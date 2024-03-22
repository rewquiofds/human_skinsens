#!/usr/bin/env python
# -*- coding:utf-8 -*-
# FileName: SkinSensPipeline.py
# Author: hqwang
# Time: 2023/3/18/018 0:54
# Note: 1.
import os
from itertools import product
# import wandb
import numpy as np
import pandas as pd
import json
from ray import tune
from ray.tune.sklearn import TuneSearchCV, TuneGridSearchCV
import optuna
from skopt.space import Real, Integer, Categorical
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer
from sklearnex import patch_sklearn

patch_sklearn()
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids

import warnings
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics
from .utils import MyEncoder, get_new_log_dir, get_logger, specificity, CCR


class SkinSensPipe():
    def __init__(self,
                 cv=None,
                 scoring=None,
                 n_trials=None,
                 search_optimization=None,
                 n_jobs=None,
                 log_dir=None,
                 details_dir=None,
                 ):
        self.cv = cv
        self.scoring = scoring
        self.n_trials = n_trials
        self.search_optimization = search_optimization
        self.n_jobs = n_jobs
        self.log_dir = log_dir
        self.details_dir = details_dir

    @staticmethod
    def load_data(fp_desc, path):
        if '+' in fp_desc:  # “+“表示特征融合
            data = pd.DataFrame()
            list = [i.strip() for i in fp_desc.split('+')]
            for idx, name in enumerate(list):
                if idx == 0:
                    df = pd.read_csv(os.path.join(path, name + '.csv'))
                    data = pd.concat([data, df], axis=1)
                else:
                    df = pd.read_csv(os.path.join(path, name + '.csv'))
                    data = pd.merge(data, df)
        else:
            data = pd.read_csv(os.path.join(path, fp_desc + '.csv'))

        x = data.iloc[:, 4:]
        y = data.iloc[:, 3]
        return x, y

    def tune_pipeline(self, init_pipe, parameters, param_distributions, xtrain, ytrain):
        model = init_pipe['model']
        if type(model) in [StackingClassifier, VotingClassifier]:
            for prefix, sub_model in model.get_params()['estimators']:
                # 参考https://github.com/MI2-Education/2022L-WB-AutoML/blob/3b28137b5e517c77c54aeda44a89bfdf57ac1929/projects/Moja%20grupa/KM4/model_selection.py
                sub_model_params = parameters[type(sub_model)]
                param_distributions.update(
                    {('model__' + prefix + "__" + key): value for key, value in sub_model_params.items()})
            if type(model) == StackingClassifier:
                final_model = model.get_params()['final_estimator']
                final_model_params = parameters[type(final_model)]
                param_distributions.update(
                    {('model__final_estimator' + "__" + key): value for key, value in final_model_params.items()})
        elif type(model) == CalibratedClassifierCV:
            param_distributions.update(
                {('model__base_estimator__' + key): value for key, value in parameters[type(model)].items()})
        else:
            param_distributions.update({('model__' + key): value for key, value in parameters[type(model)].items()})
        if self.search_optimization != "gridsearch":
            tune_search = TuneSearchCV(
                init_pipe,
                param_distributions,
                n_trials=self.n_trials,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                refit="mcc",
                # ValueError: When using multimetric scoring, refit must be the name of the scorer used to pick the best parameters.
                cv=self.cv,
                verbose=0,
                random_state=42,
                local_dir=self.log_dir,
                name='调参详情__' + self.details_dir,
                search_optimization=self.search_optimization,
                mode='max',
            )
        else:
            # tune_search = TuneGridSearchCV(
            #     init_pipe,
            #     param_distributions,
            #     scoring=self.scoring,
            #     n_jobs=self.n_jobs,
            #     cv=self.cv,
            #     verbose=0,
            #     return_train_score=False,
            #     refit="mcc",
            #     local_dir=self.log_dir,
            #     name='调参详情__' + self.details_dir,
            #     mode='max',
            # )
            tune_search = GridSearchCV(
                init_pipe,
                param_distributions,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                refit="ccr",
                cv=self.cv,
                verbose=0,
                return_train_score=False,
            )
        tune_search.fit(xtrain, ytrain)
        # print(pd.DataFrame(fps_descs=tune_search.cv_results_))
        best_params = tune_search.best_params_
        best_pipe = tune_search.best_estimator_
        best_score = tune_search.best_score_
        return best_params, best_pipe, best_score, tune_search.cv_results_

    def cv_pipeline(self, best_pipe, xtrain, ytrain):
        cv_metrics = cross_validate(
            best_pipe,
            xtrain,
            ytrain,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs
        )
        cv_metrics_mean = [np.mean(cv_metrics['test_' + score]) for score in self.scoring.keys()]
        cv_metrics_std = [np.std(cv_metrics['test_' + score]) for score in self.scoring.keys()]
        return cv_metrics_mean, cv_metrics_std

    def get_cv_results(self, cv_results, scoring):
        index = np.argmin(cv_results[f'rank_test_{scoring}'])
        cv_metrics_mean = []
        cv_metrics_std = []
        for score in self.scoring.keys():
            cv_metrics_mean.append(cv_results[f'mean_test_{score}'][index])
            cv_metrics_std.append(cv_results[f'std_test_{score}'][index])
        return cv_metrics_mean, cv_metrics_std

    def test_pipeline(self, best_pipe, xtest, ytest):
        # best_pipe.fit(xtrain, ytrain)
        ytest_pred = best_pipe.predict(xtest)
        # test_metrics = Metrics(ytest, ytest_pred)
        test_metrics = []
        for score_name, score_func in self.scoring.items():
            if score_name in ["sp", "npv"]:
                test_metrics.append(score_func._score_func(ytest, ytest_pred, pos_label=0))
            else:
                test_metrics.append(score_func._score_func(ytest, ytest_pred))
        return test_metrics

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# FileName: main.py
# Author: hqwang
# Time: 2023/3/18/018 2:06
# Note: 1.
import os
import itertools
import shutil
import json
import argparse
from datetime import datetime

# import wandb
import numpy as np
import pandas as pd
import joblib

from ray import tune
from ray.tune.sklearn import TuneSearchCV, TuneGridSearchCV
from ray.tune.search.optuna import optuna_search
from skopt.space import Real, Integer, Categorical
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, \
    cross_validate, cross_val_score, cross_val_predict
from sklearn.feature_selection import VarianceThreshold, RFECV, mutual_info_classif, SelectKBest, SelectPercentile, \
    SelectFromModel
from SkinSensPipeline import SkinSensPipe
from SkinSensPipeline import get_new_log_dir, get_logger, MyEncoder, specificity, PPV, NPV, CCR, \
    feature_importance_plot, logger_loop


## 关于对数均匀分布（tune.loguniform），这篇文章讲的非常清楚，并展示了它的概率密度函数和累积分布函数：https://ww2.mathworks.cn/help/stats/loguniform-distribution.html#mw_aa53473e-62d9-424b-b47d-0e26a26506e0
## ray官网对Search Space的讲解：https://docs.ray.io/en/latest/tune/api_docs/search_space.html
## 后来发现可以：from ray.tune.search.optuna import optuna_search，然后用optuna_search.Uniform等

def get_configs(dataset):
    if dataset == 'gpmt':
        from configs.configs_gpmt import models, fp_desc_list, parameters
        import configs.configs_gpmt as configs
        return models, fp_desc_list, parameters, configs
    elif dataset == 'dpra':
        from configs.configs_dpra import models, fp_desc_list, parameters
        import configs.configs_dpra as configs
        return models, fp_desc_list, parameters, configs
    elif dataset == 'keratinosens':
        from configs.configs_keratinosens import models, fp_desc_list, parameters
        import configs.configs_keratinosens as configs
        return models, fp_desc_list, parameters, configs
    elif dataset == 'hclat':
        from configs.configs_hclat import models, fp_desc_list, parameters
        import configs.configs_hclat as configs
        return models, fp_desc_list, parameters, configs
    elif dataset == 'llna':
        from configs.configs_llna import models, fp_desc_list, parameters
        import configs.configs_llna as configs
        return models, fp_desc_list, parameters, configs


def main(dataset):
    models, fp_desc_list, parameters, configs = get_configs(dataset)

    log_dir = get_new_log_dir(root=configs.log_root, suffix=configs.exper_name)
    logger = get_logger('train', log_dir)
    shutil.copy(f"./main.py", log_dir)
    shutil.copy(f"./configs/configs_{dataset}.py", log_dir)
    shutil.copy("./SkinSensPipeline/SkinSensPipe.py", log_dir)

    result = []
    start = datetime.now()
    for fp_desc, (model_name, model) in itertools.product(fp_desc_list, models.items()):
        logger.info('-' * 88)
        logger.info(f'{"| Experiment Name":<46}| {configs.exper_name:<27}|')
        logger.info(f'{"| Dataset":<46}| {configs.dataset:<39}|')
        logger.info(f'{"| Resampling strategy":<46}| {configs.resampling:<39}|')
        logger.info(f'{"| Molecule representation":<46}| {fp_desc:<39}|')
        logger.info(f'{"| Model":<46}| {model_name:<39}|')
        logger.info(f'{"| Searching algorithm":<46}| {configs.search_optimization:<39}|')
        logger.info('-' * 88)

        # Note: Initialize pipeline
        init_pipe = Pipeline(steps=[
            ('fs1', VarianceThreshold(threshold=0.2)),
            # ('fs2', RFECV(
            #     estimator=RandomForestClassifier(),
            #     step=50,
            #     min_features_to_select=30,
            #     cv=StratifiedKFold(5, random_state=42, shuffle=True),
            #     scoring=make_scorer(CCR, greater_is_better=True),
            #     n_jobs=configs.n_jobs)
            # ),
            # ('fs2', SelectKBest(score_func=mutual_info_classif, k=30)),
            # ('fs2', SelectFromModel(estimator=RandomForestClassifier(), threshold="1.25*mean")),
            ('fs2', SelectPercentile(score_func=mutual_info_classif)),
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        # if "Descriptors" in fp_desc:
        #     init_pipe.steps.insert(-1, ('scaler', StandardScaler()))  # 以分子指纹作为输入的时候需不需要StandardScaler()
        cv = StratifiedKFold(n_splits=configs.n_splits, shuffle=True, random_state=42)
        # cv = RepeatedStratifiedKFold(n_splits=configs.n_splits, n_repeats=5, random_state=42)
        SensPipe = SkinSensPipe(
            cv=cv,
            scoring=configs.scoring,
            n_trials=configs.n_trials,
            search_optimization=configs.search_optimization,
            n_jobs=configs.n_jobs,
            log_dir=log_dir,
            details_dir=model_name + '__' + fp_desc,  # 调参文件夹的名称
        )

        # Note: Load and resample data
        x, y = SkinSensPipe.load_data(fp_desc, configs.fp_desc_path)
        xtrainval, xtest, ytrainval, ytest = train_test_split(x, y, test_size=configs.test_size, random_state=42,
                                                              stratify=y, shuffle=True)
        if configs.resampling:
            xtrainval_raw, ytrainval_raw = xtrainval.copy(deep=True), ytrainval.copy(deep=True)
            xtrainval, ytrainval = SkinSensPipe.resampling(xtrainval, ytrainval, configs.resampling, configs.n_jobs)

        # Note: Tune pipeline, then save
        param_distributions = {
            'fs1__threshold': tune.choice([0.0]),
            'fs2__percentile': tune.choice([100]),
        }
        t0 = datetime.now()
        best_params, best_pipe, best_score, cv_results = SensPipe.tune_pipeline(init_pipe, parameters,
                                                                                param_distributions,
                                                                                xtrainval, ytrainval)
        t1 = datetime.now()
        # logger.info(f'{"| Best parameters":46}| \n{json.dumps(best_params, indent=4, cls=MyEncoder)}')
        logger_loop(logger, "Best parameters", best_params.keys(), best_params.values())
        logger.info(f'{"| Best score":46}| {best_score:<39}|')
        logger.info(f'{"| Time consumed in hyperparameter tuning":46}| {str(t1 - t0):<39}|')
        t0 = datetime.now()
        joblib.dump(best_pipe, os.path.join(log_dir, SensPipe.details_dir + '.pkl'))
        t1 = datetime.now()
        logger.info(f'{"| Time consumed in saving the best pipeline":46}| {str(t1 - t0):<39}|')
        logger.info('-' * 88)

        # Note: Reproduce feature selection steps
        logger.info(f'{"| The dimension before feature selection":46}| {str(xtrainval.shape):<39}|')
        for idx, (step_name, step) in enumerate(best_pipe.steps):
            if "fs" in step_name:
                dim_0 = xtrainval.shape[0]
                dim_1 = step.get_feature_names_out().shape[0]
                logger.info(f'{f"| The dimension after feature selection {step_name}":46}| {str((dim_0, dim_1)):<39}|')

                if idx == 1:
                    feature_index = best_pipe.steps[idx - 1][1].get_support()
                    feature_name = x.columns[feature_index]
                    try:
                        feature_importances = pd.Series(abs(best_pipe.steps[idx][1].estimator_.coef_[0]),
                                                        index=feature_name).sort_values(ascending=False).head(20)
                    # except:
                    #     feature_importances = pd.Series(abs(best_pipe.steps[idx][1].estimator_.feature_importances_),
                    #                                     index=feature_name).sort_values(ascending=False).head(20)
                    except:
                        feature_importances = pd.Series(abs(best_pipe.steps[idx][1].scores_),
                                                        index=feature_name).sort_values(ascending=False).head(20)
                    logger_loop(logger, "Ranking of important features", feature_importances.index,
                                feature_importances.values)
                    # feature_importance_plot(feature_importances, save_path=os.path.join(log_dir, model_name + '__' + fp_desc) + '.png')
        logger.info('-' * 88)

        # Note: Cross validate pipeline
        t0 = datetime.now()
        if configs.resampling:
            cv_metrics_mean, cv_metrics_std = SensPipe.cv_pipeline(best_pipe, xtrainval_raw, ytrainval_raw)
        else:
            cv_metrics_mean, cv_metrics_std = SensPipe.get_cv_results(cv_results, scoring='mcc')
        logger_loop(logger, "CV results on the validation set", SensPipe.scoring, cv_metrics_mean)
        t1 = datetime.now()
        logger.info(f'{"| Time consumed in cv on the training set":46}| {str(t1 - t0):<39}|')
        logger.info('-' * 88)

        # Note: test pipeline
        t0 = datetime.now()
        test_metrics = SensPipe.test_pipeline(best_pipe, xtest, ytest)
        t1 = datetime.now()
        logger_loop(logger, "Test on the test set", SensPipe.scoring.keys(), test_metrics)
        logger.info(f'{"| Time consumed in testing on the test set":46}| {str(t1 - t0):<39}|')
        logger.info('-' * 88)
        logger.info('\n' * 10)
        result.append(
            [configs.exper_name, fp_desc, model_name, best_params, best_pipe, best_score] +
            cv_metrics_mean +
            cv_metrics_std +
            test_metrics
        )

    result = pd.DataFrame(result)
    result.columns = [
        'task',
        'representation',
        'model',

        'best_param',
        'best_pipe',
        'best_score',

        'cv_ACC_mean',
        'cv_SE_mean',
        'cv_SP_mean',
        'cv_MCC_mean',
        'cv_PPV_mean',
        'cv_NPV_mean',
        'cv_CCR_mean',

        'cv_ACC_std',
        'cv_SE_std',
        'cv_SP_std',
        'cv_MCC_std',
        'cv_PPV_std',
        'cv_NPV_std',
        'cv_CCR_std',

        'test_ACC',
        'test_SE',
        'test_SP',
        'test_MCC',
        'test_PPV',
        'test_NPV',
        'test_CCR',
    ]
    round(result, 4).to_excel(f'{log_dir}/results.xlsx', index=False)

    end = datetime.now()
    logger.info(f'elapsed time: {end - start}')


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--dataset", type=str, default='dpra', required=False)
    args = parse.parse_args()
    main(dataset=args.dataset)

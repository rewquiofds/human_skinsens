from ray import tune
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier, \
    ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB, CategoricalNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier, BalancedBaggingClassifier, \
    BalancedRandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.metrics import make_scorer
from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
import sys

# from Sensitization_final.SkinSensPipeline import get_new_log_dir, get_logger, MyEncoder, specificity, PPV, NPV, CCR

# note ################################################################################################################
log_root = "./logs/KeratinoSens"
exper_name = "262条 100次调参"
dataset = 'KeratinoSens'
n_trials = 100
params_mode = ["gridsearch", "bayesian", "default"][1]
search_optimization = ["gridsearch", "random", "bayesian", "bohb", "hyperopt", "optuna"][4]
n_jobs = -1
fp_desc_path = "./fps/keratinosens_262"
test_size = 0.2
resampling = ["", "SMOTE", "ADASYN", "RandomUnderSampler", "ClusterCentroids"][0]
n_splits = 5
# scoring = {
#     'acc': make_scorer(metrics.accuracy_score, greater_is_better=True),
#     'se': make_scorer(metrics.recall_score, greater_is_better=True),
#     'sp': make_scorer(specificity, greater_is_better=True),
#     'mcc': make_scorer(metrics.matthews_corrcoef, greater_is_better=True),
#     'ppv': make_scorer(PPV, greater_is_better=True),
#     'npv': make_scorer(NPV, greater_is_better=True),
#     'ccr': make_scorer(CCR, greater_is_better=True),
# }
scoring = {
    'acc': make_scorer(metrics.accuracy_score, greater_is_better=True),
    'se': make_scorer(metrics.recall_score, greater_is_better=True),
    'sp': make_scorer(metrics.recall_score, greater_is_better=True, pos_label=0),
    'mcc': make_scorer(metrics.matthews_corrcoef, greater_is_better=True),
    'ppv': make_scorer(metrics.precision_score, greater_is_better=True),
    'npv': make_scorer(metrics.precision_score, greater_is_better=True, pos_label=0),
    'ccr': make_scorer(metrics.balanced_accuracy_score, greater_is_better=True),
}
# note ################################################################################################################
fp_desc_list = [
    'MACCSFP',
    'PubChemFP',  # 很好
    'SubFP',  # 好
    'SubFPC',  # 好
    # 'KRFP',
    # 'KRFPC',
    'ECFP4',
    'ECFP12',
    'RDKit_Descriptors',
    # 'Chemopy_Descriptors_1D2D',
    # 'top10'
    # 'SubFPC+PaDEL_Descriptors_EState',
    # 'rdkit_desc + opera_desc + vega_desc + padel_desc'
]

# note ################################################################################################################
models = {
    # 'LR': LogisticRegression(random_state=42, max_iter=10000),
    'RF': RandomForestClassifier(random_state=42),
    # 'SVM': CalibratedClassifierCV(SVC(random_state=42), method='sigmoid'),
    'GBDT': GradientBoostingClassifier(random_state=42),
    'ExtTree': ExtraTreesClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'MLP': MLPClassifier(random_state=42),

    # '(stacking)GBDT+XGB': StackingClassifier(estimators=[('ExtTree', GradientBoostingClassifier(random_state=42)),
    #                                                         ('XGB', XGBClassifier(random_state=42))],
    #                                             final_estimator=LogisticRegression(random_state=42),
    #                                             n_jobs=n_jobs),
    # '(voting)GBDT+XGB': VotingClassifier(estimators=[('ExtTree', GradientBoostingClassifier(random_state=42)),
    #                                                     ('XGB', XGBClassifier(random_state=42))],
    #                                         voting='soft',
    #                                         n_jobs=n_jobs,
    #                                         verbose=True),
}

if params_mode == "bayesian":
    # note 贝叶斯调参参数 ################################################################################################################
    parameters = {
        LogisticRegression: {
            'penalty': tune.choice(['l2']),
            'C': tune.loguniform(0.01, 100),
            'class_weight': tune.choice(['balanced']),
            'solver': tune.choice(['lbfgs']),
        },
        RandomForestClassifier: {
            'n_estimators': tune.randint(100, 1000),
            'criterion': tune.choice(['gini', 'entropy']),
            'max_depth': tune.randint(3, 30),
            'min_samples_split': tune.randint(2, 10),
            'min_samples_leaf': tune.randint(1, 10),
            'max_features': tune.choice(['auto', 'sqrt', 'log2']),
            'class_weight': tune.choice(['balanced_subsample', 'balanced']),
        },
        CalibratedClassifierCV: {
            'C': tune.uniform(0.01, 10),
            'kernel': tune.choice(['rbf']),
            'gamma': tune.choice(['scale', 'auto', 1e-2, 5e-2, 1e-1, 5e-1]),
            'class_weight': tune.choice(['balanced']),
        },

        GradientBoostingClassifier: {
            'loss': tune.choice(['deviance']),
            'learning_rate': tune.uniform(0.01, 0.2),
            'n_estimators': tune.randint(50, 500),
            'min_samples_split': tune.randint(2, 10),
            'min_samples_leaf': tune.randint(1, 10),
            'max_depth': tune.randint(3, 30),
            'max_features': tune.choice(['auto', 'sqrt', 'log2'])
        },
        ExtraTreesClassifier: {
            'n_estimators': tune.randint(50, 500),
            'criterion': tune.choice(["gini", "entropy"]),
            'max_depth': tune.randint(3, 30),
            'min_samples_split': tune.randint(2, 10),
            'min_samples_leaf': tune.randint(1, 10),
            'max_features': tune.choice(['auto', 'sqrt', 'log2']),
            'class_weight': tune.choice(['balanced_subsample', 'balanced']),
        },
        XGBClassifier: {
            'n_estimators': tune.randint(50, 500),
            'learning_rate': tune.uniform(0.01, 0.2),
            'gamma': tune.uniform(0, 2),
            'max_depth': tune.randint(3, 30),
            'min_child_weight': tune.uniform(1, 10),
            'subsample': tune.uniform(0.5, 1.0),
            'colsample_bytree': tune.uniform(0.5, 1.0),
            # 'reg_alpha': Real(0,0.5),
            # 'reg_lambda': Real(0,0.5)
        },
        LGBMClassifier: {
            'n_estimators': tune.randint(50, 500),
            'learning_rate': tune.uniform(0.01, 0.2),
            'max_depth': tune.randint(3, 30),
            'is_unbalance': tune.choice([True]),
            "num_leaves": tune.randint(10, 200),
            "min_child_samples": tune.randint(1, 10),
            "subsample": tune.uniform(0.5, 1),
            "colsample_bytree": tune.uniform(0.5, 1),
            # 'num_leaves': Integer(31, 127),
            # 'min_split_gain': Real(0.0,0.4),
            # 'min_child_weight': Real(0.001,0.002),
            # 'min_child_samples': Integer(15,30),
            # 'subsample': Real(0.6,1.0),
            # 'subsample_freq': Integer(3,5),
            # 'colsample_bytree': Real(0.6,1.0),
            # 'reg_alpha': Real(0,0.5),
        },

        MLPClassifier: {
            'hidden_layer_sizes': tune.choice([(256, 128), (256, 128, 64), (256, 128, 64, 32)]),
            'activation': tune.choice(['relu']),
            'solver': tune.choice(['adam']),
            'alpha': tune.uniform(0.001, 0.1),
            'batch_size': tune.choice(['auto']),
            'learning_rate': tune.choice(['constant']),
            'learning_rate_init': tune.choice([0.001]),
            'max_iter': tune.choice([100]),  # todo 这个改一下，太慢了
        }
    }

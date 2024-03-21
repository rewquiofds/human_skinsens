#!/usr/bin/env python
# -*- coding:utf-8 -*-
# FileName: utils.py
# Author: hqwang
# Time: 2022/11/11 11:42
# Note: 1.
import os
import time
import json
import logging
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import make_scorer
import plotly.graph_objs as go


class MyEncoder(
    json.JSONEncoder):  # 参考https://blog.csdn.net/zaf0516/article/details/108100862解决这个问题TypeError: Object of type bool_ is not JSON serializable
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(MyEncoder, self).default(obj)


def get_new_log_dir(root='./logs', prefix='', suffix=''):  # MinkaiXu_ConfVAE/utils/misc.py
    fn = time.strftime('%Y%m%d-%H:%M', time.localtime())
    if prefix != '':
        fn = prefix + '-' + fn
    if suffix != '':
        fn = fn + '-' + suffix
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_logger(name, log_dir=None):  # MinkaiXu_ConfVAE/utils/misc.py
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False  # 修改了这个属性后解决了重复打印的问题
    return logger


def logger_loop(logger, message, keys, values):
    for i, (k, v) in enumerate(zip(keys, values)):
        if i == 0:
            logger.info(f'{f"| {message}":46}| {(k + ": " + str(v)):<39}|')
        else:
            logger.info(f'{"| ":46}| {(k + ": " + str(v)):<39}|')


def specificity(y, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred).ravel()
    return tn / (tn + fp)


def CCR(y, y_pred):
    # BACC
    se = metrics.recall_score(y, y_pred)
    sp = specificity(y, y_pred)
    return (se + sp) / 2


def PPV(y, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred).ravel()
    return tp / (tp + fp)


def NPV(y, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred).ravel()
    return tn / (tn + fn)


# def Metrics(y, y_pred, scoring):
#     # # auc = metrics.roc_auc_score(y, y_pred)
#     # acc = metrics.accuracy_score(y, y_pred)
#     # se = metrics.recall_score(y, y_pred)  # Recall = sensitivity = TPR
#     # sp = specificity(y, y_pred)
#     # mcc = metrics.matthews_corrcoef(y, y_pred)
#     # ppv = PPV(y, y_pred)
#     # npv = NPV(y, y_pred)
#     # ccr = CCR(y, y_pred)
#     test_metrics = []
#     for score_name, score_func in scoring.items():
#         if score_name in ["sp", "npv"]:
#             test_metrics.append(score_func._score_func(y, y_pred, pos_label=0))
#         else:
#             test_metrics.append(score_func._score_func(y, y_pred))
#
#     return test_metrics


def feature_importance_plot(feature_importances, save_path):
    feature_importances = feature_importances.loc[::-1]
    data = go.Bar(x=feature_importances.values,
                  y=feature_importances.index,
                  # marker={'color': data,
                  #         'colorscale': 'Viridis'},
                  orientation='h')
    fig = go.Figure(data)
    fig.update_layout(title='feature importance ranking',
                      xaxis_title='feature importance',
                      # yaxis_title='名称',
                      template='plotly')
    fig.write_image(save_path, width=350, height=500, scale=2)

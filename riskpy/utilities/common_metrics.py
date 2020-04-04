import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import math
from statsmodels.stats.outliers_influence import variance_inflation_factor
from math import log
from scipy.stats import ttest_rel, t
from scipy.stats import spearmanr


# Вычисление gini
def gini(y_true, y_pred):
    """
    Calculate gini
    :param y_true: Fact values
    :param y_pred: Predicted values
    :return: Gini's coefficient
    """
    try:
        return roc_auc_score(y_true=y_true, y_score=y_pred) * 2 - 1
    except:
        return np.nan


# Вычисление VIF
def vif(data):
    """
    Calculate VIFs
    :param data: Data
    :return: List of VIFs for each variable in format: Variable - VIF
    """
    res = list()
    for i in range(data.shape[1]):
        res.append([data.columns[i], variance_inflation_factor(data.values, i)])
    return res


# Вычисление PSI
def psi(data1, data2, target_column=None, zone_borders=(0.1, 0.2)):
    """
    Calculate Population Stability Index (PSI)

    :param data1: First dataset
    :param data2: Second dataset
    :param target_column: Name of target columns (optional)
    :param zone_borders: validation test borders: under the first - green, above the second - red, between - yellow
    :return: Table of PSIs
    """
    result = []
    columns = [col for col in data2.columns if col in data1.columns]
    if target_column is None:
        for column in columns:
            shares1 = data1[column].value_counts() / len(data1[column])
            shares2 = data2[column].value_counts() / len(data2[column])
            dif = shares1 - shares2
            ln = (shares1 / shares2).apply((lambda x: log(x)))
            psi = dif * ln
            a = psi.sum()
            result.append([column, a])
        result = pd.DataFrame({'Column': [x[0] for x in result], 'PSI': [x[1] for x in result]})
    else:
        target_unique_values = list(set(data1[target_column].unique()).union(set(data2[target_column].unique())))
        if len(target_unique_values) > 2:
            raise Exception('Target must be binary')
        for target_value in target_unique_values:
            for column in columns:
                shares1 = data1.loc[data1[target_column] == target_value, column].value_counts() / \
                          len(data1.loc[data1[target_column] == target_value, column])
                shares2 = data2.loc[data2[target_column] == target_value, column].value_counts() / \
                          len(data2.loc[data2[target_column] == target_value, column])
                dif = shares1 - shares2
                ln = (shares1 / shares2).apply((lambda x: log(x)))
                psi = dif * ln
                a = psi.sum()
                result.append([column, target_value, a])
        result = pd.DataFrame({
            'Column': [x[0] for x in result],
            'Target': [x[1] for x in result],
            'PSI': [x[2] for x in result]}).pivot(index='Column', columns='Target', values='PSI')
    result.style.applymap(
        lambda x: "background-color: red" if x > zone_borders[1] else ("background-color: yellow" if x > zone_borders[0] else "background-color: green"),
        subset=['PSI'])
    return result


def HHI_dataset(data, adj = True, j=26):
    """
    Calculate HHI for all variable in dataframe
    :param data: Dataframe
    :param adj: Считать ли скорректированный индекс или обычный
    :param j: Количество бакетов в целевом разбиении. Применяется при рассчёте скорретированного индекса
    :return: Array in format: Variable - HHI
    """
    result = ()
    for column in data.columns:
        result.append([column, ], HHI_arr(data[column], adj, j))
    return result


def HHI_arr(arr, adj = True, j=26):
    """
    Calculate HHI for Series
    :param row: Data series
    :param adj: Считать ли скорректированный индекс или обычный
    :param j: Количество бакетов в целевом разбиении. Применяется при рассчёте скорретированного индекса
    :return: HHI
    """
    _, counts = np.unique(arr, return_counts=True)
    hhi = np.sum((counts/np.sum(counts)) ** 2)
    if adj:
        return (hhi-1/j)/(1-1/j)
    else:
        return hhi

# Доверительный интервал для среднего
def mean_CI(data, alpha):
    """
    Calculate confidence interval for mean
    :param data: Sample
    :param alpha: Confidence level
    :return: CI and average value in format [lowest, avg, highest]
    """
    n = len(data)
    t_95 = t._ppf((alpha + 1) / 2, df=n - 1)
    avg = np.mean(data)
    up = avg + t_95 * np.std(data) / (n ** 0.5)
    down = avg - t_95 * np.std(data) / (n ** 0.5)
    return [down, avg, up]


# Проверяет попадание среднего предсказание в ДИ среднего факта
def average_t_test(data, fact_name, predicted_name):
    """
    Plot CI for fact average and predicted average
    :param data: DataFrame
    :param fact_name: name of fact column
    :param predicted_name: name of predicted column
    :return: t-test for related samples result
    """
    # считаем ДИ
    t_test = ttest_rel(data[fact_name], data[predicted_name])
    ci_95 = mean_CI(data[fact_name], 0.95)
    ci_99 = mean_CI(data[fact_name], 0.99)
    # рисуем ДИ и прогнозы
    plt.figure(figsize=[15, 15])
    plt.axhline(y=ci_95[1], label='Среднее фактическое', lw=5)
    plt.fill_between(x=[0, 1], y1=ci_99[0], y2=ci_99[2], color='yellow', alpha=0.5,
                     label='99% ДИ среднего фактического')
    plt.fill_between(x=[0, 1], y1=ci_95[0], y2=ci_95[2], color='green', alpha=0.5, label='95% ДИ среднего фактического')
    plt.axhline(y=np.average(data[predicted_name]), label='Среднее предсказанное', lw=5, color='red')
    plt.legend()

    return t_test


# Вычисляет R2, корреляцию спирмена и hitrate
def hitrate(data, fact_name, prediction_name):
    """
    Calculate R2, Spearman correlation and hit rates for EAD predictions
    :param data: Dataframe
    :param fact_name: Name of fact column
    :param prediction_name: Name of prediction column
    :return: R2,Spearman correlation, hit rates
    """
    plt.figure(figsize=[10, 10])
    plt.scatter(x=data[fact_name], y=data[prediction_name], c='g', marker='.')
    plt.plot(data[fact_name], data[fact_name], 'b:')
    plt.xlabel('EAD факт')
    plt.ylabel('EAD прогноз')

    r2 = r2_score(y_true=data[fact_name], y_pred=data[prediction_name])
    spearman_c = spearmanr(data[fact_name], data[prediction_name])

    fact = data[fact_name]
    predict = data[prediction_name]
    hit_rate_5 = np.average([math.fabs(f - p) / p < 0.05 for f, p in zip(fact, predict)])
    hit_rate_10 = np.average([math.fabs(f - p) / p < 0.1 for f, p in zip(fact, predict)])
    hit_rate_20 = np.average([math.fabs(f - p) / p < 0.2 for f, p in zip(fact, predict)])

    return [['R2', r2], ['Spearman', spearman_c], ['HR5', hit_rate_5], ['HR10', hit_rate_10], ['HR20', hit_rate_20]]

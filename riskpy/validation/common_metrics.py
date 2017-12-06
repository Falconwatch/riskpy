import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from ..graphs import graphs
import numpy as np
from sklearn.metrics import roc_curve, auc
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
    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_pred)
    roc_auc = auc(fpr, tpr)
    g = 2 * roc_auc - 1
    return g


# Вычисление VIF
def vif(data):
    """
    Calculate VIFs
    :param data: Data
    :return: List of VIFs for each variable in format: Variable - VIF
    """
    res = list()
    for i in range(data.shape[1]):
        res.append([data.columns[i], variance_inflation_factor(data.as_matrix(), i)])
    return res


# Вычисление PSI
def psi(data1, data2):
    """
    Calculate Population Stability Index (PSI)
    :param data1: First dataset
    :param data2: Second dataset
    :return: Array of PSIs in format: column, PSI
    """
    result = []
    columns = [col for col in data2.columns if col in data1.columns]
    for column in columns:
        shares1 = data1[column].value_counts() / len(data1[column])
        shares2 = data2[column].value_counts() / len(data2[column])
        dif = shares1 - shares2
        ln = (shares1 / shares2).apply((lambda x: log(x)))
        psi = dif * ln
        a = psi.sum()
        result.append([column, a])
        # print (column,'{0:.40f}'.format(a))
    return result


# информация по дескриминирующей силе: джини и рок кривая
def gini_info(data, fact_name, pred_name, name=""):
    """
    Calculate gini and plot ROC-curve
    :param data: Dataframe
    :param fact_name: Name of the column with fact values
    :param pred_name: Name of the column with predicted values
    :param name: Name for the plot's title
    :return:
    """
    gini_data = data.dropna().copy()
    predicted = gini_data[pred_name]
    fact = gini_data[fact_name] > np.average(gini_data[fact_name])
    g = gini(y_true=fact, y_pred=predicted)
    graphs.roc(y_true=fact, y_pred=predicted, name=' for ' + name)
    return g


def HHI_dataset(data):
    """
    Calculate HHI for all variable in dataframe
    :param data: Dataframe
    :return: Array in format: Variable - HHI
    """
    result = ()
    for column in data.columns:
        result.append([column, ], HHI_row(data[column]))
    return result


def HHI_row(row):
    """
    Calculate HHI for Series
    :param row: Data series
    :return: HHI
    """
    return np.sum((row.value_counts() / len(row)) ** 2)


# Доверительный интервал для среднего
def mean_CI(data, alpha):
    """
    Calculate confidence interval for mean
    :param data: Sample
    :param alpha: Confidence level
    :return: CI and average value
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

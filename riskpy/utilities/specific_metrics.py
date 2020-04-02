import math
from riskpy.utilities.common_metrics import gini
import numpy as np


# gini и доверительный интервал
def pd_gini_interval(fact, predicted):
    """
    Calculate confidence interval for gini's coefficient
    :param fact: Fact values
    :param predicted: Predicted values
    :return: CI and gini
    """
    g = gini(y_true=fact, y_pred=predicted)
    if np.isnan(g):
        return [np.nan, np.nan, np.nan]
    t = (g + 1) / 2
    q1 = t / (2 - t)
    q2 = (2 * t ** 2) / (1 + t)
    goods = len(fact[fact == 0])
    bads = len(fact[fact == 1])
    res = (t * (1 - t) + (bads - 1) * (q1 - t ** 2) + (goods - 1) * (q2 - t ** 2)) / (goods * bads)
    res = math.sqrt(res)
    return [g - 3 * res, g, g + 3 * res]



# джини и ДИ во времени
def pd_gini_in_time(data, fact_name, pred_name, time_name):
    """
    Plot gini's coefficient and its confidence interval
    :param data: Dataframe
    :param fact_name: Name of column with fact values
    :param pred_name: Name of column with predicted values
    :param time_name: Name of column with datetime values
    :param name: Name for plot title
    :param zone_borders: Colored areas borders
    :param size: Result plot size
    :param colorful: color areas between borders
    """
    gini_data = data.dropna().copy()
    times = sorted(gini_data[time_name].unique())

    dates, ginis_down, ginis, ginis_up = list(), list(), list(), list()

    for time in times:
        predicted = gini_data.loc[gini_data[time_name] == time, pred_name]
        fact = gini_data.loc[gini_data[time_name] == time, fact_name]
        gini_int = pd_gini_interval(fact, predicted)

        dates.append(time)
        ginis_down.append(gini_int[0])
        ginis.append(gini_int[1])
        ginis_up.append(gini_int[2])

    return dates, ginis_down, ginis, ginis_up

import matplotlib.pyplot as plt
import math
from . import common_metrics


# gini и доверительный интервал
def pd_gini_interval(fact, predicted):
    """
    Calculate confidence interval for gini's coefficient
    :param fact: Fact values
    :param predicted: Predicted values
    :return: CI and gini
    """
    g = common_metrics.gini(y_true=fact, y_pred=predicted)
    t = (g + 1) / 2
    q1 = t / (2 - t)
    q2 = (2 * t ** 2) / (1 + t)
    goods = len(fact[fact == 0])
    bads = len(fact[fact == 1])
    res = (t * (1 - t) + (bads - 1) * (q1 - t ** 2) + (goods - 1) * (q2 - t ** 2)) / (goods * bads)
    res = math.sqrt(res)
    return [g - 3 * res, g, g + 3 * res]


# джини и ДИ во времени
def pd_gini_in_time(data, fact_name='y_fact', pred_name='y_pred', time_name='period', name='', figsize=(17, 10)):
    """
    Plot gini's coefficient and its confidence interval
    :param data: Dataframe
    :param fact_name: Name of column with fact values
    :param pred_name: Name of column with predicted values
    :param time_name: Name of column with datetime values
    :param name: Name for plot title
    :param figsize: figsize for the plot
    """
    gini_data = data.dropna().copy()
    dates = list()
    ginis = list()
    times_gini = list()

    for time in sorted(gini_data[time_name].unique(), ascending=True):
        predicted = gini_data.loc[gini_data[time_name] == time, pred_name]
        fact = gini_data.loc[gini_data[time_name] == time, fact_name]
        gini_int = pd_gini_interval(fact, predicted)
        times.append([time, gini_int])
        ginis.append(gini_int[1])
        ginis_up = gini_int[2]
        ginis_down = gini_int[0]

    plt.figure(figsize=figsize)
    plt.title('Изменение Gini во времени и его доверительный интервал' + ' ' + name)
    plt.fill_between(x=range(len(dates)), y1=0, y2=0.4, color='red', alpha=0.7)
    plt.fill_between(x=range(len(dates)), y1=0.4, y2=0.6, color='yellow', alpha=0.7)
    plt.fill_between(x=range(len(dates)), y1=0.6, y2=1, color='green', alpha=0.7)
    plt.xticks(range(len(dates)), dates, rotation='vertical')

    plt.plot(range(len(dates)), ginis, c='black', label='gini')
    plt.plot(range(len(dates)), ginis_up, c='blue')
    plt.plot(range(len(dates)), ginis_down, c='blue')
    plt.fill_between(x=range(len(dates)), y1=ginis_up, y2=ginis_down, color='blue', alpha=0.3)
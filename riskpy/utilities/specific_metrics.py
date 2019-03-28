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
def pd_gini_in_time(data, fact_name, pred_name, time_name, name=""):
    """
    Plot gini's coefficient and its confidence interval
    :param data: Dataframe
    :param fact_name: Name of column with fact values
    :param pred_name: Name of column with predicted values
    :param time_name: Name of column with datetime values
    :param name: Name for plot title
    """
    gini_data = data.dropna().copy()
    times = sorted(gini_data[time_name].unique(), ascending=True)
    times_gini = list()

    for time in times:
        predicted = gini_data.loc[gini_data[time_name] == time, pred_name]
        fact = gini_data.loc[gini_data[time_name] == time, fact_name]
        gini_int = pd_gini_interval(fact, predicted)
        times_gini.append([time, gini_int])

    dates = [mg[0] for mg in times_gini]
    ginis = [mg[1][1] for mg in times_gini]
    ginis_up = [mg[1][2] for mg in times_gini]
    ginis_down = [mg[1][0] for mg in times_gini]

    plt.figure(figsize=[17, 10])
    plt.title('Изменение Gini во времени и его доверительный интервал' + name)
    plt.fill_between(x=range(len(dates)), y1=0, y2=0.4, color='red', alpha=0.7)
    plt.fill_between(x=range(len(dates)), y1=0.4, y2=0.6, color='yellow', alpha=0.7)
    plt.fill_between(x=range(len(dates)), y1=0.6, y2=1, color='green', alpha=0.7)
    plt.xticks(range(len(dates)), dates, rotation='vertical')

    plt.plot(range(len(dates)), ginis, c='black', label='gini')
    plt.plot(range(len(dates)), ginis_up, c='blue')
    plt.plot(range(len(dates)), ginis_down, c='blue')
    plt.fill_between(x=range(len(dates)), y1=ginis_up, y2=ginis_down, color='blue', alpha=0.3)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from riskpy.graphs.graphs import roc, plot_score_distr
from riskpy.utilities.common_metrics import gini
from riskpy.utilities.specific_metrics import pd_gini_interval, pd_gini_in_time
from riskpy.validation.validation_common import TestMark, TestResult


def m_1_7(data, target_name, period_name, plot=True, red_area=0.05, yellow_area=0.15):
    """
    Тест M1.7: Анализ распределения значений разности дат целевого события и дат оценки
    :param data: Данные
    :param target_name: Имя столбца с целевой переменной (бинарной)
    :param period_name: Имя столбца с периодом до наступления события
    :param plot: Флаг, выводить ли график
    :param red_area: Верхняя граница красной зоны
    :param yellow_area: Верхняя граница жёлтой зоны
    :return: Результат теста
    """
    if plot:
        tmp = data.loc[data[target_name] == 1, period_name]
        periods = sorted(tmp.unique())
        n, bins, patches = plt.hist(tmp,
                                    bins=np.arange(0.5, max(periods) + 1, 1),
                                    weights=np.ones(len(tmp)) / len(tmp))
        plt.axis([periods[0], periods[-1], 0, 1])
        plt.hlines(red_area, periods[0], periods[-1], 'r')
        plt.hlines(yellow_area, periods[0], periods[-1], 'y')

    distribution = tmp.value_counts() / len(tmp)
    min_share = np.min(distribution)

    if min_share <= red_area:
        res = TestMark.RED
    elif min_share <= yellow_area:
        res = TestMark.YELLOW
    else:
        res = TestMark.GREEN
    return TestResult(distribution, res)


def m_2_1(data, target_name, score_name, reverse=True, plot_roc=True, plot_score=True, red_area=0.2, yellow_area=0.4):
    """
    Тест M2.1: Эффективность ранжирования всей модели
    :param data: Данные
    :param target_name: Имя столбца с целевой переменной (бинарной)
    :param score_name: Имя столбца со скором
    :param reverse: Флаг, нужно ли инвертировать скор при оценке джини
    :param plot_roc: Флаг, рисовать ли график ROC-кривой
    :param plot_score: Флаг, рисовать ли гистограмму скора
    :param red_area: Верхняя граница красной зоны
    :param yellow_area: Верхняя граница жёлтой зоны
    :return: Результат теста
    """
    if plot_roc:
        roc(y_true=data[target_name], y_pred=(-1 if reverse else 1) * data[score_name])
    g = gini(y_true=data[target_name], y_pred=(-1 if reverse else 1) * data[score_name])
    if plot_score:
        plot_score_distr(data, target_name, score_name)

    if g < red_area:
        res = TestMark.RED
    elif g < yellow_area:
        res = TestMark.YELLOW
    else:
        res = TestMark.GREEN
    return TestResult(None, res)


def m_2_2(data, factors, target_name, inverse_factor=True, figure_size=(10, 10), title='Однофакторный анализ',
          red_area=0, yellow_area=0.05):
    """
    Тест M2.2: Эффективность ранжирования отдельных факторов
    :param data: Данные для оценки
    :param factors: Список факторво для оценки
    :param target_name: Имя столбца с целевой переменной
    :param inverse_factor: Инверсия знакак фактора
    :param figure_size: Размер графика
    :param title: Заголовок графика
    :param red_area: Граница красной зоны
    :param yellow_area: Граница жёлтой зоны
    :return: Результат теста
    """
    ginis = pd.DataFrame(columns=['gini_low', 'gini', 'gini_high'])
    for factor in factors:
        g = pd_gini_interval(fact=data[target_name], predicted=(-1 if inverse_factor else 1) * data[factor])
        ginis.loc[factor] = g
    ginis['std'] = (ginis['gini_high'] - ginis['gini_low']) / 6

    # Оценка по тесту
    g_min = ginis['gini'].min()
    if g_min < red_area:
        res = TestMark.RED
    elif g_min < yellow_area:
        res = TestMark.YELLOW
    else:
        res = TestMark.GREEN

    # График
    fig = plt.figure(figsize=figure_size)
    xx = range(ginis.shape[0])
    plt.bar(xx, ginis['gini'], yerr=ginis['std'],
            error_kw=dict(elinewidth=3, ecolor='orange', capsize=10, markeredgewidth=2), color='green')
    plt.xticks(xx, list(ginis.index.values), rotation=90)
    plt.ylabel('Gini')
    plt.title(title)
    plt.show()
    plt.close(fig)

    return TestResult(ginis, res)


def m_2_4(data, fact_name, pred_name, time_name, title='', figure_size=[17, 10], red_area=0.4, yellow_area=0.5):
    """
    Тест M2.4: Динамика коэффициента Джини
    :param data: Данные
    :param fact_name: Имя столбца с целевой переменной
    :param pred_name: Имя столбца с предсказанием
    :param time_name: Имя столбца с меткой времени
    :param title: Имя графика
    :param figure_size: Размер графика
    :param red_area: Граница красной зоны
    :param yellow_area: Граница жёлтой зоны
    """
    dates, ginis_down, ginis, ginis_up = pd_gini_in_time(data, fact_name, pred_name, time_name)
    counts = data.groupby(by=time_name)[time_name].count().values

    fig, ax1 = plt.subplots(figsize=figure_size)
    ax2 = ax1.twinx()
    # counts
    ax1.fill_between(x=range(len(counts)), y1=np.zeros_like(counts), y2=counts, color='green', alpha=0.3)
    # gini
    ax2.plot(range(len(dates)), ginis, c='darkblue', label='gini', marker='o')
    ax2.plot(range(len(dates)), ginis_up, c='orange', marker='^')
    ax2.plot(range(len(dates)), ginis_down, c='orange', marker='v')
    ax2.plot(range(len(dates)), [red_area for i in dates], c='red', label='Red zone threshold', linestyle='--')
    ax2.plot(range(len(dates)), [yellow_area for i in dates], c='orange', label='Yellow zone threshold', linestyle='--')
    plt.title(title)

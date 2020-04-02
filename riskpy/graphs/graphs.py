import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import math
import itertools as it
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from riskpy.utilities.common_metrics import gini
from riskpy.utilities.specific_metrics import pd_gini_in_time


def roc(y_true, y_pred, name=""):
    ''' Plot ROC curve

         Keyword arguments:
         y_true -- true values array
         y_pred -- predicted values array
    '''
    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_pred)
    gini_value = gini(y_true=y_true, y_pred=y_pred)
    print('Gini: {}'.format(gini_value))
    plt.figure(figsize=[10, 10])
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (Gini = %0.4f)' % gini_value)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic' + name)
    plt.legend(loc="lower right")
    plt.show()


def rocs(y_true, y_pred, colors, names, name=''):
    ''' Plot several ROC curves

     Keyword arguments:
     y_true -- array of true values arrays
     y_pred -- array of predicted values arrays
     colors -- array of colors for curves
     names -- array of names for curves
     name -- name of graph (default empty string)
     '''
    if (len(y_true) != len(y_pred)) or (len(y_true) != len(colors)):
        print(len(y_true), len(y_pred), len(colors))
        raise BaseException
    plt.figure(figsize=[10, 10])
    ginis = []

    for i in range(0, len(y_true)):
        fpr, tpr, _ = roc_curve(y_true=y_true[i], y_score=y_pred[i], pos_label=None)
        gini_value = gini(y_true=y_true[i], y_pred=y_pred[i])
        ginis.append(gini_value)
        plt.plot(fpr, tpr, color=colors[i], label=names[i] + ' (Gini = %0.4f)' % gini_value)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic ' + name)
    plt.legend(loc="lower right")
    plt.show()


# джини и ДИ во времени
def plot_pd_gini_in_time(data, fact_name, pred_name, time_name, name='', zone_borders=[0, 0.4, 0.6, 1], size=[15, 10],
                         colorful=True, alpha=0.7, grid=False):
    """
    Plot gini's coefficient and its confidence interval
    :param data: Dataframe
    :param fact_name: Name of column with fact values
    :param pred_name: Name of column with predicted values
    :param time_name: Name of column with datetime values
    :param name: Name for plot title
    :param zone_borders: Colored areas borders
    :param size: Result plot size, default value is A4 format
    :param colorful: color areas between borders
    """

    dates, ginis_down, ginis, ginis_up = pd_gini_in_time(data, fact_name, pred_name, time_name)

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    title = 'Изменение Gini во времени и его доверительный интервал'
    if name != '':
        title = '{} {}'.format(title, name)
    plt.title(title)
    if colorful:
        ax.fill_between(x=dates, y1=zone_borders[0], y2=zone_borders[1], color='red', alpha=alpha)
        ax.fill_between(x=dates, y1=zone_borders[1], y2=zone_borders[2], color='yellow', alpha=alpha)
        ax.fill_between(x=dates, y1=zone_borders[2], y2=zone_borders[3], color='green', alpha=alpha)
    else:
        colors = (['black', 'yellow', 'green'] + ['black' for i in zone_borders])[1: -4]
        for zb, color in zip(zone_borders[1: -1], colors):
            ax.plot(dates, [zb for i in dates], color=color, linestyle='dashed')
        ax.set_ylim([min(ginis_down), 1])
    if grid:
        plt.grid()
    ax.plot(dates, [0 for i in dates], color='black', linestyle='dashed')
    ax.set_xticks(dates)
    ax.set_yticks(sorted([x/10 for x in range(-10, 11) if x > min(min(ginis_down), 0)] + zone_borders))
    ax.tick_params(axis='x', rotation='auto')
    l1 = ax.plot(dates, ginis, c='blue', label='Значение коэф. джини во времени')
    p1 = ax.fill_between(x=dates, y1=ginis_up, y2=ginis_down, color='blue', alpha=alpha/6, hatch='.', linestyle='dotted', label='Доверительный интервал')
    ax.legend(handles=l1 + [p1, ], )
    ax.set_xlabel('Дата')
    ax.set_ylabel('Коэф. джини')
    plt.show()


def cap(y_true, y_pred, figsize=[15, 15]):
    ''' Plot CAP curve
     Keyword arguments:
     y_true -- true values
     y_pred -- predicted values
     figsize -- size of plot figure (default [15, 15])
     '''
    data = pd.DataFrame()
    data['FACT'] = y_true
    data['PREDICTED'] = y_pred
    data.sort_values(by='PREDICTED', inplace=True)
    data['DEFAULTS_SHARE'] = data[['FACT']].cumsum(axis=0)
    data['DEFAULTS_SHARE'] = data['DEFAULTS_SHARE'] / data['DEFAULTS_SHARE'].max()

    dr = data['FACT'].mean()

    model_y = data['DEFAULTS_SHARE'].values
    model_x = np.arange(len(model_y)) / len(model_y)

    our_model = model_y.mean()
    best_model = 1 - dr / 2
    gini = (our_model - 0.5) / (best_model - 0.5)

    plt.figure(figsize=figsize)
    plt.plot(model_x, model_y, color='darkorange', label='CAP curve (Gini = %0.4f)' % gini)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random model')
    plt.plot([0, dr, 1], [0, 1, 1], color='green', linestyle='--', label='Best model')

    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])

    plt.xlabel('% sample')
    plt.ylabel('% bads')
    plt.title('CAP curve')
    plt.legend(loc="lower right")
    plt.show()


def pr_curve(y_true, y_pred):
    ''' Plot PR curve

     Keyword arguments:
     y_true -- true values
     y_pred -- predicted values
     '''
    precision, recall, thresholds = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
    avg = average_precision_score(y_true, y_pred)
    plt.figure(figsize=[10, 10])
    plt.plot(recall, precision, lw=1.5, color='navy',
             label='Precision-Recall curve')
    plt.title('Precision-Recall curve: AUC={0:0.4}'.format(avg))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()
    # print (recall)


def pr_curves(y_true, y_pred, colors, names, name=''):
    ''' Plot several PR curves

     Keyword arguments:
     y_true -- array of true values arrays
     y_pred -- array of predicted values arrays
     colors -- array of colors for curves
     names -- array of names for curves
     name -- name of graph (default empty string)
     '''
    plt.figure(figsize=[10, 10])

    for i in range(0, len(y_true)):
        precision, recall, thresholds = precision_recall_curve(y_true=y_true[i], probas_pred=y_pred[i])
        avg = average_precision_score(y_true[i], y_pred[i])
        plt.plot(recall, precision, lw=1.5, color=colors[i], label=names[i] + ' AUC={0:0.4f}'.format(avg))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="upper right")
    plt.show()


def _autolabel(ax, heights, woes):
    for i in np.arange(len(heights)):
        height = heights[i]
        woe = woes[i]
        ax.text(i, height, '%f' % height, ha='center', va='bottom')
        ax.text(i, 0, '(woe= %f )' % woe, ha='center', va='bottom')


def one_bin_barchart(bining, size=5):
    ''' Plot one binning

     Keyword arguments:
     bining -- binning to be ploted
     size -- size of one side of the plot(default 5)
     '''
    total_bins = np.arange(len(bining._woes))
    if len(total_bins) == 0:
        print('No binning for variable', bining._name)
        return

    # высота столбцов - численность
    bins_counts = [b[0] + b[1] for b in bining._counts if len(b) == 2]
    # линия - доля плохих в бакете
    bins_dr = [b[1] / (b[0] + b[1]) for b in bining._counts if len(b) == 2]

    fig, ax = plt.subplots()
    ax.bar(total_bins, bins_counts, width=0.5, color='green', )
    ax.set(ylabel='Число наблюдений в бакете')
    ax2 = ax.twinx()
    ax2.plot(bins_dr, color='orange', marker='o')
    ax2.grid(b='off')
    ax2.set(ylabel='Уровень дефолтов')

    x_labels = ['<=' + str(gap[1]) for gap in bining._gaps]
    ax.set_xticks(total_bins)
    ax.set_xticklabels(x_labels)
    plt.title(bining._name)
    plt.show()


def binning_barchart(bins, size=3):
    ''' Plot several binnings

     Keyword arguments:
     bins -- binning to be ploted
     size -- size of one side of the plot(default 3)
     '''
    for bining in bins:
        one_bin_barchart(bining, size=size)


def VariablesInteraction(X_train, y_train, X_test=None, y_test=None, classes=None,
                         figsize=[20, 20], max_tree_depth=5, mode='full', dpath='mono'):
    ''' Plot graph of variables interaction (TO DO: description)

     Keyword arguments:
     X_train -- Train X data set
     y_train -- Train y set
     X_test -- Test X data set (default None)
     y_test -- Test y data set (default None)
     classes --
     figsize --
     max_tree_depth --
     mode --
     dpath --
     '''
    plt.clf()
    possible_pairs = []
    if mode == 'single':
        possible_pairs = list(it.combinations(X_train.columns, r=1))
    elif mode == 'pairs':
        possible_pairs = list(it.combinations(X_train.columns, r=2))
    elif mode == 'full':
        possible_pairs = list(it.combinations(X_train.columns, r=1)) + list(it.combinations(X_train.columns, r=2))
    else:
        raise BaseException('Wrong mode {}'.format(mode))

    if X_test is None or y_test is None:
        X_test = X_train
        y_test = y_train

    # return possible_pairs
    plt.figure(figsize=figsize)
    for i in range(len(possible_pairs)):
        pair = list(possible_pairs[i])

        X = X_train.loc[:, pair]
        y = y_train
        X_t = X_test.loc[:, pair]
        y_t = y_test

        if len(pair) == 1:
            X['const'] = 0
            X_t['const'] = 0
            pair.append('const')

        # Train
        clf = DecisionTreeClassifier(max_depth=max_tree_depth).fit(X, y)
        # Plot the decision boundary
        h_g = min([3, len(possible_pairs)])
        plt.subplot(math.ceil(len(possible_pairs) / h_g), h_g, i + 1)
        _plotTreePath(clf, pair, X, y, X_t, y_t, dpath)

    # plt.suptitle("Decision surface of a decision tree using paired features")

    plt.axis("tight")
    plt.show()


def plot_score_distr(data, target_name, score_name):
    """
    Рисует график распределения скора плохих и скора хороших
    :param data: Данные
    :param target_name: Имя столбца с целевой (бинарной)
    :param score_name: Имя столбца со скором
    """
    tmp = [data.loc[data[target_name]==0, score_name].values,
            data.loc[data[target_name]==1, score_name].values]

    plt.hist(tmp,int(5*math.log10(max([len(l) for l in tmp])))+1,
         weights=[np.ones(len(tmp[0]))/len(tmp[0]),np.ones(len(tmp[1]))/len(tmp[1])],
         color = ['g', 'r'])
    plt.legend(['Goods', 'Bads'])
    plt.show()
    
    
def _plotTreePath(clf, pair, X, y, X_t, y_t, dpath='mono'):
    n_classes = len(y.unique())
    plot_colors = "rb"
    plot_step = 0.02

    x_min, x_max = X.loc[:, pair[0]].min() - 1, X.loc[:, pair[0]].max() + 1
    y_min, y_max = X.loc[:, pair[1]].min() - 1, X.loc[:, pair[1]].max() + 1

    x_plot_step = (x_max - x_min) / 1000
    y_plot_step = (y_max - y_min) / 1000

    xx, yy = np.meshgrid(np.arange(x_min, x_max, x_plot_step), np.arange(y_min, y_max, y_plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    if dpath == 'mono':
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = np.array([p[0] for p in clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)

    g = gini(y_pred=[p[1] for p in clf.predict_proba(X_t)], y_true=y_t)

    plt.xlabel(pair[0])
    plt.ylabel(pair[1])
    plt.title('Gini: %0.4f' % g)
    # Plot the testing points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y_t == i)
        plt.scatter(X_t.iloc[idx][pair[0]], X_t.iloc[idx][pair[1]], c=color, label=i,
                    cmap=plt.cm.RdBu, edgecolor='black', s=15)
    plt.legend(loc='lower right', borderpad=0, handletextpad=0)
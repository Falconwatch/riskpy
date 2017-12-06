import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np


def roc(y_true, y_pred, name=""):
    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_pred)
    roc_auc = auc(fpr, tpr)
    gini = 2 * roc_auc - 1
    print('Gini: {}'.format(gini))
    plt.figure(figsize=[10, 10])
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (Gini = %0.4f)' % gini)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic' + name)
    plt.legend(loc="lower right")
    plt.show()


def rocs(y_true, y_pred, colors, names, name=''):
    if len(y_true) != len(y_pred) or len(y_true) != len(colors):
        print(len(y_true), len(y_pred), len(colors))
        raise BaseException
    plt.figure(figsize=[10, 10])

    for i in range(0, len(y_true)):
        print(i)
        fpr, tpr, _ = roc_curve(y_true=y_true[i], y_score=y_pred[i], pos_label=None)
        roc_auc = auc(fpr, tpr)
        gini = 2 * roc_auc - 1
        lw = 2
        plt.plot(fpr, tpr, color=colors[i], label=names[i] + ' (Gini = %0.4f)' % gini)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic ' + name)
    plt.legend(loc="lower right")
    plt.show()


def pr_curve(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true=y_true, probas_pred=y_scores)
    avg = average_precision_score(y_true, y_scores)
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


def pr_curves(y_true, y_scores, colors, names, name=''):
    plt.figure(figsize=[10, 10])

    for i in range(0, len(y_true)):
        precision, recall, thresholds = precision_recall_curve(y_true=y_true[i], probas_pred=y_scores[i])
        avg = average_precision_score(y_true[i], y_scores[i])
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
    total_bins = np.arange(len(bining._woes))
    if len(total_bins) == 0:
        print('No binning for variable',bining._name)
        return
    heights = []
    label = "Ratio"

    if len(bining._gaps_avg) == 0:
        heights = bining._gaps_counts_shares
    else:
        heights = bining._gaps_avg
        label = 'AVG'

    max_y = np.max(heights) * 1.2
    min_y = np.min(heights + [0]) * 1.2

    plt.figure(figsize=(size * 2, size))
    ax = plt.subplot()
    plt.ylim([min_y, max_y])
    ax.bar(total_bins, height=heights, color='lightblue', label='ssss')
    _autolabel(ax, heights, bining._woes)
    ax.set_ylabel(label)
    ax.set_title(label + ' for ' + str(bining._name))
    ax.set_xticks(total_bins)
    ax.axhline(y=0, linewidth=1)

    x_labels = ['<=' + str(gap[1]) for gap in bining._gaps]

    ax.set_xticklabels(x_labels)
    plt.show()


def binning_barchart(bins, size=3):
    for bining in bins:
        one_bin_barchart(bining, size=size)
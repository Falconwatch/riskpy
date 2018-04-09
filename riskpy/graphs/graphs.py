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
    ginis=[]

    for i in range(0, len(y_true)):
        fpr, tpr, _ = roc_curve(y_true=y_true[i], y_score=y_pred[i], pos_label=None)
        roc_auc = auc(fpr, tpr)
        gini = 2 * roc_auc - 1
        ginis.append(gini)
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
    
    #высота столбцов - численность
    bins_counts=[b[0]+b[1] for b in bining._counts if len(b)==2]
    #линия - доля плохих в бакете
    bins_dr=[b[1]/(b[0]+b[1]) for b in bining._counts if len(b)==2]
    
    fig, ax =plt.subplots()   
    ax.bar(total_bins, bins_counts, width=0.5, color='green',)
    ax.set(ylabel = 'Число наблюдений в бакете')
    ax2=ax.twinx()
    ax2.plot(bins_dr, color='orange',marker='o')
    ax2.grid(b='off') 
    ax2.set(ylabel = 'Уровень дефолтов')
    
    x_labels = ['<=' + str(gap[1]) for gap in bining._gaps]
    ax.set_xticks(total_bins)
    ax.set_xticklabels(x_labels)
    plt.title(bining._name)
    plt.show()


def binning_barchart(bins, size=3):
    for bining in bins:
        one_bin_barchart(bining, size=size)
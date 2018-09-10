import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import math
import itertools as it
from ..utilities.common_metrics import gini
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


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
    
    
def cap(y_true,y_pred, figsize=[15,15]):
    data=pd.DataFrame()
    data['FACT']=y_true
    data['PREDICTED']=y_pred    
    data.sort_values(by='PREDICTED',inplace=True)
    data['DEFAULTS_SHARE']=data[['FACT']].cumsum(axis=0)
    data['DEFAULTS_SHARE'] = data['DEFAULTS_SHARE']/data['DEFAULTS_SHARE'].max()
    
    dr=data['FACT'].mean()
    
    model_y=data['DEFAULTS_SHARE'].values
    model_x=np.arange(len(model_y))/len(model_y)
    
    our_model=model_y.mean()
    best_model=1-dr/2    
    gini=(our_model-0.5)/(best_model-0.5)
    
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
        




def VariablesInteraction(X_train,y_train, X_test=None, y_test=None, classes=None,
                         figsize=[20,20], max_tree_depth=5, mode='full', dpath='mono'):
    plt.clf()
    possible_pairs=[]
    if mode=='single':
        possible_pairs=list(it.combinations(X_train.columns, r=1))
    elif mode=='pairs':
        possible_pairs=list(it.combinations(X_train.columns, r=2))
    elif mode=='full':
         possible_pairs=list(it.combinations(X_train.columns, r=1))+list(it.combinations(X_train.columns, r=2))
    else:
        raise BaseException('Wrong mode {}'.format(mode))
    
    if X_test is None or y_test is None:
            X_test=X_train
            y_test=y_train
    
    #return possible_pairs
    plt.figure(figsize=figsize)
    for i in range(len(possible_pairs)):
        pair=list(possible_pairs[i])
        
        X = X_train.loc[:, pair]
        y = y_train
        X_t=X_test.loc[:, pair]
        y_t=y_test
        
        if len(pair)==1:
            X['const']=0
            X_t['const']=0
            pair.append('const')

        # Train
        clf = DecisionTreeClassifier(max_depth=max_tree_depth).fit(X, y)
        # Plot the decision boundary
        h_g=min([3,len(possible_pairs)])
        plt.subplot(math.ceil(len(possible_pairs)/h_g), h_g, i+1)
        _plotTreePath(clf,pair,X, y, X_t, y_t, dpath)

    #plt.suptitle("Decision surface of a decision tree using paired features")
    
    plt.axis("tight")
    plt.show()
    
    
def _plotTreePath(clf, pair, X, y, X_t, y_t, dpath='mono'):
    n_classes = len(y.unique())
    plot_colors = "rb"
    plot_step = 0.02
    
    x_min, x_max = X.loc[:, pair[0]].min() - 1, X.loc[:, pair[0]].max() + 1
    y_min, y_max = X.loc[:, pair[1]].min() - 1, X.loc[:, pair[1]].max() + 1
    
    x_plot_step=(x_max-x_min)/1000
    y_plot_step=(y_max-y_min)/1000
    

    xx, yy = np.meshgrid(np.arange(x_min, x_max, x_plot_step), np.arange(y_min, y_max, y_plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    
    if dpath=='mono':
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z=np.array([p[0] for p in clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
    
    g=gini(y_pred=[p[1] for p in clf.predict_proba(X_t)],y_true=y_t)

    plt.xlabel(pair[0])
    plt.ylabel(pair[1])
    plt.title('Gini: %0.4f' % g)
    # Plot the testing points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y_t == i)
        plt.scatter(X_t.iloc[idx][pair[0]], X_t.iloc[idx][pair[1]], c=color, label=i,
                    cmap=plt.cm.RdBu, edgecolor='black', s=15)
    plt.legend(loc='lower right', borderpad=0, handletextpad=0)
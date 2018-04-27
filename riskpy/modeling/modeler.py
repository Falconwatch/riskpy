import statsmodels.formula.api as smf
from riskpy.graphs.graphs import roc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from ..utilities.common_metrics import gini
from ..graphs.graphs import rocs, roc
import pandas as pd
from ..utilities.common_metrics import vif
import matplotlib.pyplot as plt



class Modeler:
    def __init__(self, train_dataset, test_dataset, oot_dataset, target):
        self._train=train_dataset
        self._test=test_dataset
        self._oot=oot_dataset
        self._target=target
        self._fitted_model=None
        self._columns=None
        self._sw_log=None
        self._sw_models=None
        
    def ExploreOneFactorImportance(self, columns, changeSign=True):
        _multiplier=1
        if changeSign:
            multiplier=-1
            
        result=pd.DataFrame(index=columns)
        
        if self._train is not None:
            train_onefactor_gini=self._oneFactorGini(self._train, columns, multiplier)
            result['TRAIN']= [t[1] for t in train_onefactor_gini]
            
        if self._test is not None:
            test_onefactor_gini=self._oneFactorGini(self._test, columns, multiplier)
            result['TEST']= [t[1] for t in test_onefactor_gini]
            
        if self._oot is not None:
            oot_onefactor_gini=self._oneFactorGini(self._oot, columns, multiplier)
            result['OOT']= [t[1] for t in oot_onefactor_gini]

            return result    
        
    def Fit(self, columns): 
        self._columns=columns
        self._fitted_model=self._fitModel(columns)
        AIC=self._fitted_model.aic
        BIC=self._fitted_model.bic
        return AIC, BIC
    
    
    def GetModelSummary(self):
        if self._fitted_model is None:
            raise Exception('Modeller is not fitted')
        return self._fitted_model.summary()
    
    
    def GetModelQuality(self):
        if self._fitted_model is None:
            raise Exception('Modeller is not fitted')
            
        train_predicted=self._fitted_model.predict(self._train)
        train_fact=self._train[self._target]        
        test_predicted=self._fitted_model.predict(self._test)
        test_fact=self._test[self._target]        
        oot_predicted=self._fitted_model.predict(self._oot)
        oot_fact=self._oot[self._target]        
        
        return rocs([train_fact,test_fact, oot_fact],[train_predicted,test_predicted,oot_predicted],['g','r','y'],['train','test','oot'])
            
    def VIF(self):
        if self._fitted_model is None or self._columns is None:
            raise Exception('Modeller is not fitted')
        vifs=vif(self._train[self._columns])
        #return vifs
        return pd.DataFrame([v[1] for v in vifs], 
                            index=[v[0] for v in vifs], 
                            columns=['VIF'])
    
    
    def StepwiseSelection(self, factors, p_in=1, p_out=0.1):
        log=[]
        models=[]
        
        def Log(log_text, model):
            models.append(model)
            log.append(log_text)

            
        factors_importance=self._oneFactorGini(self._train, factors, -1)
        factors_importance.sort(key=lambda x: x[1],reverse=True)
        
        factors_in_model=dict()
        
        for factor_imp in factors_importance:
            factor=factor_imp[0]
            new_factors_set=list(factors_in_model.keys())+[factor]
            new_factors_fitted_model=self._fitModel(new_factors_set)
            p_values=self._getModelPvalues(new_factors_fitted_model)

            #Принимаем решение по переменной
            if p_values[factor]>p_in:
                #переменная не добавляется
                pass
            else:
                #переменная добавляется
                factors_in_model[factor]=p_values[factor]
                Log('{} added (p-value={})'.format(factor, p_values[factor]), new_factors_fitted_model)
            
            #принмаем решение по удалению переменных
            too_big_p_values = p_values[p_values>p_out]
            while len(too_big_p_values)>0:
                factor_to_remove=too_big_p_values.index[0]
                factor_to_remove_p_val=too_big_p_values[0]
                del factors_in_model[factor_to_remove]
                fitted_model=self._fitModel(list(factors_in_model.keys())) 
                p_values=self._getModelPvalues(fitted_model)
                too_big_p_values = p_values[p_values>p_out]
                Log('{} removed (p-value:{})'.format(factor_to_remove, factor_to_remove_p_val),fitted_model)
        
        self._sw_log=log
        self._sw_models=models
        return log, models
    
    def StepwiseSummary(self, figure_size=[10,10]):
        if self._sw_log is None or self._sw_models is None:
            raise Exception('Stepwise selection was not performed')
        
        sum_df=pd.DataFrame(self._sw_log, columns=['Added'])
        sum_df['Removed']=''
        sum_df.loc[sum_df['Added'].apply(lambda x: 'removed' in x), 'Removed']=sum_df['Added']
        sum_df.loc[sum_df['Added'].apply(lambda x: 'removed' in x), 'Added']=''        
        sum_df['Model description']=[m.summary() for m in self._sw_models]
        sum_df['AIC']=[m.aic for m in self._sw_models]
        sum_df['BIC']=[m.bic for m in self._sw_models]
        sum_df['Variables count']=[len(m.params)-1 for m in self._sw_models]
        
        fig, ax = plt.subplots(figsize=figure_size)
        ax.plot(sum_df['AIC'], label='AIC',color='green')
        ax.plot(sum_df['BIC'], label='BIC', color='yellow')
        plt.legend(loc='lower left')
        ax.set(ylabel = 'AIC/BIC') 
        ax2=ax.twinx()
        ax2.plot(sum_df['Variables count'], label='Variables', color='red')
        ax2.grid(b=None)
        ax2.set(ylabel = 'Variables count') 
        plt.legend(loc='upper left')
        plt.show()

        return sum_df

    ###private###   
    
    def _fitModel(self, columns):
        
        print(columns)
        columns_in_model_str=" + ".join(columns)
        print(columns_in_model_str)
        model=smf.logit("{} ~ {}".format(self._target, columns_in_model_str), data=self._train)
        fitted_model=model.fit()
        return fitted_model
    
    def _getModelPvalues(self, fitted_model):
        p_values=fitted_model.pvalues
        p_values.sort_values(ascending=False, inplace=True)
        return p_values  
    
    def _oneFactorGini(self, data, columns, multiplier):
        onefactor_result=[]
        for column in columns:
            g=gini(y_pred=multiplier*data[column],y_true=data[self._target]) 
            onefactor_result.append([column, g])
        return onefactor_result
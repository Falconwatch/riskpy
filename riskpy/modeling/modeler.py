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
from sklearn.metrics import r2_score



class Modeler:
    def __init__(self, train_dataset, test_dataset, oot_dataset, target, kind='class'):
        self._train=train_dataset
        self._test=test_dataset
        self._oot=oot_dataset
        self._target=target
        self._kind=kind
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
            train_onefactor_imp=self._oneFactorImportance(self._train, columns, multiplier)
            result['TRAIN']= [t[1] for t in train_onefactor_imp]
            
        if self._test is not None:
            train_onefactor_imp=self._oneFactorImportance(self._test, columns, multiplier)
            result['TEST']= [t[1] for t in train_onefactor_imp]
            
        if self._oot is not None:
            train_onefactor_imp=self._oneFactorImportance(self._oot, columns, multiplier)
            result['OOT']= [t[1] for t in train_onefactor_imp]

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
        fact_list = list()
        predict_list = list()
        color_list = list()
        names_list = list()
        
        train_predicted=self._fitted_model.predict(self._train)
        train_fact=self._train[self._target]        
        
        fact_list.append(train_fact)
        predict_list.append(train_predicted)
        color_list.append('g')
        names_list.append('train')
        
        try:
            test_predicted=self._fitted_model.predict(self._test)
            test_fact=self._test[self._target]     
            
            fact_list.append(test_fact)
            predict_list.append(test_predicted)
            color_list.append('r')
            names_list.append('test')
        except:
            test_predicted = None
            test_fact = None
            
        try:
            oot_predicted=self._fitted_model.predict(self._oot)
            oot_fact=self._oot[self._target]  
            
            fact_list.append(oot_fact)
            predict_list.append(oot_predicted)
            color_list.append('y')
            names_list.append('oot')
        except:
            oot_predicted = None
            oot_fact = None
        
        return rocs(fact_list, predict_list, color_list, names_list)
    
    def Predict(self, data=None):
        if data is not None:
            fact=[data[self._target].values]
            predict=[self._fitted_model.predict(data)]
            return fact, predict
        else:
            fact=[self._train[self._target].values, self._test[self._target].values, self._oot[self._target].values]
            predict=[self._fitted_model.predict(self._train), self._fitted_model.predict(self._test), self._fitted_model.predict(self._oot)]
            return fact, predict
    
    def VIF(self):
        if self._fitted_model is None or self._columns is None:
            raise Exception('Modeller is not fitted')
        vifs=vif(self._train[self._columns].dropna())
        #return vifs
        return pd.DataFrame([v[1] for v in vifs], 
                            index=[v[0] for v in vifs], 
                            columns=['VIF'])
    
    
    def StepwiseSelection(self, factors, p_in=1, p_out=0.1, kind='class'):
        log=[]
        models=[]
        
        def Log(log_text, model):
            models.append(model)
            log.append(log_text)

            
        factors_importance=self._oneFactorImportance(self._train, factors)
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
            while len(too_big_p_values)>0 and len(factors_in_model)>1:
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
    
    def _fitModel(self, columns, kind='class'):
        columns_in_model_str=" + ".join(columns)
        #print(columns_in_model_str)
        if self._kind=='class':
            model=smf.logit("{} ~ {}".format(self._target, columns_in_model_str), data=self._train)
        else:
            model=smf.ols("{} ~ {}".format(self._target, columns_in_model_str), data=self._train)
        fitted_model=model.fit()
        return fitted_model
    
    def _getModelPvalues(self, fitted_model):
        p_values=fitted_model.pvalues
        p_values.sort_values(ascending=False, inplace=True)
        return p_values[p_values.index!='Intercept']  
    
    def _oneFactorGini(self, data, columns, multiplier):
        onefactor_result=[]
        for column in columns:
            tmp_data=data[[column, self._target]].dropna()
            g=gini(y_pred=multiplier*tmp_data[column],y_true=tmp_data[self._target]) 
            onefactor_result.append([column, g])
        return onefactor_result
    
    def _oneFactorR2(self, data, columns):
        onefactor_result=[]
        for column in columns:
            tmp_data=data[[column, self._target]].dropna()
            r2=r2_score(y_pred=tmp_data[column],y_true=tmp_data[self._target]) 
            onefactor_result.append([column, r2])
        return onefactor_result
    
    def _oneFactorImportance(self, data, columns, multiplier=-1):
        if self._kind=='class':
            return self._oneFactorGini(data, columns, multiplier)
        else:
            return self._oneFactorR2( data, columns)

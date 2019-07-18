import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from riskpy.utilities.common_metrics import gini, vif
from riskpy.graphs.graphs import rocs
from statsmodels.stats.outliers_influence import variance_inflation_factor


# vif = VIF(threshold=5).fit(X_train) - if verbose=True then printing columns to be dropped
# X_train = vif.transform(X_train)
# X_test = vif.transform(X_test)
# X_oot = vif.transform(X_oot)
class VIF:
    def __init__(self, threshold=5.0):
        self.threshold = threshold
        self._init_columns = None
        self._end_columns = None
        self._drop_cols = []

    def fit(self, data, drop_cols=[], verbose=True):
        if verbose:
            print('ReduceVif fit. It can be very long if there is a lot of data')
        self._drop_cols = list(set(drop_cols).intersection(list(data.columns.tolist())))
        self._init_columns = data.drop(self._drop_cols, axis=1).columns.tolist()
        self._calculate_vif(data.drop(self._drop_cols, axis=1), self.threshold, verbose=verbose)
        return self

    def transform(self, data, verbose=True):
        if verbose:
            print('ReduceVif transform')
        return data[self._end_columns + self._drop_cols].copy()

    def _calculate_vif(self, data, threshold=5.0, verbose=True):
        dropped = True
        while dropped:
            variables = data.columns
            dropped = False
            vif = [variance_inflation_factor(data[variables].values, data.columns.get_loc(var)) for var in data.columns]
            max_vif = max(vif)
            if max_vif > threshold:
                maxloc = vif.index(max_vif)
                if verbose:
                    print('Dropping {} with vif={}'.format(data.columns[maxloc], max_vif))
                data = data.drop([data.columns.tolist()[maxloc]], axis=1)
                dropped = True
        self._end_columns = data.columns.tolist()
        return data

    @staticmethod
    def vif(data):
        variables = data.columns
        return pd.DataFrame([(var, variance_inflation_factor(data[variables].values, data.columns.get_loc(var))) for var in data.columns])


# cor = Correlations(threshold=0.95).fit(X_train) - if verbose=True then printing num of columns to be dropped
# X_train = cor.transform(X_train)
# X_test = cor.transform(X_test)
# X_oot = cor.transform(X_oot)
class Correlations:
    ''' Calculate correlations between factors
        Keyword arguments:
        threshold -- a correlation coefficient value above which is considered high enough to remove factor
    '''
    def __init__(self, threshold=0.95):
        self._threshold = threshold
        self._col_to_del = None
        self._end_cols = None
        self._init_cols = None
        self._drop_cols = None

    def fit(self, train_dataset, drop_cols=[], verbose=True, plot=False):
        self._drop_cols = list(set(drop_cols).intersection(list(train_dataset.columns.tolist())))
        self._init_cols = train_dataset.drop(self._drop_cols, axis=1).columns.tolist()
        train_corr = train_dataset.drop(self._drop_cols, axis=1).corr()
        upper = train_corr.abs().where(np.triu(np.ones(train_corr.shape), k=1).astype(np.bool))
        self._col_to_del = [c for c in upper.columns if any(upper[c] > self._threshold)]
        self._end_cols = train_dataset.drop(self._drop_cols + self._col_to_del, axis=1).columns.tolist()
        if verbose:
            print('Fitted train dataset. {} columns to be removed'.format(len(self._col_to_del)))
        if plot:
            import seaborn as sns
            plt.figure(figsize=[20, 20])
            sns.heatmap(train_corr)
            plt.title('Корреляционная матрица')
            plt.show()
        return self

    def transform(self, data, verbose=True):
        if verbose:
            print('Dropping correlated columns')
        return data[self._end_cols + self._drop_cols].copy()


class Modeler:
    def __init__(self, train_dataset=None, test_dataset=None, target=None, oot_dataset=None,  period=None, kind='class'):
        # Global vars
        self._kind = kind
        self._multiplier = -1
        # Datasets
        self._train = train_dataset
        self._test = test_dataset
        self._oot = oot_dataset
        self._period = period
        self._target = target
        # Fitted model
        self._fitted_model = None
        self._columns = None
        # Coefficients of the features calculated in last fitting
        self.coefs = None
        self.intercept = None
        self.classes = None
        self.aic = None
        self.bic = None
        # Logs
        self._sw_log = None
        self._sw_models = None

    def fit(self, columns, verbose=True, disp=False):
        ''' Fit model
        Keyword arguments:
        columns -- list of factors in model to fit
        '''
        if self._train is not None:
            raise Exception('Train dataset has not benn added')
        if self._kind == 'class':
            self._fitted_model = sm.Logit(self._train[self._target].values, sm.add_constant(self._train[columns])).fit(disp=disp)
        else:
            self._fitted_model = sm.OLS(self._train[self._target].values, sm.add_constant(self._train[columns])).fit(disp=disp)
        self.classes = self._train[self._target].values
        self._columns = columns
        self.aic = self._fitted_model.aic
        self.bic = self._fitted_model.bic
        if verbose:
            print('Model has been fitted on {} columns. AIC: {}, BIC: {}.'.format(len(columns), self.aic, self.bic))
        return self._fitted_model

    def predict(self, data='train', facts=False):
        ''' Predict with fitted model
         Keyword arguments:
         data -- data to make predictions, if None - predict on all given samples: train, test, oot (default None)
        '''
        if self._columns is not None:
            raise Exception('Modeler has not benn fitted')
        if data in ['train', 'TRAIN', 'Train']:
            if facts:
                return self._train[self._target].values, self._fitted_model.predict(self._train[self._columns])
            return self._fitted_model.predict(self._train[self._columns])
        elif data in ['test', 'TEST', 'Test']:
            if facts:
                return self._test[self._target].values, self._fitted_model.predict(self._test[self._columns])
            return self._fitted_model.predict(self._test[self._columns])
        elif data in ['oot', 'OOT', 'Oot']:
            if facts:
                return self._oot[self._target].values, self._fitted_model.predict(self._oot[self._columns])
            return self._fitted_model.predict(self._oot[self._columns])
        else:
            return self._fitted_model.predict(data[self._columns])

    def get_score(self, data='train'):
        if data in ['train', 'TRAIN', 'Train']:
            return gini(y_true=self._train[self._target], y_pred=self.predict(data='train'))
        elif data in ['test', 'TEST', 'Test']:
            return gini(y_true=self._train[self._target], y_pred=self.predict(data='test'))
        elif data in ['oot', 'OOT', 'Oot']:
            return gini(y_true=self._train[self._target], y_pred=self.predict(data='oot'))
        else:
            return gini(y_true=data[self._target], y_pred=self.predict(data=data))

    def cv_score(self, n_splits=3):
        if self._columns is not None:
            raise Exception('Modeler has not benn fitted')
        skf = StratifiedKFold(n_splits=n_splits).split(
            self._train[self._columns].append(self._test[self._columns]),
            self._train[self._target].append(self._test[self._target]))

        ginis = []
        for train_index, test_index in skf:
            if self._kind == 'class':
                fitted_model = sm.Logit(
                    self._train[self._target].append(self._test[self._target])[train_index].values,
                    sm.add_constant(
                        self._train[self._columns].append(self._test[self._columns])[train_index]
                    )).fit(disp=False)
            else:
                fitted_model = sm.OLS(
                    self._train[self._target].append(self._test[self._target])[train_index].values,
                    sm.add_constant(
                        self._train[self._columns].append(self._test[self._columns])[train_index]
                    )).fit(disp=False)
            ginis.append(
                gini(
                    y_true=self._train[self._target].append(self._test[self._target])[train_index],
                    y_pred=fitted_model.predict(self._train[self._columns].append(self._test[self._columns])[test_index])))
        return ginis

    def Fit(self, columns):
        ''' Fit model
        Keyword arguments:
        columns -- list of factors in model to fit
        '''
        self._columns = columns
        self._fitted_model = self._fitModel(columns)
        self.aic = self._fitted_model.aic
        self.bic = self._fitted_model.bic
        return self.aic, self.bic

    def ExploreOneFactorImportance(self, columns, changeSign=True):
        ''' Return factor importance for all given samples
        Keyword arguments:
        columns -- list of factors to check
        changeSign -- indicate whether change the sign of result (default True)
        '''
        multiplier = -1 if changeSign else 1

        result = pd.DataFrame(index=columns)

        if self._train is not None:
            train_one_factor_imp = self._oneFactorImportance(self._train, columns, multiplier)
            result['TRAIN'] = [t[1] for t in train_one_factor_imp]

        if self._test is not None:
            train_one_factor_imp = self._oneFactorImportance(self._test, columns, multiplier)
            result['TEST'] = [t[1] for t in train_one_factor_imp]

        if self._oot is not None:
            train_one_factor_imp = self._oneFactorImportance(self._oot, columns, multiplier)
            result['OOT'] = [t[1] for t in train_one_factor_imp]

        return result

    def ExploreOneFactorImportanceIntervals(self, columns, changeSign=True):
        ''' Return factor importance for all given samples
        Keyword arguments:
        columns -- list of factors to check
        changeSign -- indicate   whether change the sign of result (default True)
        '''
        if self._oot is None and self._test is None and self._train is None:
            raise Exception('Modeller has not have any data')
        multiplier = -1 if changeSign else 1

        if self._period is not None:
            results = []
            if self._train is not None:
                for period in self._train[self._period].unique():
                    for column in columns:
                        tmp_data = self._train.loc[self._train[self._period] == period, [column, self._target]].dropna()
                        g = gini(y_pred=multiplier * tmp_data[column], y_true=tmp_data[self._target])
                        results.append(('TRAIN', period, column, g))

            if self._test is not None:
                results = []
                for period in self._test[self._period].unique():
                    for column in columns:
                        tmp_data = self._test.loc[self._test[self._period] == period, [column, self._target]].dropna()
                        g = gini(y_pred=multiplier * tmp_data[column], y_true=tmp_data[self._target])
                        results.append(('TEST', period, column, g))

            if self._oot is not None:
                for period in self._oot[self._period].unique():
                    for column in columns:
                        tmp_data = self._oot.loc[self._oot[self._period] == period, [column, self._target]].dropna()
                        g = gini(y_pred=multiplier * tmp_data[column], y_true=tmp_data[self._target])
                        results.append(('OOT', period, column, g))

        results_df = pd.DataFrame(results)
        results_df.columns = ['subset', 'period', 'column', 'gini']
        results_df.set_index(['period', 'column'], inplace=True)

        subsets = []
        for subset in results_df['subset'].unique():
            subsets.append(results_df.loc[results_df['subset'] == subset, ['gini']].rename(columns={'gini': subset}))
        return pd.concat(subsets, axis=1).sort_index()

    def GetModelSummary(self):
        ''' Return fitted model summary'''
        if self._fitted_model is None:
            raise Exception('Modeller is not fitted')
        return self._fitted_model.summary()

    def GetModelQuality(self):
        ''' Plot ROC-curves for fitted model on all given samples (train, test, out of time)'''
        if self._fitted_model is None:
            raise Exception('Modeller is not fitted')
        fact_list = list()
        predict_list = list()
        color_list = list()
        names_list = list()
        
        train_predicted = self._fitted_model.predict(self._train)
        train_fact = self._train[self._target]
        
        fact_list.append(train_fact)
        predict_list.append(train_predicted)
        color_list.append('g')
        names_list.append('train')
        
        try:
            test_predicted = self._fitted_model.predict(self._test)
            test_fact = self._test[self._target]
            
            fact_list.append(test_fact)
            predict_list.append(test_predicted)
            color_list.append('r')
            names_list.append('test')
        except:
            pass
            
        try:
            oot_predicted = self._fitted_model.predict(self._oot)
            oot_fact = self._oot[self._target]
            
            fact_list.append(oot_fact)
            predict_list.append(oot_predicted)
            color_list.append('y')
            names_list.append('oot')
        except:
            pass
        return rocs(fact_list, predict_list, color_list, names_list)


    def Predict(self, data=None):
        ''' Predict with fitted model
        
         Keyword arguments:
         data -- data to make predictions, if None - predict on all given samples: train, test, oot (default None)
        '''
        if data is not None:
            fact = [data[self._target].values]
            predict = [self._fitted_model.predict(data)]
            return fact, predict
        else:
            fact = [self._train[self._target].values, self._test[self._target].values, self._oot[self._target].values]
            predict = [self._fitted_model.predict(self._train), self._fitted_model.predict(self._test), self._fitted_model.predict(self._oot)]
            return fact, predict
    
    def VIF(self):
        ''' Evaluate VIF for factors in model'''
        if self._fitted_model is None or self._columns is None:
            raise Exception('Modeller is not fitted')
        vifs = vif(self._train[self._columns].dropna())
        #return vifs
        return pd.DataFrame([v[1] for v in vifs], 
                            index=[v[0] for v in vifs], 
                            columns=['VIF'])

    def StepwiseSelection(self, factors, p_in=1, p_out=0.1):
        ''' Do stepwise selection
        
         Keyword arguments:
         factors -- full factors list for stepwise selection
         p_in -- p-value to include (default 1)
         p_out -- p-value to exclude (default 0.1)
         kind -- type of regression, not fully implemented (default 'class')
        '''
        print('2')
        log = []
        models = []
        
        def Log(log_text, model):
            models.append(model)
            log.append(log_text)
        print('1')
            
        factors_importance = self._oneFactorImportance(self._train, factors)
        factors_importance.sort(key=lambda x: x[1], reverse=True)
        
        factors_in_model = dict()

        for factor_imp in factors_importance:
            factor = factor_imp[0]
            new_factors_set = list(factors_in_model.keys()) + [factor, ]
            new_factors_fitted_model = self._fitModel(new_factors_set)
            p_values = self._getModelPvalues(new_factors_fitted_model)

            # Принимаем решение по переменной
            if p_values[factor] > p_in:
                # переменная не добавляется
                pass
            else:
                # переменная добавляется
                factors_in_model[factor] = p_values[factor]
                Log('{} added (p-value={})'.format(factor, p_values[factor]), new_factors_fitted_model)

                too_big_p_values = p_values[p_values > p_out]

                while len(too_big_p_values) > 0 and len(factors_in_model) > 1:
                    factor_to_remove, factor_to_remove_p_val = too_big_p_values.index[0], too_big_p_values[0]
                    del factors_in_model[factor_to_remove]
                    fitted_model = self._fitModel(list(factors_in_model.keys()))
                    p_values = self._getModelPvalues(fitted_model)
                    too_big_p_values = p_values[p_values > p_out]
                    Log('{} removed (p-value:{})'.format(factor_to_remove, factor_to_remove_p_val), fitted_model)

        self._sw_log = log
        self._sw_models = models
        return log, models

    def StepwiseSelectionInterval(self, factors, p_in=1, p_out=0.1):
        ''' Do stepwise selection
         Keyword arguments:
         factors -- full factors list for stepwise selection
         p_in -- p-value to include (default 1)
         p_out -- p-value to exclude (default 0.1)
         kind -- type of regression, not fully implemented (default 'class')
        '''
        log, models = [], []

        def Log(log_text, model):
            models.append(model)
            log.append(log_text)

        factors_importance = pd.DataFrame(
            self._oneFactorImportanceInterval(self._train, factors)
        ).rename(columns={0: 'Period', 1: 'Factor', 2: 'Gini'})
        factors_importance = factors_importance.groupby('Factor')['Gini'].mean()
        factors_importance = factors_importance.sort_values(ascending=False).index.tolist()

        factors_in_model = dict()

        for factor in factors_importance:
            new_factors_set = list(factors_in_model.keys()) + [factor, ]
            new_factors_fitted_model = self._fitModel(new_factors_set)
            p_values = self._getModelPvalues(new_factors_fitted_model)

            # Принимаем решение по переменной
            if p_values[factor] > p_in:
                # переменная не добавляется
                pass
            else:
                # переменная добавляется
                factors_in_model[factor] = p_values[factor]
                Log('{} added (p-value={})'.format(factor, p_values[factor]), new_factors_fitted_model)

                too_big_p_values = p_values[p_values > p_out]

                while len(too_big_p_values) > 0 and len(factors_in_model) > 1:
                    factor_to_remove, factor_to_remove_p_val = too_big_p_values.index[0], too_big_p_values[0]
                    del factors_in_model[factor_to_remove]
                    fitted_model = self._fitModel(list(factors_in_model.keys()))
                    p_values = self._getModelPvalues(fitted_model)
                    too_big_p_values = p_values[p_values > p_out]
                    Log('{} removed (p-value:{})'.format(factor_to_remove, factor_to_remove_p_val), fitted_model)

        self._sw_log = log
        self._sw_models = models
        return log, models
    
    def StepwiseSummary(self, figure_size=(10, 10), plot=True, add_gini=True):
        ''' Return stepwise selection results and plot stepwise path
        
         Keyword arguments:
         figure_size -- size of plot figure (default [10,10])
        '''
        if (self._sw_log is None) or (self._sw_models is None):
            raise Exception('Stepwise selection was not performed')
        
        sum_df = pd.DataFrame(self._sw_log, columns=['Added'])
        sum_df['Removed'] = ''
        sum_df.loc[sum_df['Added'].apply(lambda x: 'removed' in x), 'Removed'] = sum_df['Added']
        sum_df.loc[sum_df['Added'].apply(lambda x: 'removed' in x), 'Added'] = ''
        sum_df['Model description'] = [m.summary() for m in self._sw_models]
        sum_df['AIC'] = [m.aic for m in self._sw_models]
        sum_df['BIC'] = [m.bic for m in self._sw_models]
        if add_gini:
            sum_df['Gini TRAIN'] = [
                gini(
                    y_true=self._train[self._target],
                    y_pred=1/(1 + np.exp(-np.dot(self._train[m.params.index[1:]], m.params[1:]) + m.params[0])))
                for m in self._sw_models]
            if self._test is not None:
                sum_df['Gini TEST'] = [
                    gini(
                        y_true=self._test[self._target],
                        y_pred=1/(1 + np.exp(-np.dot(self._test[m.params.index[1:]], m.params[1:]) + m.params[0])))
                    for m in self._sw_models]
            if self._oot is not None:
                sum_df['Gini OOT'] = [
                    gini(
                        y_true=self._oot[self._target],
                        y_pred=1 / (1 + np.exp(-np.dot(self._oot[m.params.index[1:]], m.params[1:]) + m.params[0])))
                    for m in self._sw_models]
        sum_df['Variables count'] = [len(m.params)-1 for m in self._sw_models]

        if plot:
            fig, ax = plt.subplots(figsize=figure_size)
            ax.plot(sum_df['AIC'], label='AIC',color='green')
            ax.plot(sum_df['BIC'], label='BIC', color='yellow')
            plt.legend(loc='lower left')
            ax.set(ylabel='AIC/BIC')
            ax2 = ax.twinx()
            ax2.plot(sum_df['Variables count'], label='Variables', color='red')
            ax2.grid(b=None)
            ax2.set(ylabel='Variables count')
            plt.legend(loc='upper left')
            plt.show()

        return sum_df

    ###private###   
    
    def _fitModel(self, columns):
        if self._kind == 'class':
            return sm.Logit(self._train[self._target].values, sm.add_constant(self._train[columns])).fit(disp=False)
        else:
            return sm.OLS(self._train[self._target].values, sm.add_constant(self._train[columns])).fit(disp=False)
    
    def _getModelPvalues(self, fitted_model):
        p_values = fitted_model.pvalues
        p_values.sort_values(ascending=False, inplace=True)
        return p_values[p_values.index != 'Intercept']

    def _oneFactorImportance(self, data, columns, multiplier=-1):
        if self._kind == 'class':
            return self._oneFactorGini(data, columns, multiplier)
        else:
            return self._oneFactorR2(data, columns)

    def _oneFactorGini(self, data, columns, multiplier=-1):
        onefactor_result = []
        for column in columns:
            tmp_data = data[[column, self._target]].dropna()
            g = gini(y_pred=multiplier*tmp_data[column], y_true=tmp_data[self._target])
            onefactor_result.append([column, g])
        return onefactor_result

    def _oneFactorR2(self, data, columns):
        onefactor_result = []
        for column in columns:
            tmp_data = data[[column, self._target]].dropna()
            r2 = r2_score(y_pred=tmp_data[column], y_true=tmp_data[self._target])
            onefactor_result.append([column, r2])
        return onefactor_result

    def _oneFactorImportanceInterval(self, data, columns, multiplier=-1):
        if self._kind != 'class':
            raise Exception('Function work only with classification')
        if self._period is None:
            raise Exception('Modeller did not have period column')
        results = []
        for period in data[self._period].unique():
            ofgi = self._oneFactorGini(data[data[self._period] == period], columns, multiplier)
            for x in ofgi:
                results.append([period, x[0], x[1]])
        return results

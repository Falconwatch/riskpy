import numpy as np
import warnings
import statsmodels.api as sm


# Обратное преобразования для трансформации Бокса-Кокса
def invboxcox(y, lmbda):
    """
    Inverse Box-Cox transformation
    :param y: Box-Cox transformed TimeSeries
    :param lmbda: Box-Cox transformation parameter
    :return: Original TimeSeries
    """
    if lmbda == 0:
        return np.exp(y)
    else:
        return np.exp(np.log(lmbda * y + 1) / lmbda)

# Подбирает лучший набор параметров SARIMA модели из предложенного списка
def sarima_fit(time_series, parameters_list, d, D, season_len, save_all=True):
    """
    Fits SARIMA for all given parameters combinations
    :param time_series: TimeSeries to analyze
    :param parameters_list: List of parameters combinations: [p,q,P,Q]
    :param d: Order of integration
    :param D: Order of cyclic integration
    :param season_len: Cycle lenght
    :param save_all: Whether to save fitted models fully or only params and AICs
    :return: All fitted models info, best model and its parameters
    """
    results = []
    best_aic = float("inf")
    warnings.filterwarnings('ignore')

    for param in parameters_list:
        # try except нужен, потому что на некоторых наборах параметров модель не обучается
        try:
            model = sm.tsa.statespace.SARIMAX(time_series, order=(param[0], d, param[1]),
                                              seasonal_order=(param[2], D, param[3], season_len)).fit(disp=-1)
        # выводим параметры, на которых модель не обучается и переходим к следующему набору
        except ValueError:
            print('wrong parameters:', param)
            continue
        aic = model.aic
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param

        if save_all:
            results.append([param, model])
        else:
            results.append([param, model.aic])

    warnings.filterwarnings('default')

    return results,best_model,best_param

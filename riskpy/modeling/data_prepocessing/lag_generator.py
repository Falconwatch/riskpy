import pandas as pd


def add_lags(data, target_name=None, depth=4):
    """
    Adds lags to dataframe
    :param data: Dataset
    :param target_name: taget variable
    :param steps: Lags depth
    :param find_best_lags: Indicate whether algorithm must select best lags or not
    :return: Dataframe with lags
    """
    # Поиск возможных лагов
    found_lags = __find_lags_in_frame(data, target_name, depth)
    # Трансформация данных согласно найденным лагам
    transformed_data = __transform_data(data, found_lags, target_name)
    return transformed_data


# Поиск лагов по фрейму
def __find_lags_in_frame(data, target_name, max_depth):
    """
    Finds lags in Dataframe
    :return: Data about found lags
    :param max_depth: Max possible lag depth
    :param data: Dataframe
    :param target_name: Target variable
    """
    # для всех переменных осуществляется подбор
    result = list()
    for column in [c for c in data.columns if not c == target_name]:
        lag = list(range(0, max_depth + 1))
        result.append([column, lag])
    return result


# Трансорфмация данных по итогам поиска лагов - сдвиг колонок
def __transform_data(data, lags, target_name):
    # вырезаем из данных индекс
    index_column = data.index
    if index_column.name==None:
        index_column.name='index'

    data = data.reset_index().drop([index_column.name], axis=1)

    # Добавляем лаги в данные
    max_lag = 0
    a = [lag[1] for lag in lags]
    for i in range(len(a)):
        b = a[i]
        for j in range(len(b)):
            c = b[j]
            if c > max_lag:
                max_lag = c
    # Создаём результирующий датасет и добавляем туда целевую переменную
    result = pd.DataFrame()
    if target_name != None:
        target = data.loc[max_lag:, target_name]
        result[target_name] = target.as_matrix()

    # Берём кажду переменную по-отдельности
    for lag in lags:
        var_name = lag[0]
        var_steps = lag[1]
        # Для каждой переменной обрабатываем каждый найденный шаг
        for step in var_steps:
            if step == 0:
                var = data.loc[max_lag - step:, var_name][:]
            else:
                var = data.loc[max_lag - step:, var_name][:-step]
            result[var_name + "_" + str(step)] = var.as_matrix()

    # устанавливаем индекс
    result[index_column.name] = index_column[max_lag:].values
    result.set_index(index_column.name, inplace=True)

    return result

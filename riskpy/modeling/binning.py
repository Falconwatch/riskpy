import numpy as np
import pandas as pd
import math as math
import pickle
from sklearn import tree
from sklearn.linear_model import LinearRegression
from enum import Enum
import multiprocessing as mp


class BinnerType(Enum):
    IV = 1
    R2 = 2


class Binner:
    def __init__(self, binner_type=BinnerType.IV, good_mark=0, bad_mark=1):
        """
        Create binner instance
        :param binner_type: Type of binner (target metric: IV or R2)
        :param good_mark: Target variable value for "good" observations
        :param bad_mark: Target variable value for "bad" observations
        """
        if good_mark == bad_mark:
            raise Exception('Classes cant be equal!!!')
        self._good_v = good_mark
        self._bad_v = bad_mark
        self._binner_type = binner_type
        self._fitted_bins = None
        self._target_variable = None
        self._exclude = []

    def __eq__(self, other):
        assert (self._fitted_bins is not None) and (other._fitted_bins is not None), "Binner is not fitted"
        assert len(self._fitted_bins) == len(other._fitted_bins), "Not equal amount of features binned"
        flag = True
        for i in range(len(self._fitted_bins)):
            if (self._fitted_bins[i]._name != other._fitted_bins[i]._name) or \
                    (self._fitted_bins[i]._woes != other._fitted_bins[i]._woes) or \
                    (self._fitted_bins[i]._gaps != other._fitted_bins[i]._gaps):
                flag = False
        return flag

    # Фитинг биннера к данным
    def fit(self, data, target, power=5, binning_settings=(), exclude=[], verbose=False):
        """
        Fit binner to data
        :param data: Source data
        :param target: Target variable
        :param power: Max depth of splitting
        :param binning_settings: Additional parameters for binning
        :param exclude: Columns in table which are not for binning
        :return:
        """
        self._target_variable = target
        self._exclude = exclude
        bin_data = list()
        for column in [x for x in data.columns if x not in [target, ] + self._exclude]:
            if verbose:
                print(column)
            variable_settings = None
            settings_list = [bs for bs in binning_settings if bs._variable_name == column]
            if len(settings_list) == 1:
                variable_settings = settings_list[0]
            x = data[column]
            y = data[target]
            w = self._bin(x, y, column, settings=variable_settings, power=power)
            w._name = column
            bin_data.append(w)
        self._fitted_bins = bin_data
        return bin_data

    # Фитинг биннера к данным в мультипоточном режиме
    def fit_mp(self, data, target, power=5, binning_settings=(), exclude=[], n_jobs=1):
        """
        Fit binner to data with multiprocessing
        :param data: Source data
        :param target: Target variable
        :param power: Max depth of splitting
        :param binning_settings: Additional parameters for binning
        :param exclude: Columns in table which are not for binning
        :return:
        """
        self._target_variable = target
        self._exclude = exclude

        def find_settings(column_name):
            settings_list = [bs for bs in binning_settings if bs._variable_name == column_name]
            if len(settings_list) == 1:
                variable_settings = settings_list[0]
                return variable_settings
            return None

        args = [(data[column], data[target], column, power, find_settings(column)) for column in
                [x for x in data.columns if x not in [target, ] + self._exclude]]

        pool = mp.Pool(n_jobs)
        bin_data = pool.starmap(self._bin, args)
        pool.close()
        pool.terminate()

        self._fitted_bins = bin_data
        return bin_data

    # Применение бинов к данным - возвращает WOE факторы
    def transform(self, data_in, exclude=[]):
        """
        Transform data
        :param data_in: Data to be transformed
        :param exclude: List of excluded variables (will be ignored during transformation)
        :return: Transformed data
        """
        if self._fitted_bins is None or self._target_variable is None:
            raise Exception("Binner is not fitted!")

        target = self._target_variable
        data_splitting = self._fitted_bins
        data = data_in.copy()
        for ds in [bw for bw in data_splitting if
                   (bw._iv >= 0.00) & (str(bw._name) not in " ".join(exclude + self._exclude))]:
            var = str(ds._name)
            bins = [a for a in zip(ds._gaps, ds._woes)]
            # not nan
            for bi in [bb for bb in bins if bb[0][0] is not None]:
                low = bi[0][0]
                high = bi[0][1]
                data.loc[(data[var] >= low) & (data[var] < high) & (~pd.isnull(data[var])), var + '_woe'] = bi[1]

            # nan
            try:
                data.loc[np.isnan(data[var]), var + "_woe"] = [b for b in bins if b[0][0] is None][0][1]
            except:
                pass
        learn_columns = [col for col in data.columns if ('woe' in col)] + self._exclude
        learn_columns.append(target)
        learn_data = data[learn_columns]
        return learn_data

    # Возврат полученных бинов
    def get_fitted_bins(self):
        """
        Get fitted bins
        :return: Fitted bins
        """
        return self._fitted_bins

    # SQL-скрипт с бинингом
    def to_sql(self):
        """
        Generate SQL-script for WoE-transformation
        :return: SQL-script
        """
        sql = ''
        for bining in self._fitted_bins:
            sql_variable = 'case \n'
            for i in range(len(bining._gaps)):
                gap = bining._gaps[i]
                woe = bining._woes[i]
                if gap[0] == None:
                    a = "    when {} is NULL then {} \n".format(bining._name, woe)
                    sql_variable += a
                else:
                    a = "    when {}>{} and {}<={} then {} \n".format(bining._name, gap[0], bining._name, gap[1], woe)
                    sql_variable += a
            sql_variable += ' end, \n'
            sql += sql_variable
        return sql

    # Сохранение биннера в файл
    def to_file(self, filename='bins.prdb'):
        """
        Save binner to file
        :param filename: Path to file
        """
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    # Разбиение на бины одной переменной
    def _bin(self, x, y, variable_name, power=2, settings=None):
        check_mono = True
        min_leaf_ratio = 0.1
        if settings is not None:
            check_mono = settings._monotone
            min_leaf_ratio = settings._min_leaf_ratio

        # инициализируем результаты
        best_gaps = []
        best_gaps_shares = []
        best_gaps_woe = []
        best_iv = 0
        best_gaps_counts = []
        best_r2 = 0
        best_gaps_counts_shares = []
        best_gaps_avg = []
        best_hhi = 1

        # создаём переменную группировки - плохие/хорошие
        y_gr = y.copy()
        y_gr[y > np.average(y)] = 1
        y_gr[y <= np.average(y)] = 0

        # переименовываем серии для удобства обращения в дальнейшем
        x.rename("x", inplace=True)
        y.rename("y", inplace=True)
        y_gr.rename("y_gr", inplace=True)
        all_set = pd.concat([x, y, y_gr], axis=1)

        data_goods = all_set[y_gr == self._good_v]
        data_bads = all_set[y_gr == self._bad_v]

        # Не nan:
        clear_data = all_set.dropna()

        # обрабатываем nan:
        if x.isnull().sum() > 0:
            dirty_set = all_set[pd.isnull(all_set['x'])]
            dirty_bads = dirty_set[dirty_set['y_gr'] == self._bad_v]
            dirty_goods = dirty_set[dirty_set['y_gr'] == self._good_v]
            dirty_woe, dirty_goods_share, dirty_bads_share = _calc_WoE(gap_bads=dirty_bads, bads=data_bads,
                                                                       gap_goods=dirty_goods, goods=data_goods)
            dirty_iv = (dirty_goods_share - dirty_bads_share) * dirty_woe
            dirty_counts_shares = (len(dirty_goods)) / (len(dirty_bads) + 0.000000001)

        # строим деревья разной глубины, ища наилучшее разбиение на чистых (без пустого) данных
        for depth in range(1, power + 1):

            # Строим дерево
            dt = tree.DecisionTreeClassifier(max_depth=depth,
                                             min_samples_leaf=max(int(min_leaf_ratio * len(all_set['y'])), 1))
            dt.fit(clear_data['x'][:, None], clear_data['y_gr'])
            # Сохраняем полученное разбиение
            gaps = self._get_gaps(dt)
            # Считаем характеристики разбиения
            gaps_shares, gaps_woe, gaps_counts, gaps_counts_shares, gaps_avg = self._gaps_metrics(gaps, clear_data,
                                                                                                  all_set[
                                                                                                      all_set[
                                                                                                          'y_gr'] == self._good_v],
                                                                                                  all_set[all_set[
                                                                                                              'y_gr'] == self._bad_v])

            temp_mono = False

            if self._binner_type == BinnerType.IV:
                temp_mono = _is_monotonik(vals=gaps_counts_shares)

            if self._binner_type == BinnerType.R2:
                temp_mono = _is_monotonik(gaps_avg)

            is_mono = temp_mono or not check_mono  # Монотонность

            # добавляем пропущенных
            if x.isnull().sum() > 0:
                gaps.append([None, None])
                gaps_shares.append([dirty_goods_share, dirty_bads_share])
                gaps_counts.append([len(dirty_goods), len(dirty_bads)])
                gaps_woe.append(dirty_woe)
                gaps_counts_shares.append(dirty_counts_shares)

            # GINI разбиения
            gini = _calc_gini(gaps_counts)

            # HHI
            hhi = _calc_HHI(gaps_counts)

            # IV по всем бинам
            ivs = [(gs[0] - gs[1]) * gw for gs, gw in zip(gaps_shares, gaps_woe)]
            iv = np.sum(ivs)
            # Случай, когда максимизируем R2
            if self._binner_type == BinnerType.R2:
                # создаём объект-кандидат на возвращение
                b = Binning(gaps, gaps_woe, iv, gaps_shares, gaps_counts, gini, gaps_counts_shares, gaps_avg)
                # R2 по всем
                r2 = _calc_r2(x, y, b)
                # Выбор лучшего по R2
                if r2 > best_r2 and is_mono:
                    best_gaps = gaps
                    best_gaps_shares = gaps_shares
                    best_gaps_woe = gaps_woe
                    best_gaps_counts = gaps_counts
                    best_iv = iv
                    best_r2 = r2
                    best_gaps_counts_shares = []
                    best_gaps_avg = gaps_avg
                    best_hhi = hhi

            # Случай, когда максимизируем IV
            if self._binner_type == BinnerType.IV:
                # Выбор лучшего по IV
                if iv > best_iv and is_mono:
                    best_gaps = gaps
                    best_gaps_shares = gaps_shares
                    best_gaps_woe = gaps_woe
                    best_gaps_counts = gaps_counts
                    best_gaps_counts_shares = gaps_counts_shares
                    best_iv = iv
                    best_hhi = hhi

        # лучшее разбиение
        best_b = Binning(best_gaps, best_gaps_woe, best_iv, best_gaps_shares, best_gaps_counts, gini,
                         best_gaps_counts_shares, best_gaps_avg, best_hhi)
        best_b._r2 = best_r2
        best_b._name = variable_name
        return best_b

    # Вытаскиваем промежутки из дерева
    def _get_gaps(self, estimator):
        maxval = 100000000000000
        # Вытаскиваем границы промежутков без дублей
        threshold = _delete_duplicates(estimator.tree_.threshold)
        # удаляем из границ -2 (искусственную границу)
        if -2 in threshold:
            threshold.remove(-2)
        # Добавляем верхнюю и нижнуюю границу, для закрытия крайних промежутков
        threshold.append(maxval)
        threshold.append(maxval * -1)
        # отбираем все границы и сортируем
        threshold = list(set(threshold))
        threshold.sort()
        # Составляем промежуки из границ
        gaps = []
        for i in range(len(threshold) - 1):
            gaps.append([threshold[i], threshold[i + 1]])
        return gaps

    # Вычислить метрики разбиения
    def _gaps_metrics(self, gaps, data, data_good, data_bad):
        woe = 0
        gaps_shares = []
        gaps_woe = []
        gaps_counts = []
        gaps_avg = []
        # для каждого промежутка:
        for gap in gaps:
            # Нижняя и верхняя граница промежутка
            st = gap[0]
            end = gap[1]
            # наблюдения в сегменте
            gap_data = data[(data['x'] >= st) & (data['x'] <= end)]
            # Хорошие
            gap_goods = gap_data[gap_data['y_gr'] == self._good_v]
            # Плохие
            gap_bads = gap_data[gap_data['y_gr'] == self._bad_v]

            # среднее значение
            gap_avg = np.average(gap_data)
            gaps_avg.append(gap_avg)

            # counts
            gaps_counts.append([len(gap_goods), len(gap_bads)])
            # woe для группы
            woe, gap_goods_share, gap_bads_share = _calc_WoE(gap_goods=gap_goods, goods=data_good,
                                                             gap_bads=gap_bads, bads=data_bad)
            # WOE для разбиения
            gaps_shares.append([gap_goods_share, gap_bads_share])
            gaps_woe.append(woe)

        gaps_counts_shares = [(gaps_count[0] + 0.000001) / (gaps_count[1] + 0.000001) for gaps_count in gaps_counts]

        return gaps_shares, gaps_woe, gaps_counts, gaps_counts_shares, gaps_avg


"""
##############################################################
                            Functions
##############################################################
"""


# calculate gini (second way)
def calc_gini_2(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred / G_true


# Вычисление R2 однофакторной модели
def _calc_r2(x, y, b):
    bins = [a for a in zip(b._gaps, b._woes)]
    data = pd.concat([x.rename('x'), y.rename('y')], axis=1)
    # преобразуем в соответсвии с бинингом
    # наны:
    data.loc[pd.isnull(data['x']), 'x_woe'] = bins[bins[0][0] is None][1]
    # не наны
    for bi in [bb for bb in bins if bb[0][0] is not None]:
        low = bi[0][0]
        high = bi[0][1]
        data.loc[(data['x'] >= low) & (data['x'] < high), 'x_woe'] = bi[1]
        # строим регрессию
    lin = LinearRegression(fit_intercept=True)
    lin.fit(data['x_woe'][:, None], y)
    r2 = lin.score(data['x_woe'][:, None], y)
    return r2


# Удаляем те значения, которые встретились больше 1 раза
def _delete_duplicates(arr):
    new_arr = list()
    for a in arr:
        if a in new_arr:
            new_arr.remove(a)
        else:
            new_arr.append(a)
    return new_arr


# Вычисление WoE
def _calc_WoE(gap_bads, bads, gap_goods, goods):
    bads_share = (len(gap_bads) + 0.5) / (len(bads) + 0.5) * 1.0
    goods_share = (len(gap_goods) + 0.5) / (len(goods) + 0.5) * 1.0
    val = (goods_share / bads_share)
    return math.log(val), goods_share, bads_share


# Херфиндаля-Хиршмена индекс
def _calc_HHI(gaps_counts):
    total = 0
    hh_shares = []
    for counts in gaps_counts:
        total += counts[0] + counts[1]
    for counts in gaps_counts:
        hh_shares.append((counts[0] + counts[1]) / total)
    hhi = np.sum([hh ** 2 for hh in hh_shares])
    return hhi

    # Проверяем, монотонен ли показатель на разбиении


# Check if array is monotonik
def _is_monotonik(vals):
    # вычисляем разности между woe
    difs = [j - i for i, j in zip(vals[:-1], vals[1:])]
    if len(difs) == 0:
        return True
    # перебираем все разности и смотрим, чтобы знакразностей не менялся. Изменился - возвращаем False
    flag = difs[0] < 0
    for dif in difs:
        if (dif < 0) != flag:
            return False
    return True

    # Подсчёт gini по формуле из документов SAS


# Calculate gini
def _calc_gini(gaps_counts):
    # сортируем по убыванию по количеству хороших в гэпе
    s_gaps_counts = sorted(gaps_counts, key=lambda x: x[0], reverse=True)
    # первое слагаемое числителя
    first_sl = 0
    for i in range(1, len(s_gaps_counts)):
        first_sl += s_gaps_counts[i][0] * np.sum([gc[1] for gc in s_gaps_counts[:i]])

    # Второе слагаемое числителя
    second_sl = 0
    for gap_counts in s_gaps_counts:
        second_sl += gap_counts[0] * gap_counts[1]

    # Знаменатель
    znam = np.sum([c[0] for c in s_gaps_counts]) * np.sum([c[1] for c in s_gaps_counts])
    # Джини
    g = abs(1 - (2 * first_sl + second_sl) / znam)
    return g


# бининг из файла
def read_file(filename='bins.prdb'):
    """
    Read binner from file
    :param filename: Path to file
    :return: Binner instance
    """
    with open(filename, 'rb') as inp:
        binner = pickle.load(inp)
        return binner

        # бининг в файл


class Binning:
    def __init__(self, gaps, woes, iv, shares, counts, gini, gaps_counts_shares, gaps_avg=[], hhi=-1, r2=0, name=None):
        self._name = name
        self._gaps = gaps
        self._woes = woes
        self._iv = iv
        self._shares = shares
        self._counts = counts
        self._gini = gini
        self._r2 = r2
        self._hhi = hhi
        self._gaps_counts_shares = gaps_counts_shares
        self._gaps_avg = gaps_avg


class BinningSettings:
    _monotone = True
    _variable_name = ''
    _min_leaf_ratio = 0.1

    def __init__(self, variable_name='', monotone=True, min_leaf_ratio=0.1):
        self._monotone = monotone
        self._variable_name = variable_name
        self._min_leaf_ratio = min_leaf_ratio

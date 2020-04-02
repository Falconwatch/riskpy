# ver. 20190321-3
import pandas as pd
import seaborn as sns
from riskpy.common.misc import calc_size_bin, recalc_woe_and_gap_for_new_bins, gap_translate_2_human


#from ..common.misc.py import calc_size_bin, recalc_woe_and_gap_for_new_bins



def trans2score(x):
    return round(-99 * x, 0)


def riskpy_make_scorecard(binner_,
                          model_,
                          train_data,
                          src_data_=None,
                          DFCN_='',
                          bin_union_dict_={},
                          precision=2,
                          none_text_descr='NaN',
                          trans2score_function_handle=trans2score,
                          inf_re=100000000000000,
                          column_names=[
                              'Фактор',
                              'Диапазон исходной переменной',
                              'Доля в выборке для разработки, %',
                              'Значение WoE-фактора',
                              'Оценка регрессии',
                              'Итоговое значение',
                              'PD',
                              'Score'
                          ]):
    '''
    Создать скоркарту исходя из данных о модели, подготовленных в riskpy.
    binner_ - riskpy.modeling.binning.Binner биннер переменных.
    model_ - statsmodels.genmod.generalized_linear_model.GLMResultsWrapper модель.
    train_data - pandas.core.frame.DataFrame таблица с обучающей выборкой (до woe-трансформации).
    src_data_ [используется при объединении корзин]- pandas.core.frame.DataFrame таблица с исходными данными (до woe-трансформации).
    DFCN_ [используется при объединении корзин] - DefaultFlagColName имя столбца (в таблице src_data_), содеражащего целевую.
    bin_union_dict_ - словарь для объединения корзин. Например:
        bin_union_dict_={
            'K6_woe':[[0],[1],[2,3]],
            'K24_woe':[[0,3],[1],[2]]
        }
    precision - точность: число знаков после запятой при работе с интервалами.
    none_text_descr - строка, описывающая состояние None в диапазонах.
    trans2score_function_handle - функция одной перменной, определяющая правило перевода итогового значения в скор.
    inf_re - большое число, соответсвующее бесконечности в riskpy.
    column_names - имена столбцов в скоркарте.
    Для сохранения таблицы в Excel испольузйте xlsx-формат.
    '''
    data_full_size = train_data.shape[0]

    model_table = pd.DataFrame(model_.summary().tables[1].data)
    model_table.drop(0, inplace=True)
    model_table.set_index(model_table.columns[0], drop=True, inplace=True)

    bins_container = binner_.get_fitted_bins()

    scorecard_col_name = column_names[0]
    scorecard_col_diap = column_names[1]
    scorecard_col_size = column_names[2]
    scorecard_col_woe = column_names[3]
    scorecard_col_koef = column_names[4]
    scorecard_col_summary = column_names[5]
    scorecard_col_pd = column_names[6]
    scorecard_col_score = column_names[7]

    scorecard = pd.DataFrame(columns=[
        scorecard_col_name,
        scorecard_col_diap,
        scorecard_col_size,
        scorecard_col_woe,
        scorecard_col_koef,
        scorecard_col_summary,
        scorecard_col_pd,
        scorecard_col_score
    ])

    for param_name in model_table.index:

        koef = float(model_table[model_table.columns[0]][param_name])
        if param_name != 'Intercept':
            if param_name in bin_union_dict_:
                have_rebin = True
                bin_union = bin_union_dict_[param_name]
            else:
                have_rebin = False

            for bins in bins_container:
                if bins._name + '_woe' == param_name:
                    woes = bins._woes
                    gaps = bins._gaps
                    if have_rebin == True:
                        (woes, gaps) = recalc_woe_and_gap_for_new_bins(DFCN=DFCN_,
                                                                       MAINDATA=src_data_,
                                                                       SelBinObj=bins,
                                                                       new_groups=bin_union)
                    gaps_text = gap_translate_2_human(gaps, have_rebin, precision, none_text_descr, inf_re)
                    break

            for (gap_descript, gap_el, woe) in zip(gaps_text, gaps, woes):
                part_size = calc_size_bin(train_data, param_name[:-4], have_rebin, gap_el, inf_re, precision)

                summary = woe * koef
                ProbDef = 1 / (1 + pd.np.exp(-summary))
                score = trans2score_function_handle(summary)

                one_row = pd.DataFrame()
                one_row.loc[0, scorecard_col_name] = param_name
                one_row.loc[0, scorecard_col_diap] = gap_descript
                one_row.loc[0, scorecard_col_size] = round(100 * part_size / data_full_size, 2)
                one_row.loc[0, scorecard_col_woe] = woe
                one_row.loc[0, scorecard_col_koef] = koef
                one_row.loc[0, scorecard_col_summary] = summary
                one_row.loc[0, scorecard_col_pd] = ProbDef
                one_row.loc[0, scorecard_col_score] = score
                scorecard = pd.concat([scorecard, one_row])

        else:
            summary = koef
            ProbDef = 1 / (1 + pd.np.exp(-summary))
            score = trans2score_function_handle(summary)

            one_row = pd.DataFrame()
            one_row.loc[0, scorecard_col_name] = param_name
            one_row.loc[0, scorecard_col_diap] = ''
            one_row.loc[0, scorecard_col_size] = None
            one_row.loc[0, scorecard_col_woe] = ''
            one_row.loc[0, scorecard_col_koef] = koef
            one_row.loc[0, scorecard_col_summary] = summary
            one_row.loc[0, scorecard_col_pd] = ProbDef
            one_row.loc[0, scorecard_col_score] = score
            scorecard = pd.concat([scorecard, one_row])

    scorecard[scorecard_col_score] = pd.to_numeric(scorecard[scorecard_col_score], downcast='integer')

    mi = pd.MultiIndex.from_arrays([scorecard[scorecard_col_name], scorecard[scorecard_col_diap]],
                                   names=(scorecard_col_name, scorecard_col_diap))

    scorecard.set_index(mi, inplace=True)
    scorecard.drop(scorecard_col_name, axis=1, inplace=True)
    scorecard.drop(scorecard_col_diap, axis=1, inplace=True)

    scorecard_style = scorecard.style

    color_map_score = sns.light_palette('red', as_cmap=True, reverse=True)
    color_map_size = sns.light_palette('green', as_cmap=True)

    for name in model_table.index:
        if name != 'Intercept':
            scorecard_style.background_gradient(cmap=color_map_score,
                                                subset=pd.IndexSlice[[name], [scorecard_col_score]])
            scorecard_style.background_gradient(cmap=color_map_size, subset=pd.IndexSlice[[name], [scorecard_col_size]])

    return scorecard_style

# Упрощённая версия функции riskpy_make_scorecard для sklearn
def make_scorecard(data, fitted_estimator, columns, binner, precision, none_text_descr, inf_re, style=True):
    sc = []
    for i, param_name in enumerate(columns):
        bin_param = [(x._woes, x._gaps)  for x in binner.get_fitted_bins() if x._name == param_name]
        if len(bin_param) == 0:
            raise Exception('Binner does not contain features {}'.format('param_name'))
        gaps_text = gap_translate_2_human(bin_param[0][1], False, precision, none_text_descr, inf_re)
        for (gap_descript, gap_el, woe) in zip(gaps_text, bin_param[0][1], bin_param[0][0]):
            part_size = calc_size_bin(data, param_name, False, gap_el, inf_re, precision)
            sc.append((
                param_name,
                gap_descript,
                round(100 * part_size / data.shape[0], precision),
                woe,
                fitted_estimator.coef_[0][i],
            ))
    sc.append(('Intercept', None, None, None,fitted_estimator.intercept_[0],))
    scorecard = pd.DataFrame(sc, columns=['Фактор', 'Диапазон исходной переменной', 'Доля в выборке для разработки, %', 'Значение WoE-фактора', 'Оценка регрессии',])
    scorecard['Итоговое значение'] = scorecard['Значение WoE-фактора'] * scorecard['Оценка регрессии']
    scorecard['Скоринговый балл'] = [round(-(99) * x, 0) for x in scorecard['Итоговое значение']]
    if style:
        mi = pd.MultiIndex.from_arrays([scorecard['Фактор'], scorecard['Диапазон исходной переменной']], names=('Фактор', 'Диапазон исходной переменной'))
        scorecard.set_index(mi, inplace=True)
        scorecard.drop(['Фактор', 'Диапазон исходной переменной'], axis=1, inplace=True)
        scorecard_style = scorecard.style
        color_map_score = sns.light_palette('lightcoral', as_cmap=True, reverse=True)
        color_map_size = sns.light_palette('lightgreen', as_cmap=True)
        for name in columns:
            scorecard_style.background_gradient(cmap=color_map_score, subset=pd.IndexSlice[[name], ['Скоринговый балл']])
            scorecard_style.background_gradient(cmap=color_map_size, subset=pd.IndexSlice[[name], ['Доля в выборке для разработки, %']])
        return scorecard_style
    return scorecard
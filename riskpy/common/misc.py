# ver. 20190321-3
import math as math
import pandas as pd


def recalc_woe_and_gap_for_new_bins(DFCN, MAINDATA, SelBinObj, new_groups):
    RebinFactorName = SelBinObj._name
    ClassCounts = MAINDATA[DFCN].value_counts()

    if 0 not in list(ClassCounts.index):
        goods = 0
    else:
        goods = ClassCounts[0]

    if 1 not in list(ClassCounts.index):
        bads = 0
    else:
        bads = ClassCounts[1]

    woes = list()
    gaps = list()
    for new_group in new_groups:
        gap_bads = 0
        gap_goods = 0
        gap = list()
        for borders in [SelBinObj._gaps[ng_el] for ng_el in new_group]:
            if borders[0] is None:
                rule_subdata = MAINDATA.loc[pd.np.isnan(MAINDATA[RebinFactorName])]
            else:
                rule_subdata = MAINDATA.loc[
                    (MAINDATA[RebinFactorName] > borders[0]) & (MAINDATA[RebinFactorName] <= borders[1])]

            gap.append(borders)

            ClassCounts = rule_subdata[DFCN].value_counts()

            if 0 in list(ClassCounts.index):
                gap_goods += ClassCounts[0]

            if 1 in list(ClassCounts.index):
                gap_bads += ClassCounts[1]

        bads_share = (gap_bads + 0.5) / (bads + 0.5) * 1.0
        goods_share = (gap_goods + 0.5) / (goods + 0.5) * 1.0
        val = (goods_share / bads_share)
        woes.append(math.log(val))
        gaps.append(gap)

    return (woes, gaps)


def get_diap_type(element, infinity):
    if element[0] is None:
        return 'NA'

    if element[0] == (-1) * infinity:
        return 'LI'
    elif element[1] == infinity:
        return 'RI'
    else:
        return 'LR'


def gap_translate_2_human(vec, multibin, prec_, none_name, infinity_):
    def one_gap_translate_2_human(element, prec, None_text, infinity):
        txt = ''
        diap_type = get_diap_type(element, infinity)
        if diap_type == 'NA':
            txt += None_text
        elif diap_type == 'LI':
            txt += '<=' + str(round(element[1], prec))
        elif diap_type == 'RI':
            txt += '>' + str(round(element[0], prec))
        elif diap_type == 'LR':
            txt += '(' + str(round(element[0], prec)) + ';' + str(round(element[1], prec)) + ']'
        else:
            txt += 'err'

        return txt

    txts = list()
    if not multibin:
        for i, p in enumerate(vec):
            txt = one_gap_translate_2_human(p, prec_, none_name, infinity_)
            txts.append(txt)
    else:
        for i, p in enumerate(vec):
            txt = ''
            for q in p:
                if len(txt) > 0:
                    txt += ' or '
                txt += one_gap_translate_2_human(q, prec_, none_name, infinity_)
            txts.append(txt)
    return txts


def calc_size_bin(src_data_, src_name_, multibin, vec, infinity, prec):
    if multibin == False:
        vec = [vec]

    subsets_size = 0
    for p in vec:
        diap_type = get_diap_type(p, infinity)
        if diap_type == 'NA':
            subset = src_data_.loc[pd.np.isnan(src_data_[src_name_])]
        elif diap_type == 'LI':
            subset = src_data_.loc[src_data_[src_name_] <= round(p[1], prec)]
        elif diap_type == 'RI':
            subset = src_data_.loc[src_data_[src_name_] > round(p[0], prec)]
        elif diap_type == 'LR':
            subset = src_data_.loc[
                (src_data_[src_name_] > round(p[0], prec)) & (src_data_[src_name_] <= round(p[1], prec))]

        subsets_size += subset.shape[0]

    return subsets_size


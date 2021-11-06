"""
一开始配置X列和Y列的变量类型。以及谁是Y 因变量。
分别对X和Y 进行考察
将X列中的缺失样本删除，将其余 分类型变量 取值频数低于5的 值进行合并，高于5的值的进行保留。
将数值型/顺序型 变量 切分成 分类型自变量。同时使每个区间含有5条以上的数据。
进行 列联表独立性卡方检验，观察显著性。不显著的因素，思考其样本数量是否足够。重要性排序最后，可以将其从问题研究范围中剔除。
对比各个自变量因素对Y的 相关系数。得出因素重要性排序。如果最高的相关系数也不是很大，可以认为可能存在其他对Y相关的因素没有纳入问题研究域中。
"""
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import StatsUtil


def numeric_to_category(x, describe):
    if x <= describe['25%']:
        return 'A'
    elif x <= describe['50%']:
        return 'B'
    elif x <= describe['75%']:
        return 'C'
    else:
        return 'D'


def stats(sample_df: pd.DataFrame, factor_df: pd.DataFrame):
    y = factor_df.loc['is_label',][factor_df.loc['is_label',] == 1].index[0]
    x_list = factor_df.loc['is_label',][factor_df.loc['is_label',] != 1].index.to_list()
    train_df = sample_df.copy()
    chi_test_df = {}
    for x in x_list:
        train_df[x].dropna()
        if factor_df[x]['type'] == 'category':
            value_counts = train_df[x].value_counts()
            if len(value_counts[value_counts >= 5]) == 0:
                continue
            merge_category_list = value_counts[value_counts < 5].index.to_list()
            if len(merge_category_list) > 0:
                train_df[x][list(map(lambda x: x in merge_category_list, train_df[x]))] = 'merge'
        else:
            if len(train_df[x]) < 30:
                continue
            describe = train_df[x].describe()
            train_df[x] = train_df[x].apply(lambda x: numeric_to_category(x, describe))
        df = train_df.loc[:, [x, y]].dropna()
        x_index = df[x].value_counts().index.to_list()
        y_index = df[y].value_counts().index.to_list()
        d = pd.DataFrame(np.zeros((len(y_index), len(x_index))), index=y_index, columns=x_index)
        for row in zip(df[x], df[y]):
            d[row[0]][row[1]] += 1
        if len(x_index) == 1:
            print("数据列异常，只有一种取值", x)
        chi2_test = pd.Series(chi2_contingency(d.to_numpy()), index=['chi2', 'p', 'degree_of_free', 'exp'])
        chi2_test['fai'], chi2_test['c'], chi2_test['V'] = StatsUtil.cc_coefficient(
            chi2_test['chi2'], len(df), len(y_index), len(x_index))
        chi_test_df[x] = chi2_test
    chi_test_df = pd.DataFrame.from_dict(chi_test_df).T
    print(chi_test_df)
    print("统计不显著的因子:", chi_test_df[chi_test_df['p'] > 0.05])
    factor_importance = chi_test_df['V'].sort_values(ascending=False)
    print("自变量与因变量的相关性程度排序：\n", factor_importance)
    print("最高的相关系数为：", factor_importance[0])
    print("其高于0.5,此问题有一定程度上的线性规律" if factor_importance[0] > 0.5 else "其低于0.5，可能存在其他对Y相关的因素没有纳入问题研究域中")

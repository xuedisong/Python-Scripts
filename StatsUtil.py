# 多值分类型-多值分类型 列联表独立性卡方检验 相关系数

def cc_coefficient(chi2, n, r, c):
    """
    相关系数

    @param chi2: 卡方统计值
    @param n: 样本数量
    @param r: 行数
    @param c: 列数
    @return: (fai系数，c系数，V系数)
    """
    try:
        fai_correlation_coefficient = (chi2 / n) ** 0.5
        coefficient_of_contingency = (chi2 / (chi2 + n)) ** 0.5
        Gramer_V = (chi2 / (n * min(r - 1, c - 1))) ** 0.5
        return (fai_correlation_coefficient, coefficient_of_contingency, Gramer_V)
    except Exception as e:
        print(e)
        print(chi2,n,r,c)
    raise

#def cn_coefficient
#def nn_coefficient(chi2, n, r, c):
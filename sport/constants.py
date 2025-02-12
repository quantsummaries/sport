class Constants:
    """Class to store several constants."""

    # security attributes that can be used to group securities.
    grouping_attr = {'IND_CD'}

    # security attributes that can be used for nonlinear constraints.
    nonlinear_constr = {'RISK', 'AVG_MAX_DRAWDOWN'}

    # English-to-Chinese dictionary
    en_to_cn = {'': '',
                'MIN': '最小值',
                'MAX': '最大值',
                'EACH': '每个',
                'ALL': '全部',
                'ATTRIBUTE': '限制属性',
                'ATTRIBUTE_PARAMS': '限制属性参数',
                'MEAN_VARIANCE': '均值方差最优',
                'MAX_SHARPE_RATIO': '夏普信息比最大',
                'MAX_RETURN': '收益最大化',
                'MIN_RISK': '波动风险最低',
                'RISK_PARITY': '风险平价',
                'MIN_AVG_MAX_DRAWDOWN': '最大回撤均值最小',
                'SEC_ID': '证券代码',
                'WEIGHT': '证券权重',
                'SEC_NM': '证券名称',
                'RETURN': '年化收益率',
                'RISK': '年化波动率',
                'SHARPE_RATIO': '夏普信息比',
                'CTR': '风险贡献',
                'ANALYTICS': '分析量',
                'VALUE': '取值',
                'RISK_TOL': '风险容忍度',
                'BENCHMARK_RETURN': '基准收益率',
                'IND_NM': '行业名称',
                'IND_CD': '行业权重',
                '20D_AVG_MAX_DRAWDOWN': '20天最大回撤均值',
                'AVG_MAX_DRAWDOWN': '最大回撤均值',
                'PORTFOLIO': '投资组合',
                'OBJECTIVE': '优化目标',
                'CONSTRAINT': '限制条件',
                'COVAR_MATRIX': '客户的协方差矩阵',
                'EXPECTED_RETURN': '预期年化收益率',
                'CHOOSE_OBJECTIVE': '选择该优化目标',
                'PARAMETER': '参数'
                }

    # Chinese-to-English dictionary
    cn_to_en = dict()
    for k, v in en_to_cn.items():
        cn_to_en[v] = k

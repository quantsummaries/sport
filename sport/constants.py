class Constants:
    """Class to store several constants."""

    # security attributes that can be used to group securities.
    grouping_attr = {'IND_CD'}

    # security attributes that can be used for nonlinear constraints.
    nonlinear_constr = {'RISK', 'AVG_MAX_DRAWDOWN'}

    # Chinese-to-English translation
    cn_to_en = {'': '',
                '均值方差最优': 'MEAN_VARIANCE',
                '夏普信息比最大': 'MAX_SHARPE_RATIO',
                '收益最大化': 'MAX_RETURN',
                '波动风险最低': 'MIN_RISK',
                '风险平价': 'RISK_PARITY',
                '最大回撤均值最小': 'MIN_AVG_MAX_DRAWDOWN',
                '投资标的': 'SEC_ID',
                '限制属性': 'ATTRIBUTE',
                '限制属性参数': 'ATTRIBUTE_PARAMS',
                '最小值': 'MIN',
                '最大值': 'MAX',
                '每个': 'EACH',
                '全部': 'ALL',
                '证券权重': 'WEIGHT',
                '行业权重': 'IND_CD',
                '年化波动率': 'RISK',
                '最大回撤均值': 'AVG_MAX_DRAWDOWN'
                }

    # English-to-Chinese translation
    en_to_cn = {'': '',
                'MEAN_VARIANCE': '均值方差最优',
                'MAX_SHARPE_RATIO': '夏普信息比最大',
                'MAX_RETURN': '收益最大化',
                'MIN_RISK': '波动风险最低',
                'RISK_PARITY': '风险平价',
                'SEC_ID': '证券代码',
                'WEIGHT': '证券权重',
                'SEC_NM': '证券名称',
                'RETURN': '年化收益率',
                'RISK': '年化波动率',
                'SHARPE_RATIO': '夏普信息比',
                'CTR': '风险贡献',
                'OBJECTIVE': '优化目标',
                'ANALYTICS': '分析量',
                'VALUE': '取值',
                'RISK_TOL': '风险容忍度',
                'BENCHMARK_RETURN': '基准收益率',
                'IND_NM': '行业名称',
                '20D_AVG_MAX_DRAWDOWN': '20天最大回撤均值',
                'AVG_MAX_DRAWDOWN': '最大回撤均值',
                'MIN_AVG_MAX_DRAWDOWN': '最大回撤均值最小'
                }

# SPORT

SPORT (Scalable Portfolio Optimization Research Tool) provides a scalable architecture to calculate the optimal asset allocation for multi-asset-class portfolios. 

It incorporates several optimization packages (scipy.optimize, cvxopt). It provides a flexible syntax to formulate constraints and can handle a variety of target functions, including mean-variance, Sharpe ratio, volatility, risk parity, and maximum drawdown. 

Run batch_port_main.py will obtain the following output:

----- INPUT -----

             TYPE         SEC_ID         ATTRIBUTE ATTRIBUTE_PARAMS        VALUE
0     LINEAR_INEQ           EACH            WEIGHT              NaN    [0.0:1.0]
1     LINEAR_INEQ  002777+600276            WEIGHT              NaN    [0.0:0.5]
2     LINEAR_INEQ           EACH            IND_CD              NaN    [0.0:0.5]
3  NONLINEAR_INEQ            ALL              RISK              NaN   [0.0:0.35]
4  NONLINEAR_INEQ            ALL  AVG_MAX_DRAWDOWN             20.0   [0.0:0.09]
5     LINEAR_INEQ         600276            WEIGHT              NaN  [0.01:0.35]
6     LINEAR_INEQ         801150            IND_CD              NaN  [0.02:0.45]
7       LINEAR_EQ            ALL            WEIGHT             None            1

----- OUTPUT -----


MEAN_VARIANCE
     证券代码  证券名称  行业名称   证券权重   年化收益率   年化波动率
0  000000    现金    现金  0.500  0.0025  0.0000
1  000001  平安银行    银行  0.138  0.1337  0.3252
2  002777  久远银海   计算机  0.012  0.0749  0.4992
3  600276  恒瑞医药  医药生物  0.350  0.4189  0.3363

MAX_SHARPE_RATIO
     证券代码  证券名称  行业名称  证券权重   年化收益率   年化波动率   夏普信息比
0  000001  平安银行    银行  0.50  0.1337  0.3252  0.4034
1  002777  久远银海   计算机  0.15  0.0749  0.4992  0.1450
2  600276  恒瑞医药  医药生物  0.35  0.4189  0.3363  1.2381

MAX_RETURN
     证券代码  证券名称  行业名称  证券权重   年化收益率   年化波动率
0  000000    现金    现金  0.00  0.0025  0.0000
1  000001  平安银行    银行  0.50  0.1337  0.3252
2  002777  久远银海   计算机  0.15  0.0749  0.4992
3  600276  恒瑞医药  医药生物  0.35  0.4189  0.3363

MIN_RISK
     证券代码  证券名称  行业名称    证券权重   年化收益率   年化波动率
0  000000    现金    现金  0.5000  0.0025  0.0000
1  000001  平安银行    银行  0.2288  0.1337  0.3252
2  002777  久远银海   计算机  0.0873  0.0749  0.4992
3  600276  恒瑞医药  医药生物  0.1839  0.4189  0.3363

RISK_PARITY
     证券代码  证券名称  行业名称    证券权重   年化收益率   年化波动率    风险贡献
0  000001  平安银行    银行  0.5000  0.1337  0.3252  0.1314
1  002777  久远银海   计算机  0.2235  0.0749  0.4992  0.0654
2  600276  恒瑞医药  医药生物  0.2765  0.4189  0.3363  0.0637

MIN_AVG_MAX_DRAWDOWN
     证券代码  证券名称  行业名称    证券权重   年化收益率   年化波动率  最大回撤均值
0  000000    现金    现金  0.5000  0.0025  0.0000  0.0000
1  000001  平安银行    银行  0.2110  0.1337  0.3252  0.1137
2  002777  久远银海   计算机  0.0782  0.0749  0.4992  0.1737
3  600276  恒瑞医药  医药生物  0.2107  0.4189  0.3363  0.1064

Summary
       优化目标        分析量      取值
0    均值方差最优      年化收益率  0.1672
1    均值方差最优      年化波动率  0.1421
2    均值方差最优      风险容忍度     3.0
3    均值方差最优  20天最大回撤均值  0.0452
4                             
5   夏普信息比最大      年化收益率  0.2247
6   夏普信息比最大      年化波动率  0.2589
7   夏普信息比最大      夏普信息比  0.8583
8   夏普信息比最大      基准收益率  0.0025
9   夏普信息比最大  20天最大回撤均值  0.0848
10                            
11    收益最大化      年化收益率  0.2247
12    收益最大化      年化波动率  0.2589
13    收益最大化  20天最大回撤均值  0.0848
14                            
15   波动风险最低      年化收益率  0.1154
16   波动风险最低      年化波动率  0.1291
17   波动风险最低  20天最大回撤均值  0.0422
18                            
19     风险平价      年化收益率  0.1994
20     风险平价      年化波动率  0.2605
21     风险平价  20天最大回撤均值  0.0869
22                            
23   最大回撤均值      年化收益率  0.1236
24   最大回撤均值      年化波动率  0.1295
25   最大回撤均值     最大回撤均值  0.0421
26               

TODO: implement more sophisticated methods of covariance estimation, including linear shrinkage, eigenvalue clipping, eigenvalue substitution, and rotationally invariant optimal shrinkage (based on Random Matrix Theory).

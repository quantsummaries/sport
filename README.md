# SPORT

SPORT (Scalable Portfolio Optimization Research Tool) provides a scalable architecture to calculate the optimal asset allocation for multi-asset-class portfolios. 

It incorporates several optimization packages (scipy.optimize, cvxopt). It provides a flexible syntax to formulate constraints and can handle a variety of target functions, including mean-variance, Sharpe ratio, volatility, risk parity, and maximum drawdown. 

## How to Run
1. Run batch_download_data.py to download daily price data to the data folder.
2. In the data folder, run the shell script clean_yfinance_data.sh to clean up data.
3. Run btach_port_covar.py to load daily price data obtained in Step 1, and then generate covariance and correlation matrices.
4. Manually update the SEC_NM column of data/etf_covar_matrix.csv to enrich output information (if not updated, code will run without issue).
5. Manually update template_input.xlsx to generate inputs.
6. Run batch_port_main.py to generate optimal weights for various objectives and constraints.

## TODO

implement more sophisticated methods of covariance estimation, including linear shrinkage, eigenvalue clipping, eigenvalue substitution, and rotationally invariant optimal shrinkage (based on Random Matrix Theory).

## Class Documentation

### security.py

An abstraction of single security, single index etc. that carries an ID and a list of attributes.

### portfolio.py

A portfolio class that carries a list of Security objects and their covariance and weights.

### function.py

1. constr_avg_max_drawdown(x: List[float], params_constr: Dict[str: object]) -> float: Calculate average maximum drawdown of a portfolio.
2. constr_risk(x: List[float], params_constr: Dict[str: object]) -> float: Calculate risk of a portfolio.
3. obj_avg_max_drawdown(x: List[float], params_obj: Dict[str, object]) -> float: Average maximum drawdown.
4. obj_neg_rtrn(x: List[float], params_obj: Dict[str, object]) -> float: Negative of portfolio return.
5. obj_neg_sharpe_ratio(x: List[float], params_obj: Dict[str, object]) -> float: Negative of the Sharpe ratio.
6. obj_qp(x: List[float], params_obj: Dict[str, object]) -> float: Objective function for quadratic programming (minimization).
7. obj_risk(x: List[float], params_obj: Dict[str, object]) -> float: Objective function for risk.
8. obj_risk_parity(x: List[float], params_obj: Dict[str, object]) -> float: Objective function for risk parity optimization https://en.wikipedia.org/wiki/Risk_parity.
9. util_covar_to_corr_matrix(covar_matrix: pd.DataFrame) -> pd.DataFrame: convert a covar matrix to a correlation matrix.
10. util_is_valid_covar(covar_matrix: pd.DataFrame) -> tuple: validate if a matrix is a valid covar matrix (symmetric and positive definite).
11. util_md_Qn(x: float) -> float: Qn function used in maximum drawdown.
12. util_md_Qp(x: float) -> float: Qp function used in maximum drawdown.

### constraint.py

A factory class that generates constrains for portfolio optimization.

### objective.py

A factory class that generates objective functions for optimizers.

### optimizer.py

Optimization calculator.

### covar_estimator.py

Estimator of covariance matrix and average returns of assets.

### dao.py

Data access object class to get security attributes.
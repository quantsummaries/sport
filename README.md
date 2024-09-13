# SPORT

SPORT (Scalable Portfolio Optimization Research Tool) provides a scalable architecture to calculate the optimal asset allocation for multi-asset-class portfolios. 

It incorporates several optimization packages (scipy.optimize, cvxopt). It provides a flexible syntax to formulate constraints and can handle a variety of target functions, including mean-variance, Sharpe ratio, volatility, risk parity, and maximum drawdown. 

Run batch_port_main.py will get a run on sample data, with printouts stored in sample_input_output.txt.

TODO: implement more sophisticated methods of covariance estimation, including linear shrinkage, eigenvalue clipping, eigenvalue substitution, and rotationally invariant optimal shrinkage (based on Random Matrix Theory).
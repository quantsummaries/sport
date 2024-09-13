import numpy as np
import pandas as pd


def constr_avg_max_drawdown(x, params_constr):
    """Calculate average maximum drawdown of a portfolio.

    x (list): a list of unknown variables (typically a weight vector).
        params_constr (dict): {'RETURN': return vector (numpy.array), 'COVAR': covariance matrix (numpy.ndarray), 'T': number of days, 'MASK': a list of 0's and 1's}.

    Returns:
        emd (double): expected maximum drawdown of a portfolio.
    """

    if params_constr is None:
        raise Exception('Input objective parameters are None')
    if set(params_constr.keys()) != {'RETURN', 'COVAR', 'T', 'MASK'}:
        raise Exception('Parameters must be a dictionary with the keys "RETURN", "COVAR", "T" and "MASK".')

    mask = params_constr.get('MASK')

    if len(mask) != len(x):
        raise Exception('MASK parameters and x have different lengths')

    wts = np.array(x) * np.array(mask)
    rtrn = params_constr.get('RETURN')
    covar = params_constr.get('COVAR')
    T = params_constr.get('T')/250.0

    if len(x) != len(rtrn):
        raise ValueError('Input weights and returns do not have the same length in maximum drawdown objective function')

    mu = sum([a * b for a, b in zip(wts, rtrn)])
    sigma_sqrd = np.matmul(np.matmul(np.array(wts), covar), np.array(wts))

    if np.isclose(0.0, sigma_sqrd):
        if mu >= 0:
            return 0
        else:
            return -mu * T

    alpha_sqrd = mu**2 * T * 0.5 / sigma_sqrd
    gamma = np.sqrt(np.pi/8)

    if np.isclose(mu, 0):
        return 2 * gamma * np.sqrt(T * sigma_sqrd)
    elif mu > 0:
        return 2 * sigma_sqrd / mu * util_md_Qp(alpha_sqrd)
    elif mu < 0:
        return -2 * sigma_sqrd / mu * util_md_Qn(alpha_sqrd)

    raise RuntimeError('Calculation is not caught by logic in obj_avg_max_drawdown')


def constr_risk(x, params_constr):
    """Calculate risk of a portfolio.

    Args:
        x (list): a list of unknown variables (typically a weight vector).
        params_constr (dict): {'COVAR': covariance matrix (numpy.ndarray), 'MASK': a list of 0's and 1's}.

    Returns:
        risk (double): risk of a portfolio.
    """
    if params_constr is None:
        raise Exception('Input constraint parameters are None')
    if set(params_constr.keys()) != {'COVAR', 'MASK'}:
        raise Exception('Parameters must be a dictionary with keys "COVAR" and "MASK"')

    covar = params_constr.get('COVAR')
    mask = params_constr.get('MASK')

    if len(mask) != len(x):
        raise Exception('MASK parameters and x have different lengths')

    wts = np.array(x) * np.array(mask)
    wts_transpose = wts.transpose()
    risk = np.sqrt(np.matmul(np.matmul(wts, covar), wts_transpose))

    return risk


def obj_avg_max_drawdown(x, params_obj):
    """Average maximum drawdown.

    Args:
        x (list): a list of unknown variables (typically a weight vector).
        params_obj (dict): {'RETURN': return vector (numpy.array), 'COVAR': covariance matrix (numpy.ndarray), 'T': number of days}.

    Returns:
        emd (double): expected maximum drawdown.
    """
    if params_obj is None:
        raise Exception('Input objective parameters are None')
    if set(params_obj.keys()) != {'RETURN', 'COVAR', 'T'}:
        raise Exception('Parameters must be a dictionary with the keys "RETURN", "COVAR", and "T".')

    rtrn = params_obj.get('RETURN')
    covar = params_obj.get('COVAR')
    T = params_obj.get('T')/250.0

    if len(x) != len(rtrn):
        raise ValueError('Input weights and returns do not have the same length in maximum drawdown objective function')

    mu = sum([a * b for a, b in zip(x, rtrn)])
    sigma_sqrd = np.matmul(np.matmul(np.array(x), covar), np.array(x))

    if np.isclose(0.0, sigma_sqrd):
        if mu >= 0:
            return 0
        else:
            return -mu * T

    alpha_sqrd = mu**2 * T * 0.5 / sigma_sqrd
    gamma = np.sqrt(np.pi/8)

    if np.isclose(mu, 0):
        return 2 * gamma * np.sqrt(T * sigma_sqrd)
    elif mu > 0:
        return 2 * sigma_sqrd / mu * util_md_Qp(alpha_sqrd)
    elif mu < 0:
        return -2 * sigma_sqrd / mu * util_md_Qn(alpha_sqrd)

    raise RuntimeError('Calculation is not caught by logic in obj_avg_max_drawdown')


def obj_neg_rtrn(x, params_obj):
    """Negative of portfolio return.

    Args:
        x (list): a list of unknown variables.
        params_obj (dict): {'RETURN': negative of return vector (numpy.array), 'NUM_VAR': int}.

    Returns:
        value (double): negative of portfolio return.
    """

    if params_obj is None:
        raise Exception('Input objective parameters are None')
    if set(params_obj.keys()) != {'RETURN', 'NUM_VAR'}:
        raise Exception('Parameters must be a dictionary with the keyset {"RETURN", "NUM_VAR"}')

    rtrn = params_obj.get('RETURN')

    if len(x) != len(rtrn):
        raise ValueError('Input weights and returns do not have the same length in LP objective function')

    return sum([-a * b for a, b in zip(x, rtrn)])


def obj_neg_sharpe_ratio(x, params_obj):
    """Negative of the Sharpe ratio.

    Args:
        x (list): a list of unknown variables.
        params_obj (dict): {'Q': covar matrix (numpy.ndarray), 'p': returns (numpy.ndarray), 'bmk': benchmark return (float)}.

    Returns:
        value (double): value of the function.
    """
    if params_obj is None:
        raise ValueError('Input params is None')
    if set(params_obj.keys()) != {'Q', 'p', 'bmk'}:
        raise ValueError('Input params keys do not match with {"Q", "p", "bmk"}')

    Q = params_obj.get('Q')
    p = params_obj.get('p')
    bmk = params_obj.get('bmk')

    v = np.array(x)
    v_trans = v.copy().transpose()
    denom = np.matmul(np.matmul(v, Q), v_trans)
    numerator = np.dot(v, p) - bmk
    value = -numerator/np.sqrt(denom)

    return value


def obj_qp(x, params_obj):
    """Objective function for quadratic programming (minimization).

    Args:
        x (list): a list of unknown variables.
        params_obj (dict): {'Q': covariance matrix (cvxopt.matrix), 'p': negative of returns (cvxopt.matrix)}.

    Returns:
        value (double): value of the function.
    """
    if params_obj is None:
        raise ValueError('Input params is None')
    if set(params_obj.keys()) != {'Q', 'p'}:
        raise ValueError('Input params keys do not match with {"Q", "p"}')

    Q = params_obj.get('Q')
    p = params_obj.get('p')

    v = np.array(x)
    v_trans = v.copy().transpose()
    value = np.matmul(np.matmul(v, Q), v_trans) + np.dot(v, p)

    return value


def obj_risk(x, params_obj):
    """Objective function for risk.

    Args:
        x (list): a list of unknown variables (typically a weight vector).
        params_obj (dict): {'COVAR': covariance matrix (numpy.ndarray)}.

    Returns:
        risk (double): risk of a portfolio.
    """
    if params_obj is None:
        raise Exception('Input objective parameters are None')
    if set(params_obj.keys()) != {'COVAR'}:
        raise Exception('Parameters must be a dictionary with the key "COVAR"')

    covar = params_obj.get('COVAR')

    wts = np.array(x)
    wts_transpose = wts.transpose()
    risk = np.sqrt(np.matmul(np.matmul(wts, covar), wts_transpose))

    return risk


def obj_risk_parity(x, params_obj):
    """Objective function for risk parity optimization: https://en.wikipedia.org/wiki/Risk_parity.

        Args:
            x (list): a list of unknown variables (typically a weight vector).
            params_obj (dict): {'COVAR': covariance matrix (numpy.ndarray)}.

        Returns:
            risk_parity (double): risk parity value of a portfolio.
        """
    if params_obj is None:
        raise Exception('Input objective parameters are None')
    if set(params_obj.keys()) != {'COVAR'}:
        raise Exception('Parameters must be a dictionary with the key "COVAR"')

    covar = params_obj.get('COVAR')

    wts = np.array(x)
    wts_transpose = wts.transpose()
    risk_sqrd = np.matmul(np.matmul(wts, covar), wts_transpose)

    v = np.matmul(covar, wts_transpose) * len(x)

    risk_parity = sum([(w - risk_sqrd/x)**2 for w, x in zip(wts, v)])

    return risk_parity


def util_covar_to_corr_matrix(covar_matrix):
    """convert a covar matrix to a correlation matrix.

    Args:
        covar_matrix (pandas.DataFrame): convariance matrix in data frame.

    Returns:
        corr_matrix (pandas.DataFrame): correlation matrix in data frame.
    """
    valid, msg = util_is_valid_covar(covar_matrix)
    if not valid:
        raise ValueError('Input covar matrix is not valid: ' + msg)

    corr_matrix = pd.DataFrame(index=covar_matrix.index, columns=covar_matrix.columns)
    for idx_row in corr_matrix.index:
        for idx_col in corr_matrix.columns:
            denominator = np.sqrt(covar_matrix.loc[idx_row, idx_row])
            denominator = denominator * np.sqrt(covar_matrix.loc[idx_col, idx_col])
            if np.isclose(denominator, 0.0, atol=1e-16):
                corr_matrix.loc[idx_row, idx_col] = None
            else:
                corr_matrix.loc[idx_row, idx_col] = covar_matrix.loc[idx_row, idx_col] / denominator

    return corr_matrix


def util_is_valid_covar(covar_matrix):
    """validate if a matrix is a valid covar matrix (symmetric and positive definite).

    Args:
        covar_matrix (pandas.DataFrame): covar matrix in data frame.

    Returns:
        valid (bool): True if valid and False otherwise.
        err_msg (str): Error message.
    """
    valid = True

    # columns and index of covar matrix must match
    if not covar_matrix.columns.equals(covar_matrix.index):
        return False, 'columns and index do not match'

    # must be symmetric and positive definite
    matrix_values = covar_matrix.to_numpy(dtype=np.float64)

    if not np.allclose(matrix_values, matrix_values.transpose()):
        err_msg = 'not symmetric: \n'
        for idx1 in covar_matrix.index:
            for idx2 in covar_matrix.index:
                if not np.isclose(covar_matrix.loc[idx1, idx2], covar_matrix.loc[idx2, idx1]):
                    err_msg += 'Covar(' + str(idx1) + ',' + str(idx2) + ') = ' + str(covar_matrix.loc[idx1, idx2])
                    err_msg += ', Covar(' + str(idx2) + ',' + str(idx1) + ') = ' + str(covar_matrix.loc[idx2, idx1])
                    err_msg += '\n'

        return False, err_msg

    if not np.all(np.linalg.eigvals(matrix_values) >= 0):
        return False, 'eigenvalues are not all non-negative'

    return valid, None


def util_md_Qn(x):
    """Qn function used in maximum drawdown."""
    if x < 0:
        raise ValueError('Input argument x of util_md_Qn must be non-negative')

    x_array = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015,
               0.0175, 0.02, 0.0225, 0.025, 0.0275, 0.03, 0.0325, 0.035, 0.0375, 0.04,
               0.0425, 0.045, 0.0475, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08,
               0.085, 0.09, 0.095, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
               0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

    y_array = [0.019965, 0.028394, 0.034874, 0.04369, 0.045256,
               0.064633, 0.079746, 0.092708, 0.104259, 0.114814,
               0.124608, 0.133772, 0.142429, 0.15739, 0.158565,
               0.166229, 0.173756, 0.18793, 0.187739, 0.194489,
               0.20194, 0.207572, 0.213877, 0.2256, 0.231797,
               0.243374, 0.254585, 0.265472, 0.27670, 0.286406,
               0.296507, 0.306393, 0.31666, 0.325586, 0.413136,
               0.491599, 0.564333, 0.6337, 0.698849, 0.762455,
               0.884593, 1.445520, 1.97740, 2.483960, 2.99940,
               3.492520, 3.995190, 4.492380, 4.990430, 5.498820]

    assert len(x_array) == len(y_array)

    if x < x_array[0]:
        return np.sqrt(np.pi/8.0) * np.sqrt(2*x)
    elif x >= x_array[-1]:
        return x + 0.5

    for idx in range(len(x_array)):
        if (x_array[idx] <= x) and (x < x_array[idx+1]):
            return (y_array[idx+1] - y_array[idx]) / (x_array[idx+1] - x_array[idx]) * (x - x_array[idx]) + y_array[idx]

    raise RuntimeError('Calculation is not caught by logic')


def util_md_Qp(x):
    """Qp function used in maximum drawdown."""
    if x < 0:
        raise ValueError('Input argument x of util_md_Qp must be non-negative')

    x_array = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015,
               0.0175, 0.02, 0.0225, 0.025, 0.0275, 0.03, 0.0325, 0.035, 0.0375, 0.04,
               0.0425, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3,
               0.4, 0.5, 1.5, 2.5, 3.5, 4.5, 10, 20, 30, 40,
               50, 150, 250, 350, 450, 1000, 2000, 3000, 4000, 5000]

    y_array = [0.019690, 0.027694, 0.033789, 0.038896, 0.043372,
               0.06721, 0.073808, 0.084693, 0.094171, 0.102651,
               0.11375, 0.117503, 0.124142, 0.13374, 0.136259,
               0.141842, 0.147162, 0.152249, 0.157127, 0.161817,
               0.166337, 0.17702, 0.17915, 0.194248, 0.207999,
               0.22581, 0.232212, 0.24350, 0.32571, 0.38216,
               0.426452, 0.463159, 0.668992, 0.775976, 0.849298,
               0.905305, 1.088998, 1.253794, 1.351794, 1.421860,
               1.476457, 1.747485, 1.874323, 1.95837, 2.02630,
               2.219765, 2.392826, 2.494109, 2.565985, 2.621743]

    assert len(x_array) == len(y_array)

    if x < x_array[0]:
        return np.sqrt(np.pi/8.0) * np.sqrt(2*x)
    elif x >= x_array[-1]:
        return 0.25 * np.log(x) + 0.49088

    for idx in range(len(x_array)):
        if (x_array[idx] <= x) and (x < x_array[idx+1]):
            return (y_array[idx+1] - y_array[idx]) / (x_array[idx+1] - x_array[idx]) * (x - x_array[idx]) + y_array[idx]

    raise RuntimeError('Calculation is not caught by logic')



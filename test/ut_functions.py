import os
import traceback

import matplotlib.pyplot as plt
import numpy as np

from sport.functions import constr_risk
from sport.functions import obj_neg_rtrn, obj_neg_sharpe_ratio, obj_qp, obj_risk, obj_risk_parity
from sport.functions import util_md_Qn, util_md_Qp


def test():
    dir_path = os.path.abspath(os.path.dirname(__file__))

    # port obj_neg_rtrn, obj_neg_sharpe_ratio, obj_qp, obj_risk, obj_risk_parity
    Q = np.array([[1.0, 0.5], [0.5, 1.0]])
    p = np.array([1.0, 1.0])
    x = np.array([2.0, 2.0])

    assert np.isclose(-4.0, obj_neg_rtrn(x, {'RETURN': p, 'NUM_VAR': 2}), atol=1e-5)
    assert np.isclose(-3 / np.sqrt(12), obj_neg_sharpe_ratio(x, {'Q': Q, 'p': p, 'bmk': 1}), atol=1e-5)
    assert np.isclose(16, obj_qp(x, {'Q': Q, 'p': p}))
    assert np.isclose(np.sqrt(12.0), obj_risk(x, {'COVAR': Q}), atol=1e-5)
    assert np.isclose(0.9225, obj_risk_parity([1, 2], {'COVAR': Q}), atol=1e-5)

    # plot sharpe ratio function
    fig = plt.gcf()
    fig.set_size_inches((16, 12), forward=False)
    ax = plt.axes(projection="3d")

    x = np.linspace(-1, 1, 40)
    y = np.linspace(-1, 1, 40)
    X, Y = np.meshgrid(x, y)
    Z = np.ndarray(shape=(len(x), len(y)), dtype=np.double)
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i, j] = -obj_neg_sharpe_ratio([X[i, j], Y[i, j]], {'Q': Q, 'p': p, 'bmk': 1})

    ax.plot_wireframe(X, Y, Z, color='green')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='winter', edgecolor='none')
    ax.set_title('3D Sharpe Ratio Surface')

    output_dir = os.path.join(dir_path, 'test_output')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    fig.savefig(os.path.join(output_dir, '3D_Sharpe_Ratio_Surf.png'), dpi=600)
    plt.close(fig)

    # port constr_risk(...)
    x = [0.5, 0.1, 0.4]
    mask = [1.0, 0.0, 1.0]
    v = np.array(x) * np.array(mask)
    covar = np.array([[0.7, 0.5, 0.3], [0.5, 0.9, 0.4], [0.3, 0.4, 0.1]])
    result1 = np.sqrt(np.matmul(np.matmul(v, covar), v.transpose()))
    result2 = constr_risk(x, {'COVAR': covar, 'MASK': mask})
    assert np.isclose(result1, result2)

    # port lambda function
    f = lambda x: constr_risk(x, params_constr={'COVAR': covar, 'MASK': mask})
    y = f(x)
    assert np.isclose(result1, y)

    # port maximum drawdown functions
    assert np.isclose(0.0, util_md_Qn(0))
    assert np.isclose(np.sqrt(2 * 0.0001) * np.sqrt(np.pi / 8), util_md_Qn(0.0001))
    assert np.isclose(0.019965, util_md_Qn(0.0005))
    assert np.isclose((4.990430 + 5.498820) * 0.5, util_md_Qn(4.75))
    assert np.isclose(4.990430, util_md_Qn(4.5))
    assert np.isclose(5.5, util_md_Qn(5))
    assert np.isclose(6.5, util_md_Qn(6))

    assert np.isclose(0.0, util_md_Qp(0))
    assert np.isclose(np.sqrt(2*0.0001)*np.sqrt(np.pi/8), util_md_Qp(0.0001))
    assert np.isclose(0.019690, util_md_Qp(0.0005))
    assert np.isclose((2.494109+2.565985)*0.5, util_md_Qp(3500))
    assert np.isclose(2.565985, util_md_Qp(4000))
    assert np.isclose(0.25*np.log(5000)+0.49088, util_md_Qp(5000))
    assert np.isclose(0.25 * np.log(6000) + 0.49088, util_md_Qp(6000))

if __name__ == '__main__':
    try:
        test()
    except Exception as err:
        print('Functions unit test failed: ' + str(err))
        print(traceback.format_exc())

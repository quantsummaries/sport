import math
import traceback

from sport import Security


def test():
    """Unit tests."""

    print('Testing Security class...')

    # __init__(...)
    sec = Security('1234', {'duration': 7.0, 'oas': 200.0, 'dts': None})

    # reproduce(), equals(...)
    assert sec.equals(sec)
    sec1 = sec.reproduce(new_id='ABCD')
    assert sec.equals(sec1)
    assert sec1.get_id() == 'ABCD'

    # get_attr_names()
    assert {'DURATION', 'OAS'} == sec.get_attr_names()

    # get_attr_values(...)
    assert math.isclose(sec.get_attr_value('duration'), 7.0)
    assert sec.get_attr_value('abc') is None

    # get_id()
    assert sec.get_id() == '1234'

    # reproduce_by_attr(...)
    sec2 = sec.reproduce_by_attr('duration', 5.0)
    sec3 = sec.reproduce_by_attr('xyz', 100.0)
    assert math.isclose(7.0, sec.get_attr_value('duration'))
    assert math.isclose(5.0, sec2.get_attr_value('duration'))
    assert math.isclose(100.0, sec3.get_attr_value('xyz'))
    assert sec.get_attr_value('xyz') is None

    # to_dataframe()
    print(sec.to_dataframe())


if __name__ == '__main__':
    try:
        test()
    except Exception as err:
        print('Security unit port failed: ' + str(err))
        print(traceback.format_exc())

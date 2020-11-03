import pytest
import numpy as np
import pandas as pd
from ..benford import utils as ut


def test_set_N_Limit_None():
    assert ut._set_N_(100, None) == 100


def test_set_N_Limit_greater():
    assert ut._set_N_(100, 99) == 99
    assert ut._set_N_(1000, 750) == 750


def test_set_N_negative():
    with pytest.raises(ValueError) as context:
        ut._set_N_(-250, -1000)


def test_set_N_float():
    with pytest.raises(ValueError) as context:
        ut._set_N_(127.8, -100)


@pytest.fixture
def gen_array():
    arr = np.random.rand(3000) * np.random.randn(3000) * 1000
    return np.abs(arr)


@pytest.fixture
def gen_series(gen_array):
    return pd.Series(gen_array)


@pytest.fixture
def gen_data_frame(gen_array):
    return pd.DataFrame({'col1': gen_array, 'col2': gen_array})


def test_get_mantissas_less_than_1(gen_array):
    assert sum(ut.get_mantissas(gen_array) >= 1) == 0


def test_get_mantissas_less_than_0(gen_array):
    assert sum(ut.get_mantissas(gen_array) < 0) == 0


def test_input_data_Series(gen_series):
    tup = ut.input_data(gen_series)
    assert tup[0] is tup[1]


def test_input_data_array(gen_array):
    tup = ut.input_data(gen_array)
    assert tup[0] is gen_array
    assert type(tup[1]) == pd.Series


def test_input_data_wrong_tuple():
    with pytest.raises(TypeError) as context:
        ut.input_data((gen_array, 'col1'))
        ut.input_data((gen_series, 'col1'))
        ut.input_data((gen_data_frame, 2))


def test_input_data_df(gen_data_frame):
    tup = ut.input_data((gen_data_frame, 'col1'))
    assert type(tup[0]) == pd.DataFrame
    assert type(tup[1]) == pd.Series


def test_input_data_wrong_input_type(gen_array):
    with pytest.raises(TypeError) as context:
        ut.input_data(gen_array.tolist())


def test_extract_digs(gen_array):
    e_digs = ut.extract_digs(gen_array, decimals=8)
    assert len(e_digs) == len(gen_array)
    assert len(e_digs.columns) == 7

# @pytest.fixture
# def gen_input(gen_array):
#     return ut.input_data(gen_array)


# def test_prepare_1_simple_no_confidence(gen_series):
#     df = ut.prepare(gen_series, 1, simple=True)
#     assert type(df) == pd.DataFrame
#     assert len(df.columns) == 4
#     assert len(df) == 9
#     print(df.Found)
#     # assert df.Found.sum() > .999

# def test_prepare_2__conf(gen_series):
#     df = ut.prepare(gen_series, 2, confidence=95)
#     assert type(df) == tuple
#     assert len(df[1].columns) == 6
#     assert len(df[1]) == 90
#     # assert df[0] == len(gen_input[1])
#     assert 'Z_score' in df[1].columns
#     assert df[1].Found.sum() > .999

# def test_prepare_22_conf(gen_series):
#     N, df = ut.prepare(gen_series, 22, confidence=95)
#     assert len(df) == 10
#     assert df.Found.sum() == 0

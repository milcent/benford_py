import pytest
import numpy as np
import pandas as pd
from ..benford import utils as ut


@pytest.fixture
def gen_array():
    return np.abs(np.random.rand(3000) * np.random.randn(3000) * 1000)


@pytest.fixture
def gen_series(gen_array):
    return pd.Series(gen_array)


@pytest.fixture
def gen_data_frame(gen_array):
    return pd.DataFrame({'seq': gen_array, 'col2': gen_array})


@pytest.fixture
def gen_int_df(gen_data_frame):
    return gen_data_frame.astype(int)


@pytest.fixture
def small_float_array():
    return np.array([1, 2, 3, 4, 5.0, 6.3, .17])


@pytest.fixture
def small_int_array():
    return np.array([1, 2, 3, 4, 5, 6, 7])


@pytest.fixture
def small_str_dig_array():
    return np.array(['1', '2', '3', '4', '5', '6', '7'])


@pytest.fixture
def small_str_foo_array():
    return np.array(['foo', 'baar', 'baz', 'hixks'])


@pytest.fixture
def small_float_series():
    return pd.Series([1, 2, 3, 4, 5.0, 6.3, .17])


@pytest.fixture
def small_int_series():
    return pd.Series([1, 2, 3, 4, 5, 6, 7])


@pytest.fixture
def small_str_dig_series():
    return pd.Series(['1', '2', '3', '4', '5', '6', '7'])


@pytest.fixture
def small_str_foo_series():
    return pd.Series(['foo', 'baar', 'baz', 'hixks'])


@pytest.fixture
def gen_get_digs_df(gen_series):
    return ut.get_digs(gen_series, decimals=8)


@pytest.fixture
def gen_proportions_F1D(gen_get_digs_df):
    return ut.get_proportions(gen_get_digs_df.F1D)


@pytest.fixture
def gen_proportions_F2D(gen_get_digs_df):
    return ut.get_proportions(gen_get_digs_df.F2D)


@pytest.fixture
def gen_proportions_F3D(gen_get_digs_df):
    return ut.get_proportions(gen_get_digs_df.F3D)


@pytest.fixture
def gen_proportions_SD(gen_get_digs_df):
    return ut.get_proportions(gen_get_digs_df.SD)


@pytest.fixture
def gen_proportions_L2D(gen_get_digs_df):
    return ut.get_proportions(gen_get_digs_df.L2D)


@pytest.fixture
def gen_join_expect_found_diff_F1D(gen_proportions_F1D):
    return ut.join_expect_found_diff(gen_proportions_F1D, 1)


@pytest.fixture
def gen_join_expect_found_diff_F2D(gen_proportions_F2D):
    return ut.join_expect_found_diff(gen_proportions_F2D, 2)


@pytest.fixture
def gen_join_expect_found_diff_F3D(gen_proportions_F3D):
    return ut.join_expect_found_diff(gen_proportions_F3D, 3)


@pytest.fixture
def gen_join_expect_found_diff_SD(gen_proportions_SD):
    return ut.join_expect_found_diff(gen_proportions_SD, 22)


@pytest.fixture
def gen_join_expect_found_diff_L2D(gen_proportions_L2D):
    return ut.join_expect_found_diff(gen_proportions_L2D, -2)
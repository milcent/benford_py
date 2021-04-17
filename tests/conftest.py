from random import choice
import pytest
import numpy as np
import pandas as pd
from ..benford import utils as ut
from ..benford.constants import confs, len_test, rev_digs


@pytest.fixture
def gen_N():
    return np.random.randint(0, 25000)


@pytest.fixture
def gen_decimals():
    return np.random.randint(0, 8)


@pytest.fixture
def gen_N_lower(gen_N):
    return np.random.randint(0, gen_N)


@pytest.fixture
def gen_array(gen_N):
    num = gen_N
    return np.abs(np.random.rand(num) * np.random.randn(num) * 
                  np.random.randint(1, num))


@pytest.fixture
def choose_digs_rand():
    return choice([1, 2, 3, 22, -2])


@pytest.fixture
def get_random_len_by_digs(choose_digs_rand):
    return len_test[choose_digs_rand]

@pytest.fixture
def choose_test():
    return choice(["F1D","F2D","F3D","SD","L2D"])


@pytest.fixture
def choose_confidence():
    return choice(list(confs.keys())[1:])


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
def gen_get_digs_df(gen_series, gen_decimals):
    return ut.get_digs(gen_series, decimals=gen_decimals)


@pytest.fixture
def gen_proportions_F1D(gen_get_digs_df):
    return ut.get_found_proportions(gen_get_digs_df.F1D)


@pytest.fixture
def gen_proportions_F2D(gen_get_digs_df):
    return ut.get_found_proportions(gen_get_digs_df.F2D)


@pytest.fixture
def gen_proportions_F3D(gen_get_digs_df):
    return ut.get_found_proportions(gen_get_digs_df.F3D)


@pytest.fixture
def gen_proportions_SD(gen_get_digs_df):
    return ut.get_found_proportions(gen_get_digs_df.SD)


@pytest.fixture
def gen_proportions_L2D(gen_get_digs_df):
    return ut.get_found_proportions(gen_get_digs_df.L2D)


@pytest.fixture
def gen_proportions_random_test(choose_test, gen_get_digs_df):
    dig_str = choose_test
    return ut.get_found_proportions(gen_get_digs_df[dig_str]), rev_digs[dig_str]


@pytest.fixture
def gen_join_expect_found_diff_random_test(gen_proportions_random_test):
    rand_test, rand_digs = gen_proportions_random_test
    return ut.join_expect_found_diff(rand_test, rand_digs)


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

@pytest.fictue
def gen_linspaced_zero_one(cuts:int=1000):
    return np.linspace(0, 1, cuts)

@pytest.fixture
def gen_random_proportions(gen_linspaced_zero_one, get_random_len_by_digs):
    simul = np.random.choice(gen_linspaced_zero_one, get_random_len_by_digs)
    return simul / simul.sum()
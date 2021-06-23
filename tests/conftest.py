from random import choice
import pytest
import numpy as np
import pandas as pd
from ..benford import utils as ut
from ..benford.constants import CONFS, REV_DIGS
from ..benford.expected import _get_expected_digits_
from ..benford.stats import _two_dist_ks_


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
                  np.random.randint(1, num, num))

@pytest.fixture
def choose_digs_rand():
    return choice([1, 2, 3, 22, -2])


@pytest.fixture
def choose_test():
    return choice(["F1D","F2D","F3D","SD","L2D"])


@pytest.fixture
def choose_confidence():
    return choice(list(CONFS.keys())[1:])


@pytest.fixture
def gen_series(gen_array):
    return pd.Series(gen_array)


@pytest.fixture
def gen_data_frame(gen_array):
    return pd.DataFrame({'seq': gen_array, 'col2': gen_array})


@pytest.fixture
def gen_int_df(gen_data_frame):
    return gen_data_frame.astype(int)

small_arrays_type = [
    (np.array([1, 2, 3, 4, 5.0, 6.3, .17]), float),
    (np.array([1, 2, 3, 4, 5, 6, 7]), int),
    (np.array(['1', '2', '3', '4', '5', '6', '7']), float),
    (pd.Series([1, 2, 3, 4, 5.0, 6.3, .17]), float),
    (pd.Series([1, 2, 3, 4, 5, 6, 7]), int),
    (pd.Series(['1', '2', '3', '4', '5', '6', '7']), float)
]

@pytest.fixture(params=small_arrays_type)
def get_small_arrays(request):
    return request.param


@pytest.fixture
def small_str_foo_array():
    return np.array(['foo', 'baar', 'baz', 'hixks'])


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
    return ut.get_found_proportions(gen_get_digs_df[dig_str]), REV_DIGS[dig_str]


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


@pytest.fixture
def gen_linspaced_zero_one(cuts:int=1000):
    return np.linspace(0, 1, cuts)


@pytest.fixture
def gen_mantissas_ks_dists(gen_array):
    dist2 = ut.get_mantissas(gen_array)
    dist1 = np.linspace(0, 1, len(dist2), endpoint=False)
    return dist1, dist2


def gen_mantissa_distribution():
    num =  np.random.randint(1500, 5000)
    a = np.random.rand(num)
    b = np.random.randint(1, 999, num)
    c = np.random.randn(num)
    abc = np.abs(a * b * c)
    return ut.get_mantissas(abc)


mant_ks_dists_types = [
    (gen_mantissa_distribution(), np.random.choice([True, False]), np.float_) 
        for i in range(10)
]

@pytest.fixture(params=mant_ks_dists_types)
def get_mant_ks_types(request):
    dist2, cummulative, ks_type = request.param
    return np.linspace(0, 1, len(dist2), endpoint=False), \
        dist2, cummulative, ks_type


mant_dists = [
    (gen_mantissa_distribution(), np.random.choice([True, False]), 0)
        for i in range(10)
]

@pytest.fixture(params=mant_dists)
def get_mant_ks_s(request):
    dist2, cummulative, zero = request.param
    return np.linspace(0, 1, len(dist2), endpoint=False), \
        dist2, cummulative, zero

def gen_mantissa_distribution_len():
    mants = gen_mantissa_distribution()
    return mants, len(mants)

mant_ks_confidences = [
    (*gen_mantissa_distribution_len(), conf) for conf in CONFS.keys()
]

@pytest.fixture(params=mant_ks_confidences)
def get_mant_ks_confs_limit_N(request):
    mants, mants_lengths, confidence = request.param
    cap_sample = mants_lengths - np.random.randint(500, 1400)
    sample_size = np.random.choice([mants_lengths, cap_sample])
    return mants, confidence, sample_size


@pytest.fixture
def gen_random_digs_and_proportions(gen_linspaced_zero_one, choose_digs_rand):
    exp = _get_expected_digits_(choose_digs_rand).Expected.values
    rand_prop = np.random.choice(gen_linspaced_zero_one, len(exp))
    return exp, rand_prop / rand_prop.sum()
import pytest
import numpy as np
import pandas as pd
from ..benford import utils as ut



def test_set_N_Limit_None(gen_N):
    assert ut._set_N_(gen_N, None) == gen_N


def test_set_N_Limit_greater(gen_N, gen_N_lower):
    assert ut._set_N_(gen_N, gen_N_lower) == gen_N_lower


def test_set_N_negative():
    with pytest.raises(ValueError) as context:
        ut._set_N_(-250, -1000)


def test_set_N_float():
    with pytest.raises(ValueError) as context:
        ut._set_N_(127.8, -100)

def test_set_N_zero(gen_N):
    assert ut._set_N_(0, None) == 1
    assert ut._set_N_(0, gen_N) == 1

def test_get_mantissas_less_than_1(gen_array):
    assert sum(ut.get_mantissas(gen_array) > 1) == 0


def test_get_mantissas_less_than_0(gen_array):
    assert sum(ut.get_mantissas(gen_array) < 0) == 0


def test_input_data_Series(gen_series):
    tup = ut.input_data(gen_series)
    assert tup[0] is tup[1]


def test_input_data_array(gen_array):
    tup = ut.input_data(gen_array)
    assert tup[0] is gen_array
    assert type(tup[1]) == pd.Series


def test_input_data_wrong_tuple(gen_array, gen_series, gen_data_frame):
    with pytest.raises(TypeError) as context:
        ut.input_data((gen_array, 'seq'))
        ut.input_data((gen_series, 'col1'))
        ut.input_data((gen_data_frame, 2))


def test_input_data_df(gen_data_frame):
    tup = ut.input_data((gen_data_frame, 'seq'))
    assert type(tup[0]) == pd.DataFrame
    assert type(tup[1]) == pd.Series


def test_input_data_wrong_input_type(gen_array):
    with pytest.raises(TypeError) as context:
        ut.input_data(gen_array.tolist())


def test_set_sign_all(gen_data_frame):
    sign_df = ut.set_sign(gen_data_frame, 'all')
    assert len(sign_df.loc[sign_df.seq == 0]) == 0


def test_set_sign_pos(gen_data_frame):
    sign_df = ut.set_sign(gen_data_frame, 'pos')
    assert sum(sign_df.seq <= 0) == 0


def test_set_sign_neg(gen_data_frame):
    sign_df = ut.set_sign(gen_data_frame, 'neg')
    assert sum(sign_df.seq >= 0) == 0


def test_get_times_10_power_2(gen_data_frame):
    pow_df = ut.get_times_10_power(gen_data_frame)
    assert pow_df.ZN.dtype == int


def test_get_times_10_power_8(gen_data_frame):
    pow_df = ut.get_times_10_power(gen_data_frame, 8)
    assert pow_df.ZN.dtype == int
    assert (pow_df.ZN == (pow_df.seq.abs() * 10 ** 8).astype(int)).all()


def test_get_times_10_power_0(gen_int_df):
    pow_df = ut.get_times_10_power(gen_int_df)
    assert pow_df.ZN.dtype == int
    assert (pow_df.ZN == pow_df.seq.abs()).all()


def test_get_times_10_power_infer(gen_data_frame):
    pow_df = ut.get_times_10_power(gen_data_frame, 'infer')
    assert pow_df.ZN.dtype == int
    assert (pow_df.ZN.astype(str).str.len() == 5).all()


def test_get_digs_dec_8(gen_array):
    e_digs = ut.get_digs(gen_array, decimals=8)
    cols = ['seq', 'ZN', 'F1D', 'F2D', 'F3D', 'SD', 'L2D']
    assert e_digs.columns.str.contains('|'.join(cols)).all()
    assert (e_digs[['F1D', 'F2D', 'F3D', 'SD', 'L2D']].dtypes == int).all()
    assert e_digs.notna().all().all()


def test_get_digs_dec_0(gen_array):
    e_digs = ut.get_digs(gen_array, decimals=0)
    cols = ['seq', 'ZN', 'F1D', 'F2D', 'F3D', 'SD', 'L2D']
    assert e_digs.columns.str.contains('|'.join(cols)).all()
    assert (e_digs[['F1D', 'F2D', 'F3D', 'SD', 'L2D']].dtypes == int).all()
    assert e_digs.notna().all().all()


def test_get_digs_dec_2(gen_array):
    e_digs = ut.get_digs(gen_array, decimals=2)
    cols = ['seq', 'ZN', 'F1D', 'F2D', 'F3D', 'SD', 'L2D']
    assert e_digs.columns.str.contains('|'.join(cols)).all()
    assert (e_digs[['F1D', 'F2D', 'F3D', 'SD', 'L2D']].dtypes == int).all()
    assert e_digs.notna().all().all()


def test_get_digs_dec_infer(gen_array):
    e_digs = ut.get_digs(gen_array, decimals='infer')
    cols = ['seq', 'ZN', 'F1D', 'F2D', 'F3D', 'SD', 'L2D']
    assert e_digs.columns.str.contains('|'.join(cols)).all()
    assert (e_digs[['F1D', 'F2D', 'F3D', 'SD', 'L2D']].dtypes == int).all()
    assert e_digs.notna().all().all()


def test_get_proportions_F1D(gen_proportions_F1D):
    prop_f1d = gen_proportions_F1D
    assert ((prop_f1d.index >= 1) & (prop_f1d.index <= 9)).all()
    assert prop_f1d.Found.sum() > .99999
    assert (prop_f1d.Found >= 0).all()
    assert prop_f1d.Counts.dtype == int


def test_get_proportions_F2D(gen_proportions_F2D):
    prop_f2d = gen_proportions_F2D
    assert ((prop_f2d.index >= 10) & (prop_f2d.index <= 99)).all()
    assert prop_f2d.Found.sum() > .99999
    assert (prop_f2d.Found >= 0).all()
    assert prop_f2d.Counts.dtype == int


def test_get_proportions_F3D(gen_proportions_F3D):
    prop_f3d = gen_proportions_F3D
    assert ((prop_f3d.index >= 100) & (prop_f3d.index <= 999)).all()
    assert prop_f3d.Found.sum() > .99999
    assert (prop_f3d.Found >= 0).all()
    assert prop_f3d.Counts.dtype == int


def test_get_proportions_SD(gen_proportions_SD):
    prop_sd = gen_proportions_SD
    assert ((prop_sd.index >= 0) & (prop_sd.index <= 9)).all()
    assert prop_sd.Found.sum() > .99999
    assert (prop_sd.Found >= 0).all()
    assert prop_sd.Counts.dtype == int


def test_get_proportions_L2D(gen_proportions_L2D):
    prop_l2d = gen_proportions_L2D
    assert ((prop_l2d.index >= 00) & (prop_l2d.index <= 99)).all()
    assert prop_l2d.Found.sum() > .99999
    assert (prop_l2d.Found >= 0).all()
    assert prop_l2d.Counts.dtype == int


def test_join_exp_foun_diff_F1D(gen_proportions_F1D):
    jefd_F1D = ut.join_expect_found_diff(gen_proportions_F1D, 1)
    assert len(jefd_F1D) == 9
    assert (jefd_F1D.columns.str.contains('|'.join(
        ['Expected', 'Counts', 'Found', 'Dif', 'AbsDif']))).all()
    assert jefd_F1D.isna().sum().sum() == 0


def test_join_exp_foun_diff_F2D(gen_proportions_F2D):
    jefd_F2D = ut.join_expect_found_diff(gen_proportions_F2D, 2)
    assert len(jefd_F2D) == 90
    assert (jefd_F2D.columns.str.contains('|'.join(
        ['Expected', 'Counts', 'Found', 'Dif', 'AbsDif']))).all()
    assert jefd_F2D.isna().sum().sum() == 0


def test_join_exp_foun_diff_F3D(gen_proportions_F3D):
    jefd_F3D = ut.join_expect_found_diff(gen_proportions_F3D, 3)
    assert len(jefd_F3D) == 900
    assert (jefd_F3D.columns.str.contains('|'.join(
        ['Expected', 'Counts', 'Found', 'Dif', 'AbsDif']))).all()
    assert jefd_F3D.isna().sum().sum() == 0


def test_join_exp_foun_diff_SD(gen_proportions_SD):
    jefd_SD = ut.join_expect_found_diff(gen_proportions_SD, 22)
    assert len(jefd_SD) == 10
    assert (jefd_SD.columns.str.contains('|'.join(
        ['Expected', 'Counts', 'Found', 'Dif', 'AbsDif']))).all()
    assert jefd_SD.isna().sum().sum() == 0


def test_join_exp_foun_diff_L2D(gen_proportions_L2D):
    jefd_L2D = ut.join_expect_found_diff(gen_proportions_L2D, -2)
    assert len(jefd_L2D) == 100
    assert (jefd_L2D.columns.str.contains('|'.join(
        ['Expected', 'Counts', 'Found', 'Dif', 'AbsDif']))).all()
    assert jefd_L2D.isna().sum().sum() == 0


def test_prepare_F1D_simple(gen_series):
    prep_F1D = ut.prepare(gen_series, 1, simple=True)
    assert "Dif" not in prep_F1D.columns


def test_prepare_F2D_simple(gen_series):
    prep_F2D = ut.prepare(gen_series, 2, simple=True)
    assert "Dif" not in prep_F2D.columns


def test_prepare_F3D_simple(gen_series):
    prep_F3D = ut.prepare(gen_series, 3, simple=True)
    assert "Dif" not in prep_F3D.columns


def test_prepare_SD_simple(gen_series):
    prep_SD = ut.prepare(gen_series, 22, simple=True)
    assert "Dif" not in prep_SD.columns


def test_prepare_L2D_simple(gen_series):
    prep_L2D = ut.prepare(gen_series, -2, simple=True)
    assert "Dif" not in prep_L2D.columns


def test_prepare_F1D(gen_series):
    ser = gen_series
    lf = len(ser)
    num, prep_F1D = ut.prepare(ser, 1)
    assert "Z_score" in prep_F1D.columns
    assert num == lf


def test_prepare_F2D(gen_series):
    ser = gen_series
    lf = len(ser)
    num, prep_F2D = ut.prepare(ser, 2)
    assert "Z_score" in prep_F2D.columns
    assert num == lf


def test_prepare_F3D(gen_series):
    ser = gen_series
    lf = len(ser)
    num, prep_F3D = ut.prepare(ser, 3)
    assert "Z_score" in prep_F3D.columns
    assert num == lf


def test_prepare_SD(gen_series):
    ser = gen_series
    lf = len(ser)
    num, prep_SD = ut.prepare(ser, 22)
    assert "Z_score" in prep_SD.columns
    assert num == lf


def test_prepare_L2D(gen_series):
    ser = gen_series
    lf = len(ser)
    num, prep_L2D = ut.prepare(ser, -2)
    assert "Z_score" in prep_L2D.columns
    assert num == lf


def test_prepare_F1D_N(gen_N, gen_series):
    ser = gen_series
    n_diff = gen_N
    num, prep_F1D = ut.prepare(ser, 1, limit_N=n_diff)
    assert "Z_score" in prep_F1D.columns
    assert num == n_diff


def test_prepare_F2D_N(gen_N, gen_series):
    ser = gen_series
    n_diff = gen_N
    num, prep_F2D = ut.prepare(ser, 2, limit_N=n_diff)
    assert "Z_score" in prep_F2D.columns
    assert num == n_diff


def test_prepare_F3D_N(gen_N, gen_series):
    ser = gen_series
    n_diff = gen_N
    num, prep_F3D = ut.prepare(ser, 3, limit_N=n_diff)
    assert "Z_score" in prep_F3D.columns
    assert num == n_diff


def test_prepare_SD_N(gen_N, gen_series):
    ser = gen_series
    n_diff = gen_N
    num, prep_SD = ut.prepare(ser, 22, limit_N=n_diff)
    assert "Z_score" in prep_SD.columns
    assert num == n_diff


def test_prepare_L2D_N(gen_N, gen_series):
    ser = gen_series
    n_diff = gen_N
    num, prep_L2D = ut.prepare(ser, -2, limit_N=n_diff)
    assert "Z_score" in prep_L2D.columns
    assert num == n_diff
import pytest
import pandas as pd
from ..benford import utils as ut


class Test_set_N_():
        
    def test_Limit_None(self, gen_N):
        assert ut._set_N_(gen_N, None) == gen_N

    def test_Limit_greater(self, gen_N, gen_N_lower):
        assert ut._set_N_(gen_N, gen_N_lower) == gen_N_lower

    def test_negative(self, ):
        with pytest.raises(ValueError) as context:
            ut._set_N_(-250, -1000)

    def test_float(self, ):
        with pytest.raises(ValueError) as context:
            ut._set_N_(127.8, -100)

    def test_zero(self, gen_N):
        assert ut._set_N_(0, None) == 1
        assert ut._set_N_(0, gen_N) == 1


class Test_get_mantissas():
        
    def test_less_than_1(self, gen_array):
        assert sum(ut._get_mantissas_(gen_array) > 1) == 0

    def test_less_than_0(self, gen_array):
        assert sum(ut._get_mantissas_(gen_array) < 0) == 0


class Test_input_data():
        
    def test_Series(self, gen_series):
        tup = ut._input_data_(gen_series)
        assert tup[0] is tup[1]

    def test_array(self, gen_array):
        tup = ut._input_data_(gen_array)
        assert tup[0] is gen_array
        assert type(tup[1]) == pd.Series

    def test_wrong_tuple(self, gen_array, gen_series, gen_data_frame):
        with pytest.raises(TypeError) as context:
            ut._input_data_((gen_array, 'seq'))
            ut._input_data_((gen_series, 'col1'))
            ut._input_data_((gen_data_frame, 2))

    def test_df(self, gen_data_frame):
        tup = ut._input_data_((gen_data_frame, 'seq'))
        assert type(tup[0]) == pd.DataFrame
        assert type(tup[1]) == pd.Series

    def test_wrong_input_type(self, gen_array):
        with pytest.raises(TypeError) as context:
            ut._input_data_(gen_array.tolist())


class Test_set_sign():
        
    def test_all(self, gen_data_frame):
        sign_df = ut._set_sign_(gen_data_frame, 'all')
        assert len(sign_df.loc[sign_df.seq == 0]) == 0

    def test_pos(self, gen_data_frame):
        sign_df = ut._set_sign_(gen_data_frame, 'pos')
        assert sum(sign_df.seq <= 0) == 0

    def test_neg(self, gen_data_frame):
        sign_df = ut._set_sign_(gen_data_frame, 'neg')
        assert sum(sign_df.seq >= 0) == 0


class Test_get_times_10_power():
        
    def test_2(self, gen_data_frame):
        pow_df = ut.get_times_10_power(gen_data_frame)
        assert pow_df.ZN.dtype == int

    def test_8(self, gen_data_frame):
        pow_df = ut.get_times_10_power(gen_data_frame, 8)
        assert pow_df.ZN.dtype == int
        assert (pow_df.ZN == (pow_df.seq.abs() * 10 ** 8).astype(int)).all()

    def test_0(self, gen_int_df):
        pow_df = ut.get_times_10_power(gen_int_df)
        assert pow_df.ZN.dtype == int
        assert (pow_df.ZN == pow_df.seq.abs()).all()

    def test_infer(self, gen_data_frame):
        pow_df = ut.get_times_10_power(gen_data_frame, 'infer')
        assert pow_df.ZN.dtype == int
        assert (pow_df.ZN.astype(str).str.len() == 5).all()


class Test_get_all_digs():
        
    def test_dec_8(self, gen_array):
        e_digs = ut.get_all_digs(gen_array, decimals=8)
        cols = ['seq', 'ZN', 'F1D', 'F2D', 'F3D', 'SD', 'L2D']
        assert e_digs.columns.str.contains('|'.join(cols)).all()
        assert (e_digs[['F1D', 'F2D', 'F3D', 'SD', 'L2D']].dtypes == int).all()
        assert e_digs.notna().all().all()

    def test_dec_0(self, gen_array):
        e_digs = ut.get_all_digs(gen_array, decimals=0)
        cols = ['seq', 'ZN', 'F1D', 'F2D', 'F3D', 'SD', 'L2D']
        assert e_digs.columns.str.contains('|'.join(cols)).all()
        assert (e_digs[['F1D', 'F2D', 'F3D', 'SD', 'L2D']].dtypes == int).all()
        assert e_digs.notna().all().all()

    def test_dec_2(self, gen_array):
        e_digs = ut.get_all_digs(gen_array, decimals=2)
        cols = ['seq', 'ZN', 'F1D', 'F2D', 'F3D', 'SD', 'L2D']
        assert e_digs.columns.str.contains('|'.join(cols)).all()
        assert (e_digs[['F1D', 'F2D', 'F3D', 'SD', 'L2D']].dtypes == int).all()
        assert e_digs.notna().all().all()

    def test_dec_infer(self, gen_array):
        e_digs = ut.get_all_digs(gen_array, decimals='infer')
        cols = ['seq', 'ZN', 'F1D', 'F2D', 'F3D', 'SD', 'L2D']
        assert e_digs.columns.str.contains('|'.join(cols)).all()
        assert (e_digs[['F1D', 'F2D', 'F3D', 'SD', 'L2D']].dtypes == int).all()
        assert e_digs.notna().all().all()

class Test_get_found_proportions():
        
    def test_F1D(self, gen_proportions_F1D):
        prop_f1d = gen_proportions_F1D
        # assert ((prop_f1d.index >= 1) & (prop_f1d.index <= 9)).all()
        assert prop_f1d.Found.sum() > .99999
        assert (prop_f1d.Found >= 0).all()
        assert prop_f1d.Counts.dtype == int

    def test_F2D(self, gen_proportions_F2D):
        prop_f2d = gen_proportions_F2D
        # assert ((prop_f2d.index >= 10) & (prop_f2d.index <= 99)).all()
        assert prop_f2d.Found.sum() > .99999
        assert (prop_f2d.Found >= 0).all()
        assert prop_f2d.Counts.dtype == int

    def test_F3D(self, gen_proportions_F3D):
        prop_f3d = gen_proportions_F3D
        # assert ((prop_f3d.index >= 100) & (prop_f3d.index <= 999)).all()
        assert prop_f3d.Found.sum() > .99999
        assert (prop_f3d.Found >= 0).all()
        assert prop_f3d.Counts.dtype == int

    def test_SD(self, gen_proportions_SD):
        prop_sd = gen_proportions_SD
        # assert ((prop_sd.index >= 0) & (prop_sd.index <= 9)).all()
        assert prop_sd.Found.sum() > .99999
        assert (prop_sd.Found >= 0).all()
        assert prop_sd.Counts.dtype == int

    def test_L2D(self, gen_proportions_L2D):
        prop_l2d = gen_proportions_L2D
        # assert ((prop_l2d.index >= 00) & (prop_l2d.index <= 99)).all()
        assert prop_l2d.Found.sum() > .99999
        assert (prop_l2d.Found >= 0).all()
        assert prop_l2d.Counts.dtype == int


class Test_join_exp_found_diff():
        
    def test_F1D(self, gen_proportions_F1D):
        jefd_F1D = ut.join_expect_found_diff(gen_proportions_F1D, 1)
        assert len(jefd_F1D) == 9
        assert (jefd_F1D.columns.str.contains('|'.join(
            ['Expected', 'Counts', 'Found', 'Dif', 'AbsDif']))).all()
        assert jefd_F1D.isna().sum().sum() == 0

    def test_F2D(self, gen_proportions_F2D):
        jefd_F2D = ut.join_expect_found_diff(gen_proportions_F2D, 2)
        assert len(jefd_F2D) == 90
        assert (jefd_F2D.columns.str.contains('|'.join(
            ['Expected', 'Counts', 'Found', 'Dif', 'AbsDif']))).all()
        assert jefd_F2D.isna().sum().sum() == 0

    def test_F3D(self, gen_proportions_F3D):
        jefd_F3D = ut.join_expect_found_diff(gen_proportions_F3D, 3)
        assert len(jefd_F3D) == 900
        assert (jefd_F3D.columns.str.contains('|'.join(
            ['Expected', 'Counts', 'Found', 'Dif', 'AbsDif']))).all()
        assert jefd_F3D.isna().sum().sum() == 0

    def test_SD(self, gen_proportions_SD):
        jefd_SD = ut.join_expect_found_diff(gen_proportions_SD, 22)
        assert len(jefd_SD) == 10
        assert (jefd_SD.columns.str.contains('|'.join(
            ['Expected', 'Counts', 'Found', 'Dif', 'AbsDif']))).all()
        assert jefd_SD.isna().sum().sum() == 0

    def test_L2D(self, gen_proportions_L2D):
        jefd_L2D = ut.join_expect_found_diff(gen_proportions_L2D, -2)
        assert len(jefd_L2D) == 100
        assert (jefd_L2D.columns.str.contains('|'.join(
            ['Expected', 'Counts', 'Found', 'Dif', 'AbsDif']))).all()
        assert jefd_L2D.isna().sum().sum() == 0


class Test_prepare():
        
    def test_F1D_simple(self, gen_series):
        prep_F1D = ut.prepare(gen_series, 1, simple=True)
        assert "Dif" not in prep_F1D.columns

    def test_F2D_simple(self, gen_series):
        prep_F2D = ut.prepare(gen_series, 2, simple=True)
        assert "Dif" not in prep_F2D.columns

    def test_F3D_simple(self, gen_series):
        prep_F3D = ut.prepare(gen_series, 3, simple=True)
        assert "Dif" not in prep_F3D.columns

    def test_SD_simple(self, gen_series):
        prep_SD = ut.prepare(gen_series, 22, simple=True)
        assert "Dif" not in prep_SD.columns

    def test_L2D_simple(self, gen_series):
        prep_L2D = ut.prepare(gen_series, -2, simple=True)
        assert "Dif" not in prep_L2D.columns

    def test_F1D(self, gen_series):
        ser = gen_series
        lf = len(ser)
        num, prep_F1D = ut.prepare(ser, 1)
        assert "Z_score" in prep_F1D.columns
        assert num == lf

    def test_F2D(self, gen_series):
        ser = gen_series
        lf = len(ser)
        num, prep_F2D = ut.prepare(ser, 2)
        assert "Z_score" in prep_F2D.columns
        assert num == lf

    def test_F3D(self, gen_series):
        ser = gen_series
        lf = len(ser)
        num, prep_F3D = ut.prepare(ser, 3)
        assert "Z_score" in prep_F3D.columns
        assert num == lf

    def test_SD(self, gen_series):
        ser = gen_series
        lf = len(ser)
        num, prep_SD = ut.prepare(ser, 22)
        assert "Z_score" in prep_SD.columns
        assert num == lf

    def test_L2D(self, gen_series):
        ser = gen_series
        lf = len(ser)
        num, prep_L2D = ut.prepare(ser, -2)
        assert "Z_score" in prep_L2D.columns
        assert num == lf

    def test_F1D_N(self, gen_N, gen_series):
        ser = gen_series
        n_diff = gen_N
        num, prep_F1D = ut.prepare(ser, 1, limit_N=n_diff)
        assert "Z_score" in prep_F1D.columns
        assert num == n_diff

    def test_F2D_N(self, gen_N, gen_series):
        ser = gen_series
        n_diff = gen_N
        num, prep_F2D = ut.prepare(ser, 2, limit_N=n_diff)
        assert "Z_score" in prep_F2D.columns
        assert num == n_diff

    def test_F3D_N(self, gen_N, gen_series):
        ser = gen_series
        n_diff = gen_N
        num, prep_F3D = ut.prepare(ser, 3, limit_N=n_diff)
        assert "Z_score" in prep_F3D.columns
        assert num == n_diff

    def test_SD_N(self, gen_N, gen_series):
        ser = gen_series
        n_diff = gen_N
        num, prep_SD = ut.prepare(ser, 22, limit_N=n_diff)
        assert "Z_score" in prep_SD.columns
        assert num == n_diff

    def test_L2D_N(self, gen_N, gen_series):
        ser = gen_series
        n_diff = gen_N
        num, prep_L2D = ut.prepare(ser, -2, limit_N=n_diff)
        assert "Z_score" in prep_L2D.columns
        assert num == n_diff


def test_subtract_sorted(gen_series):
    ser = gen_series
    sort = ut.subtract_sorted(ser)
    assert len(ser) - len(sort) >= 1
    assert (sort != 0).all()

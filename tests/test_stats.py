import pytest
from ..benford import stats as st

def test_Z_score_F1D_N0(gen_join_expect_found_diff_F1D):
    z_f1d = st.Z_score(gen_join_expect_found_diff_F1D, 0)
    assert z_f1d == 0


def test_Z_score_F2D_N0(gen_join_expect_found_diff_F2D):
    z_f2d = st.Z_score(gen_join_expect_found_diff_F2D, 0)
    assert z_f2d == 0


def test_Z_score_F3D_N0(gen_join_expect_found_diff_F3D):
    z_f3d = st.Z_score(gen_join_expect_found_diff_F3D, 0)
    assert z_f3d == 0


def test_Z_score_SD_N0(gen_join_expect_found_diff_SD):
    z_sd = st.Z_score(gen_join_expect_found_diff_SD, 0)
    assert z_sd == 0


def test_Z_score_L2D_N0(gen_join_expect_found_diff_L2D):
    z_l2d = st.Z_score(gen_join_expect_found_diff_L2D, 0)
    assert z_l2d == 0


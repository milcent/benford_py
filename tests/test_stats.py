import pytest
from ..benford import stats as st
from ..benford.constants import crit_chi2


def test_Z_score_F1D():
    pass

class TestChiSquare():
        
    def test_conf_None(self, gen_join_expect_found_diff_F1D, capsys):
        jefd_F1D = gen_join_expect_found_diff_F1D
        chi = st.chi_sq(jefd_F1D, len(jefd_F1D) - 1, None)
        out, _ = capsys.readouterr()
        assert "Chi-square test needs confidence other than None." in out
        assert chi is None

    def test_random_conf_F1D(self, gen_join_expect_found_diff_F1D, 
                             choose_confidence):
        jefd_F1D = gen_join_expect_found_diff_F1D
        ddf = len(jefd_F1D) - 1
        confidence = choose_confidence
        chis = st.chi_sq(jefd_F1D, ddf, choose_confidence, verbose=False)
        assert chis[1] == crit_chi2[ddf][confidence]
        assert chis[0] > 0
        assert isinstance(chis[0], float)
        

    def test_random_conf_F2D(self, gen_join_expect_found_diff_F2D, 
                             choose_confidence):
        jefd_F2D = gen_join_expect_found_diff_F2D
        ddf = len(jefd_F2D) - 1
        confidence = choose_confidence
        chis = st.chi_sq(jefd_F2D, ddf, choose_confidence, verbose=False)
        assert chis[1] == crit_chi2[ddf][confidence]
        assert chis[0] > 0
        assert isinstance(chis[0], float)

    def test_random_conf_F3D(self, gen_join_expect_found_diff_F3D, 
                             choose_confidence):
        jefd_F3D = gen_join_expect_found_diff_F3D
        ddf = len(jefd_F3D) - 1
        confidence = choose_confidence
        chis = st.chi_sq(jefd_F3D, ddf, choose_confidence, verbose=False)
        assert chis[1] == crit_chi2[ddf][confidence]
        assert chis[0] > 0
        assert isinstance(chis[0], float)

    def test_random_conf_SD(self, gen_join_expect_found_diff_SD, 
                            choose_confidence):
        jefd_SD = gen_join_expect_found_diff_SD
        ddf = len(jefd_SD) - 1
        confidence = choose_confidence
        chis = st.chi_sq(jefd_SD, ddf, choose_confidence, verbose=False)
        assert chis[1] == crit_chi2[ddf][confidence]
        assert chis[0] > 0
        assert isinstance(chis[0], float)

    def test_random_conf_L2D(self, gen_join_expect_found_diff_L2D, 
                             choose_confidence):
        jefd_L2D = gen_join_expect_found_diff_L2D
        ddf = len(jefd_L2D) - 1
        confidence = choose_confidence
        chis = st.chi_sq(jefd_L2D, ddf, choose_confidence, verbose=False)
        assert chis[1] == crit_chi2[ddf][confidence]
        assert chis[0] > 0
        assert isinstance(chis[0], float)
    
    def test_rand_test_rand_conf_verbose(self, choose_confidence,
                            gen_join_expect_found_diff_random_test, capsys):
        r_test = gen_join_expect_found_diff_random_test
        ddf = len(r_test) - 1
        conf = choose_confidence
        chis = st.chi_sq(r_test, ddf, conf)
        out, _ = capsys.readouterr()
        assert f"The Chi-square statistic is {chis[0]:.4f}." in out
        assert f"Critical Chi-square for this series: {chis[1]}." in out
    
    def test_rand_test_conf_80(self, gen_join_expect_found_diff_random_test):
        r_test = gen_join_expect_found_diff_random_test
        ddf = len(r_test) - 1
        chis = st.chi_sq(r_test, ddf, 80, verbose=False)
        assert chis[1] == crit_chi2[ddf][80]
        assert chis[0] > 0
        assert isinstance(chis[0], float)

    def test_rand_test_conf_85(self, gen_join_expect_found_diff_random_test):
        r_test = gen_join_expect_found_diff_random_test
        ddf = len(r_test) - 1
        chis = st.chi_sq(r_test, ddf, 85, verbose=False)
        assert chis[1] == crit_chi2[ddf][85]
        assert chis[0] > 0
        assert isinstance(chis[0], float)

    def test_rand_test_conf_90(self, gen_join_expect_found_diff_random_test):
        r_test = gen_join_expect_found_diff_random_test
        ddf = len(r_test) - 1
        chis = st.chi_sq(r_test, ddf, 90, verbose=False)
        assert chis[1] == crit_chi2[ddf][90]
        assert chis[0] > 0
        assert isinstance(chis[0], float)

    def test_rand_test_conf_95(self, gen_join_expect_found_diff_random_test,
                               capsys):
        r_test = gen_join_expect_found_diff_random_test
        ddf = len(r_test) - 1
        chis = st.chi_sq(r_test, ddf, 95, verbose=False)
        out, _ = capsys.readouterr()
        assert chis[1] == crit_chi2[ddf][95]
        assert chis[0] > 0
        assert isinstance(chis[0], float)
        assert f"The Chi-square statistic is {chis[0]:.4f}." not in out
        assert f"Critical Chi-square for this series: {chis[1]}." not in out

    def test_rand_test_conf_99(self, gen_join_expect_found_diff_random_test):
        r_test = gen_join_expect_found_diff_random_test
        ddf = len(r_test) - 1
        chis = st.chi_sq(r_test, ddf, 99, verbose=False)
        assert chis[1] == crit_chi2[ddf][99]
        assert chis[0] > 0
        assert isinstance(chis[0], float)

    def test_rand_test_conf_999(self, gen_join_expect_found_diff_random_test):
        r_test = gen_join_expect_found_diff_random_test
        ddf = len(r_test) - 1
        chis = st.chi_sq(r_test, ddf, 99.9, verbose=False)
        assert chis[1] == crit_chi2[ddf][99.9]
        assert chis[0] > 0
        assert isinstance(chis[0], float)

    def test_rand_test_conf_9999(self, gen_join_expect_found_diff_random_test):
        r_test = gen_join_expect_found_diff_random_test
        ddf = len(r_test) - 1
        chis = st.chi_sq(r_test, ddf, 99.99, verbose=False)
        assert chis[1] == crit_chi2[ddf][99.99]
        assert chis[0] > 0
        assert isinstance(chis[0], float)

    def test_rand_test_conf_99999(self, gen_join_expect_found_diff_random_test):
        r_test = gen_join_expect_found_diff_random_test
        ddf = len(r_test) - 1
        chis = st.chi_sq(r_test, ddf, 99.999, verbose=False)
        assert chis[1] == crit_chi2[ddf][99.999]
        assert chis[0] > 0
        assert isinstance(chis[0], float)

    def test_rand_test_conf_999999(self, gen_join_expect_found_diff_random_test):
        r_test = gen_join_expect_found_diff_random_test
        ddf = len(r_test) - 1
        chis = st.chi_sq(r_test, ddf, 99.9999, verbose=False)
        assert chis[1] == crit_chi2[ddf][99.9999]
        assert chis[0] > 0
        assert isinstance(chis[0], float)

    def test_rand_test_conf_9999999(self, gen_join_expect_found_diff_random_test):
        r_test = gen_join_expect_found_diff_random_test
        ddf = len(r_test) - 1
        chis = st.chi_sq(r_test, ddf, 99.99999, verbose=False)
        assert chis[1] == crit_chi2[ddf][99.99999]
        assert chis[0] > 0
        assert isinstance(chis[0], float)

class TestBhattacharyya():

    def test_coeff(self, gen_random_digs_and_proportions):
        exp, rand_prop = gen_random_digs_and_proportions
        bhat_coeff = st._bhattacharyya_coefficient(exp, rand_prop)
        assert isinstance(bhat_coeff, float)
        assert bhat_coeff >= 0
        assert bhat_coeff <= 1
    
    def test_distance(self, gen_random_digs_and_proportions):
        exp, rand_prop = gen_random_digs_and_proportions
        bhat_dist = st._bhattacharyya_distance_(exp, rand_prop)
        assert isinstance(bhat_dist, float)
        assert bhat_dist >= 0


class TestKLDivergence():

    def test_kld(self, gen_random_digs_and_proportions):
        exp, rand_prop = gen_random_digs_and_proportions
        kl_diverg = st._kullback_leibler_divergence_(exp, rand_prop)
        assert isinstance(kl_diverg, float)
        assert kl_diverg >= 0

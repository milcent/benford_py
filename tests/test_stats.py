import pytest
from ..benford import stats as st
from ..benford.constants import crit_chi2


def test_Z_score_F1D():
    pass

class Test_chi_sq():
        
    def test_conf_None(self, gen_join_expect_found_diff_F1D, capsys):
        jefd_F1D = gen_join_expect_found_diff_F1D
        chi = st.chi_sq(jefd_F1D, len(jefd_F1D) - 1, None)
        out, _ = capsys.readouterr()
        assert out == "\nChi-square test needs confidence other than None.\n"
        assert chi is None

    def test_random_conf(self, gen_join_expect_found_diff_F1D, 
                        choose_confidence, capsys):
        jefd_F1D = gen_join_expect_found_diff_F1D
        ddf = len(jefd_F1D) - 1
        confidence = choose_confidence
        chis = st.chi_sq(jefd_F1D, ddf, choose_confidence)
        assert chis[1] == crit_chi2[ddf][confidence]
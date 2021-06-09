from numpy import abs as nabs, errstate, linspace, log, sqrt, where
from .constants import CRIT_CHI2, CRIT_KS, MAD_CONFORM, DIGS


def Z_score(frame, N):
    """Computes the Z statistics for the proportions studied

    Args:
        frame: DataFrame with the expected proportions and the already calculated
            Absolute Diferences between the found and expeccted proportions
        N: sample size

    Returns:
        Series of computed Z scores
    """
    return (frame.AbsDif - (1 / (2 * N))) / sqrt(
           (frame.Expected * (1. - frame.Expected)) / N)


def chi_sq(frame, ddf, confidence, verbose=True):
    """Comnputes the chi-square statistic of the found distributions and compares
    it with the critical chi-square of such a sample, according to the
    confidence level chosen and the degrees of freedom - len(sample) -1.

    Args:
        frame: DataFrame with Found, Expected and their difference columns.
        ddf: Degrees of freedom to consider.
        confidence: Confidence level to look up critical value.
        verbose: prints the chi-squre result and compares to the critical
            chi-square for the sample. Defaults to True.

    Returns:
        The computed Chi square statistic and the critical chi square
            (according) to the degrees of freedom and confidence level,
            for comparison. None if confidence is None
    """
    if confidence is None:
        print('\nChi-square test needs confidence other than None.')
        return
    else:
        exp_counts = frame.Counts.sum() * frame.Expected
        dif_counts = frame.Counts - exp_counts
        found_chi = (dif_counts ** 2 / exp_counts).sum()
        crit_chi = CRIT_CHI2[ddf][confidence]
        if verbose:
            print(f"\nThe Chi-square statistic is {found_chi:.4f}.\n"
                  f"Critical Chi-square for this series: {crit_chi}.")
        return (found_chi, crit_chi)


def chi_sq_2(frame):
    """Computes the chi-square statistic of the found distributions

    Args:
        frame: DataFrame with Found, Expected and their difference columns.

    Returns:
        The computed Chi square statistic 
    """
    exp_counts = frame.Counts.sum() * frame.Expected
    dif_counts = frame.Counts - exp_counts
    return (dif_counts ** 2 / exp_counts).sum()


def kolmogorov_smirnov(frame, confidence, N, verbose=True):
    """Computes the Kolmogorov-Smirnov test of the found distributions
    and compares it with the critical chi-square of such a sample,
    according to the confidence level chosen.

    Args:
        frame: DataFrame with Foud and Expected distributions.
        confidence: Confidence level to look up critical value.
        N: Sample size
        verbose: prints the KS result and the critical value for the sample.
            Defaults to True.

    Returns:
        The Suprem, which is the greatest absolute difference between the
            Found and the expected proportions, and the Kolmogorov-Smirnov
            critical value according to the confidence level, for ccomparison
    """
    if confidence is None:
        print('\nKolmogorov-Smirnov test needs confidence other than None.')
        return
    else:
        # sorting and calculating the cumulative distribution
        ks_frame = frame.sort_index()[['Found', 'Expected']].cumsum()
        # finding the supremum - the largest cumul dist difference
        suprem = ((ks_frame.Found - ks_frame.Expected).abs()).max()
        # calculating the crittical value according to confidence
        crit_KS = CRIT_KS[confidence] / sqrt(N)

        if verbose:
            print(f"\nThe Kolmogorov-Smirnov statistic is {suprem:.4f}.\n"
                  f"Critical K-S for this series: {crit_KS:.4f}")
        return (suprem, crit_KS)


def kolmogorov_smirnov_2(frame):
    """Computes the Kolmogorov-Smirnov test of the found distributions

    Args:
        frame: DataFrame with Foud and Expected distributions.

    Returns:
        The Suprem, which is the greatest absolute difference between the
            Found end th expected proportions
    """
    # sorting and calculating the cumulative distribution
    ks_frame = frame.sort_index()[['Found', 'Expected']].cumsum()
    # finding the supremum - the largest cumul dist difference
    return ((ks_frame.Found - ks_frame.Expected).abs()).max()


def _two_dist_ks_(dist1, dist2, cummulative=True):
    """Computes the Kolmogorov-Smirnov statistic between two distributions,
    a found one (dist2) and an expected one (dist1).

    Args:
        dist1 (np.arrat): array with the expected distribution
        dist2 (np.array): array with the found distribution
        cummulative (bool): makes apply cummulutative sum to the
            distributions (empirical cdf).

    Returns:
        tuple(floats): the KS statistic 
    """
    if not cummulative:
        return nabs(dist2.sort() - dist1.sort()).max()
    return nabs(dist2.sort().cumsum() - dist1.sort().cumsum()).max()


def _mantissas_ks_(mant_dist, confidence):
    """Computes the Kolmogorov-Smirnof statistic for the Mantissas, also
    providing the KS critical value according the the sample size and
    confidence level provided

    Args:
        mant_dist (np.array): array with the mantissas distribution found
        confidence (float, int): level of confidence to compute the critical
            value

    Returns:
        tuple(floats): the KS statistic and the critical value
    """ 
    size = len(mant_dist)
    crit_ks = CRIT_KS[confidence] * sqrt(2 * size / size ** 2)\
                if confidence else None
    # non-cummulative, uniformly distributed
    expected = linspace(0, 1, size, endpoint=False)
    ks = _two_dist_ks_(expected, mant_dist, cummulative=False)
    return ks, crit_ks


def mad(frame, test, verbose=True):
    """Computes the Mean Absolute Deviation (MAD) between the found and the
    expected proportions.

    Args:
        frame: DataFrame with the Absolute Deviations already calculated.
        test: Test to compute the MAD from (F1D, SD, F2D...)
        verbose: prints the MAD result and compares to limit values of
            conformity. Defaults to True.

    Returns:
        The Mean of the Absolute Deviations between the found and expected
            proportions. 
    """
    mad = frame.AbsDif.mean()

    if verbose:
        print(f"\nThe Mean Absolute Deviation is {mad}")

        if test != -2:
            print(f"For the {MAD_CONFORM[DIGS[test]]}:\n\
            - 0.0000 to {MAD_CONFORM[test][0]}: Close Conformity\n\
            - {MAD_CONFORM[test][0]} to {MAD_CONFORM[test][1]}: Acceptable Conformity\n\
            - {MAD_CONFORM[test][1]} to {MAD_CONFORM[test][2]}: Marginally Acceptable Conformity\n\
            - Above {MAD_CONFORM[test][2]}: Nonconformity")
        else:
            pass
    return mad


def mse(frame, verbose=True):
    """Computes the test's Mean Square Error

    Args:
        frame: DataFrame with the already computed Absolute Deviations between
            the found and expected proportions
        verbose: Prints the MSE. Defaults to True.

    Returns:
        Mean of the squared differences between the found and the expected proportions.
    """
    mse = (frame.AbsDif ** 2).mean()

    if verbose:
        print(f"\nMean Square Error = {mse}")

    return mse

def _bhattacharyya_coefficient(dist_1, dist_2):
    """Computes the Bhattacharyya Coeficient between two probability
    distributions, to be letar used to compute the Bhattacharyya Distance

    Args:
        dist_1 (np.array): The newly gathered distribution, to be compared
            with an older / established distribution.
        dist_2 (np.array): The older/ establhished distribution with which
            the new one will be compared. 
    
    Returns:
        bhat_coef (float)
    """
    return sqrt(dist_1 * dist_2).sum()


def _bhattacharyya_distance_(dist_1, dist_2):
    """Computes the Bhattacharyya Dsitance between two probability
    distributions

    Args:
        dist_1 (np.array): The newly gathered distribution, to be compared
            with an older / established distribution.
        dist_2 (np.array): The older/ establhished distribution with which
            the new one will be compared. 
    
    Returns:
        bhat_dist (float)
    """
    with errstate(divide='ignore'):
        bhat_dist =  -log(_bhattacharyya_coefficient(dist_1, dist_2))
    return bhat_dist


def _kullback_leibler_divergence_(dist_1, dist_2):
    """Computes the Kullback-Leibler Divergence between two probability
    distributions.

    Args:
        dist_1 (np.array): The newly gathered distribution, to be compared
            with an older / established distribution.
        dist_2 (np.array): The older/ establhished distribution with which
            the new one will be compared. 

    Returns:
        kulb_leib_diverg (float)        
    """
    # ignore divide by zero warning in np.where
    with errstate(divide='ignore'):
        kl_d = (log((dist_1 / dist_2), where=(dist_1 != 0)) * dist_1).sum()
    return kl_d

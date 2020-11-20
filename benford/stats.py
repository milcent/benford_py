from numpy import sqrt
from .constants import crit_chi2, KS_crit, mad_dict, digs_dict


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
        crit_chi = crit_chi2[ddf][confidence]
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
        crit_KS = KS_crit[confidence] / sqrt(N)

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
            print(f"For the {mad_dict[digs_dict[test]]}:\n\
            - 0.0000 to {mad_dict[test][0]}: Close Conformity\n\
            - {mad_dict[test][0]} to {mad_dict[test][1]}: Acceptable Conformity\n\
            - {mad_dict[test][1]} to {mad_dict[test][2]}: Marginally Acceptable Conformity\n\
            - Above {mad_dict[test][2]}: Nonconformity")
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

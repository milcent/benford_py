from numpy import sqrt
# from .constants import digs_dict, confs, crit_chi2, KS_crit, mad_dict


def _Z_score(frame, N):
    '''
    Returns the Z statistics for the proportions assessed

    frame -> DataFrame with the expected proportions and the already calculated
            Absolute Diferences between the found and expeccted proportions
    N -> sample size
    '''
    return (frame.AbsDif - (1 / (2 * N))) / sqrt(
           (frame.Expected * (1. - frame.Expected)) / N)


def _chi_square_(frame, ddf, confidence, verbose=True):
    '''
    Returns the chi-square statistic of the found distributions and compares
    it with the critical chi-square of such a sample, according to the
    confidence level chosen and the degrees of freedom - len(sample) -1.

    Parameters
    ----------
    frame:      DataFrame with Found, Expected and their difference columns.

    ddf:        Degrees of freedom to consider.

    confidence: Confidence level - confs dict.

    verbose:     prints the chi-squre result and compares to the critical
    chi-square for the sample. Defaults to True.
    '''
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


def _chi_square_2(frame):
    '''
    Returns the chi-square statistic of the found distributions

    Parameters
    ----------
    frame:      DataFrame with Found, Expected and their difference columns.

    '''
    exp_counts = frame.Counts.sum() * frame.Expected
    dif_counts = frame.Counts - exp_counts
    return (dif_counts ** 2 / exp_counts).sum()


def _KS_(frame, confidence, N, verbose=True):
    '''
    Returns the Kolmogorov-Smirnov test of the found distributions
    and compares it with the critical chi-square of such a sample,
    according to the confidence level chosen.

    Parameters
    ----------
    frame: DataFrame with Foud and Expected distributions.

    confidence: Confidence level - confs dict.

    N: Sample size

    verbose: prints the KS result and the critical value for the sample.
        Defaults to True.
    '''
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


def _KS_2(frame):
    '''
    Returns the Kolmogorov-Smirnov test of the found distributions.

    Parameters
    ----------
    frame: DataFrame with Foud and Expected distributions.
    '''
    # sorting and calculating the cumulative distribution
    ks_frame = frame.sort_index()[['Found', 'Expected']].cumsum()
    # finding the supremum - the largest cumul dist difference
    return ((ks_frame.Found - ks_frame.Expected).abs()).max()


def _mad_(frame, test, verbose=True):
    '''
    Returns the Mean Absolute Deviation (MAD) between the found and the
    expected proportions.

    Parameters
    ----------

    frame: DataFrame with the Absolute Deviations already calculated.

    test: Test to compute the MAD from (F1D, SD, F2D...)

    verbose: prints the MAD result and compares to limit values of
        conformity. Defaults to True.
    '''
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


def _mse_(frame, verbose=True):
    '''
    Returns the test's Mean Square Error

    frame -> DataFrame with the already computed Absolute Deviations between
            the found and expected proportions

    verbose -> Prints the MSE. Defaults to True. If False, returns MSE.
    '''
    mse = (frame.AbsDif ** 2).mean()

    if verbose:
        print(f"\nMean Square Error = {mse}")

    return mse

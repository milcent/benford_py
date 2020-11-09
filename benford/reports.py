import warnings
from .constants import mad_dict


def _inform_(df, high_Z, conf):
    """Selects and sorts by the Z_stats chosen to be considered, informing or not.
    """

    if isinstance(high_Z, int):
        if conf is not None:
            dd = df[['Expected', 'Found', 'Z_score'
                     ]].sort_values('Z_score', ascending=False).head(high_Z)
            print(f'\nThe entries with the top {high_Z} Z scores are:\n')
        # Summation Test
        else:
            dd = df[['Expected', 'Found', 'AbsDif'
                     ]].sort_values('AbsDif', ascending=False
                                    ).head(high_Z)
            print(f'\nThe entries with the top {high_Z} absolute deviations '
                  'are:\n')
    else:
        if high_Z == 'pos':
            m1 = df.Dif > 0
            m2 = df.Z_score > conf
            dd = df[['Expected', 'Found', 'Z_score'
                     ]].loc[m1 & m2].sort_values('Z_score', ascending=False)
            print('\nThe entries with the significant positive '
                  'deviations are:\n')
        elif high_Z == 'neg':
            m1 = df.Dif < 0
            m2 = df.Z_score > conf
            dd = df[['Expected', 'Found', 'Z_score'
                     ]].loc[m1 & m2].sort_values('Z_score', ascending=False)
            print('\nThe entries with the significant negative '
                  'deviations are:\n')
        else:
            dd = df[['Expected', 'Found', 'Z_score'
                     ]].loc[df.Z_score > conf].sort_values('Z_score',
                                                           ascending=False)
            print('\nThe entries with the significant deviations are:\n')
    print(dd)


def _report_mad_(digs, MAD):
    """Reports the test Mean Absolut Deviation and compares it to critical values
    """
    print(f'Mean Absolute Deviation: {MAD:.6f}')
    if digs != -2:
        mads = mad_dict[digs]
        if MAD <= mads[0]:
            print(f'MAD <= {mads[0]:.6f}: Close conformity.\n')
        elif MAD <= mads[1]:
            print(f'{mads[0]:.6f} < MAD <= {mads[1]:.6f}: '
                  'Acceptable conformity.\n')
        elif MAD <= mads[2]:
            print(f'{mads[1]:.6f} < MAD <= {mads[2]:.6f}: '
                  'Marginally Acceptable conformity.\n')
        else:
            print(f'MAD > {mads[2]:.6f}: Nonconformity.\n')
    else:
        print("There is no conformity check for this test's MAD.\n")


def _report_KS_(KS, crit_KS):
    """Reports the test Kolmogorov-Smirnov statistic and compares it to critical
    values, depending on the confidence level
    """
    result = 'PASS' if KS <= crit_KS else 'FAIL'
    print(f"\n\tKolmogorov-Smirnov: {KS:.6f}",
          f"\n\tCritical value: {crit_KS:.6f} -- {result}")


def _report_chi2_(chi2, crit_chi2):
    """Reports the test Chi-square statistic and compares it to critical values,
    depending on the confidence level
    """
    result = 'PASS' if chi2 <= crit_chi2 else 'FAIL'
    print(f"\n\tChi square: {chi2:.6f}",
          f"\n\tCritical value: {crit_chi2:.6f} -- {result}")


def _report_Z_(df, high_Z, crit_Z):
    """Reports the test Z scores and compares them to a critical value,
    depending on the confidence level
    """
    print(f"\n\tCritical Z-score:{crit_Z}.")
    _inform_(df, high_Z, crit_Z)


def _report_summ_(test, high_diff):
    """Reports the Summation Test Absolute Differences between the Found and
    the Expected proportions

    """
    if high_diff is not None:
        print(f'\nThe top {high_diff} Absolute Differences are:\n')
        print(test.sort_values('AbsDif', ascending=False).head(high_diff))
    else:
        print('\nThe top Absolute Differences are:\n')
        print(test.sort_values('AbsDif', ascending=False))


def _report_test_(test, high=None, crit_vals=None):
    """Main report function. Receives the Args: to report with, initiates
    the process, and calls the right reporting helper function(s), depending
    on the Test.
    """
    print('\n', f'  {test.name}  '.center(50, '#'), '\n')
    if not 'Summation' in test.name:
        _report_mad_(test.digs, test.MAD)
        if test.confidence is not None:
            print(f"For confidence level {test.confidence}%: ")
            _report_KS_(test.KS, crit_vals['KS'])
            _report_chi2_(test.chi_square, crit_vals['chi2'])
            _report_Z_(test, high, crit_vals['Z'])
        else:
            print('Confidence is currently `None`. Set the confidence level, '
                  'so as to generate comparable critical values.')
            if isinstance(high, int):
                _inform_(test, high, None)
    else:
        _report_summ_(test, high)


def _report_mantissa_(stats):
    """Prints the mantissas statistics and their respective reference values

    Args:
        stats (dict): 
    """
    print("\n", '  Mantissas Test  '.center(52, '#'))
    print(f"\nThe Mantissas MEAN is      {stats['Mean']:.6f}."
          "\tRef: 0.5")
    print(f"The Mantissas VARIANCE is  {stats['Var']:.6f}."
          "\tRef: 0.08333")
    print(f"The Mantissas SKEWNESS is  {stats['Skew']:.6f}."
          "\tRef: 0.0")
    print(f"The Mantissas KURTOSIS is  {stats['Kurt']:.6f}."
          "\tRef: -1.2\n")


def _deprecate_inform_(verbose, inform):
    """
    Raises:
        FutureWarning: if the arg `inform` is used (to be deprecated).    
    """
    if inform is None:
        return verbose
    else:
        warnings.warn('The parameter `inform` will be deprecated in future '
                      'versions. Use `verbose` instead.',
                      FutureWarning)
        return inform

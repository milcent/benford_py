from numpy import array, arange, maximum, sqrt
import matplotlib.pyplot as plt
from .constants import colors


def _plot_expected_(df, digs):
    '''
    Plots the Expected Benford Distributions

    df   -> DataFrame with the Expected Proportions
    digs -> Test's digit
    '''
    if digs in [1, 2, 3]:
        y_max = (df.Expected.max() + (10 ** -(digs) / 3)) * 100
        fig, ax = plt.subplots(figsize=(2 * (digs ** 2 + 5), 1.5 *
                                        (digs ** 2 + 5)))
    elif digs == 22:
        y_max = 13.
        fig, ax = plt.subplots(figsize=(14, 10.5))
    elif digs == -2:
        y_max = 1.1
        fig, ax = plt.subplots(figsize=(15, 8))
    plt.title('Expected Benford Distributions', size='xx-large')
    plt.xlabel(df.index.name, size='x-large')
    plt.ylabel('Distribution (%)', size='x-large')
    ax.set_facecolor(colors['b'])
    ax.set_ylim(0, y_max)
    ax.bar(df.index, df.Expected * 100, color=colors['t'], align='center')
    ax.set_xticks(df.index)
    ax.set_xticklabels(df.index)
    plt.show(block=False)

def _get_plot_args(digs):
    '''
    Gets the correct arguments for the plotting functions, depending on the
    the test (digs) chosen.
    '''
    if digs in [1, 2, 3]:
        text_x = False
        n, m = 10 ** (digs - 1), 10 ** (digs)
        x = arange(n, m)
        figsize = (2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5))
    elif digs == 22:
        text_x = False
        x = arange(10)
        figsize = (14, 10)
    else:
        text_x = True
        x = arange(100)
        figsize = (15, 7)
    return x, figsize, text_x
    

def _plot_dig_(df, x, y_Exp, y_Found, N, figsize, conf_Z, text_x=False):
    '''
    Plots the digits tests results

    df -> DataFrame with the data to be plotted
    x -> sequence to be used in the x axis
    y_Exp -> sequence of the expected proportions to be used in the y axis
        (line)
    y_Found -> sequence of the found proportions to be used in the y axis
        (bars)
    N -> lenght of sequence, to be used when plotting the confidence levels
    figsize - > tuple to state the size of the plot figure
    conf_Z -> Confidence level
    text_x -> Forces to show all x ticks labels. Defaluts to True.
    '''
    if len(x) > 10:
        rotation = 90
    else:
        rotation = 0
    fig, ax = plt.subplots(figsize=figsize)
    plt.title('Expected vs. Found Distributions', size='xx-large')
    plt.xlabel('Digits', size='x-large')
    plt.ylabel('Distribution (%)', size='x-large')
    if conf_Z is not None:
        sig = conf_Z * sqrt(y_Exp * (1 - y_Exp) / N)
        upper = y_Exp + sig + (1 / (2 * N))
        lower_zeros = array([0]*len(upper))
        lower = maximum(y_Exp - sig - (1 / (2 * N)), lower_zeros)
        u = (y_Found < lower) | (y_Found > upper)
        c = array([colors['m']] * len(u))
        c[u] = colors['af']
        lower *= 100.
        upper *= 100.
        ax.plot(x, upper, color=colors['s'], zorder=5)
        ax.plot(x, lower, color=colors['s'], zorder=5)
        ax.fill_between(x, upper, lower, color=colors['s'],
                        alpha=.3, label='Conf')
    else:
        c = colors['m']
    ax.bar(x, y_Found * 100., color=c, label='Found', zorder=3, align='center')
    ax.plot(x, y_Exp * 100., color=colors['s'], linewidth=2.5,
            label='Benford', zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=rotation)
    ax.set_facecolor(colors['b'])
    if text_x:
        ind = array(df.index).astype(str)
        ind[:10] = array(['00', '01', '02', '03', '04', '05',
                             '06', '07', '08', '09'])
        plt.xticks(x, ind, rotation='vertical')
    ax.legend()
    ax.set_ylim(0, max([y_Exp.max() * 100, y_Found.max() * 100]) + 10 / len(x))
    ax.set_xlim(x[0] - 1, x[-1] + 1)
    plt.show(block=False)


def _plot_sum_(df, figsize, li, text_x=False):
    '''
    Plots the summation test results

    df -> DataFrame with the data to be plotted

    figsize - > tuple to state the size of the plot figure

    li -> values with which to draw the horizontal line
    '''
    x = df.index
    rotation = 90 if len(x) > 10 else 0
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.title('Expected vs. Found Sums')
    plt.xlabel('Digits')
    plt.ylabel('Sums')
    ax.bar(x, df.Percent, color=colors['m'],
           label='Found Sums', zorder=3, align='center')
    ax.set_xlim(x[0] - 1, x[-1] + 1)
    ax.axhline(li, color=colors['s'], linewidth=2, label='Expected', zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=rotation)
    ax.set_facecolor(colors['b'])
    if text_x:
        ind = array(x).astype(str)
        ind[:10] = array(['00', '01', '02', '03', '04', '05',
                             '06', '07', '08', '09'])
        plt.xticks(x, ind, rotation='vertical')
    ax.legend()
    plt.show(block=False)

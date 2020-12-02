from numpy import array, arange, maximum, sqrt, ones
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
from .constants import colors, mad_dict


def plot_expected(df, digs, save_plot=None, save_plot_kwargs=None):
    """Plots the Expected Benford Distributions

    Args:
        df: DataFrame with the Expected Proportions
        digs: Test's digit
        save_plot: string with the path/name of the file in which the generated
            plot will be saved. Uses matplotlib.pyplot.savefig(). File format
            is infered by the file name extension.
        save_plot_kwargs: dict with any of the kwargs accepted by
            matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
    """
    if digs in [1, 2, 3]:
        y_max = (df.Expected.max() + (10 ** -(digs) / 3)) * 100
        figsize = 2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5)
    elif digs == 22:
        y_max = 13.
        figsize = 14, 10.5
    elif digs == -2:
        y_max = 1.1
        figsize = 15, 8
    fig, ax = plt.subplots(figsize=figsize)
    plt.title('Expected Benford Distributions', size='xx-large')
    plt.xlabel(df.index.name, size='x-large')
    plt.ylabel('Distribution (%)', size='x-large')
    ax.set_facecolor(colors['b'])
    ax.set_ylim(0, y_max)
    ax.bar(df.index, df.Expected * 100, color=colors['t'], align='center')
    ax.set_xticks(df.index)
    ax.set_xticklabels(df.index)

    if save_plot:
        if not save_plot_kwargs:
            save_plot_kwargs = {}
        plt.savefig(save_plot, **save_plot_kwargs)

    plt.show(block=False)


def _get_plot_args(digs):
    """Selects the correct arguments for the plotting functions, depending on the
    the test (digs) chosen.
    """
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

def plot_digs(df, x, y_Exp, y_Found, N, figsize, conf_Z, text_x=False,
              save_plot=None, save_plot_kwargs=None):
    """Plots the digits tests results

    Args:
        df: DataFrame with the data to be plotted
        x: sequence to be used in the x axis
        y_Exp: sequence of the expected proportions to be used in the y axis
            (line)
        y_Found: sequence of the found proportions to be used in the y axis
            (bars)
        N: lenght of sequence, to be used when plotting the confidence levels
        figsize: tuple to state the size of the plot figure
        conf_Z: Confidence level
        save_pic: file path to save figure
        text_x: Forces to show all x ticks labels. Defaluts to True.
        save_plot: string with the path/name of the file in which the generated
            plot will be saved. Uses matplotlib.pyplot.savefig(). File format
            is infered by the file name extension.
        save_plot_kwargs: dict with any of the kwargs accepted by
            matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
        
    """
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

    if save_plot:
        if not save_plot_kwargs:
            save_plot_kwargs = {}
        plt.savefig(save_plot, **save_plot_kwargs)

    plt.show(block=False)


def plot_sum(df, figsize, li, text_x=False, save_plot=None, save_plot_kwargs=None):
    """Plots the summation test results

    Args:
        df: DataFrame with the data to be plotted
        figsize: sets the dimensions of the plot figure
        li: value with which to draw the horizontal line
        save_plot: string with the path/name of the file in which the generated
            plot will be saved. Uses matplotlib.pyplot.savefig(). File format
            is infered by the file name extension.
        save_plot_kwargs: dict with any of the kwargs accepted by
            matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
    """
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

    if save_plot:
        if not save_plot_kwargs:
            save_plot_kwargs = {}
        plt.savefig(save_plot, **save_plot_kwargs)

    plt.show(block=False)

def plot_ordered_mantissas(col, figsize=(12, 12),
                           save_plot=None, save_plot_kwargs=None):
    """Plots the ordered mantissas and compares them to the expected, straight
        line that should be formed in a Benford-cmpliant set.

    Args:
        col (Series): column of mantissas to plot.
        figsize (tuple): sets the dimensions of the plot figure.
        save_plot: string with the path/name of the file in which the generated
            plot will be saved. Uses matplotlib.pyplot.savefig(). File format
            is infered by the file name extension.
        save_plot_kwargs: dict with any of the kwargs accepted by
            matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
 
    """
    ld = len(col)
    x = arange(1, ld + 1)
    n = ones(ld) / ld
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(x, col.sort_values(), linestyle='--',
            color=colors['s'], linewidth=3, label='Mantissas')
    ax.plot(x, n.cumsum(), color=colors['m'],
            linewidth=2, label='Expected')
    plt.ylim((0, 1.))
    plt.xlim((1, ld + 1))
    ax.set_facecolor(colors['b'])
    ax.set_title("Ordered Mantissas")
    plt.legend(loc='upper left')

    if save_plot:
        if not save_plot_kwargs:
            save_plot_kwargs = {}
        plt.savefig(save_plot, **save_plot_kwargs)

    plt.show(block=False);

def plot_mantissa_arc_test(df, gravity_center, grid=True, figsize=12,
                           save_plot=None, save_plot_kwargs=None):
    """Draws thee Mantissa Arc Test after computing X and Y circular coordinates
    for every mantissa and the center of gravity for the set

    Args:
        df (DataFrame): pandas DataFrame with the mantissas and the X and Y
            coordinates.
        gravity_center (tuple): coordinates for plottling the gravity center
        grid (bool): show grid. Defaults to True.
        figsize (int): figure dimensions. No need to be a tuple, since the
            figure is a square.
        save_plot: string with the path/name of the file in which the generated
            plot will be saved. Uses matplotlib.pyplot.savefig(). File format
            is infered by the file name extension.
        save_plot_kwargs: dict with any of the kwargs accepted by
            matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
    """
    fig = plt.figure(figsize=(figsize, figsize))
    ax = plt.subplot()
    ax.set_facecolor(colors['b'])
    ax.scatter(df.mant_x, df.mant_y, label="ARC TEST",
               color=colors['m'])
    ax.scatter(gravity_center[0], gravity_center[1],
               color=colors['s'])
    text_annotation = Annotation(
        "  Gravity Center: "
        f"x({round(gravity_center[0], 3)}),"
        f" y({round(gravity_center[1], 3)})",
        xy=(gravity_center[0] - 0.65,
            gravity_center[1] - 0.1),
        xycoords='data')
    ax.add_artist(text_annotation)
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.legend(loc='lower left')
    ax.set_title("Mantissas Arc Test")

    if save_plot:
        if not save_plot_kwargs:
            save_plot_kwargs = {}
        plt.savefig(save_plot, **save_plot_kwargs)

    plt.show(block=False);

def plot_roll_mse(roll_series, figsize, save_plot=None, save_plot_kwargs=None):
    """Shows the rolling MSE plot

    Args:
        roll_series: pd.Series resultant form rolling mse.
        figsize: the figure dimensions.
        save_plot: string with the path/name of the file in which the generated
            plot will be saved. Uses matplotlib.pyplot.savefig(). File format
            is infered by the file name extension.
        save_plot_kwargs: dict with any of the kwargs accepted by
            matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(colors['b'])
    ax.plot(roll_series, color=colors['m'])

    if save_plot:
        if not save_plot_kwargs:
            save_plot_kwargs = {}
        plt.savefig(save_plot, **save_plot_kwargs)

    plt.show(block=False)

def plot_roll_mad(roll_mad, figsize, save_plot=None, save_plot_kwargs=None):
    """Shows the rolling MAD plot

    Args:
        roll_mad: pd.Series resultant form rolling mad.
        figsize: the figure dimensions.
        save_plot: string with the path/name of the file in which the generated
            plot will be saved. Uses matplotlib.pyplot.savefig(). File format
            is infered by the file name extension.
        save_plot_kwargs: dict with any of the kwargs accepted by
            matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(colors['b'])
    ax.plot(roll_mad.roll_series, color=colors['m'])

    if roll_mad.test != -2:
        plt.axhline(y=mad_dict[roll_mad.test][0], color=colors['af'], linewidth=3)
        plt.axhline(y=mad_dict[roll_mad.test][1], color=colors['h2'], linewidth=3)
        plt.axhline(y=mad_dict[roll_mad.test][2], color=colors['s'], linewidth=3)

    if save_plot:
        if not save_plot_kwargs:
            save_plot_kwargs = {}
        plt.savefig(save_plot, **save_plot_kwargs)

    plt.show(block=False)

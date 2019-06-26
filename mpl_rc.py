from matplotlib import rcParams
import numpy as np
# eternally grateful to http://blog.dmcdougall.co.uk/publication-ready-the-first-time-beautiful-reproducible-plots-with-matplotlib/


def default():
    reference = 16
    rcParams['axes.titlesize'] = 1 * reference
    rcParams['axes.labelsize'] = 1 * reference
    rcParams['xtick.labelsize'] = 0.8 * reference
    rcParams['ytick.labelsize'] = 0.8 * reference
    rcParams['legend.fontsize'] = 0.8 * reference

    rcParams['axes.linewidth'] = .5
    rcParams['lines.linewidth'] = .5
    rcParams['patch.linewidth'] = .5

    rcParams['font.family'] = 'sans-serif'
    rcParams['font.serif'] = ['DejaVu Sans']
    rcParams['text.usetex'] = True

    WIDTH = 489.38739  # the number latex spits out
    # FACTOR = 0.49  # the fraction of the width you'd like the figure to occupy
    FACTOR = 1.0
    fig_width_pt = WIDTH * FACTOR

    inches_per_pt = 1.0 / 72.27
    golden_ratio = (np.sqrt(5) - 1.0) / 2.0  # because it looks good

    fig_width_in = fig_width_pt * inches_per_pt  # figure width in inches
    fig_height_in = fig_width_in * golden_ratio  # figure height in inches
    fig_dims = [fig_width_in, fig_height_in]   # fig dims as a list

    rcParams['figure.figsize'] = fig_dims
    return rcParams

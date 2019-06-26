import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


DIVERGING_CMAP = 'RdBu_r'


def plot_input_output(title,
                      x_label,
                      y_label,
                      z_label,
                      x_inv_label,
                      x_true,
                      y_pred,
                      z_pred,
                      x_inv,
                      figsize):
    print('x_true.shape', x_true.shape)
    print('y_pred.shape', y_pred.shape)
    print('z_pred.shape', z_pred.shape)
    print('x_inv.shape', x_inv.shape)

    labelsize = 10
    pad = 0.03
    height_ratios = [2, 10, 10, 2]
    rotation = 0
    length = x_true.shape[0]
    font_size = 8

    phase_start = 0
    phase_end = phase_start + 88

    vel_start = phase_end
    vel_end = vel_start + 88

    inst_start = vel_end
    inst_end = inst_start + 9

    fig = plt.figure(figsize=(figsize[0], figsize[1] * 0.99))
    fig.suptitle('$' + title + '$', x=0.5, y=0.99)
    gs = fig.add_gridspec(nrows=4, ncols=4, wspace=0.3, hspace=0.2, height_ratios=height_ratios)
    ax_x = fig.add_subplot(gs[:, 0])

    ax_y_inst = fig.add_subplot(gs[0, 1], sharex=ax_x)
    ax_y_vel = fig.add_subplot(gs[1, 1], sharex=ax_x)
    ax_y_phase = fig.add_subplot(gs[2, 1], sharex=ax_x)
    ax_z = fig.add_subplot(gs[3, 1], sharex=ax_x)

    ax_x_inv = fig.add_subplot(gs[:, 2], sharex=ax_x)
    ax_diff = fig.add_subplot(gs[:, 3], sharex=ax_x)

    ax_x.set_title('$' + x_label + '$')
    ax_y_inst.set_title('$' + y_label + '$')
    ax_z.set_title('$' + z_label + '$', y=-1.2)

    ax_x_inv.set_title('$' + x_inv_label + '$')
    ax_diff.set_title('$' + x_label + ' - ' + x_inv_label + '$')

    ############################################################################
    min_x = 0
    max_x = np.max(np.abs(x_true))
    mid_x = max_x * 0.5

    ax = ax_x
    im = ax.imshow(x_true.T, origin='lower', cmap='viridis', vmin=min_x, vmax=max_x, aspect='auto')
    ax.yaxis.set_major_locator(plt.NullLocator())

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=pad)
    cbar = fig.colorbar(im, cax=cax, ticks=[min_x, mid_x, max_x], orientation='vertical')
    cbar.ax.set_yticklabels([
        '{:4.1g}'.format(min_x),
        '{:4.1g}'.format(mid_x),
        '{:4.1g}'.format(max_x)
    ], rotation=rotation)
    cbar.ax.tick_params(labelsize=labelsize)

    ############################################################################
    y = y_pred[:, inst_start:inst_end]
    min_y = 0
    max_y = np.max(y)

    ax = ax_y_inst
    im = ax.imshow(y.T, origin='lower', cmap='Purples', vmin=min_y, vmax=max_y, aspect='auto')
    ax.text(length * 0.3, 2, 'instrument', fontdict=dict(size=font_size))

    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=pad)
    cbar = fig.colorbar(im, cax=cax, ticks=[min_y, max_y], orientation='vertical')
    cbar.ax.set_yticklabels([
        '{:4.1g}'.format(min_y),
        '{:4.1g}'.format(max_y)
    ], rotation=rotation)
    cbar.ax.tick_params(labelsize=labelsize)

    ############################################################################
    y = y_pred[:, vel_start:vel_end]
    min_y = 0
    max_y = np.max(y)

    ax = ax_y_vel
    im = ax.imshow(y.T, origin='lower', cmap='Oranges', vmin=min_y, vmax=max_y, aspect='auto')
    ax.text(length * 0.3, 5, 'velocity', fontdict=dict(size=font_size))
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=pad)
    cbar = fig.colorbar(im, cax=cax, ticks=[min_y, max_y], orientation='vertical')
    cbar.ax.set_yticklabels([
        '{:4.1g}'.format(min_y),
        '{:4.1g}'.format(max_y)
    ], rotation=rotation)
    cbar.ax.tick_params(labelsize=labelsize)

    ############################################################################
    y = y_pred[:, phase_start:phase_end]
    min_y = 0
    max_y = np.max(y)

    ax = ax_y_phase
    im = ax.imshow(y.T, origin='lower', cmap='gray_r', vmin=min_y, vmax=max_y, aspect='auto')
    ax.text(length * 0.3, 5, 'note phase', fontdict=dict(size=font_size))
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=pad)
    cbar = fig.colorbar(im, cax=cax, ticks=[min_y, max_y], orientation='vertical')
    cbar.ax.set_yticklabels([
        '{:4.1g}'.format(min_y),
        '{:4.1g}'.format(max_y)
    ], rotation=rotation)
    cbar.ax.tick_params(labelsize=labelsize)

    ############################################################################
    r_z = np.max(np.abs(z_pred))
    min_z = -r_z
    mid_z = 0
    max_z = r_z

    ax = ax_z
    im = ax.imshow(z_pred.T, origin='lower', cmap=DIVERGING_CMAP, vmin=min_z, vmax=max_z, aspect='auto')
    ax.yaxis.set_major_locator(plt.NullLocator())

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=pad)
    cbar = fig.colorbar(im, cax=cax, ticks=[min_z, mid_z, max_z], orientation='vertical')
    cbar.ax.set_yticklabels([
        '{:4.1g}'.format(min_z),
        '{:4.1g}'.format(mid_z),
        '{:4.1g}'.format(max_z)
    ], rotation=rotation)
    cbar.ax.tick_params(labelsize=labelsize)

    ############################################################################
    ax = ax_x_inv
    im = ax.imshow(x_inv.T, origin='lower', cmap='viridis', vmin=min_x, vmax=max_x, aspect='auto')
    ax.yaxis.set_major_locator(plt.NullLocator())

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=pad)
    cbar = fig.colorbar(im, cax=cax, ticks=[min_x, mid_x, max_x], orientation='vertical')
    cbar.ax.set_yticklabels([
        '{:4.1g}'.format(min_x),
        '{:4.1g}'.format(mid_x),
        '{:4.1g}'.format(max_x)
    ], rotation=rotation)
    cbar.ax.tick_params(labelsize=labelsize)

    ############################################################################
    ax = ax_diff
    x_diff = x_true - x_inv
    r = np.max(np.abs(x_diff))
    min_x = -r
    mid_x = 0
    max_x = r
    im = ax.imshow(x_diff.T, origin='lower', cmap=DIVERGING_CMAP, vmin=min_x, vmax=max_x, aspect='auto')
    ax.yaxis.set_major_locator(plt.NullLocator())

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=pad)
    cbar = fig.colorbar(im, cax=cax, ticks=[min_x, mid_x, max_x], orientation='vertical')
    cbar.ax.set_yticklabels([
        '{:4.1g}'.format(min_x),
        '{:4.1g}'.format(mid_x),
        '{:4.1g}'.format(max_x)
    ], rotation=rotation)
    cbar.ax.tick_params(labelsize=labelsize)

    return fig

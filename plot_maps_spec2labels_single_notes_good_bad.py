import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from reversible import ReversibleModel
from audio_midi_dataset import get_dataset_individually, Spec2MidiDataset, SqueezingDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from train_loop import normal_noise_like
import os
import re
import mpl_rc
import utils
import pretty_midi as pm
RCPARAMS = mpl_rc.default()


def collect_samples(device, model, loader, n_samples):
    model.eval()
    samples_x_true = []
    samples_x_pred = []
    for si in range(n_samples):
        x_true = []
        x_pred = []
        for batch in loader:
            x = batch['x']
            y = batch['y']

            y = y + normal_noise_like(y, model.y_noise_scale)  # tiny exaggeration

            z = torch.randn(len(y), model.ndim_z)

            x_true.append(x.numpy())

            y = y.to(device)
            z = z.to(device)

            x_hat, _ = model.decode(z, y)

            x_pred.append(x_hat.detach().cpu().numpy())

        x_true = np.vstack(x_true)
        x_pred = np.vstack(x_pred)

        samples_x_true.append(x_true)
        samples_x_pred.append(x_pred)

    samples_x_true = np.stack(samples_x_true)
    samples_x_pred = np.stack(samples_x_pred)

    print('samples_x_true.shape', samples_x_true.shape)
    print('samples_x_pred.shape', samples_x_pred.shape)

    return samples_x_true, samples_x_pred


def collect_distances(device, model, loader, n_samples):
    samples_x_true, samples_x_pred = collect_samples(device, model, loader, n_samples)

    samples_x_true = samples_x_true - samples_x_true.min()
    samples_x_true = samples_x_true / samples_x_true.max()

    samples_x_pred = samples_x_pred - samples_x_pred.min()
    samples_x_pred = samples_x_pred / samples_x_pred.max()

    samples_x_diff = samples_x_true - samples_x_pred

    return dict(
        samples_x_true=samples_x_true,
        samples_x_pred=samples_x_pred,
        samples_x_diff=samples_x_diff,
        distance=np.median(samples_x_diff ** 2)
    )


def desugar(full_path):
    # first group: F.orte, M.ezzo-Forte, P.iano
    # S: unknown meaning, ignored; |S0| + |S1| = 88
    # M: midi-note number
    match = re.match('.*(F|M|P)_S\d\_M(\d{2,3})\_.*', full_path)
    if match is None:
        raise RuntimeError('full_path {}'.format(full_path))

    volume = match.groups()[0]
    midi_note = int(match.groups()[1])
    return volume, midi_note


def escape_latex(s):
    return s.replace('#', '\#').replace('_', '\_')


def nona(midinote):
    return escape_latex(pm.note_number_to_name(midinote))


def plot_fold(direction,
              base_directory,
              instrument_filename,
              context,
              audio_options,
              batch_size,
              device,
              model,
              fold_file,
              n_samples,
              plot_output_directory):

    loaders = get_data_loaders(
        direction=direction,
        base_directory=base_directory,
        fold_file=fold_file,
        instrument_filename=instrument_filename,
        context=context,
        audio_options=audio_options,
        batch_size=batch_size
    )

    note_distances = dict()
    for fold_file, audiofilename, midifilename, loader in loaders:
        print('fold_file, audiofilename', fold_file, audiofilename)
        volume, midinote = desugar(audiofilename)
        print('volume, midinote', volume, midinote)
        note_distances[midinote] = collect_distances(device, model, loader, n_samples)
        note_distances[midinote]['midinote'] = midinote

    # best_d = np.inf
    # best_note = None
    # worst_d = -np.inf
    # worst_note = None
    # for midinote in note_distances.keys():
    #     d = note_distances[midinote]['distance']
    #     if d <= best_d:
    #         best_d = d
    #         best_note = note_distances[midinote]

    #     if d >= worst_d:
    #         worst_d = d
    #         worst_note = note_distances[midinote]

    best_note = note_distances[69]
    worst_note = note_distances[105]

    figsize = RCPARAMS['figure.figsize']
    fig = plt.figure(figsize=(figsize[0], figsize[1] * 0.5))
    gs_left = plt.GridSpec(1, 6, left=0.03, wspace=0.1)
    gs_right = plt.GridSpec(1, 6, right=0.94, wspace=0.1)

    axes = []
    for i in range(3):
        axes.append(fig.add_subplot(gs_left[0, i]))

    for i in range(3, 6):
        axes.append(fig.add_subplot(gs_right[0, i]))

    best_r = np.max(np.abs(best_note['samples_x_diff']))
    worst_r = np.max(np.abs(worst_note['samples_x_diff']))
    r = max(best_r, worst_r)

    axes[0].imshow(best_note['samples_x_true'][0].T, origin='lower', vmin=0, aspect='auto')
    axes[1].imshow(best_note['samples_x_pred'][0].T, origin='lower', vmin=0, aspect='auto')
    im2 = axes[2].imshow(best_note['samples_x_diff'][0].T, origin='lower',
                         cmap='RdBu_r', vmin=-best_r, vmax=best_r, aspect='auto')

    axes[3].imshow(worst_note['samples_x_true'][0].T, origin='lower', vmin=0, aspect='auto')
    axes[4].imshow(worst_note['samples_x_pred'][0].T, origin='lower', vmin=0, aspect='auto')
    im5 = axes[5].imshow(worst_note['samples_x_diff'][0].T, origin='lower',
                         cmap='RdBu_r', vmin=-worst_r, vmax=worst_r, aspect='auto')

    axes[1].set_title('understood, {}'.format(nona(best_note['midinote'])))
    axes[4].set_title('not understood, {}'.format(nona(worst_note['midinote'])))

    for im, ax, r in [(im2, axes[2], best_r), (im5, axes[5], worst_r)]:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.03)
        cbar = fig.colorbar(im, cax=cax, ticks=[-r, r], orientation='vertical')
        cbar.ax.set_yticklabels([
            '{:4.1g}'.format(-r),
            '{:4.1g}'.format(r)
        ], rotation=0)
        cbar.ax.tick_params(labelsize=10)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(os.path.join(plot_output_directory, 'good_bad.pdf'))


def get_data_loaders(direction,
                     base_directory,
                     fold_file,
                     instrument_filename,
                     context,
                     audio_options,
                     batch_size):

    print('-' * 30)
    print('getting data loaders:')
    print('direction', direction)
    print('base_directory', base_directory)
    print('fold_file', fold_file)
    print('instrument_filename', instrument_filename)

    clazz = Spec2MidiDataset

    datasets = get_dataset_individually(
        base_directory,
        fold_file,
        instrument_filename,
        context,
        audio_options,
        clazz
    )
    loaders = []
    for dataset in datasets:
        audiofilename = dataset.audiofilename
        midifilename = dataset.midifilename
        dataset = SqueezingDataset(dataset)
        print('len(dataset)', len(dataset))

        sampler = SequentialSampler(dataset)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True
        )
        loaders.append((fold_file, audiofilename, midifilename, loader))

    return loaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('plot_output_directory')
    parser.add_argument('--n_samples', type=int, default=30)
    args = parser.parse_args()
    batch_size = 8
    direction = 'spec2labels'
    print('direction', direction)
    utils.ensure_directory_exists(args.plot_output_directory)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    audio_options = dict(
        spectrogram_type='LogarithmicFilteredSpectrogram',
        filterbank='LogarithmicFilterbank',
        num_channels=1,
        sample_rate=44100,
        frame_size=4096,
        fft_size=4096,
        hop_size=441 * 4,  # 25 fps
        num_bands=24,
        fmin=30,
        fmax=10000.0,
        fref=440.0,
        norm_filters=True,
        unique_filters=True,
        circular_shift=False,
        add=1.
    )
    context = dict(
        frame_size=1,
        hop_size=1,
        origin='center'
    )
    base_directory = './data/maps_piano/data'

    print('loading checkpoint')
    checkpoint = torch.load(args.checkpoint)
    model = ReversibleModel(
        device=device,
        batch_size=batch_size,
        depth=5,
        ndim_tot=256,
        ndim_x=144,
        ndim_y=185,
        ndim_z=9,
        clamp=2,
        zeros_noise_scale=3e-2,  # very magic, much hack!
        y_noise_scale=3e-2
    )
    # print('model', model)
    model.to(device)
    model.load_state_dict(checkpoint)

    # instrument_filename = './splits/tiny-min/instruments'
    # fold_files = ['./splits/tiny-min/AkPnBcht_F']
    instrument_filename = './splits/maps-isolated-notes/instruments'
    # fold_files = ['./splits/maps-isolated-notes/AkPnBcht_F']
    fold_base = './splits/maps-isolated-notes'
    fold_filenames = [
        # 'AkPnBcht_F',
        # 'AkPnBsdf_F',
        # 'AkPnCGdD_F',
        # 'AkPnStgb_F',
        # 'SptkBGAm_F',
        # 'SptkBGCl_F',
        # 'StbgTGd2_F',

        # 'AkPnBcht_M',
        # 'AkPnBsdf_M',
        # 'AkPnCGdD_M',
        # 'AkPnStgb_M',
        # 'SptkBGAm_M',
        # 'SptkBGCl_M',
        # 'StbgTGd2_M',

        # 'AkPnBcht_P',
        # 'AkPnBsdf_P',
        # 'AkPnCGdD_P',
        # 'AkPnStgb_P',
        # 'SptkBGAm_P',
        # 'SptkBGCl_P',
        # 'StbgTGd2_P',

        'ENSTDkCl_F',
        # 'ENSTDkAm_F',
        # 'ENSTDkCl_M',
        # 'ENSTDkAm_M',
        # 'ENSTDkCl_P'
        # 'ENSTDkAm_P',
    ]
    fold_files = []
    for fold_filename in fold_filenames:
        fold_files.append(os.path.join(fold_base, fold_filename))

    for fold_file in fold_files:
        plot_fold(
            direction=direction,
            base_directory=base_directory,
            instrument_filename=instrument_filename,
            context=context,
            audio_options=audio_options,
            batch_size=batch_size,
            device=device,
            model=model,
            fold_file=fold_file,
            n_samples=args.n_samples,
            plot_output_directory=args.plot_output_directory
        )


if __name__ == '__main__':
    main()

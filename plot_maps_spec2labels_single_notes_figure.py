import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np
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

    x_diff = samples_x_true - samples_x_pred

    # euclidean = np.sqrt(np.sum(x_diff ** 2, axis=(1, 2)))
    euclidean = np.sqrt(np.sum(x_diff ** 2, axis=1)).flatten()
    print('euclidean.shape', euclidean.shape)
    return dict(
        euclidean=euclidean
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


def plot_dist(ax, samples, color):
    n_notes, n_samples = samples.shape
    xs = np.arange(n_notes)
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    quantiles = np.nanquantile(samples, qs, axis=1)

    for q in range(len(qs) - 1 // 2):
        ilo = q
        ihi = len(qs) - q - 1

        lo = quantiles[ilo, :]
        hi = quantiles[ihi, :]

        ax.fill_between(xs, lo, hi, facecolor=color, alpha=0.05 * q)
    med = quantiles[2, :]
    ax.plot(xs, med, c=color, linewidth=1)


# def plot_dist(ax, samples, color):
#     ax.boxplot(samples.T, whis=[5, 95], widths=0.2)

# def plot_dist(ax, samples, color):
#     ax.violinplot(samples.T, whis=[5, 95], widths=0.2)

def escape_latex(s):
    return s.replace('#', '\#').replace('_', '\_')


def nona(midinote, tick_number):
    return escape_latex(pm.note_number_to_name(midinote + 21))


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

    maxlen = 0
    unmasked_sorted_euclidean = []
    for imn, midinote in enumerate(sorted(note_distances.keys())):
        euclidean = note_distances[midinote]['euclidean']
        maxlen = max(maxlen, len(euclidean))
        unmasked_sorted_euclidean.append(euclidean)

    print('maxlen', maxlen)
    padded_sorted_euclidean = []
    for euclidean in unmasked_sorted_euclidean:
        pad_length = maxlen - len(euclidean)
        # print('pad_length', pad_length)
        if pad_length > 0:
            padded = np.pad(euclidean, (0, pad_length), mode='constant', constant_values=np.nan)
            # print('np.isnan(padded).any()', np.isnan(padded).any())
            padded_sorted_euclidean.append(padded)
        else:
            padded_sorted_euclidean.append(euclidean)

    padded_sorted_euclidean = np.vstack(padded_sorted_euclidean)
    sorted_euclidean = padded_sorted_euclidean  # use np.nanquantile
    # sorted_euclidean = np.ma.masked_invalid(padded_sorted_euclidean)
    # print('np.isnan(sorted_euclidean).any()', np.isnan(sorted_euclidean).any())
    # print('np.min(sorted_euclidean)', np.min(sorted_euclidean))
    # print('np.mean(sorted_euclidean)', np.mean(sorted_euclidean))
    # print('np.max(sorted_euclidean)', np.max(sorted_euclidean))

    ymax = 5
    figsize = RCPARAMS['figure.figsize']
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] * 0.5))
    plot_dist(ax, sorted_euclidean, color='gray')

    ax.xaxis.set_minor_locator(plt.FixedLocator(np.arange(0, 88, 1)))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    ax.xaxis.set_major_locator(plt.FixedLocator(np.arange(0, 88, 12)))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(nona))

    ax.yaxis.set_major_locator(plt.FixedLocator(np.arange(0, ymax, 1)))

    fold = os.path.basename(fold_file)
    ax.set_title(escape_latex(fold))
    ax.set_ylabel('Euclidean\nDistance')
    ax.set_xlabel('MIDI Notenumber')
    ax.set_ylim([0, ymax])

    fig_filename = os.path.join(plot_output_directory, 'notes_{}.pdf'.format(fold))
    fig.tight_layout()
    # plt.show()
    plt.savefig(fig_filename)


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
        'AkPnBcht_F',
        'AkPnBsdf_F',
        'AkPnCGdD_F',
        'AkPnStgb_F',
        'SptkBGAm_F',
        'SptkBGCl_F',
        'StbgTGd2_F',

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
        'ENSTDkAm_F',
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

import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np

from plot_input_output import plot_input_output
from reversible import ReversibleModel
from audio_midi_dataset import get_dataset_individually, Spec2MidiDataset, SqueezingDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from train_loop import normal_noise_like
import os
import mpl_rc
import utils
rcParams = mpl_rc.default()


def collect_input_output(device, model, loader, n_samples):
    model.eval()
    samples_x_true = []
    samples_x_invs = []
    samples_x_zepa = []
    samples_x_samp = []

    samples_y_true = []
    samples_y_pred = []

    samples_z_pred = []
    samples_z_samp = []
    for si in range(n_samples):
        x_true = []
        x_invs = []
        x_samp = []
        x_zepa = []

        y_pred = []
        y_true = []

        z_pred = []
        z_samp = []
        fixed_synth = torch.zeros(9)
        fixed_synth[0] = 1
        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            inst_start = 88 + 88

            # what happens if we fix a synth?
            # y[:, inst_start:] = fixed_synth

            z_hat, zy_padding, y_hat = model.encode(x)
            # print('zy_padding.mean()', zy_padding.mean().cpu().item())
            # x_inv, _ = model.decode_padding(z_hat, zy_padding, y_hat)
            z = normal_noise_like(z_hat, 1)

            # decode full bijectivity (*all* information)
            x_inv, _ = model.decode_padding(z_hat, zy_padding, y_hat)

            # decode with z_hat, zeros, y_hat
            x_zep, _ = model.decode(z_hat, y_hat)

            # decode with sampled z, zeros, perfect y ('use it like a conditional GAN')
            x_sam, _ = model.decode(z, y)

            x_true.append(x.detach().cpu().numpy())
            x_invs.append(x_inv.detach().cpu().numpy())
            x_samp.append(x_sam.detach().cpu().numpy())
            x_zepa.append(x_zep.detach().cpu().numpy())

            y_pred.append(y_hat.detach().cpu().numpy())
            y_true.append(y.detach().cpu().numpy())

            z_pred.append(z_hat.detach().cpu().numpy())
            z_samp.append(z.detach().cpu().numpy())

        x_true = np.vstack(x_true)
        x_invs = np.vstack(x_invs)
        x_zepa = np.vstack(x_zepa)
        x_samp = np.vstack(x_samp)

        y_pred = np.vstack(y_pred)
        y_true = np.vstack(y_true)

        z_pred = np.vstack(z_pred)
        z_samp = np.vstack(z_samp)

        samples_x_true.append(x_true)
        samples_x_invs.append(x_invs)
        samples_x_zepa.append(x_zepa)
        samples_x_samp.append(x_samp)

        samples_y_pred.append(y_pred)
        samples_y_true.append(y_true)

        samples_z_pred.append(z_pred)
        samples_z_samp.append(z_samp)

    samples_x_true = np.stack(samples_x_true)
    samples_x_invs = np.stack(samples_x_invs)
    samples_x_zepa = np.stack(samples_x_zepa)
    samples_x_samp = np.stack(samples_x_samp)

    samples_y_pred = np.stack(samples_y_pred)
    samples_y_true = np.stack(samples_y_true)

    samples_z_pred = np.stack(samples_z_pred)
    samples_z_samp = np.stack(samples_z_samp)

    print('samples_x_true.shape', samples_x_true.shape)
    print('samples_x_invs.shape', samples_x_invs.shape)
    print('samples_x_zepa.shape', samples_x_zepa.shape)
    print('samples_x_samp.shape', samples_x_samp.shape)

    print('samples_y_pred.shape', samples_y_pred.shape)
    print('samples_y_true.shape', samples_y_true.shape)

    print('samples_z_pred.shape', samples_z_pred.shape)
    print('samples_z_samp.shape', samples_z_samp.shape)

    return dict(
        samples_x_true=samples_x_true,
        samples_x_invs=samples_x_invs,
        samples_x_zepa=samples_x_zepa,
        samples_x_samp=samples_x_samp,

        samples_y_pred=samples_y_pred,
        samples_y_true=samples_y_true,

        samples_z_pred=samples_z_pred,
        samples_z_samp=samples_z_samp
    )


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

    for fold_file, audiofilename, midifilename, loader in loaders:
        print('fold_file', fold_file)
        print('audiofilename', audiofilename)
        print('midifilename', midifilename)
        sio = collect_input_output(device, model, loader, n_samples)

        fold = os.path.basename(fold_file)
        start = 100
        end = 150

        adjustments = dict(
            left=0.02,
            right=0.915,
            bottom=0.07,
            wspace=0.38,
            hspace=0.18
        )

        ##########################################################################
        fig = plot_input_output(
            '\hat{\mathbf{x}} = f_{\\theta}^{-1}([\hat{\mathbf{z}}; \hat{\mathbf{yz}}_{pad}; \hat{\mathbf{y}}])',
            '\mathbf{x}',
            '\hat{\mathbf{y}}',
            '\hat{\mathbf{z}}',
            '\hat{\mathbf{x}}',
            sio['samples_x_true'][0, start:end, :],
            sio['samples_y_pred'][0, start:end, :],
            sio['samples_z_pred'][0, start:end, :],
            sio['samples_x_invs'][0, start:end, :],
            rcParams['figure.figsize']
        )

        fig_filename = os.path.join(
            plot_output_directory,
            'z_hat_pad_y_hat_input_output_{}.pdf'.format(fold)
        )
        fig.subplots_adjust(**adjustments)
        fig.savefig(fig_filename)
        plt.close(fig)

        ##########################################################################
        fig = plot_input_output(
            '\hat{\mathbf{x}}^{(0)} = f_{\\theta}^{-1}([\hat{\mathbf{z}}; \mathbf{0}; \hat{\mathbf{y}}])',
            '\mathbf{x}',
            '\hat{\mathbf{y}}',
            '\hat{\mathbf{z}}',
            '\hat{\mathbf{x}}^{(0)}',
            sio['samples_x_true'][0, start:end, :],
            sio['samples_y_pred'][0, start:end, :],
            sio['samples_z_pred'][0, start:end, :],
            sio['samples_x_zepa'][0, start:end, :],
            rcParams['figure.figsize']
        )

        fig_filename = os.path.join(
            plot_output_directory,
            'z_hat_zero_y_hat_input_output_{}.pdf'.format(fold)
        )
        fig.subplots_adjust(**adjustments)
        fig.savefig(fig_filename)
        plt.close()

        ##########################################################################
        fig = plot_input_output(
            '\mathbf{x}_{sam} = f_{\\theta}^{-1}([\mathbf{z}; \mathbf{0}; \mathbf{y}])',
            '\mathbf{x}',
            '\hat{\mathbf{y}}',
            '\hat{\mathbf{z}}',
            '\mathbf{x}_{sam}',
            sio['samples_x_true'][0, start:end, :],
            sio['samples_y_true'][0, start:end, :],
            sio['samples_z_samp'][0, start:end, :],
            sio['samples_x_samp'][0, start:end, :],
            rcParams['figure.figsize']
        )

        fig_filename = os.path.join(
            plot_output_directory,
            'z_samp_zero_y_true_input_output_{}.pdf'.format(fold)
        )
        fig.subplots_adjust(**adjustments)
        fig.savefig(fig_filename)
        plt.close()


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
    parser.add_argument('--n_samples', type=int, default=1)
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

    instrument_filename = './splits/maps-individual-tracks/instruments'
    fold_base = './splits/maps-individual-tracks'
    fold_filenames = [
        'train/MAPS_MUS-chpn-p10_AkPnStgb',
        'test/MAPS_MUS-chpn-p14_ENSTDkAm',
        'test/MAPS_MUS-chpn-p19_ENSTDkCl'
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

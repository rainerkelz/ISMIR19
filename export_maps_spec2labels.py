import torch
import argparse
import numpy as np
from reversible import ReversibleModel
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from audio_midi_dataset import get_dataset_individually, Spec2MidiDataset, SqueezingDataset
import os
import utils


def export(device, model, loader):
    model.eval()
    x_pred = []
    y_pred = []
    z_pred = []

    x_true = []
    y_true = []
    z_true = []
    for batch in loader:
        x, y = batch['x'], batch['y']
        z = torch.randn(len(y), model.ndim_z)

        x_true.append(x.numpy())
        y_true.append(y.numpy())
        z_true.append(z.numpy())

        x = x.to(device)
        y = y.to(device)
        z = z.to(device)

        z_hat, _, y_hat = model.encode(x)
        x_hat, _ = model.decode(z, y)

        x_pred.append(x_hat.detach().cpu().numpy())
        y_pred.append(y_hat.detach().cpu().numpy())
        z_pred.append(z_hat.detach().cpu().numpy())

    x_pred = np.vstack(x_pred)
    y_pred = np.vstack(y_pred)
    z_pred = np.vstack(z_pred)

    x_true = np.vstack(x_true)
    y_true = np.vstack(y_true)
    z_true = np.vstack(z_true)

    return dict(
        x_pred=x_pred,
        y_pred=y_pred,
        z_pred=z_pred,

        x_true=x_true,
        y_true=y_true,
        z_true=z_true
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('output_directory')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    direction = 'spec2labels'
    print('direction', direction)

    n_epochs = 512
    meta_epoch = 12
    batch_size = 32
    gamma = 0.96

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
    model.to(device)

    print('loading checkpoint')
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)

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

    print('loading data')
    base_directory = './data/maps_piano/data'
    fold_directory = './splits/maps-non-overlapping'

    utils.ensure_directory_exists(args.output_directory)

    for fold in ['train', 'valid', 'test']:
        fold_output_directory = os.path.join(args.output_directory, fold)
        if not os.path.exists(fold_output_directory):
            os.makedirs(fold_output_directory)

        print('fold', fold)
        print('fold_output_directory', fold_output_directory)

        sequences = get_dataset_individually(
            base_directory=base_directory,
            fold_filename=os.path.join(fold_directory, fold),
            instrument_filename=os.path.join(fold_directory, 'instruments'),
            context=context,
            audio_options=audio_options,
            clazz=Spec2MidiDataset
        )

        for sequence in sequences:
            print('sequence.audiofilename', sequence.audiofilename)
            print('sequence.midifilename', sequence.midifilename)
            output_filename = os.path.basename(sequence.audiofilename)
            output_filename = os.path.splitext(output_filename)[0]
            output_filename = os.path.join(fold_output_directory, output_filename + '.pkl')

            print('output_filename', output_filename)

            loader = DataLoader(
                SqueezingDataset(sequence),
                batch_size=batch_size,
                sampler=SequentialSampler(sequence),
                drop_last=True
            )

            result = export(device, model, loader)
            result['audiofilename'] = sequence.audiofilename
            result['midifilename'] = sequence.midifilename
            torch.save(result, output_filename)


if __name__ == '__main__':
    main()

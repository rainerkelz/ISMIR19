from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from chunked_sampler import ChunkedRandomSampler
from madmom.audio.signal import FramedSignal
import madmom.audio.spectrogram as mmspec
from madmom.io import midi
from copy import deepcopy
import numpy as np
import joblib
import torch
import utils
import os


memory = joblib.memory.Memory('./joblib_cache', mmap_mode='r', verbose=1)


def curve(start, end):
    duration = end - start
    xs = np.arange(0, duration)
    return (0.99 ** xs) * 5


def get_y_from_file(midifile, n_frames, audio_options):
    pattern = midi.MIDIFile(midifile)
    dt = float(audio_options['hop_size']) / float(audio_options['sample_rate'])

    y_frames = np.zeros((n_frames, 88)).astype(np.float32)
    y_velocity = np.zeros((n_frames, 88)).astype(np.float32)
    for onset, _pitch, duration, velocity, _channel in pattern.sustained_notes:
        pitch = int(_pitch)
        label = pitch - 21

        note_start = int(np.round(onset / dt))
        note_end = int(np.round((onset + duration) / dt))

        y_frames[note_start:note_end + 1, label] = curve(note_start, note_end + 1)
        y_velocity[note_start:note_end + 1, label] = velocity / 127.

    return y_frames, y_velocity


@memory.cache
def get_xy_from_file(audiofilename, midifilename, _audio_options):
    spec_type, audio_options = utils.canonicalize_audio_options(_audio_options, mmspec)
    x = np.array(spec_type(audiofilename, **audio_options))
    y_frames, y_velocity = get_y_from_file(midifilename, len(x), audio_options)

    return x, y_frames, y_velocity


class SequenceContextDataset(Dataset):
    def __init__(self,
                 audiofilename,
                 midifilename,
                 instrument,
                 instruments,
                 context,
                 audio_options):
        super().__init__()
        self.audiofilename = audiofilename
        self.midifilename = midifilename
        self.instrument = instrument
        self.instruments = instruments
        self.instrument_number_onehot = torch.zeros(
            1, context['frame_size'], len(self.instruments)
        )
        self.instrument_number_onehot[0, :, self.instruments[self.instrument]] = 1.
        self.audio_options = deepcopy(audio_options)

        spectrogram, y_frames, y_velocity = get_xy_from_file(
            self.audiofilename,
            self.midifilename,
            self.audio_options
        )

        self.spectrogram = FramedSignal(
            spectrogram,
            frame_size=context['frame_size'],
            hop_size=context['hop_size'],
            origin=context['origin'],
        )
        self.y_frames = FramedSignal(
            y_frames,
            frame_size=context['frame_size'],
            hop_size=context['hop_size'],
            origin=context['origin'],
        )
        self.y_velocity = FramedSignal(
            y_velocity,
            frame_size=context['frame_size'],
            hop_size=context['hop_size'],
            origin=context['origin'],
        )

        self.fixed_noise = FramedSignal(
            # the noise should be strictly positive ...
            np.abs(np.random.normal(0, 1, (len(spectrogram), 7))),
            frame_size=context['frame_size'],
            hop_size=context['hop_size'],
            origin=context['origin'],
        )

        if (len(self.spectrogram) != len(self.y_frames) or
           len(self.spectrogram) != len(self.y_velocity)):
            raise RuntimeError('x and y do not have the same length.')

    def __len__(self):
        return len(self.spectrogram)

    def __getitem__(self, index):
        _, w, h = self.spectrogram.shape

        _spectrogram = torch.FloatTensor(self.spectrogram[index].reshape(1, w, h))
        _y_frames = torch.FloatTensor(self.y_frames[index].reshape(1, 1, 88))
        _y_velocity = torch.FloatTensor(self.y_velocity[index].reshape(1, 1, 88))
        _fixed_noise = torch.FloatTensor(self.fixed_noise[index].reshape(1, 1, 7))
        return dict(
            spectrogram=_spectrogram,
            y_frames=_y_frames,
            y_velocity=_y_velocity,
            instrument=self.instrument_number_onehot,
            fixed_noise=_fixed_noise
        )


class Midi2SpecDataset(SequenceContextDataset):
    def __init__(self,
                 audiofilename,
                 midifilename,
                 instrument,
                 instruments,
                 context,
                 audio_options):

        super().__init__(
            audiofilename,
            midifilename,
            instrument,
            instruments,
            context,
            audio_options
        )

    def __getitem__(self, index):
        item = super().__getitem__(index)
        return dict(
            x=torch.cat([
                item['y_frames'],
                item['y_velocity'],
                item['instrument']
            ], dim=-1),
            y=item['spectrogram']
        )


class Spec2MidiDataset(SequenceContextDataset):
    def __init__(self,
                 audiofilename,
                 midifilename,
                 instrument,
                 instruments,
                 context,
                 audio_options):

        super().__init__(
            audiofilename,
            midifilename,
            instrument,
            instruments,
            context,
            audio_options
        )

    def __getitem__(self, index):
        item = super().__getitem__(index)
        return dict(
            x=item['spectrogram'],
            y=torch.cat([
                item['y_frames'],
                item['y_velocity'],
                item['instrument']
            ], dim=-1)
        )


class MidiNoise2SpecDataset(SequenceContextDataset):
    def __init__(self,
                 audiofilename,
                 midifilename,
                 instrument,
                 instruments,
                 context,
                 audio_options):

        super().__init__(
            audiofilename,
            midifilename,
            instrument,
            instruments,
            context,
            audio_options
        )

    def __getitem__(self, index):
        item = super().__getitem__(index)
        return dict(
            x=torch.cat([
                item['y_frames'],
                item['y_velocity'],
                item['instrument'],
                item['fixed_noise']
            ], dim=-1),
            y=item['spectrogram']
        )


class SqueezingDataset(Dataset):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __getitem__(self, index):
        _item = self.wrapped[index]
        item = dict()
        for key, w in _item.items():
            item[key] = w.squeeze()
        return item

    def __len__(self):
        return len(self.wrapped)


def get_dataset(base_directory, fold_file, instrument_filename, context, audio_options, clazz):
    sequences = get_dataset_individually(
        base_directory,
        fold_file,
        instrument_filename,
        context,
        audio_options,
        clazz
    )
    return ConcatDataset(sequences)


def get_dataset_individually(base_directory,
                             fold_filename,
                             instrument_filename,
                             context,
                             audio_options,
                             clazz):

    instruments = dict()
    with open(instrument_filename, 'r') as instrument_file:
        rows = instrument_file.readlines()
        for row in rows:
            stripped_row = row.strip()
            raw_instrument_name, raw_instrument_number = stripped_row.split(',')
            instrument_name = raw_instrument_name.strip()
            instrument_number = int(raw_instrument_number.strip())
            instruments[instrument_name] = instrument_number

    parsed = []
    with open(fold_filename, 'r') as fold_file:
        rows = fold_file.readlines()
        for row in rows:
            stripped_row = row.strip()
            raw_audiofilename, raw_midifilename, raw_instrument = stripped_row.split(',')
            audiofilename = os.path.join(base_directory, raw_audiofilename.strip())
            midifilename = os.path.join(base_directory, raw_midifilename.strip())
            instrument = raw_instrument.strip()
            parsed.append((audiofilename, midifilename, instrument))

    sequences = []
    for audiofilename, midifilename, instrument in parsed:
        sequences.append(clazz(
            audiofilename, midifilename, instrument, instruments, context, audio_options
        ))
    return sequences


def get_data_loader(direction,
                    base_directory,
                    fold_file,
                    instrument_filename,
                    context,
                    audio_options,
                    batch_size,
                    sampler_classname,
                    chunk_size=None,
                    squeeze=False):

    print('-' * 30)
    print('getting data loader:')
    print('direction', direction)
    print('base_directory', base_directory)
    print('fold_file', fold_file)
    print('instrument_filename', instrument_filename)

    clazz = None
    if direction == 'spec2labels':
        clazz = Spec2MidiDataset
    elif direction == 'labels2spec':
        clazz = Midi2SpecDataset
    elif direction == 'labels_noise2spec':
        clazz = MidiNoise2SpecDataset
    else:
        raise ValueError('unknown direction')

    dataset = get_dataset(
        base_directory,
        fold_file,
        instrument_filename,
        context,
        audio_options,
        clazz
    )
    if squeeze:
        dataset = SqueezingDataset(dataset)
    print('len(dataset)', len(dataset))

    sampler = None
    if sampler_classname == 'RandomSampler':
        sampler = RandomSampler(dataset)
    elif sampler_classname == 'SequentialSampler':
        sampler = SequentialSampler(dataset)
    elif sampler_classname == 'ChunkedRandomSampler' and chunk_size is not None:
        sampler = ChunkedRandomSampler(dataset, chunk_size)
    else:
        raise ValueError('unknown sampler / incomplete parameters for sampler')

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True
    )

    return loader


def get_data_loaders(direction,
                     base_directory,
                     fold_file,
                     instrument_filename,
                     context,
                     audio_options,
                     batch_size,
                     sampler_classname,
                     chunk_size=None,
                     squeeze=False):

    print('-' * 30)
    print('getting data loaders:')
    print('direction', direction)
    print('base_directory', base_directory)
    print('fold_file', fold_file)
    print('instrument_filename', instrument_filename)

    clazz = None
    if direction == 'spec2labels':
        clazz = Spec2MidiDataset
    elif direction == 'labels2spec':
        clazz = Midi2SpecDataset
    else:
        raise ValueError('unknown direction')

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
        if squeeze:
            dataset = SqueezingDataset(dataset)
        print('len(dataset)', len(dataset))

        sampler = None
        if sampler_classname == 'RandomSampler':
            sampler = RandomSampler(dataset)
        elif sampler_classname == 'SequentialSampler':
            sampler = SequentialSampler(dataset)
        elif sampler_classname == 'ChunkedRandomSampler' and chunk_size is not None:
            sampler = ChunkedRandomSampler(dataset, chunk_size)
        else:
            raise ValueError('unknown sampler / incomplete parameters for sampler')

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True
        )
        loaders.append(loader)

    return loaders

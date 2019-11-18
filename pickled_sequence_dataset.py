from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from chunked_sampler import ChunkedRandomSampler
import numpy as np
import torch
import os


class PickledSequenceDataset(Dataset):
    def __init__(self, picklefilename):
        super().__init__()
        original_data = torch.load(picklefilename)
        # 88  frames
        # 88  velocity
        # 9   instrument
        # 185 total
        self.audiofilename = original_data['audiofilename']
        self.midifilename = original_data['midifilename']

        self.y_true_frames = original_data['y_true'][:, 0:88]
        self.y_true_velocity = original_data['y_true'][:, 88:176]

        self.y_pred_frames = original_data['y_pred'][:, 0:88]
        self.y_pred_velocity = original_data['y_pred'][:, 88:176]

        # print('self.y_true_frames.shape', self.y_true_frames.shape)
        # print('self.y_true_velocity.shape', self.y_true_velocity.shape)

        # print('self.y_pred_frames.shape', self.y_pred_frames.shape)
        # print('self.y_pred_velocity.shape', self.y_pred_velocity.shape)

    def __getitem__(self, index):
        x = torch.FloatTensor(
            np.stack(
                [
                    self.y_pred_frames[:, index],
                    self.y_pred_velocity[:, index],

                ],
                axis=-1
            )
        )
        y = torch.FloatTensor(
            np.stack(
                [
                    self.y_true_frames[:, index],
                    self.y_true_velocity[:, index],
                    (self.y_true_velocity[:, index] > 0) * 1
                ],
                axis=-1
            )
        )
        return dict(x=x, y=y)

    def __len__(self):
        return 88


def get_dataset(fold_directory):
    sequences = get_dataset_individually(fold_directory)
    return ConcatDataset(sequences)


def get_dataset_individually(fold_directory):
    sequences = []
    for base, directories, filenames in os.walk(fold_directory):
        for filename in filenames:
            if filename.endswith('.pkl'):
                sequences.append(PickledSequenceDataset(os.path.join(base, filename)))

    return sequences


def get_data_loader(fold_directory, sampler_classname, chunk_size=None):
    dataset = get_dataset(fold_directory)

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
        batch_size=1,
        sampler=sampler,
        drop_last=False
    )

    return loader

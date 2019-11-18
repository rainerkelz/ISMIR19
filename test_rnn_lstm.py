from train_rnn_gru import evaluate
import torch
import argparse
from torch import nn
from train_rnn_lstm import N_HIDDEN
from pickled_sequence_dataset import get_dataset_individually
import os
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.dataloader import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('base_directory')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    recurrent_model = nn.LSTM(
        input_size=2,
        hidden_size=N_HIDDEN,
        num_layers=1,
        bias=True,
        batch_first=True,
        dropout=0.0,
        bidirectional=False
    )
    recurrent_model.to(device)

    output_model = nn.Sequential(
        nn.ReLU(),
        nn.Linear(N_HIDDEN, 3, bias=True),
    )
    output_model.to(device)

    model_state = torch.load(args.checkpoint)
    recurrent_model.load_state_dict(model_state['recurrent_model'])
    output_model.load_state_dict(model_state['output_model'])

    base_directory = args.base_directory

    test_sequences = get_dataset_individually(os.path.join(base_directory, 'test'))
    test_loaders = []
    for sequence in test_sequences:
        loader = DataLoader(
            sequence,
            batch_size=1,
            sampler=SequentialSampler(sequence),
            drop_last=False
        )
        test_loaders.append((sequence.midifilename, loader))

    print('len(test_loaders)', len(test_loaders))
    logged = evaluate(
        None,
        'test',
        device,
        recurrent_model,
        output_model,
        test_loaders,
        1
    )
    print('logged', logged)


if __name__ == '__main__':
    main()

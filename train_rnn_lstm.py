import torch
from torch import nn
from pickled_sequence_dataset import get_data_loader, get_dataset_individually
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.dataloader import DataLoader

from tensorboardX import SummaryWriter
import numpy as np
from collections import defaultdict
import os
import argparse
from evaluation_utils import evaluate


DEBUG = False
N_HIDDEN = 4


def train(logger, device, recurrent_model, output_model, optimizer, loader, global_step):
    recurrent_model.train()
    output_model.train()
    loss_function_bce = nn.BCEWithLogitsLoss(reduction='mean')
    loss_function_mse = nn.MSELoss(reduction='mean')
    losses = defaultdict(list)
    for batch in loader:
        h_0 = torch.zeros(1, 1, N_HIDDEN).to(device)
        c_0 = torch.zeros(1, 1, N_HIDDEN).to(device)
        x = batch['x'].to(device)
        y = batch['y'].to(device)

        if DEBUG:
            print('x.size()', x.size())
            print('y.size()', y.size())

            _x = x.detach().cpu().numpy()
            _y = y.detach().cpu().numpy()
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True)

            for i, name in enumerate(['frames', 'velocity']):
                ax = axes[i]
                ax.plot(_x[0, :, i], label='x {}'.format(name))
                ax.plot(_y[0, :, i], label='y {}'.format(name))

            ax = axes[2]
            ax.plot(_y[0, :, 2], label='y bce')
            for ax in axes.flatten():
                ax.legend()

            plt.show()

        z, (h_n, c_n) = recurrent_model(x, (h_0, c_0))
        z = z.squeeze()

        y_hat = output_model(z)

        optimizer.zero_grad()
        loss_mse_frames = loss_function_mse(y_hat[:, 0], y[0, :, 0])
        loss_mse_velocity = loss_function_mse(y_hat[:, 1], y[0, :, 1])
        loss_bce = loss_function_bce(y_hat[:, 2], y[0, :, 2])

        loss = loss_mse_frames + loss_mse_velocity + loss_bce
        loss.backward()
        optimizer.step()

        logger.add_scalar('train_losses/mse_frames',
                          loss_mse_frames.detach().cpu().item(), global_step)
        logger.add_scalar('train_losses/mse_velocity',
                          loss_mse_velocity.detach().cpu().item(), global_step)
        logger.add_scalar('train_losses/mse_bce',
                          loss_bce.detach().cpu().item(), global_step)

        global_step += 1

    return global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('base_directory')
    parser.add_argument('--count-params-only', default=False, action='store_true')
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

    parameters = []
    for p in recurrent_model.parameters():
        parameters.append(p)

    for p in output_model.parameters():
        parameters.append(p)

    if args.count_params_only:
        n_total = 0
        for p in parameters:
            n_total += np.prod(list(p.size()))
        print('n_total', n_total)
        exit()

    # optimizer = torch.optim.Adam(parameters, lr=1e-3, betas=(0.9, 0.999))
    optimizer = torch.optim.SGD(parameters, lr=0.05, momentum=0.9, weight_decay=1e-5)

    base_directory = args.base_directory
    train_loader = get_data_loader(os.path.join(base_directory, 'train'), 'RandomSampler')

    # we have to decode the individual notes for individual pieces separately, of course ...
    valid_sequences = get_dataset_individually(os.path.join(base_directory, 'valid'))

    valid_loaders = []
    for sequence in valid_sequences:
        loader = DataLoader(
            sequence,
            batch_size=1,
            sampler=SequentialSampler(sequence),
            drop_last=False
        )
        valid_loaders.append((sequence.midifilename, loader))

    print('len(train_loader)', len(train_loader))

    log_dir = 'runs/rnn_lstm_maps_spec2labels_swd'
    logger = SummaryWriter(log_dir=log_dir)

    best_f = -np.inf
    global_step = 0
    for i_epoch in range(100):
        print('i_epoch', i_epoch)
        global_step = train(
            logger,
            device,
            recurrent_model,
            output_model,
            optimizer,
            train_loader,
            global_step
        )
        to_log = evaluate(
            logger,
            'valid',
            device,
            recurrent_model,
            output_model,
            valid_loaders,
            global_step
        )

        model_state = dict(
            recurrent_model=recurrent_model.state_dict(),
            output_model=output_model.state_dict()
        )
        torch.save(model_state, os.path.join(log_dir, 'model_state_{}.pkl'.format(i_epoch)))

        if best_f < to_log['valid_prf/f']:
            best_f = to_log['valid_prf/f']
            torch.save(model_state, os.path.join(log_dir, 'model_state_best.pkl'.format(i_epoch)))


if __name__ == '__main__':
    main()

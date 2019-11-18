import torch
from torch import nn

from sklearn.metrics import precision_recall_fscore_support as prfs
import numpy as np
from collections import defaultdict


def evaluate_aux(device, recurrent_model, output_model, loader, losses):
    y_true = []
    y_pred = []
    loss_function_bce = nn.BCEWithLogitsLoss(reduction='mean')
    loss_function_mse = nn.MSELoss(reduction='mean')

    with torch.no_grad():
        # each batch == one note
        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            z, h_n = recurrent_model(x)
            z = z.squeeze()

            y_hat = output_model(z)

            loss_mse_frames = loss_function_mse(y_hat[:, 0], y[0, :, 0])
            loss_mse_velocity = loss_function_mse(y_hat[:, 1], y[0, :, 1])
            loss_bce = loss_function_bce(y_hat[:, 2], y[0, :, 2])

            loss = loss_mse_frames + loss_mse_velocity + loss_bce

            losses['mse_frames'].append(loss_mse_frames.cpu().item())
            losses['mse_velocity'].append(loss_mse_velocity.cpu().item())
            losses['bce'].append(loss_bce.cpu().item())

            y_true.append(y[0, :, 2].cpu().numpy())
            y_pred.append(torch.sigmoid(y_hat[:, 2]).cpu().numpy())

    y_true = np.stack(y_true, axis=-1)
    y_pred = np.stack(y_pred, axis=-1)

    return y_true, y_pred


def evaluate(logger, tag_group, device, recurrent_model, output_model, loaders, global_step):
    recurrent_model.eval()
    output_model.eval()

    losses = defaultdict(list)
    all_prf = defaultdict(list)
    with torch.no_grad():
        loss_function_bce = nn.BCEWithLogitsLoss(reduction='mean')
        loss_function_mse = nn.MSELoss(reduction='mean')
        # each loader is for one sequence
        for midifilename, loader in loaders:
            print('evaluate midifilename', midifilename)
            y_true, y_pred = evaluate_aux(device, recurrent_model, output_model, loader, losses)
            print('y_true.shape', y_true.shape)
            print('y_pred.shape', y_pred.shape)

            y_pred = (y_pred > 0.5) * 1

            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
            # axes[0].imshow(y_true.T)
            # axes[1].imshow(y_pred.T)
            # plt.show()
            # exit()

            p, r, f, _ = prfs(y_true, y_pred, average='micro')
            print('p {:>4.2f} r {:>4.2f} f {:>4.2f}'.format(p, r, f))
            all_prf['p'].append(p)
            all_prf['r'].append(r)
            all_prf['f'].append(f)

    to_log = {
        '{}_prf/p'.format(tag_group): np.mean(all_prf['p']),
        '{}_prf/r'.format(tag_group): np.mean(all_prf['r']),
        '{}_prf/f'.format(tag_group): np.mean(all_prf['f']),
        '{}_losses/mse_frames'.format(tag_group): np.mean(losses['mse_frames']),
        '{}_losses/mse_velocity'.format(tag_group): np.mean(losses['mse_velocity']),
        '{}_losses/bce'.format(tag_group): np.mean(losses['bce'])
    }

    if logger is not None:
        for key, value in to_log.items():
            logger.add_scalar(key, value, global_step)
    return to_log

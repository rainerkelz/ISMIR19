import torch
from torch import optim
from sklearn.metrics import precision_recall_fscore_support

import numpy as np
from distances import SWD, PenalizeNegative, Joint

from tensorboardX import SummaryWriter
from train_loop import train

from audio_midi_dataset import get_data_loader
from reversible import ReversibleModel
import os


def get_loss_factor(i, n_full):
    b = np.log(2) / n_full
    a = np.exp(b)
    return np.clip((a ** i) - 1, 0., 1.)


def mse(x, y):
    return torch.mean((x - y) ** 2)


def main():
    direction = 'spec2labels'
    log_dir = 'runs/maps_{}_swd'.format(direction)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('log_dir', log_dir)
    print('using device', device)
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

    n_total = 0
    for p in model.parameters():
        n_total += torch.prod(torch.tensor(p.size()))
    print('n_total', n_total)

    lambda_padding = 1
    lambda_fit = 1
    lambda_latent = 1
    lambda_backward = 100
    loss_factor_function_backward = lambda epoch: get_loss_factor(epoch, 256)

    loss_function_padding = mse
    loss_function_fit = mse
    loss_function_latent = SWD(model.ndim_y + model.ndim_z, 2048, 2, device)
    loss_function_backward = Joint([
        PenalizeNegative(1.),
        SWD(model.ndim_x, 2048, 2, device)
    ])

    gradient_clip = 15.0
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
        betas=(0.8, 0.8),
        eps=1e-04,
        weight_decay=2e-5
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=meta_epoch,
        gamma=gamma
    )

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
    fold_directory = './splits/maps-non-overlapping'
    train_loader = get_data_loader(
        direction=direction,
        base_directory=base_directory,
        fold_file=os.path.join(fold_directory, 'train'),
        instrument_filename=os.path.join(fold_directory, 'instruments'),
        context=context,
        audio_options=audio_options,
        batch_size=batch_size,
        sampler_classname='ChunkedRandomSampler',
        chunk_size=batch_size * 2048,
        squeeze=True
    )

    valid_loader = get_data_loader(
        direction=direction,
        base_directory=base_directory,
        fold_file=os.path.join(fold_directory, 'valid'),
        instrument_filename=os.path.join(fold_directory, 'instruments'),
        context=context,
        audio_options=audio_options,
        batch_size=batch_size,
        sampler_classname='SequentialSampler',
        squeeze=True
    )

    test_loader = get_data_loader(
        direction=direction,
        base_directory=base_directory,
        fold_file=os.path.join(fold_directory, 'test'),
        instrument_filename=os.path.join(fold_directory, 'instruments'),
        context=context,
        audio_options=audio_options,
        batch_size=batch_size,
        sampler_classname='SequentialSampler',
        squeeze=True
    )

    print('len(train_loader)', len(train_loader))
    print('len(valid_loader)', len(valid_loader))
    print('len(test_loader)', len(test_loader))

    sample_fake_latent = lambda: None
    sample_real_latent = lambda: None

    sample_fake_backward = lambda: None
    sample_real_backward = lambda: None

    ###############################################################
    # start training loop
    logger = SummaryWriter(log_dir=log_dir)
    try:
        global_step = 0
        for i_epoch in range(n_epochs):
            scheduler.step()
            print('{}/{}'.format(i_epoch, n_epochs))
            global_step = train(
                logger=logger,
                tag_group='train',
                device=device,
                model=model,
                optimizer=optimizer,
                gradient_clip=gradient_clip,
                lambda_padding=lambda_padding,
                lambda_fit=lambda_fit,
                lambda_latent=lambda_latent,
                lambda_backward=lambda_backward,
                loss_function_padding=loss_function_padding,
                loss_function_fit=loss_function_fit,
                loss_function_latent=loss_function_latent,
                loss_function_backward=loss_function_backward,
                loss_factor_function_backward=loss_factor_function_backward,
                train_loader=train_loader,
                sample_fake_latent=sample_fake_latent(),
                sample_real_latent=sample_real_latent(),
                sample_fake_backward=sample_fake_backward(),
                sample_real_backward=sample_real_backward(),
                i_epoch=i_epoch,
                global_step=global_step
            )

            evaluate(
                logger=logger,
                tag_group='valid',
                loader=valid_loader,
                model=model,
                device=device,
                i_epoch=i_epoch,
                global_step=global_step
            )

            torch.save(
                model.state_dict(),
                os.path.join(log_dir, 'model_state_{}.pkl'.format(i_epoch))
            )
    except KeyboardInterrupt:
        pass
    finally:
        print('save model here')
        torch.save(
            model.state_dict(),
            os.path.join(log_dir, 'model_state_final.pkl')
        )
        logger.close()


def evaluate(logger, tag_group, loader, model, device, i_epoch, global_step):
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

    #############################################################################
    th_y_true = (y_true > 0.5) * 1
    th_y_pred = (y_pred > 0.5) * 1

    p, r, f, _ = precision_recall_fscore_support(th_y_true, th_y_pred, average='micro')
    logger.add_scalar('{}/p'.format(tag_group), p, global_step=global_step)
    logger.add_scalar('{}/r'.format(tag_group), r, global_step=global_step)
    logger.add_scalar('{}/f'.format(tag_group), f, global_step=global_step)

    output_directory = os.path.join(logger.file_writer.get_logdir(), 'model_outputs')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_name = os.path.join(output_directory, 'outputs_{:05d}.pkl'.format(i_epoch))
    torch.save(dict(
        x_pred=x_pred,
        y_pred=y_pred,
        z_pred=z_pred,

        x_true=x_true,
        y_true=y_true,
        z_true=z_true
    ), output_name)


if __name__ == '__main__':
    main()

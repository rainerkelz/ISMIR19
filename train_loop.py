import torch
import time


def normal_noise_like(a, scale):
    return a.new(a.size()).normal_(0, scale)


def output_bps_every_60s(t_start, b_start, t_current, b_current):
    t_duration = t_current - t_start
    if t_duration > 60:
        n_batches = b_current - b_start
        print('bps', float(n_batches) / float(t_duration))
        return t_current, b_current
    return t_start, b_start


def train(logger,
          tag_group,
          device,
          model,
          optimizer,
          gradient_clip,
          lambda_padding,
          lambda_fit,
          lambda_latent,
          lambda_backward,
          loss_function_padding,
          loss_function_fit,
          loss_function_latent,
          loss_function_backward,
          loss_factor_function_backward,
          train_loader,
          sample_fake_latent,
          sample_real_latent,
          sample_fake_backward,
          sample_real_backward,
          i_epoch,
          global_step):

    # dependent on the epoch, this *ramps up*, the longer we train, until it reaches 1
    # this is IMPORTANT! otherwise other losses are overwhelmed!
    loss_factor_backward = loss_factor_function_backward(i_epoch)

    t_start = time.time()
    b_start = 0
    for b_current, batch in enumerate(train_loader):
        global_step += 1

        ###################################################
        # train adversarial loss for latents
        lf_train_latent_loss = 0
        lf_train_backward_loss = 0
        if loss_function_latent.is_adversarial():
            model.eval()
            lf_train_latent_loss = loss_function_latent.train(
                sample_fake_latent,
                sample_real_latent
            )

        ###################################################
        # train adversarial loss for backward
        if loss_function_backward.is_adversarial():
            model.eval()
            lf_train_backward_loss = loss_function_backward.train(
                sample_fake_backward,
                sample_real_backward
            )

        ###################################################
        # train the invertible model itself
        model.train()
        x, y = batch['x'].to(device), batch['y'].to(device)
        optimizer.zero_grad()

        ###################################################
        # encode step
        z_hat, zy_hat_padding, y_hat = model.encode(x)

        zy_padding_noise = normal_noise_like(zy_hat_padding, model.zeros_noise_scale)
        loss_zy_padding = lambda_padding * loss_function_padding(zy_hat_padding, zy_padding_noise)

        y_noisy = y + normal_noise_like(y, model.y_noise_scale)
        loss_fit = lambda_fit * loss_function_fit(y_hat, y_noisy)

        # shorten output, and remove gradients wrt y, for latent loss
        zy_hat_detached = torch.cat([z_hat, y_hat.detach()], dim=1)
        z_proposal = normal_noise_like(z_hat, 1)
        zy = torch.cat([z_proposal, y_noisy], dim=1)
        loss_latent = lambda_latent * loss_function_latent(zy_hat_detached, zy)

        ###################################################
        # decode step
        z_hat_noisy = z_hat + normal_noise_like(z_hat, model.y_noise_scale)
        y_noisy = y + normal_noise_like(y, model.y_noise_scale)

        x_hat, x_hat_padding = model.decode(
            z_hat_noisy,
            y_noisy
        )
        z_proposal = normal_noise_like(z_hat, 1)
        x_hat_sampled, x_hat_sampled_padding = model.decode(
            z_proposal,
            y_noisy
        )

        loss_backward = (lambda_backward *
                         loss_factor_backward *
                         loss_function_backward(x_hat_sampled, x))

        loss_x_hat = (0.5 *
                      lambda_padding *
                      loss_function_padding(x_hat, x))

        x_hat_padding_noise = normal_noise_like(x_hat_padding, model.zeros_noise_scale)

        loss_x_padding = (0.5 *
                          lambda_padding *
                          loss_function_padding(x_hat_padding, x_hat_padding_noise))

        loss = (loss_fit +
                loss_latent +
                loss_backward +
                loss_x_hat +
                loss_x_padding +
                loss_zy_padding)

        loss.backward()

        # GRADIENT CLIPPING!
        for p in model.parameters():
            p.grad.data.clamp_(-gradient_clip, gradient_clip)

        optimizer.step()
        to_log = dict(
            loss_factor_backward=loss_factor_backward,
            loss_fit=loss_fit.detach().cpu().item(),
            loss_latent=loss_latent.detach().cpu().item(),
            loss_backward=loss_backward.detach().cpu().item(),
            loss_x_hat=loss_x_hat.detach().cpu().item(),
            loss_x_padding=loss_x_padding.detach().cpu().item(),
            loss_zy_padding=loss_zy_padding.detach().cpu().item(),
            loss=loss.detach().cpu().item(),
            lf_train_latent_loss=lf_train_latent_loss,
            lf_train_backward_loss=lf_train_backward_loss
        )
        for key, value in to_log.items():
            logger.add_scalar('{}/{}'.format(tag_group, key), value, global_step=global_step)

        t_start, b_start = output_bps_every_60s(t_start, b_start, time.time(), b_current)
    return global_step

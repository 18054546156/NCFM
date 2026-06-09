import torch

from .local_patch_ncfd import local_patch_feature_ncfd_loss
from .ssim_regularization import multiscale_ssim_prototype_loss


def compute_match_loss(
    args,
    loader_real,
    sample_fn,
    aug_fn,
    inner_loss_fn,
    optim_img,
    class_list,
    timing_tracker,
    model_interval,
    data_grad,
    optim_sampling_net = None,
    sampling_net =None
):

    loss_total = 0
    match_grad_mean = 0

    for c in class_list:
        timing_tracker.start_step()

        img, _ = loader_real.class_sample(c)
        timing_tracker.record("data")
        img_syn, _ = sample_fn(c)

        img_aug = aug_fn(torch.cat([img, img_syn]))
        timing_tracker.record("aug")
        n = img.shape[0]

        loss_global = inner_loss_fn(
            img_aug[:n], img_aug[n:], model_interval, sampling_net, args
        )
        loss = loss_global
        if getattr(args, "use_local_patch_feature_ncfd", False):
            loss_local = local_patch_feature_ncfd_loss(
                img_aug[:n],
                img_aug[n:],
                args.local_patch_encoder,
                args.cf_loss_func,
                args,
            )
            loss = loss_global + args.lambda_local_patch_ncfd * loss_local
        if getattr(args, "use_ssim_regularization", False):
            loss_ssim = multiscale_ssim_prototype_loss(img_aug[:n], img_aug[n:], args)
            ssim_weight = float(getattr(args, "ssim_weight", 0.1))
            loss = loss + ssim_weight * loss_ssim
            if bool(getattr(args, "ssim_log_components", True)):
                args._ssim_last_global_loss = float(loss_global.detach().item())
                args._ssim_last_total_loss = float(loss.detach().item())
        loss_total += loss.item()
        timing_tracker.record("loss")

        optim_img.zero_grad()
        if optim_sampling_net is not None:
            optim_sampling_net.zero_grad()
            loss.backward(retain_graph=True)
            optim_img.step()
            optim_img.zero_grad()
            (-loss_global).backward()
            optim_sampling_net.step()
            optim_sampling_net.zero_grad()
        else:
            loss.backward()
            optim_img.step()
        if data_grad is not None:
            match_grad_mean += torch.norm(data_grad).item()
        timing_tracker.record("backward")

    return loss_total, match_grad_mean


def compute_calib_loss(
    sample_fn,
    aug_fn,
    inter_loss_fn,
    optim_img,
    iter_calib,
    class_list,
    timing_tracker,
    model_final,
    calib_weight,
    data_grad,
):

    calib_loss_total = 0
    calib_grad_norm = 0
    for i in range(0, iter_calib):
        for c in class_list:
            timing_tracker.start_step()

            img_syn, label_syn = sample_fn(c)
            timing_tracker.record("data")

            img_aug = aug_fn(torch.cat([img_syn]))
            timing_tracker.record("aug")

            loss = calib_weight * inter_loss_fn(img_aug, label_syn, model_final)
            calib_loss_total += loss.item()
            timing_tracker.record("loss")

            optim_img.zero_grad()
            loss.backward()
            if data_grad is not None:
                calib_grad_norm = torch.norm(data_grad).item()
            optim_img.step()
            timing_tracker.record("backward")

    return calib_loss_total, calib_grad_norm

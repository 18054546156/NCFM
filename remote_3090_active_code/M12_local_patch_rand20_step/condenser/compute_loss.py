import torch

from .local_patch_ncfd import local_patch_feature_ncfd_loss, resample_local_patch_encoder


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
    lp_global_total = 0.0
    lp_local_total = 0.0
    lp_weighted_local_total = 0.0
    lp_total_loss_total = 0.0

    use_local_patch = getattr(args, "use_local_patch_feature_ncfd", False)
    if use_local_patch:
        resample_local_patch_encoder(args)

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
        if use_local_patch:
            loss_local = local_patch_feature_ncfd_loss(
                img_aug[:n],
                img_aug[n:],
                args.local_patch_encoder,
                args.cf_loss_func,
                args,
            )
            weighted_local = args.lambda_local_patch_ncfd * loss_local
            loss = loss_global + weighted_local
            lp_global_total += float(loss_global.detach().item())
            lp_local_total += float(loss_local.detach().item())
            lp_weighted_local_total += float(weighted_local.detach().item())
            lp_total_loss_total += float(loss.detach().item())
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

    if use_local_patch:
        denom = max(1, len(class_list))
        args._lp_last_global_loss = lp_global_total / denom
        args._lp_last_local_loss = lp_local_total / denom
        args._lp_last_weighted_local_loss = lp_weighted_local_total / denom
        args._lp_last_total_loss = lp_total_loss_total / denom

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

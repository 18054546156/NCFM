import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from NCFM.NCFM import match_loss
from utils.utils import define_model, load_resized_data


class SimpleTensorDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images.detach().float().cpu()
        self.labels = labels.detach().long().cpu()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def _get_targets(dataset) -> torch.Tensor:
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "labels"):
        targets = dataset.labels
    else:
        targets = [dataset[i][1] for i in range(len(dataset))]
    if isinstance(targets, torch.Tensor):
        return targets.long().view(-1).cpu()
    return torch.tensor(targets, dtype=torch.long).view(-1)


def _get_images_by_indices(dataset, indices: torch.Tensor) -> torch.Tensor:
    if hasattr(dataset, "images") and isinstance(dataset.images, torch.Tensor):
        return dataset.images[indices].float().cpu()

    samples = []
    for idx in indices.tolist():
        sample, _ = dataset[idx]
        if not isinstance(sample, torch.Tensor):
            sample = torch.as_tensor(sample)
        samples.append(sample.float())
    return torch.stack(samples, dim=0)


def _collect_balanced_subset(
    dataset, nclass: int, max_per_class: int
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    targets = _get_targets(dataset)
    image_chunks = []
    label_chunks = []
    counts = []
    for c in range(nclass):
        idx = torch.where(targets == c)[0][:max_per_class]
        counts.append(int(len(idx)))
        if len(idx) == 0:
            continue
        image_chunks.append(_get_images_by_indices(dataset, idx))
        label_chunks.append(targets[idx])
    if len(image_chunks) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long), counts
    return torch.cat(image_chunks, dim=0), torch.cat(label_chunks, dim=0), counts


def _extract_features(model, images: torch.Tensor, device: str, batch_size: int = 256):
    features = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(device)
            try:
                _, feat = model(batch, return_features=True)
            except TypeError:
                if hasattr(model, "embed"):
                    feat = model.embed(batch)
                else:
                    feat = model(batch)
            features.append(feat.detach().cpu())
    if was_training:
        model.train()
    return torch.cat(features, dim=0).numpy()


def _compute_gradient_topk(grad: torch.Tensor) -> Dict[str, float]:
    from scipy.fftpack import dct

    grad_np = grad.detach().cpu().numpy()
    bsz, channels, height, width = grad_np.shape
    grad_dct = np.zeros((bsz, channels, height, width), dtype=np.float64)
    for b in range(bsz):
        for c in range(channels):
            grad_dct[b, c] = dct(
                dct(grad_np[b, c], type=2, norm="ortho", axis=0),
                type=2,
                norm="ortho",
                axis=1,
            )

    energy_hw = np.square(grad_dct).mean(axis=(0, 1))
    flat = np.sort(energy_hw.reshape(-1))[::-1]
    cumsum = np.cumsum(flat) / (flat.sum() + 1e-12)
    out = {}
    for k in [20, 50, 100, 200, 400]:
        idx = min(k, len(cumsum)) - 1
        out[f"top{k}"] = float(cumsum[idx])
    return out


def _fisher_ratio(features: np.ndarray, labels: np.ndarray, nclass: int) -> float:
    if len(features) == 0:
        return 0.0
    global_mu = features.mean(axis=0)
    tr_sw = 0.0
    tr_sb = 0.0
    for c in range(nclass):
        mask = labels == c
        if mask.sum() < 2:
            continue
        x_c = features[mask]
        mu_c = x_c.mean(axis=0)
        tr_sw += ((x_c - mu_c) ** 2).sum()
        tr_sb += mask.sum() * ((mu_c - global_mu) ** 2).sum()
    return float(tr_sb / (tr_sw + 1e-8))


def _gaussian_mmd(x: np.ndarray, y: np.ndarray, max_samples: int = 500) -> float:
    if len(x) == 0 or len(y) == 0:
        return 0.0

    if len(x) > max_samples:
        x = x[np.random.choice(len(x), max_samples, replace=False)]
    if len(y) > max_samples:
        y = y[np.random.choice(len(y), max_samples, replace=False)]

    xy = np.concatenate([x, y], axis=0)
    dists = np.sqrt(((xy[:, None] - xy[None, :]) ** 2).sum(-1))
    sigma = np.median(dists[dists > 0])
    sigma = float(sigma) if sigma > 0 else 1.0

    def kernel(a, b):
        dist_sq = ((a[:, None] - b[None, :]) ** 2).sum(-1)
        return np.exp(-dist_sq / (2 * sigma**2))

    k_xx = kernel(x, x).mean()
    k_yy = kernel(y, y).mean()
    k_xy = kernel(x, y).mean()
    return float(k_xx + k_yy - 2 * k_xy)


def _cluster_separation_metrics(
    features: np.ndarray, labels: np.ndarray
) -> Dict[str, Optional[float]]:
    try:
        from sklearn.metrics import davies_bouldin_score, silhouette_score
    except Exception:
        return {"silhouette": None, "davies_bouldin": None}

    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2 or len(features) <= len(unique_labels):
        return {"silhouette": None, "davies_bouldin": None}
    if np.any(counts < 2):
        return {"silhouette": None, "davies_bouldin": None}

    metrics: Dict[str, Optional[float]] = {"silhouette": None, "davies_bouldin": None}
    try:
        metrics["silhouette"] = float(silhouette_score(features, labels))
    except Exception:
        metrics["silhouette"] = None
    try:
        metrics["davies_bouldin"] = float(davies_bouldin_score(features, labels))
    except Exception:
        metrics["davies_bouldin"] = None
    return metrics


def _classification_metrics(
    logits: List[np.ndarray], labels: List[np.ndarray], nclass: int
) -> Dict[str, object]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    all_logits = np.concatenate(logits, axis=0)
    all_labels = np.concatenate(labels, axis=0)
    all_probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
    all_pred = all_probs.argmax(axis=1)

    metrics = {
        "acc": float(accuracy_score(all_labels, all_pred)),
        "balanced_acc": float(balanced_accuracy_score(all_labels, all_pred)),
        "macro_f1": float(
            f1_score(all_labels, all_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(all_labels, all_pred, average="weighted", zero_division=0)
        ),
        "per_class_f1": f1_score(
            all_labels,
            all_pred,
            labels=list(range(nclass)),
            average=None,
            zero_division=0,
        ).tolist(),
        "pred_distribution": np.bincount(all_pred, minlength=nclass).tolist(),
        "label_distribution": np.bincount(all_labels, minlength=nclass).tolist(),
    }

    try:
        if nclass == 2:
            metrics["macro_auc"] = float(roc_auc_score(all_labels, all_probs[:, 1]))
        else:
            metrics["macro_auc"] = float(
                roc_auc_score(
                    all_labels,
                    all_probs,
                    multi_class="ovr",
                    average="macro",
                )
            )
    except ValueError:
        metrics["macro_auc"] = None

    return metrics


class DynamicsMonitor:
    def __init__(self, args):
        self.args = args
        self.enabled = bool(getattr(args, "enable_training_dynamics", False))
        self.logger = args.logger
        self.device = args.device
        self.rank = getattr(args, "rank", 0)
        self.log_path = os.path.join(
            args.save_dir, getattr(args, "train_log_path", "training_dynamics.json")
        )
        self.log_loss_every = int(getattr(args, "log_loss_every", 0))
        self.log_per_class_loss_every = int(
            getattr(args, "log_per_class_loss_every", 0)
        )
        self.log_grad_freq_every = int(getattr(args, "log_grad_freq_every", 0))
        self.log_feature_metrics_every = int(
            getattr(args, "log_feature_metrics_every", 0)
        )
        self.eval_during_condense_every = int(
            getattr(args, "eval_during_condense_every", 0)
        )
        self.eval_during_condense_epochs = int(
            getattr(args, "eval_during_condense_epochs", 50)
        )
        self.eval_during_condense_seed = int(
            getattr(args, "eval_during_condense_seed", 0)
        )
        self.dynamics_real_subset_per_class = int(
            getattr(args, "dynamics_real_subset_per_class", 256)
        )
        self.dynamics_feature_subset_per_class = int(
            getattr(args, "dynamics_feature_subset_per_class", 256)
        )
        self.dynamics_eval_batch_size = int(
            getattr(args, "dynamics_eval_batch_size", 256)
        )
        self.log = {
            "dataset": args.dataset,
            "ipc": int(args.ipc),
            "nclass": int(args.nclass),
            "config_path": getattr(args, "config_path", None),
            "loss": [],
            "per_class_loss": [],
            "gradient_frequency": [],
            "feature_metrics": [],
            "intermediate_eval": [],
        }

        self.real_probe_images = None
        self.real_probe_labels = None
        self.real_feature_images = None
        self.real_feature_labels = None
        self.test_loader = None

        if not self.enabled or self.rank != 0:
            return

        self._prepare_reference_data()
        self.logger(
            f"[Dynamics] enabled. log_path={self.log_path}, "
            f"loss_every={self.log_loss_every}, per_class_every={self.log_per_class_loss_every}, "
            f"grad_every={self.log_grad_freq_every}, feature_every={self.log_feature_metrics_every}, "
            f"eval_every={self.eval_during_condense_every}"
        )

    def _prepare_reference_data(self):
        try:
            import scipy  # noqa: F401
            import sklearn  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Training dynamics monitoring requires scipy and scikit-learn. "
                "Install them with `pip install -r requirements.txt`."
            ) from exc

        train_dataset, val_dataset = load_resized_data(
            self.args.dataset,
            self.args.data_dir,
            size=self.args.size,
            nclass=self.args.nclass,
            load_memory=self.args.load_memory,
        )
        (
            self.real_probe_images,
            self.real_probe_labels,
            probe_counts,
        ) = _collect_balanced_subset(
            train_dataset,
            self.args.nclass,
            self.dynamics_real_subset_per_class,
        )
        (
            self.real_feature_images,
            self.real_feature_labels,
            feature_counts,
        ) = _collect_balanced_subset(
            train_dataset,
            self.args.nclass,
            self.dynamics_feature_subset_per_class,
        )
        self.test_loader = DataLoader(
            val_dataset,
            batch_size=self.dynamics_eval_batch_size,
            shuffle=False,
            num_workers=min(4, int(getattr(self.args, "workers", 4))),
        )
        self.log["real_probe_counts"] = probe_counts
        self.log["real_feature_counts"] = feature_counts
        self.log["test_set_size"] = int(len(val_dataset))

    def should_run(self, interval: int, step: int) -> bool:
        return self.enabled and self.rank == 0 and interval > 0 and step % interval == 0

    def record_loss(self, step: int, match_loss_total: float, calib_loss_total: float):
        self.log["loss"].append(
            {
                "step": int(step),
                "match_loss_total": float(match_loss_total),
                "calib_loss_total": float(calib_loss_total),
                "total_loss": float(match_loss_total + calib_loss_total),
                "match_loss_per_class": float(match_loss_total / max(1, self.args.nclass)),
            }
        )

    def record_per_class_loss(
        self,
        step: int,
        syn_images: torch.Tensor,
        syn_labels: torch.Tensor,
        model,
    ):
        values: List[Optional[float]] = []
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for c in range(self.args.nclass):
                real_idx = torch.where(self.real_probe_labels == c)[0]
                syn_idx = torch.where(syn_labels.cpu() == c)[0]
                if len(real_idx) == 0 or len(syn_idx) == 0:
                    values.append(None)
                    continue
                real_c = self.real_probe_images[real_idx].to(self.device)
                syn_c = syn_images[syn_idx].to(self.device)
                loss_c = match_loss(real_c, syn_c, model, None, self.args)
                values.append(float(loss_c.item()))
        if was_training:
            model.train()
        self.log["per_class_loss"].append({"step": int(step), "values": values})

    def record_gradient_frequency(
        self,
        step: int,
        syn_images: torch.Tensor,
        syn_labels: torch.Tensor,
        model,
    ):
        syn_probe = syn_images.detach().clone().to(self.device).requires_grad_(True)
        total_loss = torch.tensor(0.0, device=self.device)
        was_training = model.training
        model.eval()
        for c in range(self.args.nclass):
            real_idx = torch.where(self.real_probe_labels == c)[0]
            syn_idx = torch.where(syn_labels.cpu() == c)[0]
            if len(real_idx) == 0 or len(syn_idx) == 0:
                continue
            real_c = self.real_probe_images[real_idx].to(self.device)
            syn_c = syn_probe[syn_idx]
            total_loss = total_loss + match_loss(real_c, syn_c, model, None, self.args)
        grad = torch.autograd.grad(total_loss, syn_probe)[0]
        topk = _compute_gradient_topk(grad)
        topk["step"] = int(step)
        topk["probe_loss"] = float(total_loss.item())
        self.log["gradient_frequency"].append(topk)
        if was_training:
            model.train()

    def record_feature_metrics(
        self,
        step: int,
        syn_images: torch.Tensor,
        syn_labels: torch.Tensor,
        model,
    ):
        syn_labels_np = syn_labels.detach().cpu().numpy()
        real_labels_np = self.real_feature_labels.numpy()
        syn_features = _extract_features(model, syn_images.detach().cpu(), self.device)
        real_features = _extract_features(
            model, self.real_feature_images.detach().cpu(), self.device
        )

        centroid_per_class = []
        for c in range(self.args.nclass):
            syn_mask = syn_labels_np == c
            real_mask = real_labels_np == c
            if syn_mask.sum() == 0 or real_mask.sum() == 0:
                centroid_per_class.append(None)
                continue
            mu_syn = syn_features[syn_mask].mean(axis=0)
            mu_real = real_features[real_mask].mean(axis=0)
            centroid_per_class.append(float(np.linalg.norm(mu_syn - mu_real)))

        centroid_valid = [x for x in centroid_per_class if x is not None]
        fisher_real = _fisher_ratio(real_features, real_labels_np, self.args.nclass)
        fisher_syn = _fisher_ratio(syn_features, syn_labels_np, self.args.nclass)
        real_cluster = _cluster_separation_metrics(real_features, real_labels_np)
        syn_cluster = _cluster_separation_metrics(syn_features, syn_labels_np)
        metrics = {
            "step": int(step),
            "centroid_drift_mean": float(np.mean(centroid_valid))
            if centroid_valid
            else None,
            "centroid_drift_max": float(np.max(centroid_valid))
            if centroid_valid
            else None,
            "centroid_drift_per_class": centroid_per_class,
            "fisher_ratio_real": float(fisher_real),
            "fisher_ratio_syn": float(fisher_syn),
            "fisher_retention": float(fisher_syn / (fisher_real + 1e-8)),
            "mmd": _gaussian_mmd(syn_features, real_features),
            "silhouette_real": real_cluster["silhouette"],
            "silhouette_syn": syn_cluster["silhouette"],
            "davies_bouldin_real": real_cluster["davies_bouldin"],
            "davies_bouldin_syn": syn_cluster["davies_bouldin"],
        }

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import balanced_accuracy_score

            clf = LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                multi_class="auto",
            ).fit(syn_features, syn_labels_np)
            pred = clf.predict(real_features)
            metrics["linear_probe_bacc"] = float(
                balanced_accuracy_score(real_labels_np, pred)
            )
        except Exception:
            metrics["linear_probe_bacc"] = None

        self.log["feature_metrics"].append(metrics)
        probe_msg = (
            f"{metrics['linear_probe_bacc']:.4f}"
            if metrics["linear_probe_bacc"] is not None
            else "N/A"
        )
        silhouette_msg = (
            f"{metrics['silhouette_syn']:.4f}"
            if metrics["silhouette_syn"] is not None
            else "N/A"
        )
        db_msg = (
            f"{metrics['davies_bouldin_syn']:.4f}"
            if metrics["davies_bouldin_syn"] is not None
            else "N/A"
        )
        self.logger(
            f"[Dynamics][step {step}] "
            f"fisher_ret={metrics['fisher_retention']:.4f} "
            f"mmd={metrics['mmd']:.4f} "
            f"linear_probe_bacc={probe_msg} "
            f"silhouette_syn={silhouette_msg} "
            f"davies_bouldin_syn={db_msg}"
        )

    def record_intermediate_eval(
        self, step: int, syn_images: torch.Tensor, syn_labels: torch.Tensor
    ):
        torch.manual_seed(self.eval_during_condense_seed)
        np.random.seed(self.eval_during_condense_seed)
        model = define_model(
            self.args.dataset,
            self.args.norm_type,
            self.args.net_type,
            self.args.nch,
            self.args.depth,
            self.args.width,
            self.args.nclass,
            self.args.logger,
            self.args.size,
        ).to(self.device)

        syn_dataset = SimpleTensorDataset(syn_images, syn_labels)
        syn_loader = DataLoader(
            syn_dataset,
            batch_size=min(self.dynamics_eval_batch_size, len(syn_dataset)),
            shuffle=True,
            num_workers=0,
        )

        if self.args.eval_optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.adamw_lr)
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.args.lr, momentum=self.args.momentum
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.eval_during_condense_epochs
        )
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        model.train()
        for _ in range(self.eval_during_condense_epochs):
            for inputs, targets in syn_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                if self.args.nch == 3 and torch.rand(1).item() > 0.5:
                    inputs = torch.flip(inputs, dims=[3])
                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                logits = model(inputs)
                logits_list.append(logits.detach().cpu().numpy())
                labels_list.append(targets.numpy())
        metrics = _classification_metrics(logits_list, labels_list, self.args.nclass)
        metrics["step"] = int(step)
        metrics["epochs"] = int(self.eval_during_condense_epochs)
        self.log["intermediate_eval"].append(metrics)
        self.logger(
            f"[Dynamics][step {step}] acc={metrics['acc']:.4f} "
            f"bacc={metrics['balanced_acc']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"macro_auc={metrics['macro_auc'] if metrics['macro_auc'] is not None else 'N/A'}"
        )

    def maybe_record(
        self,
        step: int,
        syn_images: torch.Tensor,
        syn_labels: torch.Tensor,
        model,
        match_loss_total: float,
        calib_loss_total: float,
    ):
        if not self.enabled or self.rank != 0:
            return

        wrote = False
        if self.should_run(self.log_loss_every, step):
            self.record_loss(step, match_loss_total, calib_loss_total)
            wrote = True
        if self.should_run(self.log_per_class_loss_every, step):
            self.record_per_class_loss(step, syn_images, syn_labels, model)
            wrote = True
        if self.should_run(self.log_grad_freq_every, step):
            self.record_gradient_frequency(step, syn_images, syn_labels, model)
            wrote = True
        if self.should_run(self.log_feature_metrics_every, step):
            self.record_feature_metrics(step, syn_images, syn_labels, model)
            wrote = True
        if step > 0 and self.should_run(self.eval_during_condense_every, step):
            self.record_intermediate_eval(step, syn_images, syn_labels)
            wrote = True
        if wrote:
            self.save()

    def save(self):
        if not self.enabled or self.rank != 0:
            return
        with open(self.log_path, "w") as f:
            json.dump(self.log, f, indent=2)


def build_dynamics_monitor(args):
    return DynamicsMonitor(args)

# train.py

import os
import json
import argparse
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from data import load_sampled_data_from_config
from models.selfgated_hierarchial_transformer import (
    SelfGatedHierarchicalTransformerEncoder,
)


# ------------- utils -------------


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_classification_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Acc={acc:.4f} | BalAcc={bal:.4f} | MacroF1={f1:.4f}")


# ------------- gating helpers -------------


def gates_to_sensor_segment_matrix(extras, reduce: str = "max"):
    gates = extras["gates"]
    W_proj = extras["W_proj"]

    if isinstance(gates, torch.Tensor):
        G = gates.detach().cpu().numpy()
    else:
        G = np.asarray(gates)

    if reduce == "max":
        G = G.max(axis=0)
    else:
        G = G.mean(axis=0)

    if isinstance(W_proj, torch.Tensor):
        Wabs_T = torch.abs(W_proj).T.detach().cpu().numpy()
    else:
        Wabs_T = np.abs(np.asarray(W_proj)).T

    M = Wabs_T @ G.T
    return M


def plot_gating_heatmap(
    M,
    sensor_names=None,
    fault_id=None,
    out_png="gating_heatmap_test.png",
    title_prefix="Gating Weights (Sensors x Segments)",
):
    F, S = M.shape
    vmax = np.max(np.abs(M)) + 1e-12

    if np.min(M) < 0:
        vmin, vmax_plot = -vmax, +vmax
        cmap_kwargs = dict(vmin=vmin, vmax=vmax_plot)
    else:
        cmap_kwargs = {}

    if sensor_names is None:
        sensor_names = [f"Var{i+1}" for i in range(F)]

    plt.figure(figsize=(max(6, 0.6 * S), max(6, 0.25 * F)))
    plt.imshow(M, aspect="auto", interpolation="nearest", **cmap_kwargs)
    cbar_label = "Gate Weight" if np.min(M) >= 0 else "Delta Gate Weight (fault - baseline)"
    plt.colorbar(label=cbar_label)
    plt.xticks(np.arange(S), [f"Seg {i}" for i in range(S)])
    plt.yticks(np.arange(F), sensor_names)
    title = title_prefix
    if fault_id is not None:
        title += f" - Fault {fault_id}"
    plt.title(title)
    plt.xlabel("Segments")
    plt.ylabel("Sensors")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_topk_sensors(
    M,
    sensor_names=None,
    k=10,
    fault_id=None,
    out_png="gating_topk.png",
    label="Mean Gate Weight (across segments)",
    title_prefix="Top Sensors",
):
    F, S = M.shape
    if sensor_names is None:
        sensor_names = [f"Var{i+1}" for i in range(F)]

    mean_per_sensor = M.mean(axis=1)
    idx = np.argsort(-mean_per_sensor)[:k]
    vals = mean_per_sensor[idx][::-1]
    names = [sensor_names[i] for i in idx][::-1]

    plt.figure(figsize=(8, max(4, 0.4 * k + 2)))
    plt.barh(np.arange(len(vals)), vals)
    plt.yticks(np.arange(len(vals)), names)
    plt.xlabel(label)
    title = title_prefix
    if fault_id is not None:
        title += f" - Fault {fault_id}"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def _compute_M_for_indices(model, X_tensor, idx_slice, reduce="max"):
    with torch.no_grad():
        logits, extras = model(X_tensor[idx_slice], return_gates=True)
    M = gates_to_sensor_segment_matrix(extras, reduce=reduce)
    return M


def make_fault_gating_plots_with_delta(
    model,
    X_test_tensor,
    y_test_tensor,
    faults_to_plot,
    baseline_fault=0,
    k_top=10,
    n_windows=128,
    reduce="max",
    out_dir="figures",
):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    idx0 = (y_test_tensor == baseline_fault).nonzero(as_tuple=False).squeeze(-1)
    M0 = None
    if idx0.numel() == 0:
        print(f"[WARN] No windows for baseline fault {baseline_fault}. Delta plots will be skipped.")
    else:
        sel0 = idx0[: min(n_windows, idx0.numel())]
        print(f"[BASELINE] fault={baseline_fault}, windows used={sel0.numel()}")
        M0 = _compute_M_for_indices(model, X_test_tensor, sel0, reduce=reduce)

    with torch.no_grad():
        for fid in faults_to_plot:
            idx = (y_test_tensor == fid).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                print(f"[WARN] No test windows found for fault {fid}; skipping.")
                continue

            sel = idx[: min(n_windows, idx.numel())]
            print(f"[FAULT] fid={fid}, windows used={sel.numel()}")

            M = _compute_M_for_indices(model, X_test_tensor, sel, reduce=reduce)
            sensor_names = [f"Var{i+1}" for i in range(M.shape[0])]

            heat_png = os.path.join(out_dir, f"gating_heatmap_fault_{fid}.png")
            plot_gating_heatmap(
                M,
                sensor_names,
                fault_id=fid,
                out_png=heat_png,
                title_prefix="Gating Weights (Sensors x Segments)",
            )

            top_png = os.path.join(out_dir, f"gating_top{k_top}_fault_{fid}.png")
            plot_topk_sensors(
                M,
                sensor_names,
                k=k_top,
                fault_id=fid,
                out_png=top_png,
                label="Mean Gate Weight (across segments)",
                title_prefix="Top Sensors by Gate Weight",
            )

            if M0 is not None and M0.shape == M.shape:
                Md = M - M0

                delta_heat_png = os.path.join(out_dir, f"gating_delta_heatmap_fault_{fid}.png")
                plot_gating_heatmap(
                    Md,
                    sensor_names,
                    fault_id=fid,
                    out_png=delta_heat_png,
                    title_prefix="Delta Gating (fault - baseline)",
                )

                delta_top_png = os.path.join(out_dir, f"gating_delta_top{k_top}_fault_{fid}.png")
                plot_topk_sensors(
                    Md,
                    sensor_names,
                    k=k_top,
                    fault_id=fid,
                    out_png=delta_top_png,
                    label="Mean Delta Gate Weight (vs baseline)",
                    title_prefix="Top Sensors by Delta Gate",
                )

                print(f"[INFO] Saved fault {fid} plots (absolute and delta).")
            else:
                print(f"[INFO] Baseline missing or shape mismatch; saved absolute plots for fault {fid} only.")


# ------------- training loop -------------


def train_one_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    cfg,
    device,
    results_dir: str,
):
    training_cfg = cfg["training"]

    epochs = int(training_cfg["epochs"])
    batch_size = int(training_cfg["batch_size"])
    eval_batch_size = int(training_cfg["eval_batch_size"])
    lr = float(training_cfg["learning_rate"])
    label_smoothing = float(training_cfg["label_smoothing"])

    model = model.to(device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = GradScaler()

    loss_history = []

    for epoch in range(epochs):
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model.train()
        permutation = torch.randperm(X_train_tensor.size(0), device=device)
        epoch_loss = 0.0

        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i : i + batch_size]
            batch_x = X_train_tensor[indices]
            batch_y = y_train_tensor[indices]

            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                with autocast(device_type="cuda"):
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y.squeeze())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(batch_x)
                loss = criterion(logits, batch_y.squeeze())
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, X_train_tensor.size(0) // batch_size)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(results_dir, "transformer_model_1.pth")
        torch.save(model.state_dict(), ckpt_path)

    # loss curve
    plt.figure()
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # evaluation
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, X_test_tensor.size(0), eval_batch_size):
            batch_x = X_test_tensor[i : i + eval_batch_size]
            logits = model(batch_x)
            preds.append(logits.argmax(dim=1).cpu())

    y_pred = torch.cat(preds).numpy()
    y_true = y_test_tensor.cpu().numpy()
    print_classification_metrics(y_true, y_pred)

    metrics = {
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }
    with open(os.path.join(results_dir, "test_raw_predictions.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return model, X_test_tensor, y_test_tensor


# ------------- argparse and main -------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Self Gated Hierarchical Transformer on TEP with config.yaml"
    )
    p.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml",
    )
    p.add_argument("--ff-path", type=str, default=None, help="Override fault free RData path")
    p.add_argument("--ft-path", type=str, default=None, help="Override faulty RData path")
    p.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    p.add_argument("--eval-batch-size", type=int, default=None, help="Override eval batch size")
    p.add_argument("--window-size", type=int, default=None, help="Override window size")
    p.add_argument("--stride", type=int, default=None, help="Override stride")
    p.add_argument("--seed", type=int, default=None, help="Override random seed")
    p.add_argument("--results-dir", type=str, default=None, help="Override results dir")
    p.add_argument(
        "--no-gating",
        action="store_true",
        help="Disable gating plots even if enabled in config",
    )
    return p.parse_args()


def main():
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # overrides
    if args.ff_path is not None:
        cfg["dataset"]["fault_free_path"] = args.ff_path
    if args.ft_path is not None:
        cfg["dataset"]["faulty_path"] = args.ft_path
    if args.window_size is not None:
        cfg["data_windowing"]["window_size"] = args.window_size
    if args.stride is not None:
        cfg["data_windowing"]["stride"] = args.stride
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        cfg["training"]["eval_batch_size"] = args.eval_batch_size
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.results_dir is not None:
        cfg["training"]["results_dir"] = args.results_dir

    results_dir = cfg["training"].get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    seed = int(cfg["training"]["seed"])
    set_seed(seed)

    print("[INFO] Loading data...")
    X_train, X_test, y_train, y_test = load_sampled_data_from_config(cfg)

    print("Training Data Shape:", X_train.shape, y_train.shape)
    print("Test Data Shape:", X_test.shape, y_test.shape)
    print("Unique classes in Training Set:", np.unique(y_train))
    print("Unique classes in Test Set:", np.unique(y_test))

    num_classes_cfg = int(cfg["model"].get("num_classes", int(np.max(y_train)) + 1))
    num_classes = max(num_classes_cfg, int(np.max(y_train)) + 1, int(np.max(y_test)) + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = SelfGatedHierarchicalTransformerEncoder(
        input_dim=X_train.shape[2],
        d_model=int(cfg["model"]["d_model"]),
        nhead=int(cfg["model"]["nhead"]),
        num_layers_low=int(cfg["model"]["num_layers_low"]),
        num_layers_high=int(cfg["model"]["num_layers_high"]),
        dim_feedforward=int(cfg["model"]["dim_feedforward"]),
        dropout=float(cfg["model"]["dropout"]),
        pool_output_size=int(cfg["model"]["pool_output_size"]),
        num_classes=num_classes,
    )

    model, X_test_tensor, y_test_tensor = train_one_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        cfg,
        device,
        results_dir=results_dir,
    )

    gating_cfg = cfg.get("gating", {})
    gating_enabled = bool(gating_cfg.get("enabled", True)) and not args.no_gating

    if gating_enabled:
        print("[INFO] Computing gating plots...")
        faults_to_plot = gating_cfg.get("faults_to_plot", [3, 9, 15])
        baseline_fault = int(gating_cfg.get("baseline_fault", 0))
        k_top = int(gating_cfg.get("k_top", 10))
        n_windows = int(gating_cfg.get("n_windows", 128))

        make_fault_gating_plots_with_delta(
            model,
            X_test_tensor,
            y_test_tensor,
            faults_to_plot=faults_to_plot,
            baseline_fault=baseline_fault,
            k_top=k_top,
            n_windows=n_windows,
            reduce="max",
            out_dir=os.path.join(results_dir, "figures"),
        )
    else:
        print("[INFO] Gating plots disabled.")


if __name__ == "__main__":
    main()

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.manifold import TSNE

from data import load_sampled_data
from utils import print_classification_metrics
from models.selfgated_hierarchial_transformer import (
    SelfGatedHierarchicalTransformerEncoder,
)


# ---------------- GATING UTILITIES (no training used here) ----------------

def gates_to_sensor_segment_matrix(extras, reduce: str = "max"):
    """
    Convert per-d_model gates into per-sensor × segment weights.

    Approx attribution: weight[f, s] ≈ ∑_d |W_proj[f,d]| * gate[s,d]
    Where W_proj is the input projection (d_model x F in PyTorch param shape).

    Args:
        extras: dict from forward(return_gates=True)
                - 'gates': (B, S, d_model)
                - 'W_proj': (d_model, F) torch.Tensor
        reduce: 'mean' or 'max' across batch dimension

    Returns:
        M: (F, S) numpy array
    """
    gates = extras["gates"]
    W_proj = extras["W_proj"]

    if isinstance(gates, torch.Tensor):
        G = gates.detach().cpu().numpy()
    else:
        G = np.asarray(gates)

    if reduce == "max":
        G = G.max(axis=0)        # (S, d_model)
    else:
        G = G.mean(axis=0)       # (S, d_model)

    if isinstance(W_proj, torch.Tensor):
        Wabs_T = torch.abs(W_proj).T.detach().cpu().numpy()  # (F, d_model)
    else:
        Wabs_T = np.abs(np.asarray(W_proj)).T

    M = Wabs_T @ G.T            # (F, S)
    return M


def plot_topk_sensors(
    M: np.ndarray,
    sensor_names=None,
    k: int = 10,
    fault_id=None,
    out_png: str = "gating_topk.png",
    label: str = "Mean Gate Weight (across segments)",
    title_prefix: str = "Top Sensors",
):
    """
    Bar plot for top-k sensors based on mean across segments.
    """
    F, S = M.shape
    if sensor_names is None:
        sensor_names = [f"Var{i+1}" for i in range(F)]

    mean_per_sensor = M.mean(axis=1)  # (F,)
    idx = np.argsort(-mean_per_sensor)[:k]
    vals = mean_per_sensor[idx][::-1]
    names = [sensor_names[i] for i in idx][::-1]

    plt.figure(figsize=(8, max(4, 0.4 * k + 2)))
    plt.barh(np.arange(len(vals)), vals)
    plt.yticks(np.arange(len(vals)), names)
    plt.xlabel(label)
    title = title_prefix
    if fault_id is not None:
        title += f" – Fault {fault_id}"
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def _compute_M_for_indices(model, X_tensor, idx_slice, reduce: str = "max"):
    """
    Forward with gates on X_tensor[idx_slice] and return sensor×segment matrix M.
    """
    with torch.no_grad():
        logits, extras = model(X_tensor[idx_slice], return_gates=True)
    M = gates_to_sensor_segment_matrix(extras, reduce=reduce)
    return M


def make_fault_delta_bar_plots(
    model: torch.nn.Module,
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    fault_ids,
    baseline_fault: int = 0,
    k_top: int = 10,
    n_windows: int = 512,
    reduce: str = "max",
):
    """
    For each fault in fault_ids, save ONLY:
      - absolute top-k bar plot
      - delta (fault - baseline fault 0) top-k bar plot

    No heatmaps here.
    """
    os.makedirs("figures", exist_ok=True)
    model.eval()

    # baseline fault 0
    idx0 = (y_tensor == baseline_fault).nonzero(as_tuple=False).squeeze(-1)
    M0 = None
    if idx0.numel() == 0:
        return
    else:
        sel0 = idx0[: min(n_windows, idx0.numel())]
        print(f"[BASELINE] fault={baseline_fault}, windows used={sel0.numel()}")
        M0 = _compute_M_for_indices(model, X_tensor, sel0, reduce=reduce)

    F, _ = M0.shape
    sensor_names = [f"Var{i+1}" for i in range(F)]

    with torch.no_grad():
        for fid in fault_ids:
            idx = (y_tensor == fid).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue

            sel = idx[: min(n_windows, idx.numel())]
            print(f"[FAULT] fid={fid}, windows used={sel.numel()}")

            M = _compute_M_for_indices(model, X_tensor, sel, reduce=reduce)  # (F, S)

            abs_png = f"figures/gating_top{k_top}_fault_{fid}.png"
            plot_topk_sensors(
                M,
                sensor_names,
                k=k_top,
                fault_id=fid,
                out_png=abs_png,
                label="Mean Gate Weight (across segments)",
                title_prefix="Top Sensors by Gate Weight",
            )

            if M0 is not None and M0.shape == M.shape:
                Md = M - M0

                mean_delta = Md.mean(axis=1)
                idx_debug = np.argsort(-np.abs(mean_delta))[:k_top]
                names_debug = [sensor_names[i] for i in idx_debug]
                vals_debug = mean_delta[idx_debug]
                print(
                    f"[DEBUG] Fault {fid} Δ top sensors (used for plot):",
                    list(zip(names_debug, vals_debug)),
                )

                delta_png = f"figures/gating_delta_top{k_top}_fault_{fid}.png"
                plot_topk_sensors(
                    Md,
                    sensor_names,
                    k=k_top,
                    fault_id=fid,
                    out_png=delta_png,
                    label="Mean Δ Gate Weight (vs baseline)",
                    title_prefix="Top Sensors by Δ Gate",
                )

                print(f"[INFO] Saved fault {fid} plots (abs + delta bar) to figures/")
            else:
                print(
                    f"[INFO] Baseline missing or shape mismatch; saved absolute top-k plot for fault {fid} only."
                )


# ---------------- MODEL BUILD / CHECKPOINT LOAD ----------------

def build_model(
    input_dim: int,
    num_classes: int = 21,
    d_model: int = 128,
    nhead: int = 4,
    num_layers_low: int = 2,
    num_layers_high: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.05,
    pool_output_size: int = 10,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    model = SelfGatedHierarchicalTransformerEncoder(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers_low=num_layers_low,
        num_layers_high=num_layers_high,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        pool_output_size=pool_output_size,
        num_classes=num_classes,
    )
    return model.to(device)


def load_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
) -> torch.nn.Module:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Loaded weights from: {ckpt_path}")
    return model


# ---------------- ACCURACY INFERENCE ----------------

def run_inference_for_accuracy(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> None:
    """
    Run inference on the TEST set (from Testing RData files),
    print metrics and per-fault accuracies.
    No plots here.
    """
    y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_np = X_test[i: i + batch_size]
            batch_tensor = torch.tensor(batch_np, dtype=torch.float32, device=device)
            out = model(batch_tensor)
            preds.append(out.argmax(dim=1).cpu())

    y_pred = torch.cat(preds).numpy()
    y_true = y_test_tensor.cpu().numpy()

    print("[INFO] Test set classification metrics:")
    print_classification_metrics(y_true, y_pred)

    print("\n[INFO] Per-fault accuracy (%):")
    faults = np.unique(y_true)
    acc_list = []

    for f in faults:
        mask = (y_true == f)
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).mean() * 100.0
            acc_list.append(acc)
            print(f"  Fault {int(f):2d}: {acc:.2f}%")
        else:
            print(f"  Fault {int(f):2d}: N/A (no samples)")

    if len(acc_list) > 0:
        macro_acc = float(np.mean(acc_list))
        # print(f"\n[INFO] Mean per-fault accuracy: {macro_acc:.2f}%")


# ---------------- T-SNE EMBEDDING PLOT ----------------

@torch.no_grad()
def compute_sght_embeddings(model, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """
    Compute embeddings from SelfGatedHierarchicalTransformerEncoder by
    reproducing the forward path up to encoder_high and mean pooling over segments.

    Returns:
        embeddings: (N, d_model) numpy array.
    """
    model.eval()
    all_emb = []

    for i in range(0, len(X), batch_size):
        batch_np = X[i: i + batch_size]
        batch = torch.tensor(batch_np, dtype=torch.float32, device=device)

        # Same internal path as forward, but stopping before classifier
        z = model.input_proj(batch)                         # (B, T, d_model)
        z = model.pos_encoder(z)                            # (B, T, d_model)
        low_out = model.encoder_low(z)                      # (B, T, d_model)
        pooled = model.pool(low_out.transpose(1, 2)).transpose(1, 2)  # (B, S, d_model)
        gated = model.self_gate(pooled)                     # (B, S, d_model)
        high_out = model.encoder_high(gated)                # (B, S, d_model)
        emb = high_out.mean(dim=1)                          # (B, d_model)

        all_emb.append(emb.cpu().numpy())

    embeddings = np.vstack(all_emb)
    return embeddings


def tsne_plot_sght(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    save_path: str = "figures/tsne_sght_test.png",
    sample_per_class: int = 800,
    seed: int = 0,
):
    """
    Compute SGHT embeddings for the test set, run t-SNE and save a 2D scatter
    colored by fault class, similar to the reference embedding plot.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    rng = np.random.default_rng(seed)
    y_test_int = y_test.astype(int)
    classes = np.unique(y_test_int)

    # Subsample per class to keep t-SNE tractable and balanced
    idx_sel = []
    for c in classes:
        idx_c = np.where(y_test_int == c)[0]
        if len(idx_c) > sample_per_class:
            idx_c = rng.choice(idx_c, size=sample_per_class, replace=False)
        idx_sel.append(idx_c)
    idx_sel = np.concatenate(idx_sel, axis=0)

    X_sel = X_test[idx_sel]
    Y_sel = y_test_int[idx_sel]

    print(f"[INFO] Computing embeddings for t-SNE on {len(X_sel)} samples")
    feats = compute_sght_embeddings(model, X_sel, device=device, batch_size=256)

    print("[INFO] Running t-SNE")
    tsne = TSNE(
        n_components=2,
        learning_rate="auto",
        init="random",
        perplexity=35,
        random_state=seed,
    )
    Z = tsne.fit_transform(feats)

    print("[INFO] Plotting t-SNE embedding")
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=Y_sel, s=6, cmap="tab20")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, ticks=classes)
    cbar.set_label("Fault class")
    ax.set_title("t-SNE embedding of SGHT features")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved t-SNE plot to: {save_path}")


# ---------------- ARGPARSE AND MAIN ----------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference (accuracy + gating Δ bar plots + t-SNE) for Self-Gated Hierarchical Transformer on TEP."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="transformer_model_1.pth",
        help="Path to the .pth checkpoint file.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Window size used during training/data loading.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Stride used during training/data loading.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="If set, skip generating gating Δ bar plots and t-SNE.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using device: {device} (CUDA enabled for inference)")
    else:
        device = torch.device("cpu")
        print(f"[INFO] Using device: {device} (CUDA disabled for inference)")

    # Testing RData: used for accuracy and t-SNE
    X_train_tmp, X_test_metrics, y_train_tmp, y_test_metrics = load_sampled_data(
        window_size=args.window_size,
        stride=args.stride,
        type_model="supervised",
        fault_free_path="/workspace/TEP_FaultFree_Testing.RData",
        faulty_path="/workspace/TEP_Faulty_Testing.RData",
        train_end=10000,
        test_start=10000,
        test_end=15000,
        train_run_start=5,
        train_run_end=20,
        test_run_start=1,
        test_run_end=80,
    )

    print("Test (Testing file) Data Shape:", X_test_metrics.shape, y_test_metrics.shape)
    print("Unique classes in Test (Testing file):", np.unique(y_test_metrics))

    num_classes = 21
    input_dim = X_train_tmp.shape[2]

    model = build_model(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=128,
        nhead=4,
        num_layers_low=2,
        num_layers_high=2,
        dim_feedforward=128,
        dropout=0.05,
        pool_output_size=10,
        device=device,
    )
    model = load_checkpoint(model, args.ckpt, device)

    # Accuracy on Testing RData
    run_inference_for_accuracy(
        model=model,
        X_test=X_test_metrics,
        y_test=y_test_metrics,
        device=device,
        batch_size=512,
    )

    # t-SNE embedding on Testing RData
    if not args.no_plots:
        tsne_plot_sght(
            model=model,
            X_test=X_test_metrics,
            y_test=y_test_metrics,
            device=device,
            save_path="figures/tsne_sght_test.png",
            sample_per_class=800,
            seed=0,
        )

    if args.no_plots:
        return

    print("\n[INFO] Plot code running for gating plots")

    # For gating plots, you use windows sampled from training file
    X_train_trainfile, X_test_plots, y_train_trainfile, y_test_plots = load_sampled_data(
        window_size=args.window_size,
        stride=args.stride,
        type_model="supervised",
        fault_free_path="/workspace/TEP_FaultFree_Training.RData",
        faulty_path="/workspace/TEP_Faulty_Training.RData",
        train_end=1000,
        test_start=100000,
        test_end=114500,
        train_run_start=5,
        train_run_end=6,
        test_run_start=100,
        test_run_end=140,
    )

    print("Test (from Training file) Data Shape:", X_test_plots.shape, y_test_plots.shape)
    print("Unique classes in Test (from Training file):", np.unique(y_test_plots))

    X_test_tensor_plots = torch.tensor(X_test_plots, dtype=torch.float32, device=device)
    y_test_tensor_plots = torch.tensor(y_test_plots, dtype=torch.long, device=device)

    make_fault_delta_bar_plots(
        model=model,
        X_tensor=X_test_tensor_plots,
        y_tensor=y_test_tensor_plots,
        fault_ids=[3, 9, 15],
        baseline_fault=0,
        k_top=10,
        n_windows=512,
        reduce="max",
    )


if __name__ == "__main__":
    main()

# data.py

import os
import pyreadr
import numpy as np
import pandas as pd

try:
    import cupy as cp
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    GPU_AVAILABLE = True
except Exception as e:
    print(f"[WARN] GPU / RAPIDS scaler not available ({e}); using CPU StandardScaler.")
    cp = np
    GPU_AVAILABLE = False

from sklearn.preprocessing import StandardScaler as skStandardScaler


def _guess_rdata_key(path: str) -> str:
    """Choose the correct key based on file name."""
    base = os.path.basename(path)
    if "Training" in base:
        return "fault_free_training" if "FaultFree" in base else "faulty_training"
    if "Testing" in base:
        return "fault_free_testing" if "FaultFree" in base else "faulty_testing"
    raise ValueError(f"Cannot determine dataset key from file name: {path}")


def read_training_data(
    fault_free_path: str,
    faulty_path: str,
):
    """
    Read TEP fault free and faulty data and return one concatenated DataFrame
    sorted by faultNumber and simulationRun.
    """
    b1_r = pyreadr.read_r(fault_free_path)
    b2_r = pyreadr.read_r(faulty_path)

    key_free = _guess_rdata_key(fault_free_path)
    key_faulty = _guess_rdata_key(faulty_path)

    if key_free not in b1_r:
        raise KeyError(f"Key '{key_free}' not found in {fault_free_path}")
    if key_faulty not in b2_r:
        raise KeyError(f"Key '{key_faulty}' not found in {faulty_path}")

    b1 = b1_r[key_free]
    b2 = b2_r[key_faulty]

    train_ts = pd.concat([b1, b2])
    return train_ts.sort_values(by=["faultNumber", "simulationRun"])


def sample_train_and_test(
    train_ts: pd.DataFrame,
    type_model: str,
    train_end: int | None = None,
    test_start: int | None = None,
    test_end: int | None = None,
    train_run_start: int | None = None,
    train_run_end: int | None = None,
    test_run_start: int | None = None,
    test_run_end: int | None = None,
):
    """
    Sample training and test data based on model type and slicing options.
    """
    sampled_train, sampled_test = pd.DataFrame(), pd.DataFrame()
    fault_0_data = train_ts[train_ts["faultNumber"] == 0]

    # default slices if not given
    if train_end is None:
        train_end = 248000
    if train_run_start is None:
        train_run_start = 1
    if train_run_end is None:
        train_run_end = 200
    if test_start is None:
        test_start = 248000
    if test_end is None:
        test_end = 250000
    if test_run_start is None:
        test_run_start = 200
    if test_run_end is None:
        test_run_end = 220

    # train
    frames_train = []
    if type_model == "supervised":
        for i in sorted(train_ts["faultNumber"].unique()):
            if i == 0:
                frames_train.append(
                    train_ts[train_ts["faultNumber"] == i].iloc[:train_end]
                )
            else:
                fr = []
                b = train_ts[train_ts["faultNumber"] == i]
                for x in range(train_run_start, train_run_end):
                    b_x = b[b["simulationRun"] == x].iloc[20:500]
                    fr.append(b_x)
                if fr:
                    frames_train.append(pd.concat(fr))
    elif type_model == "unsupervised":
        frames_train.append(fault_0_data)
    else:
        raise ValueError(f"Unknown type_model: {type_model}")

    sampled_train = pd.concat(frames_train)

    # test
    frames_test = []
    for i in sorted(train_ts["faultNumber"].unique()):
        if i == 0:
            frames_test.append(fault_0_data.iloc[test_start:test_end])
        else:
            fr = []
            b = train_ts[train_ts["faultNumber"] == i]
            for x in range(test_run_start, test_run_end):
                b_x = b[b["simulationRun"] == x].iloc[135:660]
                fr.append(b_x)
            if fr:
                frames_test.append(pd.concat(fr))

    sampled_test = pd.concat(frames_test)
    return sampled_train, sampled_test


def scale_and_window(
    X_df: pd.DataFrame,
    scaler,
    use_gpu: bool = True,
    y_col: str = "faultNumber",
    window_size: int = 20,
    stride: int = 5,
):
    """
    Apply scaling and sliding window segmentation.

    Returns:
        X_win: [num_windows, window_size, num_features]
        y_win: [num_windows]
    """
    y = X_df[y_col].values
    X = X_df.iloc[:, 3:].values

    if use_gpu and GPU_AVAILABLE:
        X_scaled = scaler.transform(cp.asarray(X))
        num_windows = (len(X_scaled) - window_size) // stride + 1
        X_indices = cp.arange(window_size)[None, :] + cp.arange(num_windows)[:, None] * stride
        y_indices = cp.arange(window_size - 1, len(X_scaled), stride)

        X_win = X_scaled[X_indices]
        y_win = cp.asarray(y)[y_indices]

        return X_win.get(), y_win.get()
    else:
        X_scaled = scaler.transform(X)
        num_windows = (len(X_scaled) - window_size) // stride + 1
        X_indices = np.arange(window_size)[None, :] + np.arange(num_windows)[:, None] * stride
        y_indices = np.arange(window_size - 1, len(X_scaled), stride)

        X_win = np.array([X_scaled[idx] for idx in X_indices])
        y_win = y[y_indices]

        return X_win, y_win


def load_sampled_data_from_config(cfg: dict):
    """
    Main entry point for training and inference.

    Uses paths and slicing from config.yaml.
    """
    ds_cfg = cfg["dataset"]
    dw_cfg = cfg["data_windowing"]

    fault_free_path = ds_cfg["fault_free_path"]
    faulty_path = ds_cfg["faulty_path"]
    type_model = ds_cfg.get("type_model", "supervised")
    use_gpu_scaler = bool(ds_cfg.get("use_gpu_scaler", True))

    train_ts = read_training_data(
        fault_free_path=fault_free_path,
        faulty_path=faulty_path,
    )

    sampled_train, sampled_test = sample_train_and_test(
        train_ts,
        type_model=type_model,
        train_end=dw_cfg.get("train_end"),
        test_start=dw_cfg.get("test_start"),
        test_end=dw_cfg.get("test_end"),
        test_run_start=dw_cfg.get("test_run_start"),
        test_run_end=dw_cfg.get("test_run_end"),
        train_run_start=dw_cfg.get("train_run_start"),
        train_run_end=dw_cfg.get("train_run_end"),
    )

    fault_free = sampled_train[sampled_train["faultNumber"] == 0].iloc[:, 3:].values

    if use_gpu_scaler and GPU_AVAILABLE:
        scaler = cuStandardScaler()
        scaler.fit(cp.asarray(fault_free))
        print(f"[INFO] Using GPU Scaler (cuML), fit on fault free samples: {fault_free.shape[0]} rows")
    else:
        scaler = skStandardScaler()
        scaler.fit(fault_free)
        print(f"[INFO] Using CPU Scaler, fit on fault free samples: {fault_free.shape[0]} rows")

    print("[INFO] Scaling and windowing training data...")
    X_train, y_train = scale_and_window(
        sampled_train,
        scaler,
        use_gpu=use_gpu_scaler,
        y_col="faultNumber",
        window_size=dw_cfg["window_size"],
        stride=dw_cfg["stride"],
    )

    print("[INFO] Scaling and windowing test data...")
    X_test, y_test = scale_and_window(
        sampled_test,
        scaler,
        use_gpu=use_gpu_scaler,
        y_col="faultNumber",
        window_size=dw_cfg["window_size"],
        stride=dw_cfg["stride"],
    )

    return X_train, X_test, y_train, y_test

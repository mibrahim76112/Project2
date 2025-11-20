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
    cp = np  # Fallback to NumPy
    GPU_AVAILABLE = False

from sklearn.preprocessing import StandardScaler as skStandardScaler


def read_training_data(
    fault_free_path: str = "/workspace/TEP_FaultFree_Training.RData",
    faulty_path: str = "/workspace/TEP_Faulty_Training.RData",
):
    """
    Reads the Tennessee Eastman Process (TEP) training or testing data from RData files.

    The function automatically chooses the correct key ('fault_free_training',
    'fault_free_testing', 'faulty_training', or 'faulty_testing') based on the filename.

    Args:
        fault_free_path (str): Path to the fault-free RData file.
        faulty_path (str): Path to the faulty RData file.

    Returns:
        pd.DataFrame: Concatenated and sorted DataFrame of fault-free and faulty data.
    """
    # --- Load RData files ---
    b1_r = pyreadr.read_r(fault_free_path)
    b2_r = pyreadr.read_r(faulty_path)

    
    if "Training" in os.path.basename(fault_free_path):
        key_free = "fault_free_training"
    elif "Testing" in os.path.basename(fault_free_path):
        key_free = "fault_free_testing"
    else:
        raise ValueError(f"Cannot determine dataset key from file name: {fault_free_path}")

    if "Training" in os.path.basename(faulty_path):
        key_faulty = "faulty_training"
    elif "Testing" in os.path.basename(faulty_path):
        key_faulty = "faulty_testing"
    else:
        raise ValueError(f"Cannot determine dataset key from file name: {faulty_path}")

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
    Samples training and test data based on model type (supervised or unsupervised),
    with configurable slicing for fault 0 and simulation run ranges.

    Args:
        train_ts (pd.DataFrame): The full dataset.
        type_model (str): "supervised" or "unsupervised".
        train_end (int or None): Row index (exclusive) for fault 0 training subset.
        test_start (int or None): Start row index for fault 0 test subset.
        test_end (int or None): End row index (exclusive) for fault 0 test subset.
        test_run_start (int or None): Start of simulationRun range for faulty test data.
        test_run_end (int or None): End (exclusive) of simulationRun range for faulty test data.
        train_run_start (int or None): Start of simulationRun range for faulty training data.
        train_run_end (int or None): End (exclusive) of simulationRun range for faulty training data.

    Returns:
        tuple: (sampled_train, sampled_test) as pandas DataFrames.
    """
    sampled_train, sampled_test = pd.DataFrame(), pd.DataFrame()
    fault_0_data = train_ts[train_ts["faultNumber"] == 0]

    # Defaults to keep your current behavior if not provided
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
    
    # -------- TRAIN --------
    frames_train = []
    if type_model == "supervised":
        for i in sorted(train_ts["faultNumber"].unique()):
            if i == 0:
                # configurable slice for fault 0 train
                frames_train.append(
                    train_ts[train_ts["faultNumber"] == i].iloc[:train_end]
                )
            else:
                fr = []
                b = train_ts[train_ts["faultNumber"] == i]
                # NOW configurable training simulationRun range
                for x in range(train_run_start, train_run_end):
                    b_x = b[b["simulationRun"] == x].iloc[20:500]
                    fr.append(b_x)
                frames_train.append(pd.concat(fr))
    elif type_model == "unsupervised":
        frames_train.append(fault_0_data)

    sampled_train = pd.concat(frames_train)

    # -------- TEST --------
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
            frames_test.append(pd.concat(fr))

    sampled_test = pd.concat(frames_test)
    return sampled_train, sampled_test


def scale_and_window(
    X_df,
    scaler,
    use_gpu: bool = True,
    y_col: str = "faultNumber",
    window_size: int = 20,
    stride: int = 5,
):
    """
    Applies scaling and sliding window segmentation to the dataset.

    Args:
        X_df (pd.DataFrame): Input time-series data with labels.
        scaler: Fitted scaler (cuML or scikit-learn).
        use_gpu (bool): Whether to use GPU-based transformations when available.
        y_col (str): Column name containing the target/fault labels.
        window_size (int): Size of each time window.
        stride (int): Step size between consecutive windows.

    Returns:
        tuple: (X_win, y_win)
            - X_win (np.ndarray): [num_windows, window_size, num_features]
            - y_win (np.ndarray): [num_windows]
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


def load_sampled_data(
    window_size: int = 20,
    stride: int = 5,
    type_model: str = "supervised",
    use_gpu: bool = True,
    fault_free_path: str = "/workspace/TEP_FaultFree_Training.RData",
    faulty_path: str = "/workspace/TEP_Faulty_Testing.RData",
    train_end: int | None = None,
    test_start: int | None = None,
    test_end: int | None = None,
    test_run_start: int | None = None,
    test_run_end: int | None = None,
    train_run_start: int | None = None,
    train_run_end: int | None = None,
):
    """
    Loads, scales, and windows the sampled training and test data.

    Args:
        window_size (int): Length of each sliding window.
        stride (int): Step size for windowing.
        type_model (str): "supervised" or "unsupervised".
        use_gpu (bool): Whether to use GPU scaler (if available).
        fault_free_path (str): Path to fault-free training RData.
        faulty_path (str): Path to faulty training/testing RData.
        train_end (int or None): Row index (exclusive) for fault 0 train subset.
        test_start (int or None): Start row index for fault 0 test subset.
        test_end (int or None): End row index (exclusive) for fault 0 test subset.
        test_run_start (int or None): Start simulationRun for faulty test data.
        test_run_end (int or None): End (exclusive) simulationRun for faulty test data.
        train_run_start (int or None): Start simulationRun for faulty train data.
        train_run_end (int or None): End (exclusive) simulationRun for faulty train data.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
            - X_train: [N_train, window_size, num_features]
            - X_test: [N_test, window_size, num_features]
            - y_train: [N_train]
            - y_test: [N_test]
    """
    train_ts = read_training_data(
        fault_free_path=fault_free_path,
        faulty_path=faulty_path,
    )

    sampled_train, sampled_test = sample_train_and_test(
        train_ts,
        type_model,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        test_run_start=test_run_start,
        test_run_end=test_run_end,
        train_run_start=train_run_start,
        train_run_end=train_run_end,
    )

    fault_free = sampled_train[sampled_train["faultNumber"] == 0].iloc[:, 3:].values

    if use_gpu and GPU_AVAILABLE:
        scaler = cuStandardScaler()
        scaler.fit(cp.asarray(fault_free))
        print(f"[INFO] Using GPU Scaler (cuML), fit on fault-free samples: {fault_free.shape[0]} rows")
    else:
        scaler = skStandardScaler()
        scaler.fit(fault_free)
   #     print(f"[INFO] Using CPU Scaler (scikit-learn), fit on fault-free samples: {fault_free.shape[0]} rows")

    print("[INFO] Scaling and windowing training data...")
    X_train, y_train = scale_and_window(
        sampled_train,
        scaler,
        use_gpu=use_gpu,
        y_col="faultNumber",
        window_size=window_size,
        stride=stride,
    )

    print("[INFO] Scaling and windowing test data...")
    X_test, y_test = scale_and_window(
        sampled_test,
        scaler,
        use_gpu=use_gpu,
        y_col="faultNumber",
        window_size=window_size,
        stride=stride,
    )

    return X_train, X_test, y_train, y_test

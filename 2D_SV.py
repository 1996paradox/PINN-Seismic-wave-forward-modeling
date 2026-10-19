# -*- coding: utf-8 -*-

import sys
import math
import time
import random
import atexit
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline

import torch
import torch.nn as nn
from torch.autograd import grad


# ============================================================
# 1. 随机种子、设备和输出目录
# ============================================================

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR
FEM_DIR = SCRIPT_DIR / "FEM_date"

INPUT_WAVE_FILE = INPUT_DIR / "input.txt"

# 显式场诊断输出目录
EXPLICIT_DIAGNOSTIC_DIR = (
    SCRIPT_DIR / "显示场诊断-2"
)

EXPLICIT_PARAMETER_FILE = (
    EXPLICIT_DIAGNOSTIC_DIR
    / "best_explicit_parameters.csv"
)

RUN_TIME = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / f"results_final_{RUN_TIME}"
)

SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"
ERROR_DIR = OUTPUT_DIR / "errors"
COMPONENT_DIR = OUTPUT_DIR / "components"
DIAGNOSTIC_DIR = OUTPUT_DIR / "diagnostics"

for directory in [
    OUTPUT_DIR,
    SNAPSHOT_DIR,
    ERROR_DIR,
    COMPONENT_DIR,
    DIAGNOSTIC_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

LOG_FILE = (
    OUTPUT_DIR
    / f"training_log_{RUN_TIME}.txt"
)

STAGE1_LOSS_FILE = (
    OUTPUT_DIR
    / "stage1_loss.txt"
)

STAGE2_LOSS_FILE = (
    OUTPUT_DIR
    / "stage2_loss.txt"
)

class Tee:
    def __init__(
        self,
        terminal,
        file_object
    ):
        self.terminal = terminal
        self.file_object = file_object

    def write(self, message):
        self.terminal.write(message)
        self.file_object.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.file_object.flush()

    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()


_original_stdout = sys.stdout
_original_stderr = sys.stderr

log_file_handle = open(
    LOG_FILE,
    "w",
    encoding="utf-8"
)

sys.stdout = Tee(
    _original_stdout,
    log_file_handle
)

sys.stderr = Tee(
    _original_stderr,
    log_file_handle
)


def close_log():
    global log_file_handle

    try:
        sys.stdout = _original_stdout
        sys.stderr = _original_stderr

        if not log_file_handle.closed:
            log_file_handle.flush()
            log_file_handle.close()

    except Exception:
        pass


atexit.register(close_log)


print("=" * 78)
print("二维 P-SV 波 PINN：诊断显式场 + PINN 修正场")
print("=" * 78)
print(f"Device                    : {DEVICE}")
print(f"Input directory           : {INPUT_DIR}")
print(f"Input wave file           : {INPUT_WAVE_FILE}")
print(f"FEM directory             : {FEM_DIR}")
print(f"Explicit diagnostic dir   : {EXPLICIT_DIAGNOSTIC_DIR}")
print(f"Explicit parameter file   : {EXPLICIT_PARAMETER_FILE}")
print(f"Output directory          : {OUTPUT_DIR}")
print(f"Log file                  : {LOG_FILE}")
print("=" * 78)


# ============================================================
# 2. 几何、时间、材料和训练参数
# ============================================================

LX = 2000.0
LZ = 1000.0

TT = 0.05
TMAX = 2.0

X0_OBS = 1000.0
Z0_OBS = 0.0

RHO = 2500.0
VS = 2000.0
NU = 0.3

VP = VS * math.sqrt(
    2.0 * (1.0 - NU)
    / (1.0 - 2.0 * NU)
)

MU = RHO * VS ** 2

LAMBDA = (
    RHO * VP ** 2
    - 2.0 * MU
)

CRITICAL_ANGLE = math.asin(
    VS / VP
)

CRITICAL_ANGLE_DEG = math.degrees(
    CRITICAL_ANGLE
)

# 网络
HIDDEN_WIDTH = 64
HIDDEN_LAYERS = 5

FOURIER_FREQUENCIES = [
    1.0,
    2.0,
    4.0,
    8.0,
    16.0
]

# 训练轮数
N_EPOCHS_IC_PRETRAIN = 3000
N_EPOCHS_PHYSICS = 2000

N_IC_BATCH_PER_TIME = 2048
N_VELOCITY_BATCH = 2048
N_R_BATCH = 10000
N_BC_BATCH = 1500

LEARNING_RATE_PRETRAIN = 1.0e-3
LEARNING_RATE_PHYSICS = 3.0e-4

PRINT_EVERY = 100

# 损失权重
LAMBDA_IC = 1.0
LAMBDA_VELOCITY = 0.20
LAMBDA_R = 1.0
LAMBDA_BC = 1.0
LAMBDA_CORRECTION = 1.0e-3

# 第一阶段采用较小边界权重
LAMBDA_BC_PRETRAIN = 0.10

# 梯度裁剪
MAX_GRAD_NORM = 5.0

# 绘图网格
NX_PLOT = 301
NZ_PLOT = 151

# 观测点高分辨率时间步
OBS_OUTPUT_DT = 0.001

print()
print("Basic settings")
print("--------------")
print(f"TT                 = {TT} s")
print(f"TMAX               = {TMAX} s")
print(f"Observation point  = ({X0_OBS}, {Z0_OBS})")
print(f"rho                = {RHO:.8e} kg/m^3")
print(f"Vs                 = {VS:.8e} m/s")
print(f"Vp                 = {VP:.8e} m/s")
print(f"mu                 = {MU:.8e} Pa")
print(f"lambda             = {LAMBDA:.8e} Pa")
print(f"critical angle     = {CRITICAL_ANGLE_DEG:.8f} degree")


# ============================================================
# 3. 数据读取
# ============================================================

def read_snapshot(filename):
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(
            f"Cannot find file: {filename}"
        )

    dataframes = []

    read_options = [
        {
            "sep": ",",
            "engine": "python"
        },
        {
            "sep": r"\s+",
            "engine": "python"
        },
        {
            "sep": ",",
            "engine": "python",
            "header": None
        },
        {
            "sep": r"\s+",
            "engine": "python",
            "header": None
        }
    ]

    for options in read_options:
        try:
            dataframe = pd.read_csv(
                filename,
                **options
            )

            if dataframe.shape[1] >= 3:
                dataframes.append(
                    dataframe
                )

        except Exception:
            pass

    if len(dataframes) == 0:
        raise ValueError(
            f"Cannot parse snapshot file: {filename}"
        )

    best_dataframe = None
    best_valid_count = -1

    for dataframe in dataframes:
        test = dataframe.iloc[:, :3].apply(
            pd.to_numeric,
            errors="coerce"
        )

        valid_count = int(
            test.notna().all(axis=1).sum()
        )

        if valid_count > best_valid_count:
            best_valid_count = valid_count
            best_dataframe = dataframe.copy()

    dataframe = best_dataframe

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    x_column = None
    z_column = None
    u_column = None

    for column in dataframe.columns:
        name = str(column).strip().lower()

        if name == "x":
            x_column = column

        elif name == "z":
            z_column = column

        elif name in [
            "u",
            "ux",
            "uz",
            "disp",
            "displacement",
            "value"
        ]:
            u_column = column

    if x_column is None:
        x_column = dataframe.columns[0]

    if z_column is None:
        z_column = dataframe.columns[1]

    if u_column is None:
        u_column = dataframe.columns[2]

    x = pd.to_numeric(
        dataframe[x_column],
        errors="coerce"
    ).to_numpy()

    z = pd.to_numeric(
        dataframe[z_column],
        errors="coerce"
    ).to_numpy()

    u = pd.to_numeric(
        dataframe[u_column],
        errors="coerce"
    ).to_numpy()

    valid = (
        np.isfinite(x)
        & np.isfinite(z)
        & np.isfinite(u)
    )

    x = x[valid].astype(
        np.float64
    )

    z = z[valid].astype(
        np.float64
    )

    u = u[valid].astype(
        np.float64
    )

    if len(x) == 0:
        raise ValueError(
            f"No valid numerical rows in: {filename}"
        )

    return x, z, u


def load_initial_snapshots():
    snapshots = []

    snapshot_times = [
        0.0,
        TT,
        2.0 * TT
    ]

    for index, time_value in enumerate(
        snapshot_times,
        start=1
    ):
        ux_file = (
            INPUT_DIR
            / f"Ux{index}.txt"
        )

        uz_file = (
            INPUT_DIR
            / f"Uz{index}.txt"
        )

        x_ux, z_ux, ux = read_snapshot(
            ux_file
        )

        x_uz, z_uz, uz = read_snapshot(
            uz_file
        )

        if len(x_ux) != len(x_uz):
            raise ValueError(
                f"Initial snapshot {index}: "
                "Ux/Uz point counts differ."
            )

        if not np.allclose(
            x_ux,
            x_uz,
            atol=1.0e-6,
            rtol=0.0
        ):
            raise ValueError(
                f"Initial snapshot {index}: "
                "x coordinates differ."
            )

        if not np.allclose(
            z_ux,
            z_uz,
            atol=1.0e-6,
            rtol=0.0
        ):
            raise ValueError(
                f"Initial snapshot {index}: "
                "z coordinates differ."
            )

        snapshots.append({
            "index": index,
            "time": float(time_value),
            "x_np": x_ux,
            "z_np": z_ux,
            "ux_np": ux,
            "uz_np": uz
        })

    return snapshots


def load_input_waveform():
    if not INPUT_WAVE_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find input wave file: "
            f"{INPUT_WAVE_FILE}"
        )

    loaded = None

    for delimiter in [
        None,
        ","
    ]:
        try:
            loaded = np.loadtxt(
                INPUT_WAVE_FILE,
                dtype=np.float64,
                delimiter=delimiter
            )
            break

        except Exception:
            pass

    if loaded is None:
        raise ValueError(
            f"Cannot parse input wave file: "
            f"{INPUT_WAVE_FILE}"
        )

    if loaded.ndim == 1:
        loaded = loaded.reshape(
            1,
            -1
        )

    if loaded.shape[1] < 2:
        raise ValueError(
            "input.txt 至少需要两列："
            "时间和位移。"
        )

    time_np = loaded[:, 0]
    displacement_np = loaded[:, 1]

    valid = (
        np.isfinite(time_np)
        & np.isfinite(displacement_np)
    )

    time_np = time_np[valid]
    displacement_np = displacement_np[valid]

    order = np.argsort(
        time_np
    )

    time_np = time_np[order]
    displacement_np = displacement_np[order]

    if len(time_np) < 2:
        raise ValueError(
            "input.txt 至少需要两个有效采样点。"
        )

    if np.any(
        np.diff(time_np) <= 0.0
    ):
        raise ValueError(
            "input.txt 时间必须严格递增。"
        )

    return time_np, displacement_np


IC_SNAPSHOTS = load_initial_snapshots()

U0_TIME_NP, U0_VALUE_NP = (
    load_input_waveform()
)

U0_TIME = torch.tensor(
    U0_TIME_NP,
    dtype=torch.float32,
    device=DEVICE
)

print()
print("Input displacement time history")
print("-------------------------------")
print(f"Input wave file      = {INPUT_WAVE_FILE}")
print(f"Number of samples    = {len(U0_TIME_NP)}")
print(
    f"Input time range     = "
    f"{U0_TIME_NP[0]:.8e} ~ "
    f"{U0_TIME_NP[-1]:.8e} s"
)
print(
    f"Displacement min/max = "
    f"{U0_VALUE_NP.min():.8e}, "
    f"{U0_VALUE_NP.max():.8e} m"
)


# ============================================================
# 4. 读取显式场诊断参数
# ============================================================

def locate_explicit_parameter_file():
    if EXPLICIT_PARAMETER_FILE.exists():
        return EXPLICIT_PARAMETER_FILE

    if EXPLICIT_DIAGNOSTIC_DIR.exists():
        candidates = list(
            EXPLICIT_DIAGNOSTIC_DIR.rglob(
                "best_explicit_parameters.csv"
            )
        )

        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1:
            candidates = sorted(
                candidates,
                key=lambda item:
                item.stat().st_mtime,
                reverse=True
            )

            print(
                "Warning: 显示场诊断-2 中发现多个参数文件，"
                "使用最新文件："
            )
            print(f"  {candidates[0]}")

            return candidates[0]

    raise FileNotFoundError(
        "没有找到显式场诊断参数文件。\n"
        "要求文件为：\n"
        f"{EXPLICIT_PARAMETER_FILE}"
    )


EXPLICIT_PARAMETER_FILE = (
    locate_explicit_parameter_file()
)


def load_explicit_parameters():
    table = pd.read_csv(
        EXPLICIT_PARAMETER_FILE
    )

    if len(table) == 0:
        raise ValueError(
            "显式场诊断参数文件为空。"
        )

    row = table.iloc[0]

    required_columns = [
        "best_alpha_deg",
        "best_amplitude_scale",
        "best_fem_output_time_offset",
        "effective_input_zero_model_time"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in table.columns
    ]

    if missing_columns:
        raise ValueError(
            "参数文件缺少字段："
            f"{missing_columns}"
        )

    result = {
        "alpha_deg": float(
            row["best_alpha_deg"]
        ),

        "amplitude_scale": float(
            row["best_amplitude_scale"]
        ),

        "fem_output_time_offset": float(
            row["best_fem_output_time_offset"]
        ),

        "effective_input_zero_model_time": float(
            row[
                "effective_input_zero_model_time"
            ]
        ),

        "input_time_zero_physical": float(
            row.get(
                "input_time_zero_physical",
                -0.3
            )
        ),

        "wave_x_ref": float(
            row.get(
                "wave_x_ref",
                0.0
            )
        ),

        "wave_z_ref": float(
            row.get(
                "wave_z_ref",
                LZ
            )
        ),

        "diagnostic_beta_deg": (
            float(row["best_beta_deg"])
            if "best_beta_deg" in table.columns
            else None
        ),

        "diagnostic_a1": (
            float(row["A1"])
            if "A1" in table.columns
            else None
        ),

        "diagnostic_a2": (
            float(row["A2"])
            if "A2" in table.columns
            else None
        )
    }

    return result


EXPLICIT_PARAMETERS = (
    load_explicit_parameters()
)

ALPHA_DEG = EXPLICIT_PARAMETERS[
    "alpha_deg"
]

ALPHA = math.radians(
    ALPHA_DEG
)

EXPLICIT_AMPLITUDE_SCALE = (
    EXPLICIT_PARAMETERS[
        "amplitude_scale"
    ]
)

FEM_OUTPUT_TIME_OFFSET = (
    EXPLICIT_PARAMETERS[
        "fem_output_time_offset"
    ]
)

EFFECTIVE_INPUT_ZERO_MODEL = (
    EXPLICIT_PARAMETERS[
        "effective_input_zero_model_time"
    ]
)

INPUT_TIME_ZERO_PHYSICAL = (
    EXPLICIT_PARAMETERS[
        "input_time_zero_physical"
    ]
)

WAVE_X_REF = EXPLICIT_PARAMETERS[
    "wave_x_ref"
]

WAVE_Z_REF = EXPLICIT_PARAMETERS[
    "wave_z_ref"
]

print()
print("Explicit-field diagnostic parameters")
print("------------------------------------")
print(
    f"Parameter file           = "
    f"{EXPLICIT_PARAMETER_FILE}"
)
print(
    f"alpha                    = "
    f"{ALPHA_DEG:.8f} degree"
)
print(
    f"amplitude scale          = "
    f"{EXPLICIT_AMPLITUDE_SCALE:.8f}"
)
print(
    f"FEM output time offset   = "
    f"{FEM_OUTPUT_TIME_OFFSET:.8f} s"
)
print(
    f"effective input zero     = "
    f"{EFFECTIVE_INPUT_ZERO_MODEL:.8f} s"
)
print(
    f"physical input zero      = "
    f"{INPUT_TIME_ZERO_PHYSICAL:.8f} s"
)
print(
    f"U0 reference point       = "
    f"({WAVE_X_REF:.8f}, "
    f"{WAVE_Z_REF:.8f})"
)


# ============================================================
# 5. 反射参数
# ============================================================

SIN_BETA = (
    VP / VS
    * math.sin(ALPHA)
)

if SIN_BETA >= 1.0:
    raise ValueError(
        "入射角超过 SV-P 临界角，"
        "反射 P 波为倏逝波。"
    )

BETA = math.asin(
    SIN_BETA
)

BETA_DEG = math.degrees(
    BETA
)

DENOMINATOR = (
    VS ** 2
    * math.sin(2.0 * ALPHA)
    * math.sin(2.0 * BETA)
    + VP ** 2
    * math.cos(2.0 * ALPHA) ** 2
)

A1_CALCULATED = (
    VS ** 2
    * math.sin(2.0 * ALPHA)
    * math.sin(2.0 * BETA)
    - VP ** 2
    * math.cos(2.0 * ALPHA) ** 2
) / DENOMINATOR

A2_CALCULATED = (
    2.0
    * VP
    * VS
    * math.sin(2.0 * ALPHA)
    * math.cos(2.0 * ALPHA)
) / DENOMINATOR

# 优先使用诊断文件中的反射系数，
# 保证和显式场诊断完全一致。
A1 = (
    EXPLICIT_PARAMETERS["diagnostic_a1"]
    if EXPLICIT_PARAMETERS["diagnostic_a1"]
    is not None
    else A1_CALCULATED
)

A2 = (
    EXPLICIT_PARAMETERS["diagnostic_a2"]
    if EXPLICIT_PARAMETERS["diagnostic_a2"]
    is not None
    else A2_CALCULATED
)

diagnostic_beta = EXPLICIT_PARAMETERS[
    "diagnostic_beta_deg"
]

if diagnostic_beta is not None:
    if not math.isclose(
        BETA_DEG,
        diagnostic_beta,
        abs_tol=1.0e-4,
        rel_tol=0.0
    ):
        print(
            "Warning: 重新计算的 beta 与诊断 beta 不一致。"
        )
        print(
            f"Calculated beta = {BETA_DEG:.8f}"
        )
        print(
            f"Diagnostic beta = {diagnostic_beta:.8f}"
        )

if not math.isclose(
    A1,
    A1_CALCULATED,
    abs_tol=1.0e-6,
    rel_tol=0.0
):
    print(
        "Warning: 使用诊断文件中的 A1，"
        "它与当前材料重新计算值略有不同。"
    )

if not math.isclose(
    A2,
    A2_CALCULATED,
    abs_tol=1.0e-6,
    rel_tol=0.0
):
    print(
        "Warning: 使用诊断文件中的 A2，"
        "它与当前材料重新计算值略有不同。"
    )

print()
print("Material and reflection parameters")
print("----------------------------------")
print(f"rho             = {RHO:.8e} kg/m^3")
print(f"Vs              = {VS:.8e} m/s")
print(f"Vp              = {VP:.8e} m/s")
print(f"mu              = {MU:.8e} Pa")
print(f"lambda          = {LAMBDA:.8e} Pa")
print(f"alpha           = {ALPHA_DEG:.8f} degree")
print(f"beta            = {BETA_DEG:.8f} degree")
print(f"critical angle  = {CRITICAL_ANGLE_DEG:.8f} degree")
print(f"A1              = {A1:.8f}")
print(f"A2              = {A2:.8f}")


# ============================================================
# 6. 位移尺度和三次样条
# ============================================================

all_ux = np.concatenate([
    snapshot["ux_np"]
    for snapshot in IC_SNAPSHOTS
])

all_uz = np.concatenate([
    snapshot["uz_np"]
    for snapshot in IC_SNAPSHOTS
])

all_magnitude = np.sqrt(
    all_ux ** 2
    + all_uz ** 2
)

explicit_input_amplitude = (
    abs(EXPLICIT_AMPLITUDE_SCALE)
    * float(
        np.max(
            np.abs(U0_VALUE_NP)
        )
    )
)

DISPLACEMENT_SCALE = max(
    float(np.max(all_magnitude)),
    explicit_input_amplitude,
    1.0e-12
)

INCIDENT_AMPLITUDE = (
    DISPLACEMENT_SCALE
)

TIME_SCALE = max(
    float(TT),
    1.0e-12
)

CHARACTERISTIC_LENGTH = max(
    VS * TIME_SCALE,
    1.0
)

if (
    abs(U0_VALUE_NP[0])
    > 1.0e-4
    * np.max(np.abs(U0_VALUE_NP))
    or abs(U0_VALUE_NP[-1])
    > 1.0e-4
    * np.max(np.abs(U0_VALUE_NP))
):
    print()
    print(
        "Warning: 输入时程两端不接近零，"
        "范围外置零可能造成边界导数不连续。"
    )


U0_SPLINE = CubicSpline(
    U0_TIME_NP,
    U0_VALUE_NP,
    bc_type="natural",
    extrapolate=False
)

U0_SPLINE_C0 = torch.tensor(
    U0_SPLINE.c[0],
    dtype=torch.float32,
    device=DEVICE
)

U0_SPLINE_C1 = torch.tensor(
    U0_SPLINE.c[1],
    dtype=torch.float32,
    device=DEVICE
)

U0_SPLINE_C2 = torch.tensor(
    U0_SPLINE.c[2],
    dtype=torch.float32,
    device=DEVICE
)

U0_SPLINE_C3 = torch.tensor(
    U0_SPLINE.c[3],
    dtype=torch.float32,
    device=DEVICE
)


def interpolate_u0(s):
    """
    C2 连续三次样条插值。

    输入时程范围之外返回零。
    """

    original_shape = s.shape
    s_flat = s.reshape(-1)

    number_of_samples = (
        U0_TIME.shape[0]
    )

    index = torch.searchsorted(
        U0_TIME,
        s_flat,
        right=True
    ) - 1

    index = torch.clamp(
        index,
        0,
        number_of_samples - 2
    )

    ds = (
        s_flat
        - U0_TIME[index]
    )

    value_inside = (
        (
            (
                U0_SPLINE_C0[index]
                * ds
                + U0_SPLINE_C1[index]
            )
            * ds
            + U0_SPLINE_C2[index]
        )
        * ds
        + U0_SPLINE_C3[index]
    )

    inside = (
        (s_flat >= U0_TIME[0])
        & (s_flat <= U0_TIME[-1])
    )

    value = torch.where(
        inside,
        value_inside,
        torch.zeros_like(
            value_inside
        )
    )

    return value.reshape(
        original_shape
    )

print()
print("Normalization scales")
print("--------------------")
print(
    f"Displacement scale    = "
    f"{DISPLACEMENT_SCALE:.8e} m"
)
print(
    f"Time scale            = "
    f"{TIME_SCALE:.8e} s"
)
print(
    f"Characteristic length = "
    f"{CHARACTERISTIC_LENGTH:.8e} m"
)


# ============================================================
# 7. 转换早期波场数据为 Torch
# ============================================================

for snapshot in IC_SNAPSHOTS:
    snapshot["x"] = torch.tensor(
        snapshot["x_np"].reshape(-1, 1),
        dtype=torch.float32,
        device=DEVICE
    )

    snapshot["z"] = torch.tensor(
        snapshot["z_np"].reshape(-1, 1),
        dtype=torch.float32,
        device=DEVICE
    )

    snapshot["t"] = torch.full_like(
        snapshot["x"],
        float(snapshot["time"])
    )

    snapshot["target"] = torch.tensor(
        np.column_stack([
            snapshot["ux_np"],
            snapshot["uz_np"]
        ]),
        dtype=torch.float32,
        device=DEVICE
    )


# 检查三个快照网格是否一致
for index in [
    1,
    2
]:
    if not np.allclose(
        IC_SNAPSHOTS[0]["x_np"],
        IC_SNAPSHOTS[index]["x_np"],
        atol=1.0e-6,
        rtol=0.0
    ):
        raise ValueError(
            "三个早期快照的 x 网格不同，"
            "无法计算中心差分速度。"
        )

    if not np.allclose(
        IC_SNAPSHOTS[0]["z_np"],
        IC_SNAPSHOTS[index]["z_np"],
        atol=1.0e-6,
        rtol=0.0
    ):
        raise ValueError(
            "三个早期快照的 z 网格不同，"
            "无法计算中心差分速度。"
        )


# 在 t=TT 处构造中心差分速度
VELOCITY_TARGET_NP = np.column_stack([
    (
        IC_SNAPSHOTS[2]["ux_np"]
        - IC_SNAPSHOTS[0]["ux_np"]
    ) / (
        2.0 * TT
    ),

    (
        IC_SNAPSHOTS[2]["uz_np"]
        - IC_SNAPSHOTS[0]["uz_np"]
    ) / (
        2.0 * TT
    )
])

VELOCITY_TARGET = torch.tensor(
    VELOCITY_TARGET_NP,
    dtype=torch.float32,
    device=DEVICE
)

VELOCITY_X = IC_SNAPSHOTS[1]["x"]
VELOCITY_Z = IC_SNAPSHOTS[1]["z"]
VELOCITY_T = IC_SNAPSHOTS[1]["t"]

velocity_magnitude_np = np.sqrt(
    VELOCITY_TARGET_NP[:, 0] ** 2
    + VELOCITY_TARGET_NP[:, 1] ** 2
)

VELOCITY_SCALE = max(
    float(
        np.max(
            velocity_magnitude_np
        )
    ),
    DISPLACEMENT_SCALE / TIME_SCALE,
    1.0e-12
)

print()
print("Early snapshot data")
print("-------------------")
print(
    f"Points per snapshot = "
    f"{len(IC_SNAPSHOTS[0]['x_np'])}"
)
print(
    f"Ux min/max          = "
    f"{all_ux.min():.8e}, "
    f"{all_ux.max():.8e}"
)
print(
    f"Uz min/max          = "
    f"{all_uz.min():.8e}, "
    f"{all_uz.max():.8e}"
)
print(
    f"Velocity scale      = "
    f"{VELOCITY_SCALE:.8e} m/s"
)


# ============================================================
# 8. Fourier-feature MLP
# ============================================================

class FourierFeatureMLP(nn.Module):
    def __init__(
        self,
        width,
        depth,
        frequencies
    ):
        super().__init__()

        self.frequencies = frequencies

        input_dimension = (
            3
            + 2 * 3 * len(frequencies)
        )

        layers = [
            nn.Linear(
                input_dimension,
                width
            ),
            nn.Tanh()
        ]

        for _ in range(
            depth - 1
        ):
            layers.extend([
                nn.Linear(
                    width,
                    width
                ),
                nn.Tanh()
            ])

        layers.append(
            nn.Linear(
                width,
                2
            )
        )

        self.net = nn.Sequential(
            *layers
        )

        self.apply(
            self.initialize_weights
        )

    @staticmethod
    def initialize_weights(module):
        if isinstance(
            module,
            nn.Linear
        ):
            nn.init.xavier_normal_(
                module.weight
            )

            nn.init.zeros_(
                module.bias
            )

    def feature_map(self, q):
        features = [q]

        for frequency in self.frequencies:
            argument = (
                2.0
                * math.pi
                * frequency
                * q
            )

            features.append(
                torch.sin(argument)
            )

            features.append(
                torch.cos(argument)
            )

        return torch.cat(
            features,
            dim=1
        )

    def forward(self, q):
        return self.net(
            self.feature_map(q)
        )


model = FourierFeatureMLP(
    width=HIDDEN_WIDTH,
    depth=HIDDEN_LAYERS,
    frequencies=FOURIER_FREQUENCIES
).to(DEVICE)


# ============================================================
# 9. 坐标归一化和自动微分
# ============================================================

def normalize_coordinates(
    x,
    z,
    t
):
    return torch.cat([
        2.0 * x / LX - 1.0,
        2.0 * z / LZ - 1.0,
        2.0 * t / TMAX - 1.0
    ], dim=1)


def derivative_tensor(
    y,
    q
):
    return grad(
        outputs=y,
        inputs=q,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=False
    )[0]


# ============================================================
# 10. 显式入射 SV、反射 SV 和反射 P
# ============================================================

def explicit_components(
    x,
    z,
    t
):
    """
    t 使用 FEM/PINN 输出标签时间。

    输入文件时间为：

        t_file = t - EFFECTIVE_INPUT_ZERO_MODEL

    U0 的参考点为：

        (WAVE_X_REF, WAVE_Z_REF)
    """

    sin_alpha = math.sin(
        ALPHA
    )

    cos_alpha = math.cos(
        ALPHA
    )

    sin_beta = math.sin(
        BETA
    )

    cos_beta = math.cos(
        BETA
    )

    dx = (
        x - WAVE_X_REF
    )

    t_file_reference = (
        t - EFFECTIVE_INPUT_ZERO_MODEL
    )

    # 入射 SV
    s_incident = (
        t_file_reference
        - (
            dx * sin_alpha
            + (
                WAVE_Z_REF - z
            ) * cos_alpha
        ) / VS
    )

    # 自由表面公共相位
    surface_phase = (
        t_file_reference
        - WAVE_Z_REF
        * cos_alpha
        / VS
        - dx
        * sin_alpha
        / VS
    )

    # 反射 SV
    s_reflected_sv = (
        surface_phase
        - z
        * cos_alpha
        / VS
    )

    # 反射 P
    s_reflected_p = (
        t_file_reference
        - WAVE_Z_REF
        * cos_alpha
        / VS
        - dx
        * sin_beta
        / VP
        - z
        * cos_beta
        / VP
    )

    u0_incident = interpolate_u0(
        s_incident
    )

    u0_reflected_sv = interpolate_u0(
        s_reflected_sv
    )

    u0_reflected_p = interpolate_u0(
        s_reflected_p
    )

    incident = (
        EXPLICIT_AMPLITUDE_SCALE
        * torch.cat([
            cos_alpha * u0_incident,
            sin_alpha * u0_incident
        ], dim=1)
    )

    reflected_sv = (
        EXPLICIT_AMPLITUDE_SCALE
        * torch.cat([
            -cos_alpha
            * A1
            * u0_reflected_sv,

            sin_alpha
            * A1
            * u0_reflected_sv
        ], dim=1)
    )

    reflected_p = (
        EXPLICIT_AMPLITUDE_SCALE
        * torch.cat([
            sin_beta
            * A2
            * u0_reflected_p,

            cos_beta
            * A2
            * u0_reflected_p
        ], dim=1)
    )

    return (
        incident,
        reflected_sv,
        reflected_p
    )


def explicit_field(
    x,
    z,
    t
):
    incident, reflected_sv, reflected_p = (
        explicit_components(
            x,
            z,
            t
        )
    )

    return (
        incident
        + reflected_sv
        + reflected_p
    )


# ============================================================
# 11. PINN 修正场和总场
# ============================================================

def correction_field(
    x,
    z,
    t
):
    q = normalize_coordinates(
        x,
        z,
        t
    )

    return (
        DISPLACEMENT_SCALE
        * model(q)
    )


def total_displacement_field(
    x,
    z,
    t
):
    """
    总场 = 诊断校准后的显式场 + PINN 修正场。
    """

    return (
        explicit_field(
            x,
            z,
            t
        )
        + correction_field(
            x,
            z,
            t
        )
    )


# ============================================================
# 12. 自动微分和 PDE
# ============================================================

def compute_derivatives(
    field_function,
    x,
    z,
    t
):
    q = torch.cat([
        x.clone().detach(),
        z.clone().detach(),
        t.clone().detach()
    ], dim=1)

    q.requires_grad_(True)

    field = field_function(
        q[:, 0:1],
        q[:, 1:2],
        q[:, 2:3]
    )

    ux = field[:, 0:1]
    uz = field[:, 1:2]

    gradient_ux = derivative_tensor(
        ux,
        q
    )

    gradient_uz = derivative_tensor(
        uz,
        q
    )

    ux_x = gradient_ux[:, 0:1]
    ux_z = gradient_ux[:, 1:2]
    ux_t = gradient_ux[:, 2:3]

    uz_x = gradient_uz[:, 0:1]
    uz_z = gradient_uz[:, 1:2]
    uz_t = gradient_uz[:, 2:3]

    gradient_ux_x = derivative_tensor(
        ux_x,
        q
    )

    gradient_ux_z = derivative_tensor(
        ux_z,
        q
    )

    gradient_ux_t = derivative_tensor(
        ux_t,
        q
    )

    gradient_uz_x = derivative_tensor(
        uz_x,
        q
    )

    gradient_uz_z = derivative_tensor(
        uz_z,
        q
    )

    gradient_uz_t = derivative_tensor(
        uz_t,
        q
    )

    return {
        "ux": ux,
        "uz": uz,

        "ux_x": ux_x,
        "ux_z": ux_z,
        "ux_t": ux_t,

        "uz_x": uz_x,
        "uz_z": uz_z,
        "uz_t": uz_t,

        "ux_xx": (
            gradient_ux_x[:, 0:1]
        ),

        "ux_xz": (
            gradient_ux_x[:, 1:2]
        ),

        "ux_zz": (
            gradient_ux_z[:, 1:2]
        ),

        "ux_tt": (
            gradient_ux_t[:, 2:3]
        ),

        "uz_xx": (
            gradient_uz_x[:, 0:1]
        ),

        "uz_xz": (
            gradient_uz_x[:, 1:2]
        ),

        "uz_zz": (
            gradient_uz_z[:, 1:2]
        ),

        "uz_tt": (
            gradient_uz_t[:, 2:3]
        )
    }


def pde_loss(
    x,
    z,
    t
):
    derivatives = compute_derivatives(
        total_displacement_field,
        x,
        z,
        t
    )

    residual_x = (
        derivatives["ux_tt"]
        - VP ** 2
        * derivatives["ux_xx"]
        - VS ** 2
        * derivatives["ux_zz"]
        - (
            VP ** 2 - VS ** 2
        )
        * derivatives["uz_xz"]
    )

    residual_z = (
        derivatives["uz_tt"]
        - VS ** 2
        * derivatives["uz_xx"]
        - VP ** 2
        * derivatives["uz_zz"]
        - (
            VP ** 2 - VS ** 2
        )
        * derivatives["ux_xz"]
    )

    residual_scale = (
        DISPLACEMENT_SCALE
        / TIME_SCALE ** 2
    )

    residual_x_nd = (
        residual_x
        / residual_scale
    )

    residual_z_nd = (
        residual_z
        / residual_scale
    )

    loss_x = torch.mean(
        residual_x_nd ** 2
    )

    loss_z = torch.mean(
        residual_z_nd ** 2
    )

    return (
        loss_x + loss_z,
        loss_x,
        loss_z
    )


def boundary_loss(
    x,
    t
):
    z = torch.zeros_like(
        x
    )

    derivatives = compute_derivatives(
        total_displacement_field,
        x,
        z,
        t
    )

    sigma_zz = (
        LAMBDA
        * derivatives["ux_x"]
        + (
            LAMBDA + 2.0 * MU
        )
        * derivatives["uz_z"]
    )

    sigma_xz = (
        MU
        * (
            derivatives["ux_z"]
            + derivatives["uz_x"]
        )
    )

    stress_scale = (
        MU
        * DISPLACEMENT_SCALE
        / CHARACTERISTIC_LENGTH
    )

    sigma_zz_nd = (
        sigma_zz
        / stress_scale
    )

    sigma_xz_nd = (
        sigma_xz
        / stress_scale
    )

    loss_zz = torch.mean(
        sigma_zz_nd ** 2
    )

    loss_xz = torch.mean(
        sigma_xz_nd ** 2
    )

    return (
        loss_zz + loss_xz,
        loss_zz,
        loss_xz
    )


# ============================================================
# 13. 早期波场、速度和正则化损失
# ============================================================

def sample_snapshot_batch(
    snapshot,
    batch_size
):
    number_of_points = (
        snapshot["x"].shape[0]
    )

    if batch_size >= number_of_points:
        index = torch.arange(
            number_of_points,
            device=DEVICE
        )

    else:
        index = torch.randint(
            0,
            number_of_points,
            (batch_size,),
            device=DEVICE
        )

    return (
        snapshot["x"][index],
        snapshot["z"][index],
        snapshot["t"][index],
        snapshot["target"][index]
    )


def early_snapshot_loss():
    loss_total = 0.0
    loss_ux_total = 0.0
    loss_uz_total = 0.0

    for snapshot in IC_SNAPSHOTS:
        x, z, t, target = (
            sample_snapshot_batch(
                snapshot,
                N_IC_BATCH_PER_TIME
            )
        )

        prediction = (
            total_displacement_field(
                x,
                z,
                t
            )
        )

        prediction_nd = (
            prediction
            / DISPLACEMENT_SCALE
        )

        target_nd = (
            target
            / DISPLACEMENT_SCALE
        )

        loss_ux = torch.mean(
            (
                prediction_nd[:, 0]
                - target_nd[:, 0]
            ) ** 2
        )

        loss_uz = torch.mean(
            (
                prediction_nd[:, 1]
                - target_nd[:, 1]
            ) ** 2
        )

        loss_total = (
            loss_total
            + loss_ux
            + loss_uz
        )

        loss_ux_total = (
            loss_ux_total
            + loss_ux
        )

        loss_uz_total = (
            loss_uz_total
            + loss_uz
        )

    number_of_snapshots = float(
        len(IC_SNAPSHOTS)
    )

    return (
        loss_total
        / number_of_snapshots,

        loss_ux_total
        / number_of_snapshots,

        loss_uz_total
        / number_of_snapshots
    )


def sample_velocity_batch(
    batch_size
):
    number_of_points = (
        VELOCITY_X.shape[0]
    )

    if batch_size >= number_of_points:
        index = torch.arange(
            number_of_points,
            device=DEVICE
        )

    else:
        index = torch.randint(
            0,
            number_of_points,
            (batch_size,),
            device=DEVICE
        )

    return (
        VELOCITY_X[index],
        VELOCITY_Z[index],
        VELOCITY_T[index],
        VELOCITY_TARGET[index]
    )


def initial_velocity_loss():
    x, z, t, target_velocity = (
        sample_velocity_batch(
            N_VELOCITY_BATCH
        )
    )

    q = torch.cat([
        x.clone().detach(),
        z.clone().detach(),
        t.clone().detach()
    ], dim=1)

    q.requires_grad_(True)

    prediction = (
        total_displacement_field(
            q[:, 0:1],
            q[:, 1:2],
            q[:, 2:3]
        )
    )

    gradient_ux = derivative_tensor(
        prediction[:, 0:1],
        q
    )

    gradient_uz = derivative_tensor(
        prediction[:, 1:2],
        q
    )

    predicted_velocity = torch.cat([
        gradient_ux[:, 2:3],
        gradient_uz[:, 2:3]
    ], dim=1)

    predicted_velocity_nd = (
        predicted_velocity
        / VELOCITY_SCALE
    )

    target_velocity_nd = (
        target_velocity
        / VELOCITY_SCALE
    )

    loss_ux_t = torch.mean(
        (
            predicted_velocity_nd[:, 0]
            - target_velocity_nd[:, 0]
        ) ** 2
    )

    loss_uz_t = torch.mean(
        (
            predicted_velocity_nd[:, 1]
            - target_velocity_nd[:, 1]
        ) ** 2
    )

    return (
        loss_ux_t + loss_uz_t,
        loss_ux_t,
        loss_uz_t
    )


def correction_regularization(
    x,
    z,
    t
):
    correction = correction_field(
        x,
        z,
        t
    )

    return torch.mean(
        (
            correction
            / DISPLACEMENT_SCALE
        ) ** 2
    )


def sample_pde_points(
    number_of_points
):
    x = (
        torch.rand(
            number_of_points,
            1,
            device=DEVICE
        )
        * LX
    )

    z = (
        torch.rand(
            number_of_points,
            1,
            device=DEVICE
        )
        * LZ
    )

    t = (
        torch.rand(
            number_of_points,
            1,
            device=DEVICE
        )
        * TMAX
    )

    return x, z, t


def sample_boundary_points(
    number_of_points
):
    x = (
        torch.rand(
            number_of_points,
            1,
            device=DEVICE
        )
        * LX
    )

    t = (
        torch.rand(
            number_of_points,
            1,
            device=DEVICE
        )
        * TMAX
    )

    return x, t


# ============================================================
# 14. 训练前显式场诊断
# ============================================================

def relative_l2_torch(
    prediction,
    target
):
    return (
        torch.linalg.norm(
            prediction - target
        )
        / torch.clamp(
            torch.linalg.norm(target),
            min=1.0e-30
        )
    )


def explicit_before_training_diagnostic():
    records = []

    print()
    print("Explicit-field diagnostic before training")
    print("-----------------------------------------")

    for snapshot in IC_SNAPSHOTS:
        with torch.no_grad():
            prediction = explicit_field(
                snapshot["x"],
                snapshot["z"],
                snapshot["t"]
            )

        target = snapshot["target"]

        ux_l2 = relative_l2_torch(
            prediction[:, 0],
            target[:, 0]
        )

        uz_l2 = relative_l2_torch(
            prediction[:, 1],
            target[:, 1]
        )

        vector_l2 = relative_l2_torch(
            prediction,
            target
        )

        records.append({
            "time": snapshot["time"],
            "Ux_relative_L2": ux_l2.item(),
            "Uz_relative_L2": uz_l2.item(),
            "Vector_relative_L2": (
                vector_l2.item()
            )
        })

        print(
            f"t={snapshot['time']:.4f} s | "
            f"Ux L2={ux_l2.item():.6e} | "
            f"Uz L2={uz_l2.item():.6e} | "
            f"Vector L2="
            f"{vector_l2.item():.6e}"
        )

    table = pd.DataFrame(
        records
    )

    table.to_csv(
        DIAGNOSTIC_DIR
        / "explicit_before_training.csv",
        index=False,
        encoding="utf-8-sig"
    )


explicit_before_training_diagnostic()


# ============================================================
# 15. 训练历史
# ============================================================

history = {
    "stage": [],
    "total": [],

    "snapshot": [],
    "snapshot_ux": [],
    "snapshot_uz": [],

    "velocity": [],
    "velocity_ux": [],
    "velocity_uz": [],

    "pde": [],
    "pde_x": [],
    "pde_z": [],

    "bc": [],
    "bc_zz": [],
    "bc_xz": [],

    "correction": []
}


def record_history(
    stage,
    total,
    snapshot,
    snapshot_ux,
    snapshot_uz,
    velocity,
    velocity_ux,
    velocity_uz,
    pde,
    pde_x,
    pde_z,
    bc,
    bc_zz,
    bc_xz,
    correction
):
    history["stage"].append(
        int(stage)
    )

    history["total"].append(
        float(total)
    )

    history["snapshot"].append(
        float(snapshot)
    )

    history["snapshot_ux"].append(
        float(snapshot_ux)
    )

    history["snapshot_uz"].append(
        float(snapshot_uz)
    )

    history["velocity"].append(
        float(velocity)
    )

    history["velocity_ux"].append(
        float(velocity_ux)
    )

    history["velocity_uz"].append(
        float(velocity_uz)
    )

    history["pde"].append(
        float(pde)
    )

    history["pde_x"].append(
        float(pde_x)
    )

    history["pde_z"].append(
        float(pde_z)
    )

    history["bc"].append(
        float(bc)
    )

    history["bc_zz"].append(
        float(bc_zz)
    )

    history["bc_xz"].append(
        float(bc_xz)
    )

    history["correction"].append(
        float(correction)
    )

def save_stage1_loss_file(
    filename
):
    """
    保存第一阶段每一步的 loss。

    列：
        iteration, Lic, Lpde, Lcorr

    第一阶段未计算 PDE loss，因此 Lpde 记为 0。
    """

    stage_array = np.asarray(
        history["stage"],
        dtype=np.int64
    )

    snapshot_array = np.asarray(
        history["snapshot"],
        dtype=np.float64
    )

    velocity_array = np.asarray(
        history["velocity"],
        dtype=np.float64
    )

    correction_array = np.asarray(
        history["correction"],
        dtype=np.float64
    )

    selected_indices = np.where(
        stage_array == 1
    )[0]

    if len(selected_indices) != N_EPOCHS_IC_PRETRAIN:
        raise ValueError(
            "Stage 1 loss record count mismatch: "
            f"expected {N_EPOCHS_IC_PRETRAIN}, "
            f"got {len(selected_indices)}."
        )

    iteration = np.arange(
        1,
        len(selected_indices) + 1,
        dtype=np.int64
    )

    loss_ic = (
        LAMBDA_IC
        * snapshot_array[selected_indices]
        + LAMBDA_VELOCITY
        * velocity_array[selected_indices]
    )

    # 第一阶段没有计算 PDE loss
    loss_pde = np.zeros(
        len(selected_indices),
        dtype=np.float64
    )

    loss_correction = (
        correction_array[selected_indices]
    )

    loss_table = np.column_stack([
        iteration,
        loss_ic,
        loss_pde,
        loss_correction
    ])

    np.savetxt(
        filename,
        loss_table,
        delimiter=" ",
        header="iteration Lic Lpde Lcorr",
        comments="",
        fmt=[
            "%d",
            "%.12e",
            "%.12e",
            "%.12e"
        ]
    )

    print(
        f"Stage 1 loss file saved: "
        f"{filename}"
    )


def save_stage2_loss_file(
    filename
):
    """
    保存第二阶段每一步的 loss。

    列：
        iteration, Lic, Lpde, Lcorr, Ltotal
    """

    stage_array = np.asarray(
        history["stage"],
        dtype=np.int64
    )

    total_array = np.asarray(
        history["total"],
        dtype=np.float64
    )

    snapshot_array = np.asarray(
        history["snapshot"],
        dtype=np.float64
    )

    velocity_array = np.asarray(
        history["velocity"],
        dtype=np.float64
    )

    pde_array = np.asarray(
        history["pde"],
        dtype=np.float64
    )

    correction_array = np.asarray(
        history["correction"],
        dtype=np.float64
    )

    selected_indices = np.where(
        stage_array == 2
    )[0]

    if len(selected_indices) != N_EPOCHS_PHYSICS:
        raise ValueError(
            "Stage 2 loss record count mismatch: "
            f"expected {N_EPOCHS_PHYSICS}, "
            f"got {len(selected_indices)}."
        )

    iteration = np.arange(
        1,
        len(selected_indices) + 1,
        dtype=np.int64
    )

    loss_ic = (
        LAMBDA_IC
        * snapshot_array[selected_indices]
        + LAMBDA_VELOCITY
        * velocity_array[selected_indices]
    )

    loss_pde = (
        pde_array[selected_indices]
    )

    loss_correction = (
        correction_array[selected_indices]
    )

    # 这是第二阶段反向传播实际使用的总损失
    loss_total = (
        total_array[selected_indices]
    )

    loss_table = np.column_stack([
        iteration,
        loss_ic,
        loss_pde,
        loss_correction,
        loss_total
    ])

    np.savetxt(
        filename,
        loss_table,
        delimiter=" ",
        header=(
            "iteration Lic Lpde "
            "Lcorr Ltotal"
        ),
        comments="",
        fmt=[
            "%d",
            "%.12e",
            "%.12e",
            "%.12e",
            "%.12e"
        ]
    )

    print(
        f"Stage 2 loss file saved: "
        f"{filename}"
    )




# ============================================================
# 16. 第一阶段：早期波场预训练
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE_PRETRAIN
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=1000,
    gamma=0.7
)

print()
print("Stage 1: early-snapshot pretraining")
print("-----------------------------------")

training_start_timestamp = datetime.now()
training_start_counter = time.perf_counter()

for epoch in range(
    1,
    N_EPOCHS_IC_PRETRAIN + 1
):
    optimizer.zero_grad()

    (
        loss_snapshot,
        loss_snapshot_ux,
        loss_snapshot_uz
    ) = early_snapshot_loss()

    (
        loss_velocity,
        loss_velocity_ux,
        loss_velocity_uz
    ) = initial_velocity_loss()

    xr, zr, tr = sample_pde_points(
        1500
    )

    loss_correction = (
        correction_regularization(
            xr,
            zr,
            tr
        )
    )

    xb, tb = sample_boundary_points(
        N_BC_BATCH
    )

    (
        loss_bc,
        loss_bc_zz,
        loss_bc_xz
    ) = boundary_loss(
        xb,
        tb
    )

    loss_total = (
        loss_snapshot
        + LAMBDA_VELOCITY
        * loss_velocity
        + LAMBDA_BC_PRETRAIN
        * loss_bc
        + LAMBDA_CORRECTION
        * loss_correction
    )

    loss_total.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=MAX_GRAD_NORM
    )

    optimizer.step()
    scheduler.step()

    record_history(
        stage=1,

        total=loss_total.detach().item(),

        snapshot=(
            loss_snapshot.detach().item()
        ),

        snapshot_ux=(
            loss_snapshot_ux.detach().item()
        ),

        snapshot_uz=(
            loss_snapshot_uz.detach().item()
        ),

        velocity=(
            loss_velocity.detach().item()
        ),

        velocity_ux=(
            loss_velocity_ux.detach().item()
        ),

        velocity_uz=(
            loss_velocity_uz.detach().item()
        ),

        pde=1.0e-30,
        pde_x=1.0e-30,
        pde_z=1.0e-30,

        bc=loss_bc.detach().item(),
        bc_zz=loss_bc_zz.detach().item(),
        bc_xz=loss_bc_xz.detach().item(),

        correction=(
            loss_correction.detach().item()
        )
    )

    if (
        epoch == 1
        or epoch % PRINT_EVERY == 0
    ):
        print(
            f"Pretrain epoch {epoch:6d} | "
            f"Total {loss_total.item():.5e} | "
            f"Snapshot {loss_snapshot.item():.5e} | "
            f"Velocity {loss_velocity.item():.5e} | "
            f"BC {loss_bc.item():.5e} | "
            f"Correction {loss_correction.item():.5e} | "
            f"time "
            f"{time.perf_counter() - training_start_counter:.1f}s"
        )


# ============================================================
# 17. 第二阶段：物理训练
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE_PHYSICS
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3000,
    gamma=0.6
)

print()
print("Stage 2: snapshot + velocity + PDE + BC")
print("---------------------------------------")

for epoch in range(
    1,
    N_EPOCHS_PHYSICS + 1
):
    optimizer.zero_grad()

    (
        loss_snapshot,
        loss_snapshot_ux,
        loss_snapshot_uz
    ) = early_snapshot_loss()

    (
        loss_velocity,
        loss_velocity_ux,
        loss_velocity_uz
    ) = initial_velocity_loss()

    xr, zr, tr = sample_pde_points(
        N_R_BATCH
    )

    (
        loss_pde,
        loss_pde_x,
        loss_pde_z
    ) = pde_loss(
        xr,
        zr,
        tr
    )

    loss_correction = (
        correction_regularization(
            xr,
            zr,
            tr
        )
    )

    xb, tb = sample_boundary_points(
        N_BC_BATCH
    )

    (
        loss_bc,
        loss_bc_zz,
        loss_bc_xz
    ) = boundary_loss(
        xb,
        tb
    )

    loss_total = (
        LAMBDA_IC
        * loss_snapshot
        + LAMBDA_VELOCITY
        * loss_velocity
        + LAMBDA_R
        * loss_pde
        + LAMBDA_BC
        * loss_bc
        + LAMBDA_CORRECTION
        * loss_correction
    )

    loss_total.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=MAX_GRAD_NORM
    )

    optimizer.step()
    scheduler.step()

    record_history(
        stage=2,

        total=loss_total.detach().item(),

        snapshot=(
            loss_snapshot.detach().item()
        ),

        snapshot_ux=(
            loss_snapshot_ux.detach().item()
        ),

        snapshot_uz=(
            loss_snapshot_uz.detach().item()
        ),

        velocity=(
            loss_velocity.detach().item()
        ),

        velocity_ux=(
            loss_velocity_ux.detach().item()
        ),

        velocity_uz=(
            loss_velocity_uz.detach().item()
        ),

        pde=loss_pde.detach().item(),
        pde_x=loss_pde_x.detach().item(),
        pde_z=loss_pde_z.detach().item(),

        bc=loss_bc.detach().item(),
        bc_zz=loss_bc_zz.detach().item(),
        bc_xz=loss_bc_xz.detach().item(),

        correction=(
            loss_correction.detach().item()
        )
    )

    if (
        epoch == 1
        or epoch % PRINT_EVERY == 0
    ):
        print(
            f"Physics epoch {epoch:6d} | "
            f"Total {loss_total.item():.5e} | "
            f"Snapshot {loss_snapshot.item():.5e} | "
            f"Velocity {loss_velocity.item():.5e} | "
            f"PDE {loss_pde.item():.5e} | "
            f"BC {loss_bc.item():.5e} | "
            f"Correction {loss_correction.item():.5e} | "
            f"time "
            f"{time.perf_counter() - training_start_counter:.1f}s"
        )

# ============================================================
# 保存两个阶段的逐步 loss
# ============================================================

save_stage1_loss_file(
    STAGE1_LOSS_FILE
)

save_stage2_loss_file(
    STAGE2_LOSS_FILE
)




# ============================================================
# 训练总时长统计
# ============================================================

training_end_timestamp = datetime.now()
training_total_seconds = (
    time.perf_counter()
    - training_start_counter
)

training_total_hours = (
    training_total_seconds / 3600.0
)

training_total_minutes = (
    training_total_seconds / 60.0
)

TRAINING_TIME_FILE = (
    OUTPUT_DIR
    / "training_duration.txt"
)

with open(
    TRAINING_TIME_FILE,
    "w",
    encoding="utf-8"
) as file:
    file.write(
        "PINN training duration report\n"
    )
    file.write(
        "============================\n"
    )
    file.write(
        f"Training start time : "
        f"{training_start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    file.write(
        f"Training end time   : "
        f"{training_end_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    file.write(
        f"Stage 1 epochs     : "
        f"{N_EPOCHS_IC_PRETRAIN}\n"
    )
    file.write(
        f"Stage 2 epochs     : "
        f"{N_EPOCHS_PHYSICS}\n"
    )
    file.write(
        f"Total epochs       : "
        f"{N_EPOCHS_IC_PRETRAIN + N_EPOCHS_PHYSICS}\n"
    )
    file.write(
        f"Training seconds   : "
        f"{training_total_seconds:.6f}\n"
    )
    file.write(
        f"Training minutes   : "
        f"{training_total_minutes:.6f}\n"
    )
    file.write(
        f"Training hours     : "
        f"{training_total_hours:.6f}\n"
    )
    file.write(
        f"Training time      : "
        f"{training_total_seconds // 3600:.0f} h "
        f"{(training_total_seconds % 3600) // 60:.0f} min "
        f"{training_total_seconds % 60:.3f} s\n"
    )

print()
print(
    f"Training duration saved: "
    f"{TRAINING_TIME_FILE}"
)
print(
    f"Total training time: "
    f"{training_total_seconds:.3f} s "
    f"({training_total_hours:.4f} h)"
)








# ============================================================
# 18. 保存模型和训练历史
# ============================================================

MODEL_FILE = (
    OUTPUT_DIR
    / "pinn_psv_diagnostic_explicit.pt"
)

HISTORY_FILE = (
    OUTPUT_DIR
    / "training_history.npz"
)

torch_save_data = {
    "model_state_dict": (
        model.state_dict()
    ),

    "LX": LX,
    "LZ": LZ,
    "TMAX": TMAX,
    "TT": TT,

    "RHO": RHO,
    "VS": VS,
    "VP": VP,
    "NU": NU,
    "MU": MU,
    "LAMBDA": LAMBDA,

    "ALPHA_DEG": ALPHA_DEG,
    "BETA_DEG": BETA_DEG,

    "A1": A1,
    "A2": A2,

    "WAVE_X_REF": WAVE_X_REF,
    "WAVE_Z_REF": WAVE_Z_REF,

    "INPUT_TIME_ZERO_PHYSICAL": (
        INPUT_TIME_ZERO_PHYSICAL
    ),

    "FEM_OUTPUT_TIME_OFFSET": (
        FEM_OUTPUT_TIME_OFFSET
    ),

    "EFFECTIVE_INPUT_ZERO_MODEL": (
        EFFECTIVE_INPUT_ZERO_MODEL
    ),

    "EXPLICIT_AMPLITUDE_SCALE": (
        EXPLICIT_AMPLITUDE_SCALE
    ),

    "DISPLACEMENT_SCALE": (
        DISPLACEMENT_SCALE
    ),

    "VELOCITY_SCALE": (
        VELOCITY_SCALE
    ),

    "TIME_SCALE": TIME_SCALE,

    "FOURIER_FREQUENCIES": (
        FOURIER_FREQUENCIES
    ),

    "HIDDEN_WIDTH": HIDDEN_WIDTH,
    "HIDDEN_LAYERS": HIDDEN_LAYERS,

    "LAMBDA_IC": LAMBDA_IC,
    "LAMBDA_VELOCITY": (
        LAMBDA_VELOCITY
    ),
    "LAMBDA_R": LAMBDA_R,
    "LAMBDA_BC": LAMBDA_BC,
    "LAMBDA_CORRECTION": (
        LAMBDA_CORRECTION
    ),

    "input_wave_file": str(
        INPUT_WAVE_FILE
    ),

    "explicit_parameter_file": str(
        EXPLICIT_PARAMETER_FILE
    ),

    "input_wave_time": (
        U0_TIME_NP
    ),

    "input_wave_displacement": (
        U0_VALUE_NP
    )
}

torch.save(
    torch_save_data,
    MODEL_FILE
)

np.savez(
    HISTORY_FILE,
    **{
        key: np.asarray(value)
        for key, value in history.items()
    }
)

print()
print(f"Model saved   : {MODEL_FILE}")
print(f"History saved : {HISTORY_FILE}")


# ============================================================
# 19. 绘制损失历史
# ============================================================

iterations = np.arange(
    1,
    len(history["total"]) + 1
)

plt.figure(
    figsize=(12, 8)
)

for key, label in [
    ("total", "Total"),
    ("snapshot", "Early snapshots"),
    ("velocity", "Initial velocity"),
    ("pde", "PDE"),
    ("bc", "Free-surface BC"),
    ("correction", "Correction")
]:
    plt.semilogy(
        iterations,
        np.maximum(
            history[key],
            1.0e-30
        ),
        label=label
    )

plt.axvline(
    N_EPOCHS_IC_PRETRAIN,
    color="black",
    linestyle="--",
    label="Physics stage"
)

plt.xlabel(
    "Training iteration"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training losses"
)

plt.grid(
    True,
    which="both",
    alpha=0.3
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "loss_history.png",
    dpi=300
)

plt.close()


plt.figure(
    figsize=(12, 8)
)

for key, label in [
    ("snapshot_ux", "Snapshot ux"),
    ("snapshot_uz", "Snapshot uz"),
    ("velocity_ux", "Velocity ux"),
    ("velocity_uz", "Velocity uz"),
    ("pde_x", "PDE x"),
    ("pde_z", "PDE z"),
    ("bc_zz", "BC sigma_zz"),
    ("bc_xz", "BC sigma_xz")
]:
    plt.semilogy(
        iterations,
        np.maximum(
            history[key],
            1.0e-30
        ),
        label=label
    )

plt.axvline(
    N_EPOCHS_IC_PRETRAIN,
    color="black",
    linestyle="--"
)

plt.xlabel(
    "Training iteration"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Component losses"
)

plt.grid(
    True,
    which="both",
    alpha=0.3
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "component_losses.png",
    dpi=300
)

plt.close()


# ============================================================
# 20. FEM 数据读取
# ============================================================

def load_fem_snapshots():
    if not FEM_DIR.exists():
        raise FileNotFoundError(
            f"FEM folder does not exist: "
            f"{FEM_DIR}"
        )

    ux_files = {}
    uz_files = {}

    for file in FEM_DIR.glob(
        "Ux*.txt"
    ):
        number_string = (
            file.stem[2:]
        )

        if number_string.isdigit():
            ux_files[
                int(number_string)
            ] = file

    for file in FEM_DIR.glob(
        "Uz*.txt"
    ):
        number_string = (
            file.stem[2:]
        )

        if number_string.isdigit():
            uz_files[
                int(number_string)
            ] = file

    common_indices = sorted(
        set(ux_files.keys())
        .intersection(
            set(uz_files.keys())
        )
    )

    if len(common_indices) == 0:
        raise FileNotFoundError(
            "FEM_date 中没有成对的 "
            "Ux/Uz 文件。"
        )

    snapshots = []

    for index in common_indices:
        x_ux, z_ux, ux = read_snapshot(
            ux_files[index]
        )

        x_uz, z_uz, uz = read_snapshot(
            uz_files[index]
        )

        if len(x_ux) != len(x_uz):
            raise ValueError(
                f"FEM snapshot {index}: "
                "point counts differ."
            )

        if not np.allclose(
            x_ux,
            x_uz,
            atol=1.0e-6,
            rtol=0.0
        ):
            raise ValueError(
                f"FEM snapshot {index}: "
                "x coordinates differ."
            )

        if not np.allclose(
            z_ux,
            z_uz,
            atol=1.0e-6,
            rtol=0.0
        ):
            raise ValueError(
                f"FEM snapshot {index}: "
                "z coordinates differ."
            )

        time_value = (
            index - 1
        ) * TT

        snapshots.append({
            "index": index,
            "time": float(time_value),
            "x": x_ux,
            "z": z_ux,
            "ux": ux,
            "uz": uz
        })

    print()
    print("FEM data")
    print("--------")
    print(
        f"Number of snapshots = "
        f"{len(snapshots)}"
    )
    print(
        f"Time range          = "
        f"{snapshots[0]['time']:.6f} ~ "
        f"{snapshots[-1]['time']:.6f} s"
    )

    return snapshots


FEM_SNAPSHOTS = load_fem_snapshots()


# ============================================================
# 21. 预测函数
# ============================================================

def predict_function(
    field_function,
    x_np,
    z_np,
    time_value,
    batch_size=8000
):
    x_np = np.asarray(
        x_np,
        dtype=np.float32
    ).reshape(-1, 1)

    z_np = np.asarray(
        z_np,
        dtype=np.float32
    ).reshape(-1, 1)

    ux_list = []
    uz_list = []

    for start in range(
        0,
        len(x_np),
        batch_size
    ):
        end = min(
            start + batch_size,
            len(x_np)
        )

        x = torch.tensor(
            x_np[start:end],
            dtype=torch.float32,
            device=DEVICE
        )

        z = torch.tensor(
            z_np[start:end],
            dtype=torch.float32,
            device=DEVICE
        )

        t = torch.full_like(
            x,
            float(time_value)
        )

        with torch.no_grad():
            field = field_function(
                x,
                z,
                t
            )

        ux_list.append(
            field[:, 0]
            .cpu()
            .numpy()
        )

        uz_list.append(
            field[:, 1]
            .cpu()
            .numpy()
        )

    return (
        np.concatenate(
            ux_list
        ),
        np.concatenate(
            uz_list
        )
    )


# ============================================================
# 22. FEM 对比和误差计算
# ============================================================

error_records = []

for fem in FEM_SNAPSHOTS:
    time_value = fem["time"]

    ux_pinn, uz_pinn = predict_function(
        total_displacement_field,
        fem["x"],
        fem["z"],
        time_value
    )

    ux_explicit, uz_explicit = (
        predict_function(
            explicit_field,
            fem["x"],
            fem["z"],
            time_value
        )
    )

    ux_correction, uz_correction = (
        predict_function(
            correction_field,
            fem["x"],
            fem["z"],
            time_value
        )
    )

    ux_error = (
        ux_pinn - fem["ux"]
    )

    uz_error = (
        uz_pinn - fem["uz"]
    )

    # 总位移幅值
    fem_displacement = np.sqrt(
        fem["ux"] ** 2
        + fem["uz"] ** 2
    )

    pinn_displacement = np.sqrt(
        ux_pinn ** 2
        + uz_pinn ** 2
    )

    displacement_error = (
        pinn_displacement
        - fem_displacement
    )

    displacement_abs_error = np.abs(
        displacement_error
    )





    ux_abs_error = np.abs(
        ux_error
    )

    uz_abs_error = np.abs(
        uz_error
    )

    ux_l2 = (
        np.linalg.norm(ux_error)
        / max(
            np.linalg.norm(fem["ux"]),
            1.0e-30
        )
    )

    uz_l2 = (
        np.linalg.norm(uz_error)
        / max(
            np.linalg.norm(fem["uz"]),
            1.0e-30
        )
    )

    error_records.append({
        "time": time_value,

        "x": fem["x"],
        "z": fem["z"],

        "ux_pinn": ux_pinn,
        "uz_pinn": uz_pinn,

        "ux_explicit": ux_explicit,
        "uz_explicit": uz_explicit,

        "ux_correction": ux_correction,
        "uz_correction": uz_correction,

        "ux_fem": fem["ux"],
        "uz_fem": fem["uz"],

        "ux_error": ux_error,
        "uz_error": uz_error,

        "ux_abs_error": ux_abs_error,
        "uz_abs_error": uz_abs_error,

        "fem_displacement": fem_displacement,
        "pinn_displacement": pinn_displacement,
        "displacement_error": displacement_error,
        "displacement_abs_error": displacement_abs_error,

        "ux_l2": ux_l2,
        "uz_l2": uz_l2
    })

    print(
        f"FEM comparison "
        f"t={time_value:.2f} s | "
        f"Ux relative L2="
        f"{ux_l2:.6e} | "
        f"Uz relative L2="
        f"{uz_l2:.6e}"
    )


error_table = pd.DataFrame([
    {
        "time": record["time"],

        "Ux_relative_L2": (
            record["ux_l2"]
        ),

        "Uz_relative_L2": (
            record["uz_l2"]
        ),

        "Ux_MAE": np.mean(
            record["ux_abs_error"]
        ),

        "Uz_MAE": np.mean(
            record["uz_abs_error"]
        ),

        "Ux_max_abs_error": np.max(
            record["ux_abs_error"]
        ),

        "Uz_max_abs_error": np.max(
            record["uz_abs_error"]
        ),

        "Ux_correction_RMS": np.sqrt(
            np.mean(
                record["ux_correction"] ** 2
            )
        ),

        "Uz_correction_RMS": np.sqrt(
            np.mean(
                record["uz_correction"] ** 2
            )
        )
    }
    for record in error_records
])

error_table.to_csv(
    OUTPUT_DIR
    / "fem_comparison_error_statistics.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 23. 波场绘图函数
# ============================================================

def plot_field(
    axis,
    x,
    z,
    values,
    title,
    cmap="seismic",
    vmin=None,
    vmax=None
):
    unique_x = np.unique(x)
    unique_z = np.unique(z)

    if (
        len(unique_x)
        * len(unique_z)
        == len(values)
    ):
        try:
            ix = np.searchsorted(
                unique_x,
                x
            )

            iz = np.searchsorted(
                unique_z,
                z
            )

            field = np.full(
                (
                    len(unique_z),
                    len(unique_x)
                ),
                np.nan
            )

            field[iz, ix] = values

            X, Z = np.meshgrid(
                unique_x,
                unique_z
            )

            image = axis.pcolormesh(
                X,
                Z,
                field,
                shading="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )

        except Exception:
            image = axis.scatter(
                x,
                z,
                c=values,
                s=5,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )

    else:
        image = axis.scatter(
            x,
            z,
            c=values,
            s=5,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )

    axis.set_xlabel(
        "x (m)"
    )

    axis.set_ylabel(
        "z (m)"
    )

    axis.set_title(
        title
    )

    axis.invert_yaxis()

    return image


# ============================================================
# 24. FEM 对比图
# ============================================================

for record in error_records:
    time_value = record["time"]
    # 例如：
    # 0.0 s  -> 00p0
    # 0.1 s  -> 00p1
    # 1.2 s  -> 01p2
    time_tag = f"{time_value:04.1f}".replace(
        ".",
        "p"
    )

    ux_scale = max(
        np.max(
            np.abs(record["ux_pinn"])
        ),
        np.max(
            np.abs(record["ux_fem"])
        ),
        1.0e-12
    )

    uz_scale = max(
        np.max(
            np.abs(record["uz_pinn"])
        ),
        np.max(
            np.abs(record["uz_fem"])
        ),
        1.0e-12
    )

    ux_error_scale = max(
        np.max(
            record["ux_abs_error"]
        ),
        1.0e-12
    )

    uz_error_scale = max(
        np.max(
            record["uz_abs_error"]
        ),
        1.0e-12
    )

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(20, 10),
        constrained_layout=True
    )

    fields = [
        (
            record["ux_fem"],
            r"FEM $u_x$",
            -ux_scale,
            ux_scale
        ),
        (
            record["ux_explicit"],
            r"Explicit $u_x$",
            -ux_scale,
            ux_scale
        ),
        (
            record["ux_pinn"],
            r"PINN total $u_x$",
            -ux_scale,
            ux_scale
        ),
        (
            record["ux_error"],
            r"Error $u_x$",
            -ux_error_scale,
            ux_error_scale
        ),
        (
            record["uz_fem"],
            r"FEM $u_z$",
            -uz_scale,
            uz_scale
        ),
        (
            record["uz_explicit"],
            r"Explicit $u_z$",
            -uz_scale,
            uz_scale
        ),
        (
            record["uz_pinn"],
            r"PINN total $u_z$",
            -uz_scale,
            uz_scale
        ),
        (
            record["uz_error"],
            r"Error $u_z$",
            -uz_error_scale,
            uz_error_scale
        )
    ]

    for index, (
        values,
        title,
        vmin,
        vmax
    ) in enumerate(fields):
        row = index // 4
        column = index % 4

        image = plot_field(
            axes[row, column],
            record["x"],
            record["z"],
            values,
            title
            + f", t={time_value:.2f} s",
            cmap="seismic",
            vmin=vmin,
            vmax=vmax
        )

        figure.colorbar(
            image,
            ax=axes[row, column]
        )

    figure.suptitle(
        f"PINN-FEM comparison "
        f"at t={time_value:.2f} s"
    )

    plt.savefig(
        ERROR_DIR
        / f"comparison_t_{time_value:05.2f}.png",
        dpi=220
    )

    plt.close(
        figure
    )
    # --------------------------------------------------------
    # 总位移波场快照
    # --------------------------------------------------------

    total_scale = max(
        np.max(
            record["fem_displacement"]
        ),
        np.max(
            record["pinn_displacement"]
        ),
        1.0e-12
    )

    total_error_scale = max(
        np.max(
            record["displacement_abs_error"]
        ),
        1.0e-12
    )

    total_figure, total_axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.5),
        constrained_layout=True
    )

    fem_image = plot_field(
        total_axes[0],
        record["x"],
        record["z"],
        record["fem_displacement"],
        f"FEM total displacement, "
        f"t={time_value:.2f} s",
        cmap="viridis",
        vmin=0.0,
        vmax=total_scale
    )

    total_figure.colorbar(
        fem_image,
        ax=total_axes[0],
        label="displacement (m)"
    )

    pinn_image = plot_field(
        total_axes[1],
        record["x"],
        record["z"],
        record["pinn_displacement"],
        f"PINN total displacement, "
        f"t={time_value:.2f} s",
        cmap="viridis",
        vmin=0.0,
        vmax=total_scale
    )

    total_figure.colorbar(
        pinn_image,
        ax=total_axes[1],
        label="displacement (m)"
    )

    error_image = plot_field(
        total_axes[2],
        record["x"],
        record["z"],
        record["displacement_error"],
        f"Total displacement error, "
        f"t={time_value:.2f} s",
        cmap="seismic",
        vmin=-total_error_scale,
        vmax=total_error_scale
    )

    total_figure.colorbar(
        error_image,
        ax=total_axes[2],
        label="error (m)"
    )

    total_figure.suptitle(
        f"Total displacement comparison, "
        f"t={time_value:.2f} s"
    )

    total_figure.savefig(
        ERROR_DIR
        / f"total_displacement_t_{time_tag}.png",
        dpi=220
    )

    plt.close(
        total_figure
    )

    # --------------------------------------------------------
    # 输出 FEM 总位移和 PINN 总位移 TXT 文件
    # --------------------------------------------------------

    fem_total_table = np.column_stack([
        record["x"],
        record["z"],
        record["fem_displacement"]
    ])

    pinn_total_table = np.column_stack([
        record["x"],
        record["z"],
        record["pinn_displacement"]
    ])

    np.savetxt(
        ERROR_DIR
        / f"fem_t_{time_tag}.txt",
        fem_total_table,
        delimiter=" ",
        header="x z displacement",
        comments="",
        fmt=[
            "%.10e",
            "%.10e",
            "%.10e"
        ]
    )

    np.savetxt(
        ERROR_DIR
        / f"vsrw-pinn_t_{time_tag}.txt",
        pinn_total_table,
        delimiter=" ",
        header="x z displacement",
        comments="",
        fmt=[
            "%.10e",
            "%.10e",
            "%.10e"
        ]
    )

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    error_table["time"],
    error_table["Ux_relative_L2"],
    "o-",
    label="Ux relative L2"
)

plt.plot(
    error_table["time"],
    error_table["Uz_relative_L2"],
    "s-",
    label="Uz relative L2"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Relative L2 error"
)

plt.title(
    "PINN-FEM relative L2 errors"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "fem_relative_l2_error.png",
    dpi=300
)

plt.close()


# ============================================================
# 25. FEM 观测点时程
# ============================================================

def nearest_index(
    x,
    z,
    x0,
    z0
):
    distance = (
        (x - x0) ** 2
        + (z - z0) ** 2
    )

    return int(
        np.argmin(distance)
    )


observation_records = []

for record in error_records:
    point_index = nearest_index(
        record["x"],
        record["z"],
        X0_OBS,
        Z0_OBS
    )

    observation_records.append({
        "time": record["time"],

        "x_actual": (
            record["x"][point_index]
        ),

        "z_actual": (
            record["z"][point_index]
        ),

        "ux_fem": (
            record["ux_fem"][point_index]
        ),

        "ux_explicit": (
            record["ux_explicit"][point_index]
        ),

        "ux_correction": (
            record["ux_correction"][point_index]
        ),

        "ux_pinn": (
            record["ux_pinn"][point_index]
        ),

        "uz_fem": (
            record["uz_fem"][point_index]
        ),

        "uz_explicit": (
            record["uz_explicit"][point_index]
        ),

        "uz_correction": (
            record["uz_correction"][point_index]
        ),

        "uz_pinn": (
            record["uz_pinn"][point_index]
        )
    })


observation_table = pd.DataFrame(
    observation_records
)

observation_table[
    "ux_error"
] = (
    observation_table["ux_pinn"]
    - observation_table["ux_fem"]
)

observation_table[
    "uz_error"
] = (
    observation_table["uz_pinn"]
    - observation_table["uz_fem"]
)

observation_table.to_csv(
    OUTPUT_DIR
    / "observation_point_timeseries.csv",
    index=False,
    encoding="utf-8-sig"
)


figure, axes = plt.subplots(
    2,
    1,
    figsize=(12, 10),
    sharex=True,
    constrained_layout=True
)

axes[0].plot(
    observation_table["time"],
    observation_table["ux_fem"],
    "k--",
    linewidth=1.4,
    label="FEM ux"
)

axes[0].plot(
    observation_table["time"],
    observation_table["ux_explicit"],
    color="blue",
    linewidth=1.2,
    label="Explicit ux"
)

axes[0].plot(
    observation_table["time"],
    observation_table["ux_pinn"],
    color="red",
    linewidth=1.5,
    label="PINN total ux"
)

axes[0].set_ylabel(
    "ux displacement (m)"
)

axes[0].set_title(
    f"Observation point horizontal displacement "
    f"({X0_OBS}, {Z0_OBS})"
)

axes[0].grid(
    True,
    alpha=0.3
)

axes[0].legend()


axes[1].plot(
    observation_table["time"],
    observation_table["uz_fem"],
    "k--",
    linewidth=1.4,
    label="FEM uz"
)

axes[1].plot(
    observation_table["time"],
    observation_table["uz_explicit"],
    color="blue",
    linewidth=1.2,
    label="Explicit uz"
)

axes[1].plot(
    observation_table["time"],
    observation_table["uz_pinn"],
    color="red",
    linewidth=1.5,
    label="PINN total uz"
)

axes[1].set_xlabel(
    "Time label (s)"
)

axes[1].set_ylabel(
    "uz displacement (m)"
)

axes[1].set_title(
    f"Observation point vertical displacement "
    f"({X0_OBS}, {Z0_OBS})"
)

axes[1].grid(
    True,
    alpha=0.3
)

axes[1].legend()

plt.savefig(
    OUTPUT_DIR
    / "observation_point_timeseries.png",
    dpi=300
)

plt.close(
    figure
)


# ============================================================
# 26. 高时间分辨率观测点时程
# ============================================================

observation_output_times = np.arange(
    0.0,
    TMAX + 0.5 * OBS_OUTPUT_DT,
    OBS_OUTPUT_DT,
    dtype=np.float64
)

observation_output_x = np.full(
    observation_output_times.shape,
    X0_OBS,
    dtype=np.float32
)

observation_output_z = np.full(
    observation_output_times.shape,
    Z0_OBS,
    dtype=np.float32
)

result_lists = {
    "ux_explicit": [],
    "uz_explicit": [],

    "ux_correction": [],
    "uz_correction": [],

    "ux_total": [],
    "uz_total": []
}

batch_size = 8000

for start in range(
    0,
    len(observation_output_times),
    batch_size
):
    end = min(
        start + batch_size,
        len(observation_output_times)
    )

    x = torch.tensor(
        observation_output_x[
            start:end
        ].reshape(-1, 1),
        dtype=torch.float32,
        device=DEVICE
    )

    z = torch.tensor(
        observation_output_z[
            start:end
        ].reshape(-1, 1),
        dtype=torch.float32,
        device=DEVICE
    )

    t = torch.tensor(
        observation_output_times[
            start:end
        ].reshape(-1, 1),
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():
        explicit_displacement = (
            explicit_field(
                x,
                z,
                t
            )
        )

        correction_displacement = (
            correction_field(
                x,
                z,
                t
            )
        )

        total_displacement = (
            explicit_displacement
            + correction_displacement
        )

    result_lists["ux_explicit"].append(
        explicit_displacement[:, 0]
        .cpu()
        .numpy()
    )

    result_lists["uz_explicit"].append(
        explicit_displacement[:, 1]
        .cpu()
        .numpy()
    )

    result_lists["ux_correction"].append(
        correction_displacement[:, 0]
        .cpu()
        .numpy()
    )

    result_lists["uz_correction"].append(
        correction_displacement[:, 1]
        .cpu()
        .numpy()
    )

    result_lists["ux_total"].append(
        total_displacement[:, 0]
        .cpu()
        .numpy()
    )

    result_lists["uz_total"].append(
        total_displacement[:, 1]
        .cpu()
        .numpy()
    )


for key in result_lists:
    result_lists[key] = np.concatenate(
        result_lists[key]
    )


high_resolution_table = pd.DataFrame({
    "time_label": (
        observation_output_times
    ),

    "time_physical": (
        observation_output_times
        + FEM_OUTPUT_TIME_OFFSET
    ),

    "ux_explicit": (
        result_lists["ux_explicit"]
    ),

    "ux_correction": (
        result_lists["ux_correction"]
    ),

    "ux_total": (
        result_lists["ux_total"]
    ),

    "uz_explicit": (
        result_lists["uz_explicit"]
    ),

    "uz_correction": (
        result_lists["uz_correction"]
    ),

    "uz_total": (
        result_lists["uz_total"]
    )
})

high_resolution_table.to_csv(
    OUTPUT_DIR
    / "observation_point_high_resolution_components.csv",
    index=False,
    encoding="utf-8-sig"
)


np.savetxt(
    OUTPUT_DIR
    / "observation_point_ux_timeseries.txt",

    np.column_stack([
        observation_output_times,
        result_lists["ux_total"]
    ]),

    delimiter=",",
    header="Time,Displacement",
    comments="",
    fmt=[
        "%.6f",
        "%.10e"
    ]
)

np.savetxt(
    OUTPUT_DIR
    / "observation_point_uz_timeseries.txt",

    np.column_stack([
        observation_output_times,
        result_lists["uz_total"]
    ]),

    delimiter=",",
    header="Time,Displacement",
    comments="",
    fmt=[
        "%.6f",
        "%.10e"
    ]
)


figure, axes = plt.subplots(
    2,
    1,
    figsize=(12, 10),
    sharex=True,
    constrained_layout=True
)

axes[0].plot(
    observation_output_times,
    result_lists["ux_explicit"],
    color="blue",
    linewidth=1.2,
    label="Explicit ux"
)

axes[0].plot(
    observation_output_times,
    result_lists["ux_correction"],
    color="green",
    linewidth=1.0,
    label="PINN correction ux"
)

axes[0].plot(
    observation_output_times,
    result_lists["ux_total"],
    color="black",
    linewidth=1.8,
    label="Total ux"
)

axes[0].set_ylabel(
    "ux displacement (m)"
)

axes[0].set_title(
    "High-resolution horizontal displacement"
)

axes[0].grid(
    True,
    alpha=0.3
)

axes[0].legend()


axes[1].plot(
    observation_output_times,
    result_lists["uz_explicit"],
    color="blue",
    linewidth=1.2,
    label="Explicit uz"
)

axes[1].plot(
    observation_output_times,
    result_lists["uz_correction"],
    color="green",
    linewidth=1.0,
    label="PINN correction uz"
)

axes[1].plot(
    observation_output_times,
    result_lists["uz_total"],
    color="black",
    linewidth=1.8,
    label="Total uz"
)

axes[1].set_xlabel(
    "Time label (s)"
)

axes[1].set_ylabel(
    "uz displacement (m)"
)

axes[1].set_title(
    "High-resolution vertical displacement"
)

axes[1].grid(
    True,
    alpha=0.3
)

axes[1].legend()

plt.savefig(
    OUTPUT_DIR
    / "observation_high_resolution_components.png",
    dpi=300
)

plt.close(
    figure
)


# ============================================================
# 27. PINN 输出快照
# ============================================================

x_plot = np.linspace(
    0.0,
    LX,
    NX_PLOT
)

z_plot = np.linspace(
    0.0,
    LZ,
    NZ_PLOT
)

X_GRID, Z_GRID = np.meshgrid(
    x_plot,
    z_plot
)

x_flat = X_GRID.reshape(-1)
z_flat = Z_GRID.reshape(-1)

output_times = np.arange(
    0.0,
    TMAX + 0.5 * TT,
    TT
)

ux_snapshots = []
uz_snapshots = []

print()
print("Predicting output snapshots")
print("---------------------------")

for time_value in output_times:
    print(
        f"Predicting t={time_value:.2f} s"
    )

    ux, uz = predict_function(
        total_displacement_field,
        x_flat,
        z_flat,
        time_value
    )

    ux_snapshots.append(
        ux.reshape(
            NZ_PLOT,
            NX_PLOT
        )
    )

    uz_snapshots.append(
        uz.reshape(
            NZ_PLOT,
            NX_PLOT
        )
    )


ux_snapshots = np.asarray(
    ux_snapshots
)

uz_snapshots = np.asarray(
    uz_snapshots
)

common_scale = max(
    np.max(
        np.abs(ux_snapshots)
    ),
    np.max(
        np.abs(uz_snapshots)
    ),
    1.0e-12
)

np.savez(
    OUTPUT_DIR
    / "pinn_wavefield_snapshots.npz",

    times=output_times,
    physical_times=(
        output_times
        + FEM_OUTPUT_TIME_OFFSET
    ),

    x=x_plot,
    z=z_plot,

    ux=ux_snapshots,
    uz=uz_snapshots
)


for index, time_value in enumerate(
    output_times
):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        constrained_layout=True
    )

    image_x = axes[0].pcolormesh(
        X_GRID,
        Z_GRID,
        ux_snapshots[index],
        shading="auto",
        cmap="seismic",
        vmin=-common_scale,
        vmax=common_scale
    )

    axes[0].set_xlabel(
        "x (m)"
    )

    axes[0].set_ylabel(
        "z (m)"
    )

    axes[0].set_title(
        rf"PINN $u_x$, "
        rf"$t={time_value:.2f}\,\mathrm{{s}}$"
    )

    axes[0].invert_yaxis()
    axes[0].set_aspect(
        "equal"
    )

    figure.colorbar(
        image_x,
        ax=axes[0],
        label="ux (m)"
    )

    image_z = axes[1].pcolormesh(
        X_GRID,
        Z_GRID,
        uz_snapshots[index],
        shading="auto",
        cmap="seismic",
        vmin=-common_scale,
        vmax=common_scale
    )

    axes[1].set_xlabel(
        "x (m)"
    )

    axes[1].set_ylabel(
        "z (m)"
    )

    axes[1].set_title(
        rf"PINN $u_z$, "
        rf"$t={time_value:.2f}\,\mathrm{{s}}$"
    )

    axes[1].invert_yaxis()
    axes[1].set_aspect(
        "equal"
    )

    figure.colorbar(
        image_z,
        ax=axes[1],
        label="uz (m)"
    )

    figure.suptitle(
        f"PINN P-SV wavefield, "
        f"time label={time_value:.2f} s"
    )

    plt.savefig(
        SNAPSHOT_DIR
        / f"pinn_t_{time_value:05.2f}.png",
        dpi=220
    )

    plt.close(
        figure
    )


# ============================================================
# 28. 显式分量和 PINN 修正场输出
# ============================================================

component_times = sorted(
    set([
        0.0,
        min(0.2, TMAX),
        min(0.4, TMAX),
        min(0.6, TMAX)
    ])
)

component_functions = [
    (
        "incident_SV",
        lambda x, z, t:
        explicit_components(
            x,
            z,
            t
        )[0]
    ),

    (
        "reflected_SV",
        lambda x, z, t:
        explicit_components(
            x,
            z,
            t
        )[1]
    ),

    (
        "reflected_P",
        lambda x, z, t:
        explicit_components(
            x,
            z,
            t
        )[2]
    ),

    (
        "PINN_correction",
        correction_field
    ),

    (
        "total",
        total_displacement_field
    )
]


for time_value in component_times:
    fields = []
    scale = 1.0e-12

    for name, function in component_functions:
        ux, uz = predict_function(
            function,
            x_flat,
            z_flat,
            time_value
        )

        ux = ux.reshape(
            NZ_PLOT,
            NX_PLOT
        )

        uz = uz.reshape(
            NZ_PLOT,
            NX_PLOT
        )

        fields.append(
            (
                name,
                ux,
                uz
            )
        )

        scale = max(
            scale,
            np.max(np.abs(ux)),
            np.max(np.abs(uz))
        )

    number_of_rows = len(fields)

    figure, axes = plt.subplots(
        number_of_rows,
        2,
        figsize=(
            14,
            4.5 * number_of_rows
        ),
        constrained_layout=True
    )

    for row, (
        name,
        ux,
        uz
    ) in enumerate(fields):
        image_ux = axes[
            row,
            0
        ].pcolormesh(
            X_GRID,
            Z_GRID,
            ux,
            shading="auto",
            cmap="seismic",
            vmin=-scale,
            vmax=scale
        )

        axes[row, 0].set_title(
            f"{name}: ux"
        )

        axes[row, 0].set_xlabel(
            "x (m)"
        )

        axes[row, 0].set_ylabel(
            "z (m)"
        )

        axes[row, 0].invert_yaxis()
        axes[row, 0].set_aspect(
            "equal"
        )

        figure.colorbar(
            image_ux,
            ax=axes[row, 0]
        )

        image_uz = axes[
            row,
            1
        ].pcolormesh(
            X_GRID,
            Z_GRID,
            uz,
            shading="auto",
            cmap="seismic",
            vmin=-scale,
            vmax=scale
        )

        axes[row, 1].set_title(
            f"{name}: uz"
        )

        axes[row, 1].set_xlabel(
            "x (m)"
        )

        axes[row, 1].set_ylabel(
            "z (m)"
        )

        axes[row, 1].invert_yaxis()
        axes[row, 1].set_aspect(
            "equal"
        )

        figure.colorbar(
            image_uz,
            ax=axes[row, 1]
        )

    figure.suptitle(
        f"P-SV component fields, "
        f"time label={time_value:.2f} s"
    )

    plt.savefig(
        COMPONENT_DIR
        / f"components_t_{time_value:05.2f}.png",
        dpi=220
    )

    plt.close(
        figure
    )


# ============================================================
# 29. 最终诊断
# ============================================================

xf, zf, tf = sample_pde_points(
    3000
)

final_pde, final_pde_x, final_pde_z = (
    pde_loss(
        xf,
        zf,
        tf
    )
)

xbf, tbf = sample_boundary_points(
    3000
)

final_bc, final_bc_zz, final_bc_xz = (
    boundary_loss(
        xbf,
        tbf
    )
)

final_correction = (
    correction_regularization(
        xf,
        zf,
        tf
    )
)

print()
print("Final diagnostics")
print("-----------------")
print(
    f"Final PDE total    = "
    f"{final_pde.item():.8e}"
)
print(
    f"Final PDE x        = "
    f"{final_pde_x.item():.8e}"
)
print(
    f"Final PDE z        = "
    f"{final_pde_z.item():.8e}"
)
print(
    f"Final BC total     = "
    f"{final_bc.item():.8e}"
)
print(
    f"Final BC sigma_zz  = "
    f"{final_bc_zz.item():.8e}"
)
print(
    f"Final BC sigma_xz  = "
    f"{final_bc_xz.item():.8e}"
)
print(
    f"Final correction   = "
    f"{final_correction.item():.8e}"
)
print(
    f"Mean Ux relative L2 = "
    f"{error_table['Ux_relative_L2'].mean():.8e}"
)
print(
    f"Mean Uz relative L2 = "
    f"{error_table['Uz_relative_L2'].mean():.8e}"
)

print()
print("=" * 78)
print("All calculations finished.")
print("=" * 78)
print(f"Output directory  : {OUTPUT_DIR}")
print(f"Model file        : {MODEL_FILE}")
print(f"History file      : {HISTORY_FILE}")
print(f"FEM error dir     : {ERROR_DIR}")
print(f"Snapshot dir      : {SNAPSHOT_DIR}")
print(f"Component dir     : {COMPONENT_DIR}")
print(f"Diagnostic dir    : {DIAGNOSTIC_DIR}")
print(f"Log file          : {LOG_FILE}")
print("=" * 78)

close_log()
# -*- coding: utf-8 -*-

"""
基于 T1 加权结构 MRI 的 ASD / HC 多尺度特征优化分类软件

目录结构：

MRI_ASD_Projec/
│
├── excise/
│   ├── train/
│   ├── test/
│   ├── train_labels.csv
│   └── submission_example.csv
│
└── main.py

标签：
1 = ASD
2 = HC
"""

import os
import hashlib
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk

from scipy.ndimage import (
    zoom, gaussian_filter, gaussian_gradient_magnitude,
    binary_closing, binary_fill_holes
)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# 一、项目路径
# ============================================================

# 当前 Python 文件所在目录
BASE_DIR = Path(__file__).resolve().parent

# 数据集目录
DATA_DIR = BASE_DIR / "excise"

# 训练集
TRAIN_DIR = DATA_DIR / "train"

# 测试集
TEST_DIR = DATA_DIR / "test"

# 训练标签
LABEL_FILE = DATA_DIR / "train_labels.csv"

# 测试提交模板
SUBMISSION_TEMPLATE = DATA_DIR / "submission_example.csv"

# 最终结果
OUTPUT_FILE = BASE_DIR / "submission.csv"

# 三模型结果保存目录
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

# 特征缓存目录：N4 / MNI 很耗时，同一预处理配置下避免重复计算
FEATURE_CACHE_DIR = BASE_DIR / ".feature_cache"
FEATURE_CACHE_DIR.mkdir(exist_ok=True)

# 修改特征提取逻辑后请同步修改版本号，以自动失效旧缓存
FEATURE_VERSION = "multiscale_v2"

# 网格搜索并行数。Windows / PyCharm 下 1 最稳；机器内存充足可改为 -1
N_JOBS = 1


# ============================================================
# 二、基础图像处理函数
# ============================================================

def get_subject_id(file_path):
    """
    从文件名：

    sub-001_T1w.nii.gz

    获取：

    sub-001
    """
    name = Path(file_path).name
    return name.replace("_T1w.nii.gz", "")


def foreground_mask(volume):
    """
    构造稳健的前景/脑区近似掩膜。

    目的不是替代专业 skull-stripping，而是：
    1. 排除大面积零背景；
    2. 降低极低强度噪声和边缘伪影的影响；
    3. 为稳健强度标准化、形态统计提供统一 mask。
    """
    volume = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(volume)
    positive = volume[finite & (volume > 0)]

    if positive.size == 0:
        return np.zeros_like(volume, dtype=bool)

    # 采用较低分位阈值保留绝大多数头部/脑组织，同时去掉背景噪声
    threshold = np.percentile(positive, 7.5)
    mask = finite & (volume > threshold)

    # 轻度闭运算 + 填洞，使 mask 更连续
    mask = binary_closing(mask, iterations=1)
    mask = binary_fill_holes(mask)

    return mask.astype(bool)


def intensity_normalization(volume, mask=None):
    """
    稳健 MRI 强度归一化：
        前景 mask -> 百分位裁剪 -> Z-score。

    与直接对所有正值做 Z-score 相比，对极端高亮、扫描强度差异更稳健。
    """
    volume = np.asarray(volume, dtype=np.float32)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)

    if mask is None:
        mask = foreground_mask(volume)

    values = volume[mask]
    if values.size < 32:
        return volume

    low, high = np.percentile(values, [1.0, 99.0])
    clipped = np.clip(volume, low, high)

    values = clipped[mask]
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-8:
        std = 1.0

    out = np.zeros_like(clipped, dtype=np.float32)
    out[mask] = (clipped[mask] - mean) / std
    return out


def crop_brain(volume, mask=None, margin=4):
    """根据前景 mask 裁剪，主要用于未进行 MNI 配准时。"""
    if mask is None:
        mask = foreground_mask(volume)

    coords = np.argwhere(mask)
    if coords.size == 0:
        return volume

    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1

    lo = np.maximum(lo - margin, 0)
    hi = np.minimum(hi + margin, np.array(volume.shape))

    return volume[
        lo[0]:hi[0],
        lo[1]:hi[1],
        lo[2]:hi[2]
    ]


def resize_volume(volume, target_shape=(24, 24, 24), order=1):
    """三维重采样到固定大小。"""
    factors = [
        target_shape[i] / volume.shape[i]
        for i in range(3)
    ]
    return zoom(volume, zoom=factors, order=order).astype(np.float32)


def block_statistics(volume16):
    """
    对 16x16x16 体数据划分 4x4x4 个空间块，
    每块提取 mean/std/mean(abs)，增强对局部结构差异的表达。
    """
    features = []
    block = 4
    for x in range(0, 16, block):
        for y in range(0, 16, block):
            for z in range(0, 16, block):
                patch = volume16[x:x+block, y:y+block, z:z+block]
                features.extend([
                    float(np.mean(patch)),
                    float(np.std(patch)),
                    float(np.mean(np.abs(patch)))
                ])
    return np.asarray(features, dtype=np.float32)


def global_structural_features(volume, mask):
    """提取少量全局强度和形态统计特征。"""
    values = volume[mask]
    if values.size == 0:
        return np.zeros(50, dtype=np.float32)

    # 固定范围的标准化强度直方图
    hist, _ = np.histogram(values, bins=32, range=(-3.0, 3.0))
    hist = hist.astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)

    quantiles = np.percentile(
        values, [1, 5, 10, 25, 50, 75, 90, 95, 99]
    ).astype(np.float32)

    coords = np.argwhere(mask)
    shape = np.asarray(mask.shape, dtype=np.float32)

    centroid = coords.mean(axis=0) / np.maximum(shape - 1.0, 1.0)
    spread = coords.std(axis=0) / np.maximum(shape, 1.0)
    occupancy = np.array([mask.mean()], dtype=np.float32)
    intensity_stats = np.array([
        values.mean(),
        values.std()
    ], dtype=np.float32)

    morphology = np.concatenate([
        centroid.astype(np.float32),
        spread.astype(np.float32),
        occupancy,
        intensity_stats
    ])

    return np.concatenate([hist, quantiles, morphology]).astype(np.float32)


def n4_bias_correction(image):
    """SimpleITK N4 偏置场校正。"""
    image = sitk.Cast(image, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([30, 20, 10])
    return corrector.Execute(image, mask)


def affine_registration_to_mni(moving_image, template_path):
    """将 T1 MRI 仿射配准至 MNI T1 模板。"""
    fixed_image = sitk.ReadImage(str(template_path), sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    fixed_norm = sitk.Normalize(fixed_image)
    moving_norm = sitk.Normalize(moving_image)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_norm,
        moving_norm,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=40)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.02)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=80,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration.Execute(fixed_norm, moving_norm)

    return sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32
    )


def _load_preprocessed_volume(image_path, use_n4=False, template_path=None):
    """读取 MRI 并执行几何/强度预处理，返回标准化体数据及 mask。"""
    if use_n4 or template_path is not None:
        image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)

        if use_n4:
            image = n4_bias_correction(image)

        if template_path is not None:
            image = affine_registration_to_mni(image, template_path)

        volume = sitk.GetArrayFromImage(image)
        volume = np.transpose(volume, (2, 1, 0))
    else:
        nii = nib.load(str(image_path))
        nii = nib.as_closest_canonical(nii)
        volume = nii.get_fdata(dtype=np.float32)

    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    mask = foreground_mask(volume)

    # 未进行 MNI 配准时进行紧凑裁剪；配准后保留 MNI 空间位置关系
    if template_path is None:
        coords = np.argwhere(mask)
        if coords.size:
            lo = np.maximum(coords.min(axis=0) - 4, 0)
            hi = np.minimum(coords.max(axis=0) + 5, np.array(volume.shape))
            volume = volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            mask = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    volume = intensity_normalization(volume, mask)
    # 归一化后脑组织会同时包含正/负 Z-score，不能再次按 >0 重建 mask；
    # 继续使用归一化前得到的前景 mask。
    return volume, mask


def extract_feature(image_path, use_n4=False, template_path=None):
    """
    多尺度结构特征：
      1) 平滑 T1 体素特征 24^3；
      2) 梯度幅值特征 16^3；
      3) 高频残差特征 16^3；
      4) 4x4x4 空间块统计；
      5) 强度直方图 / 分位数 / 形态统计。

    相比单纯 24^3 flatten，保留了更多边缘、局部结构和全局形态信息，
    同时仍由 Pipeline 内的 SelectKBest 在每个训练折独立选特征。
    """
    volume, mask = _load_preprocessed_volume(
        image_path=image_path,
        use_n4=use_n4,
        template_path=template_path
    )

    # 轻度平滑：抑制随机噪声，尽量不过度模糊结构边界
    smooth = gaussian_filter(volume, sigma=0.8)
    smooth *= mask.astype(np.float32)

    # 1. 基础强度体素特征
    intensity_24 = resize_volume(smooth, (24, 24, 24), order=1)

    # 2. 梯度特征
    gradient = gaussian_gradient_magnitude(smooth, sigma=1.0)
    gradient *= mask.astype(np.float32)
    gradient_16 = resize_volume(gradient, (16, 16, 16), order=1)

    # 3. 多尺度高频残差：突出局部形态/纹理变化
    low_frequency = gaussian_filter(smooth, sigma=2.0)
    high_frequency = (smooth - low_frequency) * mask.astype(np.float32)
    high_16 = resize_volume(high_frequency, (16, 16, 16), order=1)

    # 4. 空间块统计
    smooth_16 = resize_volume(smooth, (16, 16, 16), order=1)
    block_feat = block_statistics(smooth_16)

    # 5. 全局统计
    global_feat = global_structural_features(volume, mask)

    feature = np.concatenate([
        intensity_24.ravel(),
        gradient_16.ravel(),
        high_16.ravel(),
        block_feat,
        global_feat
    ])

    return np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _feature_cache_key(image_path, use_n4=False, template_path=None):
    """根据图像、预处理选项、模板和特征版本生成缓存键。"""
    image_path = Path(image_path)
    image_stat = image_path.stat()

    parts = [
        FEATURE_VERSION,
        str(image_path.resolve()),
        str(image_stat.st_size),
        str(image_stat.st_mtime_ns),
        f"n4={int(use_n4)}"
    ]

    if template_path is not None:
        template_path = Path(template_path)
        stat = template_path.stat()
        parts.extend([
            str(template_path.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns)
        ])
    else:
        parts.append("template=None")

    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def extract_feature_cached(image_path, use_n4=False, template_path=None):
    """特征缓存：重复实验时无需再次执行 N4/MNI/多尺度特征提取。"""
    key = _feature_cache_key(image_path, use_n4, template_path)
    cache_file = FEATURE_CACHE_DIR / f"{key}.npy"

    if cache_file.exists():
        return np.load(cache_file, allow_pickle=False).astype(np.float32)

    feature = extract_feature(image_path, use_n4, template_path)
    np.save(cache_file, feature, allow_pickle=False)
    return feature


# ============================================================
# 六、构建多个分类模型
# ============================================================

def build_model_specs(feature_num):
    """
    为三个模型建立 Pipeline 和轻量参数网格。

    特征选择、标准化均在 Pipeline 内部完成，避免交叉验证信息泄漏。
    网格规模控制在小数据集可接受范围，重点搜索 k 与正则化强度。
    """
    k_candidates = [k for k in (200, 500, 1000) if k < feature_num]
    if not k_candidates:
        k_candidates = [max(1, feature_num // 2)]

    common_steps = [
        ("variance", VarianceThreshold(threshold=1e-10)),
        ("feature_selection", SelectKBest(score_func=f_classif, k=k_candidates[0])),
        ("scaler", StandardScaler())
    ]

    linear = Pipeline(common_steps + [
        ("classifier", SVC(
            kernel="linear",
            class_weight="balanced"
        ))
    ])

    logistic = Pipeline(common_steps + [
        ("classifier", LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=42
        ))
    ])

    rbf = Pipeline(common_steps + [
        ("classifier", SVC(
            kernel="rbf",
            class_weight="balanced"
        ))
    ])

    return {
        "Linear SVM": {
            "pipeline": linear,
            "grid": {
                "feature_selection__k": k_candidates,
                "classifier__C": [0.1, 1.0, 10.0]
            }
        },
        "Logistic Regression": {
            "pipeline": logistic,
            "grid": {
                "feature_selection__k": k_candidates,
                "classifier__C": [0.1, 1.0, 10.0]
            }
        },
        "RBF-SVM": {
            "pipeline": rbf,
            "grid": {
                "feature_selection__k": k_candidates[:2],
                "classifier__C": [1.0, 10.0],
                "classifier__gamma": ["scale", 0.001]
            }
        }
    }


def _asd_decision_score(fitted_model, X):
    """将 decision_function 统一转换为“分数越大越倾向 ASD(label=1)”。"""
    score = np.asarray(fitted_model.decision_function(X), dtype=np.float64)
    classes = np.asarray(fitted_model.classes_)

    if classes.size != 2:
        raise ValueError("当前程序仅支持二分类。")

    # sklearn 二分类 decision_function 正方向对应 classes_[1]
    return score if classes[1] == 1 else -score


def nested_evaluate_and_fit(model_name, spec, X, y):
    """
    外层 5 折用于泛化评估，内层 3 折只用于选参数；
    最后再在全部训练集上进行 5 折参数搜索并拟合最终模型。

    这样既尽量提高最终模型表现，又不会把调参结果直接当成独立验证成绩。
    """
    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    y_pred = np.zeros_like(y)
    y_score = np.zeros(len(y), dtype=np.float64)
    fold_details = []

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
        inner_cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=100 + fold
        )

        search = GridSearchCV(
            estimator=spec["pipeline"],
            param_grid=spec["grid"],
            scoring="balanced_accuracy",
            cv=inner_cv,
            n_jobs=N_JOBS,
            refit=True,
            error_score="raise"
        )

        search.fit(X[train_idx], y[train_idx])
        best = search.best_estimator_

        pred = best.predict(X[val_idx])
        score = _asd_decision_score(best, X[val_idx])

        y_pred[val_idx] = pred
        y_score[val_idx] = score

        fold_details.append({
            "fold": fold,
            "best_params": search.best_params_,
            "inner_bacc": float(search.best_score_)
        })

    acc = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    sensitivity = recall_score(y, y_pred, pos_label=1, zero_division=0)
    specificity = recall_score(y, y_pred, pos_label=2, zero_division=0)
    auc = roc_auc_score((y == 1).astype(int), y_score)
    cm = confusion_matrix(y, y_pred, labels=[1, 2])

    # 全训练集上再次搜索参数，得到用于 test 的最终模型
    final_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=2026
    )
    final_search = GridSearchCV(
        estimator=spec["pipeline"],
        param_grid=spec["grid"],
        scoring="balanced_accuracy",
        cv=final_cv,
        n_jobs=N_JOBS,
        refit=True,
        error_score="raise"
    )
    final_search.fit(X, y)

    return {
        "ACC": acc,
        "BACC": bacc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "AUC": auc,
        "CM": cm,
        "FoldDetails": fold_details,
        "BestParams": final_search.best_params_,
        "FinalCVBACC": float(final_search.best_score_),
        "FinalModel": final_search.best_estimator_
    }


# ============================================================
# 六、GUI 主程序
# ============================================================

class MRIClassifierApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "基于 T1 MRI 的 ASD / HC 医学图像分类系统"
        )

        self.root.geometry(
            "1350x850"
        )

        self.root.minsize(
            1100,
            700
        )

        # ----------------------------------------------------
        # 数据路径
        # ----------------------------------------------------

        self.train_dir = TRAIN_DIR
        self.test_dir = TEST_DIR
        self.label_file = LABEL_FILE
        self.submission_template = (
            SUBMISSION_TEMPLATE
        )

        # MNI 模板默认为空
        self.template_path = None

        # ----------------------------------------------------
        # 模型变量
        # ----------------------------------------------------

        # 三个最终分类模型
        self.models = {}

        # 三个模型的交叉验证结果
        self.model_results = {}

        # 训练集交叉验证表现最好的模型名称
        self.best_model_name = None

        # 保留与原代码兼容的单模型变量
        self.model = None

        self.prediction_df = None

        self.current_volume = None

        self.current_file = None

        # ----------------------------------------------------
        # 预处理选项
        # ----------------------------------------------------

        self.use_n4_var = tk.BooleanVar(
            value=False
        )

        self.use_mni_var = tk.BooleanVar(
            value=False
        )

        # 创建界面
        self.create_ui()

        # 检查数据
        self.check_dataset()


    # ========================================================
    # 七、创建 GUI
    # ========================================================

    def create_ui(self):

        # ----------------------------------------------------
        # 左侧控制区域
        # ----------------------------------------------------

        left = ttk.Frame(
            self.root,
            width=280
        )

        left.pack(
            side=tk.LEFT,
            fill=tk.Y,
            padx=10,
            pady=10
        )

        # ----------------------------------------------------
        # 右侧 Notebook
        # ----------------------------------------------------

        self.notebook = ttk.Notebook(
            self.root
        )

        self.notebook.pack(
            side=tk.RIGHT,
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10
        )

        # 三个标签页
        self.viewer_tab = ttk.Frame(
            self.notebook
        )

        self.result_tab = ttk.Frame(
            self.notebook
        )

        self.log_tab = ttk.Frame(
            self.notebook
        )

        self.notebook.add(
            self.viewer_tab,
            text="MRI 影像浏览"
        )

        self.notebook.add(
            self.result_tab,
            text="分类结果"
        )

        self.notebook.add(
            self.log_tab,
            text="运行日志"
        )

        # ====================================================
        # 左侧标题
        # ====================================================

        ttk.Label(
            left,
            text="ASD / HC",
            font=(
                "Microsoft YaHei",
                20,
                "bold"
            )
        ).pack(
            pady=(15, 0)
        )

        ttk.Label(
            left,
            text="医学图像分类系统",
            font=(
                "Microsoft YaHei",
                13
            )
        ).pack(
            pady=(0, 20)
        )

        # ====================================================
        # 数据状态
        # ====================================================

        status_frame = ttk.LabelFrame(
            left,
            text="数据集状态"
        )

        status_frame.pack(
            fill=tk.X,
            pady=5
        )

        self.train_status = ttk.Label(
            status_frame,
            text="训练集：检查中..."
        )

        self.train_status.pack(
            anchor="w",
            padx=10,
            pady=4
        )

        self.test_status = ttk.Label(
            status_frame,
            text="测试集：检查中..."
        )

        self.test_status.pack(
            anchor="w",
            padx=10,
            pady=4
        )

        self.label_status = ttk.Label(
            status_frame,
            text="标签：检查中..."
        )

        self.label_status.pack(
            anchor="w",
            padx=10,
            pady=4
        )

        # ====================================================
        # MRI 浏览
        # ====================================================

        viewer_frame = ttk.LabelFrame(
            left,
            text="医学影像"
        )

        viewer_frame.pack(
            fill=tk.X,
            pady=10
        )

        ttk.Button(
            viewer_frame,
            text="打开 MRI 影像",
            command=self.open_mri
        ).pack(
            fill=tk.X,
            padx=8,
            pady=6
        )

        # ====================================================
        # 预处理设置
        # ====================================================

        preprocessing = ttk.LabelFrame(
            left,
            text="预处理设置"
        )

        preprocessing.pack(
            fill=tk.X,
            pady=10
        )

        ttk.Checkbutton(
            preprocessing,
            text="N4 偏置场校正",
            variable=self.use_n4_var
        ).pack(
            anchor="w",
            padx=10,
            pady=5
        )

        ttk.Checkbutton(
            preprocessing,
            text="MNI 仿射配准",
            variable=self.use_mni_var
        ).pack(
            anchor="w",
            padx=10,
            pady=5
        )

        ttk.Button(
            preprocessing,
            text="选择 MNI T1 模板",
            command=self.select_mni_template
        ).pack(
            fill=tk.X,
            padx=8,
            pady=6
        )

        self.template_label = ttk.Label(
            preprocessing,
            text="未选择模板",
            wraplength=240
        )

        self.template_label.pack(
            padx=8,
            pady=4
        )

        # ====================================================
        # 模型训练
        # ====================================================

        model_frame = ttk.LabelFrame(
            left,
            text="模式识别"
        )

        model_frame.pack(
            fill=tk.X,
            pady=10
        )

        self.train_button = ttk.Button(
            model_frame,
            text="优化训练并比较三个模型",
            command=lambda:
            self.start_thread(
                self.train_model
            )
        )

        self.train_button.pack(
            fill=tk.X,
            padx=8,
            pady=6
        )

        self.predict_button = ttk.Button(
            model_frame,
            text="预测测试集 + 生成3份优化结果",
            command=lambda:
            self.start_thread(
                self.predict_test
            )
        )

        self.predict_button.pack(
            fill=tk.X,
            padx=8,
            pady=6
        )

        ttk.Button(
            model_frame,
            text="导出 submission.csv",
            command=self.export_submission
        ).pack(
            fill=tk.X,
            padx=8,
            pady=6
        )

        # ====================================================
        # 进度条
        # ====================================================

        ttk.Label(
            left,
            text="处理进度"
        ).pack(
            anchor="w",
            pady=(15, 3)
        )

        self.progress = ttk.Progressbar(
            left,
            orient="horizontal",
            mode="determinate",
            maximum=100
        )

        self.progress.pack(
            fill=tk.X,
            pady=5
        )

        self.progress_label = ttk.Label(
            left,
            text="0%"
        )

        self.progress_label.pack()

        # ====================================================
        # MRI 浏览页面
        # ====================================================

        self.figure = Figure(
            figsize=(9, 5),
            dpi=100
        )

        self.ax1 = self.figure.add_subplot(
            131
        )

        self.ax2 = self.figure.add_subplot(
            132
        )

        self.ax3 = self.figure.add_subplot(
            133
        )

        self.canvas = FigureCanvasTkAgg(
            self.figure,
            master=self.viewer_tab
        )

        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

        slider_frame = ttk.Frame(
            self.viewer_tab
        )

        slider_frame.pack(
            fill=tk.X,
            padx=20,
            pady=10
        )

        ttk.Label(
            slider_frame,
            text="轴状位"
        ).grid(
            row=0,
            column=0,
            padx=5
        )

        self.axial_slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=100,
            command=self.update_mri_view
        )

        self.axial_slider.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=10
        )

        ttk.Label(
            slider_frame,
            text="冠状位"
        ).grid(
            row=1,
            column=0,
            padx=5
        )

        self.coronal_slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=100,
            command=self.update_mri_view
        )

        self.coronal_slider.grid(
            row=1,
            column=1,
            sticky="ew",
            padx=10
        )

        ttk.Label(
            slider_frame,
            text="矢状位"
        ).grid(
            row=2,
            column=0,
            padx=5
        )

        self.sagittal_slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=100,
            command=self.update_mri_view
        )

        self.sagittal_slider.grid(
            row=2,
            column=1,
            sticky="ew",
            padx=10
        )

        slider_frame.columnconfigure(
            1,
            weight=1
        )

        # ====================================================
        # 分类结果页面
        # ====================================================

        metric_frame = ttk.LabelFrame(
            self.result_tab,
            text="交叉验证性能"
        )

        metric_frame.pack(
            fill=tk.X,
            padx=20,
            pady=20
        )

        self.metric_label = ttk.Label(
            metric_frame,
            text=(
                "ACC：--     "
                "Sensitivity：--     "
                "Specificity：--"
            ),
            font=(
                "Microsoft YaHei",
                14,
                "bold"
            )
        )

        self.metric_label.pack(
            pady=15
        )

        self.cm_label = ttk.Label(
            metric_frame,
            text="混淆矩阵：--",
            font=(
                "Consolas",
                12
            )
        )

        self.cm_label.pack(
            pady=10
        )

        # ----------------------------------------------------
        # 测试预测表
        # ----------------------------------------------------

        prediction_frame = ttk.LabelFrame(
            self.result_tab,
            text="测试集预测结果"
        )

        prediction_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=20,
            pady=10
        )

        self.tree = ttk.Treeview(
            prediction_frame,
            columns=(
                "subject",
                "label",
                "diagnosis"
            ),
            show="headings"
        )

        self.tree.heading(
            "subject",
            text="Subject ID"
        )

        self.tree.heading(
            "label",
            text="Label"
        )

        self.tree.heading(
            "diagnosis",
            text="Diagnosis"
        )

        self.tree.column(
            "subject",
            width=200,
            anchor="center"
        )

        self.tree.column(
            "label",
            width=100,
            anchor="center"
        )

        self.tree.column(
            "diagnosis",
            width=150,
            anchor="center"
        )

        self.tree.pack(
            fill=tk.BOTH,
            expand=True
        )

        # ====================================================
        # 日志页
        # ====================================================

        self.log_text = tk.Text(
            self.log_tab,
            font=(
                "Consolas",
                10
            )
        )

        self.log_text.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10
        )


    # ========================================================
    # 八、数据检查
    # ========================================================

    def check_dataset(self):

        train_files = []

        test_files = []

        if TRAIN_DIR.exists():

            train_files = list(
                TRAIN_DIR.glob(
                    "*_T1w.nii.gz"
                )
            )

        if TEST_DIR.exists():

            test_files = list(
                TEST_DIR.glob(
                    "*_T1w.nii.gz"
                )
            )

        # ----------------------------------------------------
        # 更新训练集状态
        # ----------------------------------------------------

        if len(train_files) > 0:

            self.train_status.config(
                text=f"训练集：{len(train_files)} 例 ✓"
            )

        else:

            self.train_status.config(
                text="训练集：未找到 ✗"
            )

        # ----------------------------------------------------

        if len(test_files) > 0:

            self.test_status.config(
                text=f"测试集：{len(test_files)} 例 ✓"
            )

        else:

            self.test_status.config(
                text="测试集：未找到 ✗"
            )

        # ----------------------------------------------------

        if LABEL_FILE.exists():

            try:

                labels = pd.read_csv(
                    LABEL_FILE
                )

                self.label_status.config(
                    text=f"标签：{len(labels)} 条 ✓"
                )

            except Exception:

                self.label_status.config(
                    text="标签：读取失败 ✗"
                )

        else:

            self.label_status.config(
                text="标签：未找到 ✗"
            )

        # ----------------------------------------------------

        self.log("=" * 60)

        self.log("程序启动成功")

        self.log(
            f"项目目录：{BASE_DIR}"
        )

        self.log(
            f"训练目录：{TRAIN_DIR}"
        )

        self.log(
            f"测试目录：{TEST_DIR}"
        )

        self.log(
            f"训练样本：{len(train_files)}"
        )

        self.log(
            f"测试样本：{len(test_files)}"
        )

        self.log("=" * 60)


    # ========================================================
    # 九、日志
    # ========================================================

    def log(self, text):

        def update():

            self.log_text.insert(
                tk.END,
                str(text) + "\n"
            )

            self.log_text.see(
                tk.END
            )

        self.root.after(
            0,
            update
        )


    # ========================================================
    # 十、进度条
    # ========================================================

    def set_progress(self, value):

        value = max(
            0,
            min(
                100,
                value
            )
        )

        def update():

            self.progress[
                "value"
            ] = value

            self.progress_label.config(
                text=f"{value:.0f}%"
            )

        self.root.after(
            0,
            update
        )


    # ========================================================
    # 十一、多线程
    # ========================================================

    def start_thread(self, function):

        thread = threading.Thread(
            target=function,
            daemon=True
        )

        thread.start()


    # ========================================================
    # 十二、选择 MNI 模板
    # ========================================================

    def select_mni_template(self):

        path = filedialog.askopenfilename(
            title="选择 MNI152 T1 模板",
            filetypes=[
                (
                    "NIfTI",
                    "*.nii *.nii.gz"
                ),
                (
                    "All files",
                    "*.*"
                )
            ]
        )

        if not path:
            return

        self.template_path = Path(path)

        self.template_label.config(
            text=self.template_path.name
        )

        self.use_mni_var.set(
            True
        )

        self.log(
            f"MNI 模板：{path}"
        )


    # ========================================================
    # 十三、MRI 浏览
    # ========================================================

    def open_mri(self):

        # 默认打开训练集目录
        initial_dir = (
            str(TRAIN_DIR)
            if TRAIN_DIR.exists()
            else str(BASE_DIR)
        )

        path = filedialog.askopenfilename(
            title="选择 MRI",
            initialdir=initial_dir,
            filetypes=[
                (
                    "NIfTI",
                    "*.nii *.nii.gz"
                ),
                (
                    "All files",
                    "*.*"
                )
            ]
        )

        if not path:
            return

        try:

            nii = nib.load(
                path
            )

            nii = nib.as_closest_canonical(
                nii
            )

            volume = nii.get_fdata(
                dtype=np.float32
            )

            volume = np.nan_to_num(
                volume
            )

            self.current_volume = volume

            self.current_file = path

            # 设置滑块范围
            self.sagittal_slider.config(
                from_=0,
                to=volume.shape[0] - 1
            )

            self.coronal_slider.config(
                from_=0,
                to=volume.shape[1] - 1
            )

            self.axial_slider.config(
                from_=0,
                to=volume.shape[2] - 1
            )

            # 设置中间切片
            self.sagittal_slider.set(
                volume.shape[0] // 2
            )

            self.coronal_slider.set(
                volume.shape[1] // 2
            )

            self.axial_slider.set(
                volume.shape[2] // 2
            )

            self.update_mri_view()

            self.notebook.select(
                self.viewer_tab
            )

            self.log(
                f"打开 MRI：{Path(path).name}"
            )

            self.log(
                f"图像尺寸：{volume.shape}"
            )

            self.log(
                "体素大小："
                f"{nii.header.get_zooms()[:3]}"
            )

        except Exception as e:

            messagebox.showerror(
                "错误",
                f"MRI 读取失败：\n{e}"
            )


    # ========================================================
    # 十四、更新 MRI 三视图
    # ========================================================

    def update_mri_view(self, *_):

        if self.current_volume is None:
            return

        volume = self.current_volume

        sagittal = int(
            self.sagittal_slider.get()
        )

        coronal = int(
            self.coronal_slider.get()
        )

        axial = int(
            self.axial_slider.get()
        )

        sagittal = min(
            sagittal,
            volume.shape[0] - 1
        )

        coronal = min(
            coronal,
            volume.shape[1] - 1
        )

        axial = min(
            axial,
            volume.shape[2] - 1
        )

        # 显示窗口
        positive = volume[
            np.isfinite(volume)
        ]

        if len(positive) > 0:

            vmin = np.percentile(
                positive,
                1
            )

            vmax = np.percentile(
                positive,
                99
            )

        else:

            vmin = None
            vmax = None

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Axial
        self.ax1.imshow(
            np.rot90(
                volume[:, :, axial]
            ),
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        self.ax1.set_title(
            f"Axial\nSlice {axial}"
        )

        self.ax1.axis(
            "off"
        )

        # Coronal
        self.ax2.imshow(
            np.rot90(
                volume[:, coronal, :]
            ),
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        self.ax2.set_title(
            f"Coronal\nSlice {coronal}"
        )

        self.ax2.axis(
            "off"
        )

        # Sagittal
        self.ax3.imshow(
            np.rot90(
                volume[sagittal, :, :]
            ),
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        self.ax3.set_title(
            f"Sagittal\nSlice {sagittal}"
        )

        self.ax3.axis(
            "off"
        )

        self.figure.tight_layout()

        self.canvas.draw_idle()


    # ========================================================
    # 十五、获取预处理设置
    # ========================================================

    def get_processing_settings(self):

        use_n4 = self.use_n4_var.get()

        use_mni = self.use_mni_var.get()

        template = None

        if use_mni:

            if self.template_path is None:

                self.log(
                    "警告：开启了 MNI 配准，但未选择 MNI 模板。"
                )

                self.log(
                    "本次训练自动关闭 MNI 配准。"
                )

            else:

                template = self.template_path

        return (
            use_n4,
            template
        )


    # ========================================================
    # 十六、训练并优化三个模型
    # ========================================================

    def train_model(self):

        self.set_progress(0)

        if not TRAIN_DIR.exists():
            self.log("错误：找不到训练集目录。")
            return

        if not LABEL_FILE.exists():
            self.log("错误：找不到 train_labels.csv。")
            return

        labels_df = pd.read_csv(LABEL_FILE)
        required_columns = {"subject_id", "label"}
        if not required_columns.issubset(labels_df.columns):
            self.log("标签文件必须包含 subject_id 和 label 两列。")
            return

        labels_df = labels_df.set_index("subject_id")
        files = sorted(TRAIN_DIR.glob("*_T1w.nii.gz"))
        if not files:
            self.log("训练目录没有 NIfTI 文件。")
            return

        use_n4, template = self.get_processing_settings()

        self.log("")
        self.log("=" * 70)
        self.log("开始：优化预处理 + 多尺度特征 + 三模型参数搜索")
        self.log(f"训练样本：{len(files)}")
        self.log(f"N4：{'开启' if use_n4 else '关闭'}")
        self.log(f"MNI：{'开启' if template else '关闭'}")
        self.log(f"特征版本：{FEATURE_VERSION}")
        self.log("提示：相同预处理配置会自动读取特征缓存，加快重复实验。")

        X, y = [], []
        total = len(files)

        for index, file in enumerate(files, start=1):
            sid = get_subject_id(file)
            if sid not in labels_df.index:
                self.log(f"跳过 {sid}：无标签")
                continue

            try:
                self.log(f"[{index}/{total}] 处理 {sid}")
                feature = extract_feature_cached(
                    image_path=file,
                    use_n4=use_n4,
                    template_path=template
                )
                X.append(feature)
                y.append(int(labels_df.loc[sid, "label"]))
            except Exception as e:
                self.log(f"{sid} 失败：{e}")

            self.set_progress(index / total * 55)

        if len(X) < 10:
            self.log("错误：有效样本数量不足。")
            return

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)

        unique, counts = np.unique(y, return_counts=True)
        self.log("")
        self.log(f"有效样本：{len(y)}")
        self.log(f"重设计后的原始特征维度：{X.shape[1]}")
        self.log(f"类别分布：{dict(zip(unique.tolist(), counts.tolist()))}")

        specs = build_model_specs(X.shape[1])
        self.models = {}
        self.model_results = {}
        comparison_rows = []

        for index, (name, spec) in enumerate(specs.items(), start=1):
            self.log("")
            self.log("-" * 70)
            self.log(f"模型 {index}/3：{name}")
            self.log("外层5折评估 + 内层3折轻量调参...")

            result = nested_evaluate_and_fit(name, spec, X, y)
            self.model_results[name] = result
            self.models[name] = result["FinalModel"]

            for fold_info in result["FoldDetails"]:
                self.log(
                    f"Outer Fold {fold_info['fold']}/5 | "
                    f"inner BACC={fold_info['inner_bacc']:.3f} | "
                    f"best={fold_info['best_params']}"
                )

            self.log(
                f"{name} OOF：ACC={result['ACC']:.4f}, "
                f"BACC={result['BACC']:.4f}, "
                f"SEN={result['Sensitivity']:.4f}, "
                f"SPE={result['Specificity']:.4f}, "
                f"AUC={result['AUC']:.4f}"
            )
            self.log(f"全训练集最终参数：{result['BestParams']}")
            self.log(f"全训练集内部CV BACC：{result['FinalCVBACC']:.4f}")

            cm = result["CM"]
            self.log("          Pred ASD   Pred HC")
            self.log(f"True ASD     {cm[0, 0]:4d}       {cm[0, 1]:4d}")
            self.log(f"True HC      {cm[1, 0]:4d}       {cm[1, 1]:4d}")

            comparison_rows.append({
                "Model": name,
                "ACC": result["ACC"],
                "BACC": result["BACC"],
                "Sensitivity": result["Sensitivity"],
                "Specificity": result["Specificity"],
                "AUC": result["AUC"],
                "FinalCV_BACC": result["FinalCVBACC"],
                "BestParams": str(result["BestParams"])
            })

            self.set_progress(55 + index / 3 * 45)

        comparison_df = pd.DataFrame(comparison_rows)
        comparison_path = RESULT_DIR / "model_comparison_optimized.csv"
        comparison_df.to_csv(comparison_path, index=False)

        # 用外层 OOF BACC 决定默认 submission.csv 对应模型
        self.best_model_name = max(
            self.model_results,
            key=lambda n: self.model_results[n]["BACC"]
        )
        self.model = self.models[self.best_model_name]
        self.model_use_n4 = use_n4
        self.model_template = template

        def update_metrics():
            lines = []
            for name, result in self.model_results.items():
                lines.append(
                    f"{name}: ACC={result['ACC']:.3f}, "
                    f"BACC={result['BACC']:.3f}, "
                    f"SEN={result['Sensitivity']:.3f}, "
                    f"SPE={result['Specificity']:.3f}, "
                    f"AUC={result['AUC']:.3f}"
                )

            self.metric_label.config(
                text=(
                    "优化后三模型交叉验证结果\n\n"
                    + "\n".join(lines)
                    + f"\n\nOOF BACC最佳：{self.best_model_name}"
                )
            )
            self.cm_label.config(text="详细参数与混淆矩阵请查看运行日志。")
            self.notebook.select(self.result_tab)

        self.root.after(0, update_metrics)
        self.set_progress(100)

        self.log("")
        self.log("=" * 70)
        self.log("三模型优化训练完成。")
        self.log(f"OOF BACC最佳模型：{self.best_model_name}")
        self.log(f"模型比较结果：{comparison_path}")
        self.log("现在可以点击“预测测试集 + 生成3份结果”。")
        self.log("=" * 70)


    # ========================================================
    # 十七、测试集预测：生成三份 submission
    # ========================================================

    def predict_test(self):

        if len(self.models) != 3:
            self.log("错误：请先完成三个模型的优化训练。")
            return

        files = sorted(TEST_DIR.glob("*_T1w.nii.gz"))
        if not files:
            self.log("错误：测试集为空或目录不存在。")
            return

        self.set_progress(0)
        self.log("")
        self.log("=" * 70)
        self.log("开始处理测试集（支持特征缓存）...")

        X_test, subject_ids = [], []
        total = len(files)

        for index, file in enumerate(files, start=1):
            sid = get_subject_id(file)
            try:
                feature = extract_feature_cached(
                    image_path=file,
                    use_n4=self.model_use_n4,
                    template_path=self.model_template
                )
                X_test.append(feature)
                subject_ids.append(sid)
                self.log(f"[{index}/{total}] {sid}")
            except Exception as e:
                self.log(f"{sid} 失败：{e}")

            self.set_progress(index / total * 70)

        if len(X_test) != len(files):
            self.log("错误：部分测试 MRI 处理失败，停止生成结果。")
            return

        X_test = np.asarray(X_test, dtype=np.float32)
        predictions = {
            name: model.predict(X_test).astype(int)
            for name, model in self.models.items()
        }

        file_names = {
            "Linear SVM": "submission_1_linear_svm_optimized.csv",
            "Logistic Regression": "submission_2_logistic_optimized.csv",
            "RBF-SVM": "submission_3_rbf_svm_optimized.csv"
        }

        result_frames = {}

        for name, pred in predictions.items():
            prediction_map = dict(zip(subject_ids, pred.tolist()))

            if SUBMISSION_TEMPLATE.exists():
                result = pd.read_csv(SUBMISSION_TEMPLATE)[["subject_id"]].copy()
                result["label"] = result["subject_id"].map(prediction_map)
            else:
                result = pd.DataFrame({"subject_id": subject_ids, "label": pred})

            if result["label"].isna().any():
                self.log(f"错误：{name} 存在未匹配的测试 subject_id。")
                return

            result["label"] = result["label"].astype(int)
            output_path = RESULT_DIR / file_names[name]
            result.to_csv(output_path, index=False)
            result_frames[name] = result
            self.log(f"{name} 已保存：{output_path}")

        best_result = result_frames[self.best_model_name]
        best_result.to_csv(OUTPUT_FILE, index=False)
        self.prediction_df = best_result

        def update_table():
            for item in self.tree.get_children():
                self.tree.delete(item)

            for _, row in best_result.iterrows():
                label_value = int(row["label"])
                diagnosis = "ASD" if label_value == 1 else "HC"
                self.tree.insert(
                    "", tk.END,
                    values=(row["subject_id"], label_value, diagnosis)
                )

            self.notebook.select(self.result_tab)

        self.root.after(0, update_table)

        self.log("")
        self.log(f"默认 submission.csv 使用模型：{self.best_model_name}")
        self.log(f"默认结果：{OUTPUT_FILE}")
        self.log("=" * 70)
        self.set_progress(100)

        self.root.after(
            0,
            lambda: messagebox.showinfo(
                "完成",
                "优化后的三份预测已生成到 results 文件夹。\n"
                f"默认 submission.csv：{self.best_model_name}"
            )
        )


    # ========================================================
    # 十八、导出 submission.csv
    # ========================================================

    def export_submission(self):

        if self.prediction_df is None:

            messagebox.showwarning(
                "提示",
                "请先执行测试集预测。"
            )

            return

        path = filedialog.asksaveasfilename(
            title="保存 submission.csv",
            initialfile="submission.csv",
            defaultextension=".csv",
            filetypes=[
                (
                    "CSV 文件",
                    "*.csv"
                )
            ]
        )

        if not path:
            return

        try:

            self.prediction_df[
                [
                    "subject_id",
                    "label"
                ]
            ].to_csv(
                path,
                index=False
            )

            messagebox.showinfo(
                "完成",
                "submission.csv 保存成功！"
            )

            self.log(
                f"文件已导出：{path}"
            )

        except Exception as e:

            messagebox.showerror(
                "错误",
                str(e)
            )


# ============================================================
# 十九、主程序入口
# ============================================================

def main():

    root = tk.Tk()

    MRIClassifierApp(
        root
    )

    root.mainloop()


if __name__ == "__main__":

    main()
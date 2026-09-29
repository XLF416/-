# 基于 T1 加权结构 MRI 的 ASD / HC 分类系统

## 一、设计目的

本项目面向 T1 加权结构磁共振影像（T1-weighted structural MRI），构建一个用于区分自闭症谱系障碍（ASD）与健康对照（HC）的二分类模式识别系统。系统目标不是单纯训练某一个分类器，而是形成一套较完整、可复现的医学影像机器学习流程，包括影像读取、可选预处理、多尺度特征提取、特征筛选、模型训练、交叉验证、测试集预测和结果导出。

程序同时提供图形化界面，可浏览 MRI 三视图、查看训练日志和交叉验证结果，并自动生成测试集预测文件。标签约定为：

- `1 = ASD`
- `2 = HC`

本项目主要用于课程教学、模式识别实验和医学影像机器学习方法验证，不作为临床诊断工具。

---

## 二、实际应用价值

1. **辅助研究 ASD 相关脑结构差异**  
   T1 结构 MRI 可以反映脑组织形态及局部结构特征。本系统通过强度、梯度、高频残差和空间统计等多尺度特征，对 ASD 与 HC 之间可能存在的结构差异进行数据驱动分析。

2. **建立完整的医学影像机器学习流程**  
   系统从原始 NIfTI 影像开始，经过预处理、特征工程、特征选择、分类建模和交叉验证，覆盖医学影像模式识别任务中的主要环节，适合作为课程设计和实验教学案例。

3. **提高小样本条件下模型评估的可靠性**  
   针对样本量较小、类别数量不完全均衡的问题，程序使用分层交叉验证、Balanced Accuracy、Sensitivity、Specificity 和 AUC 等指标综合评价模型，避免只依赖单一 Accuracy。

4. **支持多模型比较**  
   系统同时比较 Linear SVM、Logistic Regression 和 RBF-SVM 三种分类器，并分别进行参数搜索，使不同分类边界假设能够在同一套特征上公平比较。

5. **生成可直接提交的预测结果**  
   程序可对测试集自动预测，并生成三份不同模型对应的 `submission` 文件，同时将外层交叉验证 BACC 最优的模型结果额外保存为默认 `submission.csv`。

---

## 三、技术路线

### 3.1 总体流程

```text
T1 MRI（.nii.gz）
        ↓
影像读取与方向处理
        ↓
前景区域近似提取
        ↓
N4 偏置场校正（可选）
        ↓
MNI 仿射配准（可选）
        ↓
稳健强度标准化
        ↓
多尺度结构特征提取
        ↓
低方差特征去除
        ↓
SelectKBest 特征选择
        ↓
StandardScaler 标准化
        ↓
┌─────────────────────────────┐
│ Linear SVM                  │
│ Logistic Regression         │
│ RBF-SVM                     │
└─────────────────────────────┘
        ↓
外层 5 折 + 内层 3 折嵌套交叉验证
        ↓
ACC / BACC / SEN / SPE / AUC
        ↓
全部训练集重新搜索参数并训练最终模型
        ↓
测试集预测
        ↓
生成 3 份预测文件 + submission.csv
```

### 3.2 影像预处理

程序中的预处理包含以下步骤：

#### （1）前景区域近似提取

通过非零体素和低分位强度阈值构建前景掩膜，并使用轻度闭运算和孔洞填充提高区域连续性。该步骤主要用于减少大面积背景和边缘噪声对后续统计的影响。

#### （2）N4 偏置场校正（可选）

使用 SimpleITK 的 `N4BiasFieldCorrectionImageFilter` 对 MRI 中缓慢变化的强度不均匀进行校正。该功能默认关闭，可在 GUI 中手动启用。

#### （3）MNI 仿射配准（可选）

使用 SimpleITK 将受试者 MRI 仿射配准到用户选择的 MNI T1 模板。配准相似性度量采用 Mattes Mutual Information，并使用多分辨率优化策略。该功能默认关闭，需要用户提供 MNI 模板后手动启用。

#### （4）稳健强度归一化

在前景区域内先进行 1%～99% 百分位裁剪，再进行 Z-score 标准化，以减小异常高亮体素和不同扫描强度尺度的影响。

#### （5）未使用 MNI 时的紧凑裁剪

如果没有进行 MNI 配准，程序会根据前景区域自动裁剪影像，减少无效背景，同时降低不同影像周围空白区域造成的影响。

### 3.3 多尺度特征提取

与简单地将单一降采样体数据直接展平相比，本程序同时提取多种结构信息。

#### （1）低分辨率强度体素特征

对轻度高斯平滑后的 MRI 重采样到：

```text
24 × 24 × 24 = 13,824 维
```

用于保留整体脑结构及粗粒度强度分布信息。

#### （2）梯度幅值特征

计算三维 Gaussian Gradient Magnitude，并重采样至：

```text
16 × 16 × 16 = 4,096 维
```

主要反映组织边界、局部结构变化和形态差异。

#### （3）高频残差特征

使用原平滑图像减去更大尺度 Gaussian 平滑结果，获得局部高频残差信息，再重采样至：

```text
16 × 16 × 16 = 4,096 维
```

用于增强局部纹理和较细尺度结构变化的表达。

#### （4）空间块统计特征

将 `16 × 16 × 16` 体数据划分为 `4 × 4 × 4` 大小的空间块，共 64 个块。每个块计算：

- 均值
- 标准差
- 绝对值均值

共得到：

```text
64 × 3 = 192 维
```

#### （5）全局结构统计特征

程序进一步提取：

- 32 维标准化强度直方图
- 9 个强度分位数
- 前景区域质心位置
- 空间扩散程度
- 前景占比
- 强度均值与标准差

共得到 50 维全局统计特征。

因此，在进入特征筛选前，每名受试者的原始特征维度约为：

```text
13,824 + 4,096 + 4,096 + 192 + 50 = 22,258 维
```

### 3.4 特征缓存

N4 校正、MNI 配准和三维特征提取计算量较大，因此程序提供 `.feature_cache/` 缓存机制。

当以下条件没有变化时：

- MRI 文件
- N4 是否启用
- MNI 是否启用
- MNI 模板
- 特征版本

再次训练会直接读取已经保存的 `.npy` 特征文件，避免重复进行耗时的三维处理。

### 3.5 特征选择与标准化

每个分类模型都使用以下 Pipeline：

```text
VarianceThreshold
        ↓
SelectKBest（ANOVA F-test）
        ↓
StandardScaler
        ↓
Classifier
```

其中：

- `VarianceThreshold` 用于移除近乎恒定的无效特征；
- `SelectKBest` 使用 `f_classif` 对 ASD 和 HC 的特征差异进行单变量筛选；
- `StandardScaler` 对最终保留特征进行标准化。

这些步骤均位于 sklearn `Pipeline` 内部，并在每一个交叉验证训练折中重新拟合，从而降低特征选择产生的信息泄漏风险。

### 3.6 分类模型与参数搜索

系统比较 3 个分类器。

#### Linear SVM

搜索参数：

```text
SelectKBest k ∈ {200, 500, 1000}
C ∈ {0.1, 1.0, 10.0}
```

#### Logistic Regression

搜索参数：

```text
SelectKBest k ∈ {200, 500, 1000}
C ∈ {0.1, 1.0, 10.0}
```

使用 L2 正则化、`liblinear` 求解器和类别平衡权重。

#### RBF-SVM

搜索参数：

```text
SelectKBest k ∈ {200, 500}
C ∈ {1.0, 10.0}
gamma ∈ {scale, 0.001}
```

三个模型均使用 `class_weight="balanced"` 或等效类别平衡设置，以降低类别不均衡造成的偏差。

### 3.7 嵌套交叉验证

为了避免“使用同一批交叉验证数据既调参又报告最终结果”造成的乐观偏差，程序使用嵌套交叉验证：

```text
外层：Stratified 5-fold CV
    ↓
每个外层训练集
    ↓
内层：Stratified 3-fold GridSearchCV
    ↓
以 Balanced Accuracy 选择最优参数
    ↓
在对应外层验证折评价
```

最终得到完整的 OOF（Out-of-Fold）预测结果，并据此计算各项指标。

完成外层评价后，程序还会在全部训练数据上执行一次 5 折 GridSearchCV，得到用于预测测试集的最终模型。

---

## 四、结果评价指标

本项目使用以下指标综合评价模型。

### 4.1 Accuracy（ACC）

表示全部样本中预测正确的比例：

```text
ACC = (TP + TN) / (TP + TN + FP + FN)
```

ACC 可以反映整体分类正确率，但在类别不均衡时不能单独作为评价依据。

### 4.2 Balanced Accuracy（BACC）

```text
BACC = (Sensitivity + Specificity) / 2
```

Balanced Accuracy 对两类给予相同权重，因此更适合 ASD / HC 样本数量不完全相等的情况。本程序的参数搜索也使用 BACC 作为主要评分指标。

### 4.3 Sensitivity（灵敏度）

本项目规定：

```text
ASD = 正类 = label 1
```

因此：

```text
Sensitivity = TP / (TP + FN)
```

表示真实 ASD 受试者中被模型正确识别为 ASD 的比例。

### 4.4 Specificity（特异度）

```text
Specificity = TN / (TN + FP)
```

表示真实 HC 受试者中被正确识别为 HC 的比例。

### 4.5 ROC-AUC

AUC 表示模型对 ASD 和 HC 进行排序区分的整体能力。程序统一将 ASD（label=1）定义为阳性类，并根据分类器的 `decision_function` 计算 AUC。

通常：

```text
AUC = 0.5   接近随机分类
AUC 越接近 1.0，区分能力越强
```

### 4.6 混淆矩阵

程序同时输出：

```text
                 预测 ASD     预测 HC
真实 ASD            TP           FN
真实 HC             FP           TN
```

用于直观观察模型对 ASD 和 HC 两类样本的具体预测情况。

---

## 五、程序运行说明

### 5.1 项目目录

推荐目录结构：

```text
MRI_ASD_Project/
│
├── excise/
│   ├── train/
│   │   ├── sub-001_T1w.nii.gz
│   │   └── ...
│   │
│   ├── test/
│   │   ├── sub-xxx_T1w.nii.gz
│   │   └── ...
│   │
│   ├── train_labels.csv
│   └── submission_example.csv
│
├── main_optimized.py
│
├── results/               # 程序自动生成
└── .feature_cache/        # 程序自动生成
```

如果使用 MNI 配准，可自行准备 MNI T1 模板并在软件中选择。

### 5.2 安装依赖

```bash
pip install numpy pandas nibabel SimpleITK scipy matplotlib scikit-learn
```

如果系统无法直接识别 `pip`，Windows 下可使用：

```bash
python -m pip install numpy pandas nibabel SimpleITK scipy matplotlib scikit-learn
```

### 5.3 运行程序

```bash
python main_optimized.py
```

### 5.4 GUI 操作顺序

1. 启动程序，确认训练集、测试集和标签状态正常；
2. 可使用“打开 MRI 影像”浏览轴状位、冠状位和矢状位；
3. 根据实验方案选择是否启用 N4；
4. 如需 MNI 配准，先选择 MNI T1 模板，再勾选 MNI；
5. 点击“优化训练并比较三个模型”；
6. 等待三个模型完成嵌套交叉验证和最终参数搜索；
7. 查看分类结果和运行日志；
8. 点击“预测测试集 + 生成3份优化结果”；
9. 在 `results/` 目录查看三份测试集预测文件。

---

## 六、输出文件

训练完成后会生成：

```text
results/model_comparison_optimized.csv
```

记录三个模型的：

- ACC
- BACC
- Sensitivity
- Specificity
- AUC
- 全训练集内部交叉验证 BACC
- 最终最优参数

测试集预测完成后生成：

```text
results/submission_1_linear_svm_optimized.csv
results/submission_2_logistic_optimized.csv
results/submission_3_rbf_svm_optimized.csv
```

程序还会根据外层 OOF BACC 自动选择交叉验证表现最好的模型，并将其测试集结果额外保存为：

```text
submission.csv
```

预测文件格式：

```csv
subject_id,label
sub-xxx,1
sub-xxx,2
```

其中：

```text
1 = ASD
2 = HC
```

---

## 七、注意事项与局限性

1. **本程序不是临床诊断软件**  
   分类结果仅用于教学和实验分析，不能代替医学诊断。

2. **MNI 配准为仿射配准**  
   当前 SimpleITK 实现主要校正平移、旋转、缩放和剪切等全局空间差异，并不等同于 ANTs SyN、SPM/CAT12 等非线性标准化流程。

3. **前景掩膜不是专业脑提取**  
   当前 `foreground_mask()` 为稳健的近似前景提取方法，并非 FSL BET、HD-BET 等专业 skull-stripping 方法。

4. **多尺度特征属于手工结构特征**  
   当前方法主要提取强度、梯度、高频残差及局部/全局统计信息，并没有显式进行灰质、白质和脑脊液的专业组织分割。

5. **小样本结果可能波动**  
   训练样本数量有限，因此单次交叉验证或测试集结果可能存在较大方差，应结合多项指标进行分析。

6. **不要根据匿名编号推测诊断标签**  
   测试受试者编号仅为匿名编号，应完全依据影像模型进行预测。

---

## 八、总结

本项目实现了一套针对 T1 结构 MRI 的 ASD / HC 多模型分类流程。相较于单一低分辨率体素特征，优化版方法融合了原始强度、空间梯度、高频残差、局部块统计和全局结构统计等多尺度特征，并通过 Pipeline、SelectKBest、类别平衡和嵌套交叉验证控制高维小样本任务中的过拟合与信息泄漏风险。最终通过 Linear SVM、Logistic Regression 和 RBF-SVM 三种模型完成训练和测试集预测，为后续分析不同预处理策略、特征设计和分类算法对结果的影响提供了统一实验框架。

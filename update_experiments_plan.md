# UniLasso 合成数据模拟实验设计

## 背景
UniLasso 算法相较于传统方法有两个关键升级：
1. **分组约束升级** - 对高度相关的特征进行分组，促进分组内变量的同时选择/剔除
2. **自适应加权** - 根据单变量显著性调整惩罚项，降低显著变量的惩罚，增加不显著变量的惩罚

需要设计合成数据实验来系统验证这两个升级的效果，并覆盖三个版本：
1. **基础线性版本** - 线性高斯稀疏回归
2. **GLM 扩展版本** - 所有 GLM 族（高斯、二项/逻辑、泊松、多项、Cox）
3. **非线性版本** - 使用样条和决策树单变量模型处理非线性关系

实验需要特别验证：分组约束和自适应加权在不同场景下的效果提升。

## 现有基础设施复用

### 已有代码
| 文件 | 用途 | 如何复用 |
|------|------|----------|
| `unilasso/utils.py` | 基础模拟函数：`simulate_gaussian_data`, `simulate_binomial_data`, `simulate_cox_data` | 直接导入复用 |
| `data_generators.py` | 高级线性生成器：AR(1)、高维、符号不一致、因子模型 | **扩展**该文件，添加新生成器 |
| `unilasso/uni_lasso.py` | 主 `fit_uni()`, `cv_uni()` API 已支持所有参数 | 直接使用，无需修改 |
| `tests/test_performance.py` | 已有基准测试模式 | 参考结构 |
| `tests/test_glm_families.py` | GLM 测试模式 | 参考结构 |
| `tests/test_nonlinear.py` | 非线性测试模式 | 参考结构 |

## 实现计划

### 阶段一：扩展数据生成器
**文件：`data_generators.py`** - 在现有文件末尾添加新生成器：

1. **`simulate_poisson_data()`** - 生成泊松计数数据：
   - 线性预测器：`log_lambda = X @ beta`
   - 响应：`y ~ Poisson(exp(log_lambda))`

2. **`simulate_multinomial_data()`** - 生成多类分类数据：
   - k个类，每个类有自己的线性预测器
   - 响应：`y ~ Multinomial(1, p)` 其中 `p = softmax(X @ B)`

3. **`simulate_nonlinear_gaussian_data()`** - 非线性高斯响应：
   - 支持多种非线性模式：sine、quadratic、step、mixed
   - 仅子集特征有非线性效应（保持稀疏性）
   - 真实模型：`y = sum f_j(x_j) + 噪声`

4. **`simulate_nonlinear_glm_data()`** - 通用非线性GLM生成器：
   - 支持任意 GLM 族（gaussian、binomial、poisson）
   - 真实预测器：`eta = sum f_j(x_j)`，其中 f_j 是非线性函数

5. **`simulate_mixed_data()`** - 线性 + 非线性 + 无关特征的混合数据
   - 更真实的场景，测试特征选择

### 阶段二：创建实验框架

**新目录结构：**
```
/workspaces/uni-lasso/
├── experiments/
│   ├── __init__.py
│   ├── base_experiment.py         # 所有实验的基类
│   ├── linear_experiment.py       # 线性高斯实验
│   ├── glm_experiment.py          # 所有 GLM 族实验
│   ├── nonlinear_experiment.py    # 非线性（样条/树）实验
│   └── visualization.py           # 共享绘图工具
├── scripts/
│   ├── run_linear_experiment.py   # 线性实验命令行接口
│   ├── run_glm_experiment.py      # GLM 实验命令行接口
│   └── run_nonlinear_experiment.py # 非线性实验命令行接口
└── experiments/results/           # 输出目录（加入 .gitignore）
```

**基类：`experiments/base_experiment.py`**
```python
class BaseSimulationExperiment:
    def __init__(self, n_repeats=20, test_size=0.3, random_seed=42):
        self.n_repeats = n_repeats
        self.test_size = test_size
        self.random_seed = random_seed
        self.results = []

    def generate_data(self, repeat_seed):
        raise NotImplementedError

    def fit_model(self, X_train, y_train, **kwargs):
        # 使用 cv_uni 进行交叉验证
        raise NotImplementedError

    def evaluate_model(self, model, X_test, y_test, beta_true):
        # 计算指标：外样本误差、TPR、FPR、F1、运行时间
        raise NotImplementedError

    def run(self):
        # 运行所有重复，汇总结果
        pass

    def aggregate_results(self):
        # 计算多次重复的均值 ± 标准差，返回 DataFrame
        pass

    def save_results(self, filepath):
        # 保存到 CSV
        pass

    def plot_results(self, output_dir):
        # 生成图像
        pass
```

### 阶段三：线性高斯实验 - 重点验证分组约束和自适应加权

**文件：`experiments/linear_experiment.py`**

**核心：四种方法对比验证升级效果**

在每个数据场景中，都对比四种配置：
| 方法 | 分组约束 | 自适应加权 | 说明 |
|------|----------|------------|------|
| UniLasso (baseline) | ❌ | ❌ | 无升级，普通 Lasso 类似 |
| UniLasso + 自适应加权 | ❌ | ✅ | 只看自适应加权效果 |
| UniLasso + 分组约束 | ✅ | ❌ | 只看分组约束效果 |
| UniLasso + 双升级 | ✅ | ✅ | 完整版算法 |

**实验场景设计（针对分组约束和自适应加权）：**

| 场景 | 维度 | 稀疏性 | 相关性结构 | 信噪比 | 重复次数 | 研究目的 |
|------|------|--------|------------|--------|----------|----------|
| 独立特征 | n=500, p=100 | 10 | 无 | 中等 (2.0) | 30 | 无相关时，自适应加权是否仍有帮助 |
| AR(1) 相关 | n=500, p=100 | 10 | ρ=0.7 | 中等 | 30 | 连续相关性下分组效果 |
| 块结构相关 **(重点)** | n=500, p=100 | 10个真变量，1个/块 | 块内 0.7 | 中等 | 30 | 分组约束能否正确整组选入真变量 |
| 块结构假相关 | n=500, p=100 | 5个真变量都在第1块 | 块内 0.7 | 中等 | 30 | 分组约束会不会误选整个块？ |
| 符号不一致 **(重点)** | n=300, p=20 | 2个真变量 (符号相反) | 两者高度相关 (0.95) | 中等 | 30 | 分组约束能否处理高度相关但符号相反 |
| 因子模型 | n=300, p=50 | 10 | 通过潜因子 | 中等 | 30 | 隐因子结构下的表现 |
| **高维** | n=300, p=1000 | 20 | 块结构 | 低/中/高 | 20 | 高维下两个升级是否还有效 |

**评估指标 (特别关注变量选择)：**
- 外样本 MSE
- **真阳性率 (TPR)** - 真非零系数被选中的比例
- **假阳性率 (FPR)** - 零系数被错误选中的比例
- F1-score（变量选择）
- 系数估计的均方误差
- **分组准确率** - 真变量同组是否一起选入/剔除
- 计算时间

**预期结果：**
- 在有相关性的数据上，分组约束 + 自适应加权应该同时提高 TPR 并降低 FPR
- 在独立特征上，自适应加权应该仍能提高变量选择准确性

### 阶段四：GLM 扩展实验 - 验证各GLM族下升级仍然有效

**文件：`experiments/glm_experiment.py`**

**核心设计：每个 GLM 族都对比四种配置**

和线性实验相同，每个场景都对比：
- ❌无分组 ❌无自适应 → baseline
- ❌无分组 ✅自适应加权 → 只看自适应效果
- ✅有分组 ❌无自适应 → 只看分组效果
- ✅有分组 ✅自适应加权 → 完整版

**每个 GLM 族单独实验：**

| GLM 族 | 数据生成 | 样本量 | 维度 | 评估指标 | 重复次数 |
|--------|---------|--------|------|----------|----------|
| **Gaussian** | 线性稀疏 + 块相关 | n=500 | p=100 | 外样本 MSE, TPR/FPR | 20 |
| **Binomial (逻辑)** | 线性 → logits → 二元响应 | n=500 | p=100 | Deviance, Accuracy, AUC, TPR/FPR | 20 |
| **Poisson** | log λ = Xβ → 计数数据 | n=500 | p=100 | Deviance, 计数 RMSE, TPR/FPR | 20 |
| **Multinomial** | 多类线性预测器 | n=500 | p=100 (3类) | 分类准确率，多类 AUC | 20 |
| **Cox** | 比例风险 + 删失 | n=500 | p=100 | 一致性指数 (C-index) | 20 |

**研究问题：**
- 在非高斯 GLM 下，分组约束和自适应加权是否仍然有效？
- 不同 GLM 族上的性能提升幅度是否一致？

**预期结果：**
- 在所有 GLM 族上，两个升级都应该带来变量选择准确性提升

### 阶段五：非线性实验 - 验证非线性场景下升级仍然有效

**文件：`experiments/nonlinear_experiment.py`**

**二维比较设计：**
1. **横向**：单变量模型类型对比：linear vs spline vs tree
2. **纵向**：四种方法配置对比（和线性实验一致）：无升级 vs 仅自适应 vs 仅分组 vs 双升级

**数据类型：**
1. **纯非线性** - 所有相关特征都是非线性关系
   - 模式：正弦波（非单调）、二次函数（曲率）、阶梯函数（阈值）、混合
2. **混合非线性** **(更实际)** - 部分线性、部分非线性、部分无关
3. **块相关非线性** **(重点)** - 真非线性特征在相关块中，测试分组约束是否仍然有效
4. **零模型** - 所有特征都无关，测试 FDR 控制

**完整比较设计：**

| 单变量模型 | 无约束无自适应 | 无约束有自适应 | 有约束无自适应 | 有约束有自适应 |
|------------|----------------|----------------|----------------|----------------|
| linear | ✓ | ✓ | ✓ | ✓ |
| spline | ✓ | ✓ | ✓ | ✓ |
| tree | ✓ | ✓ | ✓ | ✓ |

**研究问题：**
- 在真实非线性数据上，非线性单变量模型（spline/tree）是否比 linear 提升预测精度？
- 分组约束和自适应加权在非线性场景下是否仍然有效？
- spline 和 tree 各在什么场景下表现更好？

**评估指标：**
- 外样本预测误差（和线性基线比较）
- 特征选择准确率（能否正确找出非线性相关特征？）
- **分组准确率**（当非线性特征在相关块中，能否整组正确选择？）
- 计算时间对比

**预期结果：**
- 在真非线性数据上，spline/tree > linear
- 无论单变量模型是哪种，分组约束 + 自适应加权都能进一步提升变量选择准确性

### 阶段六：可视化和结果分析

**文件：`experiments/visualization.py`**

共享绘图函数：
1. 箱线图展示多次重复间指标分布
2. 性能热图（稀疏度 × 信噪比）
3. 变量选择的 TPR-FPR 曲线（类似ROC）
4. 拟合非线性函数和真实函数对比示例图
5. 方法对比汇总条形图

**输出：**
- 汇总结果为 CSV 文件
- LaTeX 格式汇总表格
- 所有图像保存为 PNG (300dpi) 和 PDF

### 阶段七：运行脚本

**`scripts/run_linear_experiment.py`** - 可配置参数运行线性实验的命令行接口
**`scripts/run_glm_experiment.py`** - 运行 GLM 实验
**`scripts/run_nonlinear_experiment.py`** - 运行非线性实验

## 新建文件列表

| 路径 | 用途 | 预估行数 |
|------|------|----------|
| `experiments/__init__.py` | 包初始化 | ~10 行 |
| `experiments/base_experiment.py` | 实验基类 | ~150 行 |
| `experiments/linear_experiment.py` | 线性高斯实验 | ~200 行 |
| `experiments/glm_experiment.py` | 所有 GLM 族实验 | ~200 行 |
| `experiments/nonlinear_experiment.py` | 非线性对比实验 | ~250 行 |
| `experiments/visualization.py` | 共享绘图工具 | ~150 行 |
| `scripts/run_linear_experiment.py` | 线性运行脚本 | ~50 行 |
| `scripts/run_glm_experiment.py` | GLM 运行脚本 | ~50 行 |
| `scripts/run_nonlinear_experiment.py` | 非线性运行脚本 | ~50 行 |

**需要修改的文件：**
- `data_generators.py` - 添加约 200 行新生成器（追加到现有文件末尾）

## 实现顺序

1. 扩展 `data_generators.py`，添加新的 GLM 和非线性生成器
2. 创建 `experiments/` 目录结构和 `base_experiment.py`
3. 实现 `visualization.py` 共享工具
4. 实现 `linear_experiment.py`
5. 实现 `glm_experiment.py`
6. 实现 `nonlinear_experiment.py`
7. 在 `scripts/` 中创建运行脚本
8. 在 `experiments/results/` 添加 `.gitignore` 排除输出文件

## 验证

实现完成后，我们可以：
1. 用较少重复次数运行每个实验，验证其正确性
2. 检查所有指标计算正确
3. 验证图像成功生成
4. 确保固定随机种子下框架可复现

## 实验设计要点

本实验设计统计上合理：
- 使用多次独立重复（20-30次）来估计均值性能和变异度
- 使用训练/测试分割进行诚实的外样本评估
- 系统覆盖了三个算法版本（线性、GLM扩展、非线性）
- **特别设计了四组对比，清晰分离出分组约束和自适应加权各自的贡献**
  - ❌无约束 ❌无自适应 → 基线
  - ❌无约束 ✅自适应 → 看自适应加权单独效果
  - ✅有约束 ❌无自适应 → 看分组约束单独效果
  - ✅有约束 ✅自适应 → 完整版，看组合效果

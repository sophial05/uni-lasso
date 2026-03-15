# GroupAdaUniLasso 全GLM家族升级计划

## 项目概述
将 UniLasso 升级为支持全 GLM 家族的 GroupAdaUniLasso，新增自适应惩罚和分组约束功能。

---

## 第一阶段：核心功能实现 ✅ 已完成

**完成日期：2026-03-15**

### 1.1 新增辅助工具函数 ✅

| 函数名 | 功能 | 状态 |
|--------|------|------|
| `_compute_feature_significance_weights` | 计算特征显著性权重（t统计量、p值、相关性） | ✅ |
| `_greedy_correlation_grouping` | 基于相关性的贪心分组算法 | ✅ |
| `_compute_group_penalty_weights` | 计算组惩罚权重和主导符号 | ✅ |

### 1.2 改造核心求解器 ✅

| 求解器 | 升级内容 | 状态 |
|--------|----------|------|
| `_fit_numba_lasso_path` | 双层非对称软阈值，GLM家族支持 | ✅ |
| `_fit_numba_lasso_path_accelerated` | 加速版本（Numba优化） | ✅ |
| `_fit_pytorch_lasso_path` | PyTorch后端同步升级 | ✅ |

### 1.3 升级 fit_uni 和 cv_uni 接口 ✅

| 功能 | 说明 | 状态 |
|------|------|------|
| 自适应惩罚参数 | `adaptive_weighting`, `weight_method`, `weight_max_scale` | ✅ |
| 分组约束参数 | `enable_group_constraint`, `corr_threshold`, `group_penalty`, `max_group_size` | ✅ |
| 向后兼容 | 新功能默认关闭，保持原有行为 | ✅ |

### 1.4 结果增强 ✅

| 属性 | 说明 | 状态 |
|------|------|------|
| `groups` | 特征分组信息 | ✅ |
| `group_signs` | 每组的主导符号 | ✅ |

### 1.5 基础验证 ✅

- ✅ 所有向后兼容性测试通过
- ✅ 新功能基础测试通过

---

## 第二阶段：全GLM家族适配 ✅ 已完成

**完成日期：2026-03-15**

### 2.1 更新配置文件 ✅

**文件：** `unilasso/config.py`

```python
VALID_FAMILIES = {"gaussian", "binomial", "multinomial", "poisson", "cox"}
```

### 2.2 新增 Poisson 和 Multinomial 支持 ✅

**文件：** `unilasso/univariate_regression.py`

| 函数 | 功能 | 状态 |
|------|------|------|
| `leave_one_out_poisson()` | 泊松回归 LOO 拟合 | ✅ |
| `leave_one_out_multinomial()` | 多分类回归 LOO 拟合 | ✅ |
| `fit_loo_univariate_models()` | 分派函数更新 | ✅ |

### 2.3 泛化求解器支持各GLM损失函数 ✅

**文件：** `unilasso/solvers.py`

| GLM家族 | 损失函数 | 状态 |
|---------|----------|------|
| Gaussian | MSE 损失 | ✅ |
| Binomial | Logistic 损失 | ✅ |
| Poisson | 对数线性损失 | ✅ |
| Multinomial | Softmax 损失（one-vs-rest）| ✅ |
| Cox | 部分似然损失（保留原始）| ✅ |

### 2.4 更新 fit_uni 和 cv_uni 移除 Gaussian 限制 ✅

**文件：** `unilasso/uni_lasso.py`

| 功能 | 说明 | 状态 |
|------|------|------|
| 移除回退逻辑 | 不再限制 family="gaussian" | ✅ |
| 交叉验证损失 | 支持各 GLM 的损失计算 | ✅ |
| lambda_max 计算 | 支持各 GLM 的正则化路径 | ✅ |
| predict 函数 | 支持各 GLM 的预测 | ✅ |

### 2.5 修复 _prepare_unilasso_input ✅

**问题：** `_get_glm_family()` 不支持 poisson 和 multinomial

**修复方案：**
```python
# 只对 adelie 支持的家族获取 glm_family 和 constraints
if family in {"gaussian", "binomial", "cox"}:
    glm_family = _get_glm_family(family, y)
    constraints = [ad.constraint.lower(b=np.zeros(1)) for _ in range(X.shape[1])]
else:
    glm_family = None
    constraints = None
```

### 2.6 新增全GLM测试用例 ✅

**文件：** `tests/test_glm_families.py`

| 测试 | 状态 |
|------|------|
| Gaussian 回归 | ✅ |
| Gaussian + 新功能 | ✅ |
| Binomial 分类 | ✅ |
| Binomial + 新功能 | ✅ |
| Poisson 回归 | ✅ |
| Poisson + 新功能 | ✅ |
| Multinomial 分类 | ✅ |
| 后端选择 | ✅ |
| 配置验证 | ✅ |

---

## 测试结果汇总 ✅

| 测试文件 | 通过数/总数 | 状态 |
|----------|-------------|------|
| `test_glm_families.py` | 9/9 | ✅ |
| `test_groupada.py` | 7/7 | ✅ |
| `test_fit_uni.py` | 9/9 | ✅ |
| **总计** | **25/25** | **✅** |

---

## 第三阶段：非线性单变量模型扩展 ✅ 已完成

**完成日期：2026-03-15**

### 3.1 非线性单变量模型扩展 ✅

| 任务 | 状态 |
|------|------|
| 新增 `univariate_model` 参数 (linear/spline/tree) | ✅ |
| 实现单变量样条回归拟合 | ✅ |
| 实现单变量浅决策树拟合 | ✅ |
| 适配非线性模型的LOO拟合 | ✅ |
| 适配非线性模型的显著性计算 | ✅ |

**新增函数：**
- `leave_one_out_spline()` - 样条回归 LOO 拟合，支持 B-spline 基函数
- `leave_one_out_tree()` - 决策树 LOO 拟合，使用决策树桩（stump）
- `_bspline_basis()` - B-spline 基函数计算
- `_weighted_least_squares()` - 加权最小二乘回归
- `_find_best_split()` - 决策树最佳分裂点查找
- `_DecisionTreeStump` - 决策树桩类

### 3.2 接口扩展与兼容性处理 ✅

| 任务 | 状态 |
|------|------|
| 新增 `univariate_model`, `spline_df`, `tree_max_depth` 参数 | ✅ |
| 参数合法性校验 | ✅ |

**修改的接口：**
- `fit_uni()` - 新增 `univariate_model`, `spline_df`, `spline_degree`, `tree_max_depth` 参数
- `cv_uni()` - 新增 `univariate_model`, `spline_df`, `spline_degree`, `tree_max_depth` 参数
- `fit_univariate_models()` - 支持非线性模型参数
- `_prepare_unilasso_input()` - 支持非线性模型参数
- `fit_loo_univariate_models()` - 支持非线性模型分派

**向后兼容性：**
- `univariate_model` 默认值为 `"linear"`，保持原有行为
- 当使用非线性模型时，自动禁用原始 adelie 回退
- 所有原有测试通过，验证向后兼容性

### 3.3 新增配置 ✅

**文件：** `unilasso/config.py`
```python
VALID_UNIVARIATE_MODELS = {"linear", "spline", "tree"}
```

### 3.4 新增测试用例 ✅

**文件：** `tests/test_nonlinear.py`

| 测试类 | 功能 | 状态 |
|--------|------|------|
| `TestSplineRegression` | 样条回归基础测试 | ✅ |
| `TestTreeRegression` | 决策树基础测试 | ✅ |
| `TestFitUniNonlinear` | fit_uni 非线性模型测试 | ✅ |
| `TestCVUniNonlinear` | cv_uni 非线性模型测试 | ✅ |
| `TestBackwardCompatibility` | 向后兼容性测试 | ✅ |

**测试统计：** 17 个测试全部通过

---

## 测试结果汇总 ✅

| 测试文件 | 通过数/总数 | 状态 |
|----------|-------------|------|
| `test_glm_families.py` | 9/9 | ✅ |
| `test_groupada.py` | 7/7 | ✅ |
| `test_fit_uni.py` | 9/9 | ✅ |
| `test_nonlinear.py` | 17/17 | ✅ |
| **总计** | **42/42** | **✅** |

---

## 涉及文件清单

| 文件路径 | 修改/新增 | 说明 |
|----------|-----------|------|
| `unilasso/config.py` | 修改 | 扩展 VALID_FAMILIES、新增 VALID_UNIVARIATE_MODELS |
| `unilasso/uni_lasso.py` | 修改 | 核心算法升级，新增非线性模型参数 |
| `unilasso/solvers.py` | 修改 | 求解器泛化 |
| `unilasso/univariate_regression.py` | 修改 | 新增 Poisson/Multinomial/Spline/Tree |
| `tests/test_glm_families.py` | 新增 | 全GLM测试 |
| `tests/test_groupada.py` | 新增 | GroupAda测试 |
| `tests/test_fit_uni.py` | 新增 | fit_uni测试 |
| `tests/test_nonlinear.py` | 新增 | 非线性模型测试 |
| `PROJECT_PLAN.md` | 新增 | 本文档 |

---

## 功能支持矩阵

### GLM 家族支持

| GLM 家族 | fit_unilasso | fit_uni | cv_unilasso | cv_uni | 自适应惩罚 | 分组约束 | 非线性模型 |
|----------|--------------|---------|-------------|--------|------------|----------|----------|
| Gaussian | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Binomial | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Poisson | ❌* | ✅ | ❌* | ✅ | ✅ | ✅ | ✅ |
| Multinomial | ❌* | ✅ | ❌* | ✅ | ✅ | ✅ | ✅ |
| Cox | ✅ | ✅** | ✅ | ✅** | ✅ | ✅ | ❌ |

*: 仅在新功能开启时可用（使用自定义求解器）
**: Cox 使用原始 adelie 实现（新功能关闭时）

### 非线性单变量模型支持

| 单变量模型 | fit_uni | cv_uni | 自适应惩罚 | 分组约束 |
|-----------|---------|--------|------------|----------|
| Linear (线性) | ✅ | ✅ | ✅ | ✅ |
| Spline (样条) | ✅ | ✅ | ✅ | ✅ |
| Tree (决策树) | ✅ | ✅ | ✅ | ✅ |

---

## 快速开始

```python
import numpy as np
from unilasso.uni_lasso import fit_uni, cv_uni

# Poisson 回归
X, y = ...  # 计数数据
result = fit_uni(X, y, family="poisson",
                 adaptive_weighting=True,
                 enable_group_constraint=True)

# Multinomial 分类
X, y = ...  # 多分类数据
cv_result = cv_uni(X, y, family="multinomial",
                   adaptive_weighting=True)

# 样条回归（非线性）
X, y = ...  # 非线性关系数据
result_spline = fit_uni(X, y, family="gaussian",
                         univariate_model="spline",
                         spline_df=5,
                         spline_degree=3)

# 决策树回归（非线性）
result_tree = fit_uni(X, y, family="gaussian",
                       univariate_model="tree",
                       tree_max_depth=2)

# 非线性 + 自适应惩罚 + 分组约束
result_combined = fit_uni(X, y, family="gaussian",
                           univariate_model="spline",
                           spline_df=5,
                           adaptive_weighting=True,
                           enable_group_constraint=True)
```

---

## 参考文献

- Chatterjee, S., Hastie, T., & Tibshirani, R. (2025). Univariate-Guided Sparse Regression. arXiv:2501.18360

---

---

## 快速总结

### 已完成的三个阶段：

1. **第一阶段：核心功能实现** ✅
   - 自适应惩罚
   - 分组约束
   - 双层非对称软阈值

2. **第二阶段：全GLM家族适配** ✅
   - Gaussian, Binomial, Poisson, Multinomial, Cox 全部支持

3. **第三阶段：非线性单变量模型扩展** ✅
   - Spline（样条）回归
   - Tree（决策树）回归
   - 与自适应惩罚、分组约束完美结合

### 测试统计：
- 总测试数：42 个
- 通过数：42 个
- 通过率：100%

**最后更新：** 2026-03-15

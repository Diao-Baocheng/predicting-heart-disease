# 数据加载器使用说明 (Data Loader Documentation)

## 简介

`DataLoader` 是一个用于加载和分析心脏病预测数据集的工具类。它提供了以下功能：

- ✅ 使用 `pathlib` 进行路径处理
- ✅ 自动加载 `data/raw/heart.csv` 数据文件
- ✅ 自动报告数据集中的缺失值情况
- ✅ 自动报告每个特征的类型（数值型/分类型）
- ✅ 提供数据统计摘要

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 基本用法

```python
from src.data_loader import DataLoader

# 创建数据加载器实例（使用默认路径）
loader = DataLoader()

# 加载数据（会自动显示缺失值和特征类型报告）
data = loader.load_data()

# 查看数据
print(data.head())
```

### 使用自定义路径

```python
from pathlib import Path
from src.data_loader import DataLoader

# 指定自定义数据路径
custom_path = Path("path/to/your/data.csv")
loader = DataLoader(data_path=custom_path)

# 加载数据
data = loader.load_data()
```

### 查看数据摘要

```python
from src.data_loader import DataLoader

loader = DataLoader()
data = loader.load_data()

# 显示统计摘要
loader.get_data_summary()
```

## 输出示例

运行 `python src/data_loader.py` 会输出：

```
正在从 /path/to/data/raw/heart.csv 加载数据...
成功加载 25 条记录，14 个特征

============================================================
缺失值报告 (Missing Values Report)
============================================================
✓ 数据完整，无缺失值

总缺失值数量: 0
============================================================

============================================================
特征类型报告 (Feature Types Report)
============================================================
    特征名称    数据类型  唯一值数量               特征类型
     age   int64     20   数值特征 (Numerical)
     sex   int64      2 分类特征 (Categorical)
      cp   int64      4 分类特征 (Categorical)
...
============================================================
```

## 目录结构

```
predicting-heart-disease/
├── data/
│   ├── raw/
│   │   └── heart.csv          # 原始数据文件
│   └── processed/             # 处理后的数据（待实现）
├── src/
│   ├── __init__.py
│   └── data_loader.py         # 数据加载类
├── requirements.txt           # 项目依赖
└── test_data_loader.py        # 测试脚本
```

## API 文档

### DataLoader 类

#### `__init__(data_path: Optional[Path] = None)`

初始化数据加载器。

**参数:**
- `data_path` (Optional[Path]): 数据文件路径。如果为 None，使用默认路径 `data/raw/heart.csv`

#### `load_data() -> pd.DataFrame`

加载数据文件并自动报告数据质量信息。

**返回:**
- `pd.DataFrame`: 加载的数据框

**异常:**
- `FileNotFoundError`: 当数据文件不存在时

#### `report_missing_values() -> None`

报告数据集中的缺失值情况，包括：
- 每个特征的缺失值数量
- 缺失值百分比
- 总缺失值数量

#### `report_feature_types() -> None`

报告数据集中每个特征的类型信息，包括：
- 特征名称
- 数据类型
- 唯一值数量
- 推断的特征类型（数值型/分类型）

#### `get_data_summary() -> None`

显示数据的统计摘要（调用 pandas 的 `describe()` 方法）。

## 数据集说明

`heart.csv` 包含心脏病预测相关的特征：

- `age`: 年龄
- `sex`: 性别 (1 = 男性; 0 = 女性)
- `cp`: 胸痛类型 (0-3)
- `trestbps`: 静息血压 (mm Hg)
- `chol`: 血清胆固醇 (mg/dl)
- `fbs`: 空腹血糖 > 120 mg/dl (1 = true; 0 = false)
- `restecg`: 静息心电图结果 (0-2)
- `thalach`: 达到的最大心率
- `exang`: 运动诱发的心绞痛 (1 = yes; 0 = no)
- `oldpeak`: 运动相对于休息的 ST 段压低
- `slope`: 运动 ST 段的斜率
- `ca`: 主要血管数量 (0-3)
- `thal`: 地中海贫血 (1 = 正常; 2 = 固定缺陷; 3 = 可逆缺陷)
- `target`: 心脏病诊断 (1 = 有; 0 = 无)

## 测试

运行测试脚本：

```bash
python test_data_loader.py
```

## 许可证

请参考项目根目录的 LICENSE 文件。

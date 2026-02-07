# 心脏病预测项目

通过机器学习算法预测心脏病发生概率的完整项目。

## 项目结构

```
predicting-heart-disease/
│
├── main.py                    # 主程序入口
├── data_processing.py         # 数据处理模块
├── model.py                   # 模型训练和预测模块
├── stats_analysis.py          # 统计分析模块
├── visualization.py           # 数据可视化模块
├── requirements.txt           # 项目依赖
│
├── data/                      # 数据目录
│   └── playground-series-s6e2/
│       ├── train.csv          # 训练数据
│       ├── test.csv           # 测试数据
│       └── sample_submission.csv
│
├── plots/                     # 可视化图表输出目录（自动生成）
└── submission.csv             # 预测结果提交文件（自动生成）
```

## 模块说明

### 1. data_processing.py - 数据处理模块
- 数据加载和探索
- 数据预处理和特征工程
- 缺失值检查
- 数据采样

### 2. model.py - 模型模块
- 随机森林分类器
- 梯度提升分类器
- 模型集成预测
- 特征重要性分析
- 预测结果保存

### 3. stats_analysis.py - 统计分析模块
- 基础统计信息
- 特征统计分析
- 相关性分析
- 分布分析
- 异常值检测
- 类别分布分析

### 4. visualization.py - 可视化模块
- 目标变量分布图
- 特征分布图
- 相关性矩阵热力图
- 特征重要性图
- 预测结果分布图
- 箱线图

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行项目

```bash
python main.py
```

## 项目流程

1. **数据加载与处理**：加载训练和测试数据，进行预处理
2. **统计分析**：对数据进行全面的统计分析
3. **模型训练**：训练随机森林和梯度提升模型
4. **生成预测**：使用集成方法生成预测结果
5. **可视化报告**：生成完整的数据和结果可视化图表

## 输出文件

- `submission.csv`：Kaggle提交格式的预测结果
- `plots/`目录：包含所有可视化图表
  - `target_distribution.png`：目标变量分布
  - `feature_distributions.png`：特征分布
  - `correlation_matrix.png`：相关性矩阵
  - `feature_importance.png`：特征重要性
  - `prediction_distribution.png`：预测结果分布

## 数据集信息

- 训练集：630,000条记录
- 测试集：270,000条记录
- 特征数：13个
- 目标变量：Heart Disease（Presence/Absence）

## 主要特征

- Age：年龄
- Sex：性别
- Chest pain type：胸痛类型
- BP：血压
- Cholesterol：胆固醇
- FBS over 120：空腹血糖>120
- EKG results：心电图结果
- Max HR：最大心率
- Exercise angina：运动性心绞痛
- ST depression：ST段压低
- Slope of ST：ST斜率
- Number of vessels fluro：荧光血管数
- Thallium：铊

## 性能优化

为了加快训练速度，项目使用了数据采样策略（默认10万样本）。
如需使用完整数据集，请修改 `main.py` 中的参数：

```python
processor, X_train, y_train, X_test, test_ids = load_and_process_data(
    train_path=train_path,
    test_path=test_path,
    use_sample=False,  # 改为 False
    sample_size=100000
)
```

## License

MIT License

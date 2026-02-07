"""
Data Processing Module
Responsible for data loading, preprocessing and feature engineering
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DataProcessor:
    """数据处理器类"""
    
    def __init__(self, train_path, test_path):
        """
        初始化数据处理器
        
        Args:
            train_path: 训练数据路径
            test_path: 测试数据路径
        """
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.test_ids = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """加载训练和测试数据"""
        print("="*60)
        print("加载数据")
        print("="*60)
        
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)
        
        print(f"训练数据形状: {self.train_df.shape}")
        print(f"测试数据形状: {self.test_df.shape}")
        
        return self.train_df, self.test_df
    
    def explore_data(self):
        """数据探索"""
        print("\n" + "="*60)
        print("数据探索")
        print("="*60)
        
        print("\n训练数据前5行:")
        print(self.train_df.head())
        
        print("\n数据信息:")
        print(self.train_df.info())
        
        print("\n数据统计描述:")
        print(self.train_df.describe())
        
        print("\n目标变量分布:")
        print(self.train_df['Heart Disease'].value_counts())
        
        return self.train_df.describe()
    
    def check_missing_values(self):
        """检查缺失值"""
        print("\n" + "="*60)
        print("缺失值检查")
        print("="*60)
        
        train_missing = self.train_df.isnull().sum()
        test_missing = self.test_df.isnull().sum()
        
        print("\n训练集缺失值:")
        print(train_missing[train_missing > 0] if train_missing.sum() > 0 else "无缺失值")
        
        print("\n测试集缺失值:")
        print(test_missing[test_missing > 0] if test_missing.sum() > 0 else "无缺失值")
        
        return train_missing, test_missing
    
    def preprocess_data(self):
        """数据预处理"""
        print("\n" + "="*60)
        print("数据预处理")
        print("="*60)
        
        # 分离特征和目标变量
        self.X_train = self.train_df.drop(['id', 'Heart Disease'], axis=1)
        self.y_train = self.train_df['Heart Disease']
        
        # 编码目标变量：Presence -> 1, Absence -> 0
        y_train_encoded = self.label_encoder.fit_transform(self.y_train)
        
        # 测试数据
        self.X_test = self.test_df.drop(['id'], axis=1)
        self.test_ids = self.test_df['id']
        
        print(f"\n特征数量: {self.X_train.shape[1]}")
        print(f"特征列: {self.X_train.columns.tolist()}")
        print(f"\n目标变量编码后的类别分布:")
        print(pd.Series(y_train_encoded).value_counts())
        
        return self.X_train, y_train_encoded, self.X_test, self.test_ids
    
    def create_sample_data(self, sample_size=100000, random_state=42):
        """
        创建采样数据用于快速训练
        
        Args:
            sample_size: 采样大小
            random_state: 随机种子
            
        Returns:
            采样后的特征和标签
        """
        if self.X_train is None:
            raise ValueError("请先调用 preprocess_data() 方法")
        
        sample_size = min(sample_size, len(self.X_train))
        sample_indices = np.random.RandomState(random_state).choice(
            len(self.X_train), sample_size, replace=False
        )
        
        X_train_sample = self.X_train.iloc[sample_indices]
        y_train_sample = self.label_encoder.transform(
            self.y_train.iloc[sample_indices]
        )
        
        print(f"\n创建采样数据: {sample_size} 样本")
        
        return X_train_sample, y_train_sample
    
    def get_feature_names(self):
        """获取特征名称"""
        return self.X_train.columns.tolist()


def load_and_process_data(train_path, test_path, use_sample=True, sample_size=100000):
    """
    便捷函数：加载和处理数据
    
    Args:
        train_path: 训练数据路径
        test_path: 测试数据路径
        use_sample: 是否使用采样数据
        sample_size: 采样大小
        
    Returns:
        处理器对象，训练特征，训练标签，测试特征，测试ID
    """
    processor = DataProcessor(train_path, test_path)
    processor.load_data()
    processor.explore_data()
    processor.check_missing_values()
    X_train, y_train, X_test, test_ids = processor.preprocess_data()
    
    if use_sample:
        X_train, y_train = processor.create_sample_data(sample_size)
    
    return processor, X_train, y_train, X_test, test_ids

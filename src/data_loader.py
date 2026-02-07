"""
数据加载模块 (Data Loader Module)

该模块提供数据加载和基本分析功能，包括：
- 使用 pathlib 进行路径处理
- 读取 CSV 数据文件
- 自动报告缺失值统计
- 自动报告特征类型信息
"""

from pathlib import Path
import pandas as pd
from typing import Optional


class DataLoader:
    """
    数据加载类 (Data Loader Class)
    
    用于加载心脏病预测数据集，并提供基本的数据质量报告功能。
    
    Attributes:
        data_path (Path): 数据文件路径
        data (pd.DataFrame): 加载的数据框
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        初始化数据加载器
        
        Args:
            data_path (Optional[Path]): 数据文件路径。如果为 None，则使用默认路径 data/raw/heart.csv
        """
        if data_path is None:
            # 获取项目根目录（假设 src 在项目根目录下）
            project_root = Path(__file__).parent.parent
            self.data_path = project_root / "data" / "raw" / "heart.csv"
        else:
            self.data_path = Path(data_path)
        
        self.data: Optional[pd.DataFrame] = None
    
    def load_data(self) -> pd.DataFrame:
        """
        加载数据文件
        
        Returns:
            pd.DataFrame: 加载的数据框
            
        Raises:
            FileNotFoundError: 当数据文件不存在时
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        print(f"正在从 {self.data_path} 加载数据...")
        self.data = pd.read_csv(self.data_path)
        print(f"成功加载 {len(self.data)} 条记录，{len(self.data.columns)} 个特征\n")
        
        # 自动报告数据信息
        self.report_missing_values()
        self.report_feature_types()
        
        return self.data
    
    def report_missing_values(self) -> None:
        """
        报告数据集中的缺失值情况
        
        打印每个特征的缺失值数量和百分比
        """
        if self.data is None:
            print("警告: 尚未加载数据，请先调用 load_data() 方法")
            return
        
        print("=" * 60)
        print("缺失值报告 (Missing Values Report)")
        print("=" * 60)
        
        missing_counts = self.data.isnull().sum()
        missing_percentages = (missing_counts / len(self.data)) * 100
        
        # 创建缺失值报告表格
        missing_info = pd.DataFrame({
            '特征名称': missing_counts.index,
            '缺失数量': missing_counts.values,
            '缺失百分比': missing_percentages.values
        })
        
        # 只显示有缺失值的特征
        missing_info = missing_info[missing_info['缺失数量'] > 0]
        
        if len(missing_info) == 0:
            print("✓ 数据完整，无缺失值")
        else:
            print(missing_info.to_string(index=False))
        
        print(f"\n总缺失值数量: {missing_counts.sum()}")
        print("=" * 60)
        print()
    
    def report_feature_types(self) -> None:
        """
        报告数据集中每个特征的类型
        
        打印每个特征的数据类型、唯一值数量等信息
        """
        if self.data is None:
            print("警告: 尚未加载数据，请先调用 load_data() 方法")
            return
        
        print("=" * 60)
        print("特征类型报告 (Feature Types Report)")
        print("=" * 60)
        
        feature_info = []
        for column in self.data.columns:
            col_dtype = str(self.data[column].dtype)
            unique_count = self.data[column].nunique()
            
            # 推断特征类型
            if col_dtype in ['int64', 'float64']:
                if unique_count <= 10:
                    feature_type = '分类特征 (Categorical)'
                else:
                    feature_type = '数值特征 (Numerical)'
            else:
                feature_type = '分类特征 (Categorical)'
            
            feature_info.append({
                '特征名称': column,
                '数据类型': col_dtype,
                '唯一值数量': unique_count,
                '特征类型': feature_type
            })
        
        feature_df = pd.DataFrame(feature_info)
        print(feature_df.to_string(index=False))
        print("=" * 60)
        print()
    
    def get_data_summary(self) -> None:
        """
        获取数据的统计摘要
        
        打印数据的基本统计信息
        """
        if self.data is None:
            print("警告: 尚未加载数据，请先调用 load_data() 方法")
            return
        
        print("=" * 60)
        print("数据统计摘要 (Data Summary)")
        print("=" * 60)
        print(self.data.describe())
        print("=" * 60)
        print()


if __name__ == "__main__":
    # 示例用法
    loader = DataLoader()
    data = loader.load_data()
    loader.get_data_summary()

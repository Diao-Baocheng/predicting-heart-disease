"""
演示缺失值报告功能 (Demonstrate Missing Values Reporting)

该脚本创建一个包含缺失值的测试数据集，并演示 DataLoader 的缺失值报告功能
"""

from pathlib import Path
import pandas as pd
from src.data_loader import DataLoader


def create_test_data_with_missing_values():
    """创建包含缺失值的测试数据"""
    print("创建包含缺失值的测试数据...")
    
    # 创建临时目录
    temp_dir = Path("/tmp/heart_disease_test")
    temp_dir.mkdir(exist_ok=True)
    
    # 创建包含缺失值的数据
    data = {
        'age': [63, 37, None, 56, 57],
        'sex': [1, 1, 0, 1, None],
        'cp': [3, 2, 1, 1, 0],
        'trestbps': [145, None, 130, 120, 120],
        'chol': [233, 250, 204, 236, 354],
        'target': [1, 1, 1, None, 1]
    }
    
    df = pd.DataFrame(data)
    
    # 保存到临时文件
    test_file = temp_dir / "heart_test.csv"
    df.to_csv(test_file, index=False)
    
    print(f"测试数据已保存到: {test_file}\n")
    
    return test_file


def test_missing_values_report():
    """测试缺失值报告功能"""
    # 创建测试数据
    test_file = create_test_data_with_missing_values()
    
    # 使用 DataLoader 加载数据
    print("=" * 60)
    print("使用 DataLoader 加载包含缺失值的数据")
    print("=" * 60)
    
    loader = DataLoader(data_path=test_file)
    data = loader.load_data()
    
    print("\n数据预览:")
    print(data)
    print()


if __name__ == "__main__":
    test_missing_values_report()

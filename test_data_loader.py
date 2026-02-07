"""
测试数据加载器 (Test Data Loader)

该脚本演示如何使用 DataLoader 类
"""

from pathlib import Path
from src.data_loader import DataLoader


def test_default_path():
    """测试使用默认路径加载数据"""
    print("测试1: 使用默认路径加载数据")
    print("-" * 60)
    loader = DataLoader()
    data = loader.load_data()
    
    print(f"数据形状: {data.shape}")
    print(f"列名: {list(data.columns)}")
    print("-" * 60)
    print()


def test_custom_path():
    """测试使用自定义路径加载数据"""
    print("测试2: 使用自定义路径加载数据")
    print("-" * 60)
    
    # 构建自定义路径
    custom_path = Path(__file__).parent / "data" / "raw" / "heart.csv"
    loader = DataLoader(data_path=custom_path)
    data = loader.load_data()
    
    print(f"数据形状: {data.shape}")
    print("-" * 60)
    print()


def test_individual_reports():
    """测试单独调用报告方法"""
    print("测试3: 单独调用报告方法")
    print("-" * 60)
    
    loader = DataLoader()
    # load_data() 会自动调用报告方法并设置 self.data
    data = loader.load_data()
    
    # 获取数据摘要
    loader.get_data_summary()
    print("-" * 60)
    print()


if __name__ == "__main__":
    test_default_path()
    # test_custom_path()  # 取消注释以测试自定义路径
    # test_individual_reports()  # 取消注释以测试单独报告

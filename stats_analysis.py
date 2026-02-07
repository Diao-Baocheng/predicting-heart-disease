"""
Statistical Analysis Module
Responsible for statistical analysis and feature correlation analysis of data
"""
import pandas as pd
import numpy as np
from scipy import stats


class StatisticsAnalyzer:
    """统计分析器类"""
    
    def __init__(self, data):
        """
        初始化统计分析器
        
        Args:
            data: DataFrame数据
        """
        self.data = data
        
    def basic_statistics(self):
        """基础统计分析"""
        print("\n" + "="*60)
        print("基础统计信息")
        print("="*60)
        
        stats_info = {
            '数值列数量': len(self.data.select_dtypes(include=[np.number]).columns),
            '分类列数量': len(self.data.select_dtypes(exclude=[np.number]).columns),
            '总行数': len(self.data),
            '总列数': len(self.data.columns),
            '缺失值总数': self.data.isnull().sum().sum()
        }
        
        print("\n数据集基本信息:")
        for key, value in stats_info.items():
            print(f"  {key}: {value}")
        
        return stats_info
    
    def feature_statistics(self):
        """特征统计分析"""
        print("\n" + "="*60)
        print("特征统计分析")
        print("="*60)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        stats_df = pd.DataFrame()
        for col in numeric_cols:
            if col != 'id':
                stats_df[col] = {
                    '均值': self.data[col].mean(),
                    '中位数': self.data[col].median(),
                    '标准差': self.data[col].std(),
                    '最小值': self.data[col].min(),
                    '最大值': self.data[col].max(),
                    '偏度': self.data[col].skew(),
                    '峰度': self.data[col].kurtosis()
                }
        
        print("\n数值特征统计摘要:")
        print(stats_df.T.round(2))
        
        return stats_df.T
    
    def correlation_analysis(self, target_col=None):
        """
        相关性分析
        
        Args:
            target_col: 目标列名
        """
        print("\n" + "="*60)
        print("相关性分析")
        print("="*60)
        
        # 只选择数值列
        numeric_data = self.data.select_dtypes(include=[np.number])
        if 'id' in numeric_data.columns:
            numeric_data = numeric_data.drop('id', axis=1)
        
        # 计算相关性矩阵
        correlation_matrix = numeric_data.corr()
        
        print("\n特征相关性矩阵 (前5x5):")
        print(correlation_matrix.iloc[:5, :5].round(3))
        
        # 如果指定了目标列，显示与目标列的相关性
        if target_col and target_col in correlation_matrix.columns:
            target_corr = correlation_matrix[target_col].sort_values(ascending=False)
            print(f"\n与 {target_col} 的相关性排序:")
            print(target_corr.round(3))
            return correlation_matrix, target_corr
        
        return correlation_matrix
    
    def distribution_analysis(self):
        """分布分析"""
        print("\n" + "="*60)
        print("数据分布分析")
        print("="*60)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        distribution_info = {}
        
        for col in numeric_cols:
            if col != 'id':
                # 正态性检验 (Shapiro-Wilk test)
                # 对大数据集采样进行检验
                sample_size = min(5000, len(self.data))
                sample_data = self.data[col].sample(n=sample_size, random_state=42)
                
                try:
                    statistic, p_value = stats.shapiro(sample_data)
                    is_normal = p_value > 0.05
                    
                    distribution_info[col] = {
                        '正态性检验统计量': statistic,
                        'p值': p_value,
                        '是否正态分布': '是' if is_normal else '否'
                    }
                except:
                    distribution_info[col] = {
                        '正态性检验统计量': None,
                        'p值': None,
                        '是否正态分布': '无法检验'
                    }
        
        dist_df = pd.DataFrame(distribution_info).T
        print("\n特征分布检验结果:")
        print(dist_df)
        
        return dist_df
    
    def outlier_detection(self, method='iqr'):
        """
        异常值检测
        
        Args:
            method: 检测方法 ('iqr' 或 'zscore')
        """
        print("\n" + "="*60)
        print(f"异常值检测 (方法: {method.upper()})")
        print("="*60)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            if col != 'id':
                if method == 'iqr':
                    Q1 = self.data[col].quantile(0.25)
                    Q3 = self.data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = self.data[(self.data[col] < lower_bound) | 
                                        (self.data[col] > upper_bound)]
                    
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(self.data[col]))
                    outliers = self.data[z_scores > 3]
                
                outlier_info[col] = {
                    '异常值数量': len(outliers),
                    '异常值比例': f"{len(outliers) / len(self.data) * 100:.2f}%"
                }
        
        outlier_df = pd.DataFrame(outlier_info).T
        print("\n异常值检测结果:")
        print(outlier_df)
        
        return outlier_df
    
    def class_distribution(self, target_col):
        """
        类别分布分析
        
        Args:
            target_col: 目标列名
        """
        print("\n" + "="*60)
        print("目标变量分布分析")
        print("="*60)
        
        if target_col not in self.data.columns:
            print(f"列 '{target_col}' 不存在")
            return None
        
        # 统计各类别数量
        class_counts = self.data[target_col].value_counts()
        class_percent = self.data[target_col].value_counts(normalize=True) * 100
        
        distribution = pd.DataFrame({
            '数量': class_counts,
            '百分比': class_percent.round(2)
        })
        
        print(f"\n{target_col} 分布:")
        print(distribution)
        
        # 计算类别不平衡比例
        imbalance_ratio = class_counts.max() / class_counts.min()
        print(f"\n类别不平衡比例: {imbalance_ratio:.2f}:1")
        
        return distribution


def analyze_data(data, target_col=None):
    """
    便捷函数：执行完整的统计分析
    
    Args:
        data: DataFrame数据
        target_col: 目标列名
        
    Returns:
        统计分析器对象
    """
    analyzer = StatisticsAnalyzer(data)
    
    analyzer.basic_statistics()
    analyzer.feature_statistics()
    analyzer.correlation_analysis(target_col)
    analyzer.distribution_analysis()
    analyzer.outlier_detection()
    
    if target_col:
        analyzer.class_distribution(target_col)
    
    return analyzer


def compare_predictions(y_true, y_pred, labels=['Absence', 'Presence']):
    """
    比较预测结果的统计分析
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        labels: 类别标签
    """
    from sklearn.metrics import confusion_matrix, classification_report
    
    print("\n" + "="*60)
    print("预测结果统计分析")
    print("="*60)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print("\n混淆矩阵:")
    print(pd.DataFrame(cm, index=labels, columns=labels))
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=labels))

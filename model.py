"""
Model Training Module
Responsible for model training, prediction and evaluation
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class HeartDiseaseModel:
    """心脏病预测模型类"""
    
    def __init__(self):
        """初始化模型"""
        self.rf_model = None
        self.gb_model = None
        self.rf_score = 0
        self.gb_score = 0
        
    def train_random_forest(self, X_train, y_train, n_estimators=50, max_depth=12):
        """
        训练随机森林模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            n_estimators: 树的数量
            max_depth: 树的最大深度
        """
        print("\n训练随机森林模型...")
        
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(X_train, y_train)
        self.rf_score = self.rf_model.score(X_train, y_train)
        
        print(f"随机森林训练完成 - 训练准确率: {self.rf_score:.4f}")
        
        return self.rf_model
    
    def train_gradient_boosting(self, X_train, y_train, n_estimators=100, learning_rate=0.1):
        """
        训练梯度提升模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            n_estimators: 树的数量
            learning_rate: 学习率
        """
        print("\n训练梯度提升模型...")
        
        self.gb_model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        self.gb_model.fit(X_train, y_train)
        self.gb_score = self.gb_model.score(X_train, y_train)
        
        print(f"梯度提升训练完成 - 训练准确率: {self.gb_score:.4f}")
        
        return self.gb_model
    
    def train_all_models(self, X_train, y_train):
        """
        训练所有模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
        """
        print("\n" + "="*60)
        print("模型训练")
        print("="*60)
        
        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        
        print(f"\n所有模型训练完成！")
        print(f"随机森林得分: {self.rf_score:.4f}")
        print(f"梯度提升得分: {self.gb_score:.4f}")
        
        return self.rf_model, self.gb_model
    
    def predict(self, X_test):
        """
        Generate predictions using ensemble method
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted labels and predicted probabilities
        """
        if self.rf_model is None or self.gb_model is None:
            raise ValueError("Please train models first")
        
        print("\n" + "="*60)
        print("Generating Predictions")
        print("="*60)
        
        # Get prediction probabilities from each model
        rf_pred_proba = self.rf_model.predict_proba(X_test)[:, 1]
        gb_pred_proba = self.gb_model.predict_proba(X_test)[:, 1]
        
        # Ensemble prediction using weighted average
        rf_weight = self.rf_score
        gb_weight = self.gb_score
        total_weight = rf_weight + gb_weight
        
        ensemble_pred_proba = (rf_pred_proba * rf_weight + gb_pred_proba * gb_weight) / total_weight
        
        # Convert to binary labels
        ensemble_pred = (ensemble_pred_proba >= 0.5).astype(int)
        
        print(f"Prediction completed - Sample count: {len(ensemble_pred)}")
        print(f"Predicted Presence (1): {ensemble_pred.sum()}")
        print(f"Predicted Absence (0): {(ensemble_pred == 0).sum()}")
        
        return ensemble_pred, ensemble_pred_proba
    
    def predict_proba_on_train(self, X_train):
        """
        Get prediction probabilities on training data (for ROC curve)
        
        Args:
            X_train: Training features
            
        Returns:
            Prediction probabilities
        """
        if self.rf_model is None or self.gb_model is None:
            raise ValueError("Please train models first")
        
        # Get prediction probabilities from each model
        rf_pred_proba = self.rf_model.predict_proba(X_train)[:, 1]
        gb_pred_proba = self.gb_model.predict_proba(X_train)[:, 1]
        
        # Ensemble prediction using weighted average
        rf_weight = self.rf_score
        gb_weight = self.gb_score
        total_weight = rf_weight + gb_weight
        
        ensemble_pred_proba = (rf_pred_proba * rf_weight + gb_pred_proba * gb_weight) / total_weight
        
        return ensemble_pred_proba
    
    def get_feature_importance(self, feature_names):
        """
        获取特征重要性
        
        Args:
            feature_names: 特征名称列表
            
        Returns:
            特征重要性DataFrame
        """
        if self.rf_model is None:
            raise ValueError("请先训练随机森林模型")
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def evaluate(self, X_val, y_val):
        """
        评估模型性能
        
        Args:
            X_val: 验证特征
            y_val: 验证标签
        """
        if self.rf_model is None or self.gb_model is None:
            raise ValueError("请先训练模型")
        
        print("\n" + "="*60)
        print("模型评估")
        print("="*60)
        
        # 随机森林评估
        rf_pred = self.rf_model.predict(X_val)
        rf_acc = accuracy_score(y_val, rf_pred)
        print(f"\n随机森林准确率: {rf_acc:.4f}")
        
        # 梯度提升评估
        gb_pred = self.gb_model.predict(X_val)
        gb_acc = accuracy_score(y_val, gb_pred)
        print(f"梯度提升准确率: {gb_acc:.4f}")
        
        return rf_acc, gb_acc


def save_submission(predictions, test_ids, output_path='submission.csv'):
    """
    保存预测结果为提交文件
    
    Args:
        predictions: 预测标签
        test_ids: 测试集ID
        output_path: 输出文件路径
    """
    submission_df = pd.DataFrame({
        'id': test_ids,
        'Heart Disease': predictions
    })
    
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n提交文件已保存: {output_path}")
    print(f"\n预览前10行:")
    print(submission_df.head(10))
    
    return submission_df

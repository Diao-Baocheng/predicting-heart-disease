"""
Visualization Module
Handles data visualization and result presentation
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")


class DataVisualizer:
    """Data Visualizer Class"""
    
    def __init__(self, figsize=(12, 8)):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        
    def plot_target_distribution(self, data, target_col, save_path=None):
        """
        Plot target variable distribution
        
        Args:
            data: DataFrame
            target_col: Target column name
            save_path: Save path
        """
        plt.figure(figsize=(10, 6))
        
        # Count data
        counts = data[target_col].value_counts()
        
        # Plot bar chart
        ax = counts.plot(kind='bar', color=['#3498db', '#e74c3c'])
        plt.title('Target Variable Distribution', fontsize=16, fontweight='bold')
        plt.xlabel(target_col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=0)
        
        # 添加数值标签
        for i, v in enumerate(counts):
            ax.text(i, v + 1000, str(v), ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved: {save_path}")
        
        plt.show()
        
    def plot_feature_distributions(self, data, features=None, save_path=None):
        """
        Plot feature distributions
        
        Args:
            data: DataFrame
            features: Feature list (None for all numeric features)
            save_path: Save path
        """
        if features is None:
            features = data.select_dtypes(include=[np.number]).columns.tolist()
            if 'id' in features:
                features.remove('id')
        
        # Limit number of features
        features = features[:12]
        
        n_features = len(features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(features):
            axes[idx].hist(data[feature], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            axes[idx].set_title(feature, fontsize=10, fontweight='bold')
            axes[idx].set_xlabel('Value', fontsize=8)
            axes[idx].set_ylabel('Frequency', fontsize=8)
            axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold', y=1.0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved: {save_path}")
        
        plt.show()
        
    def plot_correlation_matrix(self, data, save_path=None):
        """
        Plot correlation matrix heatmap
        
        Args:
            data: DataFrame
            save_path: Save path
        """
        # Select numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        if 'id' in numeric_data.columns:
            numeric_data = numeric_data.drop('id', axis=1)
        
        # Calculate correlation matrix
        correlation = numeric_data.corr()
        
        plt.figure(figsize=(14, 12))
        
        # Plot heatmap
        sns.heatmap(
            correlation, 
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved: {save_path}")
        
        plt.show()
        
    def plot_feature_importance(self, feature_importance_df, top_n=15, save_path=None):
        """
        Plot feature importance
        
        Args:
            feature_importance_df: Feature importance DataFrame
            top_n: Show top N important features
            save_path: Save path
        """
        # Select top N features
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        
        # Plot horizontal bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(top_features['importance']):
            plt.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved: {save_path}")
        
        plt.show()
        
    def plot_prediction_distribution(self, predictions, save_path=None):
        """
        Plot prediction distribution
        
        Args:
            predictions: Predictions
            save_path: Save path
        """
        plt.figure(figsize=(10, 6))
        
        # Count predictions
        pred_counts = pd.Series(predictions).value_counts().sort_index()
        labels = ['Absence (0)', 'Presence (1)']
        
        # Plot pie chart
        colors = ['#3498db', '#e74c3c']
        plt.pie(pred_counts, labels=labels, autopct='%1.1f%%', 
                startangle=90, colors=colors, textprops={'fontsize': 12})
        plt.title('Prediction Distribution', fontsize=16, fontweight='bold')
        plt.axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved: {save_path}")
        
        plt.show()
        
    def plot_box_plots(self, data, features=None, target_col=None, save_path=None):
        """
        Plot box plots
        
        Args:
            data: DataFrame
            features: Feature list
            target_col: Target column name (for grouping)
            save_path: Save path
        """
        if features is None:
            features = data.select_dtypes(include=[np.number]).columns.tolist()
            if 'id' in features:
                features.remove('id')
        
        # Limit number of features
        features = features[:12]
        
        n_features = len(features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(features):
            if target_col and target_col in data.columns:
                data.boxplot(column=feature, by=target_col, ax=axes[idx])
                axes[idx].set_title(feature, fontsize=10, fontweight='bold')
                axes[idx].set_xlabel(target_col, fontsize=8)
            else:
                axes[idx].boxplot(data[feature].dropna())
                axes[idx].set_title(feature, fontsize=10, fontweight='bold')
            
            axes[idx].set_ylabel('Value', fontsize=8)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Feature Box Plots', fontsize=16, fontweight='bold', y=1.0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved: {save_path}")
        
        plt.show()
        
    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Plot ROC Curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Save path
        """
        from sklearn.metrics import roc_curve, auc
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='#e74c3c', lw=3, 
                label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Heart Disease Prediction', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved: {save_path}")
        
        plt.show()
        
        return roc_auc
        
    def create_summary_report(self, data, target_col, feature_importance, predictions, 
                             y_true=None, pred_proba=None, output_dir='./plots'):
        """
        Generate complete visualization report
        
        Args:
            data: DataFrame
            target_col: Target column name
            feature_importance: Feature importance DataFrame
            predictions: Predictions
            y_true: True labels (for ROC curve)
            pred_proba: Predicted probabilities (for ROC curve)
            output_dir: Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("Generating Visualization Report")
        print("="*60)
        
        # 1. Target distribution
        print("\nGenerating target distribution plot...")
        self.plot_target_distribution(data, target_col, 
                                     save_path=f'{output_dir}/target_distribution.png')
        
        # 2. Feature distributions
        print("Generating feature distribution plots...")
        self.plot_feature_distributions(data, 
                                       save_path=f'{output_dir}/feature_distributions.png')
        
        # 3. Correlation matrix
        print("Generating correlation matrix...")
        self.plot_correlation_matrix(data, 
                                    save_path=f'{output_dir}/correlation_matrix.png')
        
        # 4. Feature importance
        print("Generating feature importance plot...")
        self.plot_feature_importance(feature_importance, 
                                    save_path=f'{output_dir}/feature_importance.png')
        
        # 5. Prediction distribution
        print("Generating prediction distribution plot...")
        self.plot_prediction_distribution(predictions, 
                                        save_path=f'{output_dir}/prediction_distribution.png')
        
        # 6. ROC Curve (if true labels and probabilities are provided)
        if y_true is not None and pred_proba is not None:
            print("Generating ROC curve...")
            self.plot_roc_curve(y_true, pred_proba, 
                              save_path=f'{output_dir}/roc_curve.png')
        
        print(f"\nAll charts saved to: {output_dir}/")

"""
Heart Disease Prediction Project - Main Program
Integrates data processing, model training, statistical analysis and visualization modules
"""
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_processing import load_and_process_data
from model import HeartDiseaseModel, save_submission
from stats_analysis import analyze_data
from visualization import DataVisualizer

def main():
    """Main function"""
    print("="*60)
    print("Heart Disease Prediction Project")
    print("="*60)
    
    # ==================== 1. Data Processing ====================
    print("\nStep 1/5: Data Loading and Processing")
    
    train_path = 'data/playground-series-s6e2/train.csv'
    test_path = 'data/playground-series-s6e2/test.csv'
    
    # 加载和处理数据（使用采样以加快训练速度）
    processor, X_train, y_train, X_test, test_ids = load_and_process_data(
        train_path=train_path,
        test_path=test_path,
        use_sample=True,
        sample_size=100000
    )
    
    # ==================== 2. Statistical Analysis ====================
    print("\nStep 2/5: Statistical Analysis")
    
    # Perform statistical analysis on full training data
    analyzer = analyze_data(processor.train_df, target_col='Heart Disease')
    
    # ==================== 3. Model Training ====================
    print("\nStep 3/5: Model Training")
    
    # Create model instance
    model = HeartDiseaseModel()
    
    # Train all models
    model.train_all_models(X_train, y_train)
    
    # Get feature importance
    feature_names = processor.get_feature_names()
    feature_importance = model.get_feature_importance(feature_names)
    
    print("\n" + "="*60)
    print("Feature Importance (Top 10)")
    print("="*60)
    print(feature_importance.head(10).to_string(index=False))
    
    # ==================== 4. Prediction ====================
    print("\nStep 4/5: Generate Predictions")
    
    # Generate prediction results
    predictions, pred_probas = model.predict(X_test)
    
    # Save submission file
    submission = save_submission(predictions, test_ids, output_path='submission.csv')
    
    # ==================== 5. Visualization Report ====================
    print("\nStep 5/5: Generate Visualization Report")
    
    # Create visualizer
    visualizer = DataVisualizer()
    
    # Get prediction probabilities on training data for ROC curve
    train_pred_proba = model.predict_proba_on_train(X_train)
    
    # Generate complete visualization report
    visualizer.create_summary_report(
        data=processor.train_df,
        target_col='Heart Disease',
        feature_importance=feature_importance,
        predictions=predictions,
        y_true=y_train,
        pred_proba=train_pred_proba,
        output_dir='./plots'
    )
    
    # ==================== Completion ====================
    print("\n" + "="*60)
    print("Project Execution Complete!")
    print("="*60)
    print(f"\nOutput Files:")
    print(f"  - Submission File: submission.csv")
    print(f"  - Visualization Charts: plots/ Directory")
    print(f"\nModel Performance:")
    print(f"  - Random Forest Accuracy: {model.rf_score:.4f}")
    print(f"  - Gradient Boosting Accuracy: {model.gb_score:.4f}")
    print(f"\nPrediction Results:")
    print(f"  - Total Samples: {len(predictions)}")
    print(f"  - Predicted Presence: {predictions.sum()}")
    print(f"  - Predicted Absence: {(predictions == 0).sum()}")

if __name__ == '__main__':
    main()


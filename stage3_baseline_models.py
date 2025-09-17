"""
Stage 3: Baseline Model Training (No Differential Privacy)
Save this as: stage3_baseline_models.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_recall_fscore_support)
import warnings
warnings.filterwarnings('ignore')

def load_previous_results():
    try:
        with open('results/stage2_features.pkl', 'rb') as f:
            stage2_results = pickle.load(f)
        
        print("✅ Loaded Stage 2 results successfully")
        return stage2_results
    
    except FileNotFoundError:
        print("❌ Could not find Stage 2 results. Please run stage2_feature_engineering.py first")
        return None

def prepare_data_for_modeling(X, y, test_size=0.2, random_state=42):
    print(f"\n📊 PREPARING DATA FOR MODELING")
    print("-" * 40)
    
    print("Class distribution:")
    class_counts = pd.Series(y).value_counts()
    for class_name, count in class_counts.items():
        percentage = count / len(y) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain/Test Split:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    print(f"  Features: {X_train.shape[1]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

def train_baseline_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    print(f"\n🤖 TRAINING BASELINE MODELS")
    print("-" * 35)
    
    models = {
        'Logistic_Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr'),
            'use_scaled': True,
            'description': 'Linear model with regularization'
        },
        'Random_Forest': {
            'model': RandomForestClassifier(
                n_estimators=100, 
                max_depth=None, 
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            ),
            'use_scaled': False,
            'description': 'Ensemble of decision trees'
        },
        'Gradient_Boosting': {
            'model': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ),
            'use_scaled': False,
            'description': 'Boosted decision trees'
        },
        'SVM': {
            'model': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'use_scaled': True,
            'description': 'Support Vector Machine'
        }
    }
    
    results = {}
    
    for name, model_info in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print(f"Description: {model_info['description']}")
        print(f"{'='*50}")
        
        model = model_info['model']
        
        if model_info['use_scaled']:
            X_train_use = X_train_scaled
            X_test_use = X_test_scaled
        else:
            X_train_use = X_train
            X_test_use = X_test
        
        print("Training model...")
        model.fit(X_train_use, y_train)
        
        y_pred = model.predict(X_test_use)
        
        try:
            y_prob = model.predict_proba(X_test_use)
        except:
            y_prob = None
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        cv_scores = cross_val_score(
            model, X_train_use, y_train, cv=5, scoring='accuracy'
        )
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            print("\nTop 5 Important Features:")
            feature_indices = np.argsort(feature_importance)[::-1][:5]
            for i, idx in enumerate(feature_indices):
                print(f"  {i+1}. Feature {idx}: {feature_importance[idx]:.4f}")
        
        elif hasattr(model, 'coef_'):
            if len(model.coef_.shape) > 1:
                coef_importance = np.mean(np.abs(model.coef_), axis=0)
            else:
                coef_importance = np.abs(model.coef_)
            
            print("\nTop 5 Important Features (by coefficient magnitude):")
            feature_indices = np.argsort(coef_importance)[::-1][:5]
            for i, idx in enumerate(feature_indices):
                print(f"  {i+1}. Feature {idx}: {coef_importance[idx]:.4f}")
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'probabilities': y_prob,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': feature_importance,
            'use_scaled': model_info['use_scaled'],
            'description': model_info['description']
        }
    
    return results

def evaluate_model_performance(results, y_test, feature_names):
    print(f"\n📈 COMPREHENSIVE MODEL EVALUATION")
    print("-" * 45)
    
    print("\nModel Performance Summary:")
    print("=" * 80)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'CV Score':<10}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<20} {result['accuracy']:<10.4f} {result['precision']:<10.4f} "
              f"{result['recall']:<10.4f} {result['f1_score']:<10.4f} {result['cv_mean']:<10.4f}")
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]
    
    print(f"\n🏆 Best Performing Model: {best_model_name}")
    print(f"   Accuracy: {best_model['accuracy']:.4f}")
    print(f"   F1-Score: {best_model['f1_score']:.4f}")
    print(f"   CV Score: {best_model['cv_mean']:.4f} (+/- {best_model['cv_std']*2:.4f})")
    
    print(f"\n📊 Detailed Analysis of {best_model_name}:")
    print("-" * 50)
    
    cm = best_model['confusion_matrix']
    print("Confusion Matrix:")
    print(cm)
    
    classes = sorted(set(y_test))
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_test, best_model['predictions'], labels=classes, zero_division=0
    )
    
    print("\nPer-Class Performance:")
    for i, class_name in enumerate(classes):
        print(f"  {class_name}:")
        print(f"    Precision: {precision_per_class[i]:.4f}")
        print(f"    Recall: {recall_per_class[i]:.4f}")
        print(f"    F1-Score: {f1_per_class[i]:.4f}")
        print(f"    Support: {support[i]}")
    
    return best_model_name, best_model

def create_performance_visualizations(results, y_test, feature_names, save_plots=True):
    print(f"\n📊 CREATING PERFORMANCE VISUALIZATIONS")
    print("-" * 45)
    
    if save_plots:
        os.makedirs('plots', exist_ok=True)
    
    plt.style.use('seaborn-v0_8')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Baseline Model Performance Analysis', fontsize=16, fontweight='bold')
    
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    bars = axes[0, 0].bar(model_names, accuracies, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Model Accuracy Comparison', fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    for bar, acc in zip(bars, accuracies):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom')
    
    f1_scores = [results[name]['f1_score'] for name in model_names]
    
    bars = axes[0, 1].bar(model_names, f1_scores, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('F1-Score Comparison', fontweight='bold')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    for bar, f1 in zip(bars, f1_scores):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{f1:.3f}', ha='center', va='bottom')
    
    cv_means = [results[name]['cv_mean'] for name in model_names]
    cv_stds = [results[name]['cv_std'] for name in model_names]
    
    bars = axes[0, 2].bar(model_names, cv_means, yerr=cv_stds, 
                         color='lightcoral', alpha=0.7, capsize=5)
    axes[0, 2].set_title('Cross-Validation Scores', fontweight='bold')
    axes[0, 2].set_ylabel('CV Accuracy')
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    cm = results[best_model_name]['confusion_matrix']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    
    best_model = results[best_model_name]
    if best_model['feature_importance'] is not None:
        feature_imp = best_model['feature_importance']
        top_indices = np.argsort(feature_imp)[::-1][:10]
        
        top_features = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' 
                       for i in top_indices]
        top_importances = feature_imp[top_indices]
        
        axes[1, 1].barh(range(len(top_features)), top_importances, color='gold', alpha=0.7)
        axes[1, 1].set_yticks(range(len(top_features)))
        axes[1, 1].set_yticklabels(top_features)
        axes[1, 1].set_title(f'Top 10 Features - {best_model_name}', fontweight='bold')
        axes[1, 1].set_xlabel('Importance')
    else:
        axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Feature Importance', fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    best_metrics = [
        best_model['accuracy'],
        best_model['precision'], 
        best_model['recall'],
        best_model['f1_score']
    ]
    
    axes[1, 2].bar(metrics, best_metrics, color='mediumpurple', alpha=0.7)
    axes[1, 2].set_title(f'Performance Metrics - {best_model_name}', fontweight='bold')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('plots/stage3_baseline_performance.png', dpi=300, bbox_inches='tight')
        print("📈 Performance plots saved to plots/stage3_baseline_performance.png")
    
    plt.show()

def save_baseline_results(results, best_model_name, feature_names, X_test, y_test, scaler):
    print(f"\n💾 SAVING BASELINE RESULTS")
    print("-" * 30)
    
    baseline_results = {
        'model_results': results,
        'best_model_name': best_model_name,
        'best_model': results[best_model_name],
        'feature_names': feature_names,
        'test_data': {
            'X_test': X_test,
            'y_test': y_test
        },
        'scaler': scaler,
        'baseline_info': {
            'total_models': len(results),
            'best_accuracy': results[best_model_name]['accuracy'],
            'best_f1': results[best_model_name]['f1_score'],
            'processing_date': datetime.now()
        }
    }
    
    with open('results/stage3_baseline_results.pkl', 'wb') as f:
        pickle.dump(baseline_results, f)
    
    print(f"✅ Baseline results saved to results/stage3_baseline_results.pkl")
    print(f"🏆 Best model: {best_model_name}")
    print(f"📊 Best accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"📈 Best F1-score: {results[best_model_name]['f1_score']:.4f}")

def main():
    print("🏥 DIFFERENTIAL PRIVACY HEALTHCARE BREACH ANALYSIS")
    print("=" * 80)
    print("STAGE 3: BASELINE MODEL TRAINING (NO PRIVACY)")
    print("=" * 80)
    
    stage2_results = load_previous_results()
    if stage2_results is None:
        return
    
    X = stage2_results['X']
    y = stage2_results['y']
    feature_names = stage2_results['features']
    
    print(f"✅ Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"🎯 Target classes: {sorted(y.unique())}")
    
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_data_for_modeling(X, y)
    
    results = train_baseline_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
    
    best_model_name, best_model = evaluate_model_performance(results, y_test, feature_names)
    
    create_performance_visualizations(results, y_test, feature_names, save_plots=True)
    
    save_baseline_results(results, best_model_name, feature_names, X_test, y_test, scaler)
    
    print(f"\n✅ STAGE 3 COMPLETED SUCCESSFULLY!")
    print(f"📁 Results saved in 'results/' directory")
    print(f"📊 Plots saved in 'plots/' directory")
    print(f"➡️  Run stage4_private_models.py next")

if __name__ == "__main__":
    main()
"""
Stage 4: Private Model Training with Differential Privacy
Save this as: stage4_private_models.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class DifferentialPrivacyMechanisms:
    @staticmethod
    def laplace_mechanism(true_value, sensitivity, epsilon):
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        return true_value + noise
    
    @staticmethod
    def gaussian_mechanism(true_value, sensitivity, epsilon, delta=1e-5):
        sigma = np.sqrt(2 * np.log(1.25/delta)) * sensitivity / epsilon
        noise = np.random.normal(0, sigma)
        return true_value + noise
    
    @staticmethod
    def exponential_mechanism(candidates, utility_scores, sensitivity, epsilon):
        scaled_utilities = np.array(utility_scores) * epsilon / (2 * sensitivity)
        max_utility = np.max(scaled_utilities)
        exp_utilities = np.exp(scaled_utilities - max_utility)
        probabilities = exp_utilities / np.sum(exp_utilities)
        
        selected_idx = np.random.choice(len(candidates), p=probabilities)
        return candidates[selected_idx]
    
    @staticmethod
    def add_noise_to_features(X, noise_multiplier, feature_bounds=None):
        X_noisy = X.copy()
        
        for col in range(X.shape[1]):
            if feature_bounds and col < len(feature_bounds):
                sensitivity = feature_bounds[col][1] - feature_bounds[col][0]
            else:
                sensitivity = np.max(X[:, col]) - np.min(X[:, col])
                sensitivity = max(sensitivity, 1e-6)
            
            noise = np.random.laplace(0, noise_multiplier * sensitivity, size=X.shape[0])
            X_noisy[:, col] += noise
            
        return X_noisy
    
    @staticmethod
    def add_gaussian_noise_to_features(X, noise_multiplier, feature_bounds=None):
        X_noisy = X.copy()
        
        for col in range(X.shape[1]):
            if feature_bounds and col < len(feature_bounds):
                sensitivity = feature_bounds[col][1] - feature_bounds[col][0]
            else:
                sensitivity = np.max(X[:, col]) - np.min(X[:, col])
                sensitivity = max(sensitivity, 1e-6)
            
            noise = np.random.normal(0, noise_multiplier * sensitivity, size=X.shape[0])
            X_noisy[:, col] += noise
            
        return X_noisy

class PrivateModelTrainer:
    def __init__(self, budget_tracker):
        self.budget_tracker = budget_tracker
        self.dp_mechanisms = DifferentialPrivacyMechanisms()
        self.trained_models = {}
    
    def input_perturbation_training(self, X_train, y_train, X_test, epsilon, model_type='random_forest'):
        print(f"\n🔒 INPUT PERTURBATION TRAINING (ε = {epsilon})")
        print("-" * 50)
        
        feature_bounds = []
        for i in range(X_train.shape[1]):
            bounds = (np.min(X_train[:, i]), np.max(X_train[:, i]))
            feature_bounds.append(bounds)
        
        noise_multiplier = 2.0 / epsilon
        
        print(f"Noise multiplier: {noise_multiplier:.3f}")
        print("Adding Laplace noise to training features...")
        
        X_train_noisy = self.dp_mechanisms.add_noise_to_features(
            X_train, noise_multiplier, feature_bounds
        )
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
        else:
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            )
        
        print(f"Training {model_type} on noisy data...")
        model.fit(X_train_noisy, y_train)
        
        y_pred = model.predict(X_test)
        
        print("✅ Input perturbation training completed")
        
        return model, y_pred
    
    def output_perturbation_training(self, X_train, y_train, X_test, epsilon, model_type='random_forest'):
        print(f"\n🔒 OUTPUT PERTURBATION TRAINING (ε = {epsilon})")
        print("-" * 52)
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
        else:
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            )
        
        print(f"Training {model_type}...")
        model.fit(X_train, y_train)
        
        y_prob = model.predict_proba(X_test)
        
        print("Applying exponential mechanism to predictions...")
        
        y_pred_private = []
        classes = model.classes_
        
        for i in range(len(X_test)):
            utility_scores = y_prob[i]
            
            private_pred = self.dp_mechanisms.exponential_mechanism(
                classes, utility_scores, sensitivity=1.0, epsilon=epsilon
            )
            y_pred_private.append(private_pred)
        
        y_pred_private = np.array(y_pred_private)
        
        print("✅ Output perturbation training completed")
        
        return model, y_pred_private
    
    def gaussian_mechanism_training(self, X_train, y_train, X_test, epsilon, delta=1e-5, model_type='random_forest'):
        print(f"\n🔒 GAUSSIAN MECHANISM TRAINING (ε = {epsilon}, δ = {delta})")
        print("-" * 60)
        
        feature_bounds = []
        for i in range(X_train.shape[1]):
            bounds = (np.min(X_train[:, i]), np.max(X_train[:, i]))
            feature_bounds.append(bounds)
        
        noise_multiplier = np.sqrt(2 * np.log(1.25/delta)) / epsilon
        
        print(f"Noise multiplier: {noise_multiplier:.3f}")
        print("Adding Gaussian noise to training features...")
        
        X_train_noisy = self.dp_mechanisms.add_gaussian_noise_to_features(
            X_train, noise_multiplier, feature_bounds
        )
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
        else:
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            )
        
        print(f"Training {model_type} on noisy data...")
        model.fit(X_train_noisy, y_train)
        
        y_pred = model.predict(X_test)
        
        print("✅ Gaussian mechanism training completed")
        
        return model, y_pred

def load_previous_results():
    try:
        with open('results/stage3_baseline_results.pkl', 'rb') as f:
            baseline_results = pickle.load(f)
        
        with open('results/stage2_features.pkl', 'rb') as f:
            stage2_results = pickle.load(f)
        
        print("✅ Loaded previous stage results successfully")
        return baseline_results, stage2_results
    
    except FileNotFoundError:
        print("❌ Could not find previous stage results. Please run previous stages first")
        return None, None

def train_private_models_at_multiple_privacy_levels(X_train, X_test, y_train, y_test, 
                                                   budget_tracker, privacy_levels=[0.5, 1.0, 2.0]):
    print(f"\n🔐 TRAINING PRIVATE MODELS AT MULTIPLE PRIVACY LEVELS")
    print("=" * 65)
    
    trainer = PrivateModelTrainer(budget_tracker)
    all_results = {}
    
    for epsilon in privacy_levels:
        print(f"\n{'='*20} PRIVACY LEVEL: ε = {epsilon} {'='*20}")
        
        budget_fraction = 0.25
        eps_training, _ = budget_tracker.allocate_budget(
            f"Private Training (ε={epsilon})", budget_fraction
        )
        
        level_results = {}
        
        # 1. Input Perturbation (Laplace)
        try:
            model_input, pred_input = trainer.input_perturbation_training(
                X_train, y_train, X_test, epsilon, 'random_forest'
            )
            
            accuracy_input = accuracy_score(y_test, pred_input)
            precision_input, recall_input, f1_input, _ = precision_recall_fscore_support(
                y_test, pred_input, average='weighted', zero_division=0
            )
            
            level_results['input_perturbation'] = {
                'model': model_input,
                'predictions': pred_input,
                'accuracy': accuracy_input,
                'precision': precision_input,
                'recall': recall_input,
                'f1_score': f1_input,
                'method': 'Input Perturbation (Laplace)',
                'epsilon_used': epsilon
            }
            
            print(f"Input Perturbation Results:")
            print(f"  Accuracy: {accuracy_input:.4f}")
            print(f"  F1-Score: {f1_input:.4f}")
            
        except Exception as e:
            print(f"❌ Input perturbation failed: {e}")
            level_results['input_perturbation'] = None
        
        # 2. Output Perturbation (Exponential)
        try:
            model_output, pred_output = trainer.output_perturbation_training(
                X_train, y_train, X_test, epsilon, 'random_forest'
            )
            
            accuracy_output = accuracy_score(y_test, pred_output)
            precision_output, recall_output, f1_output, _ = precision_recall_fscore_support(
                y_test, pred_output, average='weighted', zero_division=0
            )
            
            level_results['output_perturbation'] = {
                'model': model_output,
                'predictions': pred_output,
                'accuracy': accuracy_output,
                'precision': precision_output,
                'recall': recall_output,
                'f1_score': f1_output,
                'method': 'Output Perturbation (Exponential)',
                'epsilon_used': epsilon
            }
            
            print(f"Output Perturbation Results:")
            print(f"  Accuracy: {accuracy_output:.4f}")
            print(f"  F1-Score: {f1_output:.4f}")
            
        except Exception as e:
            print(f"❌ Output perturbation failed: {e}")
            level_results['output_perturbation'] = None
        
        # 3. Gaussian Mechanism
        try:
            model_gaussian, pred_gaussian = trainer.gaussian_mechanism_training(
                X_train, y_train, X_test, epsilon, 1e-5, 'random_forest'
            )
            
            accuracy_gaussian = accuracy_score(y_test, pred_gaussian)
            precision_gaussian, recall_gaussian, f1_gaussian, _ = precision_recall_fscore_support(
                y_test, pred_gaussian, average='weighted', zero_division=0
            )
            
            level_results['gaussian_mechanism'] = {
                'model': model_gaussian,
                'predictions': pred_gaussian,
                'accuracy': accuracy_gaussian,
                'precision': precision_gaussian,
                'recall': recall_gaussian,
                'f1_score': f1_gaussian,
                'method': 'Gaussian Mechanism',
                'epsilon_used': epsilon
            }
            
            print(f"Gaussian Mechanism Results:")
            print(f"  Accuracy: {accuracy_gaussian:.4f}")
            print(f"  F1-Score: {f1_gaussian:.4f}")
            
        except Exception as e:
            print(f"❌ Gaussian mechanism failed: {e}")
            level_results['gaussian_mechanism'] = None
        
        all_results[epsilon] = level_results
        
        print(f"\nPrivacy budget used for ε={epsilon}: {eps_training:.4f}")
    
    return all_results

def evaluate_privacy_utility_tradeoff(private_results, baseline_results):
    print(f"\n📊 PRIVACY-UTILITY TRADEOFF EVALUATION")
    print("=" * 50)
    
    baseline_accuracy = baseline_results['best_model']['accuracy']
    baseline_f1 = baseline_results['best_model']['f1_score']
    
    print(f"Baseline Performance (No Privacy):")
    print(f"  Best Model: {baseline_results['best_model_name']}")
    print(f"  Accuracy: {baseline_accuracy:.4f}")
    print(f"  F1-Score: {baseline_f1:.4f}")
    
    print(f"\nPrivate Model Performance:")
    print("=" * 80)
    print(f"{'Privacy Level':<15} {'Method':<25} {'Accuracy':<10} {'F1-Score':<10} {'Privacy Cost':<12}")
    print("-" * 80)
    
    tradeoff_data = []
    
    for epsilon, level_results in private_results.items():
        for method_name, method_results in level_results.items():
            if method_results is not None:
                accuracy = method_results['accuracy']
                f1_score = method_results['f1_score']
                privacy_cost = baseline_accuracy - accuracy
                
                print(f"ε = {epsilon:<10} {method_results['method']:<25} {accuracy:<10.4f} "
                      f"{f1_score:<10.4f} {privacy_cost:<12.4f}")
                
                tradeoff_data.append({
                    'epsilon': epsilon,
                    'method': method_results['method'],
                    'accuracy': accuracy,
                    'f1_score': f1_score,
                    'privacy_cost': privacy_cost,
                    'utility_privacy_ratio': accuracy / epsilon
                })
    
    print(f"\n🏆 BEST CONFIGURATIONS:")
    print("-" * 30)
    
    best_accuracy = max(tradeoff_data, key=lambda x: x['accuracy'])
    print(f"Highest Accuracy:")
    print(f"  Method: {best_accuracy['method']}")
    print(f"  Privacy Level: ε = {best_accuracy['epsilon']}")
    print(f"  Accuracy: {best_accuracy['accuracy']:.4f}")
    print(f"  Privacy Cost: {best_accuracy['privacy_cost']:.4f}")
    
    best_ratio = max(tradeoff_data, key=lambda x: x['utility_privacy_ratio'])
    print(f"\nBest Utility-Privacy Balance:")
    print(f"  Method: {best_ratio['method']}")
    print(f"  Privacy Level: ε = {best_ratio['epsilon']}")
    print(f"  Accuracy: {best_ratio['accuracy']:.4f}")
    print(f"  Utility/Privacy Ratio: {best_ratio['utility_privacy_ratio']:.3f}")
    
    return tradeoff_data

def save_private_model_results(private_results, tradeoff_data, budget_tracker):
    print(f"\n💾 SAVING PRIVATE MODEL RESULTS")
    print("-" * 35)
    
    stage4_results = {
        'private_model_results': private_results,
        'tradeoff_analysis': tradeoff_data,
        'budget_tracker': budget_tracker,
        'stage4_info': {
            'privacy_levels_tested': list(private_results.keys()),
            'dp_techniques': ['Input Perturbation', 'Output Perturbation', 'Gaussian Mechanism'],
            'best_accuracy': max(tradeoff_data, key=lambda x: x['accuracy'])['accuracy'],
            'best_method': max(tradeoff_data, key=lambda x: x['accuracy'])['method'],
            'processing_date': datetime.now()
        }
    }
    
    with open('results/stage4_private_results.pkl', 'wb') as f:
        pickle.dump(stage4_results, f)
    
    budget_tracker.save_budget('results/privacy_budget_stage4.pkl')
    
    print(f"✅ Private model results saved to results/stage4_private_results.pkl")
    
    best_result = max(tradeoff_data, key=lambda x: x['accuracy'])
    print(f"🏆 Best private model:")
    print(f"   Method: {best_result['method']}")
    print(f"   Privacy Level: ε = {best_result['epsilon']}")
    print(f"   Accuracy: {best_result['accuracy']:.4f}")
    print(f"   Privacy Cost: {best_result['privacy_cost']:.4f}")

def main():
    print("🏥 DIFFERENTIAL PRIVACY HEALTHCARE BREACH ANALYSIS")
    print("=" * 80)
    print("STAGE 4: PRIVATE MODEL TRAINING WITH DIFFERENTIAL PRIVACY")
    print("=" * 80)
    
    baseline_results, stage2_results = load_previous_results()
    if baseline_results is None or stage2_results is None:
        return
    
    budget_tracker = stage2_results['budget_tracker']
    
    X = stage2_results['X'].values
    y = stage2_results['y'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Data prepared: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"📊 Features: {X_train.shape[1]}")
    
    privacy_levels = [0.5, 1.0, 2.0]
    private_results = train_private_models_at_multiple_privacy_levels(
        X_train, X_test, y_train, y_test, budget_tracker, privacy_levels
    )
    
    tradeoff_data = evaluate_privacy_utility_tradeoff(private_results, baseline_results)
    
    budget_tracker.print_budget_summary()
    
    save_private_model_results(private_results, tradeoff_data, budget_tracker)
    
    print(f"\n✅ STAGE 4 COMPLETED SUCCESSFULLY!")
    print(f"📁 Results saved in 'results/' directory")
    print(f"➡️  Run stage5_final_report.py next")

if __name__ == "__main__":
    main()
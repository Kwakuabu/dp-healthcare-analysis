"""
Stage 1: Data Loading and Private Exploration
Save this as: stage1_data_exploration.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime

class PrivacyBudgetTracker:
    def __init__(self, total_epsilon, total_delta=1e-5):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.consumed_epsilon = 0
        self.consumed_delta = 0
        self.operations = []
    
    def allocate_budget(self, operation_name, epsilon_fraction, delta_fraction=0):
        epsilon_used = self.total_epsilon * epsilon_fraction
        delta_used = self.total_delta * delta_fraction
        
        if self.consumed_epsilon + epsilon_used > self.total_epsilon:
            raise ValueError(f"Privacy budget exhausted! Requested: {epsilon_used:.4f}, Available: {self.total_epsilon - self.consumed_epsilon:.4f}")
        
        self.consumed_epsilon += epsilon_used
        self.consumed_delta += delta_used
        
        self.operations.append({
            'operation': operation_name,
            'epsilon_used': epsilon_used,
            'delta_used': delta_used,
            'cumulative_epsilon': self.consumed_epsilon,
            'cumulative_delta': self.consumed_delta,
            'timestamp': datetime.now()
        })
        
        return epsilon_used, delta_used
    
    def print_budget_summary(self):
        print("=" * 60)
        print("PRIVACY BUDGET CONSUMPTION SUMMARY")
        print("=" * 60)
        print(f"Total Budget: ε = {self.total_epsilon}, δ = {self.total_delta}")
        print(f"Consumed: ε = {self.consumed_epsilon:.4f} ({self.consumed_epsilon/self.total_epsilon*100:.1f}%)")
        print(f"Remaining: ε = {self.total_epsilon - self.consumed_epsilon:.4f}")
        print("\nOperation-by-Operation Breakdown:")
        print("-" * 60)
        
        for op in self.operations:
            print(f"{op['operation']:<25} | ε: {op['epsilon_used']:.4f} | δ: {op['delta_used']:.6f} | Cumulative ε: {op['cumulative_epsilon']:.4f}")
    
    def save_budget(self, filename='privacy_budget.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Privacy budget saved to {filename}")

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

def load_and_explore_data(filepath, budget_tracker):
    print("STAGE 1: DATA LOADING AND PRIVATE EXPLORATION")
    print("=" * 50)
    
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Successfully loaded dataset: {filepath}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find file {filepath}")
        print("Please ensure breach_report.csv is in the same directory as this script")
        return None, None
    
    eps_explore, delta_explore = budget_tracker.allocate_budget("Data Exploration", 0.15)
    
    print(f"\n📊 DATASET OVERVIEW (with ε = {eps_explore:.4f} privacy protection)")
    print("-" * 50)
    
    true_size = len(df)
    noisy_size = int(DifferentialPrivacyMechanisms.laplace_mechanism(
        true_size, sensitivity=1, epsilon=eps_explore/4
    ))
    print(f"Dataset size (private): ~{noisy_size} records")
    print(f"Actual size: {true_size} records")
    
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\n📈 CATEGORICAL FEATURES ANALYSIS (private)")
    print("-" * 45)
    
    categorical_cols = ['Covered Entity Type', 'Type of Breach', 'Location of Breached Information', 
                       'Business Associate Present', 'State']
    
    categorical_stats = {}
    for col in categorical_cols:
        if col in df.columns:
            value_counts = df[col].value_counts()
            print(f"\n{col}:")
            
            private_counts = {}
            for value, count in value_counts.head(5).items():
                noisy_count = max(0, int(DifferentialPrivacyMechanisms.laplace_mechanism(
                    count, sensitivity=1, epsilon=eps_explore/(len(categorical_cols)*5)
                )))
                private_counts[value] = {'actual': count, 'private': noisy_count}
                percentage = count / len(df) * 100
                print(f"  {value}: ~{noisy_count} ({percentage:.1f}% actual)")
            
            categorical_stats[col] = {
                'unique_count': len(value_counts),
                'private_counts': private_counts
            }
    
    if 'Individuals Affected' in df.columns:
        individuals_col = df['Individuals Affected'].dropna()
        if len(individuals_col) > 0:
            true_mean = individuals_col.mean()
            true_median = individuals_col.median()
            true_max = individuals_col.max()
            true_min = individuals_col.min()
            
            sensitivity = true_max - true_min
            private_mean = DifferentialPrivacyMechanisms.laplace_mechanism(
                true_mean, sensitivity/len(individuals_col), eps_explore/8
            )
            private_median = DifferentialPrivacyMechanisms.laplace_mechanism(
                true_median, sensitivity/len(individuals_col), eps_explore/8
            )
            
            print(f"\nIndividuals Affected:")
            print(f"  Mean (private): ~{int(private_mean):,} (actual: {int(true_mean):,})")
            print(f"  Median (private): ~{int(private_median):,} (actual: {int(true_median):,})")
            print(f"  Range: {int(true_min):,} - {int(true_max):,}")
    
    exploration_results = {
        'dataset_info': {
            'filepath': filepath,
            'true_size': true_size,
            'private_size': noisy_size,
            'columns': list(df.columns),
            'exploration_epsilon': eps_explore
        },
        'categorical_stats': categorical_stats,
        'budget_tracker': budget_tracker
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/stage1_exploration.pkl', 'wb') as f:
        pickle.dump(exploration_results, f)
    
    print(f"\n💾 Exploration results saved to results/stage1_exploration.pkl")
    
    budget_tracker.print_budget_summary()
    budget_tracker.save_budget('results/privacy_budget_stage1.pkl')
    
    return df, exploration_results

def create_visualizations(df, save_plots=True):
    print(f"\n📊 CREATING PRIVATE VISUALIZATIONS")
    print("-" * 40)
    
    if save_plots:
        os.makedirs('plots', exist_ok=True)
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Private Healthcare Breach Data Exploration', fontsize=16, fontweight='bold')
    
    if 'Covered Entity Type' in df.columns:
        entity_counts = df['Covered Entity Type'].value_counts()
        noisy_counts = []
        labels = []
        for entity, count in entity_counts.items():
            noisy_count = max(0, int(DifferentialPrivacyMechanisms.laplace_mechanism(count, 1, 0.1)))
            noisy_counts.append(noisy_count)
            labels.append(entity)
        
        axes[0, 0].bar(range(len(labels)), noisy_counts, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Entity Types Distribution (Private)', fontweight='bold')
        axes[0, 0].set_xticks(range(len(labels)))
        axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Count (with DP noise)')
    
    if 'Type of Breach' in df.columns:
        breach_counts = df['Type of Breach'].value_counts().head(8)
        noisy_breach_counts = []
        breach_labels = []
        for breach, count in breach_counts.items():
            noisy_count = max(0, int(DifferentialPrivacyMechanisms.laplace_mechanism(count, 1, 0.1)))
            noisy_breach_counts.append(noisy_count)
            breach_labels.append(breach[:20] + '...' if len(breach) > 20 else breach)
        
        axes[0, 1].bar(range(len(breach_labels)), noisy_breach_counts, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Top Breach Types (Private)', fontweight='bold')
        axes[0, 1].set_xticks(range(len(breach_labels)))
        axes[0, 1].set_xticklabels(breach_labels, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Count (with DP noise)')
    
    if 'Individuals Affected' in df.columns:
        individuals = df['Individuals Affected'].dropna()
        def categorize_impact(value):
            if value < 500: return 'Small'
            elif value < 5000: return 'Medium'
            elif value < 50000: return 'Large'
            else: return 'Massive'
        
        categories = individuals.apply(categorize_impact).value_counts()
        noisy_categories = []
        cat_labels = []
        for cat, count in categories.items():
            noisy_count = max(0, int(DifferentialPrivacyMechanisms.laplace_mechanism(count, 1, 0.1)))
            noisy_categories.append(noisy_count)
            cat_labels.append(cat)
        
        colors = ['lightgreen', 'yellow', 'orange', 'red']
        axes[1, 0].pie(noisy_categories, labels=cat_labels, autopct='%1.1f%%', 
                      colors=colors[:len(cat_labels)], startangle=90)
        axes[1, 0].set_title('Breach Impact Categories (Private)', fontweight='bold')
    
    if 'State' in df.columns:
        state_counts = df['State'].value_counts().head(10)
        noisy_state_counts = []
        state_labels = []
        for state, count in state_counts.items():
            noisy_count = max(0, int(DifferentialPrivacyMechanisms.laplace_mechanism(count, 1, 0.1)))
            noisy_state_counts.append(noisy_count)
            state_labels.append(state)
        
        axes[1, 1].barh(range(len(state_labels)), noisy_state_counts, color='lightseagreen', alpha=0.7)
        axes[1, 1].set_title('Top 10 States by Breach Count (Private)', fontweight='bold')
        axes[1, 1].set_yticks(range(len(state_labels)))
        axes[1, 1].set_yticklabels(state_labels)
        axes[1, 1].set_xlabel('Count (with DP noise)')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('plots/stage1_private_exploration.png', dpi=300, bbox_inches='tight')
        print(f"📈 Plots saved to plots/stage1_private_exploration.png")
    
    plt.show()

def main():
    print("🏥 DIFFERENTIAL PRIVACY HEALTHCARE BREACH ANALYSIS")
    print("=" * 80)
    print("STAGE 1: DATA LOADING AND PRIVATE EXPLORATION")
    print("=" * 80)
    
    budget_tracker = PrivacyBudgetTracker(total_epsilon=3.0, total_delta=1e-5)
    
    filepath = 'breach_report.csv'
    df, exploration_results = load_and_explore_data(filepath, budget_tracker)
    
    if df is not None:
        create_visualizations(df, save_plots=True)
        
        print(f"\n✅ STAGE 1 COMPLETED SUCCESSFULLY!")
        print(f"📁 Results saved in 'results/' directory")
        print(f"📊 Plots saved in 'plots/' directory")
        print(f"➡️  Run stage2_feature_engineering.py next")
    else:
        print(f"\n❌ STAGE 1 FAILED - Could not load data")
        print(f"Please check that breach_report.csv exists in the current directory")

if __name__ == "__main__":
    main()
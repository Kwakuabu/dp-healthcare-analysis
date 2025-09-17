"""
Stage 2: Feature Engineering with Privacy Considerations
Save this as: stage2_feature_engineering.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from stage1_data_exploration import DifferentialPrivacyMechanisms, PrivacyBudgetTracker

def load_previous_results():
    try:
        with open('results/stage1_exploration.pkl', 'rb') as f:
            exploration_results = pickle.load(f)
        
        with open('results/privacy_budget_stage1.pkl', 'rb') as f:
            budget_tracker = pickle.load(f)
        
        print("✅ Loaded Stage 1 results successfully")
        return exploration_results, budget_tracker
    
    except FileNotFoundError:
        print("❌ Could not find Stage 1 results. Please run stage1_data_exploration.py first")
        return None, None

def create_target_variable(df, budget_tracker):
    print("\n🎯 CREATING TARGET VARIABLE")
    print("-" * 35)
    
    eps_target, _ = budget_tracker.allocate_budget("Target Creation", 0.05)
    
    def categorize_breach_size(individuals_affected):
        if pd.isna(individuals_affected) or individuals_affected <= 0:
            return 'Unknown'
        elif individuals_affected < 500:
            return 'Small'
        elif individuals_affected < 5000:
            return 'Medium'
        elif individuals_affected < 50000:
            return 'Large'
        else:
            return 'Massive'
    
    df['Breach_Impact_Category'] = df['Individuals Affected'].apply(categorize_breach_size)
    
    target_dist = df['Breach_Impact_Category'].value_counts()
    print("Target Distribution (with privacy noise):")
    
    from stage1_data_exploration import DifferentialPrivacyMechanisms, PrivacyBudgetTracker
    
    private_target_stats = {}
    for category, count in target_dist.items():
        noisy_count = max(0, int(DifferentialPrivacyMechanisms.laplace_mechanism(
            count, sensitivity=1, epsilon=eps_target/len(target_dist)
        )))
        private_target_stats[category] = {'actual': count, 'private': noisy_count}
        percentage = count / len(df) * 100
        print(f"  {category}: ~{noisy_count} ({percentage:.1f}% actual)")
    
    return df, private_target_stats

def engineer_temporal_features(df):
    print("\n📅 ENGINEERING TEMPORAL FEATURES")
    print("-" * 40)
    
    def extract_year(date_str):
        try:
            if pd.isna(date_str):
                return 2015
            
            parts = str(date_str).split('/')
            if len(parts) >= 3:
                year = int(parts[2])
                if year < 30:
                    year += 2000
                elif year < 100:
                    year += 1900
                return year
            return 2015
        except:
            return 2015
    
    df['Submission_Year'] = df['Breach Submission Date'].apply(extract_year)
    
    df['Year_Category'] = pd.cut(df['Submission_Year'], 
                                bins=[2008, 2011, 2014, 2017, 2025], 
                                labels=['Early', 'Growth', 'Peak', 'Recent'])
    
    year_min, year_max = df['Submission_Year'].min(), df['Submission_Year'].max()
    df['Submission_Year_Normalized'] = (df['Submission_Year'] - year_min) / (year_max - year_min)
    
    print(f"✓ Extracted submission year (range: {year_min}-{year_max})")
    print(f"✓ Created year categories and normalized features")
    
    return df

def engineer_categorical_features(df):
    print("\n🏷️ ENGINEERING CATEGORICAL FEATURES")
    print("-" * 45)
    
    def simplify_breach_type(breach_type):
        if pd.isna(breach_type):
            return 'Other'
        
        breach_type = str(breach_type).lower()
        if 'theft' in breach_type:
            return 'Theft'
        elif 'hack' in breach_type or 'it incident' in breach_type:
            return 'Hacking'
        elif 'unauthorized' in breach_type:
            return 'Unauthorized_Access'
        elif 'loss' in breach_type:
            return 'Loss'
        elif 'disposal' in breach_type:
            return 'Improper_Disposal'
        else:
            return 'Other'
    
    df['Breach_Type_Simple'] = df['Type of Breach'].apply(simplify_breach_type)
    print("✓ Simplified breach types into 6 categories")
    
    def simplify_location(location):
        if pd.isna(location):
            return 'Unknown'
        
        location = str(location).lower()
        if 'network' in location or 'server' in location:
            return 'Network'
        elif 'laptop' in location:
            return 'Laptop'
        elif 'paper' in location or 'film' in location:
            return 'Paper'
        elif 'email' in location:
            return 'Email'
        elif 'desktop' in location:
            return 'Desktop'
        elif 'portable' in location or 'device' in location:
            return 'Portable_Device'
        else:
            return 'Other_Electronic'
    
    df['Location_Simple'] = df['Location of Breached Information'].apply(simplify_location)
    print("✓ Simplified location types into 7 categories")
    
    df['Has_Business_Associate'] = (df['Business Associate Present'] == 'Yes').astype(int)
    print("✓ Created business associate binary flag")
    
    entity_mapping = {
        'Healthcare Provider': 0,
        'Health Plan': 1,
        'Business Associate': 2,
        'Healthcare Clearing House': 3
    }
    df['Entity_Type_Encoded'] = df['Covered Entity Type'].map(entity_mapping).fillna(0)
    print("✓ Encoded entity types as ordinal features")
    
    return df

def engineer_geographic_features(df):
    print("\n🗺️ ENGINEERING GEOGRAPHIC FEATURES")
    print("-" * 45)
    
    state_counts = df['State'].value_counts()
    high_risk_threshold = state_counts.median()
    high_risk_states = state_counts[state_counts >= high_risk_threshold].index.tolist()
    
    df['High_Risk_State'] = df['State'].isin(high_risk_states).astype(int)
    print(f"✓ Identified {len(high_risk_states)} high-risk states (threshold: {high_risk_threshold:.0f} breaches)")
    
    def get_region(state):
        northeast = ['CT', 'ME', 'MA', 'NH', 'NJ', 'NY', 'PA', 'RI', 'VT']
        midwest = ['IL', 'IN', 'IA', 'KS', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI']
        south = ['AL', 'AR', 'DE', 'FL', 'GA', 'KY', 'LA', 'MD', 'MS', 'NC', 'OK', 'SC', 'TN', 'TX', 'VA', 'WV']
        west = ['AK', 'AZ', 'CA', 'CO', 'HI', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY']
        
        if state in northeast:
            return 'Northeast'
        elif state in midwest:
            return 'Midwest'
        elif state in south:
            return 'South'
        elif state in west:
            return 'West'
        else:
            return 'Other'
    
    df['Region'] = df['State'].apply(get_region)
    print("✓ Mapped states to US regions")
    
    return df

def engineer_derived_features(df):
    print("\n⚙️ ENGINEERING DERIVED FEATURES")
    print("-" * 40)
    
    def calculate_severity_score(row):
        score = 0
        
        if row['Entity_Type_Encoded'] == 0:  # Healthcare Provider
            score += 2
        elif row['Entity_Type_Encoded'] == 1:  # Health Plan
            score += 3
        else:
            score += 1
        
        breach_type = row['Breach_Type_Simple']
        if breach_type == 'Hacking':
            score += 3
        elif breach_type == 'Theft':
            score += 2
        elif breach_type == 'Unauthorized_Access':
            score += 2
        else:
            score += 1
        
        location = row['Location_Simple']
        if location == 'Network':
            score += 3
        elif location in ['Laptop', 'Portable_Device']:
            score += 2
        else:
            score += 1
        
        return score
    
    df['Severity_Score'] = df.apply(calculate_severity_score, axis=1)
    
    severity_min, severity_max = df['Severity_Score'].min(), df['Severity_Score'].max()
    df['Severity_Score_Normalized'] = (df['Severity_Score'] - severity_min) / (severity_max - severity_min)
    
    print(f"✓ Created severity score (range: {severity_min}-{severity_max})")
    
    tech_locations = ['Network', 'Laptop', 'Desktop', 'Email', 'Portable_Device']
    df['Technology_Involved'] = df['Location_Simple'].isin(tech_locations).astype(int)
    print("✓ Created technology involvement flag")
    
    df['Multi_Factor_Breach'] = df['Type of Breach'].str.contains(',', na=False).astype(int)
    print("✓ Created multi-factor breach flag")
    
    return df

def encode_categorical_features(df):
    print("\n🔢 ENCODING CATEGORICAL FEATURES")
    print("-" * 40)
    
    encoders = {}
    
    categorical_features = [
        'Breach_Type_Simple',
        'Location_Simple', 
        'Region',
        'Year_Category'
    ]
    
    for feature in categorical_features:
        if feature in df.columns:
            le = LabelEncoder()
            feature_values = df[feature].astype(str).fillna('Unknown')
            df[f'{feature}_encoded'] = le.fit_transform(feature_values)
            encoders[feature] = le
            print(f"✓ Encoded {feature} ({len(le.classes_)} categories)")
    
    return df, encoders

def select_final_features(df):
    print("\n✨ SELECTING FINAL FEATURES")
    print("-" * 35)
    
    final_features = [
        'Submission_Year_Normalized',
        'Year_Category_encoded',
        'Entity_Type_Encoded',
        'Breach_Type_Simple_encoded',
        'Location_Simple_encoded',
        'Region_encoded',
        'Has_Business_Associate',
        'High_Risk_State',
        'Technology_Involved',
        'Multi_Factor_Breach',
        'Severity_Score_Normalized'
    ]
    
    available_features = [f for f in final_features if f in df.columns]
    missing_features = [f for f in final_features if f not in df.columns]
    
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
    
    print(f"✓ Selected {len(available_features)} features for modeling:")
    for i, feature in enumerate(available_features, 1):
        print(f"  {i:2d}. {feature}")
    
    return available_features

def save_results(df, features, encoders, private_target_stats, budget_tracker):
    print("\n💾 SAVING STAGE 2 RESULTS")
    print("-" * 30)
    
    df_modeling = df[df['Breach_Impact_Category'] != 'Unknown'].copy()
    
    X = df_modeling[features].copy()
    y = df_modeling['Breach_Impact_Category'].copy()
    
    stage2_results = {
        'df_full': df,
        'df_modeling': df_modeling,
        'X': X,
        'y': y,
        'features': features,
        'encoders': encoders,
        'private_target_stats': private_target_stats,
        'budget_tracker': budget_tracker,
        'stage2_info': {
            'total_records': len(df),
            'modeling_records': len(df_modeling),
            'feature_count': len(features),
            'target_classes': list(y.unique()),
            'processing_date': datetime.now()
        }
    }
    
    with open('results/stage2_features.pkl', 'wb') as f:
        pickle.dump(stage2_results, f)
    
    budget_tracker.save_budget('results/privacy_budget_stage2.pkl')
    
    print(f"✅ Stage 2 results saved to results/stage2_features.pkl")
    print(f"📊 Modeling dataset: {len(df_modeling)} records, {len(features)} features")
    print(f"🎯 Target classes: {list(y.unique())}")

def main():
    print("🏥 DIFFERENTIAL PRIVACY HEALTHCARE BREACH ANALYSIS")
    print("=" * 80)
    print("STAGE 2: FEATURE ENGINEERING WITH PRIVACY")
    print("=" * 80)
    
    exploration_results, budget_tracker = load_previous_results()
    if exploration_results is None:
        return
    
    try:
        df = pd.read_csv('breach_report.csv')
        print(f"✅ Loaded dataset: {len(df)} records")
    except FileNotFoundError:
        print("❌ Could not find breach_report.csv")
        return
    
    df, private_target_stats = create_target_variable(df, budget_tracker)
    df = engineer_temporal_features(df)
    df = engineer_categorical_features(df)
    df = engineer_geographic_features(df)
    df = engineer_derived_features(df)
    df, encoders = encode_categorical_features(df)
    features = select_final_features(df)
    
    save_results(df, features, encoders, private_target_stats, budget_tracker)
    
    budget_tracker.print_budget_summary()
    
    print(f"\n✅ STAGE 2 COMPLETED SUCCESSFULLY!")
    print(f"📁 Results saved in 'results/' directory")
    print(f"➡️  Run stage3_baseline_models.py next")

if __name__ == "__main__":
    main()
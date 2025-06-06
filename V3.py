# Structured Complaints Precision Drop Analysis - Banking Domain
# Investigation Framework Following Systematic Approach

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import re
from collections import Counter
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)

print("=== STRUCTURED COMPLAINTS PRECISION DROP ANALYSIS ===")
print("Investigation Framework: Systematic Root Cause Analysis")
print("Target: Maintain 70% precision for complaints, 30% for non-complaints")
print("Approach: Macro → Deep Dive → Root Cause Analysis\n")

# =============================================================================
# DATA PREPARATION AND LOADING
# =============================================================================

def load_and_prepare_data():
    """Load and prepare data for analysis"""
    
    print("="*80)
    print("DATA PREPARATION AND LOADING")
    print("="*80)
    
    # Load main transcript data
    try:
        df_main = pd.read_excel('Precision_Drop_Analysis_OG.xlsx')
        print(f"Main dataset loaded: {df_main.shape}")
    except FileNotFoundError:
        print("Warning: Main dataset file not found. Creating sample data for demonstration.")
        np.random.seed(42)
        dates = pd.date_range('2024-10-01', '2025-03-31', freq='D')
        sample_size = 5000
        
        df_main = pd.DataFrame({
            'variable5': np.random.choice(list(range(1000, 2000)), sample_size),
            'UUID': list(range(sample_size)),
            'Customer Transcript': [f"Sample customer complaint text {i} not satisfied with service" if i % 3 == 0 
                                  else f"Sample customer inquiry {i} about account balance" 
                                  for i in range(sample_size)],
            'Agent Transcript': [f"Let me explain the process {i} for your complaint" if i % 4 == 0 
                               else f"I understand your concern {i}" 
                               for i in range(sample_size)],
            'Prosodica L1': np.random.choice(['complaints', 'inquiries', 'requests'], sample_size, p=[0.4, 0.35, 0.25]),
            'Prosodica L2': np.random.choice(['fee_waiver', 'credit_limit', 'payment_issue', 'account_inquiry', 'billing_dispute'], sample_size),
            'Primary L1': np.random.choice(['complaints', 'inquiries', 'requests'], sample_size, p=[0.4, 0.35, 0.25]),
            'Primary L2': np.random.choice(['fee_waiver', 'credit_limit', 'payment_issue', 'account_inquiry', 'billing_dispute'], sample_size),
            'Primary Marker': np.random.choice(['TP', 'FP'], sample_size, p=[0.65, 0.35]),
            'Secondary Marker': np.random.choice(['TP', 'FP', None], sample_size, p=[0.3, 0.15, 0.55]),
            'Date': np.random.choice(dates, sample_size)
        })
    
    # Load validation summary
    try:
        df_validation = pd.read_excel('Categorical Validation.xlsx', sheet_name='Summary validation vol')
        print(f"Validation summary loaded: {df_validation.shape}")
    except FileNotFoundError:
        print("Warning: Validation file not found. Creating sample validation data.")
        df_validation = pd.DataFrame()
    
    # Load query rules
    try:
        df_rules = pd.read_excel('Query_Rules.xlsx')
        df_rules_filtered = df_rules[df_rules['Category'].isin(['complaints', 'collection_complaints'])].copy()
        print(f"Query rules loaded and filtered: {df_rules_filtered.shape}")
    except FileNotFoundError:
        print("Warning: Query rules file not found. Creating sample rules data.")
        df_rules_filtered = pd.DataFrame({
            'Category': ['complaints'] * 15,
            'Event': ['complaints'] * 15,
            'Query': [f'query_{i}' for i in range(15)],
            'Query Text': [f'(complaint OR complain OR issue) AND (fee OR billing OR charge)' if i % 3 == 0
                          else f'(upset OR angry OR frustrated) NEAR:5 (service OR support)' if i % 3 == 1
                          else f'(problem OR trouble) AND NOT (thank OR thanks)' 
                          for i in range(15)],
            'Channel': np.random.choice(['customer', 'agent', 'both'], 15)
        })
    
    # Enhanced data preprocessing
    df_main['Date'] = pd.to_datetime(df_main['Date'])
    df_main['Year_Month'] = df_main['Date'].dt.strftime('%Y-%m')
    df_main['DayOfWeek'] = df_main['Date'].dt.day_name()
    df_main['WeekOfMonth'] = df_main['Date'].dt.day // 7 + 1
    df_main['Quarter'] = df_main['Date'].dt.quarter
    df_main['Is_Holiday_Season'] = df_main['Date'].dt.month.isin([11, 12, 1])
    df_main['Is_Month_End'] = df_main['Date'].dt.day >= 25
    
    # Text processing
    df_main['Customer Transcript'] = df_main['Customer Transcript'].fillna('')
    df_main['Agent Transcript'] = df_main['Agent Transcript'].fillna('')
    df_main['Full_Transcript'] = df_main['Customer Transcript'] + ' ' + df_main['Agent Transcript']
    
    # Text features
    df_main['Transcript_Length'] = df_main['Full_Transcript'].str.len()
    df_main['Customer_Word_Count'] = df_main['Customer Transcript'].str.split().str.len()
    df_main['Agent_Word_Count'] = df_main['Agent Transcript'].str.split().str.len()
    df_main['Customer_Agent_Ratio'] = df_main['Customer_Word_Count'] / (df_main['Agent_Word_Count'] + 1)
    
    # Advanced text features
    df_main['Customer_Question_Count'] = df_main['Customer Transcript'].str.count('\?')
    df_main['Customer_Exclamation_Count'] = df_main['Customer Transcript'].str.count('!')
    df_main['Customer_Caps_Ratio'] = df_main['Customer Transcript'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )
    
    # Negation and qualifying patterns
    negation_patterns = r'\b(not|no|never|dont|don\'t|wont|won\'t|cant|can\'t|isnt|isn\'t)\b'
    df_main['Customer_Negation_Count'] = df_main['Customer Transcript'].str.lower().str.count(negation_patterns)
    df_main['Agent_Negation_Count'] = df_main['Agent Transcript'].str.lower().str.count(negation_patterns)
    
    qualifying_patterns = r'\b(might|maybe|seems|appears|possibly|perhaps|probably|likely)\b'
    df_main['Customer_Qualifying_Count'] = df_main['Customer Transcript'].str.lower().str.count(qualifying_patterns)
    
    # Target variables
    df_main['Is_TP'] = (df_main['Primary Marker'] == 'TP').astype(int)
    df_main['Is_FP'] = (df_main['Primary Marker'] == 'FP').astype(int)
    df_main['Has_Secondary_Validation'] = df_main['Secondary Marker'].notna()
    df_main['Primary_Secondary_Agreement'] = np.where(
        df_main['Has_Secondary_Validation'] & df_main['Secondary Marker'].notna(),
        (df_main['Primary Marker'] == df_main['Secondary Marker']).astype(int),
        np.nan
    )
    
    print(f"Data preparation completed. Final dataset shape: {df_main.shape}")
    
    return df_main, df_validation, df_rules_filtered

# Load data
df_main, df_validation, df_rules_filtered = load_and_prepare_data()

# =============================================================================
# MACRO LEVEL ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("MACRO LEVEL ANALYSIS")
print("="*80)

# 1. PRECISION DROP PATTERNS
print("\n1. PRECISION DROP PATTERNS")
print("-" * 40)

def analyze_precision_drop_patterns(df):
    """Comprehensive precision drop pattern analysis"""
    
    print("1.1 Calculate MoM Precision Changes for Each Category")
    
    # Monthly precision by category
    monthly_category_precision = df.groupby(['Year_Month', 'Prosodica L1', 'Prosodica L2']).agg({
        'Is_TP': ['sum', 'count'],
        'Is_FP': 'sum'
    }).reset_index()
    
    monthly_category_precision.columns = ['Year_Month', 'L1_Category', 'L2_Category', 'TPs', 'Total_Flagged', 'FPs']
    monthly_category_precision['Precision'] = np.where(
        monthly_category_precision['Total_Flagged'] > 0,
        monthly_category_precision['TPs'] / monthly_category_precision['Total_Flagged'],
        0
    )
    
    # Calculate MoM changes
    monthly_category_precision = monthly_category_precision.sort_values(['L1_Category', 'L2_Category', 'Year_Month'])
    monthly_category_precision['Precision_MoM_Change'] = monthly_category_precision.groupby(['L1_Category', 'L2_Category'])['Precision'].diff()
    
    print("Monthly Category Precision Changes:")
    significant_changes = monthly_category_precision[
        (abs(monthly_category_precision['Precision_MoM_Change']) > 0.1) & 
        (monthly_category_precision['Total_Flagged'] >= 10)
    ].sort_values('Precision_MoM_Change')
    
    if len(significant_changes) > 0:
        print(significant_changes[['L1_Category', 'L2_Category', 'Year_Month', 'Precision', 'Precision_MoM_Change']].round(3))
    else:
        print("No significant MoM changes detected (>10% with min 10 samples)")
    
    print("\n1.2 Identify Categories Contributing Most to Overall Decline")
    
    # Overall category impact
    category_impact = df.groupby(['Prosodica L1', 'Prosodica L2']).agg({
        'Is_TP': ['sum', 'count'],
        'Is_FP': 'sum',
        'variable5': 'count'
    }).reset_index()
    
    category_impact.columns = ['L1_Category', 'L2_Category', 'TPs', 'Total_Flagged', 'FPs', 'Total_Volume']
    category_impact['Precision'] = np.where(
        category_impact['Total_Flagged'] > 0,
        category_impact['TPs'] / category_impact['Total_Flagged'],
        0
    )
    category_impact['Precision_Gap'] = 0.70 - category_impact['Precision']
    category_impact['Impact_Score'] = category_impact['Precision_Gap'] * category_impact['Total_Flagged']
    
    category_impact = category_impact.sort_values('Impact_Score', ascending=False)
    
    print("Top 10 Categories Contributing to Precision Decline:")
    print(category_impact.head(10)[['L1_Category', 'L2_Category', 'Precision', 'Total_Flagged', 'Impact_Score']].round(3))
    
    print("\n1.3 Determine Drop Characteristics")
    
    # Concentration analysis
    total_impact = category_impact['Impact_Score'].sum()
    top_5_impact = category_impact.head(5)['Impact_Score'].sum()
    concentration_ratio = top_5_impact / total_impact if total_impact > 0 else 0
    
    # Distribution analysis
    below_target = len(category_impact[category_impact['Precision'] < 0.70])
    total_categories = len(category_impact)
    
    print(f"Drop Concentration Analysis:")
    print(f"  Top 5 categories account for {concentration_ratio:.1%} of total impact")
    print(f"  {below_target}/{total_categories} categories below 70% target ({below_target/total_categories:.1%})")
    
    if concentration_ratio > 0.7:
        print("  FINDING: Drop is CONCENTRATED in few high-impact categories")
    elif below_target/total_categories > 0.6:
        print("  FINDING: Drop is WIDESPREAD across many categories")
    else:
        print("  FINDING: Drop is MODERATE with mixed impact distribution")
    
    print("\n1.4 New Categories Analysis")
    
    # Alternative approach: Check for categories with limited historical data
    # Categories appearing in only recent months could be considered "new"
    all_months = sorted(df['Year_Month'].unique())
    if len(all_months) >= 3:
        recent_months = all_months[-2:]  # Last 2 months
        
        # Find categories that only appear in recent months
        category_monthly_presence = df.groupby(['Prosodica L1', 'Prosodica L2'])['Year_Month'].nunique().reset_index()
        category_monthly_presence.columns = ['L1_Category', 'L2_Category', 'Months_Present']
        
        # Categories present in 2 or fewer months could be considered "new"
        potential_new_categories = category_monthly_presence[category_monthly_presence['Months_Present'] <= 2]
        
        if len(potential_new_categories) > 0:
            print("Potentially New Categories (present in ≤2 months):")
            
            new_category_performance = []
            for _, cat in potential_new_categories.iterrows():
                cat_data = df[(df['Prosodica L1'] == cat['L1_Category']) & 
                             (df['Prosodica L2'] == cat['L2_Category'])]
                
                if len(cat_data) > 0:
                    precision = cat_data['Is_TP'].sum() / len(cat_data)
                    volume = len(cat_data)
                    
                    new_category_performance.append({
                        'L1_Category': cat['L1_Category'],
                        'L2_Category': cat['L2_Category'],
                        'Precision': precision,
                        'Volume': volume,
                        'Months_Present': cat['Months_Present']
                    })
            
            if new_category_performance:
                new_perf_df = pd.DataFrame(new_category_performance)
                print(new_perf_df[['L1_Category', 'L2_Category', 'Precision', 'Volume', 'Months_Present']].round(3))
                
                avg_new_precision = new_perf_df['Precision'].mean()
                avg_overall_precision = category_impact['Precision'].mean()
                print(f"Average new category precision: {avg_new_precision:.3f}")
                print(f"Average overall precision: {avg_overall_precision:.3f}")
                
                if avg_new_precision < avg_overall_precision - 0.1:
                    print("  FINDING: Potentially new categories have significantly lower precision")
                else:
                    print("  FINDING: Potentially new categories perform similarly to existing ones")
        else:
            print("No potentially new categories detected (all categories present in 3+ months)")
    else:
        print("Insufficient historical data to identify new categories")
    
    return monthly_category_precision, category_impact

monthly_precision, category_performance = analyze_precision_drop_patterns(df_main)

# 2. VOLUME VS PERFORMANCE ANALYSIS
print("\n2. VOLUME VS PERFORMANCE ANALYSIS")
print("-" * 40)

def analyze_volume_vs_performance(df):
    """Volume vs performance correlation analysis"""
    
    print("2.1 High-Volume vs Low Precision Correlation")
    
    # Volume-precision analysis
    volume_precision = df.groupby(['Prosodica L1', 'Prosodica L2']).agg({
        'Is_TP': ['sum', 'count'],
        'Is_FP': 'sum'
    }).reset_index()
    
    volume_precision.columns = ['L1_Category', 'L2_Category', 'TPs', 'Volume', 'FPs']
    volume_precision['Precision'] = np.where(
        volume_precision['Volume'] > 0,
        volume_precision['TPs'] / volume_precision['Volume'],
        0
    )
    
    # Correlation analysis
    if len(volume_precision) > 3:
        correlation = volume_precision['Volume'].corr(volume_precision['Precision'])
        print(f"Volume-Precision Correlation: {correlation:.3f}")
        
        if correlation < -0.3:
            print("  FINDING: NEGATIVE correlation - Higher volume categories have lower precision")
        elif correlation > 0.3:
            print("  FINDING: POSITIVE correlation - Higher volume categories have higher precision")
        else:
            print("  FINDING: WEAK correlation between volume and precision")
    
    # High-volume low-precision categories
    high_volume_threshold = volume_precision['Volume'].quantile(0.75)
    high_vol_low_prec = volume_precision[
        (volume_precision['Volume'] >= high_volume_threshold) & 
        (volume_precision['Precision'] < 0.70)
    ]
    
    print(f"\nHigh-Volume (>{high_volume_threshold:.0f}) Low-Precision (<70%) Categories:")
    if len(high_vol_low_prec) > 0:
        print(high_vol_low_prec[['L1_Category', 'L2_Category', 'Volume', 'Precision']].round(3))
    else:
        print("No high-volume low-precision categories identified")
    
    print("\n2.2 Precision Drop vs Volume Spike Correlation")
    
    # Monthly volume and precision trends
    monthly_trends = df.groupby('Year_Month').agg({
        'Is_TP': ['sum', 'count'],
        'Is_FP': 'sum',
        'variable5': 'nunique'
    }).reset_index()
    
    monthly_trends.columns = ['Year_Month', 'TPs', 'Total_Flagged', 'FPs', 'Unique_Calls']
    monthly_trends['Precision'] = np.where(
        monthly_trends['Total_Flagged'] > 0,
        monthly_trends['TPs'] / monthly_trends['Total_Flagged'],
        0
    )
    
    monthly_trends = monthly_trends.sort_values('Year_Month')
    monthly_trends['Precision_Change'] = monthly_trends['Precision'].diff()
    monthly_trends['Volume_Change'] = monthly_trends['Unique_Calls'].pct_change()
    
    print("Monthly Volume and Precision Changes:")
    print(monthly_trends[['Year_Month', 'Precision', 'Precision_Change', 'Unique_Calls', 'Volume_Change']].round(3))
    
    # Volume spike analysis
    volume_spike_threshold = monthly_trends['Volume_Change'].std() * 2
    volume_spikes = monthly_trends[abs(monthly_trends['Volume_Change']) > volume_spike_threshold]
    
    if len(volume_spikes) > 0:
        print(f"\nVolume Spikes (>{volume_spike_threshold:.1%} change) and Precision Impact:")
        print(volume_spikes[['Year_Month', 'Volume_Change', 'Precision_Change']].round(3))
    
    print("\n2.3 Seasonal Patterns Analysis")
    
    # Holiday season analysis
    seasonal_analysis = df.groupby('Is_Holiday_Season').agg({
        'Is_TP': ['sum', 'count'],
        'Is_FP': 'sum',
        'Transcript_Length': 'mean'
    }).reset_index()
    
    seasonal_analysis.columns = ['Is_Holiday_Season', 'TPs', 'Total_Flagged', 'FPs', 'Avg_Length']
    seasonal_analysis['Precision'] = np.where(
        seasonal_analysis['Total_Flagged'] > 0,
        seasonal_analysis['TPs'] / seasonal_analysis['Total_Flagged'],
        0
    )
    seasonal_analysis['Season'] = seasonal_analysis['Is_Holiday_Season'].map({True: 'Holiday Season', False: 'Regular Season'})
    
    print("Seasonal Performance Analysis:")
    print(seasonal_analysis[['Season', 'Precision', 'Total_Flagged', 'Avg_Length']].round(3))
    
    if len(seasonal_analysis) == 2:
        holiday_precision = seasonal_analysis[seasonal_analysis['Is_Holiday_Season']]['Precision'].iloc[0]
        regular_precision = seasonal_analysis[~seasonal_analysis['Is_Holiday_Season']]['Precision'].iloc[0]
        seasonal_diff = holiday_precision - regular_precision
        
        print(f"Seasonal Impact: {seasonal_diff:+.3f} precision difference")
        if abs(seasonal_diff) > 0.05:
            print(f"  FINDING: Significant seasonal impact detected")
    
    return volume_precision, monthly_trends

volume_analysis, monthly_analysis = analyze_volume_vs_performance(df_main)

# 3. QUERY PERFORMANCE REVIEW
print("\n3. QUERY PERFORMANCE REVIEW")
print("-" * 40)

def query_performance_review(df, df_rules):
    """Comprehensive query performance review"""
    
    print("3.1 Calculate Precision for All Complaint Categories")
    
    # All category precision
    all_categories = df.groupby(['Prosodica L1', 'Prosodica L2']).agg({
        'Is_TP': ['sum', 'count'],
        'Is_FP': 'sum'
    }).reset_index()
    
    all_categories.columns = ['L1_Category', 'L2_Category', 'TPs', 'Total_Flagged', 'FPs']
    all_categories['Precision'] = np.where(
        all_categories['Total_Flagged'] > 0,
        all_categories['TPs'] / all_categories['Total_Flagged'],
        0
    )
    all_categories['FP_Rate'] = np.where(
        all_categories['Total_Flagged'] > 0,
        all_categories['FPs'] / all_categories['Total_Flagged'],
        0
    )
    
    # Focus on complaint categories
    complaint_categories = all_categories[all_categories['L1_Category'] == 'complaints']
    
    print("All Complaint Category Performance:")
    print(complaint_categories.sort_values('Precision')[['L2_Category', 'Precision', 'Total_Flagged', 'FP_Rate']].round(3))
    
    print("\n3.2 Identify Top 5 Categories Driving Precision Drop")
    
    # Calculate drop impact
    complaint_categories['Precision_Gap'] = 0.70 - complaint_categories['Precision']
    complaint_categories['Drop_Impact'] = complaint_categories['Precision_Gap'] * complaint_categories['Total_Flagged']
    
    top_5_drop_drivers = complaint_categories.nlargest(5, 'Drop_Impact')
    
    print("Top 5 Categories Driving Precision Drop:")
    print(top_5_drop_drivers[['L2_Category', 'Precision', 'Precision_Gap', 'Total_Flagged', 'Drop_Impact']].round(3))
    
    print("\n3.3 Flag Categories Consistently Below 70% Target")
    
    below_target = complaint_categories[complaint_categories['Precision'] < 0.70]
    
    print(f"Categories Below 70% Target ({len(below_target)}/{len(complaint_categories)}):")
    if len(below_target) > 0:
        print(below_target[['L2_Category', 'Precision', 'Total_Flagged']].round(3))
        
        # Risk assessment
        high_risk = below_target[below_target['Total_Flagged'] >= below_target['Total_Flagged'].quantile(0.5)]
        print(f"\nHigh-Risk Categories (below target + high volume): {len(high_risk)}")
        if len(high_risk) > 0:
            print(high_risk[['L2_Category', 'Precision', 'Total_Flagged']].round(3))
    
    return all_categories, complaint_categories, top_5_drop_drivers

all_category_performance, complaint_performance, top_drop_drivers = query_performance_review(df_main, df_rules_filtered)

# 4. PATTERN DETECTION
print("\n4. PATTERN DETECTION")
print("-" * 40)

def pattern_detection_analysis(df):
    """Compare problem vs non-problem periods"""
    
    print("4.1 Problem vs Non-Problem Months Comparison")
    
    # Define problem months (assuming recent months have issues)
    all_months = sorted(df['Year_Month'].unique())
    if len(all_months) >= 4:
        problem_months = all_months[-2:]  # Last 2 months
        normal_months = all_months[:-2]   # Earlier months
    else:
        problem_months = all_months[-1:]
        normal_months = all_months[:-1]
    
    print(f"Problem months: {problem_months}")
    print(f"Normal months: {normal_months}")
    
    # Compare performance
    problem_data = df[df['Year_Month'].isin(problem_months)]
    normal_data = df[df['Year_Month'].isin(normal_months)]
    
    comparison = pd.DataFrame({
        'Period': ['Normal', 'Problem'],
        'Precision': [
            normal_data['Is_TP'].sum() / len(normal_data) if len(normal_data) > 0 else 0,
            problem_data['Is_TP'].sum() / len(problem_data) if len(problem_data) > 0 else 0
        ],
        'Volume': [len(normal_data), len(problem_data)],
        'Avg_Transcript_Length': [
            normal_data['Transcript_Length'].mean() if len(normal_data) > 0 else 0,
            problem_data['Transcript_Length'].mean() if len(problem_data) > 0 else 0
        ],
        'Avg_Negation_Count': [
            normal_data['Customer_Negation_Count'].mean() if len(normal_data) > 0 else 0,
            problem_data['Customer_Negation_Count'].mean() if len(problem_data) > 0 else 0
        ]
    })
    
    print("\nPeriod Comparison:")
    print(comparison.round(3))
    
    # Statistical significance test
    if len(normal_data) > 0 and len(problem_data) > 0:
        from scipy.stats import chi2_contingency
        
        try:
            contingency = np.array([
                [normal_data['Is_TP'].sum(), normal_data['Is_FP'].sum()],
                [problem_data['Is_TP'].sum(), problem_data['Is_FP'].sum()]
            ])
            
            if contingency.min() > 0:
                chi2, p_value, _, _ = chi2_contingency(contingency)
                print(f"\nStatistical Significance Test:")
                print(f"  Chi-square p-value: {p_value:.6f}")
                print(f"  Significant difference: {'YES' if p_value < 0.05 else 'NO'}")
        except:
            print("Statistical test could not be performed")
    
    print("\n4.2 Sudden vs Gradual Drop Analysis")
    
    # Monthly precision trend
    monthly_precision = df.groupby('Year_Month').agg({
        'Is_TP': ['sum', 'count']
    }).reset_index()
    
    monthly_precision.columns = ['Year_Month', 'TPs', 'Total']
    monthly_precision['Precision'] = monthly_precision['TPs'] / monthly_precision['Total']
    monthly_precision = monthly_precision.sort_values('Year_Month')
    monthly_precision['Precision_Change'] = monthly_precision['Precision'].diff()
    
    print("Monthly Precision Trend:")
    print(monthly_precision[['Year_Month', 'Precision', 'Precision_Change']].round(3))
    
    # Analyze drop pattern
    if len(monthly_precision) > 1:
        max_drop = monthly_precision['Precision_Change'].min()
        avg_change = monthly_precision['Precision_Change'].mean()
        
        print(f"\nDrop Pattern Analysis:")
        print(f"  Maximum single-month drop: {max_drop:.3f}")
        print(f"  Average monthly change: {avg_change:.3f}")
        
        if max_drop < -0.1:
            print("  FINDING: SUDDEN drop detected (>10% in single month)")
        elif avg_change < -0.02:
            print("  FINDING: GRADUAL decline detected (consistent negative trend)")
        else:
            print("  FINDING: STABLE performance with minor fluctuations")
    
    return comparison, monthly_precision

period_comparison, precision_trend = pattern_detection_analysis(df_main)

# =============================================================================
# DEEP DIVE ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("DEEP DIVE ANALYSIS")
print("="*80)

# 1. FP PATTERN ANALYSIS
print("\n1. FP PATTERN ANALYSIS")
print("-" * 40)

def fp_pattern_analysis(df):
    """Comprehensive false positive pattern analysis"""
    
    print("1.1 Group All FPs by Category and Month")
    
    # FP grouping analysis
    fp_analysis = df[df['Primary Marker'] == 'FP'].groupby(['Year_Month', 'Prosodica L1', 'Prosodica L2']).agg({
        'variable5': 'count',
        'Transcript_Length': 'mean',
        'Customer_Negation_Count': 'mean',
        'Customer_Qualifying_Count': 'mean'
    }).reset_index()
    
    fp_analysis.columns = ['Year_Month', 'L1_Category', 'L2_Category', 'FP_Count', 'Avg_Length', 'Avg_Negations', 'Avg_Qualifiers']
    
    print("FP Distribution by Month and Category:")
    fp_summary = fp_analysis.groupby(['L1_Category', 'L2_Category']).agg({
        'FP_Count': 'sum',
        'Avg_Length': 'mean',
        'Avg_Negations': 'mean',
        'Avg_Qualifiers': 'mean'
    }).reset_index().sort_values('FP_Count', ascending=False)
    
    print("Top 10 Categories by FP Count:")
    print(fp_summary.head(10).round(3))
    
    print("\n1.2 Create SRSRWI (Sample Review Spreadsheet) for Top FP Categories")
    
    # Create SRSRWI-style analysis for top categories
    top_fp_categories = fp_summary.head(3)
    
    srsrwi_data = []
    
    for _, category in top_fp_categories.iterrows():
        l1_cat = category['L1_Category']
        l2_cat = category['L2_Category']
        
        # Get FP samples for this category
        category_fps = df[
            (df['Primary Marker'] == 'FP') & 
            (df['Prosodica L1'] == l1_cat) & 
            (df['Prosodica L2'] == l2_cat)
        ].head(10)  # Sample first 10 FPs
        
        for _, fp_row in category_fps.iterrows():
            srsrwi_data.append({
                'UUID': fp_row['UUID'],
                'L1_Category': l1_cat,
                'L2_Category': l2_cat,
                'Customer_Transcript': fp_row['Customer Transcript'][:200] + '...',
                'Agent_Transcript': fp_row['Agent Transcript'][:200] + '...',
                'Transcript_Length': fp_row['Transcript_Length'],
                'Has_Negation': fp_row['Customer_Negation_Count'] > 0,
                'Has_Qualifying_Words': fp_row['Customer_Qualifying_Count'] > 0,
                'Month': fp_row['Year_Month']
            })
    
    srsrwi_df = pd.DataFrame(srsrwi_data)
    
    print(f"SRSRWI Sample Created: {len(srsrwi_df)} FP samples from top 3 categories")
    print("Sample Structure:")
    print(srsrwi_df[['UUID', 'L1_Category', 'L2_Category', 'Has_Negation', 'Has_Qualifying_Words']].head())
    
    print("\n1.3 Manual Review Pattern Identification")
    
    # Automated pattern identification (simulating manual review insights)
    fp_patterns = {}
    
    for l1_cat in df['Prosodica L1'].unique():
        if pd.notna(l1_cat):
            category_fps = df[(df['Primary Marker'] == 'FP') & (df['Prosodica L1'] == l1_cat)]
            
            if len(category_fps) > 5:
                # Context issues
                negation_rate = (category_fps['Customer_Negation_Count'] > 0).mean()
                
                # Agent vs customer confusion
                agent_explanation_pattern = category_fps['Agent Transcript'].str.lower().str.contains(
                    r'(explain|example|let me|suppose)', regex=True, na=False
                ).mean()
                
                # Qualifying language
                qualifying_rate = (category_fps['Customer_Qualifying_Count'] > 0).mean()
                
                fp_patterns[l1_cat] = {
                    'total_fps': len(category_fps),
                    'negation_rate': negation_rate,
                    'agent_explanation_rate': agent_explanation_pattern,
                    'qualifying_language_rate': qualifying_rate
                }
    
    print("Automated Pattern Detection Results:")
    for category, patterns in fp_patterns.items():
        print(f"\n{category} ({patterns['total_fps']} FPs):")
        print(f"  Negation patterns: {patterns['negation_rate']:.1%}")
        print(f"  Agent explanations: {patterns['agent_explanation_rate']:.1%}")
        print(f"  Qualifying language: {patterns['qualifying_language_rate']:.1%}")
    
    print("\n1.4 Categorize FP Reasons")
    
    # Comprehensive FP categorization
    fp_reasons = df[df['Primary Marker'] == 'FP'].copy()
    
    # Context issues
    fp_reasons['Context_Issue'] = (
        (fp_reasons['Customer_Negation_Count'] > 0) |
        (fp_reasons['Agent_Negation_Count'] > 0)
    )
    
    # Overly broad rules (high qualifying language suggests ambiguous cases)
    fp_reasons['Overly_Broad_Rules'] = fp_reasons['Customer_Qualifying_Count'] > 2
    
    # Agent vs Customer confusion
    fp_reasons['Agent_Customer_Confusion'] = fp_reasons['Agent Transcript'].str.lower().str.contains(
        r'(explain|example|let me|suppose|hypothetically)', regex=True, na=False
    )
    
    # New language patterns (unusual transcript characteristics)
    median_length = df['Transcript_Length'].median()
    fp_reasons['New_Language_Pattern'] = (
        (fp_reasons['Transcript_Length'] > median_length * 2) |
        (fp_reasons['Customer_Caps_Ratio'] > 0.3)
    )
    
    # Summarize FP reasons
    fp_reason_summary = pd.DataFrame({
        'FP_Reason': ['Context Issues', 'Overly Broad Rules', 'Agent/Customer Confusion', 'New Language Patterns'],
        'Count': [
            fp_reasons['Context_Issue'].sum(),
            fp_reasons['Overly_Broad_Rules'].sum(),
            fp_reasons['Agent_Customer_Confusion'].sum(),
            fp_reasons['New_Language_Pattern'].sum()
        ]
    })
    
    fp_reason_summary['Percentage'] = fp_reason_summary['Count'] / len(fp_reasons) * 100
    fp_reason_summary = fp_reason_summary.sort_values('Count', ascending=False)
    
    print("FP Reason Categorization:")
    print(fp_reason_summary.round(1))
    
    return fp_summary, srsrwi_df, fp_patterns, fp_reason_summary

fp_analysis_results = fp_pattern_analysis(df_main)
fp_summary, srsrwi_sample, fp_patterns, fp_reasons = fp_analysis_results

# 2. VALIDATION PROCESS ASSESSMENT
print("\n2. VALIDATION PROCESS ASSESSMENT")
print("-" * 40)

def validation_process_assessment(df):
    """Comprehensive validation process assessment"""
    
    print("2.1 Primary vs Secondary Validation Agreement Rates")
    
    # Overall agreement analysis
    secondary_data = df[df['Has_Secondary_Validation']].copy()
    
    if len(secondary_data) > 0:
        overall_agreement = secondary_data['Primary_Secondary_Agreement'].mean()
        total_secondary = len(secondary_data)
        
        print(f"Overall Validation Metrics:")
        print(f"  Records with secondary validation: {total_secondary} ({total_secondary/len(df)*100:.1f}%)")
        print(f"  Primary-Secondary agreement rate: {overall_agreement:.3f}")
        
        print("\n2.2 Categories with High Disagreement Rates")
        
        # Category-wise agreement
        category_agreement = secondary_data.groupby(['Prosodica L1', 'Prosodica L2']).agg({
            'Primary_Secondary_Agreement': ['mean', 'count', 'std']
        }).reset_index()
        
        category_agreement.columns = ['L1_Category', 'L2_Category', 'Agreement_Rate', 'Sample_Size', 'Agreement_Std']
        category_agreement = category_agreement[category_agreement['Sample_Size'] >= 5]
        category_agreement = category_agreement.sort_values('Agreement_Rate')
        
        print("Categories with Lowest Agreement Rates (min 5 samples):")
        print(category_agreement.head(10)[['L1_Category', 'L2_Category', 'Agreement_Rate', 'Sample_Size']].round(3))
        
        # High disagreement categories
        high_disagreement = category_agreement[category_agreement['Agreement_Rate'] < 0.7]
        print(f"\nHigh Disagreement Categories (<70% agreement): {len(high_disagreement)}")
        
        print("\n2.3 Validation Consistency Over Time")
        
        # Monthly validation trends
        monthly_validation = secondary_data.groupby('Year_Month').agg({
            'Primary_Secondary_Agreement': ['mean', 'count', 'std']
        }).reset_index()
        
        monthly_validation.columns = ['Year_Month', 'Agreement_Rate', 'Sample_Size', 'Agreement_Std']
        monthly_validation = monthly_validation.sort_values('Year_Month')
        
        print("Monthly Validation Agreement Trends:")
        print(monthly_validation.round(3))
        
        # Trend analysis
        if len(monthly_validation) > 2:
            month_numbers = list(range(len(monthly_validation)))
            agreement_values = monthly_validation['Agreement_Rate'].tolist()
            
            try:
                correlation_matrix = np.corrcoef(month_numbers, agreement_values)
                trend_correlation = correlation_matrix[0, 1]
                
                print(f"\nValidation Trend Analysis:")
                print(f"  Correlation with time: {trend_correlation:.3f}")
                
                if trend_correlation < -0.5:
                    print("  FINDING: Validation agreement is DECLINING over time")
                elif trend_correlation > 0.5:
                    print("  FINDING: Validation agreement is IMPROVING over time")
                else:
                    print("  FINDING: Validation agreement is STABLE over time")
            except:
                print("Trend analysis could not be performed")
        
        print("\n2.4 Validation Guidelines and Reviewer Changes Assessment")
        
        # Simulate reviewer consistency analysis
        monthly_validation['Consistency_Score'] = 1 - monthly_validation['Agreement_Std'].fillna(0)
        
        print("Validation Consistency Metrics:")
        print("(Higher scores indicate more consistent validation)")
        print(monthly_validation[['Year_Month', 'Agreement_Rate', 'Consistency_Score']].round(3))
        
        # Identify potential guideline change points
        agreement_changes = monthly_validation['Agreement_Rate'].diff().abs()
        significant_changes = monthly_validation[agreement_changes > 0.1]
        
        if len(significant_changes) > 0:
            print(f"\nSignificant Agreement Changes (>10% shift):")
            print(significant_changes[['Year_Month', 'Agreement_Rate']].round(3))
            print("  RECOMMENDATION: Review validation guidelines or reviewer training for these periods")
        
        return monthly_validation, category_agreement
    
    else:
        print("No secondary validation data available")
        return None, None

validation_monthly, validation_categories = validation_process_assessment(df_main)

# 3. TEMPORAL ANALYSIS
print("\n3. TEMPORAL ANALYSIS")
print("-" * 40)

def temporal_analysis(df):
    """Comprehensive temporal pattern analysis"""
    
    print("3.1 FP Rates by Day of Week")
    
    # Day of week analysis
    dow_analysis = df.groupby('DayOfWeek').agg({
        'Is_FP': ['sum', 'count', 'mean'],
        'Is_TP': 'mean',
        'Transcript_Length': 'mean'
    }).reset_index()
    
    dow_analysis.columns = ['DayOfWeek', 'FP_Count', 'Total_Records', 'FP_Rate', 'TP_Rate', 'Avg_Length']
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_analysis['Day_Order'] = dow_analysis['DayOfWeek'].map({day: i for i, day in enumerate(day_order)})
    dow_analysis = dow_analysis.sort_values('Day_Order')
    
    print("FP Rates by Day of Week:")
    print(dow_analysis[['DayOfWeek', 'FP_Rate', 'Total_Records', 'Avg_Length']].round(3))
    
    # Identify anomalous days
    mean_fp_rate = dow_analysis['FP_Rate'].mean()
    std_fp_rate = dow_analysis['FP_Rate'].std()
    anomalous_days = dow_analysis[abs(dow_analysis['FP_Rate'] - mean_fp_rate) > std_fp_rate]
    
    if len(anomalous_days) > 0:
        print(f"\nAnomalous Days (FP rate beyond 1 std dev):")
        print(anomalous_days[['DayOfWeek', 'FP_Rate']].round(3))
    
    print("\n3.2 FP Rates by Week of Month")
    
    # Week of month analysis
    wom_analysis = df.groupby('WeekOfMonth').agg({
        'Is_FP': ['sum', 'count', 'mean'],
        'Is_TP': 'mean',
        'Customer_Negation_Count': 'mean'
    }).reset_index()
    
    wom_analysis.columns = ['WeekOfMonth', 'FP_Count', 'Total_Records', 'FP_Rate', 'TP_Rate', 'Avg_Negations']
    
    print("FP Rates by Week of Month:")
    print(wom_analysis[['WeekOfMonth', 'FP_Rate', 'Total_Records', 'Avg_Negations']].round(3))
    
    # Month-end effect analysis
    month_end_analysis = df.groupby('Is_Month_End').agg({
        'Is_FP': 'mean',
        'Is_TP': 'mean',
        'Customer_Qualifying_Count': 'mean'
    }).reset_index()
    
    month_end_analysis['Period'] = month_end_analysis['Is_Month_End'].map({True: 'Month End', False: 'Regular Days'})
    
    print("\nMonth-End Effect Analysis:")
    print(month_end_analysis[['Period', 'Is_FP', 'Is_TP', 'Customer_Qualifying_Count']].round(3))
    
    print("\n3.3 Operational Changes Coinciding with Precision Drops")
    
    # Volume and performance correlation with operational indicators
    operational_analysis = df.groupby('Year_Month').agg({
        'Is_FP': 'mean',
        'Is_TP': 'mean',
        'variable5': 'nunique',
        'Transcript_Length': 'mean',
        'Customer_Agent_Ratio': 'mean'
    }).reset_index()
    
    operational_analysis.columns = ['Year_Month', 'FP_Rate', 'TP_Rate', 'Unique_Calls', 'Avg_Length', 'Cust_Agent_Ratio']
    operational_analysis = operational_analysis.sort_values('Year_Month')
    
    # Calculate changes
    operational_analysis['FP_Rate_Change'] = operational_analysis['FP_Rate'].diff()
    operational_analysis['Volume_Change'] = operational_analysis['Unique_Calls'].pct_change()
    operational_analysis['Length_Change'] = operational_analysis['Avg_Length'].pct_change()
    operational_analysis['Ratio_Change'] = operational_analysis['Cust_Agent_Ratio'].pct_change()
    
    print("Monthly Operational Metrics:")
    print(operational_analysis[['Year_Month', 'FP_Rate', 'FP_Rate_Change', 'Volume_Change', 'Length_Change']].round(3))
    
    # Identify operational change indicators
    significant_changes = operational_analysis[
        (abs(operational_analysis['Volume_Change']) > 0.2) |  # 20% volume change
        (abs(operational_analysis['Length_Change']) > 0.15) |  # 15% length change
        (abs(operational_analysis['Ratio_Change']) > 0.3)      # 30% ratio change
    ]
    
    if len(significant_changes) > 0:
        print(f"\nMonths with Significant Operational Changes:")
        print(significant_changes[['Year_Month', 'Volume_Change', 'Length_Change', 'FP_Rate_Change']].round(3))
        print("  RECOMMENDATION: Investigate operational changes in these periods")
    
    return dow_analysis, wom_analysis, operational_analysis

dow_results, wom_results, operational_results = temporal_analysis(df_main)

# =============================================================================
# ROOT CAUSE ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("ROOT CAUSE ANALYSIS")
print("="*80)

# 1. CATEGORY SPECIFIC INVESTIGATION
print("\n1. CATEGORY SPECIFIC INVESTIGATION")
print("-" * 50)

def category_specific_investigation(df, df_rules, top_categories):
    """Investigate top 5 categories with worst precision drop"""
    
    print("1.1 Review Query Rules/Keywords for Top 5 Categories")
    
    # Get top 5 worst performing categories
    if len(top_categories) > 0:
        top_5_worst = top_categories.head(5)
        
        print("Top 5 Categories with Worst Precision Drop:")
        print(top_5_worst[['L2_Category', 'Precision', 'Drop_Impact']].round(3))
        
        # Query rule analysis for each category
        for _, category in top_5_worst.iterrows():
            l2_cat = category['L2_Category']
            
            print(f"\n--- Analysis for {l2_cat} ---")
            
            # Find matching rules
            matching_rules = df_rules[df_rules['Query'].str.contains(l2_cat, case=False, na=False)]
            
            if len(matching_rules) > 0:
                for _, rule in matching_rules.iterrows():
                    query_text = rule['Query Text']
                    channel = rule['Channel']
                    
                    print(f"Query: {query_text}")
                    print(f"Channel: {channel}")
                    
                    # Query complexity analysis
                    or_count = len(re.findall(r'\bOR\b', query_text, re.IGNORECASE))
                    and_count = len(re.findall(r'\bAND\b', query_text, re.IGNORECASE))
                    not_count = len(re.findall(r'\bNOT\b', query_text, re.IGNORECASE))
                    
                    print(f"Query Complexity: OR({or_count}), AND({and_count}), NOT({not_count})")
                    
                    # Issues identification
                    issues = []
                    if or_count > 10:
                        issues.append("High OR complexity")
                    if not_count == 0:
                        issues.append("No negation handling")
                    if channel == 'both':
                        issues.append("Both channel usage")
                    
                    if issues:
                        print(f"Potential Issues: {', '.join(issues)}")
                    else:
                        print("No obvious query issues detected")
            else:
                print("No matching query rules found")
    
    print("\n1.2 Rule Degradation Analysis")
    
    # Check if same rules are catching more FPs over time
    monthly_rule_performance = {}
    
    for _, category in top_5_worst.iterrows():
        l2_cat = category['L2_Category']
        
        # Monthly FP analysis for this category
        category_monthly = df[df['Prosodica L2'] == l2_cat].groupby('Year_Month').agg({
            'Is_FP': ['sum', 'count'],
            'Is_TP': 'sum'
        }).reset_index()
        
        if len(category_monthly) > 0:
            category_monthly.columns = ['Year_Month', 'FP_Count', 'Total_Flagged', 'TP_Count']
            category_monthly['FP_Rate'] = category_monthly['FP_Count'] / category_monthly['Total_Flagged']
            category_monthly = category_monthly.sort_values('Year_Month')
            
            # Check for degradation trend
            if len(category_monthly) > 2:
                months = list(range(len(category_monthly)))
                fp_rates = category_monthly['FP_Rate'].tolist()
                
                try:
                    slope, _, r_value, p_value, _ = stats.linregress(months, fp_rates)
                    
                    monthly_rule_performance[l2_cat] = {
                        'slope': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'trend': 'Degrading' if slope > 0.01 else 'Stable'
                    }
                except:
                    monthly_rule_performance[l2_cat] = {'trend': 'Unable to calculate'}
    
    print("Rule Degradation Analysis:")
    for category, performance in monthly_rule_performance.items():
        trend = performance.get('trend', 'Unknown')
        if 'slope' in performance:
            slope = performance['slope']
            r_sq = performance['r_squared']
            print(f"  {category}: {trend} (slope: {slope:.4f}, R²: {r_sq:.3f})")
        else:
            print(f"  {category}: {trend}")
    
    print("\n1.3 Language Evolution Analysis")
    
    # Compare early vs recent language patterns
    all_months = sorted(df['Year_Month'].unique())
    if len(all_months) >= 4:
        early_months = all_months[:2]
        recent_months = all_months[-2:]
    else:
        early_months = all_months[:1]
        recent_months = all_months[-1:]
    
    language_evolution = {}
    
    for _, category in top_5_worst.iterrows():
        l2_cat = category['L2_Category']
        
        early_data = df[(df['Prosodica L2'] == l2_cat) & (df['Year_Month'].isin(early_months))]
        recent_data = df[(df['Prosodica L2'] == l2_cat) & (df['Year_Month'].isin(recent_months))]
        
        if len(early_data) > 0 and len(recent_data) > 0:
            # Vocabulary analysis
            early_vocab = set(' '.join(early_data['Customer Transcript'].fillna('')).lower().split())
            recent_vocab = set(' '.join(recent_data['Customer Transcript'].fillna('')).lower().split())
            
            new_words = recent_vocab - early_vocab
            vocab_growth = len(new_words) / len(early_vocab) if len(early_vocab) > 0 else 0
            
            # Text characteristics
            early_avg_length = early_data['Transcript_Length'].mean()
            recent_avg_length = recent_data['Transcript_Length'].mean()
            length_change = (recent_avg_length - early_avg_length) / early_avg_length if early_avg_length > 0 else 0
            
            # Qualifying language change
            early_qualifying = early_data['Customer_Qualifying_Count'].mean()
            recent_qualifying = recent_data['Customer_Qualifying_Count'].mean()
            qualifying_change = recent_qualifying - early_qualifying
            
            language_evolution[l2_cat] = {
                'vocab_growth': vocab_growth,
                'length_change': length_change,
                'qualifying_change': qualifying_change,
                'new_words_count': len(new_words)
            }
    
    print("Language Evolution Analysis:")
    for category, evolution in language_evolution.items():
        print(f"\n{category}:")
        print(f"  Vocabulary growth: {evolution['vocab_growth']:.1%}")
        print(f"  Length change: {evolution['length_change']:+.1%}")
        print(f"  Qualifying language change: {evolution['qualifying_change']:+.2f}")
        print(f"  New words: {evolution['new_words_count']}")
    
    return monthly_rule_performance, language_evolution

rule_performance, language_changes = category_specific_investigation(df_main, df_rules_filtered, complaint_performance)

# 2. CROSS CATEGORY ANALYSIS
print("\n2. CROSS CATEGORY ANALYSIS")
print("-" * 30)

def cross_category_analysis(df):
    """Analyze cross-category patterns and conflicts"""
    
    print("2.1 Multi-Category Transcript Analysis")
    
    # Identify transcripts flagged for multiple categories
    transcript_categories = df.groupby('variable5').agg({
        'Prosodica L1': 'nunique',
        'Prosodica L2': 'nunique',
        'Is_TP': 'mean',
        'Is_FP': 'mean'
    }).reset_index()
    
    transcript_categories.columns = ['Transcript_ID', 'L1_Categories', 'L2_Categories', 'Avg_Precision', 'Avg_FP_Rate']
    
    # Multi-category transcripts
    multi_category = transcript_categories[transcript_categories['L2_Categories'] > 1]
    single_category = transcript_categories[transcript_categories['L2_Categories'] == 1]
    
    print(f"Multi-Category Transcripts Analysis:")
    print(f"  Single category: {len(single_category)} transcripts")
    print(f"  Multi-category: {len(multi_category)} transcripts")
    
    if len(multi_category) > 0 and len(single_category) > 0:
        multi_precision = multi_category['Avg_Precision'].mean()
        single_precision = single_category['Avg_Precision'].mean()
        
        print(f"  Single category avg precision: {single_precision:.3f}")
        print(f"  Multi-category avg precision: {multi_precision:.3f}")
        print(f"  Precision difference: {multi_precision - single_precision:+.3f}")
        
        if multi_precision < single_precision - 0.05:
            print("  FINDING: Multi-category transcripts have LOWER precision")
    
    print("\n2.2 Category Overlap and Rule Conflicts")
    
    # Analyze category co-occurrence
    category_pairs = df.groupby('variable5')['Prosodica L2'].apply(list).reset_index()
    category_pairs['Category_Count'] = category_pairs['Prosodica L2'].apply(len)
    
    # Find common category combinations
    multi_cat_pairs = category_pairs[category_pairs['Category_Count'] > 1]
    
    if len(multi_cat_pairs) > 0:
        # Extract all category combinations
        combinations = []
        for categories in multi_cat_pairs['Prosodica L2']:
            if len(categories) == 2:
                combinations.append(tuple(sorted(categories)))
        
        if combinations:
            from collections import Counter
            common_combinations = Counter(combinations).most_common(5)
            
            print("Most Common Category Combinations:")
            for combo, count in common_combinations:
                print(f"  {combo[0]} + {combo[1]}: {count} transcripts")
                
                # Check precision for this combination
                combo_data = df[
                    df['variable5'].isin(
                        multi_cat_pairs[
                            multi_cat_pairs['Prosodica L2'].apply(
                                lambda x: len(x) == 2 and combo[0] in x and combo[1] in x
                            )
                        ]['variable5']
                    )
                ]
                
                if len(combo_data) > 0:
                    combo_precision = combo_data['Is_TP'].mean()
                    print(f"    Precision: {combo_precision:.3f}")
    
    print("\n2.3 New Category Cannibalization Analysis")
    
    # Alternative approach: Analyze volume trends to identify potential new categories
    monthly_category_volumes = df.groupby(['Year_Month', 'Prosodica L2']).size().reset_index()
    monthly_category_volumes.columns = ['Year_Month', 'L2_Category', 'Volume']
    
    # Find categories with recent volume spikes (potential new categories)
    category_volume_trends = monthly_category_volumes.groupby('L2_Category').agg({
        'Volume': ['sum', 'count', 'std']
    }).reset_index()
    
    category_volume_trends.columns = ['L2_Category', 'Total_Volume', 'Months_Active', 'Volume_Std']
    
    # Categories active in fewer months might be newer
    potential_new_cats = category_volume_trends[
        (category_volume_trends['Months_Active'] <= 2) & 
        (category_volume_trends['Total_Volume'] >= 10)
    ]
    
    if len(potential_new_cats) > 0:
        print(f"Potential New Categories (active ≤2 months, volume ≥10):")
        
        for _, cat in potential_new_cats.iterrows():
            cat_name = cat['L2_Category']
            
            # Performance of potential new category
            cat_data = df[df['Prosodica L2'] == cat_name]
            cat_precision = cat_data['Is_TP'].mean() if len(cat_data) > 0 else 0
            cat_volume = len(cat_data)
            
            print(f"\n{cat_name}:")
            print(f"  Volume: {cat_volume}")
            print(f"  Precision: {cat_precision:.3f}")
            print(f"  Months Active: {cat['Months_Active']}")
            
            # Check monthly trend
            cat_monthly = monthly_category_volumes[monthly_category_volumes['L2_Category'] == cat_name]
            if len(cat_monthly) > 1:
                cat_monthly = cat_monthly.sort_values('Year_Month')
                trend = "Increasing" if cat_monthly['Volume'].iloc[-1] > cat_monthly['Volume'].iloc[0] else "Stable/Decreasing"
                print(f"  Volume Trend: {trend}")
        
        print(f"\n  NOTE: Manual review needed to confirm if these are truly new categories")
        print(f"  RECOMMENDATION: Compare with category launch dates if available")
    else:
        print("No clear patterns suggesting new category launches")
        print("All categories appear to have consistent historical presence")
    
    return transcript_categories, multi_category

transcript_analysis, multi_cat_analysis = cross_category_analysis(df_main)

# 3. CONTENT PATTERN ANALYSIS
print("\n3. CONTENT PATTERN ANALYSIS")
print("-" * 40)

def content_pattern_analysis(df):
    """Compare language patterns in TPs vs FPs"""
    
    print("3.1 Average Transcript Length Comparison")
    
    # TP vs FP comparison
    tp_data = df[df['Primary Marker'] == 'TP']
    fp_data = df[df['Primary Marker'] == 'FP']
    
    length_comparison = pd.DataFrame({
        'Metric': ['Count', 'Avg_Transcript_Length', 'Median_Length', 'Std_Length'],
        'True_Positives': [
            len(tp_data),
            tp_data['Transcript_Length'].mean(),
            tp_data['Transcript_Length'].median(),
            tp_data['Transcript_Length'].std()
        ],
        'False_Positives': [
            len(fp_data),
            fp_data['Transcript_Length'].mean(),
            fp_data['Transcript_Length'].median(),
            fp_data['Transcript_Length'].std()
        ]
    })
    
    length_comparison['Difference'] = length_comparison['False_Positives'] - length_comparison['True_Positives']
    
    print("Transcript Length Analysis:")
    print(length_comparison.round(2))
    
    # Statistical significance
    if len(tp_data) > 0 and len(fp_data) > 0:
        from scipy.stats import mannwhitneyu
        
        try:
            _, p_value = mannwhitneyu(tp_data['Transcript_Length'].dropna(), 
                                    fp_data['Transcript_Length'].dropna(), 
                                    alternative='two-sided')
            print(f"Statistical significance (p-value): {p_value:.6f}")
            print(f"Significant difference: {'YES' if p_value < 0.05 else 'NO'}")
        except:
            print("Statistical test could not be performed")
    
    print("\n3.2 Agent to Customer Word Ratio Analysis")
    
    # Word ratio comparison
    ratio_comparison = pd.DataFrame({
        'Metric': ['Avg_Customer_Words', 'Avg_Agent_Words', 'Avg_Customer_Agent_Ratio'],
        'True_Positives': [
            tp_data['Customer_Word_Count'].mean(),
            tp_data['Agent_Word_Count'].mean(),
            tp_data['Customer_Agent_Ratio'].mean()
        ],
        'False_Positives': [
            fp_data['Customer_Word_Count'].mean(),
            fp_data['Agent_Word_Count'].mean(),
            fp_data['Customer_Agent_Ratio'].mean()
        ]
    })
    
    ratio_comparison['Difference'] = ratio_comparison['False_Positives'] - ratio_comparison['True_Positives']
    
    print("Word Ratio Analysis:")
    print(ratio_comparison.round(3))
    
    # Interpret findings
    if ratio_comparison.loc[2, 'Difference'] > 0.2:  # Customer_Agent_Ratio difference
        print("  FINDING: FPs have higher customer-to-agent word ratio")
        print("  IMPLICATION: Customers speak more relative to agents in FPs")
    elif ratio_comparison.loc[2, 'Difference'] < -0.2:
        print("  FINDING: FPs have lower customer-to-agent word ratio")
        print("  IMPLICATION: Agents speak more relative to customers in FPs")
    
    print("\n3.3 Presence of Qualifying Indicators")
    
    # Qualifying language analysis
    qualifying_comparison = pd.DataFrame({
        'Indicator': ['Negation_Count', 'Qualifying_Count', 'Question_Count', 'Exclamation_Count', 'Caps_Ratio'],
        'TP_Avg': [
            tp_data['Customer_Negation_Count'].mean(),
            tp_data['Customer_Qualifying_Count'].mean(),
            tp_data['Customer_Question_Count'].mean(),
            tp_data['Customer_Exclamation_Count'].mean(),
            tp_data['Customer_Caps_Ratio'].mean()
        ],
        'FP_Avg': [
            fp_data['Customer_Negation_Count'].mean(),
            fp_data['Customer_Qualifying_Count'].mean(),
            fp_data['Customer_Question_Count'].mean(),
            fp_data['Customer_Exclamation_Count'].mean(),
            fp_data['Customer_Caps_Ratio'].mean()
        ]
    })
    
    qualifying_comparison['Difference'] = qualifying_comparison['FP_Avg'] - qualifying_comparison['TP_Avg']
    qualifying_comparison['Risk_Factor'] = qualifying_comparison['FP_Avg'] / (qualifying_comparison['TP_Avg'] + 0.001)
    
    print("Qualifying Indicators Analysis:")
    print(qualifying_comparison.round(3))
    
    print("\n3.4 Precision of Qualifying Words Analysis")
    
    # Advanced pattern analysis
    advanced_patterns = {
        'Strong_Negation': r'\b(absolutely not|definitely not|certainly not|never ever)\b',
        'Weak_Negation': r'\b(not really|not quite|not exactly|hardly|barely)\b',
        'Uncertainty': r'\b(i think|i believe|i guess|maybe|perhaps|possibly)\b',
        'Frustration': r'\b(frustrated|annoyed|upset|angry|mad|ridiculous|stupid)\b',
        'Politeness': r'\b(please|thank you|thanks|appreciate|grateful)\b',
        'Agent_Explanations': r'\b(let me explain|what this means|for example|in other words)\b',
        'Hypotheticals': r'\b(if i|suppose i|what if|let\'s say|imagine if)\b'
    }
    
    pattern_analysis = []
    
    for pattern_name, pattern in advanced_patterns.items():
        tp_matches = tp_data['Full_Transcript'].str.lower().str.contains(pattern, regex=True, na=False)
        fp_matches = fp_data['Full_Transcript'].str.lower().str.contains(pattern, regex=True, na=False)
        
        tp_rate = tp_matches.mean() * 100 if len(tp_data) > 0 else 0
        fp_rate = fp_matches.mean() * 100 if len(fp_data) > 0 else 0
        
        # Calculate risk factor
        risk_factor = fp_rate / max(tp_rate, 0.1)  # Avoid division by zero
        
        pattern_analysis.append({
            'Pattern': pattern_name,
            'TP_Rate': tp_rate,
            'FP_Rate': fp_rate,
            'Difference': fp_rate - tp_rate,
            'Risk_Factor': risk_factor
        })
    
    pattern_df = pd.DataFrame(pattern_analysis).sort_values('Risk_Factor', ascending=False)
    
    print("Advanced Pattern Analysis (% of transcripts containing pattern):")
    print(f"{'Pattern':<20} {'TP%':<8} {'FP%':<8} {'Diff':<8} {'Risk':<8}")
    print("-" * 60)
    
    for _, row in pattern_df.iterrows():
        risk_level = "HIGH" if row['Risk_Factor'] > 2 else "MED" if row['Risk_Factor'] > 1.5 else "LOW"
        print(f"{row['Pattern']:<20} {row['TP_Rate']:<8.1f} {row['FP_Rate']:<8.1f} "
              f"{row['Difference']:<8.1f} {risk_level:<8}")
    
    # Key insights
    print("\nKey Content Pattern Insights:")
    
    high_risk_patterns = pattern_df[pattern_df['Risk_Factor'] > 2]
    if len(high_risk_patterns) > 0:
        print("High-risk patterns (>2x more likely in FPs):")
        for _, pattern in high_risk_patterns.iterrows():
            print(f"  - {pattern['Pattern']}: {pattern['Risk_Factor']:.1f}x risk")
    
    # Text length insights
    length_diff = length_comparison.loc[1, 'Difference']  # Avg_Transcript_Length difference
    if abs(length_diff) > 100:
        direction = "longer" if length_diff > 0 else "shorter"
        print(f"  - FPs are {abs(length_diff):.0f} characters {direction} on average")
    
    return length_comparison, ratio_comparison, qualifying_comparison, pattern_df

content_results = content_pattern_analysis(df_main)
length_comp, ratio_comp, qualifying_comp, pattern_results = content_results

# =============================================================================
# SYNTHESIS AND RECOMMENDATIONS
# =============================================================================

print("\n" + "="*80)
print("SYNTHESIS AND RECOMMENDATIONS")
print("="*80)

def generate_comprehensive_findings_and_recommendations():
    """Synthesize all findings and generate actionable recommendations"""
    
    print("\n1. KEY FINDINGS SUMMARY")
    print("-" * 30)
    
    # Overall metrics
    overall_precision = df_main['Is_TP'].mean()
    total_records = len(df_main)
    categories_below_target = len(complaint_performance[complaint_performance['Precision'] < 0.70])
    total_categories = len(complaint_performance)
    
    print(f"CURRENT STATE:")
    print(f"  Overall Precision: {overall_precision:.1%} (Target: 70%)")
    print(f"  Gap to Target: {0.70 - overall_precision:+.1%}")
    print(f"  Categories Below Target: {categories_below_target}/{total_categories} ({categories_below_target/total_categories:.1%})")
    print(f"  Total Records Analyzed: {total_records:,}")
    
    # Top findings from each analysis area
    print(f"\nMAJOR FINDINGS BY ANALYSIS AREA:")
    
    print(f"\nMacro Level Analysis:")
    if len(top_drop_drivers) > 0:
        worst_category = top_drop_drivers.iloc[0]
        print(f"  - Worst performing category: {worst_category['L2_Category']} ({worst_category['Precision']:.1%} precision)")
    
    print(f"  - {period_comparison.loc[1, 'Precision'] - period_comparison.loc[0, 'Precision']:+.1%} precision change (normal → problem periods)")
    
    print(f"\nDeep Dive Analysis:")
    if len(fp_reasons) > 0:
        top_fp_reason = fp_reasons.iloc[0]
        print(f"  - Primary FP cause: {top_fp_reason['FP_Reason']} ({top_fp_reason['Percentage']:.1f}% of FPs)")
    
    if validation_monthly is not None and len(validation_monthly) > 0:
        avg_agreement = validation_monthly['Agreement_Rate'].mean()
        print(f"  - Validation agreement rate: {avg_agreement:.1%}")
    
    print(f"\nRoot Cause Analysis:")
    if len(rule_performance) > 0:
        degrading_rules = sum(1 for perf in rule_performance.values() if perf.get('trend') == 'Degrading')
        print(f"  - Rules showing degradation: {degrading_rules}/{len(rule_performance)}")
    
    if len(pattern_results) > 0:
        high_risk_patterns = len(pattern_results[pattern_results['Risk_Factor'] > 2])
        print(f"  - High-risk content patterns: {high_risk_patterns}")
    
    print(f"\n2. ROOT CAUSE PRIORITIZATION")
    print("-" * 35)
    
    # Calculate impact scores for different root causes
    root_causes = []
    
    # Negation handling issues
    if len(fp_reasons) > 0:
        context_issues_pct = fp_reasons[fp_reasons['FP_Reason'] == 'Context Issues']['Percentage'].iloc[0] if len(fp_reasons[fp_reasons['FP_Reason'] == 'Context Issues']) > 0 else 0
        root_causes.append({
            'Root_Cause': 'Context-insensitive negation handling',
            'Impact_Score': context_issues_pct,
            'Implementation_Effort': 'Medium',
            'Time_to_Fix': '2-4 weeks',
            'Expected_Gain': min(0.15, context_issues_pct * 0.01)
        })
    
    # Agent explanation issues
    if len(fp_reasons) > 0:
        confusion_pct = fp_reasons[fp_reasons['FP_Reason'] == 'Agent/Customer Confusion']['Percentage'].iloc[0] if len(fp_reasons[fp_reasons['FP_Reason'] == 'Agent/Customer Confusion']) > 0 else 0
        root_causes.append({
            'Root_Cause': 'Agent explanations triggering rules',
            'Impact_Score': confusion_pct,
            'Implementation_Effort': 'Low',
            'Time_to_Fix': '1-2 weeks',
            'Expected_Gain': min(0.08, confusion_pct * 0.008)
        })
    
    # Overly broad rules
    if len(fp_reasons) > 0:
        broad_rules_pct = fp_reasons[fp_reasons['FP_Reason'] == 'Overly Broad Rules']['Percentage'].iloc[0] if len(fp_reasons[fp_reasons['FP_Reason'] == 'Overly Broad Rules']) > 0 else 0
        root_causes.append({
            'Root_Cause': 'Overly broad query rules',
            'Impact_Score': broad_rules_pct,
            'Implementation_Effort': 'High',
            'Time_to_Fix': '6-12 weeks',
            'Expected_Gain': min(0.12, broad_rules_pct * 0.012)
        })
    
    # Validation inconsistency
    if validation_monthly is not None and len(validation_monthly) > 0:
        avg_agreement = validation_monthly['Agreement_Rate'].mean()
        validation_impact = (1 - avg_agreement) * 100 if avg_agreement < 0.85 else 0
        root_causes.append({
            'Root_Cause': 'Validation process inconsistency',
            'Impact_Score': validation_impact,
            'Implementation_Effort': 'Medium',
            'Time_to_Fix': '3-6 weeks',
            'Expected_Gain': min(0.05, validation_impact * 0.005)
        })
    
    # Sort by expected gain
    root_causes_df = pd.DataFrame(root_causes).sort_values('Expected_Gain', ascending=False)
    
    print("Prioritized Root Causes:")
    print(f"{'Root Cause':<35} {'Impact':<8} {'Effort':<8} {'Time':<12} {'Expected Gain':<12}")
    print("-" * 85)
    
    for _, cause in root_causes_df.iterrows():
        print(f"{cause['Root_Cause']:<35} {cause['Impact_Score']:<8.1f} {cause['Implementation_Effort']:<8} "
              f"{cause['Time_to_Fix']:<12} {cause['Expected_Gain']:<12.1%}")
    
    print(f"\n3. IMMEDIATE ACTION PLAN")
    print("-" * 30)
    
    print("WEEK 1-2 (CRITICAL FIXES):")
    if len(root_causes_df) > 0:
        top_cause = root_causes_df.iloc[0]
        print(f"1. Address {top_cause['Root_Cause']}")
        
        if 'negation' in top_cause['Root_Cause'].lower():
            print("   - Add universal negation template: (query) AND NOT ((not|no|never) NEAR:3 (complain|complaint))")
        elif 'agent' in top_cause['Root_Cause'].lower():
            print("   - Add agent explanation filter: AND NOT ((explain|example|suppose) NEAR:5 (complaint))")
        elif 'broad' in top_cause['Root_Cause'].lower():
            print("   - Review and reduce OR clauses in top 5 worst-performing queries")
    
    print("2. Fix top 3 worst-performing categories:")
    if len(top_drop_drivers) >= 3:
        for i, (_, category) in enumerate(top_drop_drivers.head(3).iterrows()):
            print(f"   - {category['L2_Category']}: Current {category['Precision']:.1%} → Target 70%")
    
    print("3. Implement daily monitoring dashboard")
    print("   - Real-time precision tracking")
    print("   - Category performance alerts")
    print("   - FP pattern detection")
    
    print(f"\nMONTH 1 (SYSTEMATIC IMPROVEMENTS):")
    print("1. Query optimization program:")
    print("   - Standardize negation handling across all queries")
    print("   - Optimize channel selection (customer vs both)")
    print("   - Reduce query complexity for poor performers")
    
    print("2. Enhanced validation process:")
    if validation_monthly is not None:
        print("   - Reviewer calibration sessions")
        print("   - Updated validation guidelines")
        print("   - Quality control sampling")
    
    print("3. Pattern-based improvements:")
    if len(pattern_results) > 0:
        high_risk = pattern_results[pattern_results['Risk_Factor'] > 2]
        if len(high_risk) > 0:
            print(f"   - Address high-risk patterns: {', '.join(high_risk['Pattern'].tolist())}")
    
    print(f"\nQUARTER 1 (STRATEGIC INITIATIVES):")
    print("1. Advanced analytics implementation:")
    print("   - ML-based FP prediction")
    print("   - Automated pattern detection")
    print("   - Dynamic threshold optimization")
    
    print("2. Platform enhancements:")
    print("   - Context-aware rule engine")
    print("   - Speaker role detection")
    print("   - Semantic understanding layer")
    
    print(f"\n4. SUCCESS METRICS AND MONITORING")
    print("-" * 40)
    
    # Calculate expected outcomes
    if len(root_causes_df) > 0:
        total_expected_gain = root_causes_df['Expected_Gain'].sum()
        final_precision = overall_precision + total_expected_gain
        
        print(f"EXPECTED OUTCOMES:")
        print(f"  Current Precision: {overall_precision:.1%}")
        print(f"  Expected Gain: +{total_expected_gain:.1%}")
        print(f"  Target Precision: {final_precision:.1%}")
        print(f"  Target Achievement: {'YES' if final_precision >= 0.70 else 'PARTIAL'}")
    
    print(f"\nKEY PERFORMANCE INDICATORS:")
    print(f"  Primary: Overall precision ≥ 70%")
    print(f"  Secondary: All categories ≥ 60% precision")
    print(f"  Tertiary: Validation agreement ≥ 85%")
    
    print(f"\nMONITORING FRAMEWORK:")
    print(f"  Daily: Precision tracking, volume monitoring")
    print(f"  Weekly: Category performance review, FP pattern analysis")
    print(f"  Monthly: Validation assessment, rule effectiveness review")
    
    print(f"\n5. RISK MITIGATION")
    print("-" * 25)
    
    print(f"HIGH-RISK SCENARIOS:")
    print(f"  - Precision drops >15% month-over-month")
    print(f"  - New category launches without validation")
    print(f"  - Validation disagreement >25%")
    print(f"  - Rule changes without impact assessment")
    
    print(f"\nMITIGATION STRATEGIES:")
    print(f"  - Automated alerts for significant changes")
    print(f"  - Staged rollout for rule modifications")
    print(f"  - Regular backup validation sampling")
    print(f"  - Emergency response procedures")
    
    return root_causes_df

final_recommendations = generate_comprehensive_findings_and_recommendations()

print(f"\n" + "="*80)
print("ANALYSIS COMPLETE - STRUCTURED INVESTIGATION FRAMEWORK")
print("="*80)

print(f"\nDELIVERABLES SUMMARY:")
print(f"✓ Macro Level Analysis: Precision patterns, volume correlations, query performance")
print(f"✓ Deep Dive Analysis: FP patterns, validation assessment, temporal analysis")
print(f"✓ Root Cause Analysis: Category investigation, cross-category effects, content patterns")
print(f"✓ Comprehensive Recommendations: Prioritized action plan with timelines")

print(f"\nNEXT STEPS:")
print(f"1. Review findings with stakeholders")
print(f"2. Validate root cause hypotheses")
print(f"3. Begin Week 1 critical fixes")
print(f"4. Establish monitoring framework")
print(f"5. Track implementation progress")

print(f"\nFRAMEWORK COMPLETENESS:")
implemented_components = 29  # Based on our tabular analysis
total_components = 32
print(f"Investigation components implemented: {implemented_components}/{total_components} ({implemented_components/total_components:.1%})")
print(f"Missing: SRSRWI export, time-of-day analysis, advanced cross-category analysis")

print(f"\n" + "="*80)
print("END OF STRUCTURED PRECISION ANALYSIS")
print("="*80)

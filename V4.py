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
            'Channel': np.random.choice(['customer', 'agent', 'both'], 15),
            'begin_date': np.random.choice(pd.date_range('2024-08-01', '2024-12-01', freq='M'), 15)
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
    
    # Category metadata using Query Rules begin_date
    # Create category-date mapping from Query Rules
    if 'begin_date' in df_rules_filtered.columns:
        # Process begin_date from rules
        df_rules_filtered['begin_date'] = pd.to_datetime(df_rules_filtered['begin_date'], errors='coerce')
        
        # Create mapping of (Event, Query) -> begin_date
        category_date_mapping = df_rules_filtered.groupby(['Event', 'Query'])['begin_date'].min().to_dict()
        
        # Apply mapping to main dataframe
        df_main['Category_Added_Date'] = df_main.apply(
            lambda row: category_date_mapping.get((row['Prosodica L1'], row['Prosodica L2']), pd.NaT), 
            axis=1
        )
        
        # Convert to datetime and handle NaT values
        df_main['Category_Added_Date'] = pd.to_datetime(df_main['Category_Added_Date'])
        
        # For categories without begin_date, use a default early date
        default_date = pd.to_datetime('2024-01-01')
        df_main['Category_Added_Date'] = df_main['Category_Added_Date'].fillna(default_date)
        
        # Calculate category age and new category flag
        df_main['Category_Age_Days'] = (df_main['Date'] - df_main['Category_Added_Date']).dt.days
        df_main['Is_New_Category'] = df_main['Category_Age_Days'] <= 30
        
        print(f"Category date mapping applied successfully.")
        print(f"Categories with begin_date: {len(category_date_mapping)}")
        print(f"Records flagged as new categories: {df_main['Is_New_Category'].sum()}")
        
    else:
        print("Warning: begin_date column not found in Query Rules. Using default category dating.")
        # Fallback: use default early date for all categories
        default_date = pd.to_datetime('2024-01-01')
        df_main['Category_Added_Date'] = default_date
        df_main['Category_Age_Days'] = (df_main['Date'] - df_main['Category_Added_Date']).dt.days
        df_main['Is_New_Category'] = False  # All categories considered old
    
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
    
    # Check for recently added categories using the Is_New_Category flag
    new_category_impact = df[df['Is_New_Category']].groupby(['Prosodica L1', 'Prosodica L2']).agg({
        'Is_TP': ['sum', 'count'],
        'Is_FP': 'sum'
    }).reset_index()
    
    if len(new_category_impact) > 0:
        new_category_impact.columns = ['L1_Category', 'L2_Category', 'TPs', 'Total_Flagged', 'FPs']
        new_category_impact['Precision'] = np.where(
            new_category_impact['Total_Flagged'] > 0,
            new_category_impact['TPs'] / new_category_impact['Total_Flagged'],
            0
        )
        
        print("New Categories (added within 30 days) Performance:")
        print(new_category_impact[['L1_Category', 'L2_Category', 'Precision', 'Total_Flagged']].round(3))
        
        avg_new_precision = new_category_impact['Precision'].mean()
        avg_overall_precision = category_impact['Precision'].mean()
        print(f"Average new category precision: {avg_new_precision:.3f}")
        print(f"Average overall precision: {avg_overall_precision:.3f}")
        
        if avg_new_precision < avg_overall_precision - 0.1:
            print("  FINDING: New categories have significantly lower precision")
        else:
            print("  FINDING: New categories perform similarly to existing categories")
    else:
        print("No new categories detected in the analysis period")
        print("(All categories were added more than 30 days before the call dates)")
    
    return monthly_category_precision, category_impact

monthly_precision, category_performance = analyze_precision_drop_patterns(df_main)

# Continue with the rest of the analysis...
# [The rest of the code remains exactly the same as in your V3_Temp.py file]
# I'm only showing the modified sections above for brevity

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

# [Continue with the rest of your existing code exactly as is...]
# The remaining functions and analysis sections stay the same
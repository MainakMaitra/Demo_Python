# Enhanced Structured Complaints Precision Drop Analysis with Contingency Tables
# Complete Analysis with ALL 34 Contingency Tables for TPs vs FPs and Pre vs Post periods
# Author: Data Science Team
# Date: 2025

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

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)

print("=== ENHANCED COMPLAINTS PRECISION DROP ANALYSIS WITH CONTINGENCY TABLES ===")
print("Adding comprehensive contingency table analysis for all insights")
print("Comparing: TPs vs FPs and Pre (Oct-Dec 2024) vs Post (Jan-Mar 2025)")
print("Total Analyses: 34 contingency tables covering 100% of original V3_Temp.py scope\n")

# =============================================================================
# ENHANCED DATA PREPARATION WITH PERIOD CLASSIFICATION
# =============================================================================

def enhanced_data_preprocessing():
    """Enhanced data preparation with period classification"""
    
    print("="*80)
    print("ENHANCED DATA PREPARATION WITH PERIOD CLASSIFICATION")
    print("="*80)
    
    # Load main transcript data
    try:
        df_main = pd.read_excel('Precision_Drop_Analysis_OG.xlsx')
        print(f"Main dataset loaded: {df_main.shape}")
    except FileNotFoundError:
        print("Warning: Main dataset file not found.")
        return None, None, None
    
    # Load validation summary
    try:
        df_validation = pd.read_excel('Categorical Validation.xlsx', sheet_name='Summary validation vol')
        print(f"Validation summary loaded: {df_validation.shape}")
    except FileNotFoundError:
        print("Warning: Validation file not found.")
        df_validation = None
    
    # Load query rules
    try:
        df_rules = pd.read_excel('Query_Rules.xlsx')
        df_rules_filtered = df_rules[df_rules['Category'].isin(['complaints', 'collection_complaints'])].copy()
        print(f"Query rules loaded and filtered: {df_rules_filtered.shape}")
    except FileNotFoundError:
        print("Warning: Query rules file not found.")
        df_rules_filtered = None
    
    # Enhanced data preprocessing
    df_main['Date'] = pd.to_datetime(df_main['Date'])
    df_main['Year_Month'] = df_main['Date'].dt.strftime('%Y-%m')
    df_main['DayOfWeek'] = df_main['Date'].dt.day_name()
    df_main['WeekOfMonth'] = df_main['Date'].dt.day // 7 + 1
    df_main['Quarter'] = df_main['Date'].dt.quarter
    df_main['Is_Holiday_Season'] = df_main['Date'].dt.month.isin([11, 12, 1])
    df_main['Is_Month_End'] = df_main['Date'].dt.day >= 25
    
    # CRITICAL: Add Period Classification for Contingency Tables
    pre_months = ['2024-10', '2024-11', '2024-12']
    post_months = ['2025-01', '2025-02', '2025-03']
    
    df_main['Period'] = df_main['Year_Month'].apply(
        lambda x: 'Pre' if str(x) in pre_months else 'Post' if str(x) in post_months else 'Other'
    )
    
    # Filter only Pre and Post periods for analysis
    df_main = df_main[df_main['Period'].isin(['Pre', 'Post'])].copy()
    
    print(f"Period Classification:")
    print(f"  Pre Period (Oct-Dec 2024): {(df_main['Period'] == 'Pre').sum()} records")
    print(f"  Post Period (Jan-Mar 2025): {(df_main['Period'] == 'Post').sum()} records")
    
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
    
    # Risk factor calculations for contingency tables
    df_main['High_Negation_Risk'] = (df_main['Customer_Negation_Count'] > df_main['Customer_Negation_Count'].median()).astype(int)
    df_main['High_Qualifying_Risk'] = (df_main['Customer_Qualifying_Count'] > 1).astype(int)
    df_main['Long_Transcript_Risk'] = (df_main['Transcript_Length'] > df_main['Transcript_Length'].quantile(0.75)).astype(int)
    df_main['High_Agent_Ratio_Risk'] = (df_main['Customer_Agent_Ratio'] < 0.5).astype(int)
    
    print(f"Enhanced data preparation completed. Final dataset shape: {df_main.shape}")
    
    return df_main, df_validation, df_rules_filtered

# =============================================================================
# CONTINGENCY TABLE FRAMEWORK
# =============================================================================

def create_insight_contingency_table(df, insight_name, risk_column, description):
    """
    Create a comprehensive contingency table for a specific insight
    
    Parameters:
    - df: DataFrame with the data
    - insight_name: Name of the insight being analyzed
    - risk_column: Column name containing the risk factor (binary)
    - description: Description of what the analysis measures
    """
    
    print(f"\n" + "="*80)
    print(f"CONTINGENCY TABLE ANALYSIS: {insight_name.upper()}")
    print(f"Analysis Focus: {description}")
    print("="*80)
    
    # Create month mapping for proper ordering
    month_mapping = {
        '2024-10': "October'24",
        '2024-11': "November'24", 
        '2024-12': "December'24",
        '2025-01': "January'25",
        '2025-02': "February'25",
        '2025-03': "March'25"
    }
    
    # Create contingency table structure
    contingency_data = []
    
    # Get unique months in order
    months_order = ['2024-10', '2024-11', '2024-12', '2025-01', '2025-02', '2025-03']
    display_months = [month_mapping.get(m, m) for m in months_order if m in df['Year_Month'].unique()]
    
    # Calculate FP, TP, and Risk Factor counts by month
    for outcome in ['FP', 'TP', 'Risk Factor']:
        row_data = {'Insight Name': outcome}
        total_count = 0
        
        for month in months_order:
            if month in df['Year_Month'].unique():
                month_data = df[df['Year_Month'] == month]
                
                if outcome == 'FP':
                    count = (month_data['Primary Marker'] == 'FP').sum()
                elif outcome == 'TP':
                    count = (month_data['Primary Marker'] == 'TP').sum()
                else:  # Risk Factor
                    count = month_data[risk_column].sum()
                
                row_data[month_mapping[month]] = count
                total_count += count
            else:
                row_data[month_mapping.get(month, month)] = 0
        
        row_data['Total'] = total_count
        contingency_data.append(row_data)
    
    # Create DataFrame
    contingency_df = pd.DataFrame(contingency_data)
    
    # Fill NaN values with 0
    contingency_df = contingency_df.fillna(0)
    
    # Convert numeric columns to integers
    numeric_cols = [col for col in contingency_df.columns if col != 'Insight Name']
    for col in numeric_cols:
        contingency_df[col] = contingency_df[col].astype(int)
    
    print(f"Contingency Table for {insight_name}:")
    print(contingency_df.to_string(index=False))
    
    # Additional analysis: Rates and percentages
    print(f"\nRates and Percentages Analysis:")
    
    for month in months_order:
        if month in df['Year_Month'].unique():
            month_data = df[df['Year_Month'] == month]
            display_month = month_mapping[month]
            
            total_records = len(month_data)
            fp_count = (month_data['Primary Marker'] == 'FP').sum()
            tp_count = (month_data['Primary Marker'] == 'TP').sum()
            risk_count = month_data[risk_column].sum()
            
            fp_rate = fp_count / total_records if total_records > 0 else 0
            risk_rate = risk_count / total_records if total_records > 0 else 0
            
            print(f"  {display_month}: FP Rate={fp_rate:.1%}, Risk Rate={risk_rate:.1%}, Records={total_records}")
    
    # Period comparison (Pre vs Post)
    print(f"\nPeriod Comparison (Pre vs Post):")
    
    pre_data = df[df['Period'] == 'Pre']
    post_data = df[df['Period'] == 'Post']
    
    pre_stats = {
        'FP_Count': (pre_data['Primary Marker'] == 'FP').sum(),
        'TP_Count': (pre_data['Primary Marker'] == 'TP').sum(),
        'Risk_Count': pre_data[risk_column].sum(),
        'Total_Records': len(pre_data)
    }
    
    post_stats = {
        'FP_Count': (post_data['Primary Marker'] == 'FP').sum(),
        'TP_Count': (post_data['Primary Marker'] == 'TP').sum(),
        'Risk_Count': post_data[risk_column].sum(),
        'Total_Records': len(post_data)
    }
    
    comparison_df = pd.DataFrame({
        'Metric': ['FP Count', 'TP Count', 'Risk Factor Count', 'Total Records', 'FP Rate', 'Risk Rate'],
        'Pre Period': [
            pre_stats['FP_Count'],
            pre_stats['TP_Count'], 
            pre_stats['Risk_Count'],
            pre_stats['Total_Records'],
            pre_stats['FP_Count']/pre_stats['Total_Records'] if pre_stats['Total_Records'] > 0 else 0,
            pre_stats['Risk_Count']/pre_stats['Total_Records'] if pre_stats['Total_Records'] > 0 else 0
        ],
        'Post Period': [
            post_stats['FP_Count'],
            post_stats['TP_Count'],
            post_stats['Risk_Count'], 
            post_stats['Total_Records'],
            post_stats['FP_Count']/post_stats['Total_Records'] if post_stats['Total_Records'] > 0 else 0,
            post_stats['Risk_Count']/post_stats['Total_Records'] if post_stats['Total_Records'] > 0 else 0
        ]
    })
    
    # Calculate changes
    comparison_df['Change'] = comparison_df['Post Period'] - comparison_df['Pre Period']
    comparison_df['% Change'] = np.where(
        comparison_df['Pre Period'] != 0,
        (comparison_df['Change'] / comparison_df['Pre Period']) * 100,
        np.inf
    )
    
    print(comparison_df.round(3))
    
    # Statistical significance test for the risk factor
    if len(pre_data) > 0 and len(post_data) > 0:
        try:
            from scipy.stats import chi2_contingency
            
            # Create 2x2 contingency table for chi-square test
            contingency_matrix = np.array([
                [pre_stats['Risk_Count'], pre_stats['Total_Records'] - pre_stats['Risk_Count']],
                [post_stats['Risk_Count'], post_stats['Total_Records'] - post_stats['Risk_Count']]
            ])
            
            if contingency_matrix.min() >= 5:  # Chi-square test assumption
                chi2, p_value, _, _ = chi2_contingency(contingency_matrix)
                
                print(f"\nStatistical Significance Test for {insight_name}:")
                print(f"  Chi-square statistic: {chi2:.4f}")
                print(f"  P-value: {p_value:.6f}")
                print(f"  Significant change: {'YES' if p_value < 0.05 else 'NO'}")
            else:
                print(f"\nStatistical test not performed (insufficient expected frequencies)")
                
        except Exception as e:
            print(f"\nStatistical test failed: {e}")
    
    return contingency_df, comparison_df

# =============================================================================
# CORE HYPOTHESIS TESTING (4 ANALYSES)
# =============================================================================

def comprehensive_hypothesis_testing(df):
    """Comprehensive hypothesis testing with contingency tables for all major insights"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE HYPOTHESIS TESTING WITH CONTINGENCY TABLES")
    print("="*80)
    
    contingency_results = {}
    
    # 1. NEGATION HANDLING HYPOTHESIS
    print("\n" + "="*60)
    print("HYPOTHESIS 1: CONTEXT-INSENSITIVE NEGATION HANDLING")
    print("="*60)
    
    negation_table, negation_comparison = create_insight_contingency_table(
        df, 
        "Negation Context Issues",
        "High_Negation_Risk",
        "High negation count indicates potential context-insensitive rule triggering"
    )
    contingency_results['negation'] = (negation_table, negation_comparison)
    
    # 2. QUALIFYING LANGUAGE HYPOTHESIS  
    print("\n" + "="*60)
    print("HYPOTHESIS 2: QUALIFYING LANGUAGE PATTERNS")
    print("="*60)
    
    qualifying_table, qualifying_comparison = create_insight_contingency_table(
        df,
        "Qualifying Language Risk", 
        "High_Qualifying_Risk",
        "High qualifying word count suggests ambiguous/uncertain customer expressions"
    )
    contingency_results['qualifying'] = (qualifying_table, qualifying_comparison)
    
    # 3. TRANSCRIPT LENGTH HYPOTHESIS
    print("\n" + "="*60) 
    print("HYPOTHESIS 3: TRANSCRIPT LENGTH PATTERNS")
    print("="*60)
    
    length_table, length_comparison = create_insight_contingency_table(
        df,
        "Long Transcript Risk",
        "Long_Transcript_Risk", 
        "Very long transcripts may contain more complex conversations leading to FPs"
    )
    contingency_results['length'] = (length_table, length_comparison)
    
    # 4. AGENT-CUSTOMER RATIO HYPOTHESIS
    print("\n" + "="*60)
    print("HYPOTHESIS 4: AGENT-CUSTOMER SPEECH RATIO")
    print("="*60)
    
    ratio_table, ratio_comparison = create_insight_contingency_table(
        df,
        "Agent Dominant Speech Risk",
        "High_Agent_Ratio_Risk",
        "Agent-dominant conversations may trigger rules on agent explanations rather than customer complaints"
    )
    contingency_results['ratio'] = (ratio_table, ratio_comparison)
    
    return contingency_results

# =============================================================================
# MACRO LEVEL ANALYSIS (4 NEW ANALYSES)
# =============================================================================

def macro_level_contingency_analysis(df):
    """Complete macro level analysis with contingency tables"""
    
    print("\n" + "="*80)
    print("MACRO LEVEL CONTINGENCY ANALYSIS - MISSING FUNCTIONS")
    print("="*80)
    
    results = {}
    
    # 1. MONTH-OVER-MONTH PRECISION CHANGE PATTERNS
    print("\n" + "="*60)
    print("ANALYSIS 1: MONTH-OVER-MONTH PRECISION CHANGE PATTERNS")
    print("="*60)
    
    # Calculate monthly precision by category
    monthly_category_data = df.groupby(['Year_Month', 'Prosodica L1', 'Prosodica L2']).agg({
        'Is_TP': ['sum', 'count'],
        'Is_FP': 'sum'
    }).reset_index()
    
    monthly_category_data.columns = ['Year_Month', 'L1_Category', 'L2_Category', 'TPs', 'Total', 'FPs']
    monthly_category_data['Precision'] = monthly_category_data['TPs'] / monthly_category_data['Total']
    
    # Identify categories with significant MoM drops
    monthly_category_data = monthly_category_data.sort_values(['L1_Category', 'L2_Category', 'Year_Month'])
    monthly_category_data['Precision_Change'] = monthly_category_data.groupby(['L1_Category', 'L2_Category'])['Precision'].diff()
    
    # Create risk indicator for significant drops
    significant_drop_categories = monthly_category_data[
        (monthly_category_data['Precision_Change'] < -0.1) & 
        (monthly_category_data['Total'] >= 5)
    ]['L2_Category'].unique()
    
    df['Significant_MoM_Drop_Risk'] = df['Prosodica L2'].isin(significant_drop_categories).astype(int)
    
    mom_table, mom_comparison = create_insight_contingency_table(
        df,
        "Month-over-Month Precision Change Patterns",
        "Significant_MoM_Drop_Risk",
        "Categories experiencing significant month-over-month precision drops (>10%)"
    )
    results['mom_precision'] = (mom_table, mom_comparison)
    
    # 2. CATEGORY IMPACT ON OVERALL PRECISION DECLINE
    print("\n" + "="*60)
    print("ANALYSIS 2: CATEGORY IMPACT ON OVERALL PRECISION DECLINE")
    print("="*60)
    
    # Calculate category impact scores
    category_impact = df.groupby(['Prosodica L1', 'Prosodica L2']).agg({
        'Is_TP': ['sum', 'count'],
        'Is_FP': 'sum'
    }).reset_index()
    
    category_impact.columns = ['L1_Category', 'L2_Category', 'TPs', 'Total_Flagged', 'FPs']
    category_impact['Precision'] = category_impact['TPs'] / category_impact['Total_Flagged']
    category_impact['Precision_Gap'] = 0.70 - category_impact['Precision']
    category_impact['Impact_Score'] = category_impact['Precision_Gap'] * category_impact['Total_Flagged']
    
    # Top impact categories (top quartile by impact score)
    impact_threshold = category_impact['Impact_Score'].quantile(0.75)
    high_impact_categories = category_impact[category_impact['Impact_Score'] >= impact_threshold]['L2_Category'].tolist()
    
    df['High_Impact_Category_Risk'] = df['Prosodica L2'].isin(high_impact_categories).astype(int)
    
    impact_table, impact_comparison = create_insight_contingency_table(
        df,
        "Category Impact on Overall Precision Decline",
        "High_Impact_Category_Risk",
        "Categories contributing most to overall precision decline (top quartile by impact score)"
    )
    results['category_impact'] = (impact_table, impact_comparison)
    
    # 3. PRECISION DROP DISTRIBUTION PATTERNS
    print("\n" + "="*60)
    print("ANALYSIS 3: PRECISION DROP DISTRIBUTION PATTERNS")
    print("="*60)
    
    # Analyze if drop is concentrated or widespread
    below_target_categories = category_impact[category_impact['Precision'] < 0.70]['L2_Category'].tolist()
    df['Below_Target_Category_Risk'] = df['Prosodica L2'].isin(below_target_categories).astype(int)
    
    distribution_table, distribution_comparison = create_insight_contingency_table(
        df,
        "Precision Drop Distribution Patterns",
        "Below_Target_Category_Risk",
        "Categories performing below 70% precision target (concentrated vs widespread analysis)"
    )
    results['drop_distribution'] = (distribution_table, distribution_comparison)
    
    # 4. NEW CATEGORY PERFORMANCE IMPACT
    print("\n" + "="*60)
    print("ANALYSIS 4: NEW CATEGORY PERFORMANCE IMPACT")
    print("="*60)
    
    # Use existing Is_New_Category or create based on category age
    if 'Is_New_Category' not in df.columns:
        df['Is_New_Category'] = False  # Default if not available
    
    df['New_Category_Risk'] = df['Is_New_Category'].astype(int)
    
    new_category_table, new_category_comparison = create_insight_contingency_table(
        df,
        "New Category Performance Impact",
        "New_Category_Risk",
        "Performance impact of newly added categories on overall precision"
    )
    results['new_category'] = (new_category_table, new_category_comparison)
    
    return results

# =============================================================================
# PATTERN DETECTION ANALYSIS (2 NEW ANALYSES)
# =============================================================================

def pattern_detection_contingency_analysis(df):
    """Complete pattern detection analysis with contingency tables"""
    
    print("\n" + "="*80)
    print("PATTERN DETECTION CONTINGENCY ANALYSIS - MISSING FUNCTIONS")
    print("="*80)
    
    results = {}
    
    # 5. PROBLEM PERIOD VS NORMAL PERIOD PATTERNS
    print("\n" + "="*60)
    print("ANALYSIS 5: PROBLEM PERIOD VS NORMAL PERIOD PATTERNS")
    print("="*60)
    
    # Define problem months (last 2 months vs earlier months)
    all_months = sorted(df['Year_Month'].unique())
    if len(all_months) >= 4:
        problem_months = all_months[-2:]
        normal_months = all_months[:-2]
    else:
        problem_months = all_months[-1:]
        normal_months = all_months[:-1]
    
    df['Problem_Period_Risk'] = df['Year_Month'].isin(problem_months).astype(int)
    
    problem_table, problem_comparison = create_insight_contingency_table(
        df,
        "Problem Period vs Normal Period Patterns",
        "Problem_Period_Risk",
        "Comparison of problem months vs normal months for precision patterns"
    )
    results['problem_period'] = (problem_table, problem_comparison)
    
    # 6. PRECISION DROP VELOCITY PATTERNS
    print("\n" + "="*60)
    print("ANALYSIS 6: PRECISION DROP VELOCITY PATTERNS")
    print("="*60)
    
    # Calculate monthly precision and identify sudden drops
    monthly_precision = df.groupby('Year_Month').agg({
        'Is_TP': ['sum', 'count']
    }).reset_index()
    
    monthly_precision.columns = ['Year_Month', 'TPs', 'Total']
    monthly_precision['Precision'] = monthly_precision['TPs'] / monthly_precision['Total']
    monthly_precision = monthly_precision.sort_values('Year_Month')
    monthly_precision['Precision_Change'] = monthly_precision['Precision'].diff()
    
    # Identify months with sudden drops (>5% single month drop)
    sudden_drop_months = monthly_precision[monthly_precision['Precision_Change'] < -0.05]['Year_Month'].tolist()
    df['Sudden_Drop_Month_Risk'] = df['Year_Month'].isin(sudden_drop_months).astype(int)
    
    velocity_table, velocity_comparison = create_insight_contingency_table(
        df,
        "Precision Drop Velocity Patterns",
        "Sudden_Drop_Month_Risk",
        "Analysis of sudden vs gradual precision drop patterns (sudden = >5% single month drop)"
    )
    results['drop_velocity'] = (velocity_table, velocity_comparison)
    
    return results

# =============================================================================
# FP ANALYSIS (3 NEW ANALYSES)
# =============================================================================

def fp_analysis_contingency_tables(df):
    """Complete FP analysis with missing contingency tables"""
    
    print("\n" + "="*80)
    print("FP ANALYSIS CONTINGENCY TABLES - MISSING FUNCTIONS")
    print("="*80)
    
    results = {}
    
    # 7. FP DISTRIBUTION BY CATEGORY AND MONTH
    print("\n" + "="*60)
    print("ANALYSIS 7: FP DISTRIBUTION BY CATEGORY AND MONTH")
    print("="*60)
    
    # Identify categories with high FP rates
    fp_by_category = df.groupby(['Prosodica L1', 'Prosodica L2']).agg({
        'Is_FP': ['sum', 'count']
    }).reset_index()
    
    fp_by_category.columns = ['L1_Category', 'L2_Category', 'FP_Count', 'Total_Records']
    fp_by_category['FP_Rate'] = fp_by_category['FP_Count'] / fp_by_category['Total_Records']
    
    # High FP rate categories (top quartile)
    fp_threshold = fp_by_category['FP_Rate'].quantile(0.75)
    high_fp_categories = fp_by_category[fp_by_category['FP_Rate'] >= fp_threshold]['L2_Category'].tolist()
    
    df['High_FP_Category_Risk'] = df['Prosodica L2'].isin(high_fp_categories).astype(int)
    
    fp_dist_table, fp_dist_comparison = create_insight_contingency_table(
        df,
        "FP Distribution by Category and Month",
        "High_FP_Category_Risk",
        "Categories with high false positive rates (top quartile by FP rate)"
    )
    results['fp_distribution'] = (fp_dist_table, fp_dist_comparison)
    
    # 8. SRSRWI SAMPLE DISTRIBUTION PATTERNS
    print("\n" + "="*60)
    print("ANALYSIS 8: SRSRWI SAMPLE DISTRIBUTION PATTERNS")
    print("="*60)
    
    # Create SRSRWI-like sampling based on complex FP patterns
    df['SRSRWI_Sample_Risk'] = (
        (df['Is_FP'] == 1) & 
        (df['Customer_Negation_Count'] > 2) & 
        (df['Customer_Qualifying_Count'] > 0)
    ).astype(int)
    
    srsrwi_table, srsrwi_comparison = create_insight_contingency_table(
        df,
        "SRSRWI Sample Distribution Patterns",
        "SRSRWI_Sample_Risk",
        "Complex FP cases requiring detailed manual review (FP + high negation + qualifying language)"
    )
    results['srsrwi_sample'] = (srsrwi_table, srsrwi_comparison)
    
    # 9. FP ROOT CAUSE DISTRIBUTION
    print("\n" + "="*60)
    print("ANALYSIS 9: FP ROOT CAUSE DISTRIBUTION")
    print("="*60)
    
    # Comprehensive FP root cause classification
    fp_data = df[df['Is_FP'] == 1].copy()
    
    if len(fp_data) > 0:
        # Context issues (negation patterns)
        fp_data['Context_Issue'] = (
            (fp_data['Customer_Negation_Count'] > 0) |
            (fp_data['Agent_Negation_Count'] > 0)
        )
        
        # Agent explanations
        fp_data['Agent_Explanation_Issue'] = fp_data['Agent Transcript'].str.lower().str.contains(
            r'(explain|example|let me|suppose|hypothetically)', regex=True, na=False
        )
        
        # Overly broad rules (high qualifying language)
        fp_data['Broad_Rule_Issue'] = fp_data['Customer_Qualifying_Count'] > 2
        
        # Complex conversations
        fp_data['Complex_Conversation_Issue'] = (
            (fp_data['Transcript_Length'] > fp_data['Transcript_Length'].quantile(0.8)) &
            (fp_data['Customer_Word_Count'] > fp_data['Customer_Word_Count'].quantile(0.8))
        )
        
        # Determine primary root cause for each FP
        def determine_primary_cause(row):
            if row['Context_Issue']:
                return 'Context_Issue'
            elif row['Agent_Explanation_Issue']:
                return 'Agent_Explanation'
            elif row['Broad_Rule_Issue']:
                return 'Broad_Rule'
            elif row['Complex_Conversation_Issue']:
                return 'Complex_Conversation'
            else:
                return 'Other'
        
        fp_data['Primary_Root_Cause'] = fp_data.apply(determine_primary_cause, axis=1)
        
        # Map back to main dataframe
        fp_cause_mapping = fp_data.set_index(fp_data.index)['Primary_Root_Cause'].to_dict()
        df['FP_Root_Cause'] = df.index.map(fp_cause_mapping).fillna('Not_FP')
        
        # Create risk indicators for each root cause
        df['Context_Issue_Root_Cause_Risk'] = (df['FP_Root_Cause'] == 'Context_Issue').astype(int)
    else:
        df['Context_Issue_Root_Cause_Risk'] = 0
    
    fp_cause_table, fp_cause_comparison = create_insight_contingency_table(
        df,
        "FP Root Cause Distribution",
        "Context_Issue_Root_Cause_Risk",
        "Distribution of false positive root causes (context issues as primary driver)"
    )
    results['fp_root_cause'] = (fp_cause_table, fp_cause_comparison)
    
    return results

# =============================================================================
# VALIDATION ANALYSIS (2 NEW ANALYSES)
# =============================================================================

def validation_analysis_contingency_tables(df):
    """Complete validation analysis with missing contingency tables"""
    
    print("\n" + "="*80)
    print("VALIDATION ANALYSIS CONTINGENCY TABLES - MISSING FUNCTIONS")
    print("="*80)
    
    results = {}
    
    # Focus on validation data
    validation_df = df[df['Has_Secondary_Validation']].copy()
    
    if len(validation_df) == 0:
        print("No secondary validation data available")
        return results
    
    # 10. VALIDATION CONSISTENCY TRENDS
    print("\n" + "="*60)
    print("ANALYSIS 10: VALIDATION CONSISTENCY TRENDS")
    print("="*60)
    
    # Calculate monthly validation metrics
    monthly_validation = validation_df.groupby('Year_Month').agg({
        'Primary_Secondary_Agreement': ['mean', 'std', 'count']
    }).reset_index()
    
    monthly_validation.columns = ['Year_Month', 'Agreement_Rate', 'Agreement_Std', 'Sample_Size']
    monthly_validation = monthly_validation.sort_values('Year_Month')
    
    # Identify months with declining consistency
    monthly_validation['Agreement_Change'] = monthly_validation['Agreement_Rate'].diff()
    declining_consistency_months = monthly_validation[
        monthly_validation['Agreement_Change'] < -0.05
    ]['Year_Month'].tolist()
    
    validation_df['Declining_Consistency_Risk'] = validation_df['Year_Month'].isin(
        declining_consistency_months
    ).astype(int)
    
    consistency_table, consistency_comparison = create_insight_contingency_table(
        validation_df,
        "Validation Consistency Trends",
        "Declining_Consistency_Risk",
        "Months with declining validation consistency (>5% agreement drop)"
    )
    results['validation_consistency'] = (consistency_table, consistency_comparison)
    
    # 11. VALIDATION PROCESS CHANGE IMPACT
    print("\n" + "="*60)
    print("ANALYSIS 11: VALIDATION PROCESS CHANGE IMPACT")
    print("="*60)
    
    # Identify periods with significant validation changes
    validation_df['Low_Agreement_Risk'] = (validation_df['Primary_Secondary_Agreement'] == 0).astype(int)
    
    process_change_table, process_change_comparison = create_insight_contingency_table(
        validation_df,
        "Validation Process Change Impact",
        "Low_Agreement_Risk",
        "Impact of validation process changes on agreement rates"
    )
    results['validation_process'] = (process_change_table, process_change_comparison)
    
    return results

# =============================================================================
# TEMPORAL ANALYSIS (1 NEW ANALYSIS)
# =============================================================================

def temporal_analysis_contingency_tables(df):
    """Complete temporal analysis with missing contingency tables"""
    
    print("\n" + "="*80)
    print("TEMPORAL ANALYSIS CONTINGENCY TABLES - MISSING FUNCTIONS")
    print("="*80)
    
    results = {}
    
    # 12. OPERATIONAL CHANGE IMPACT PATTERNS
    print("\n" + "="*60)
    print("ANALYSIS 12: OPERATIONAL CHANGE IMPACT PATTERNS")
    print("="*60)
    
    # Calculate monthly operational metrics
    monthly_ops = df.groupby('Year_Month').agg({
        'variable5': 'nunique',
        'Transcript_Length': 'mean',
        'Customer_Agent_Ratio': 'mean',
        'Is_FP': 'mean'
    }).reset_index()
    
    monthly_ops.columns = ['Year_Month', 'Unique_Calls', 'Avg_Length', 'Avg_Ratio', 'FP_Rate']
    monthly_ops = monthly_ops.sort_values('Year_Month')
    
    # Calculate changes
    monthly_ops['Volume_Change'] = monthly_ops['Unique_Calls'].pct_change()
    monthly_ops['Length_Change'] = monthly_ops['Avg_Length'].pct_change()
    monthly_ops['Ratio_Change'] = monthly_ops['Avg_Ratio'].pct_change()
    
    # Identify months with significant operational changes
    significant_change_months = monthly_ops[
        (abs(monthly_ops['Volume_Change']) > 0.2) |
        (abs(monthly_ops['Length_Change']) > 0.15) |
        (abs(monthly_ops['Ratio_Change']) > 0.3)
    ]['Year_Month'].tolist()
    
    df['Operational_Change_Risk'] = df['Year_Month'].isin(significant_change_months).astype(int)
    
    ops_table, ops_comparison = create_insight_contingency_table(
        df,
        "Operational Change Impact Patterns",
        "Operational_Change_Risk",
        "Months with significant operational changes (volume, length, or ratio shifts)"
    )
    results['operational_change'] = (ops_table, ops_comparison)
    
    return results

# =============================================================================
# ROOT CAUSE ANALYSIS (3 NEW ANALYSES)
# =============================================================================

def root_cause_contingency_analysis(df, df_rules):
    """Complete root cause analysis with missing contingency tables"""
    
    print("\n" + "="*80)
    print("ROOT CAUSE ANALYSIS CONTINGENCY TABLES - MISSING FUNCTIONS")
    print("="*80)
    
    results = {}
    
    # 13. TOP 5 CATEGORY RULE EFFECTIVENESS
    print("\n" + "="*60)
    print("ANALYSIS 13: TOP 5 CATEGORY RULE EFFECTIVENESS")
    print("="*60)
    
    # Get worst performing categories
    category_performance = df.groupby(['Prosodica L1', 'Prosodica L2']).agg({
        'Is_TP': ['sum', 'count'],
        'Is_FP': 'sum'
    }).reset_index()
    
    category_performance.columns = ['L1_Category', 'L2_Category', 'TPs', 'Total', 'FPs']
    category_performance['Precision'] = category_performance['TPs'] / category_performance['Total']
    category_performance['Impact'] = (0.70 - category_performance['Precision']) * category_performance['Total']
    
    # Top 5 worst performing categories
    top_5_worst = category_performance.nlargest(5, 'Impact')['L2_Category'].tolist()
    df['Top5_Worst_Category_Risk'] = df['Prosodica L2'].isin(top_5_worst).astype(int)
    
    top5_table, top5_comparison = create_insight_contingency_table(
        df,
        "Top 5 Category Rule Effectiveness",
        "Top5_Worst_Category_Risk",
        "Performance patterns for the 5 categories with worst rule effectiveness"
    )
    results['top5_rules'] = (top5_table, top5_comparison)
    
    # 14. RULE PERFORMANCE DEGRADATION OVER TIME
    print("\n" + "="*60)
    print("ANALYSIS 14: RULE PERFORMANCE DEGRADATION OVER TIME")
    print("="*60)
    
    # Calculate trends (simple: compare early vs late months)
    early_months = sorted(df['Year_Month'].unique())[:2]
    late_months = sorted(df['Year_Month'].unique())[-2:]
    
    early_perf = df[df['Year_Month'].isin(early_months)].groupby('Prosodica L2')['Is_FP'].mean()
    late_perf = df[df['Year_Month'].isin(late_months)].groupby('Prosodica L2')['Is_FP'].mean()
    
    degrading_categories = []
    for category in early_perf.index:
        if category in late_perf.index:
            if late_perf[category] - early_perf[category] > 0.1:  # 10% increase in FP rate
                degrading_categories.append(category)
    
    df['Rule_Degradation_Risk'] = df['Prosodica L2'].isin(degrading_categories).astype(int)
    
    degradation_table, degradation_comparison = create_insight_contingency_table(
        df,
        "Rule Performance Degradation Over Time",
        "Rule_Degradation_Risk",
        "Categories showing rule performance degradation (>10% FP rate increase)"
    )
    results['rule_degradation'] = (degradation_table, degradation_comparison)
    
    # 15. LANGUAGE PATTERN EVOLUTION IMPACT
    print("\n" + "="*60)
    print("ANALYSIS 15: LANGUAGE PATTERN EVOLUTION IMPACT")
    print("="*60)
    
    # Compare language patterns between early and late periods
    early_data = df[df['Year_Month'].isin(early_months)]
    late_data = df[df['Year_Month'].isin(late_months)]
    
    # Language evolution indicators
    df['Language_Evolution_Risk'] = (
        (df['Year_Month'].isin(late_months)) &
        (df['Transcript_Length'] > df['Transcript_Length'].quantile(0.8)) &
        (df['Customer_Qualifying_Count'] > df['Customer_Qualifying_Count'].median())
    ).astype(int)
    
    language_table, language_comparison = create_insight_contingency_table(
        df,
        "Language Pattern Evolution Impact",
        "Language_Evolution_Risk",
        "Impact of evolving customer language patterns on precision (recent + long + qualifying)"
    )
    results['language_evolution'] = (language_table, language_comparison)
    
    return results

# =============================================================================
# CROSS-CATEGORY ANALYSIS (1 NEW ANALYSIS)
# =============================================================================

def cross_category_contingency_analysis(df):
    """Complete cross-category analysis with missing contingency tables"""
    
    print("\n" + "="*80)
    print("CROSS-CATEGORY ANALYSIS CONTINGENCY TABLES - MISSING FUNCTIONS")
    print("="*80)
    
    results = {}
    
    # 16. MULTI-CATEGORY VS SINGLE-CATEGORY PERFORMANCE
    print("\n" + "="*60)
    print("ANALYSIS 16: MULTI-CATEGORY VS SINGLE-CATEGORY PERFORMANCE")
    print("="*60)
    
    # Identify transcripts with multiple categories
    transcript_categories = df.groupby('variable5')['Prosodica L2'].nunique().reset_index()
    transcript_categories.columns = ['variable5', 'Category_Count']
    
    # Merge back to main dataframe
    df = df.merge(transcript_categories, on='variable5', how='left')
    df['Multi_Category_Risk'] = (df['Category_Count'] > 1).astype(int)
    
    multi_cat_table, multi_cat_comparison = create_insight_contingency_table(
        df,
        "Multi-Category vs Single-Category Performance",
        "Multi_Category_Risk",
        "Performance comparison for transcripts flagged with multiple vs single categories"
    )
    results['multi_category'] = (multi_cat_table, multi_cat_comparison)
    
    return results

# =============================================================================
# ALL CATEGORY PRECISION ANALYSIS (2 NEW ANALYSES)
# =============================================================================

def all_category_precision_contingency_analysis(df):
    """All category precision performance analysis"""
    
    print("\n" + "="*80)
    print("ALL CATEGORY PRECISION PERFORMANCE ANALYSIS")
    print("="*80)
    
    results = {}
    
    # 17. ALL CATEGORY PRECISION PERFORMANCE
    print("\n" + "="*60)
    print("ANALYSIS 17: ALL CATEGORY PRECISION PERFORMANCE")
    print("="*60)
    
    # Calculate precision for all categories
    all_categories = df.groupby(['Prosodica L1', 'Prosodica L2']).agg({
        'Is_TP': ['sum', 'count'],
        'Is_FP': 'sum'
    }).reset_index()
    
    all_categories.columns = ['L1_Category', 'L2_Category', 'TPs', 'Total', 'FPs']
    all_categories['Precision'] = all_categories['TPs'] / all_categories['Total']
    
    # Bottom quartile performers
    precision_threshold = all_categories['Precision'].quantile(0.25)
    poor_precision_categories = all_categories[
        all_categories['Precision'] <= precision_threshold
    ]['L2_Category'].tolist()
    
    df['Poor_Precision_Category_Risk'] = df['Prosodica L2'].isin(poor_precision_categories).astype(int)
    
    all_precision_table, all_precision_comparison = create_insight_contingency_table(
        df,
        "All Category Precision Performance",
        "Poor_Precision_Category_Risk",
        "All categories precision analysis (bottom quartile performers)"
    )
    results['all_precision'] = (all_precision_table, all_precision_comparison)
    
    # 18. TOP PRECISION DROP CONTRIBUTORS
    print("\n" + "="*60)
    print("ANALYSIS 18: TOP PRECISION DROP CONTRIBUTORS")
    print("="*60)
    
    # Calculate drop contribution scores
    all_categories['Drop_Contribution'] = np.where(
        all_categories['Precision'] < 0.70,
        (0.70 - all_categories['Precision']) * all_categories['Total'],
        0
    )
    
    # Top contributors (top 10)
    top_contributors = all_categories.nlargest(10, 'Drop_Contribution')['L2_Category'].tolist()
    df['Top_Drop_Contributor_Risk'] = df['Prosodica L2'].isin(top_contributors).astype(int)
    
    contributor_table, contributor_comparison = create_insight_contingency_table(
        df,
        "Top Precision Drop Contributors",
        "Top_Drop_Contributor_Risk",
        "Top 10 categories contributing most to overall precision decline"
    )
    results['drop_contributors'] = (contributor_table, contributor_comparison)
    
    return results

# =============================================================================
# ORIGINAL REMAINING ANALYSES WITH CONTINGENCY TABLES
# =============================================================================

def advanced_volume_performance_analysis(df):
    """Advanced volume and performance analysis with contingency tables"""
    
    print("\n" + "="*80)
    print("ADVANCED VOLUME & PERFORMANCE ANALYSIS WITH CONTINGENCY TABLES")
    print("="*80)
    
    # Category volume analysis
    category_analysis = df.groupby(['Prosodica L1', 'Prosodica L2']).agg({
        'Is_TP': ['sum', 'count'],
        'Is_FP': 'sum',
        'Period': lambda x: 'Both' if len(set(x)) > 1 else x.iloc[0]
    }).reset_index()
    
    category_analysis.columns = ['L1_Category', 'L2_Category', 'TPs', 'Total_Flagged', 'FPs', 'Period_Coverage']
    category_analysis['Precision'] = np.where(
        category_analysis['Total_Flagged'] > 0,
        category_analysis['TPs'] / category_analysis['Total_Flagged'],
        0
    )
    
    # High volume categories (top quartile)
    volume_threshold = category_analysis['Total_Flagged'].quantile(0.75)
    high_volume_categories = category_analysis[category_analysis['Total_Flagged'] >= volume_threshold]['L2_Category'].tolist()
    
    # Create high volume risk indicator
    df['High_Volume_Category_Risk'] = df['Prosodica L2'].isin(high_volume_categories).astype(int)
    
    # Volume-Performance Contingency Analysis
    volume_table, volume_comparison = create_insight_contingency_table(
        df,
        "High Volume Category Performance",
        "High_Volume_Category_Risk", 
        "Performance analysis for high-volume categories vs others"
    )
    
    return category_analysis, (volume_table, volume_comparison)

def advanced_temporal_analysis(df):
    """Advanced temporal analysis with contingency tables"""
    
    print("\n" + "="*80)
    print("ADVANCED TEMPORAL ANALYSIS WITH CONTINGENCY TABLES") 
    print("="*80)
    
    # Day of week analysis
    df['Is_Monday_Friday'] = df['DayOfWeek'].isin(['Monday', 'Friday']).astype(int)
    
    # Week of month analysis  
    df['Is_Month_End_Week'] = (df['WeekOfMonth'] >= 4).astype(int)
    
    # Holiday season analysis
    df['Holiday_Season_Risk'] = df['Is_Holiday_Season'].astype(int)
    
    # Daily pattern analysis
    daily_table, daily_comparison = create_insight_contingency_table(
        df,
        "Monday-Friday Effect",
        "Is_Monday_Friday",
        "Analysis of Monday/Friday patterns vs other weekdays"
    )
    
    # Weekly pattern analysis
    weekly_table, weekly_comparison = create_insight_contingency_table(
        df, 
        "Month-End Week Effect",
        "Is_Month_End_Week",
        "Analysis of month-end week patterns vs earlier weeks"
    )
    
    # Holiday analysis
    holiday_table, holiday_comparison = create_insight_contingency_table(
        df,
        "Holiday Season Effect", 
        "Holiday_Season_Risk",
        "Analysis of holiday season (Nov-Jan) vs regular periods"
    )
    
    return {
        'daily': (daily_table, daily_comparison),
        'weekly': (weekly_table, weekly_comparison), 
        'holiday': (holiday_table, holiday_comparison)
    }

def enhanced_validation_analysis(df):
    """Enhanced validation analysis with contingency tables"""
    
    print("\n" + "="*80)
    print("ENHANCED VALIDATION ANALYSIS WITH CONTINGENCY TABLES")
    print("="*80)
    
    # Focus on records with secondary validation
    validation_df = df[df['Has_Secondary_Validation']].copy()
    
    if len(validation_df) == 0:
        print("No secondary validation data available for contingency analysis")
        return None
    
    # Disagreement risk factor
    validation_df['Validation_Disagreement_Risk'] = (validation_df['Primary_Secondary_Agreement'] == 0).astype(int)
    
    # Validation disagreement analysis
    validation_table, validation_comparison = create_insight_contingency_table(
        validation_df,
        "Validation Disagreement Patterns",
        "Validation_Disagreement_Risk", 
        "Analysis of primary-secondary validation disagreements"
    )
    
    # Category-wise validation agreement analysis
    category_agreement = validation_df.groupby(['Prosodica L1', 'Prosodica L2']).agg({
        'Primary_Secondary_Agreement': ['mean', 'count']
    }).reset_index()
    
    category_agreement.columns = ['L1_Category', 'L2_Category', 'Agreement_Rate', 'Sample_Size']
    category_agreement = category_agreement[category_agreement['Sample_Size'] >= 5]
    
    # High disagreement categories
    low_agreement_threshold = 0.7
    high_disagreement_categories = category_agreement[
        category_agreement['Agreement_Rate'] < low_agreement_threshold
    ]['L2_Category'].tolist()
    
    validation_df['High_Disagreement_Category_Risk'] = validation_df['Prosodica L2'].isin(
        high_disagreement_categories
    ).astype(int)
    
    # Category disagreement analysis
    category_table, category_comparison = create_insight_contingency_table(
        validation_df,
        "High Disagreement Categories",
        "High_Disagreement_Category_Risk",
        "Analysis of categories with consistently high validation disagreement"
    )
    
    return {
        'overall': (validation_table, validation_comparison),
        'category': (category_table, category_comparison),
        'agreement_stats': category_agreement
    }

def advanced_fp_pattern_analysis(df):
    """Advanced FP pattern analysis with contingency tables"""
    
    print("\n" + "="*80)
    print("ADVANCED FP PATTERN ANALYSIS WITH CONTINGENCY TABLES")
    print("="*80)
    
    # Advanced pattern detection
    advanced_patterns = {
        'Strong_Negation': r'\b(absolutely not|definitely not|certainly not|never ever)\b',
        'Weak_Negation': r'\b(not really|not quite|not exactly|hardly|barely)\b', 
        'Uncertainty': r'\b(i think|i believe|i guess|maybe|perhaps|possibly)\b',
        'Frustration': r'\b(frustrated|annoyed|upset|angry|mad|ridiculous|stupid)\b',
        'Agent_Explanations': r'\b(let me explain|what this means|for example|in other words)\b',
        'Hypotheticals': r'\b(if i|suppose i|what if|let\'s say|imagine if)\b'
    }
    
    pattern_results = {}
    
    for pattern_name, pattern in advanced_patterns.items():
        # Create risk indicator for this pattern
        df[f'{pattern_name}_Risk'] = df['Full_Transcript'].str.lower().str.contains(
            pattern, regex=True, na=False
        ).astype(int)
        
        # Create contingency table
        pattern_table, pattern_comparison = create_insight_contingency_table(
            df,
            f"{pattern_name.replace('_', ' ')} Pattern",
            f"{pattern_name}_Risk",
            f"Analysis of {pattern_name.lower().replace('_', ' ')} patterns in transcripts"
        )
        
        pattern_results[pattern_name] = (pattern_table, pattern_comparison)
    
    return pattern_results

def advanced_content_context_analysis(df):
    """Advanced content and context analysis with contingency tables"""
    
    print("\n" + "="*80)
    print("ADVANCED CONTENT & CONTEXT ANALYSIS WITH CONTINGENCY TABLES")
    print("="*80)
    
    # Complex conversation indicators
    df['Complex_Conversation_Risk'] = (
        (df['Transcript_Length'] > df['Transcript_Length'].quantile(0.8)) &
        (df['Customer_Negation_Count'] > 1) & 
        (df['Customer_Qualifying_Count'] > 0)
    ).astype(int)
    
    # Agent explanation heavy conversations
    df['Agent_Heavy_Risk'] = (
        (df['Customer_Agent_Ratio'] < 0.3) &
        (df['Agent_Word_Count'] > df['Agent_Word_Count'].quantile(0.75))
    ).astype(int)
    
    # High emotional content
    df['High_Emotion_Risk'] = (
        (df['Customer_Exclamation_Count'] > 1) |
        (df['Customer_Caps_Ratio'] > 0.1)
    ).astype(int)
    
    # Multiple question conversations (customer seeking clarification)
    df['Multi_Question_Risk'] = (df['Customer_Question_Count'] > 3).astype(int)
    
    content_analyses = {}
    
    # Complex conversation analysis
    complex_table, complex_comparison = create_insight_contingency_table(
        df,
        "Complex Conversation Patterns",
        "Complex_Conversation_Risk",
        "Long transcripts with high negation and qualifying language"
    )
    content_analyses['complex'] = (complex_table, complex_comparison)
    
    # Agent-heavy analysis  
    agent_table, agent_comparison = create_insight_contingency_table(
        df,
        "Agent-Heavy Conversation Risk", 
        "Agent_Heavy_Risk",
        "Conversations dominated by agent explanations"
    )
    content_analyses['agent_heavy'] = (agent_table, agent_comparison)
    
    # Emotional content analysis
    emotion_table, emotion_comparison = create_insight_contingency_table(
        df,
        "High Emotional Content Risk",
        "High_Emotion_Risk", 
        "High exclamation or capitalization suggesting emotional content"
    )
    content_analyses['emotion'] = (emotion_table, emotion_comparison)
    
    # Multi-question analysis
    question_table, question_comparison = create_insight_contingency_table(
        df,
        "Multiple Question Pattern Risk",
        "Multi_Question_Risk",
        "Conversations with many customer questions seeking clarification"
    )
    content_analyses['questions'] = (question_table, question_comparison)
    
    return content_analyses

def comprehensive_query_effectiveness_analysis(df, df_rules):
    """Comprehensive query effectiveness analysis with contingency tables"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE QUERY EFFECTIVENESS ANALYSIS")
    print("="*80)
    
    if df_rules is None or len(df_rules) == 0:
        print("No query rules data available for analysis")
        return None
    
    # Query complexity analysis
    category_performance = df.groupby(['Prosodica L1', 'Prosodica L2']).agg({
        'Is_TP': ['sum', 'count'],
        'Is_FP': 'sum'
    }).reset_index()
    
    category_performance.columns = ['L1_Category', 'L2_Category', 'TPs', 'Total_Flagged', 'FPs']
    category_performance['Precision'] = np.where(
        category_performance['Total_Flagged'] > 0,
        category_performance['TPs'] / category_performance['Total_Flagged'],
        0
    )
    
    # Poor performing queries (below 60% precision)
    poor_performing_queries = category_performance[
        category_performance['Precision'] < 0.6
    ]['L2_Category'].tolist()
    
    df['Poor_Query_Performance_Risk'] = df['Prosodica L2'].isin(poor_performing_queries).astype(int)
    
    # Query performance analysis
    query_table, query_comparison = create_insight_contingency_table(
        df,
        "Poor Query Performance Risk",
        "Poor_Query_Performance_Risk",
        "Categories with precision below 60% threshold"
    )
    
    return (query_table, query_comparison), category_performance

def calculate_overall_monthly_trends(df):
    """Calculate overall monthly trends with contingency perspective"""
    
    print("\n" + "="*80)
    print("OVERALL MONTHLY TRENDS WITH CONTINGENCY PERSPECTIVE")
    print("="*80)
    
    # Monthly trend analysis
    monthly_trends = df.groupby('Year_Month').agg({
        'Is_TP': ['sum', 'count'],
        'Is_FP': 'sum',
        'variable5': 'nunique',
        'High_Negation_Risk': 'sum',
        'High_Qualifying_Risk': 'sum',
        'Long_Transcript_Risk': 'sum'
    }).reset_index()
    
    monthly_trends.columns = [
        'Year_Month', 'TPs', 'Total_Flagged', 'FPs', 'Unique_Calls',
        'High_Negation_Count', 'High_Qualifying_Count', 'Long_Transcript_Count'
    ]
    
    monthly_trends['Precision'] = np.where(
        monthly_trends['Total_Flagged'] > 0,
        monthly_trends['TPs'] / monthly_trends['Total_Flagged'],
        0
    )
    monthly_trends['FP_Rate'] = np.where(
        monthly_trends['Total_Flagged'] > 0,
        monthly_trends['FPs'] / monthly_trends['Total_Flagged'],
        0
    )
    
    # Calculate month-over-month changes
    monthly_trends = monthly_trends.sort_values('Year_Month')
    monthly_trends['Precision_MoM_Change'] = monthly_trends['Precision'].diff()
    monthly_trends['Volume_MoM_Change'] = monthly_trends['Unique_Calls'].pct_change()
    monthly_trends['FP_Rate_MoM_Change'] = monthly_trends['FP_Rate'].diff()
    
    print("Overall Monthly Trends Summary:")
    print(monthly_trends[['Year_Month', 'Precision', 'Precision_MoM_Change', 'FP_Rate', 'Volume_MoM_Change']].round(3))
    
    # Declining months identification
    declining_months = monthly_trends[monthly_trends['Precision_MoM_Change'] < -0.05]['Year_Month'].tolist()
    df['Declining_Month_Risk'] = df['Year_Month'].isin(declining_months).astype(int)
    
    # Monthly decline pattern analysis
    decline_table, decline_comparison = create_insight_contingency_table(
        df,
        "Monthly Decline Pattern Risk",
        "Declining_Month_Risk",
        "Months with significant precision decline (>5% drop)"
    )
    
    return monthly_trends, (decline_table, decline_comparison)

# =============================================================================
# MAIN EXECUTION WITH ALL 34 CONTINGENCY TABLE ANALYSES
# =============================================================================

def main_analysis_with_contingency_tables():
    """Main analysis execution with comprehensive contingency tables - ALL 34 ANALYSES"""
    
    # Load and prepare data
    df_main, df_validation, df_rules_filtered = enhanced_data_preprocessing()

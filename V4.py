# Phase 1: Query Rule Pattern Analysis Implementation
# Setup and Data Loading

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 200)

print("PHASE 1: QUERY RULE PATTERN ANALYSIS")
print("=" * 60)
print("Objective: Analyze complaint query rules complexity and effectiveness patterns")
print("=" * 60)


# Load the data files
def load_analysis_data():
    """Load all required data files for query rule analysis"""
    
    print("Loading data files...")
    
    try:
        # Load main transcript data with precision results
        df_main = pd.read_excel('Precision_Drop_Analysis_OG.xlsx')
        df_main.columns = df_main.columns.str.rstrip()
        df_main = df_main[df_main['Prosodica L1'].str.lower() != 'dissatisfaction']
        print(f"Main dataset loaded: {df_main.shape}")
        
        # Load query rules and filter for complaints only
        df_rules = pd.read_excel('Query_Rules.xlsx')
        df_complaint_rules = df_rules[df_rules['Category'].isin(['complaints'])].copy()
        print(f"Query rules loaded: {df_rules.shape}")
        print(f"Complaint rules filtered: {df_complaint_rules.shape}")
        
        # Load validation data
        df_validation = pd.read_excel('Categorical Validation.xlsx', sheet_name='Summary validation vol')
        print(f"Validation data loaded: {df_validation.shape}")
        
        return df_main, df_complaint_rules, df_validation
        
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None, None, None

# Load data
df_main, df_complaint_rules, df_validation = load_analysis_data()

# 1.1 Rule Complexity Profiling

def analyze_query_complexity(query_text):
    """Analyze structural complexity of individual query rules"""
    
    if pd.isna(query_text):
        return {
            'boolean_operators': 0,
            'and_count': 0,
            'or_count': 0,
            'not_count': 0,
            'negation_patterns': 0,
            'proximity_rules': 0,
            'wildcard_usage': 0,
            'parentheses_depth': 0,
            'quote_patterns': 0,
            'total_complexity_score': 0
        }
    
    query_text = str(query_text).upper()
    
    # Count boolean operators
    and_count = len(re.findall(r'\bAND\b', query_text))
    or_count = len(re.findall(r'\bOR\b', query_text))
    not_count = len(re.findall(r'\bNOT\b', query_text))
    
    # Count negation patterns (context-aware negation)
    negation_patterns = len(re.findall(r'\b(NOT|NO|NEVER|DON\'T|WON\'T|CAN\'T|ISN\'T|DOESN\'T)\b', query_text))
    
    # Count proximity operators
    proximity_rules = len(re.findall(r'(NEAR|BEFORE|AFTER):\d+[WS]?', query_text))
    
    # Count wildcards
    wildcard_usage = query_text.count('*') + query_text.count('?')
    
    # Calculate parentheses depth
    max_depth = 0
    current_depth = 0
    for char in query_text:
        if char == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            current_depth -= 1
    
    # Count quoted patterns
    quote_patterns = len(re.findall(r'"[^"]*"', query_text))
    
    # Calculate total complexity score
    complexity_score = (
        and_count * 1 +
        or_count * 2 +  # OR adds more complexity
        not_count * 3 +  # NOT adds significant complexity
        proximity_rules * 4 +  # Proximity rules are complex
        wildcard_usage * 1 +
        max_depth * 2 +
        quote_patterns * 1
    )
    
    return {
        'boolean_operators': and_count + or_count + not_count,
        'and_count': and_count,
        'or_count': or_count,
        'not_count': not_count,
        'negation_patterns': negation_patterns,
        'proximity_rules': proximity_rules,
        'wildcard_usage': wildcard_usage,
        'parentheses_depth': max_depth,
        'quote_patterns': quote_patterns,
        'total_complexity_score': complexity_score
    }

def profile_all_complaint_rules(df_rules):
    """Profile complexity for all complaint rules"""
    
    print("1.1 RULE COMPLEXITY PROFILING")
    print("-" * 40)
    
    if df_rules is None or len(df_rules) == 0:
        print("No complaint rules data available")
        return pd.DataFrame()
    
    # Analyze complexity for each rule
    complexity_results = []
    
    for idx, rule in df_rules.iterrows():
        query_text = rule.get('Query Text', '')
        
        complexity = analyze_query_complexity(query_text)
        
        result = {
            'Event': rule.get('Event', ''),
            'Query': rule.get('Query', ''),
            'Channel': rule.get('Channel', ''),
            'Query_Text_Length': len(str(query_text)) if pd.notna(query_text) else 0,
            **complexity
        }
        
        complexity_results.append(result)
    
    complexity_df = pd.DataFrame(complexity_results)
    
    print(f"Analyzed {len(complexity_df)} complaint rules")
    print("\nComplexity Distribution Summary:")
    
    # Summary statistics
    summary_stats = complexity_df[['total_complexity_score', 'boolean_operators', 'negation_patterns', 
                                 'proximity_rules', 'wildcard_usage']].describe()
    print(summary_stats.round(2))
    
    # Categorize rules by complexity
    complexity_df['complexity_category'] = pd.cut(
        complexity_df['total_complexity_score'], 
        bins=[0, 5, 15, 30, float('inf')], 
        labels=['Simple', 'Moderate', 'Complex', 'Very Complex']
    )
    
    print("\nComplexity Category Distribution:")
    complexity_dist = complexity_df['complexity_category'].value_counts()
    print(complexity_dist)
    
    # Identify most complex rules
    print("\nTop 10 Most Complex Rules:")
    most_complex = complexity_df.nlargest(10, 'total_complexity_score')[
        ['Event', 'Query', 'total_complexity_score', 'or_count', 'not_count', 'proximity_rules']
    ]
    print(most_complex)
    
    # Identify rules with poor negation handling
    print("\nRules with High Negation Usage but No Context Handling:")
    high_negation_rules = complexity_df[
        (complexity_df['negation_patterns'] > 3) & 
        (complexity_df['proximity_rules'] == 0)
    ][['Event', 'Query', 'negation_patterns', 'proximity_rules']]
    
    if len(high_negation_rules) > 0:
        print(high_negation_rules.head(10))
    else:
        print("No rules identified with this pattern")
    
    return complexity_df

# Execute complexity profiling
complexity_df = profile_all_complaint_rules(df_complaint_rules)

# 1.2 Rule-Category Effectiveness Mapping

def map_rule_category_effectiveness(df_main, df_rules, complexity_df):
    """Map rule effectiveness by category using actual performance data"""
    
    print("\n1.2 RULE-CATEGORY EFFECTIVENESS MAPPING")
    print("-" * 45)
    
    if df_main is None or df_rules is None:
        print("Missing required data for effectiveness mapping")
        return pd.DataFrame(), pd.DataFrame()
    
    # Calculate actual category performance from main data
    category_performance = df_main.groupby(['Prosodica L1', 'Prosodica L2']).agg({
        'Primary Marker': lambda x: (x == 'TP').mean(),  # Precision
        'variable5': 'count'  # Volume
    }).reset_index()
    
    category_performance.columns = ['L1_Category', 'L2_Category', 'Actual_Precision', 'Volume']
    category_performance['FP_Rate'] = 1 - category_performance['Actual_Precision']
    
    print(f"Category Performance Analysis - {len(category_performance)} categories")
    
    # Map rules to categories
    rule_category_mapping = []
    
    for idx, rule in df_rules.iterrows():
        event = rule.get('Event', '')
        query = rule.get('Query', '')
        channel = rule.get('Channel', '')
        
        # Find matching category performance
        matching_categories = category_performance[
            (category_performance['L1_Category'].str.lower() == event.lower()) |
            (category_performance['L2_Category'].str.lower() == query.lower())
        ]
        
        if len(matching_categories) > 0:
            for _, cat in matching_categories.iterrows():
                
                # Get complexity data
                complexity_info = complexity_df[
                    (complexity_df['Event'] == event) & 
                    (complexity_df['Query'] == query)
                ]
                
                if len(complexity_info) > 0:
                    complexity_score = complexity_info['total_complexity_score'].iloc[0]
                    negation_patterns = complexity_info['negation_patterns'].iloc[0]
                    proximity_rules = complexity_info['proximity_rules'].iloc[0]
                else:
                    complexity_score = 0
                    negation_patterns = 0
                    proximity_rules = 0
                
                rule_category_mapping.append({
                    'Event': event,
                    'Query': query,
                    'Channel': channel,
                    'L1_Category': cat['L1_Category'],
                    'L2_Category': cat['L2_Category'],
                    'Actual_Precision': cat['Actual_Precision'],
                    'Volume': cat['Volume'],
                    'FP_Rate': cat['FP_Rate'],
                    'Complexity_Score': complexity_score,
                    'Negation_Patterns': negation_patterns,
                    'Proximity_Rules': proximity_rules
                })
    
    effectiveness_df = pd.DataFrame(rule_category_mapping)
    
    if len(effectiveness_df) == 0:
        print("No rule-category mappings found")
        return pd.DataFrame(), category_performance
    
    print(f"Mapped {len(effectiveness_df)} rule-category combinations")
    
    # Analyze effectiveness patterns
    print("\nEffectiveness Analysis by Channel:")
    channel_effectiveness = effectiveness_df.groupby('Channel').agg({
        'Actual_Precision': 'mean',
        'FP_Rate': 'mean',
        'Volume': 'sum',
        'Complexity_Score': 'mean'
    }).round(3)
    print(channel_effectiveness)
    
    # Analyze effectiveness by complexity
    print("\nEffectiveness Analysis by Complexity:")
    effectiveness_df['complexity_bin'] = pd.cut(
        effectiveness_df['Complexity_Score'],
        bins=[0, 5, 15, 30, float('inf')],
        labels=['Simple', 'Moderate', 'Complex', 'Very Complex']
    )
    
    complexity_effectiveness = effectiveness_df.groupby('complexity_bin').agg({
        'Actual_Precision': 'mean',
        'FP_Rate': 'mean',
        'Volume': 'mean'
    }).round(3)
    print(complexity_effectiveness)
    
    # Identify problematic rule-category combinations
    print("\nWorst Performing Rule-Category Combinations (Low Precision, High Volume):")
    problematic = effectiveness_df[
        (effectiveness_df['Actual_Precision'] < 0.7) & 
        (effectiveness_df['Volume'] > effectiveness_df['Volume'].quantile(0.5))
    ].sort_values('Actual_Precision')
    
    if len(problematic) > 0:
        print(problematic[['Event', 'Query', 'Channel', 'Actual_Precision', 'Volume', 'Complexity_Score']].head(10))
    else:
        print("No high-volume low-precision combinations identified")
    
    # Multi-category analysis
    print("\nMulti-Category Rule Analysis:")
    multi_category_rules = effectiveness_df.groupby(['Event', 'Query']).size().reset_index(name='category_count')
    multi_category_rules = multi_category_rules[multi_category_rules['category_count'] > 1]
    
    if len(multi_category_rules) > 0:
        print(f"Found {len(multi_category_rules)} rules triggering multiple categories")
        
        # Analyze average precision for multi-category rules
        multi_category_analysis = effectiveness_df.merge(
            multi_category_rules[['Event', 'Query']], 
            on=['Event', 'Query']
        ).groupby(['Event', 'Query']).agg({
            'Actual_Precision': 'mean',
            'Volume': 'sum',
            'category_count': 'first'
        }).reset_index()
        
        print("Multi-Category Rule Performance:")
        print(multi_category_analysis.sort_values('Actual_Precision').head(10))
    else:
        print("No multi-category rules detected")
    
    return effectiveness_df, category_performance

# Execute effectiveness mapping
effectiveness_df, category_performance = map_rule_category_effectiveness(df_main, df_complaint_rules, complexity_df)

# 1.3 Temporal Rule Performance Tracking

def analyze_temporal_rule_performance(df_main):
    """Analyze rule performance changes over time periods"""
    
    print("\n1.3 TEMPORAL RULE PERFORMANCE TRACKING")
    print("-" * 42)
    
    if df_main is None:
        print("No main data available for temporal analysis")
        return pd.DataFrame()
    
    # Prepare temporal data
    df_main['Date'] = pd.to_datetime(df_main['Date'])
    df_main['Year_Month'] = df_main['Date'].dt.strftime('%Y-%m')
    
    # Define periods
    pre_months = ['2024-10', '2024-11', '2024-12']
    post_months = ['2025-01', '2025-02', '2025-03']
    
    df_main['Period'] = df_main['Year_Month'].apply(
        lambda x: 'Pre' if str(x) in pre_months else 'Post' if str(x) in post_months else 'Other'
    )
    
    # Filter to Pre and Post periods only
    df_temporal = df_main[df_main['Period'].isin(['Pre', 'Post'])].copy()
    
    print(f"Temporal analysis data: {len(df_temporal)} records")
    print(f"Pre period: {(df_temporal['Period'] == 'Pre').sum()} records")
    print(f"Post period: {(df_temporal['Period'] == 'Post').sum()} records")
    
    # Calculate precision by category and period
    temporal_performance = df_temporal.groupby(['Prosodica L1', 'Prosodica L2', 'Period']).agg({
        'Primary Marker': lambda x: (x == 'TP').mean(),
        'variable5': 'count'
    }).reset_index()
    
    temporal_performance.columns = ['L1_Category', 'L2_Category', 'Period', 'Precision', 'Volume']
    
    # Pivot to compare Pre vs Post
    precision_comparison = temporal_performance.pivot_table(
        index=['L1_Category', 'L2_Category'],
        columns='Period',
        values='Precision',
        fill_value=0
    ).reset_index()
    
    # Calculate precision change
    if 'Pre' in precision_comparison.columns and 'Post' in precision_comparison.columns:
        precision_comparison['Precision_Change'] = precision_comparison['Post'] - precision_comparison['Pre']
        precision_comparison['Percent_Change'] = (
            (precision_comparison['Post'] - precision_comparison['Pre']) / 
            precision_comparison['Pre'] * 100
        ).fillna(0)
    else:
        print("Missing Pre or Post period data for comparison")
        return pd.DataFrame()
    
    # Volume comparison
    volume_comparison = temporal_performance.pivot_table(
        index=['L1_Category', 'L2_Category'],
        columns='Period',
        values='Volume',
        fill_value=0
    ).reset_index()
    
    if 'Pre' in volume_comparison.columns and 'Post' in volume_comparison.columns:
        volume_comparison['Volume_Change'] = volume_comparison['Post'] - volume_comparison['Pre']
        volume_comparison['Volume_Percent_Change'] = (
            (volume_comparison['Post'] - volume_comparison['Pre']) / 
            volume_comparison['Pre'] * 100
        ).fillna(0)
    
    print("\nPrecision Changes by Category (Pre vs Post):")
    
    # Filter for categories with sufficient data in both periods
    significant_categories = precision_comparison[
        (precision_comparison['Pre'] > 0) & 
        (precision_comparison['Post'] > 0)
    ].copy()
    
    if len(significant_categories) > 0:
        print(f"Categories with data in both periods: {len(significant_categories)}")
        
        # Biggest precision drops
        biggest_drops = significant_categories[
            significant_categories['Precision_Change'] < -0.1
        ].sort_values('Precision_Change')
        
        print("\nBiggest Precision Drops (>10%):")
        if len(biggest_drops) > 0:
            print(biggest_drops[['L1_Category', 'L2_Category', 'Pre', 'Post', 'Precision_Change', 'Percent_Change']].head(10))
        else:
            print("No categories with >10% precision drop identified")
        
        # Biggest precision improvements
        biggest_improvements = significant_categories[
            significant_categories['Precision_Change'] > 0.1
        ].sort_values('Precision_Change', ascending=False)
        
        print("\nBiggest Precision Improvements (>10%):")
        if len(biggest_improvements) > 0:
            print(biggest_improvements[['L1_Category', 'L2_Category', 'Pre', 'Post', 'Precision_Change', 'Percent_Change']].head(10))
        else:
            print("No categories with >10% precision improvement identified")
    
    # Monthly trend analysis
    print("\nMonthly Performance Trends:")
    monthly_performance = df_temporal.groupby('Year_Month').agg({
        'Primary Marker': lambda x: (x == 'TP').mean(),
        'variable5': 'count'
    }).reset_index()
    
    monthly_performance.columns = ['Year_Month', 'Overall_Precision', 'Volume']
    monthly_performance = monthly_performance.sort_values('Year_Month')
    monthly_performance['Precision_Change'] = monthly_performance['Overall_Precision'].diff()
    
    print(monthly_performance.round(3))
    
    # Identify critical months
    critical_months = monthly_performance[abs(monthly_performance['Precision_Change']) > 0.05]
    
    if len(critical_months) > 0:
        print("\nCritical Months (>5% precision change):")
        print(critical_months[['Year_Month', 'Overall_Precision', 'Precision_Change']].round(3))
    
    # Category-specific temporal patterns
    print("\nCategory-Specific Temporal Analysis:")
    
    # Categories with consistent degradation
    degrading_categories = significant_categories[
        significant_categories['Precision_Change'] < 0
    ].sort_values('Precision_Change')
    
    if len(degrading_categories) > 0:
        print(f"\nCategories with Precision Degradation: {len(degrading_categories)}")
        print("Top 10 Degrading Categories:")
        print(degrading_categories[['L1_Category', 'L2_Category', 'Precision_Change']].head(10).round(3))
        
        # Check if degrading categories have common patterns
        if len(effectiveness_df) > 0:
            degrading_with_complexity = degrading_categories.merge(
                effectiveness_df[['L1_Category', 'L2_Category', 'Complexity_Score', 'Channel']].drop_duplicates(),
                on=['L1_Category', 'L2_Category'],
                how='left'
            )
            
            print("\nComplexity Analysis of Degrading Categories:")
            degrading_complexity = degrading_with_complexity.groupby('Channel')['Complexity_Score'].mean()
            print(degrading_complexity.round(2))
    
    # Return comprehensive temporal analysis
    temporal_summary = {
        'precision_comparison': precision_comparison,
        'volume_comparison': volume_comparison,
        'monthly_performance': monthly_performance,
        'degrading_categories': degrading_categories if len(degrading_categories) > 0 else pd.DataFrame()
    }
    
    return temporal_summary

# Execute temporal analysis
temporal_analysis = analyze_temporal_rule_performance(df_main)

# Summary and Rule Prioritization

def generate_rule_analysis_summary(complexity_df, effectiveness_df, temporal_analysis):
    """Generate comprehensive summary of rule analysis findings"""
    
    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY: QUERY RULE PATTERN ANALYSIS")
    print("=" * 60)
    
    # Key findings summary
    key_findings = []
    
    # Complexity findings
    if len(complexity_df) > 0:
        avg_complexity = complexity_df['total_complexity_score'].mean()
        complex_rules = len(complexity_df[complexity_df['total_complexity_score'] > 20])
        high_negation_rules = len(complexity_df[complexity_df['negation_patterns'] > 3])
        
        key_findings.append(f"Average rule complexity score: {avg_complexity:.1f}")
        key_findings.append(f"Complex rules (score >20): {complex_rules} ({complex_rules/len(complexity_df)*100:.1f}%)")
        key_findings.append(f"High negation rules (>3 patterns): {high_negation_rules}")
    
    # Effectiveness findings
    if len(effectiveness_df) > 0:
        avg_precision = effectiveness_df['Actual_Precision'].mean()
        low_precision_rules = len(effectiveness_df[effectiveness_df['Actual_Precision'] < 0.7])
        
        key_findings.append(f"Average rule precision: {avg_precision:.3f}")
        key_findings.append(f"Low precision rules (<70%): {low_precision_rules}")
        
        # Channel analysis
        channel_performance = effectiveness_df.groupby('Channel')['Actual_Precision'].mean()
        worst_channel = channel_performance.idxmin()
        key_findings.append(f"Worst performing channel: {worst_channel} ({channel_performance[worst_channel]:.3f})")
    
    # Temporal findings
    if 'precision_comparison' in temporal_analysis and len(temporal_analysis['precision_comparison']) > 0:
        precision_changes = temporal_analysis['precision_comparison']['Precision_Change'].dropna()
        if len(precision_changes) > 0:
            avg_change = precision_changes.mean()
            degrading_count = len(precision_changes[precision_changes < -0.05])
            key_findings.append(f"Average precision change (Pre to Post): {avg_change:+.3f}")
            key_findings.append(f"Categories with >5% degradation: {degrading_count}")
    
    print("KEY FINDINGS:")
    for i, finding in enumerate(key_findings, 1):
        print(f"{i}. {finding}")
    
    # Priority rules for improvement
    print("\nPRIORITY RULES FOR IMPROVEMENT:")
    
    priority_rules = []
    
    if len(effectiveness_df) > 0:
        # High volume, low precision rules
        high_impact_rules = effectiveness_df[
            (effectiveness_df['Actual_Precision'] < 0.7) &
            (effectiveness_df['Volume'] > effectiveness_df['Volume'].quantile(0.7))
        ].sort_values(['Actual_Precision', 'Volume'], ascending=[True, False])
        
        if len(high_impact_rules) > 0:
            print("\n1. HIGH IMPACT RULES (Low Precision + High Volume):")
            print(high_impact_rules[['Event', 'Query', 'Channel', 'Actual_Precision', 'Volume']].head(5).to_string(index=False))
            priority_rules.extend(high_impact_rules[['Event', 'Query']].head(5).to_dict('records'))
    
    if len(complexity_df) > 0:
        # High complexity, potentially over-engineered rules
        complex_rules = complexity_df[
            complexity_df['total_complexity_score'] > complexity_df['total_complexity_score'].quantile(0.9)
        ].sort_values('total_complexity_score', ascending=False)
        
        if len(complex_rules) > 0:
            print("\n2. OVER-COMPLEX RULES (Top 10% Complexity):")
            print(complex_rules[['Event', 'Query', 'total_complexity_score', 'or_count', 'not_count']].head(5).to_string(index=False))
    
    # Rules with poor negation handling
    if len(complexity_df) > 0:
        poor_negation_rules = complexity_df[
            (complexity_df['negation_patterns'] > 2) &
            (complexity_df['proximity_rules'] == 0)
        ]
        
        if len(poor_negation_rules) > 0:
            print("\n3. POOR NEGATION HANDLING RULES:")
            print(poor_negation_rules[['Event', 'Query', 'negation_patterns', 'proximity_rules']].head(5).to_string(index=False))
    
    # Temporal degradation rules
    if 'degrading_categories' in temporal_analysis and len(temporal_analysis['degrading_categories']) > 0:
        print("\n4. TEMPORALLY DEGRADING RULES:")
        degrading = temporal_analysis['degrading_categories'].head(5)
        print(degrading[['L1_Category', 'L2_Category', 'Precision_Change']].to_string(index=False))
    
    print("\nRECOMMENDATIONS FOR NEXT PHASES:")
    recommendations = [
        "Focus NLP correlation analysis on high-impact low-precision rules",
        "Implement context-aware negation handling for rules with poor negation patterns",
        "Simplify over-complex rules while maintaining effectiveness",
        "Investigate temporal degradation causes for declining categories",
        "Enhance channel-specific logic for 'both' channel rules",
        "Implement proximity operators for high-negation rules without context"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    return {
        'key_findings': key_findings,
        'priority_rules': priority_rules,
        'recommendations': recommendations
    }

# Generate summary
analysis_summary = generate_rule_analysis_summary(complexity_df, effectiveness_df, temporal_analysis)

# Export Results
def export_phase1_results(complexity_df, effectiveness_df, temporal_analysis, analysis_summary):
    """Export all Phase 1 analysis results to Excel"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'Phase1_Query_Rule_Analysis_{timestamp}.xlsx'
    
    print(f"\nExporting Phase 1 results to: {filename}")
    
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        
        # Rule complexity analysis
        if len(complexity_df) > 0:
            complexity_df.to_excel(writer, sheet_name='Rule_Complexity', index=False)
        
        # Rule effectiveness mapping
        if len(effectiveness_df) > 0:
            effectiveness_df.to_excel(writer, sheet_name='Rule_Effectiveness', index=False)
        
        # Temporal analysis results
        if 'precision_comparison' in temporal_analysis:
            temporal_analysis['precision_comparison'].to_excel(writer, sheet_name='Precision_Changes', index=False)
        
        if 'monthly_performance' in temporal_analysis:
            temporal_analysis['monthly_performance'].to_excel(writer, sheet_name='Monthly_Performance', index=False)
        
        # Summary findings
        summary_df = pd.DataFrame({
            'Finding_Type': ['Key Finding'] * len(analysis_summary['key_findings']) + 
                          ['Recommendation'] * len(analysis_summary['recommendations']),
            'Description': analysis_summary['key_findings'] + analysis_summary['recommendations']
        })
        summary_df.to_excel(writer, sheet_name='Summary_Findings', index=False)
        
        # Priority rules
        if len(analysis_summary['priority_rules']) > 0:
            priority_df = pd.DataFrame(analysis_summary['priority_rules'])
            priority_df.to_excel(writer, sheet_name='Priority_Rules', index=False)
    
    print("Phase 1 analysis export completed successfully!")
    print("\nFiles generated:")
    print(f"- {filename} (Complete Phase 1 analysis results)")
    
    return filename

# Export results
export_filename = export_phase1_results(complexity_df, effectiveness_df, temporal_analysis, analysis_summary)

print("\n" + "=" * 60)
print("PHASE 1 QUERY RULE PATTERN ANALYSIS COMPLETED")
print("=" * 60)
print("Ready to proceed to Phase 2: NLP Signal-Rule Correlation Framework")

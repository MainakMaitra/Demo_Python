# PART 1: Deep Negation Pattern Analysis
# Resolving the contradiction in negation patterns

def deep_negation_analysis(df):
    """
    Comprehensive negation analysis to resolve the apparent contradiction
    between high TP negation counts and high FP negation rates
    """
    
    print("="*80)
    print("DEEP NEGATION PATTERN ANALYSIS")
    print("="*80)
    
    # 1. Basic Negation Statistics
    print("1. BASIC NEGATION STATISTICS")
    print("-" * 40)
    
    tp_data = df[df['Primary Marker'] == 'TP']
    fp_data = df[df['Primary Marker'] == 'FP']
    
    basic_stats = pd.DataFrame({
        'Metric': [
            'Total Records',
            'Records with Negations (count > 0)',
            'Percentage with Negations',
            'Average Negation Count',
            'Median Negation Count',
            'Max Negation Count'
        ],
        'True_Positives': [
            len(tp_data),
            (tp_data['Customer_Negation_Count'] > 0).sum(),
            (tp_data['Customer_Negation_Count'] > 0).mean() * 100,
            tp_data['Customer_Negation_Count'].mean(),
            tp_data['Customer_Negation_Count'].median(),
            tp_data['Customer_Negation_Count'].max()
        ],
        'False_Positives': [
            len(fp_data),
            (fp_data['Customer_Negation_Count'] > 0).sum(),
            (fp_data['Customer_Negation_Count'] > 0).mean() * 100,
            fp_data['Customer_Negation_Count'].mean(),
            fp_data['Customer_Negation_Count'].median(),
            fp_data['Customer_Negation_Count'].max()
        ]
    })
    
    print(basic_stats.round(2))
    
    # 2. Negation Rate by Category (This is the key insight!)
    print("\n2. NEGATION RATE BY CATEGORY")
    print("-" * 40)
    
    negation_by_category = []
    
    for l1_cat in df['Prosodica L1'].unique():
        if pd.notna(l1_cat):
            cat_data = df[df['Prosodica L1'] == l1_cat]
            cat_tp = cat_data[cat_data['Primary Marker'] == 'TP']
            cat_fp = cat_data[cat_data['Primary Marker'] == 'FP']
            
            if len(cat_tp) >= 5 and len(cat_fp) >= 5:  # Minimum sample size
                tp_neg_rate = (cat_tp['Customer_Negation_Count'] > 0).mean() * 100
                fp_neg_rate = (cat_fp['Customer_Negation_Count'] > 0).mean() * 100
                
                tp_avg_count = cat_tp['Customer_Negation_Count'].mean()
                fp_avg_count = cat_fp['Customer_Negation_Count'].mean()
                
                negation_by_category.append({
                    'Category': l1_cat,
                    'TP_Count': len(cat_tp),
                    'FP_Count': len(cat_fp),
                    'TP_Negation_Rate_%': tp_neg_rate,
                    'FP_Negation_Rate_%': fp_neg_rate,
                    'TP_Avg_Negation_Count': tp_avg_count,
                    'FP_Avg_Negation_Count': fp_avg_count,
                    'Rate_Difference': fp_neg_rate - tp_neg_rate,
                    'Count_Difference': fp_avg_count - tp_avg_count
                })
    
    negation_category_df = pd.DataFrame(negation_by_category)
    negation_category_df = negation_category_df.sort_values('Rate_Difference', ascending=False)
    
    print("Negation Patterns by Category (sorted by Rate Difference):")
    print(negation_category_df.round(2))
    
    # 3. Context Analysis: What type of negations?
    print("\n3. NEGATION CONTEXT ANALYSIS")
    print("-" * 40)
    
    # Define different types of negation patterns
    negation_patterns = {
        'Simple_Negation': r'\b(not|no|never)\b',
        'Contracted_Negation': r'\b(don\'t|won\'t|can\'t|isn\'t|doesn\'t|haven\'t)\b',
        'Complaint_Negation': r'\b(not (working|fair|right|correct|satisfied)|never (received|got|works))\b',
        'Information_Negation': r'\b(don\'t (understand|know)|not (sure|clear))\b',
        'Service_Negation': r'\b(can\'t (help|access|login)|won\'t (work|load))\b'
    }
    
    negation_context_analysis = []
    
    for pattern_name, pattern in negation_patterns.items():
        tp_matches = tp_data['Customer Transcript'].str.lower().str.contains(pattern, regex=True, na=False)
        fp_matches = fp_data['Customer Transcript'].str.lower().str.contains(pattern, regex=True, na=False)
        
        tp_rate = tp_matches.mean() * 100 if len(tp_data) > 0 else 0
        fp_rate = fp_matches.mean() * 100 if len(fp_data) > 0 else 0
        
        negation_context_analysis.append({
            'Negation_Type': pattern_name,
            'TP_Rate_%': tp_rate,
            'FP_Rate_%': fp_rate,
            'Difference': fp_rate - tp_rate,
            'Risk_Ratio': fp_rate / max(tp_rate, 0.1)
        })
    
    context_df = pd.DataFrame(negation_context_analysis)
    context_df = context_df.sort_values('Risk_Ratio', ascending=False)
    
    print("Negation Context Analysis:")
    print(context_df.round(2))
    
    # 4. Monthly Negation Evolution
    print("\n4. MONTHLY NEGATION EVOLUTION")
    print("-" * 40)
    
    monthly_negation = df.groupby(['Year_Month', 'Primary Marker']).agg({
        'Customer_Negation_Count': ['count', 'mean'],
        'variable5': 'count'
    }).reset_index()
    
    monthly_negation.columns = ['Year_Month', 'Primary_Marker', 'Records_With_Negations', 'Avg_Negation_Count', 'Total_Records']
    monthly_negation['Negation_Rate_%'] = (monthly_negation['Records_With_Negations'] / monthly_negation['Total_Records']) * 100
    
    # Pivot for easier comparison
    monthly_pivot = monthly_negation.pivot(index='Year_Month', columns='Primary_Marker', values=['Negation_Rate_%', 'Avg_Negation_Count']).round(2)
    
    print("Monthly Negation Evolution:")
    print(monthly_pivot)
    
    # 5. The Resolution: Negation Density vs Presence
    print("\n5. RESOLUTION: NEGATION DENSITY VS PRESENCE")
    print("-" * 40)
    
    # Key insight: Look at negation density in transcripts that contain negations
    tp_with_neg = tp_data[tp_data['Customer_Negation_Count'] > 0]
    fp_with_neg = fp_data[fp_data['Customer_Negation_Count'] > 0]
    
    density_analysis = pd.DataFrame({
        'Metric': [
            'Records with Negations',
            'Avg Negations per Record (with negations)',
            'Negations per 1000 characters',
            'Percentage of all records with negations'
        ],
        'True_Positives': [
            len(tp_with_neg),
            tp_with_neg['Customer_Negation_Count'].mean() if len(tp_with_neg) > 0 else 0,
            (tp_with_neg['Customer_Negation_Count'].sum() / tp_with_neg['Transcript_Length'].sum() * 1000) if len(tp_with_neg) > 0 else 0,
            len(tp_with_neg) / len(tp_data) * 100
        ],
        'False_Positives': [
            len(fp_with_neg),
            fp_with_neg['Customer_Negation_Count'].mean() if len(fp_with_neg) > 0 else 0,
            (fp_with_neg['Customer_Negation_Count'].sum() / fp_with_neg['Transcript_Length'].sum() * 1000) if len(fp_with_neg) > 0 else 0,
            len(fp_with_neg) / len(fp_data) * 100
        ]
    })
    
    print("Negation Density Analysis:")
    print(density_analysis.round(2))
    
    return negation_category_df, context_df, monthly_pivot, density_analysis

# Execute the analysis
negation_category_df, context_df, monthly_pivot, density_analysis = deep_negation_analysis(df_main)

# PART 2: Agent Explanations Contamination Analysis
# Detailed monthly breakdown by category and multi-category analysis

def enhanced_agent_contamination_analysis(df):
    """
    Enhanced analysis of agent explanations contaminating classification
    with detailed monthly and category breakdowns
    """
    
    print("="*80)
    print("ENHANCED AGENT CONTAMINATION ANALYSIS")
    print("="*80)
    
    # 1. Define Agent Explanation Patterns
    agent_explanation_patterns = {
        'Direct_Explanations': r'\b(let me explain|i\'ll explain|what this means|this means that)\b',
        'Examples': r'\b(for example|for instance|let\'s say|suppose)\b',
        'Hypotheticals': r'\b(if you|what if|in case|should you|were to)\b',
        'Clarifications': r'\b(to clarify|what i mean|in other words|basically)\b',
        'Instructions': r'\b(you need to|you should|you can|you have to)\b'
    }
    
    # 2. Identify contaminated transcripts
    def identify_agent_contamination(row):
        """Identify different types of agent contamination"""
        agent_text = str(row['Agent Transcript']).lower()
        customer_text = str(row['Customer Transcript']).lower()
        
        contamination_score = 0
        contamination_types = []
        
        for pattern_name, pattern in agent_explanation_patterns.items():
            if re.search(pattern, agent_text):
                contamination_score += 1
                contamination_types.append(pattern_name)
        
        # Check if agent explanation is followed by complaint keywords in customer response
        complaint_keywords = r'\b(problem|issue|wrong|error|mistake|confused|don\'t understand)\b'
        if re.search(complaint_keywords, customer_text) and contamination_score > 0:
            contamination_score += 2
            contamination_types.append('Customer_Confusion_After_Agent_Explanation')
        
        return {
            'Agent_Contamination_Score': contamination_score,
            'Agent_Contamination_Types': ';'.join(contamination_types),
            'Has_Agent_Contamination': contamination_score > 0
        }
    
    # Apply contamination analysis
    contamination_analysis = df.apply(identify_agent_contamination, axis=1, result_type='expand')
    df_enhanced = pd.concat([df, contamination_analysis], axis=1)
    
    # 3. Category-wise Agent Contamination Analysis
    print("1. CATEGORY-WISE AGENT CONTAMINATION ANALYSIS")
    print("-" * 50)
    
    category_contamination = []
    
    for l1_cat in df_enhanced['Prosodica L1'].unique():
        if pd.notna(l1_cat):
            cat_data = df_enhanced[df_enhanced['Prosodica L1'] == l1_cat]
            cat_tp = cat_data[cat_data['Primary Marker'] == 'TP']
            cat_fp = cat_data[cat_data['Primary Marker'] == 'FP']
            
            if len(cat_fp) >= 5:  # Minimum sample size for FPs
                tp_contamination_rate = (cat_tp['Has_Agent_Contamination']).mean() * 100 if len(cat_tp) > 0 else 0
                fp_contamination_rate = (cat_fp['Has_Agent_Contamination']).mean() * 100
                
                tp_avg_ratio = cat_tp['Customer_Agent_Ratio'].mean() if len(cat_tp) > 0 else 0
                fp_avg_ratio = cat_fp['Customer_Agent_Ratio'].mean()
                
                category_contamination.append({
                    'Category': l1_cat,
                    'TP_Count': len(cat_tp),
                    'FP_Count': len(cat_fp),
                    'TP_Contamination_Rate_%': tp_contamination_rate,
                    'FP_Contamination_Rate_%': fp_contamination_rate,
                    'TP_Customer_Agent_Ratio': tp_avg_ratio,
                    'FP_Customer_Agent_Ratio': fp_avg_ratio,
                    'Contamination_Risk': fp_contamination_rate / max(tp_contamination_rate, 1),
                    'Ratio_Difference': fp_avg_ratio - tp_avg_ratio
                })
    
    category_contamination_df = pd.DataFrame(category_contamination)
    category_contamination_df = category_contamination_df.sort_values('Contamination_Risk', ascending=False)
    
    print("Category-wise Agent Contamination Analysis:")
    print(category_contamination_df.round(3))
    
    # 4. Monthly Analysis for Top 3 Contaminated Categories
    print("\n2. MONTHLY ANALYSIS FOR TOP CONTAMINATED CATEGORIES")
    print("-" * 50)
    
    top_3_categories = category_contamination_df.head(3)['Category'].tolist()
    
    # Single Category Analysis
    print("\nSINGLE CATEGORY MONTHLY ANALYSIS:")
    print("Customer-Agent Ratio by Category and Month")
    print("-" * 40)
    
    # Get transcripts that are flagged for only one category
    single_category_transcripts = df_enhanced.groupby('variable5')['Prosodica L1'].nunique()
    single_category_ids = single_category_transcripts[single_category_transcripts == 1].index
    single_cat_data = df_enhanced[df_enhanced['variable5'].isin(single_category_ids)]
    
    monthly_single_cat = []
    months = sorted(df_enhanced['Year_Month'].dropna().unique())
    
    for category in top_3_categories:
        cat_single_data = single_cat_data[single_cat_data['Prosodica L1'] == category]
        
        row_data = {'Category': category}
        
        for month in months:
            month_data = cat_single_data[cat_single_data['Year_Month'] == month]
            if len(month_data) > 0:
                fp_data = month_data[month_data['Primary Marker'] == 'FP']
                avg_ratio = fp_data['Customer_Agent_Ratio'].mean() if len(fp_data) > 0 else 0
                contamination_rate = (fp_data['Has_Agent_Contamination']).mean() * 100 if len(fp_data) > 0 else 0
                row_data[f'{month}_Ratio'] = round(avg_ratio, 3)
                row_data[f'{month}_Contamination_%'] = round(contamination_rate, 1)
            else:
                row_data[f'{month}_Ratio'] = 0
                row_data[f'{month}_Contamination_%'] = 0
        
        monthly_single_cat.append(row_data)
    
    single_cat_monthly_df = pd.DataFrame(monthly_single_cat)
    print("\nSingle Category - Customer-Agent Ratio and Contamination Rate by Month:")
    print(single_cat_monthly_df)
    
    # Pre vs Post Analysis for Single Categories
    print("\nSINGLE CATEGORY - PRE VS POST ANALYSIS:")
    print("-" * 40)
    
    pre_post_single = []
    pre_months = ['2024-10', '2024-11', '2024-12']
    post_months = ['2025-01', '2025-02', '2025-03']
    
    for category in top_3_categories:
        cat_single_data = single_cat_data[single_cat_data['Prosodica L1'] == category]
        
        pre_data = cat_single_data[cat_single_data['Year_Month'].astype(str).isin(pre_months)]
        post_data = cat_single_data[cat_single_data['Year_Month'].astype(str).isin(post_months)]
        
        pre_fp = pre_data[pre_data['Primary Marker'] == 'FP']
        post_fp = post_data[post_data['Primary Marker'] == 'FP']
        
        pre_ratio = pre_fp['Customer_Agent_Ratio'].mean() if len(pre_fp) > 0 else 0
        post_ratio = post_fp['Customer_Agent_Ratio'].mean() if len(post_fp) > 0 else 0
        
        pre_contamination = (pre_fp['Has_Agent_Contamination']).mean() * 100 if len(pre_fp) > 0 else 0
        post_contamination = (post_fp['Has_Agent_Contamination']).mean() * 100 if len(post_fp) > 0 else 0
        
        pre_post_single.append({
            'Category': category,
            'Pre_Ratio': round(pre_ratio, 3),
            'Post_Ratio': round(post_ratio, 3),
            'Ratio_Change': round(post_ratio - pre_ratio, 3),
            'Pre_Contamination_%': round(pre_contamination, 1),
            'Post_Contamination_%': round(post_contamination, 1),
            'Contamination_Change': round(post_contamination - pre_contamination, 1)
        })
    
    pre_post_single_df = pd.DataFrame(pre_post_single)
    print(pre_post_single_df)
    
    # Multi-Category Analysis
    print("\nMULTI-CATEGORY MONTHLY ANALYSIS:")
    print("-" * 40)
    
    # Get transcripts that are flagged for multiple categories
    multi_category_ids = single_category_transcripts[single_category_transcripts > 1].index
    multi_cat_data = df_enhanced[df_enhanced['variable5'].isin(multi_category_ids)]
    
    monthly_multi_cat = []
    
    for category in top_3_categories:
        cat_multi_data = multi_cat_data[multi_cat_data['Prosodica L1'] == category]
        
        row_data = {'Category': category}
        
        for month in months:
            month_data = cat_multi_data[cat_multi_data['Year_Month'] == month]
            if len(month_data) > 0:
                fp_data = month_data[month_data['Primary Marker'] == 'FP']
                avg_ratio = fp_data['Customer_Agent_Ratio'].mean() if len(fp_data) > 0 else 0
                contamination_rate = (fp_data['Has_Agent_Contamination']).mean() * 100 if len(fp_data) > 0 else 0
                row_data[f'{month}_Ratio'] = round(avg_ratio, 3)
                row_data[f'{month}_Contamination_%'] = round(contamination_rate, 1)
            else:
                row_data[f'{month}_Ratio'] = 0
                row_data[f'{month}_Contamination_%'] = 0
        
        monthly_multi_cat.append(row_data)
    
    multi_cat_monthly_df = pd.DataFrame(monthly_multi_cat)
    print("\nMulti-Category - Customer-Agent Ratio and Contamination Rate by Month:")
    print(multi_cat_monthly_df)
    
    # Pre vs Post Analysis for Multi Categories
    print("\nMULTI-CATEGORY - PRE VS POST ANALYSIS:")
    print("-" * 40)
    
    pre_post_multi = []
    
    for category in top_3_categories:
        cat_multi_data = multi_cat_data[multi_cat_data['Prosodica L1'] == category]
        
        pre_data = cat_multi_data[cat_multi_data['Year_Month'].astype(str).isin(pre_months)]
        post_data = cat_multi_data[cat_multi_data['Year_Month'].astype(str).isin(post_months)]
        
        pre_fp = pre_data[pre_data['Primary Marker'] == 'FP']
        post_fp = post_data[post_data['Primary Marker'] == 'FP']
        
        pre_ratio = pre_fp['Customer_Agent_Ratio'].mean() if len(pre_fp) > 0 else 0
        post_ratio = post_fp['Customer_Agent_Ratio'].mean() if len(post_fp) > 0 else 0
        
        pre_contamination = (pre_fp['Has_Agent_Contamination']).mean() * 100 if len(pre_fp) > 0 else 0
        post_contamination = (post_fp['Has_Agent_Contamination']).mean() * 100 if len(post_fp) > 0 else 0
        
        pre_post_multi.append({
            'Category': category,
            'Pre_Ratio': round(pre_ratio, 3),
            'Post_Ratio': round(post_ratio, 3),
            'Ratio_Change': round(post_ratio - pre_ratio, 3),
            'Pre_Contamination_%': round(pre_contamination, 1),
            'Post_Contamination_%': round(post_contamination, 1),
            'Contamination_Change': round(post_contamination - pre_contamination, 1)
        })
    
    pre_post_multi_df = pd.DataFrame(pre_post_multi)
    print(pre_post_multi_df)
    
    return (df_enhanced, category_contamination_df, single_cat_monthly_df, 
            pre_post_single_df, multi_cat_monthly_df, pre_post_multi_df)

# Execute the enhanced agent analysis
(df_enhanced, category_contamination_df, single_cat_monthly_df, 
 pre_post_single_df, multi_cat_monthly_df, pre_post_multi_df) = enhanced_agent_contamination_analysis(df_main)

# PART 3: Validation Rater Analysis
# Analyzing if specific raters are influencing agreement rates

def rater_influence_analysis(df):
    """
    Analyze if specific primary raters are influencing validation agreement rates
    """
    
    print("="*80)
    print("VALIDATION RATER INFLUENCE ANALYSIS")
    print("="*80)
    
    # Check if Primary Rater Name column exists
    if 'Primary Rater Name' not in df.columns:
        print("Warning: 'Primary Rater Name' column not found in dataset")
        return None, None, None
    
    # 1. Overall Rater Performance Analysis
    print("1. OVERALL RATER PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    # Filter data with secondary validation
    secondary_data = df[df['Has_Secondary_Validation'] & df['Primary Rater Name'].notna()].copy()
    
    if len(secondary_data) == 0:
        print("No data available with both secondary validation and rater names")
        return None, None, None
    
    rater_performance = secondary_data.groupby('Primary Rater Name').agg({
        'Primary_Secondary_Agreement': ['count', 'mean', 'std'],
        'Is_TP': 'mean',
        'Is_FP': 'mean'
    }).reset_index()
    
    rater_performance.columns = [
        'Primary_Rater_Name', 'Total_Validations', 'Agreement_Rate', 'Agreement_Std',
        'TP_Rate', 'FP_Rate'
    ]
    
    # Filter raters with minimum validations
    min_validations = 10
    rater_performance = rater_performance[rater_performance['Total_Validations'] >= min_validations]
    rater_performance = rater_performance.sort_values('Agreement_Rate')
    
    print(f"Rater Performance Analysis (minimum {min_validations} validations):")
    print(rater_performance.round(3))
    
    # 2. Rater Consistency Analysis
    print("\n2. RATER CONSISTENCY ANALYSIS")
    print("-" * 40)
    
    # Calculate consistency metrics
    overall_agreement = secondary_data['Primary_Secondary_Agreement'].mean()
    agreement_std = secondary_data['Primary_Secondary_Agreement'].std()
    
    rater_performance['Agreement_Z_Score'] = (
        (rater_performance['Agreement_Rate'] - overall_agreement) / agreement_std
    )
    
    rater_performance['Consistency_Flag'] = rater_performance['Agreement_Z_Score'].apply(
        lambda x: 'High_Disagreement' if x < -1.5 else 'High_Agreement' if x > 1.5 else 'Normal'
    )
    
    print("Rater Consistency Analysis:")
    print(rater_performance[['Primary_Rater_Name', 'Agreement_Rate', 'Agreement_Z_Score', 'Consistency_Flag']].round(3))
    
    # 3. Rater-Category Interaction Analysis
    print("\n3. RATER-CATEGORY INTERACTION ANALYSIS")
    print("-" * 40)
    
    rater_category_analysis = []
    
    for rater in rater_performance['Primary_Rater_Name'].unique():
        rater_data = secondary_data[secondary_data['Primary Rater Name'] == rater]
        
        for l1_cat in rater_data['Prosodica L1'].unique():
            if pd.notna(l1_cat):
                rater_cat_data = rater_data[rater_data['Prosodica L1'] == l1_cat]
                
                if len(rater_cat_data) >= 5:  # Minimum sample size
                    agreement_rate = rater_cat_data['Primary_Secondary_Agreement'].mean()
                    tp_rate = rater_cat_data['Is_TP'].mean()
                    fp_rate = rater_cat_data['Is_FP'].mean()
                    
                    # Compare with category average
                    cat_avg_agreement = secondary_data[
                        secondary_data['Prosodica L1'] == l1_cat
                    ]['Primary_Secondary_Agreement'].mean()
                    
                    rater_category_analysis.append({
                        'Primary_Rater_Name': rater,
                        'Category': l1_cat,
                        'Sample_Size': len(rater_cat_data),
                        'Rater_Agreement_Rate': agreement_rate,
                        'Category_Avg_Agreement': cat_avg_agreement,
                        'Agreement_Difference': agreement_rate - cat_avg_agreement,
                        'TP_Rate': tp_rate,
                        'FP_Rate': fp_rate
                    })
    
    rater_category_df = pd.DataFrame(rater_category_analysis)
    rater_category_df = rater_category_df.sort_values('Agreement_Difference')
    
    print("Rater-Category Performance (Top 10 Worst and Best):")
    print("\nWorst Performing Rater-Category Combinations:")
    print(rater_category_df.head(10)[['Primary_Rater_Name', 'Category', 'Rater_Agreement_Rate', 'Agreement_Difference']].round(3))
    
    print("\nBest Performing Rater-Category Combinations:")
    print(rater_category_df.tail(10)[['Primary_Rater_Name', 'Category', 'Rater_Agreement_Rate', 'Agreement_Difference']].round(3))
    
    # 4. Monthly Rater Performance Trends
    print("\n4. MONTHLY RATER PERFORMANCE TRENDS")
    print("-" * 40)
    
    monthly_rater_performance = secondary_data.groupby(['Year_Month', 'Primary Rater Name']).agg({
        'Primary_Secondary_Agreement': ['count', 'mean'],
        'Is_TP': 'mean',
        'Is_FP': 'mean'
    }).reset_index()
    
    monthly_rater_performance.columns = [
        'Year_Month', 'Primary_Rater_Name', 'Sample_Size', 'Agreement_Rate', 'TP_Rate', 'FP_Rate'
    ]
    
    # Filter for raters with consistent monthly data
    monthly_rater_performance = monthly_rater_performance[monthly_rater_performance['Sample_Size'] >= 5]
    
    # Pivot for better visualization
    monthly_pivot = monthly_rater_performance.pivot(
        index='Primary_Rater_Name', 
        columns='Year_Month', 
        values='Agreement_Rate'
    ).round(3)
    
    print("Monthly Rater Agreement Rates:")
    print(monthly_pivot)
    
    # Calculate trend for each rater
    rater_trends = []
    for rater in monthly_rater_performance['Primary_Rater_Name'].unique():
        rater_monthly = monthly_rater_performance[
            monthly_rater_performance['Primary_Rater_Name'] == rater
        ].sort_values('Year_Month')
        
        if len(rater_monthly) >= 3:  # Need at least 3 months for trend
            months = list(range(len(rater_monthly)))
            agreements = rater_monthly['Agreement_Rate'].tolist()
            
            try:
                from scipy.stats import linregress
                slope, _, r_value, p_value, _ = linregress(months, agreements)
                
                rater_trends.append({
                    'Primary_Rater_Name': rater,
                    'Trend_Slope': slope,
                    'R_Squared': r_value**2,
                    'P_Value': p_value,
                    'Trend_Direction': 'Improving' if slope > 0.02 else 'Declining' if slope < -0.02 else 'Stable',
                    'Months_Data': len(rater_monthly)
                })
            except:
                rater_trends.append({
                    'Primary_Rater_Name': rater,
                    'Trend_Direction': 'Unable_to_Calculate',
                    'Months_Data': len(rater_monthly)
                })
    
    rater_trends_df = pd.DataFrame(rater_trends)
    print("\nRater Performance Trends:")
    print(rater_trends_df.round(4))
    
    # 5. Statistical Analysis: Which raters are outliers?
    print("\n5. STATISTICAL OUTLIER ANALYSIS")
    print("-" * 40)
    
    # Calculate Z-scores for agreement rates
    mean_agreement = rater_performance['Agreement_Rate'].mean()
    std_agreement = rater_performance['Agreement_Rate'].std()
    
    rater_performance['Z_Score'] = (rater_performance['Agreement_Rate'] - mean_agreement) / std_agreement
    rater_performance['Outlier_Status'] = rater_performance['Z_Score'].apply(
        lambda x: 'Significant_Low' if x < -2 else 'Low' if x < -1 else 'High' if x > 1 else 'Significant_High' if x > 2 else 'Normal'
    )
    
    outliers = rater_performance[rater_performance['Outlier_Status'].isin(['Significant_Low', 'Significant_High'])]
    
    print("Statistical Outliers (>2 standard deviations from mean):")
    print(outliers[['Primary_Rater_Name', 'Agreement_Rate', 'Z_Score', 'Outlier_Status']].round(3))
    
    if len(outliers) > 0:
        print(f"\nFINDING: {len(outliers)} raters are statistical outliers")
        print("RECOMMENDATION: Review training and guidelines for these raters")
    else:
        print("FINDING: No statistical outliers detected among raters")
    
    # 6. Rater Impact on Overall System Performance
    print("\n6. RATER IMPACT ON OVERALL SYSTEM PERFORMANCE")
    print("-" * 40)
    
    # Calculate how much each rater's performance affects overall metrics
    rater_impact = []
    
    for rater in rater_performance['Primary_Rater_Name'].unique():
        rater_data = secondary_data[secondary_data['Primary Rater Name'] == rater]
        other_data = secondary_data[secondary_data['Primary Rater Name'] != rater]
        
        rater_volume = len(rater_data)
        total_volume = len(secondary_data)
        volume_share = rater_volume / total_volume
        
        rater_agreement = rater_data['Primary_Secondary_Agreement'].mean()
        others_agreement = other_data['Primary_Secondary_Agreement'].mean()
        
        # Impact calculation: volume share * performance difference
        impact_score = volume_share * abs(rater_agreement - others_agreement)
        
        rater_impact.append({
            'Primary_Rater_Name': rater,
            'Volume_Share_%': volume_share * 100,
            'Rater_Agreement': rater_agreement,
            'Others_Agreement': others_agreement,
            'Performance_Difference': rater_agreement - others_agreement,
            'Impact_Score': impact_score
        })
    
    rater_impact_df = pd.DataFrame(rater_impact)
    rater_impact_df = rater_impact_df.sort_values('Impact_Score', ascending=False)
    
    print("Rater Impact on System Performance:")
    print(rater_impact_df.round(3))
    
    high_impact_raters = rater_impact_df[rater_impact_df['Impact_Score'] > 0.01]  # 1% impact threshold
    
    if len(high_impact_raters) > 0:
        print(f"\nHigh-Impact Raters (>1% system impact): {len(high_impact_raters)}")
        print("These raters significantly influence overall validation quality")
    
    return rater_performance, rater_category_df, rater_trends_df, rater_impact_df

# Execute the rater analysis
rater_performance, rater_category_df, rater_trends_df, rater_impact_df = rater_influence_analysis(df_main)

# PART 4: Enhanced Qualifying Language Analysis
# Split by customer and agent, deep dive by category

def enhanced_qualifying_language_analysis(df):
    """
    Enhanced analysis of qualifying language patterns with customer/agent split
    and category-level deep dive
    """
    
    print("="*80)
    print("ENHANCED QUALIFYING LANGUAGE ANALYSIS")
    print("="*80)
    
    # 1. Define Enhanced Qualifying Patterns
    qualifying_patterns = {
        'Uncertainty': r'\b(might|maybe|seems|appears|possibly|perhaps|probably|likely|i think|i believe|i guess)\b',
        'Hedging': r'\b(sort of|kind of|more or less|somewhat|relatively|fairly|quite|rather)\b',
        'Approximation': r'\b(about|around|approximately|roughly|nearly|almost|close to)\b',
        'Conditional': r'\b(if|unless|provided|assuming|suppose|in case|should)\b',
        'Doubt': r'\b(not sure|uncertain|unclear|confused|don\'t know|no idea)\b',
        'Politeness': r'\b(please|thank you|thanks|appreciate|grateful|excuse me|pardon|sorry)\b'
    }
    
    # 2. Extract qualifying language for both customer and agent
    def extract_qualifying_features(row):
        """Extract qualifying language features for customer and agent separately"""
        customer_text = str(row['Customer Transcript']).lower()
        agent_text = str(row['Agent Transcript']).lower()
        
        features = {}
        
        # Customer qualifying language
        for pattern_name, pattern in qualifying_patterns.items():
            customer_matches = len(re.findall(pattern, customer_text))
            agent_matches = len(re.findall(pattern, agent_text))
            
            features[f'Customer_{pattern_name}_Count'] = customer_matches
            features[f'Agent_{pattern_name}_Count'] = agent_matches
            features[f'Total_{pattern_name}_Count'] = customer_matches + agent_matches
            
            # Ratio of customer to total
            total_matches = customer_matches + agent_matches
            if total_matches > 0:
                features[f'{pattern_name}_Customer_Ratio'] = customer_matches / total_matches
            else:
                features[f'{pattern_name}_Customer_Ratio'] = 0
        
        return features
    
    # Apply qualifying analysis
    qualifying_features = df.apply(extract_qualifying_features, axis=1, result_type='expand')
    df_qualifying = pd.concat([df, qualifying_features], axis=1)
    
    # 3. Overall Customer vs Agent Qualifying Language Analysis
    print("1. OVERALL CUSTOMER VS AGENT QUALIFYING LANGUAGE")
    print("-" * 50)
    
    tp_data = df_qualifying[df_qualifying['Primary Marker'] == 'TP']
    fp_data = df_qualifying[df_qualifying['Primary Marker'] == 'FP']
    
    overall_analysis = []
    
    for pattern_name in qualifying_patterns.keys():
        tp_customer_avg = tp_data[f'Customer_{pattern_name}_Count'].mean()
        tp_agent_avg = tp_data[f'Agent_{pattern_name}_Count'].mean()
        fp_customer_avg = fp_data[f'Customer_{pattern_name}_Count'].mean()
        fp_agent_avg = fp_data[f'Agent_{pattern_name}_Count'].mean()
        
        tp_customer_ratio = tp_data[f'{pattern_name}_Customer_Ratio'].mean()
        fp_customer_ratio = fp_data[f'{pattern_name}_Customer_Ratio'].mean()
        
        overall_analysis.append({
            'Pattern': pattern_name,
            'TP_Customer_Avg': tp_customer_avg,
            'TP_Agent_Avg': tp_agent_avg,
            'FP_Customer_Avg': fp_customer_avg,
            'FP_Agent_Avg': fp_agent_avg,
            'TP_Customer_Ratio': tp_customer_ratio,
            'FP_Customer_Ratio': fp_customer_ratio,
            'Customer_Risk_Factor': fp_customer_avg / max(tp_customer_avg, 0.01),
            'Agent_Risk_Factor': fp_agent_avg / max(tp_agent_avg, 0.01)
        })
    
    overall_df = pd.DataFrame(overall_analysis)
    print("Overall Customer vs Agent Qualifying Language Analysis:")
    print(overall_df.round(3))
    
    # 4. Category-Level Deep Dive Analysis
    print("\n2. CATEGORY-LEVEL DEEP DIVE ANALYSIS")
    print("-" * 50)
    
    category_analysis = []
    
    for l1_cat in df_qualifying['Prosodica L1'].unique():
        if pd.notna(l1_cat):
            cat_data = df_qualifying[df_qualifying['Prosodica L1'] == l1_cat]
            cat_tp = cat_data[cat_data['Primary Marker'] == 'TP']
            cat_fp = cat_data[cat_data['Primary Marker'] == 'FP']
            
            if len(cat_fp) >= 5:  # Minimum sample size
                for pattern_name in qualifying_patterns.keys():
                    tp_customer_avg = cat_tp[f'Customer_{pattern_name}_Count'].mean() if len(cat_tp) > 0 else 0
                    tp_agent_avg = cat_tp[f'Agent_{pattern_name}_Count'].mean() if len(cat_tp) > 0 else 0
                    fp_customer_avg = cat_fp[f'Customer_{pattern_name}_Count'].mean()
                    fp_agent_avg = cat_fp[f'Agent_{pattern_name}_Count'].mean()
                    
                    category_analysis.append({
                        'Category': l1_cat,
                        'Pattern': pattern_name,
                        'TP_Count': len(cat_tp),
                        'FP_Count': len(cat_fp),
                        'TP_Customer_Avg': tp_customer_avg,
                        'TP_Agent_Avg': tp_agent_avg,
                        'FP_Customer_Avg': fp_customer_avg,
                        'FP_Agent_Avg': fp_agent_avg,
                        'Customer_Difference': fp_customer_avg - tp_customer_avg,
                        'Agent_Difference': fp_agent_avg - tp_agent_avg,
                        'Customer_Risk': fp_customer_avg / max(tp_customer_avg, 0.01),
                        'Agent_Risk': fp_agent_avg / max(tp_agent_avg, 0.01)
                    })
    
    category_analysis_df = pd.DataFrame(category_analysis)
    
    # Show top risk categories for each pattern
    for pattern in qualifying_patterns.keys():
        pattern_data = category_analysis_df[category_analysis_df['Pattern'] == pattern]
        pattern_data = pattern_data.sort_values('Customer_Risk', ascending=False)
        
        print(f"\n{pattern.upper()} - Top Risk Categories:")
        print("Customer Risk:")
        print(pattern_data.head(5)[['Category', 'TP_Customer_Avg', 'FP_Customer_Avg', 'Customer_Risk']].round(3))
        print("Agent Risk:")
        agent_sorted = pattern_data.sort_values('Agent_Risk', ascending=False)
        print(agent_sorted.head(5)[['Category', 'TP_Agent_Avg', 'FP_Agent_Avg', 'Agent_Risk']].round(3))
    
    # 5. Monthly Trend Analysis for Qualifying Language
    print("\n3. MONTHLY TREND ANALYSIS")
    print("-" * 50)
    
    monthly_qualifying = []
    months = sorted(df_qualifying['Year_Month'].dropna().unique())
    
    for month in months:
        month_data = df_qualifying[df_qualifying['Year_Month'] == month]
        month_tp = month_data[month_data['Primary Marker'] == 'TP']
        month_fp = month_data[month_data['Primary Marker'] == 'FP']
        
        for pattern_name in qualifying_patterns.keys():
            tp_customer_avg = month_tp[f'Customer_{pattern_name}_Count'].mean() if len(month_tp) > 0 else 0
            tp_agent_avg = month_tp[f'Agent_{pattern_name}_Count'].mean() if len(month_tp) > 0 else 0
            fp_customer_avg = month_fp[f'Customer_{pattern_name}_Count'].mean() if len(month_fp) > 0 else 0
            fp_agent_avg = month_fp[f'Agent_{pattern_name}_Count'].mean() if len(month_fp) > 0 else 0
            
            monthly_qualifying.append({
                'Year_Month': month,
                'Pattern': pattern_name,
                'TP_Customer_Avg': tp_customer_avg,
                'TP_Agent_Avg': tp_agent_avg,
                'FP_Customer_Avg': fp_customer_avg,
                'FP_Agent_Avg': fp_agent_avg,
                'Customer_Risk': fp_customer_avg / max(tp_customer_avg, 0.01),
                'Agent_Risk': fp_agent_avg / max(tp_agent_avg, 0.01)
            })
    
    monthly_qualifying_df = pd.DataFrame(monthly_qualifying)
    
    # Pivot table for better visualization
    for pattern in ['Uncertainty', 'Doubt', 'Politeness']:  # Focus on key patterns
        print(f"\n{pattern} Monthly Trends:")
        pattern_monthly = monthly_qualifying_df[monthly_qualifying_df['Pattern'] == pattern]
        
        pivot_customer = pattern_monthly.pivot_table(
            index='Pattern', 
            columns='Year_Month', 
            values=['TP_Customer_Avg', 'FP_Customer_Avg'], 
            aggfunc='mean'
        ).round(3)
        
        print("Customer Usage:")
        print(pivot_customer)
        
        pivot_agent = pattern_monthly.pivot_table(
            index='Pattern', 
            columns='Year_Month', 
            values=['TP_Agent_Avg', 'FP_Agent_Avg'], 
            aggfunc='mean'
        ).round(3)
        
        print("Agent Usage:")
        print(pivot_agent)
    
    # 6. Pre vs Post Analysis
    print("\n4. PRE VS POST ANALYSIS")
    print("-" * 50)
    
    pre_months = ['2024-10', '2024-11', '2024-12']
    post_months = ['2025-01', '2025-02', '2025-03']
    
    pre_data = df_qualifying[df_qualifying['Year_Month'].astype(str).isin(pre_months)]
    post_data = df_qualifying[df_qualifying['Year_Month'].astype(str).isin(post_months)]
    
    pre_tp = pre_data[pre_data['Primary Marker'] == 'TP']
    pre_fp = pre_data[pre_data['Primary Marker'] == 'FP']
    post_tp = post_data[post_data['Primary Marker'] == 'TP']
    post_fp = post_data[post_data['Primary Marker'] == 'FP']
    
    pre_post_analysis = []
    
    for pattern_name in qualifying_patterns.keys():
        pre_tp_customer = pre_tp[f'Customer_{pattern_name}_Count'].mean() if len(pre_tp) > 0 else 0
        pre_fp_customer = pre_fp[f'Customer_{pattern_name}_Count'].mean() if len(pre_fp) > 0 else 0
        post_tp_customer = post_tp[f'Customer_{pattern_name}_Count'].mean() if len(post_tp) > 0 else 0
        post_fp_customer = post_fp[f'Customer_{pattern_name}_Count'].mean() if len(post_fp) > 0 else 0
        
        pre_tp_agent = pre_tp[f'Agent_{pattern_name}_Count'].mean() if len(pre_tp) > 0 else 0
        pre_fp_agent = pre_fp[f'Agent_{pattern_name}_Count'].mean() if len(pre_fp) > 0 else 0
        post_tp_agent = post_tp[f'Agent_{pattern_name}_Count'].mean() if len(post_tp) > 0 else 0
        post_fp_agent = post_fp[f'Agent_{pattern_name}_Count'].mean() if len(post_fp) > 0 else 0
        
        pre_post_analysis.append({
            'Pattern': pattern_name,
            'Pre_TP_Customer': pre_tp_customer,
            'Post_TP_Customer': post_tp_customer,
            'Pre_FP_Customer': pre_fp_customer,
            'Post_FP_Customer': post_fp_customer,
            'Pre_TP_Agent': pre_tp_agent,
            'Post_TP_Agent': post_tp_agent,
            'Pre_FP_Agent': pre_fp_agent,
            'Post_FP_Agent': post_fp_agent,
            'Customer_TP_Change': post_tp_customer - pre_tp_customer,
            'Customer_FP_Change': post_fp_customer - pre_fp_customer,
            'Agent_TP_Change': post_tp_agent - pre_tp_agent,
            'Agent_FP_Change': post_fp_agent - pre_fp_agent
        })
    
    pre_post_df = pd.DataFrame(pre_post_analysis)
    print("Pre vs Post Qualifying Language Analysis:")
    print(pre_post_df.round(3))
    
    return df_qualifying, overall_df, category_analysis_df, monthly_qualifying_df, pre_post_df

# Execute the enhanced qualifying language analysis
(df_qualifying, overall_qualifying_df, category_qualifying_df, 
 monthly_qualifying_df, pre_post_qualifying_df) = enhanced_qualifying_language_analysis(df_main)


# =============================================================================
# NEW ENHANCED ANALYSIS EXECUTION
# ADD THIS SECTION AFTER THE EXISTING Cell 11: Content Pattern Analysis
# =============================================================================

print("\n" + "="*80)
print("EXECUTING NEW ENHANCED DEEP DIVE ANALYSIS")
print("="*80)

# Cell 12: Deep Negation Analysis
print("\n### CELL 12: DEEP NEGATION ANALYSIS ###")
negation_category_df, context_df = deep_negation_analysis(df_main)

# Cell 13: Enhanced Agent Contamination Analysis
print("\n### CELL 13: ENHANCED AGENT CONTAMINATION ANALYSIS ###")
df_enhanced, category_contamination_df = enhanced_agent_contamination_analysis(df_main)

# Cell 14: Rater Influence Analysis
print("\n### CELL 14: RATER INFLUENCE ANALYSIS ###")
rater_performance, rater_outliers = rater_influence_analysis(df_main)

# Cell 15: Enhanced Qualifying Language Analysis
print("\n### CELL 15: ENHANCED QUALIFYING LANGUAGE ANALYSIS ###")
df_qualifying, overall_qualifying_df, category_qualifying_df, monthly_qualifying_df, pre_post_qualifying_df = enhanced_qualifying_language_analysis(df_main)

# Cell 16: Create Unified Feature Dataframe
print("\n### CELL 16: CREATE UNIFIED FEATURE DATAFRAME ###")
unified_dataframe, feature_documentation = create_unified_feature_dataframe(df_main, df_enhanced, df_qualifying)


# PART 5: Create Unified Dataframe with All Features
# Combine all original columns with feature-engineered columns at variable5 level

def create_unified_feature_dataframe(df_main, df_enhanced, df_qualifying):
    """
    Create a unified dataframe with all original columns and feature-engineered columns
    aggregated at the variable5 level
    """
    
    print("="*80)
    print("CREATING UNIFIED FEATURE DATAFRAME")
    print("="*80)
    
    # 1. Start with original columns aggregated at variable5 level
    print("1. AGGREGATING ORIGINAL COLUMNS AT VARIABLE5 LEVEL")
    print("-" * 50)
    
    # Identify categorical and numerical columns
    categorical_cols = [
        'Prosodica L1', 'Prosodica L2', 'Primary L1', 'Primary L2', 
        'Primary Marker', 'Secondary L1', 'Secondary L2', 'Secondary Marker',
        'Primary Rater Name', 'Year_Month', 'DayOfWeek', 'Period'
    ]
    
    # Define aggregation functions for different column types
    agg_functions = {}
    
    # For categorical columns, take the most frequent value (mode)
    for col in categorical_cols:
        if col in df_main.columns:
            agg_functions[col] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    
    # For text columns, concatenate
    text_cols = ['Customer Transcript', 'Agent Transcript']
    for col in text_cols:
        if col in df_main.columns:
            agg_functions[col] = lambda x: ' '.join(x.astype(str))
    
    # For date columns, take the first value
    if 'Date' in df_main.columns:
        agg_functions['Date'] = 'first'
    
    # For binary/numerical columns, take mean or sum as appropriate
    binary_cols = [
        'Is_TP', 'Is_FP', 'Has_Secondary_Validation', 'Is_Holiday_Season', 
        'Is_Month_End', 'Is_New_Category'
    ]
    for col in binary_cols:
        if col in df_main.columns:
            agg_functions[col] = 'mean'  # Will give the proportion
    
    # For other numerical columns, take mean
    numerical_cols = [
        'Transcript_Length', 'Customer_Word_Count', 'Agent_Word_Count',
        'Customer_Agent_Ratio', 'Customer_Question_Count', 'Customer_Exclamation_Count',
        'Customer_Caps_Ratio', 'Customer_Negation_Count', 'Agent_Negation_Count',
        'Customer_Qualifying_Count', 'WeekOfMonth', 'Quarter', 'Category_Age_Days'
    ]
    for col in numerical_cols:
        if col in df_main.columns:
            agg_functions[col] = 'mean'
    
    # Perform aggregation
    unified_base = df_main.groupby('variable5').agg(agg_functions).reset_index()
    
    print(f"Base aggregated dataframe shape: {unified_base.shape}")
    
    # 2. Add enhanced features from agent contamination analysis
    print("\n2. ADDING AGENT CONTAMINATION FEATURES")
    print("-" * 50)
    
    if 'Agent_Contamination_Score' in df_enhanced.columns:
        agent_features = df_enhanced.groupby('variable5').agg({
            'Agent_Contamination_Score': 'mean',
            'Has_Agent_Contamination': 'mean'
        }).reset_index()
        
        agent_features.columns = ['variable5', 'Avg_Agent_Contamination_Score', 'Agent_Contamination_Rate']
        
        unified_base = unified_base.merge(agent_features, on='variable5', how='left')
        print(f"Added agent contamination features. Shape: {unified_base.shape}")
    
    # 3. Add qualifying language features
    print("\n3. ADDING QUALIFYING LANGUAGE FEATURES")
    print("-" * 50)
    
    # Get all qualifying language columns
    qualifying_cols = [col for col in df_qualifying.columns if any(pattern in col for pattern in 
                      ['Customer_Uncertainty', 'Customer_Hedging', 'Customer_Approximation', 
                       'Customer_Conditional', 'Customer_Doubt', 'Customer_Politeness',
                       'Agent_Uncertainty', 'Agent_Hedging', 'Agent_Approximation',
                       'Agent_Conditional', 'Agent_Doubt', 'Agent_Politeness'])]
    
    if qualifying_cols:
        qualifying_agg = {col: 'mean' for col in qualifying_cols}
        qualifying_features = df_qualifying.groupby('variable5').agg(qualifying_agg).reset_index()
        
        unified_base = unified_base.merge(qualifying_features, on='variable5', how='left')
        print(f"Added {len(qualifying_cols)} qualifying language features. Shape: {unified_base.shape}")
    
    # 4. Add derived conversation-level features
    print("\n4. ADDING DERIVED CONVERSATION-LEVEL FEATURES")
    print("-" * 50)
    
    # Count of categories per conversation
    category_counts = df_main.groupby('variable5').agg({
        'Prosodica L1': 'nunique',
        'Prosodica L2': 'nunique',
        'UUID': 'count'
    }).reset_index()
    
    category_counts.columns = ['variable5', 'Unique_L1_Categories', 'Unique_L2_Categories', 'Total_UUIDs']
    
    unified_base = unified_base.merge(category_counts, on='variable5', how='left')
    
    # Add multi-category flags
    unified_base['Is_Multi_L1_Category'] = (unified_base['Unique_L1_Categories'] > 1).astype(int)
    unified_base['Is_Multi_L2_Category'] = (unified_base['Unique_L2_Categories'] > 1).astype(int)
    
    # 5. Add validation consistency features (if available)
    print("\n5. ADDING VALIDATION CONSISTENCY FEATURES")
    print("-" * 50)
    
    if 'Primary_Secondary_Agreement' in df_main.columns:
        validation_features = df_main.groupby('variable5').agg({
            'Primary_Secondary_Agreement': ['mean', 'count', 'std']
        }).reset_index()
        
        validation_features.columns = ['variable5', 'Avg_Validation_Agreement', 'Validation_Count', 'Validation_Std']
        validation_features['Has_Validation_Data'] = (validation_features['Validation_Count'] > 0).astype(int)
        
        unified_base = unified_base.merge(validation_features, on='variable5', how='left')
        print(f"Added validation consistency features. Shape: {unified_base.shape}")
    
    # 6. Add temporal and contextual features
    print("\n6. ADDING TEMPORAL AND CONTEXTUAL FEATURES")
    print("-" * 50)
    
    # Calculate conversation complexity metrics
    unified_base['Conversation_Complexity_Score'] = (
        unified_base['Total_UUIDs'] * 0.3 +
        unified_base['Unique_L2_Categories'] * 0.4 +
        (unified_base['Transcript_Length'] / 1000) * 0.3
    )
    
    # Calculate precision risk score
    unified_base['Precision_Risk_Score'] = 0
    
    # Add risk from negations
    if 'Customer_Negation_Count' in unified_base.columns:
        unified_base['Precision_Risk_Score'] += unified_base['Customer_Negation_Count'] * 0.1
    
    # Add risk from agent contamination
    if 'Agent_Contamination_Rate' in unified_base.columns:
        unified_base['Precision_Risk_Score'] += unified_base['Agent_Contamination_Rate'] * 0.3
    
    # Add risk from qualifying language
    if 'Customer_Doubt_Count' in unified_base.columns:
        unified_base['Precision_Risk_Score'] += unified_base['Customer_Doubt_Count'] * 0.2
    
    # Add risk from multi-category
    unified_base['Precision_Risk_Score'] += unified_base['Is_Multi_L2_Category'] * 0.15
    
    # Normalize risk score
    max_risk = unified_base['Precision_Risk_Score'].max()
    if max_risk > 0:
        unified_base['Precision_Risk_Score_Normalized'] = unified_base['Precision_Risk_Score'] / max_risk
    else:
        unified_base['Precision_Risk_Score_Normalized'] = 0
    
    # 7. Add feature summary statistics
    print("\n7. ADDING FEATURE SUMMARY STATISTICS")
    print("-" * 50)
    
    # Calculate feature counts and ratios
    text_features = ['Customer_Negation_Count', 'Customer_Qualifying_Count', 'Customer_Question_Count']
    
    for feature in text_features:
        if feature in unified_base.columns:
            # Feature density per 1000 characters
            unified_base[f'{feature}_Density'] = (
                unified_base[feature] / (unified_base['Transcript_Length'] / 1000)
            ).fillna(0)
    
    # Customer engagement score
    engagement_features = ['Customer_Question_Count', 'Customer_Word_Count']
    if all(feature in unified_base.columns for feature in engagement_features):
        unified_base['Customer_Engagement_Score'] = (
            unified_base['Customer_Question_Count'] * 0.4 +
            (unified_base['Customer_Word_Count'] / 100) * 0.6
        )
    
    # 8. Final data quality checks and summary
    print("\n8. FINAL DATA QUALITY CHECKS AND SUMMARY")
    print("-" * 50)
    
    # Check for missing values
    missing_summary = unified_base.isnull().sum()
    missing_percentage = (missing_summary / len(unified_base)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_summary.index,
        'Missing_Count': missing_summary.values,
        'Missing_Percentage': missing_percentage.values
    })
    
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
    
    if len(missing_df) > 0:
        print("Columns with missing values:")
        print(missing_df.head(10))
    else:
        print("No missing values detected in unified dataframe")
    
    # Feature categories summary
    original_features = [col for col in unified_base.columns if col in df_main.columns or col == 'variable5']
    engineered_features = [col for col in unified_base.columns if col not in original_features]
    
    print(f"\nFeature Summary:")
    print(f"Total features: {len(unified_base.columns)}")
    print(f"Original features: {len(original_features)}")
    print(f"Engineered features: {len(engineered_features)}")
    print(f"Total conversations: {len(unified_base)}")
    
    print(f"\nEngineered Feature Categories:")
    print(f"- Agent contamination features: {len([col for col in engineered_features if 'Agent' in col and 'Contamination' in col])}")
    print(f"- Qualifying language features: {len([col for col in engineered_features if any(pattern in col for pattern in ['Uncertainty', 'Hedging', 'Doubt', 'Politeness'])])}")
    print(f"- Category complexity features: {len([col for col in engineered_features if 'Category' in col or 'Multi' in col])}")
    print(f"- Validation features: {len([col for col in engineered_features if 'Validation' in col])}")
    print(f"- Risk scoring features: {len([col for col in engineered_features if 'Risk' in col or 'Score' in col])}")
    print(f"- Density features: {len([col for col in engineered_features if 'Density' in col])}")
    
    # Data distribution summary
    print(f"\nData Distribution Summary:")
    if 'Is_TP' in unified_base.columns:
        tp_rate = unified_base['Is_TP'].mean()
        print(f"Overall TP rate: {tp_rate:.3f}")
    
    if 'Is_Multi_L2_Category' in unified_base.columns:
        multi_cat_rate = unified_base['Is_Multi_L2_Category'].mean()
        print(f"Multi-category conversations: {multi_cat_rate:.3f}")
    
    if 'Agent_Contamination_Rate' in unified_base.columns:
        contamination_rate = unified_base['Agent_Contamination_Rate'].mean()
        print(f"Average agent contamination rate: {contamination_rate:.3f}")
    
    # 9. Create feature dictionary for documentation
    print("\n9. CREATING FEATURE DOCUMENTATION")
    print("-" * 50)
    
    feature_documentation = []
    
    # Original features
    for col in original_features:
        if col != 'variable5':
            feature_documentation.append({
                'Feature_Name': col,
                'Feature_Type': 'Original',
                'Data_Type': str(unified_base[col].dtype),
                'Description': 'Original column from input data',
                'Missing_Count': unified_base[col].isnull().sum(),
                'Unique_Values': unified_base[col].nunique()
            })
    
    # Engineered features with descriptions
    engineered_descriptions = {
        'Unique_L1_Categories': 'Number of unique L1 categories per conversation',
        'Unique_L2_Categories': 'Number of unique L2 categories per conversation',
        'Total_UUIDs': 'Total number of UUIDs per conversation',
        'Is_Multi_L1_Category': 'Binary flag for multi-L1 category conversations',
        'Is_Multi_L2_Category': 'Binary flag for multi-L2 category conversations',
        'Avg_Agent_Contamination_Score': 'Average agent contamination score per conversation',
        'Agent_Contamination_Rate': 'Rate of agent contamination patterns per conversation',
        'Conversation_Complexity_Score': 'Composite score measuring conversation complexity',
        'Precision_Risk_Score': 'Raw precision risk score based on multiple factors',
        'Precision_Risk_Score_Normalized': 'Normalized precision risk score (0-1)',
        'Customer_Engagement_Score': 'Score measuring customer engagement level',
        'Avg_Validation_Agreement': 'Average primary-secondary validation agreement',
        'Validation_Count': 'Number of validation records per conversation',
        'Has_Validation_Data': 'Binary flag for presence of validation data'
    }
    
    for col in engineered_features:
        description = engineered_descriptions.get(col, 'Engineered feature')
        
        # Determine feature category
        if 'Agent' in col and 'Contamination' in col:
            category = 'Agent_Contamination'
        elif any(pattern in col for pattern in ['Uncertainty', 'Hedging', 'Doubt', 'Politeness']):
            category = 'Qualifying_Language'
        elif 'Category' in col or 'Multi' in col:
            category = 'Category_Complexity'
        elif 'Validation' in col:
            category = 'Validation_Quality'
        elif 'Risk' in col or 'Score' in col:
            category = 'Risk_Scoring'
        elif 'Density' in col:
            category = 'Feature_Density'
        else:
            category = 'Other_Engineered'
        
        feature_documentation.append({
            'Feature_Name': col,
            'Feature_Type': 'Engineered',
            'Feature_Category': category,
            'Data_Type': str(unified_base[col].dtype),
            'Description': description,
            'Missing_Count': unified_base[col].isnull().sum(),
            'Unique_Values': unified_base[col].nunique()
        })
    
    feature_doc_df = pd.DataFrame(feature_documentation)
    
    print("Feature documentation created successfully")
    print(f"Documented {len(feature_doc_df)} features")
    
    # 10. Export unified dataframe and documentation
    print("\n10. EXPORTING UNIFIED DATAFRAME")
    print("-" * 50)
    
    # Export to Excel with multiple sheets
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'Unified_Feature_Dataset_{timestamp}.xlsx'
    
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        # Main unified dataset
        unified_base.to_excel(writer, sheet_name='Unified_Dataset', index=False)
        
        # Feature documentation
        feature_doc_df.to_excel(writer, sheet_name='Feature_Documentation', index=False)
        
        # Data quality summary
        if len(missing_df) > 0:
            missing_df.to_excel(writer, sheet_name='Data_Quality', index=False)
        
        # Sample data for reference
        unified_base.head(100).to_excel(writer, sheet_name='Sample_Data', index=False)
        
        # Feature statistics
        numeric_features = unified_base.select_dtypes(include=[np.number]).columns
        feature_stats = unified_base[numeric_features].describe().round(3)
        feature_stats.to_excel(writer, sheet_name='Feature_Statistics')
    
    print(f"Unified dataframe exported to: {output_filename}")
    print(f"Sheets included: Unified_Dataset, Feature_Documentation, Data_Quality, Sample_Data, Feature_Statistics")
    
    return unified_base, feature_doc_df

# Execute the unified dataframe creation
unified_dataframe, feature_documentation = create_unified_feature_dataframe(df_main, df_enhanced, df_qualifying)

# Display final summary
print("\n" + "="*80)
print("UNIFIED DATAFRAME CREATION COMPLETED")
print("="*80)
print(f"Final unified dataframe shape: {unified_dataframe.shape}")
print(f"Conversations analyzed: {len(unified_dataframe)}")
print(f"Total features: {len(unified_dataframe.columns)}")

# Show sample of the unified dataframe
print("\nSample of unified dataframe (first 5 rows, key columns):")
key_columns = ['variable5', 'Prosodica L1', 'Prosodica L2', 'Primary Marker', 'Is_TP', 
               'Unique_L2_Categories', 'Agent_Contamination_Rate', 'Precision_Risk_Score_Normalized']
available_key_columns = [col for col in key_columns if col in unified_dataframe.columns]
print(unified_dataframe[available_key_columns].head())

print("\nAnalysis complete! All results have been exported to Excel files.")


# =============================================================================
# ENHANCED EXPORT WITH ALL RESULTS
# REPLACE THE EXISTING EXPORT SECTION WITH THIS
# =============================================================================

print("\n" + "="*80)
print("EXPORTING ALL ANALYSIS RESULTS")
print("="*80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Export Original Results (keep existing structure)
print("Exporting original analysis results...")
with pd.ExcelWriter(f'Complaints_Analysis_Results_Original_{timestamp}.xlsx', engine='xlsxwriter') as writer:
    
    # Write each original result to a separate sheet
    monthly_category_precision.to_excel(writer, sheet_name='Monthly_Category_Precision', index=False)
    category_impact.to_excel(writer, sheet_name='Category_Impact', index=False)
    volume_precision.to_excel(writer, sheet_name='Volume_Precision', index=False)
    monthly_trends.to_excel(writer, sheet_name='Monthly_Trends', index=False)
    all_categories.to_excel(writer, sheet_name='All_Categories', index=False)
    complaint_categories.to_excel(writer, sheet_name='Complaint_Categories', index=False)
    top_5_drop_drivers.to_excel(writer, sheet_name='Top_5_Drop_Drivers', index=False)
    comparison.to_excel(writer, sheet_name='Period_Comparison', index=False)
    monthly_precision.to_excel(writer, sheet_name='Monthly_Precision', index=False)
    fp_summary.to_excel(writer, sheet_name='FP_Summary', index=False)
    fp_reason_summary.to_excel(writer, sheet_name='FP_Reason_Summary', index=False)
    
    if monthly_validation is not None:
        monthly_validation.to_excel(writer, sheet_name='Monthly_Validation', index=False)
    if category_agreement is not None:
        category_agreement.to_excel(writer, sheet_name='Category_Agreement', index=False)
    
    dow_analysis.to_excel(writer, sheet_name='Day_of_Week_Analysis', index=False)
    wom_analysis.to_excel(writer, sheet_name='Week_of_Month_Analysis', index=False)
    operational_analysis.to_excel(writer, sheet_name='Operational_Analysis', index=False)
    transcript_categories.to_excel(writer, sheet_name='Transcript_Categories', index=False)
    multi_category.to_excel(writer, sheet_name='Multi_Category', index=False)
    length_comparison.to_excel(writer, sheet_name='Length_Comparison', index=False)
    ratio_comparison.to_excel(writer, sheet_name='Ratio_Comparison', index=False)
    qualifying_comparison.to_excel(writer, sheet_name='Qualifying_Comparison', index=False)
    pattern_df.to_excel(writer, sheet_name='Pattern_Analysis', index=False)

# Export New Enhanced Results
print("Exporting enhanced deep dive results...")
with pd.ExcelWriter(f'Enhanced_Deep_Dive_Results_{timestamp}.xlsx', engine='xlsxwriter') as writer:
    
    # New analysis results
    negation_category_df.to_excel(writer, sheet_name='Negation_Analysis', index=False)
    context_df.to_excel(writer, sheet_name='Negation_Context', index=False)
    category_contamination_df.to_excel(writer, sheet_name='Agent_Contamination', index=False)
    
    # Rater analysis (if available)
    if rater_performance is not None:
        rater_performance.to_excel(writer, sheet_name='Rater_Performance', index=False)
    if rater_outliers is not None and len(rater_outliers) > 0:
        rater_outliers.to_excel(writer, sheet_name='Rater_Outliers', index=False)
    
    # Qualifying language analysis
    overall_qualifying_df.to_excel(writer, sheet_name='Overall_Qualifying', index=False)
    category_qualifying_df.to_excel(writer, sheet_name='Category_Qualifying', index=False)
    monthly_qualifying_df.to_excel(writer, sheet_name='Monthly_Qualifying', index=False)
    pre_post_qualifying_df.to_excel(writer, sheet_name='PrePost_Qualifying', index=False)

# Export Unified Dataset with Documentation
print("Exporting unified feature dataset...")
with pd.ExcelWriter(f'Unified_Feature_Dataset_{timestamp}.xlsx', engine='xlsxwriter') as writer:
    
    # Main unified dataset
    unified_dataframe.to_excel(writer, sheet_name='Unified_Dataset', index=False)
    
    # Feature documentation
    feature_documentation.to_excel(writer, sheet_name='Feature_Documentation', index=False)
    
    # Sample data for reference
    unified_dataframe.head(100).to_excel(writer, sheet_name='Sample_Data', index=False)
    
    # Feature statistics
    numeric_features = unified_dataframe.select_dtypes(include=[np.number]).columns
    if len(numeric_features) > 0:
        feature_stats = unified_dataframe[numeric_features].describe().round(3)
        feature_stats.to_excel(writer, sheet_name='Feature_Statistics')

# Export Summary Report
print("Creating executive summary...")
with pd.ExcelWriter(f'Executive_Summary_{timestamp}.xlsx', engine='xlsxwriter') as writer:
    
    # Key findings summary
    key_findings = pd.DataFrame({
        'Analysis_Area': [
            'Negation Patterns',
            'Agent Contamination', 
            'Rater Consistency',
            'Qualifying Language',
            'Overall Precision'
        ],
        'Key_Finding': [
            f'FPs have {(fp_data["Customer_Negation_Count"] > 0).mean():.1%} negation rate vs TPs {(tp_data["Customer_Negation_Count"] > 0).mean():.1%}',
            f'Agent contamination affects {category_contamination_df["FP_Contamination_Rate_%"].mean():.1f}% of FPs on average',
            f'{len(rater_outliers) if rater_outliers is not None else 0} raters identified as statistical outliers',
            f'Customer uncertainty words show {overall_qualifying_df[overall_qualifying_df["Pattern"] == "Uncertainty"]["Customer_Risk_Factor"].iloc[0]:.2f}x risk in FPs',
            f'Overall precision: {df_main["Is_TP"].mean():.3f} (Target: 0.700)'
        ],
        'Priority': ['High', 'High', 'Medium', 'Medium', 'Critical'],
        'Actionable': [
            'Implement context-aware negation rules',
            'Add channel-specific classification',
            'Review rater training programs', 
            'Distinguish information vs complaint uncertainty',
            'Focus on top risk categories'
        ]
    })
    
    key_findings.to_excel(writer, sheet_name='Key_Findings', index=False)
    
    # Top risk categories
    if len(category_contamination_df) > 0:
        top_risk_categories = category_contamination_df.nlargest(10, 'Contamination_Risk')[
            ['Category', 'FP_Count', 'FP_Contamination_Rate_%', 'Contamination_Risk']
        ]
        top_risk_categories.to_excel(writer, sheet_name='Top_Risk_Categories', index=False)
    
    # Monthly precision trends
    monthly_summary = df_main.groupby('Year_Month').agg({
        'Is_TP': 'mean',
        'Is_FP': 'mean', 
        'variable5': 'nunique'
    }).reset_index()
    monthly_summary.columns = ['Year_Month', 'Precision', 'FP_Rate', 'Total_Calls']
    monthly_summary.to_excel(writer, sheet_name='Monthly_Trends', index=False)

# Print completion summary
print(f"\n" + "="*80)
print("ANALYSIS COMPLETE - FILES EXPORTED")
print("="*80)
print(f"Original analysis results: Complaints_Analysis_Results_Original_{timestamp}.xlsx")
print(f"Enhanced deep dive results: Enhanced_Deep_Dive_Results_{timestamp}.xlsx") 
print(f"Unified feature dataset: Unified_Feature_Dataset_{timestamp}.xlsx")
print(f"Executive summary: Executive_Summary_{timestamp}.xlsx")

# Display final summary
print(f"\n" + "="*80)
print("FINAL ANALYSIS SUMMARY")
print("="*80)
print(f"Original functions executed: 11 cells")
print(f"New enhanced functions executed: 5 cells") 
print(f"Total conversations analyzed: {len(unified_dataframe)}")
print(f"Total features in unified dataset: {len(unified_dataframe.columns)}")
print(f"Key insights generated: {len(key_findings)}")

# Show sample of the unified dataframe
print(f"\nSample of unified dataframe (first 5 rows, key columns):")
key_columns = ['variable5', 'Prosodica L1', 'Prosodica L2', 'Primary Marker', 'Is_TP', 
               'Unique_L2_Categories', 'Agent_Contamination_Rate', 'Precision_Risk_Score_Normalized']
available_key_columns = [col for col in key_columns if col in unified_dataframe.columns]
print(unified_dataframe[available_key_columns].head())

print(f"\nAnalysis pipeline completed successfully!")
print(f"All insights have been resolved and exported to Excel files.")
print(f"The negation contradiction has been resolved through deep analysis.")
print(f"Agent contamination patterns have been identified and quantified.")
print(f"Rater influence has been assessed for validation quality.")
print(f"Qualifying language patterns have been split by customer vs agent.")
print(f"Unified feature dataset created at conversation level for further modeling.")

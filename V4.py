# Enhanced Data Preparation with Monthly Tracking Framework
# Modification 1: Add Period Classification and Monthly Analysis Framework

def load_and_prepare_data():
    """Enhanced data preparation with monthly tracking capabilities"""
    
    print("="*80)
    print("ENHANCED DATA PREPARATION WITH MONTHLY TRACKING")
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
    
    # CRITICAL ADDITION: Period Classification for Pre vs Post Analysis
    pre_months = ['2024-10', '2024-11', '2024-12']
    post_months = ['2025-01', '2025-02', '2025-03']
    
    df_main['Period'] = df_main['Year_Month'].apply(
        lambda x: 'Pre' if str(x) in pre_months else 'Post' if str(x) in post_months else 'Other'
    )
    
    print(f"Period Classification:")
    print(f"  Pre Period (Oct-Dec 2024): {(df_main['Period'] == 'Pre').sum()} records")
    print(f"  Post Period (Jan-Mar 2025): {(df_main['Period'] == 'Post').sum()} records")
    print(f"  Other Periods: {(df_main['Period'] == 'Other').sum()} records")
    
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
    if df_rules_filtered is not None and 'begin_date' in df_rules_filtered.columns:
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
    
    print(f"Enhanced data preparation completed. Final dataset shape: {df_main.shape}")
    
    return df_main, df_validation, df_rules_filtered

# Monthly Analysis Framework
def create_monthly_insight_table(df, insight_name, metric_columns, description):
    """
    Create monthly breakdown table for any metric using the original approach
    
    Parameters:
    - df: DataFrame with the data
    - insight_name: Name of the insight being analyzed
    - metric_columns: List of column names to analyze
    - description: Description of what the analysis measures
    """
    
    print(f"\n" + "="*80)
    print(f"MONTHLY BREAKDOWN: {insight_name.upper()}")
    print(f"Analysis Focus: {description}")
    print("="*80)
    
    # Month mapping for proper ordering
    month_mapping = {
        '2024-10': "October'24",
        '2024-11': "November'24", 
        '2024-12': "December'24",
        '2025-01': "January'25",
        '2025-02': "February'25",
        '2025-03': "March'25"
    }
    
    months_order = ['2024-10', '2024-11', '2024-12', '2025-01', '2025-02', '2025-03']
    display_months = [month_mapping.get(m, m) for m in months_order if m in df['Year_Month'].unique()]
    
    # Create results for each metric
    monthly_results = []
    
    for metric in metric_columns:
        # Calculate TP, FP averages, and risk factor for each metric
        for outcome in ['FP_Avg', 'TP_Avg', 'Risk_Factor']:
            row_data = {'Insight': f"{metric}_{outcome}"}
            
            for month in months_order:
                if month in df['Year_Month'].unique():
                    month_data = df[df['Year_Month'] == month]
                    
                    if outcome == 'FP_Avg':
                        # Average for FPs in this month
                        fp_data = month_data[month_data['Primary Marker'] == 'FP']
                        value = fp_data[metric].mean() if len(fp_data) > 0 else 0
                        
                    elif outcome == 'TP_Avg':
                        # Average for TPs in this month
                        tp_data = month_data[month_data['Primary Marker'] == 'TP']
                        value = tp_data[metric].mean() if len(tp_data) > 0 else 0
                        
                    else:  # Risk_Factor
                        # Calculate risk factor for this month (FP_Avg / TP_Avg)
                        fp_data = month_data[month_data['Primary Marker'] == 'FP']
                        tp_data = month_data[month_data['Primary Marker'] == 'TP']
                        
                        fp_avg = fp_data[metric].mean() if len(fp_data) > 0 else 0
                        tp_avg = tp_data[metric].mean() if len(tp_data) > 0 else 0
                        
                        value = fp_avg / (tp_avg + 0.001) if tp_avg > 0 else 0
                    
                    row_data[month_mapping[month]] = round(value, 3)
                else:
                    row_data[month_mapping.get(month, month)] = 0.000
            
            # Calculate total/overall averages
            if outcome == 'FP_Avg':
                overall_fp = df[df['Primary Marker'] == 'FP']
                total_value = overall_fp[metric].mean() if len(overall_fp) > 0 else 0
            elif outcome == 'TP_Avg':
                overall_tp = df[df['Primary Marker'] == 'TP']
                total_value = overall_tp[metric].mean() if len(overall_tp) > 0 else 0
            else:  # Risk_Factor
                overall_fp = df[df['Primary Marker'] == 'FP']
                overall_tp = df[df['Primary Marker'] == 'TP']
                fp_avg = overall_fp[metric].mean() if len(overall_fp) > 0 else 0
                tp_avg = overall_tp[metric].mean() if len(overall_tp) > 0 else 0
                total_value = fp_avg / (tp_avg + 0.001) if tp_avg > 0 else 0
            
            row_data['Total'] = round(total_value, 3)
            monthly_results.append(row_data)
    
    # Create DataFrame
    contingency_df = pd.DataFrame(monthly_results)
    
    print(f"Monthly Breakdown Table for {insight_name}:")
    print(contingency_df.to_string(index=False))
    
    # Period Comparison (Pre vs Post)
    print(f"\nPeriod Comparison (Pre vs Post):")
    
    pre_data = df[df['Period'] == 'Pre']
    post_data = df[df['Period'] == 'Post']
    
    for metric in metric_columns:
        print(f"\n{metric.upper()} Analysis:")
        
        # Pre period averages
        pre_fp = pre_data[pre_data['Primary Marker'] == 'FP']
        pre_tp = pre_data[pre_data['Primary Marker'] == 'TP']
        pre_fp_avg = pre_fp[metric].mean() if len(pre_fp) > 0 else 0
        pre_tp_avg = pre_tp[metric].mean() if len(pre_tp) > 0 else 0
        pre_risk = pre_fp_avg / (pre_tp_avg + 0.001) if pre_tp_avg > 0 else 0
        
        # Post period averages
        post_fp = post_data[post_data['Primary Marker'] == 'FP']
        post_tp = post_data[post_data['Primary Marker'] == 'TP']
        post_fp_avg = post_fp[metric].mean() if len(post_fp) > 0 else 0
        post_tp_avg = post_tp[metric].mean() if len(post_tp) > 0 else 0
        post_risk = post_fp_avg / (post_tp_avg + 0.001) if post_tp_avg > 0 else 0
        
        comparison_df = pd.DataFrame({
            'Metric': [f'{metric}_FP_Avg', f'{metric}_TP_Avg', f'{metric}_Risk_Factor'],
            'Pre Period': [pre_fp_avg, pre_tp_avg, pre_risk],
            'Post Period': [post_fp_avg, post_tp_avg, post_risk],
            'Change': [
                post_fp_avg - pre_fp_avg,
                post_tp_avg - pre_tp_avg, 
                post_risk - pre_risk
            ],
            '% Change': [
                ((post_fp_avg - pre_fp_avg) / pre_fp_avg * 100) if pre_fp_avg > 0 else 0,
                ((post_tp_avg - pre_tp_avg) / pre_tp_avg * 100) if pre_tp_avg > 0 else 0,
                ((post_risk - pre_risk) / pre_risk * 100) if pre_risk > 0 else 0
            ]
        })
        
        print(comparison_df.round(3))
        
        # Insights
        fp_change = post_fp_avg - pre_fp_avg
        tp_change = post_tp_avg - pre_tp_avg
        risk_change = post_risk - pre_risk
        
        if abs(fp_change) > 0.5:
            direction = "increased" if fp_change > 0 else "decreased"
            print(f"  - FP {metric} {direction} significantly ({fp_change:+.2f})")
        
        if abs(tp_change) > 0.5:
            direction = "increased" if tp_change > 0 else "decreased"  
            print(f"  - TP {metric} {direction} significantly ({tp_change:+.2f})")
        
        if abs(risk_change) > 0.1:
            direction = "worsened" if risk_change > 0 else "improved"
            print(f"  - Risk factor {direction}: {post_risk:.3f} vs {pre_risk:.3f}")
    
    return contingency_df

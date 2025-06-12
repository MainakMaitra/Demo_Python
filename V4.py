# =============================================================================
# DEEP DIVE ANALYSIS
# =============================================================================


# Enhanced Deep Dive Analysis with Monthly Breakdown
# Modification 3: Add monthly tracking to all deep dive analyses

def fp_pattern_analysis_enhanced(df):
    """Enhanced false positive pattern analysis with monthly breakdown"""
    
    print("1.1 Group All FPs by Category and Month")
    
    # Original FP grouping analysis (unchanged)
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
    
    # NEW: Core Negation Analysis - The Key Insight!
    print("\n1.1.1 CORE INSIGHT: Monthly Negation Pattern Analysis")
    create_monthly_insight_table(
        df,
        "Context-Insensitive Negation Handling",
        ['Customer_Negation_Count'],
        "CRITICAL: Monthly tracking of TP vs FP negation patterns - the $2.4M insight"
    )
    
    print("\n1.2 Create SRSRWI (Sample Review Spreadsheet) for Top FP Categories")
    
    # Original SRSRWI analysis (unchanged)
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
    
    # Original automated pattern identification (unchanged)
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
    
    # NEW: Monthly Qualifying Language Analysis
    create_monthly_insight_table(
        df,
        "Qualifying Language Patterns",
        ['Customer_Qualifying_Count'],
        "Monthly qualifying language analysis: Uncertainty indicators in TPs vs FPs"
    )
    
    print("\n1.4 Categorize FP Reasons")
    
    # Original FP categorization (unchanged)
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
    
    # NEW: Monthly Agent Conversation Analysis
    create_monthly_insight_table(
        df,
        "Agent Explanation Contamination",
        ['Customer_Agent_Ratio'],
        "Monthly agent-customer ratio analysis: Agent explanations triggering false positives"
    )
    
    return fp_summary, srsrwi_df, fp_patterns, fp_reason_summary

def validation_process_assessment_enhanced(df):
    """Enhanced validation process assessment with monthly breakdown"""
    
    print("2.1 Primary vs Secondary Validation Agreement Rates")
    
    # Original overall agreement analysis (unchanged)
    secondary_data = df[df['Has_Secondary_Validation']].copy()
    
    if len(secondary_data) > 0:
        overall_agreement = secondary_data['Primary_Secondary_Agreement'].mean()
        total_secondary = len(secondary_data)
        
        print(f"Overall Validation Metrics:")
        print(f"  Records with secondary validation: {total_secondary} ({total_secondary/len(df)*100:.1f}%)")
        print(f"  Primary-Secondary agreement rate: {overall_agreement:.3f}")
        
        print("\n2.2 Categories with High Disagreement Rates")
        
        # Original category-wise agreement (unchanged)
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
        
        # Original monthly validation trends (unchanged)
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
        
        # NEW: Monthly Validation Quality Analysis
        if len(secondary_data) > 0:
            create_monthly_insight_table(
                secondary_data,
                "Validation Process Quality",
                ['Primary_Secondary_Agreement'],
                "Monthly validation agreement tracking: Process consistency over time"
            )
        
        print("\n2.4 Validation Guidelines and Reviewer Changes Assessment")
        
        # Original reviewer consistency analysis (unchanged)
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

def temporal_analysis_enhanced(df):
    """Enhanced temporal pattern analysis with monthly breakdown"""
    
    print("3.1 FP Rates by Day of Week")
    
    # Original day of week analysis (unchanged)
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
    
    # Original week of month analysis (unchanged)
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
    
    # NEW: Monthly Temporal Pattern Analysis
    create_monthly_insight_table(
        df,
        "Temporal Operational Patterns",
        ['Customer_Question_Count', 'Customer_Exclamation_Count'],
        "Monthly temporal analysis: Question and exclamation patterns indicating customer sentiment"
    )
    
    print("\n3.3 Operational Changes Coinciding with Precision Drops")
    
    # Original operational analysis (unchanged)
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

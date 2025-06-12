# Enhanced Macro Level Analysis with Monthly Breakdown
# Modification 2: Add monthly tracking to all macro-level analyses

def analyze_precision_drop_patterns_enhanced(df):
    """Enhanced precision drop pattern analysis with monthly breakdown"""
    
    print("1.1 Calculate MoM Precision Changes for Each Category")
    
    # Original analysis (unchanged)
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
    
    # NEW: Monthly Precision Tracking Analysis
    create_monthly_insight_table(
        df, 
        "Overall Precision Patterns",
        ['Is_TP'],  # We'll analyze precision through TP rates
        "Monthly precision tracking: TP rates for overall system performance"
    )
    
    print("\n1.2 Identify Categories Contributing Most to Overall Decline")
    
    # Original category impact analysis (unchanged)
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
    
    # NEW: Monthly Volume Impact Analysis
    create_monthly_insight_table(
        df,
        "Volume Impact Patterns", 
        ['Is_FP'],  # Analyze FP volume patterns monthly
        "Monthly FP volume tracking: How false positive rates change over time"
    )
    
    print("\n1.3 Determine Drop Characteristics")
    
    # Original concentration analysis (unchanged)
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
    
    # Original new category analysis (unchanged)
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

def analyze_volume_vs_performance_enhanced(df):
    """Enhanced volume vs performance analysis with monthly breakdown"""
    
    print("2.1 High-Volume vs Low Precision Correlation")
    
    # Original volume-precision analysis (unchanged)
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
    
    # Original monthly volume and precision trends (unchanged)
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
    
    # Original seasonal analysis (unchanged)
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
    
    # NEW: Monthly Transcript Length Analysis
    create_monthly_insight_table(
        df,
        "Transcript Length Patterns",
        ['Transcript_Length'],
        "Monthly transcript length analysis: How conversation length affects TP vs FP classification"
    )
    
    return volume_precision, monthly_trends

def query_performance_review_enhanced(df, df_rules):
    """Enhanced query performance review with monthly breakdown"""
    
    print("3.1 Calculate Precision for All Complaint Categories")
    
    # Original all category precision (unchanged)
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
    
    # NEW: Monthly Category Performance Analysis
    if len(complaint_categories) > 0:
        # Focus on top problematic categories for monthly analysis
        top_problem_categories = complaint_categories.nsmallest(3, 'Precision')['L2_Category'].tolist()
        
        if len(top_problem_categories) > 0:
            print(f"\nMonthly Analysis for Top 3 Worst Performing Categories:")
            print(f"Categories: {', '.join(top_problem_categories)}")
            
            # Filter data to focus on these categories
            category_subset = df[df['Prosodica L2'].isin(top_problem_categories)]
            
            if len(category_subset) > 0:
                create_monthly_insight_table(
                    category_subset,
                    "Worst Performing Categories",
                    ['Is_TP', 'Is_FP'],
                    "Monthly tracking of worst performing complaint categories"
                )
    
    return all_categories, complaint_categories, top_5_drop_drivers

def pattern_detection_analysis_enhanced(df):
    """Enhanced pattern detection analysis with monthly breakdown"""
    
    print("4.1 Problem vs Non-Problem Months Comparison")
    
    # Original problem month analysis (unchanged)
    unique_months = df['Year_Month'].dropna().unique()
    
    # Convert to strings and filter out any remaining NaN/None values
    valid_months = []
    for month in unique_months:
        if pd.notna(month) and month is not None:
            valid_months.append(str(month))
    
    # Sort the valid months
    all_months = sorted(valid_months)
    
    if len(all_months) >= 4:
        problem_months = all_months[-2:]  # Last 2 months
        normal_months = all_months[:-2]   # Earlier months
    elif len(all_months) >= 2:
        problem_months = all_months[-1:]  # Last 1 month
        normal_months = all_months[:-1]   # Earlier months
    else:
        print("Insufficient months for comparison")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"Problem months: {problem_months}")
    print(f"Normal months: {normal_months}")
    
    # Compare performance
    problem_data = df[df['Year_Month'].astype(str).isin(problem_months)]
    normal_data = df[df['Year_Month'].astype(str).isin(normal_months)]
    
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
    
    # NEW: Monthly Pattern Analysis
    create_monthly_insight_table(
        df,
        "Overall Pattern Detection",
        ['Customer_Negation_Count', 'Transcript_Length'],
        "Monthly pattern detection: Key indicators for problem vs normal periods"
    )
    
    print("\n4.2 Sudden vs Gradual Drop Analysis")
    
    # Original monthly precision trend analysis (unchanged)
    monthly_precision = df.groupby('Year_Month').agg({
        'Is_TP': ['sum', 'count']
    }).reset_index()
    
    monthly_precision.columns = ['Year_Month', 'TPs', 'Total']
    monthly_precision['Precision'] = monthly_precision['TPs'] / monthly_precision['Total']
    
    # Convert Year_Month to string for consistent sorting
    monthly_precision['Year_Month'] = monthly_precision['Year_Month'].astype(str)
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

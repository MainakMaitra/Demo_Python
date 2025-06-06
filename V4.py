def pattern_detection_analysis(df):
    """Compare problem vs non-problem periods"""
    
    print("4.1 Problem vs Non-Problem Months Comparison")
    
    # Define problem months (assuming recent months have issues)
    # FIXED: Handle mixed data types in Year_Month column
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
    
    print("\n4.2 Sudden vs Gradual Drop Analysis")
    
    # Monthly precision trend - ALSO FIXED
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

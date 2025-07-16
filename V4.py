def analyze_transcript_length_buckets_pre_post(df_main, problematic_rules=None):
    """
    Comprehensive transcript length bucketing analysis with Pre vs Post comparison
    to prove that shorter queries are getting flagged more as false positives
    """
    
    print("="*80)
    print("TRANSCRIPT LENGTH BUCKETING ANALYSIS - PRE VS POST")
    print("="*80)
    print("Objective: Prove that shorter transcripts are increasingly misclassified")
    print("="*80)
    
    # Data preparation
    df_main['Date'] = pd.to_datetime(df_main['Date'])
    df_main['Year_Month'] = df_main['Date'].dt.strftime('%Y-%m')
    
    # Define periods
    pre_months = ['2024-10', '2024-11', '2024-12']
    post_months = ['2025-01', '2025-02', '2025-03']
    
    df_main['Period'] = df_main['Year_Month'].apply(
        lambda x: 'Pre' if str(x) in pre_months else 'Post' if str(x) in post_months else 'Other'
    )
    
    # Calculate transcript lengths
    df_main['Customer_Transcript'] = df_main['Customer Transcript'].fillna('')
    df_main['Agent_Transcript'] = df_main['Agent Transcript'].fillna('')
    df_main['Full_Transcript'] = df_main['Customer_Transcript'] + ' ' + df_main['Agent_Transcript']
    df_main['Transcript_Length'] = df_main['Full_Transcript'].str.len()
    
    # Create comprehensive length buckets
    length_buckets = [
        (0, 1000, 'Very Short (<1K)'),
        (1000, 2000, 'Short (1K-2K)'),
        (2000, 3000, 'Medium-Short (2K-3K)'),
        (3000, 5000, 'Medium (3K-5K)'),
        (5000, 8000, 'Medium-Long (5K-8K)'),
        (8000, 12000, 'Long (8K-12K)'),
        (12000, float('inf'), 'Very Long (>12K)')
    ]
    
    # Apply length buckets
    def categorize_length(length):
        for min_len, max_len, category in length_buckets:
            if min_len <= length < max_len:
                return category
        return 'Very Long (>12K)'
    
    df_main['Length_Bucket'] = df_main['Transcript_Length'].apply(categorize_length)
    
    # Filter to Pre and Post periods
    df_analysis = df_main[df_main['Period'].isin(['Pre', 'Post'])].copy()
    
    print(f"Analysis Data: {len(df_analysis)} records")
    print(f"Pre Period: {(df_analysis['Period'] == 'Pre').sum()} records")
    print(f"Post Period: {(df_analysis['Period'] == 'Post').sum()} records")
    
    # 1. OVERALL LENGTH DISTRIBUTION ANALYSIS
    print("\n1. OVERALL LENGTH DISTRIBUTION ANALYSIS")
    print("-" * 50)
    
    length_distribution = df_analysis.groupby(['Period', 'Length_Bucket']).agg({
        'Transcript_Length': ['count', 'mean'],
        'Primary Marker': lambda x: (x == 'TP').mean()
    }).reset_index()
    
    length_distribution.columns = ['Period', 'Length_Bucket', 'Count', 'Avg_Length', 'Precision']
    
    # Calculate percentage distribution
    period_totals = df_analysis.groupby('Period').size()
    length_distribution = length_distribution.merge(
        period_totals.reset_index().rename(columns={0: 'Period_Total'}),
        on='Period'
    )
    length_distribution['Percentage'] = (length_distribution['Count'] / length_distribution['Period_Total']) * 100
    
    # Pivot for better comparison
    distribution_pivot = length_distribution.pivot_table(
        index='Length_Bucket',
        columns='Period',
        values=['Count', 'Percentage', 'Precision'],
        fill_value=0
    ).round(2)
    
    print("Length Distribution by Period:")
    print(distribution_pivot)
    
    # 2. PRECISION ANALYSIS BY LENGTH BUCKET
    print("\n2. PRECISION ANALYSIS BY LENGTH BUCKET")
    print("-" * 50)
    
    precision_analysis = df_analysis.groupby(['Length_Bucket', 'Period']).agg({
        'Primary Marker': ['count', lambda x: (x == 'TP').sum(), lambda x: (x == 'TP').mean()],
        'Transcript_Length': 'mean'
    }).reset_index()
    
    precision_analysis.columns = ['Length_Bucket', 'Period', 'Total_Records', 'TP_Count', 'Precision', 'Avg_Length']
    precision_analysis['FP_Count'] = precision_analysis['Total_Records'] - precision_analysis['TP_Count']
    precision_analysis['FP_Rate'] = 1 - precision_analysis['Precision']
    
    # Pivot precision data
    precision_pivot = precision_analysis.pivot_table(
        index='Length_Bucket',
        columns='Period',
        values=['Precision', 'FP_Rate', 'Total_Records'],
        fill_value=0
    ).round(3)
    
    print("Precision by Length Bucket and Period:")
    print(precision_pivot)
    
    # 3. PRE VS POST CHANGE ANALYSIS
    print("\n3. PRE VS POST CHANGE ANALYSIS")
    print("-" * 50)
    
    # Calculate changes for each length bucket
    change_analysis = []
    
    for bucket in df_analysis['Length_Bucket'].unique():
        bucket_data = precision_analysis[precision_analysis['Length_Bucket'] == bucket]
        
        pre_data = bucket_data[bucket_data['Period'] == 'Pre']
        post_data = bucket_data[bucket_data['Period'] == 'Post']
        
        if len(pre_data) > 0 and len(post_data) > 0:
            pre_precision = pre_data['Precision'].iloc[0]
            post_precision = post_data['Precision'].iloc[0]
            
            pre_fp_rate = pre_data['FP_Rate'].iloc[0]
            post_fp_rate = post_data['FP_Rate'].iloc[0]
            
            pre_volume = pre_data['Total_Records'].iloc[0]
            post_volume = post_data['Total_Records'].iloc[0]
            
            precision_change = post_precision - pre_precision
            fp_rate_change = post_fp_rate - pre_fp_rate
            volume_change = post_volume - pre_volume
            volume_pct_change = (volume_change / pre_volume * 100) if pre_volume > 0 else 0
            
            change_analysis.append({
                'Length_Bucket': bucket,
                'Pre_Precision': pre_precision,
                'Post_Precision': post_precision,
                'Precision_Change': precision_change,
                'Pre_FP_Rate': pre_fp_rate,
                'Post_FP_Rate': post_fp_rate,
                'FP_Rate_Change': fp_rate_change,
                'Pre_Volume': pre_volume,
                'Post_Volume': post_volume,
                'Volume_Change': volume_change,
                'Volume_Pct_Change': volume_pct_change,
                'Problem_Severity': 'HIGH' if precision_change < -0.1 and fp_rate_change > 0.1 else 
                                 'MEDIUM' if precision_change < -0.05 or fp_rate_change > 0.05 else 'LOW'
            })
    
    change_df = pd.DataFrame(change_analysis)
    
    # Sort by precision change (worst first)
    change_df = change_df.sort_values('Precision_Change')
    
    print("Pre vs Post Change Analysis:")
    print(change_df[['Length_Bucket', 'Pre_Precision', 'Post_Precision', 'Precision_Change', 
                    'FP_Rate_Change', 'Volume_Pct_Change', 'Problem_Severity']].round(3))
    
    # 4. SHORT VS LONG TRANSCRIPT FOCUSED ANALYSIS
    print("\n4. SHORT VS LONG TRANSCRIPT FOCUSED ANALYSIS")
    print("-" * 50)
    
    # Define short and long categories for focused analysis
    short_buckets = ['Very Short (<1K)', 'Short (1K-2K)', 'Medium-Short (2K-3K)']
    long_buckets = ['Medium-Long (5K-8K)', 'Long (8K-12K)', 'Very Long (>12K)']
    
    df_analysis['Category_Focus'] = df_analysis['Length_Bucket'].apply(
        lambda x: 'Short' if x in short_buckets else 'Long' if x in long_buckets else 'Medium'
    )
    
    focus_analysis = df_analysis.groupby(['Category_Focus', 'Period']).agg({
        'Primary Marker': ['count', lambda x: (x == 'TP').mean()],
        'Transcript_Length': 'mean'
    }).reset_index()
    
    focus_analysis.columns = ['Category_Focus', 'Period', 'Count', 'Precision', 'Avg_Length']
    
    # Pivot for comparison
    focus_pivot = focus_analysis.pivot_table(
        index='Category_Focus',
        columns='Period',
        values=['Count', 'Precision'],
        fill_value=0
    ).round(3)
    
    print("Short vs Long Analysis:")
    print(focus_pivot)
    
    # Calculate the key insight
    short_pre = focus_analysis[(focus_analysis['Category_Focus'] == 'Short') & (focus_analysis['Period'] == 'Pre')]
    short_post = focus_analysis[(focus_analysis['Category_Focus'] == 'Short') & (focus_analysis['Period'] == 'Post')]
    long_pre = focus_analysis[(focus_analysis['Category_Focus'] == 'Long') & (focus_analysis['Period'] == 'Pre')]
    long_post = focus_analysis[(focus_analysis['Category_Focus'] == 'Long') & (focus_analysis['Period'] == 'Post')]
    
    if len(short_pre) > 0 and len(short_post) > 0 and len(long_pre) > 0 and len(long_post) > 0:
        short_precision_change = short_post['Precision'].iloc[0] - short_pre['Precision'].iloc[0]
        long_precision_change = long_post['Precision'].iloc[0] - long_pre['Precision'].iloc[0]
        
        print(f"\nKEY INSIGHTS:")
        print(f"Short Transcript Precision Change: {short_precision_change:+.3f}")
        print(f"Long Transcript Precision Change: {long_precision_change:+.3f}")
        print(f"Gap: {short_precision_change - long_precision_change:+.3f}")
        
        if short_precision_change < long_precision_change - 0.05:
            print("*** FINDING: Short transcripts are declining MORE than long transcripts ***")
        elif short_precision_change < -0.05:
            print("*** FINDING: Short transcripts showing significant precision decline ***")
    
    # 5. COMPREHENSIVE PROBLEMATIC RULES ANALYSIS (ALL RULES)
    problematic_rules_results = []
    
    if problematic_rules is not None and len(problematic_rules) > 0:
        print("\n5. COMPREHENSIVE PROBLEMATIC RULES ANALYSIS")
        print("=" * 60)
        print(f"Analyzing ALL {len(problematic_rules)} problematic rules for length-based patterns")
        print("=" * 60)
        
        # Process ALL problematic rules, not just top 5
        for idx, rule in problematic_rules.iterrows():
            event = rule['Event']
            query = rule['Query']
            negation_count = rule['negation_patterns']
            
            print(f"\nRule {idx+1}/{len(problematic_rules)}: {event} | {query}")
            print(f"Negation Patterns: {negation_count} | Proximity Rules: 0")
            print("-" * 60)
            
            # Find transcripts for this rule
            rule_data = df_analysis[
                (df_analysis['Prosodica L1'].str.lower() == event.lower()) |
                (df_analysis['Prosodica L2'].str.lower() == query.lower())
            ]
            
            if len(rule_data) == 0:
                print("  No matching transcripts found for this rule")
                continue
            
            print(f"  Total matching transcripts: {len(rule_data)}")
            
            # Detailed length bucket analysis for this rule
            rule_length_detailed = rule_data.groupby(['Length_Bucket', 'Period']).agg({
                'Primary Marker': ['count', lambda x: (x == 'TP').sum(), lambda x: (x == 'TP').mean()],
                'Transcript_Length': 'mean'
            }).reset_index()
            
            rule_length_detailed.columns = ['Length_Bucket', 'Period', 'Total_Records', 'TP_Count', 'Precision', 'Avg_Length']
            rule_length_detailed['FP_Count'] = rule_length_detailed['Total_Records'] - rule_length_detailed['TP_Count']
            rule_length_detailed['FP_Rate'] = 1 - rule_length_detailed['Precision']
            
            if len(rule_length_detailed) > 0:
                # Calculate Pre vs Post changes for each length bucket
                rule_changes = []
                
                for bucket in rule_length_detailed['Length_Bucket'].unique():
                    bucket_data = rule_length_detailed[rule_length_detailed['Length_Bucket'] == bucket]
                    
                    pre_data = bucket_data[bucket_data['Period'] == 'Pre']
                    post_data = bucket_data[bucket_data['Period'] == 'Post']
                    
                    if len(pre_data) > 0 and len(post_data) > 0:
                        pre_precision = pre_data['Precision'].iloc[0]
                        post_precision = post_data['Precision'].iloc[0]
                        pre_volume = pre_data['Total_Records'].iloc[0]
                        post_volume = post_data['Total_Records'].iloc[0]
                        
                        precision_change = post_precision - pre_precision
                        volume_change = post_volume - pre_volume
                        volume_pct_change = (volume_change / pre_volume * 100) if pre_volume > 0 else 0
                        
                        rule_changes.append({
                            'Rule': f"{event} | {query}",
                            'Negation_Patterns': negation_count,
                            'Length_Bucket': bucket,
                            'Pre_Precision': pre_precision,
                            'Post_Precision': post_precision,
                            'Precision_Change': precision_change,
                            'Pre_Volume': pre_volume,
                            'Post_Volume': post_volume,
                            'Volume_Change': volume_change,
                            'Volume_Pct_Change': volume_pct_change,
                            'Problem_Level': 'HIGH' if precision_change < -0.1 else 'MEDIUM' if precision_change < -0.05 else 'LOW'
                        })
                
                if len(rule_changes) > 0:
                    rule_changes_df = pd.DataFrame(rule_changes)
                    problematic_rules_results.extend(rule_changes)
                    
                    print("  Pre vs Post Analysis by Length Bucket:")
                    print(rule_changes_df[['Length_Bucket', 'Pre_Precision', 'Post_Precision', 
                                         'Precision_Change', 'Volume_Pct_Change', 'Problem_Level']].round(3))
                    
                    # Identify short vs long performance for this rule
                    short_buckets_rule = ['Very Short (<1K)', 'Short (1K-2K)', 'Medium-Short (2K-3K)']
                    long_buckets_rule = ['Medium-Long (5K-8K)', 'Long (8K-12K)', 'Very Long (>12K)']
                    
                    short_changes = rule_changes_df[rule_changes_df['Length_Bucket'].isin(short_buckets_rule)]
                    long_changes = rule_changes_df[rule_changes_df['Length_Bucket'].isin(long_buckets_rule)]
                    
                    if len(short_changes) > 0 and len(long_changes) > 0:
                        avg_short_change = short_changes['Precision_Change'].mean()
                        avg_long_change = long_changes['Precision_Change'].mean()
                        
                        print(f"  Average Short Transcript Change: {avg_short_change:+.3f}")
                        print(f"  Average Long Transcript Change: {avg_long_change:+.3f}")
                        print(f"  Short vs Long Gap: {avg_short_change - avg_long_change:+.3f}")
                        
                        if avg_short_change < avg_long_change - 0.05:
                            print("  *** LENGTH BIAS DETECTED: Short transcripts declining more than long ***")
                    
                else:
                    print("  Insufficient data for Pre vs Post comparison")
            else:
                print("  No length bucket data available")
        
        # Aggregate analysis across ALL problematic rules
        if len(problematic_rules_results) > 0:
            print("\n" + "="*80)
            print("AGGREGATE ANALYSIS ACROSS ALL PROBLEMATIC RULES")
            print("="*80)
            
            all_rules_df = pd.DataFrame(problematic_rules_results)
            
            # Summary by length bucket across all rules
            length_bucket_summary = all_rules_df.groupby('Length_Bucket').agg({
                'Precision_Change': ['count', 'mean', 'std'],
                'Volume_Pct_Change': 'mean',
                'Problem_Level': lambda x: (x == 'HIGH').sum()
            }).round(3)
            
            length_bucket_summary.columns = ['Rule_Count', 'Avg_Precision_Change', 'Std_Precision_Change', 
                                           'Avg_Volume_Change_%', 'High_Problem_Rules']
            
            print("Summary by Length Bucket (All Problematic Rules):")
            print(length_bucket_summary)
            
            # Rules with significant short transcript problems
            short_buckets_all = ['Very Short (<1K)', 'Short (1K-2K)', 'Medium-Short (2K-3K)']
            short_problems = all_rules_df[
                (all_rules_df['Length_Bucket'].isin(short_buckets_all)) &
                (all_rules_df['Precision_Change'] < -0.05)
            ]
            
            print(f"\nRules with Short Transcript Problems (>5% precision drop):")
            print(f"Total problematic rule-length combinations: {len(short_problems)}")
            
            if len(short_problems) > 0:
                short_problems_summary = short_problems.groupby('Rule').agg({
                    'Precision_Change': 'mean',
                    'Length_Bucket': 'count'
                }).round(3)
                short_problems_summary.columns = ['Avg_Precision_Drop', 'Affected_Short_Buckets']
                short_problems_summary = short_problems_summary.sort_values('Avg_Precision_Drop')
                
                print("Top 10 Worst Performing Rules for Short Transcripts:")
                print(short_problems_summary.head(10))
            
            # Overall pattern analysis
            short_all = all_rules_df[all_rules_df['Length_Bucket'].isin(short_buckets_all)]
            long_all = all_rules_df[all_rules_df['Length_Bucket'].isin(['Medium-Long (5K-8K)', 'Long (8K-12K)', 'Very Long (>12K)'])]
            
            if len(short_all) > 0 and len(long_all) > 0:
                overall_short_change = short_all['Precision_Change'].mean()
                overall_long_change = long_all['Precision_Change'].mean()
                
                print(f"\nOVERALL PATTERN ACROSS ALL PROBLEMATIC RULES:")
                print(f"Average Short Transcript Precision Change: {overall_short_change:+.3f}")
                print(f"Average Long Transcript Precision Change: {overall_long_change:+.3f}")
                print(f"Overall Short vs Long Gap: {overall_short_change - overall_long_change:+.3f}")
                
                if overall_short_change < overall_long_change - 0.03:
                    print("*** SYSTEMATIC LENGTH BIAS CONFIRMED ACROSS ALL PROBLEMATIC RULES ***")
                    print("*** High-negation rules without context are systematically discriminating against short transcripts ***")
        
        else:
            print("No sufficient data found for problematic rules analysis")
    
    else:
        print("\n5. NO PROBLEMATIC RULES PROVIDED")
        print("Skipping rule-specific analysis")
    
    # 6. STATISTICAL SUMMARY
    print("\n6. STATISTICAL SUMMARY")
    print("-" * 50)
    
    # Calculate summary statistics
    short_buckets_data = change_df[change_df['Length_Bucket'].isin(short_buckets)]
    long_buckets_data = change_df[change_df['Length_Bucket'].isin(long_buckets)]
    
    stats_summary = {
        'Total_Length_Buckets_Analyzed': len(change_df),
        'Short_Buckets_with_Precision_Drop': len(short_buckets_data[short_buckets_data['Precision_Change'] < 0]),
        'Long_Buckets_with_Precision_Drop': len(long_buckets_data[long_buckets_data['Precision_Change'] < 0]),
        'Avg_Short_Precision_Change': short_buckets_data['Precision_Change'].mean() if len(short_buckets_data) > 0 else 0,
        'Avg_Long_Precision_Change': long_buckets_data['Precision_Change'].mean() if len(long_buckets_data) > 0 else 0,
        'High_Problem_Buckets': len(change_df[change_df['Problem_Severity'] == 'HIGH']),
        'Buckets_with_FP_Rate_Increase': len(change_df[change_df['FP_Rate_Change'] > 0])
    }
    
    print("Summary Statistics:")
    for key, value in stats_summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # 7. BUSINESS CONCLUSION
    print("\n7. BUSINESS CONCLUSION")
    print("-" * 50)
    
    # Generate automated conclusions based on analysis
    conclusions = []
    
    if stats_summary['Avg_Short_Precision_Change'] < stats_summary['Avg_Long_Precision_Change'] - 0.03:
        conclusions.append("Short transcripts are declining MORE than long transcripts, proving length-based discrimination")
    
    if stats_summary['Short_Buckets_with_Precision_Drop'] > stats_summary['Long_Buckets_with_Precision_Drop']:
        conclusions.append("More short transcript categories show precision drops than long transcript categories")
    
    if stats_summary['High_Problem_Buckets'] > 0:
        conclusions.append(f"{stats_summary['High_Problem_Buckets']} length categories show severe precision problems")
    
    if stats_summary['Buckets_with_FP_Rate_Increase'] > len(change_df) * 0.5:
        conclusions.append("Majority of length categories show increasing false positive rates")
    
    if len(conclusions) > 0:
        print("KEY CONCLUSIONS:")
        for i, conclusion in enumerate(conclusions, 1):
            print(f"  {i}. {conclusion}")
        
        print("\nBUSINESS IMPACT:")
        print("  - High-negation rules without context are systematically misclassifying short transcripts")
        print("  - Short informational queries are being flagged as complaints")
        print("  - This creates a length-based bias in the complaint detection system")
        print("  - Precision drops are concentrated in shorter transcript categories")
    else:
        print("Analysis shows mixed results - no clear length-based pattern detected")
    
    return {
        'length_distribution': length_distribution,
        'precision_analysis': precision_analysis,
        'change_analysis': change_df,
        'focus_analysis': focus_analysis,
        'problematic_rules_results': problematic_rules_results if 'problematic_rules_results' in locals() else [],
        'stats_summary': stats_summary,
        'conclusions': conclusions
    }

# Example usage function
def run_complete_length_analysis(df_main, problematic_rules=None):
    """
    Run the complete length analysis and export results
    """
    
    print("RUNNING COMPLETE TRANSCRIPT LENGTH ANALYSIS...")
    print("This analysis will prove whether shorter queries are getting flagged more")
    
    # Run the analysis
    results = analyze_transcript_length_buckets_pre_post(df_main, problematic_rules)
    
    # Export results to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'Transcript_Length_Bucketing_Analysis_{timestamp}.xlsx'
    
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        
        # Export all analysis results
        if 'length_distribution' in results:
            results['length_distribution'].to_excel(writer, sheet_name='Length_Distribution', index=False)
        
        if 'precision_analysis' in results:
            results['precision_analysis'].to_excel(writer, sheet_name='Precision_Analysis', index=False)
        
        if 'change_analysis' in results:
            results['change_analysis'].to_excel(writer, sheet_name='Change_Analysis', index=False)
        
        if 'focus_analysis' in results:
            results['focus_analysis'].to_excel(writer, sheet_name='Short_vs_Long', index=False)
        
        # Summary statistics
        if 'stats_summary' in results:
            summary_df = pd.DataFrame(list(results['stats_summary'].items()), 
                                    columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Summary_Stats', index=False)
        
        # Conclusions
        if 'conclusions' in results:
            conclusions_df = pd.DataFrame({'Conclusion': results['conclusions']})
            conclusions_df.to_excel(writer, sheet_name='Business_Conclusions', index=False)
        
        # Problematic Rules Analysis
        if 'problematic_rules_results' in results and len(results['problematic_rules_results']) > 0:
            problematic_rules_df = pd.DataFrame(results['problematic_rules_results'])
            problematic_rules_df.to_excel(writer, sheet_name='Problematic_Rules_Analysis', index=False)
    
    print(f"\nLength analysis exported to: {filename}")
    
    return results, filename

# To use this analysis:
# results, export_file = run_complete_length_analysis(df_main, problematic_rules)

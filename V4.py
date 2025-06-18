def analyze_query_rule_evolution(df_main, df_rules):
    """Analyze how query modifications correlate with precision changes"""
    
    if df_rules is None or 'begin_date' not in df_rules.columns:
        print("Query rules data not available or missing begin_date column")
        return pd.DataFrame()
    
    # Track when each query was last modified
    query_modifications = df_rules.groupby(['Event', 'Query']).agg({
        'begin_date': ['min', 'max', 'count']
    }).reset_index()
    
    # Flatten column names
    query_modifications.columns = ['Event', 'Query', 'begin_date_min', 'begin_date_max', 'modification_count']
    
    results = []
    
    # Correlate query changes with precision drops
    for idx, row in query_modifications.iterrows():
        event = row['Event']
        query = row['Query']
        begin_date_max = row['begin_date_max']
        
        if pd.isna(begin_date_max):
            continue
            
        # Find precision before and after query modification
        before_data = df_main[
            (df_main['Prosodica L1'] == event) & 
            (df_main['Prosodica L2'] == query) &
            (df_main['Date'] < begin_date_max)
        ]
        
        after_data = df_main[
            (df_main['Prosodica L1'] == event) & 
            (df_main['Prosodica L2'] == query) &
            (df_main['Date'] >= begin_date_max)
        ]
        
        if len(before_data) >= 10 and len(after_data) >= 10:
            before_precision = before_data['Is_TP'].mean()
            after_precision = after_data['Is_TP'].mean()
            precision_change = after_precision - before_precision
            
            # Calculate volume changes
            before_volume = len(before_data)
            after_volume = len(after_data)
            volume_change = after_volume - before_volume
            
            # Calculate FP rate changes
            before_fp_rate = before_data['Is_FP'].mean()
            after_fp_rate = after_data['Is_FP'].mean()
            fp_rate_change = after_fp_rate - before_fp_rate
            
            results.append({
                'Event': event,
                'Query': query,
                'Modification_Date': begin_date_max,
                'Before_Precision': before_precision,
                'After_Precision': after_precision,
                'Precision_Change': precision_change,
                'Before_Volume': before_volume,
                'After_Volume': after_volume,
                'Volume_Change': volume_change,
                'FP_Rate_Change': fp_rate_change,
                'Modification_Count': row['modification_count']
            })
            
            print(f"{query}: {precision_change:+.3f} precision change after modification on {begin_date_max}")
    
    if results:
        results_df = pd.DataFrame(results).sort_values('Precision_Change')
        
        print("\n" + "="*60)
        print("QUERY MODIFICATION IMPACT ANALYSIS")
        print("="*60)
        
        # Queries that got worse after modification
        worse_queries = results_df[results_df['Precision_Change'] < -0.05]
        if len(worse_queries) > 0:
            print("\nQueries that WORSENED after modification (>5% drop):")
            print(worse_queries[['Query', 'Before_Precision', 'After_Precision', 'Precision_Change', 'FP_Rate_Change']].round(3))
        
        # Queries that improved after modification
        better_queries = results_df[results_df['Precision_Change'] > 0.05]
        if len(better_queries) > 0:
            print("\nQueries that IMPROVED after modification (>5% gain):")
            print(better_queries[['Query', 'Before_Precision', 'After_Precision', 'Precision_Change', 'FP_Rate_Change']].round(3))
        
        # Summary statistics
        print("\nSummary Statistics:")
        print(f"Total queries analyzed: {len(results_df)}")
        print(f"Queries that worsened: {len(worse_queries)} ({len(worse_queries)/len(results_df)*100:.1f}%)")
        print(f"Queries that improved: {len(better_queries)} ({len(better_queries)/len(results_df)*100:.1f}%)")
        print(f"Average precision change: {results_df['Precision_Change'].mean():.3f}")
        
        # Correlation analysis
        if len(results_df) > 3:
            print("\nCorrelation Analysis:")
            print(f"Correlation between modification count and precision change: {results_df['Modification_Count'].corr(results_df['Precision_Change']):.3f}")
            print(f"Correlation between volume change and precision change: {results_df['Volume_Change'].corr(results_df['Precision_Change']):.3f}")
        
        return results_df
    else:
        print("Insufficient data for query evolution analysis")
        return pd.DataFrame()

# Call the function
query_evolution_results = analyze_query_rule_evolution(df_main, df_rules)

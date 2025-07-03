def create_compact_problematic_rules_view_pivot(df_main, problematic_rules):
    """
    Create a compact view with multi-level column structure matching the requested format:
    
    DataFrame Structure:
    - Level 0: Event, Query, Negation Patterns, Short_Precision, Medium_Precision, Long_Precision
    - Level 1: blank, blank, blank, [Pre, Post, Precision_Change], [Pre, Post, Precision_Change], [Pre, Post, Precision_Change]
    """
    
    print("\n" + "="*120)
    print("PIVOT-STRUCTURED COMPACT ANALYSIS: PROBLEMATIC RULES")
    print("="*120)
    print("Creating DataFrame with multi-level columns as requested")
    print("="*120)
    
    if df_main is None or len(problematic_rules) == 0:
        print("Missing required data for compact analysis")
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
    
    # Calculate transcript lengths
    df_main['Customer_Transcript'] = df_main['Customer Transcript'].fillna('')
    df_main['Agent_Transcript'] = df_main['Agent Transcript'].fillna('')
    df_main['Full_Transcript'] = df_main['Customer_Transcript'] + ' ' + df_main['Agent_Transcript']
    df_main['Transcript_Length'] = df_main['Full_Transcript'].str.len()
    
    # Define length categories
    df_main['Length_Category'] = pd.cut(
        df_main['Transcript_Length'],
        bins=[0, 4000, 8000, float('inf')],
        labels=['Short', 'Medium', 'Long']
    )
    
    print("Length Category Distribution:")
    length_dist = df_main['Length_Category'].value_counts()
    for category, count in length_dist.items():
        percentage = count / len(df_main) * 100
        print(f"  {category}: {count:,} transcripts ({percentage:.1f}%)")
    
    # Collect data for each rule
    analysis_data = []
    
    print(f"\nAnalyzing top {len(problematic_rules.head(15))} problematic rules...")
    
    for idx, rule in problematic_rules.head(15).iterrows():
        event = rule['Event']
        query = rule['Query']
        negation_count = rule['negation_patterns']
        
        # Find matching transcripts
        rule_transcripts = df_main[
            (df_main['Prosodica L1'].str.lower() == event.lower()) |
            (df_main['Prosodica L2'].str.lower() == query.lower())
        ]
        
        if len(rule_transcripts) == 0:
            continue
        
        # Split by period
        pre_data = rule_transcripts[rule_transcripts['Period'] == 'Pre']
        post_data = rule_transcripts[rule_transcripts['Period'] == 'Post']
        
        if len(pre_data) == 0 or len(post_data) == 0:
            continue
        
        # Calculate precision for each length category and period
        row_data = {
            'Event': event,
            'Query': query,
            'Negation_Patterns': negation_count
        }
        
        for length_cat in ['Short', 'Medium', 'Long']:
            # Pre period
            pre_length_data = pre_data[pre_data['Length_Category'] == length_cat]
            pre_precision = (pre_length_data['Primary Marker'] == 'TP').mean() if len(pre_length_data) > 0 else 0.0
            
            # Post period
            post_length_data = post_data[post_data['Length_Category'] == length_cat]
            post_precision = (post_length_data['Primary Marker'] == 'TP').mean() if len(post_length_data) > 0 else 0.0
            
            # Calculate change
            precision_change = post_precision - pre_precision
            
            # Store data
            row_data[f'{length_cat}_Pre'] = round(pre_precision, 3)
            row_data[f'{length_cat}_Post'] = round(post_precision, 3)
            row_data[f'{length_cat}_Change'] = round(precision_change, 3)
        
        analysis_data.append(row_data)
    
    # Create DataFrame from collected data
    if len(analysis_data) == 0:
        print("No analysis data collected")
        return pd.DataFrame()
    
    base_df = pd.DataFrame(analysis_data)
    
    # Sort by Short precision change (most negative first)
    base_df = base_df.sort_values('Short_Change')
    
    # Create the multi-level column structure
    print("\nCreating multi-level column structure...")
    
    # Define the column structure
    # Level 0: Main category headers
    # Level 1: Sub-headers (Pre, Post, Precision_Change)
    
    columns_level_0 = (
        ['Event', 'Query', 'Negation Patterns'] +
        ['Short_Precision'] * 3 +
        ['Medium_Precision'] * 3 +
        ['Long_Precision'] * 3
    )
    
    columns_level_1 = (
        ['', '', ''] +  # Empty for basic info columns
        ['Pre', 'Post', 'Precision_Change'] +  # Short
        ['Pre', 'Post', 'Precision_Change'] +  # Medium
        ['Pre', 'Post', 'Precision_Change']    # Long
    )
    
    # Create MultiIndex columns
    multi_columns = pd.MultiIndex.from_tuples(
        list(zip(columns_level_0, columns_level_1)),
        names=['Category', 'Metric']
    )
    
    # Create the final DataFrame with multi-level columns
    final_data = []
    
    for _, row in base_df.iterrows():
        final_row = [
            row['Event'],
            row['Query'], 
            row['Negation_Patterns'],
            row['Short_Pre'],
            row['Short_Post'],
            row['Short_Change'],
            row['Medium_Pre'],
            row['Medium_Post'],
            row['Medium_Change'],
            row['Long_Pre'],
            row['Long_Post'],
            row['Long_Change']
        ]
        final_data.append(final_row)
    
    # Create the final DataFrame with MultiIndex columns
    pivot_df = pd.DataFrame(final_data, columns=multi_columns)
    
    print(f"\nPivot DataFrame created with shape: {pivot_df.shape}")
    print("Column structure:")
    print(f"Level 0: {pivot_df.columns.get_level_values(0).unique().tolist()}")
    print(f"Level 1: {pivot_df.columns.get_level_values(1).unique().tolist()}")
    
    # Display preview
    print(f"\nPreview of top 5 rules:")
    print("-" * 120)
    
    # Display with proper formatting
    display_preview(pivot_df.head(5))
    
    # Calculate and display summary statistics
    print(f"\n" + "="*120)
    print("SUMMARY STATISTICS:")
    print("="*120)
    
    # Extract precision change columns for analysis
    short_changes = pivot_df[('Short_Precision', 'Precision_Change')]
    medium_changes = pivot_df[('Medium_Precision', 'Precision_Change')]
    long_changes = pivot_df[('Long_Precision', 'Precision_Change')]
    
    print(f"Average Precision Changes:")
    print(f"  Short Transcripts:  {short_changes.mean():+.3f}")
    print(f"  Medium Transcripts: {medium_changes.mean():+.3f}")
    print(f"  Long Transcripts:   {long_changes.mean():+.3f}")
    
    # Count significant drops
    short_drops = (short_changes < -0.05).sum()
    medium_drops = (medium_changes < -0.05).sum()
    long_drops = (long_changes < -0.05).sum()
    
    print(f"\nRules with >5% Precision Drop:")
    print(f"  Short Transcripts:  {short_drops}/{len(pivot_df)} ({short_drops/len(pivot_df)*100:.1f}%)")
    print(f"  Medium Transcripts: {medium_drops}/{len(pivot_df)} ({medium_drops/len(pivot_df)*100:.1f}%)")
    print(f"  Long Transcripts:   {long_drops}/{len(pivot_df)} ({long_drops/len(pivot_df)*100:.1f}%)")
    
    # Key insights
    if short_drops > medium_drops and short_drops > long_drops:
        print(f"\n*** KEY FINDING: Short transcripts are disproportionately affected ***")
        print(f"*** This confirms the length-based discrimination hypothesis ***")
    
    return pivot_df

def display_preview(df):
    """Display a nicely formatted preview of the multi-level DataFrame"""
    
    print("Event                Query                Neg |    Short_Precision    |   Medium_Precision    |    Long_Precision     |")
    print("                                              |  Pre  Post  Change   |  Pre  Post  Change   |  Pre  Post  Change   |")
    print("-" * 120)
    
    for idx, row in df.iterrows():
        event = str(row[('Event', '')])[:20].ljust(20)
        query = str(row[('Query', '')])[:20].ljust(20)
        neg = str(int(row[('Negation Patterns', '')])).rjust(3)
        
        # Short precision
        short_pre = f"{row[('Short_Precision', 'Pre')]:.3f}"
        short_post = f"{row[('Short_Precision', 'Post')]:.3f}"
        short_change = f"{row[('Short_Precision', 'Precision_Change')]:+.3f}"
        
        # Medium precision
        medium_pre = f"{row[('Medium_Precision', 'Pre')]:.3f}"
        medium_post = f"{row[('Medium_Precision', 'Post')]:.3f}"
        medium_change = f"{row[('Medium_Precision', 'Precision_Change')]:+.3f}"
        
        # Long precision
        long_pre = f"{row[('Long_Precision', 'Pre')]:.3f}"
        long_post = f"{row[('Long_Precision', 'Post')]:.3f}"
        long_change = f"{row[('Long_Precision', 'Precision_Change')]:+.3f}"
        
        print(f"{event} {query} {neg} | {short_pre} {short_post} {short_change} | {medium_pre} {medium_post} {medium_change} | {long_pre} {long_post} {long_change} |")

def export_pivot_results(pivot_df):
    """Export the pivot DataFrame to Excel with proper multi-level headers"""
    
    if len(pivot_df) == 0:
        print("No data to export")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'Pivot_Problematic_Rules_Analysis_{timestamp}.xlsx'
    
    print(f"\nExporting pivot results to: {filename}")
    
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        # Write the multi-level DataFrame
        pivot_df.to_excel(writer, sheet_name='Pivot_Analysis', index=False)
        
        # Get workbook and worksheet for formatting
        workbook = writer.book
        worksheet = writer.sheets['Pivot_Analysis']
        
        # Create formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1,
            'align': 'center'
        })
        
        # Format the multi-level headers
        # Row 0: Level 0 headers
        # Row 1: Level 1 headers
        
        # Apply header formatting
        for col in range(len(pivot_df.columns)):
            worksheet.write(0, col, pivot_df.columns[col][0], header_format)
            worksheet.write(1, col, pivot_df.columns[col][1], header_format)
    
    print(f"Pivot analysis exported successfully with multi-level headers!")
    return filename

def run_pivot_analysis(df_main, problematic_rules):
    """Run the complete pivot analysis"""
    
    pivot_results = create_compact_problematic_rules_view_pivot(df_main, problematic_rules)
    
    if len(pivot_results) > 0:
        export_filename = export_pivot_results(pivot_results)
        return pivot_results, export_filename
    else:
        return pd.DataFrame(), None

# Usage in your main script:
"""
# After identifying problematic rules:
pivot_results, export_file = run_pivot_analysis(df_main, problematic_rules)

# To access specific columns in the multi-level DataFrame:
short_precision_changes = pivot_results[('Short_Precision', 'Precision_Change')]
medium_pre_values = pivot_results[('Medium_Precision', 'Pre')]
long_post_values = pivot_results[('Long_Precision', 'Post')]
"""

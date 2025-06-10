# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

def read_excel_file(filename):
    """Read Excel file and return dataframe."""
    try:
        df = pd.read_excel(filename)
        print(f"Successfully read {filename} with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def analyze_file_structure(df, filename):
    """Analyze and print file structure."""
    print(f"\n--- Structure Analysis for {filename} ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col} (Type: {df[col].dtype})")
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    for col, count in zip(missing.index, missing.values):
        if count > 0:
            print(f"- {col}: {count} ({count/len(df)*100:.2f}%)")

def compare_columns(df1, df2, filename1, filename2):
    """Compare columns between two dataframes."""
    print(f"\n--- Column Comparison between {filename1} and {filename2} ---")
    
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    common_cols = cols1.intersection(cols2)
    only_in_df1 = cols1 - cols2
    only_in_df2 = cols2 - cols1
    
    print(f"Common columns: {len(common_cols)}")
    print(f"Columns only in {filename1}: {list(only_in_df1) if only_in_df1 else 'None'}")
    print(f"Columns only in {filename2}: {list(only_in_df2) if only_in_df2 else 'None'}")

def analyze_variable5_relation(df1, df2, filename1, filename2):
    """Analyze variable5 which should be common between files."""
    print(f"\n--- variable5 Analysis ---")
    
    # Check if variable5 exists in both dataframes
    if 'variable5' not in df1.columns or 'variable5' not in df2.columns:
        print("variable5 not found in one or both files")
        return
    
    # Count unique variable5 values in each file
    var5_unique1 = df1['variable5'].nunique()
    var5_unique2 = df2['variable5'].nunique()
    
    print(f"Unique variable5 values in {filename1}: {var5_unique1}")
    print(f"Unique variable5 values in {filename2}: {var5_unique2}")
    
    # Check for variable5 values that exist in one file but not the other
    var5_values1 = set(df1['variable5'].unique())
    var5_values2 = set(df2['variable5'].unique())
    
    only_in_file1 = var5_values1 - var5_values2
    only_in_file2 = var5_values2 - var5_values1
    
    print(f"variable5 values only in {filename1}: {len(only_in_file1)}")
    print(f"variable5 values only in {filename2}: {len(only_in_file2)}")
    
    # For file1, analyze how many UUIDs per variable5
    if 'UUID' in df1.columns:
        uuid_per_var5 = df1.groupby('variable5')['UUID'].nunique()
        
        print("\nUUID distribution per variable5 in original file:")
        print(f"Min UUIDs per variable5: {uuid_per_var5.min()}")
        print(f"Max UUIDs per variable5: {uuid_per_var5.max()}")
        print(f"Mean UUIDs per variable5: {uuid_per_var5.mean():.2f}")
        print(f"Median UUIDs per variable5: {uuid_per_var5.median()}")
        
        # Distribution of number of UUIDs per variable5
        counts = uuid_per_var5.value_counts().sort_index()
        print("\nDistribution of UUID counts per variable5:")
        for count, frequency in counts.items():
            print(f"{count} UUID(s): {frequency} variable5 values")

def analyze_call_integrity(df1, df2, filename1, filename2):
    """Analyze the integrity of calls without UUID in the second file."""
    print(f"\n--- Call Integrity Analysis ---")
    
    # Since UUID is missing in file2, check if each variable5 has the same number of entries
    var5_counts1 = df1['variable5'].value_counts()
    
    if 'variable5' in df2.columns:
        var5_counts2 = df2['variable5'].value_counts()
        
        # Get common variable5 values
        common_var5 = set(var5_counts1.index).intersection(set(var5_counts2.index))
        
        # Compare counts for common variable5 values
        count_diffs = []
        for var5 in common_var5:
            count1 = var5_counts1[var5]
            count2 = var5_counts2[var5]
            if count1 != count2:
                count_diffs.append((var5, count1, count2, count2 - count1))
        
        if count_diffs:
            print(f"Found {len(count_diffs)} variable5 values with different entry counts")
            count_diffs.sort(key=lambda x: abs(x[3]), reverse=True)
            print("\nTop differences (by absolute difference):")
            for var5, count1, count2, diff in count_diffs[:10]:
                print(f"variable5: {var5}, {filename1}: {count1} entries, {filename2}: {count2} entries, Difference: {diff}")
        else:
            print("All common variable5 values have the same number of entries in both files")

def analyze_precision_metrics(df1, df2, filename1, filename2):
    """Analyze precision metrics between files."""
    print(f"\n--- Precision Metrics Analysis ---")
    
    if 'Primary Marker' not in df1.columns or 'Primary Marker' not in df2.columns:
        print("Primary Marker not found in one or both files")
        return
    
    # Calculate metrics for file1
    tp_count1 = (df1['Primary Marker'] == 'TP').sum()
    fp_count1 = (df1['Primary Marker'] == 'FP').sum()
    total_marked1 = tp_count1 + fp_count1
    precision1 = tp_count1 / total_marked1 if total_marked1 > 0 else 0
    
    # Calculate metrics for file2
    tp_count2 = (df2['Primary Marker'] == 'TP').sum()
    fp_count2 = (df2['Primary Marker'] == 'FP').sum()
    total_marked2 = tp_count2 + fp_count2
    precision2 = tp_count2 / total_marked2 if total_marked2 > 0 else 0
    
    print(f"{filename1} Precision: {precision1:.4f} ({tp_count1} TP, {fp_count1} FP, {total_marked1} total)")
    print(f"{filename2} Precision: {precision2:.4f} ({tp_count2} TP, {fp_count2} FP, {total_marked2} total)")
    print(f"Precision change: {precision2 - precision1:.4f} ({(precision2 - precision1) / precision1 * 100 if precision1 > 0 else 0:.2f}%)")
    
    # Analyze by category
    for category_col in ['Prosodica L1', 'Primary L1']:
        if category_col in df1.columns and category_col in df2.columns:
            print(f"\nPrecision by {category_col}:")
            
            # Calculate precision by category for file1
            cat_metrics1 = {}
            for cat in df1[category_col].dropna().unique():
                cat_df = df1[df1[category_col] == cat]
                cat_tp = (cat_df['Primary Marker'] == 'TP').sum()
                cat_fp = (cat_df['Primary Marker'] == 'FP').sum()
                cat_total = cat_tp + cat_fp
                cat_precision = cat_tp / cat_total if cat_total > 0 else 0
                cat_metrics1[cat] = {'tp': cat_tp, 'fp': cat_fp, 'precision': cat_precision, 'total': cat_total}
            
            # Calculate precision by category for file2
            cat_metrics2 = {}
            for cat in df2[category_col].dropna().unique():
                cat_df = df2[df2[category_col] == cat]
                cat_tp = (cat_df['Primary Marker'] == 'TP').sum()
                cat_fp = (cat_df['Primary Marker'] == 'FP').sum()
                cat_total = cat_tp + cat_fp
                cat_precision = cat_tp / cat_total if cat_total > 0 else 0
                cat_metrics2[cat] = {'tp': cat_tp, 'fp': cat_fp, 'precision': cat_precision, 'total': cat_total}
            
            # Get all categories from both files
            all_cats = set(cat_metrics1.keys()).union(set(cat_metrics2.keys()))
            
            # Print category comparison
            print(f"{'Category':<20} | {'File1 Prec':<10} | {'File2 Prec':<10} | {'Change':<10} | {'File1 Count':<12} | {'File2 Count':<12}")
            print("-" * 90)
            
            for cat in sorted(all_cats):
                metrics1 = cat_metrics1.get(cat, {'precision': 0, 'total': 0})
                metrics2 = cat_metrics2.get(cat, {'precision': 0, 'total': 0})
                
                prec1 = metrics1['precision']
                prec2 = metrics2['precision']
                change = prec2 - prec1
                count1 = metrics1['total']
                count2 = metrics2['total']
                
                print(f"{str(cat)[:20]:<20} | {prec1:.4f}     | {prec2:.4f}     | {change:+.4f}    | {count1:<12} | {count2:<12}")

def analyze_transcript_data(df1, df2, filename1, filename2):
    """Analyze transcript data for differences."""
    print(f"\n--- Transcript Data Analysis ---")
    
    transcript_cols = ['Customer Transcript', 'Agent Transcript']
    
    for col in transcript_cols:
        if col in df1.columns and col in df2.columns:
            # Check for empty transcripts
            empty1 = df1[col].isna().sum()
            empty2 = df2[col].isna().sum()
            
            print(f"\n{col} analysis:")
            print(f"Empty transcripts in {filename1}: {empty1} ({empty1/len(df1)*100:.2f}%)")
            print(f"Empty transcripts in {filename2}: {empty2} ({empty2/len(df2)*100:.2f}%)")
            
            # Analyze transcript lengths
            if df1[col].dtype == object and df2[col].dtype == object:
                df1['transcript_len'] = df1[col].fillna('').apply(len)
                df2['transcript_len'] = df2[col].fillna('').apply(len)
                
                print(f"Avg transcript length in {filename1}: {df1['transcript_len'].mean():.2f} chars")
                print(f"Avg transcript length in {filename2}: {df2['transcript_len'].mean():.2f} chars")
                print(f"Max transcript length in {filename1}: {df1['transcript_len'].max()} chars")
                print(f"Max transcript length in {filename2}: {df2['transcript_len'].max()} chars")
                
                # Remove temporary columns
                df1.drop('transcript_len', axis=1, inplace=True)
                df2.drop('transcript_len', axis=1, inplace=True)

def analyze_date_distribution(df1, df2, filename1, filename2):
    """Analyze date distribution between files."""
    print(f"\n--- Date Distribution Analysis ---")
    
    if 'Date' not in df1.columns or 'Date' not in df2.columns:
        print("Date column not found in one or both files")
        return
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_dtype(df1['Date']):
        df1['Date'] = pd.to_datetime(df1['Date'], errors='coerce')
    
    if not pd.api.types.is_datetime64_dtype(df2['Date']):
        df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')
    
    # Analyze date ranges
    min_date1 = df1['Date'].min()
    max_date1 = df1['Date'].max()
    min_date2 = df2['Date'].min()
    max_date2 = df2['Date'].max()
    
    print(f"{filename1} date range: {min_date1} to {max_date1}")
    print(f"{filename2} date range: {min_date2} to {max_date2}")
    
    # Count by month
    df1['month'] = df1['Date'].dt.to_period('M')
    df2['month'] = df2['Date'].dt.to_period('M')
    
    counts1 = df1['month'].value_counts().sort_index()
    counts2 = df2['month'].value_counts().sort_index()
    
    # Find all months across both dataframes
    all_months = sorted(set(counts1.index) | set(counts2.index))
    
    print("\nEntry counts by month:")
    print(f"{'Month':<10} | {filename1:<10} | {filename2:<10} | {'Difference':<10}")
    print("-" * 50)
    
    for month in all_months:
        count1 = counts1.get(month, 0)
        count2 = counts2.get(month, 0)
        diff = count2 - count1
        print(f"{str(month):<10} | {count1:<10} | {count2:<10} | {diff:+}")
    
    # Remove temporary columns
    df1.drop('month', axis=1, inplace=True)
    df2.drop('month', axis=1, inplace=True)

def summarize_key_findings(df1, df2, filename1, filename2):
    """Summarize key findings from the analysis."""
    print(f"\n--- Key Findings Summary ---")
    
    # 1. Structure differences
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    struct_diff = cols1 - cols2
    
    print(f"1. Structure Differences: {', '.join(struct_diff) if struct_diff else 'None'}")
    
    # 2. Size comparison
    rows_diff = len(df2) - len(df1)
    print(f"2. Size Difference: {filename2} has {abs(rows_diff)} {'more' if rows_diff > 0 else 'fewer'} rows than {filename1}")
    
    # 3. Precision metrics comparison
    if 'Primary Marker' in df1.columns and 'Primary Marker' in df2.columns:
        tp_count1 = (df1['Primary Marker'] == 'TP').sum()
        fp_count1 = (df1['Primary Marker'] == 'FP').sum()
        total_marked1 = tp_count1 + fp_count1
        precision1 = tp_count1 / total_marked1 if total_marked1 > 0 else 0
        
        tp_count2 = (df2['Primary Marker'] == 'TP').sum()
        fp_count2 = (df2['Primary Marker'] == 'FP').sum()
        total_marked2 = tp_count2 + fp_count2
        precision2 = tp_count2 / total_marked2 if total_marked2 > 0 else 0
        
        precision_change = precision2 - precision1
        print(f"3. Precision Change: {precision_change:.4f} ({precision_change/precision1*100 if precision1 > 0 else 0:.2f}%)")
    
    # 4. Variable5 completeness
    if 'variable5' in df1.columns and 'variable5' in df2.columns:
        var5_only_in_1 = set(df1['variable5'].unique()) - set(df2['variable5'].unique())
        var5_only_in_2 = set(df2['variable5'].unique()) - set(df1['variable5'].unique())
        
        print(f"4. variable5 Completeness: {len(var5_only_in_1)} values only in {filename1}, {len(var5_only_in_2)} values only in {filename2}")
    
    # 5. UUID impact
    print("5. UUID Impact: Without UUID in the new file, call-level grouping requires using variable5 alone")
    
    # 6. Additional findings
    print("6. Additional findings:")
    if 'UUID' in df1.columns:
        multi_uuid = df1.groupby('variable5')['UUID'].nunique()
        pct_multi = (multi_uuid > 1).mean() * 100
        print(f"   - {pct_multi:.2f}% of variable5 values in {filename1} have multiple UUIDs")
    
    if 'Primary Marker' in df1.columns and 'Primary Marker' in df2.columns:
        fp_rate1 = fp_count1 / total_marked1 if total_marked1 > 0 else 0
        fp_rate2 = fp_count2 / total_marked2 if total_marked2 > 0 else 0
        print(f"   - False positive rate changed from {fp_rate1:.4f} to {fp_rate2:.4f}")

def main():
    # Define filenames
    filename1 = "Precision_Drop_Analysis_OG.xlsx"
    filename2 = "Precision_Drop_Analysis_NEW.xlsx"
    
    # Read the Excel files
    df1 = read_excel_file(filename1)
    df2 = read_excel_file(filename2)
    
    if df1 is None or df2 is None:
        print("Error: Could not read one or both of the files. Exiting.")
        return
    
    # Run analysis functions
    analyze_file_structure(df1, filename1)
    analyze_file_structure(df2, filename2)
    compare_columns(df1, df2, filename1, filename2)
    analyze_variable5_relation(df1, df2, filename1, filename2)
    analyze_call_integrity(df1, df2, filename1, filename2)
    analyze_precision_metrics(df1, df2, filename1, filename2)
    analyze_transcript_data(df1, df2, filename1, filename2)
    analyze_date_distribution(df1, df2, filename1, filename2)
    summarize_key_findings(df1, df2, filename1, filename2)
    
    print("\nAnalysis complete!")

# Run the script
if __name__ == "__main__":
    main()

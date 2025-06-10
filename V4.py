# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

def read_excel_file(filename):
    """Read Excel file and return dataframe."""
    try:
        return pd.read_excel(filename)
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
    
    print("\nSample Data (First 3 rows):")
    print(df.head(3))

def compare_dataframes(df1, df2, filename1, filename2):
    """Compare two dataframes and identify differences."""
    print(f"\n--- Comparison between {filename1} and {filename2} ---")
    
    # Compare columns
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    common_cols = cols1.intersection(cols2)
    only_in_df1 = cols1 - cols2
    only_in_df2 = cols2 - cols1
    
    print(f"Common columns: {len(common_cols)}")
    print(f"Columns only in {filename1}: {only_in_df1 if only_in_df1 else 'None'}")
    print(f"Columns only in {filename2}: {only_in_df2 if only_in_df2 else 'None'}")
    
    # Check for primary key
    print("\nChecking uniqueness of potential keys:")
    for col in ['variable5', 'UUID']:
        if col in common_cols:
            unique1 = df1[col].nunique()
            unique2 = df2[col].nunique()
            print(f"- {col}: {unique1} unique values in {filename1}, {unique2} unique values in {filename2}")
    
    # Compare data distributions for common columns
    print("\nData distribution comparison for common columns:")
    for col in common_cols:
        if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
            mean1, mean2 = df1[col].mean(), df2[col].mean()
            print(f"- {col}: Mean {mean1:.2f} in {filename1}, {mean2:.2f} in {filename2}, Difference: {abs(mean1-mean2):.2f}")
        elif pd.api.types.is_string_dtype(df1[col]) and pd.api.types.is_string_dtype(df2[col]):
            unique_vals1 = df1[col].nunique()
            unique_vals2 = df2[col].nunique()
            print(f"- {col}: {unique_vals1} unique values in {filename1}, {unique_vals2} unique values in {filename2}")
            
            # For categorical columns with reasonable number of values, show value counts difference
            if unique_vals1 < 20 and unique_vals2 < 20:
                counts1 = df1[col].value_counts().to_dict()
                counts2 = df2[col].value_counts().to_dict()
                
                all_values = set(counts1.keys()).union(set(counts2.keys()))
                print(f"  Value distribution changes:")
                for val in all_values:
                    count1 = counts1.get(val, 0)
                    count2 = counts2.get(val, 0)
                    if count1 != count2:
                        print(f"    '{val}': {count1} -> {count2}, Change: {count2 - count1}")

def analyze_variable5_uuid_relationship(df1, df2, filename1, filename2):
    """Analyze the relationship between variable5 and UUID in both files."""
    print("\n--- variable5 and UUID Relationship Analysis ---")
    
    # Check if both columns exist in both dataframes
    if 'variable5' in df1.columns and 'UUID' in df1.columns and 'variable5' in df2.columns and 'UUID' in df2.columns:
        # For each file, count how many UUIDs per variable5
        var5_counts1 = df1.groupby('variable5')['UUID'].nunique().reset_index()
        var5_counts1.columns = ['variable5', 'uuid_count']
        
        var5_counts2 = df2.groupby('variable5')['UUID'].nunique().reset_index()
        var5_counts2.columns = ['variable5', 'uuid_count']
        
        print(f"\nDistribution of UUIDs per variable5 in {filename1}:")
        print(var5_counts1['uuid_count'].describe())
        
        print(f"\nDistribution of UUIDs per variable5 in {filename2}:")
        print(var5_counts2['uuid_count'].describe())
        
        # Check for variable5 values that exist in one file but not the other
        var5_in_1_not_2 = set(df1['variable5'].unique()) - set(df2['variable5'].unique())
        var5_in_2_not_1 = set(df2['variable5'].unique()) - set(df1['variable5'].unique())
        
        print(f"\nNumber of variable5 values in {filename1} but not in {filename2}: {len(var5_in_1_not_2)}")
        print(f"Number of variable5 values in {filename2} but not in {filename1}: {len(var5_in_2_not_1)}")
        
        # Analyze changes in UUID counts for common variable5 values
        common_var5 = set(df1['variable5'].unique()).intersection(set(df2['variable5'].unique()))
        print(f"\nAnalyzing {len(common_var5)} common variable5 values between files")
        
        var5_counts1_dict = dict(zip(var5_counts1['variable5'], var5_counts1['uuid_count']))
        var5_counts2_dict = dict(zip(var5_counts2['variable5'], var5_counts2['uuid_count']))
        
        changes = []
        for var5 in common_var5:
            count1 = var5_counts1_dict.get(var5, 0)
            count2 = var5_counts2_dict.get(var5, 0)
            if count1 != count2:
                changes.append((var5, count1, count2, count2 - count1))
        
        if changes:
            print(f"\nFound {len(changes)} variable5 values with different UUID counts")
            changes.sort(key=lambda x: abs(x[3]), reverse=True)
            print("Top 5 changes (by absolute difference):")
            for var5, count1, count2, diff in changes[:5]:
                print(f"variable5: {var5}, {filename1}: {count1} UUIDs, {filename2}: {count2} UUIDs, Difference: {diff}")

def analyze_marker_distribution(df1, df2, filename1, filename2):
    """Analyze the distribution of Primary and Secondary Markers."""
    print("\n--- Marker Distribution Analysis ---")
    
    # Check if marker columns exist in both dataframes
    marker_cols = ['Primary Marker', 'Secondary Marker']
    
    for col in marker_cols:
        if col in df1.columns and col in df2.columns:
            print(f"\n{col} distribution:")
            
            counts1 = df1[col].value_counts(dropna=False).to_dict()
            counts2 = df2[col].value_counts(dropna=False).to_dict()
            
            all_values = set(counts1.keys()).union(set(counts2.keys()))
            
            print(f"{'Value':<10} | {filename1:<10} | {filename2:<10} | Change")
            print("-" * 50)
            
            for val in all_values:
                count1 = counts1.get(val, 0)
                count2 = counts2.get(val, 0)
                change = count2 - count1
                val_str = str(val) if pd.notna(val) else "NaN"
                print(f"{val_str:<10} | {count1:<10} | {count2:<10} | {change:+}")

def analyze_precision_metrics(df1, df2, filename1, filename2):
    """Calculate and compare precision metrics between the two files."""
    print("\n--- Precision Metrics Analysis ---")
    
    # Check if we have the necessary columns
    if 'Primary Marker' in df1.columns and 'Primary Marker' in df2.columns:
        # Calculate precision metrics for file 1
        tp1 = (df1['Primary Marker'] == 'TP').sum()
        fp1 = (df1['Primary Marker'] == 'FP').sum()
        precision1 = tp1 / (tp1 + fp1) if (tp1 + fp1) > 0 else 0
        
        # Calculate precision metrics for file 2
        tp2 = (df2['Primary Marker'] == 'TP').sum()
        fp2 = (df2['Primary Marker'] == 'FP').sum()
        precision2 = tp2 / (tp2 + fp1) if (tp2 + fp2) > 0 else 0
        
        print(f"\nPrecision Metrics for {filename1}:")
        print(f"True Positives: {tp1}")
        print(f"False Positives: {fp1}")
        print(f"Precision: {precision1:.4f}")
        
        print(f"\nPrecision Metrics for {filename2}:")
        print(f"True Positives: {tp2}")
        print(f"False Positives: {fp2}")
        print(f"Precision: {precision2:.4f}")
        
        print(f"\nPrecision Change: {precision2-precision1:.4f} ({(precision2-precision1)/precision1*100:.2f}%)")
        
        # Analyze by categories if they exist
        if 'Prosodica L1' in df1.columns and 'Prosodica L1' in df2.columns:
            print("\nPrecision by Prosodica L1 Category:")
            
            # For file 1
            cat_precision1 = {}
            for cat in df1['Prosodica L1'].unique():
                if pd.notna(cat):
                    cat_df = df1[df1['Prosodica L1'] == cat]
                    cat_tp = (cat_df['Primary Marker'] == 'TP').sum()
                    cat_fp = (cat_df['Primary Marker'] == 'FP').sum()
                    cat_precision1[cat] = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0
            
            # For file 2
            cat_precision2 = {}
            for cat in df2['Prosodica L1'].unique():
                if pd.notna(cat):
                    cat_df = df2[df2['Prosodica L1'] == cat]
                    cat_tp = (cat_df['Primary Marker'] == 'TP').sum()
                    cat_fp = (cat_df['Primary Marker'] == 'FP').sum()
                    cat_precision2[cat] = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0
            
            # Show changes for common categories
            all_cats = set(cat_precision1.keys()).union(set(cat_precision2.keys()))
            
            print(f"{'Category':<20} | {filename1:<10} | {filename2:<10} | Change")
            print("-" * 60)
            
            for cat in all_cats:
                prec1 = cat_precision1.get(cat, 0)
                prec2 = cat_precision2.get(cat, 0)
                change = prec2 - prec1
                print(f"{cat:<20} | {prec1:.4f}     | {prec2:.4f}     | {change:+.4f}")

def main():
    # Define filenames
    filename1 = "Precision_Drop_Analysis_OG.xlsx"
    filename2 = "Precision_Drop_Analysis_NEW.xlsx"
    
    # Read the Excel files
    print(f"Reading {filename1}...")
    df1 = read_excel_file(filename1)
    
    print(f"Reading {filename2}...")
    df2 = read_excel_file(filename2)
    
    if df1 is None or df2 is None:
        print("Error: Could not read one or both of the files. Exiting.")
        return
    
    # Analyze file structures
    analyze_file_structure(df1, filename1)
    analyze_file_structure(df2, filename2)
    
    # Compare the dataframes
    compare_dataframes(df1, df2, filename1, filename2)
    
    # Analyze variable5 and UUID relationship
    analyze_variable5_uuid_relationship(df1, df2, filename1, filename2)
    
    # Analyze marker distributions
    analyze_marker_distribution(df1, df2, filename1, filename2)
    
    # Analyze precision metrics
    analyze_precision_metrics(df1, df2, filename1, filename2)
    
    print("\n--- Analysis Complete ---")

# Run the main function
if __name__ == "__main__":
    main()

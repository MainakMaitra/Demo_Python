# Cell 1: Data Preparation
df_main, df_validation, df_rules = load_and_prepare_data()

# Cell 2: Macro Level Analysis
monthly_category_precision, category_impact = analyze_precision_drop_patterns_enhanced(df_main)

# Cell 3: Volume vs Performance Analysis
volume_precision, monthly_trends = analyze_volume_vs_performance_enhanced(df_main)

# Cell 4: Query Performance Review
all_categories, complaint_categories, top_5_drop_drivers = query_performance_review_enhanced(df_main, df_rules)

# Cell 5: Pattern Detection Analysis
comparison, monthly_precision = pattern_detection_analysis_enhanced(df_main)

# Cell 6: False Positive Pattern Analysis
fp_summary, srsrwi_df, fp_patterns, fp_reason_summary = fp_pattern_analysis_enhanced(df_main)

# Cell 7: Validation Process Assessment
monthly_validation, category_agreement = validation_process_assessment_enhanced(df_main)

# Cell 8: Temporal Analysis
dow_analysis, wom_analysis, operational_analysis = temporal_analysis_enhanced(df_main)

# Cell 9: Category-Specific Investigation
monthly_rule_performance, language_evolution = category_specific_investigation_enhanced(df_main, df_rules, complaint_categories)

# Cell 10: Cross-Category Analysis
transcript_categories, multi_category = cross_category_analysis_enhanced(df_main)

# Cell 11: Content Pattern Analysis
length_comparison, ratio_comparison, qualifying_comparison, pattern_df = content_pattern_analysis_enhanced(df_main)

# Cell 12: Export All Results to Excel
import pandas as pd
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

with pd.ExcelWriter(f'Complaints_Analysis_Results_{timestamp}.xlsx', engine='xlsxwriter') as writer:
    
    # Write each result to a separate sheet
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
    srsrwi_df.to_excel(writer, sheet_name='SRSRWI_Sample', index=False)
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

print(f"All results exported to: Complaints_Analysis_Results_{timestamp}.xlsx")

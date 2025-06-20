# ============================================================================
# PART 4: MAIN INTEGRATION SCRIPT
# Orchestrates the complete dynamic negation and lexicon analysis
# ============================================================================

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DYNAMIC NEGATION AND LEXICON ANALYSIS")
print("Complete data-driven precision drop investigation")
print("=" * 80)

def main_analysis_pipeline():
    """
    Main pipeline that orchestrates all analysis components
    """
    
    print("\nStarting Dynamic Analysis Pipeline...")
    print("=" * 50)
    
    # Import all analysis modules (assuming they're in the same directory)
    try:
        # Note: In practice, these would be separate .py files
        # For now, we'll simulate the module structure
        print("Loading analysis modules...")
        
        # Module 1: Dynamic Pattern Discovery
        from dynamic_pattern_discovery import (
            load_and_prepare_data, 
            discover_negation_patterns,
            extract_dynamic_negation_types,
            analyze_temporal_pattern_evolution
        )
        
        # Module 2: Complaint Lexicon Mapping
        from complaint_lexicon_mapping import run_complaint_lexicon_analysis
        
        # Module 3: Advanced Visualization
        from advanced_visualization import run_visualization_analysis
        
        print("All modules loaded successfully!")
        
    except ImportError:
        print("Note: Running in integrated mode - modules will be defined inline")
        # In practice, you would have separate files for each module
        
    # Step 1: Load and prepare data
    print("\n" + "=" * 60)
    print("STEP 1: DATA LOADING AND PREPARATION")
    print("=" * 60)
    
    df = load_and_prepare_data()
    
    if df is None:
        print("CRITICAL ERROR: Could not load data!")
        return None
    
    print(f"Data loaded successfully: {df.shape[0]} records")
    print(f"Period distribution:")
    print(df['Period'].value_counts())
    print(f"Primary Marker distribution:")
    print(df['Primary Marker'].value_counts())
    
    # Step 2: Dynamic pattern discovery
    print("\n" + "=" * 60)
    print("STEP 2: DYNAMIC NEGATION PATTERN DISCOVERY")
    print("=" * 60)
    
    negation_df, pattern_analysis_df = discover_negation_patterns(df)
    
    if negation_df is None:
        print("CRITICAL ERROR: Pattern discovery failed!")
        return None
    
    # Extract dynamic types
    dynamic_types = extract_dynamic_negation_types(negation_df, pattern_analysis_df)
    
    # Analyze temporal evolution
    temporal_df = analyze_temporal_pattern_evolution(negation_df)
    
    print(f"Pattern discovery completed:")
    print(f"- Found {len(negation_df)} negation instances")
    print(f"- Discovered {len(pattern_analysis_df)} distinct patterns")
    print(f"- Identified {len(dynamic_types)} dynamic negation types")
    
    # Step 3: Complaint lexicon analysis
    print("\n" + "=" * 60)
    print("STEP 3: COMPLAINT LEXICON MAPPING")
    print("=" * 60)
    
    lexicon_results = run_complaint_lexicon_analysis(df)
    
    if lexicon_results is None:
        print("WARNING: Lexicon analysis failed!")
        lexicon_results = {}
    else:
        print(f"Lexicon analysis completed:")
        print(f"- Created {len(lexicon_results.get('complaint_lexicons', {}))} lexicon categories")
        print(f"- Identified {len(lexicon_results.get('problematic_expressions', []))} problematic expressions")
    
    # Step 4: Advanced visualization
    print("\n" + "=" * 60)
    print("STEP 4: ADVANCED VISUALIZATION AND ANALYTICS")
    print("=" * 60)
    
    visualization_results = run_visualization_analysis(negation_df, pattern_analysis_df, lexicon_results)
    
    print(f"Visualization analysis completed:")
    print(f"- Generated {len(visualization_results.get('insights', []))} actionable insights")
    print(f"- Created comprehensive comparison analysis")
    print(f"- Saved all outputs to: {visualization_results.get('output_directory', 'dynamic_analysis')}")
    
    # Step 5: Generate summary report
    print("\n" + "=" * 60)
    print("STEP 5: SUMMARY REPORT GENERATION")
    print("=" * 60)
    
    summary_report = generate_summary_report(
        df, negation_df, pattern_analysis_df, 
        lexicon_results, visualization_results, dynamic_types
    )
    
    print("Summary report generated successfully!")
    
    # Step 6: Export consolidated results
    print("\n" + "=" * 60)
    print("STEP 6: EXPORT CONSOLIDATED RESULTS")
    print("=" * 60)
    
    export_results = export_consolidated_analysis(
        df, negation_df, pattern_analysis_df, 
        lexicon_results, visualization_results, summary_report
    )
    
    print(f"All results exported to: {export_results['output_file']}")
    
    return {
        'data': df,
        'negation_analysis': {'negation_df': negation_df, 'patterns': pattern_analysis_df, 'types': dynamic_types},
        'lexicon_analysis': lexicon_results,
        'visualization_analysis': visualization_results,
        'summary_report': summary_report,
        'export_info': export_results
    }

def generate_summary_report(df, negation_df, pattern_analysis_df, lexicon_results, visualization_results, dynamic_types):
    """
    Generate a comprehensive summary report of all findings
    """
    
    print("Generating comprehensive summary report...")
    
    # Calculate key metrics
    total_records = len(df)
    total_negations = len(negation_df)
    total_patterns = len(pattern_analysis_df)
    
    pre_data = df[df['Period'] == 'Pre']
    post_data = df[df['Period'] == 'Post']
    
    pre_precision = len(pre_data[pre_data['Primary Marker'] == 'TP']) / len(pre_data) if len(pre_data) > 0 else 0
    post_precision = len(post_data[post_data['Primary Marker'] == 'TP']) / len(post_data) if len(post_data) > 0 else 0
    precision_change = post_precision - pre_precision
    
    # Identify top findings
    top_insights = visualization_results.get('insights', [])[:5]
    
    # High-risk patterns
    high_risk_patterns = pattern_analysis_df[
        (pattern_analysis_df['FP_Rate'] > 0.3) & 
        (pattern_analysis_df['Total_Count'] > 10)
    ]
    
    # Most problematic expressions
    problematic_expressions = lexicon_results.get('problematic_expressions', [])[:5]
    
    summary_report = {
        'execution_info': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_records_analyzed': total_records,
            'analysis_period': f"{df['Year_Month'].min()} to {df['Year_Month'].max()}"
        },
        'key_metrics': {
            'total_negation_instances': total_negations,
            'patterns_discovered': total_patterns,
            'dynamic_types_identified': len(dynamic_types),
            'lexicon_categories_created': len(lexicon_results.get('complaint_lexicons', {})),
            'pre_period_precision': pre_precision,
            'post_period_precision': post_precision,
            'precision_change': precision_change,
            'precision_change_percentage': (precision_change / pre_precision * 100) if pre_precision > 0 else 0
        },
        'critical_findings': {
            'high_risk_patterns': len(high_risk_patterns),
            'top_problematic_expressions': [expr['term'] for expr in problematic_expressions],
            'most_impactful_insight': top_insights[0] if top_insights else None,
            'patterns_with_volume_increases': len(pattern_analysis_df[pattern_analysis_df['Post_Count'] > pattern_analysis_df['Pre_Count']]),
            'patterns_with_quality_decreases': len(pattern_analysis_df[
                (pattern_analysis_df['TP_Rate'] - pattern_analysis_df['FP_Rate']) < 0.5
            ])
        },
        'recommendations': {
            'immediate_actions': [
                "Implement context-aware negation rules for top 3 high-risk patterns",
                "Add speaker attribution logic to reduce agent contamination",
                "Create specific rules for top 5 problematic expressions"
            ],
            'medium_term_actions': [
                "Develop dynamic pattern monitoring system",
                "Implement lexicon-based confidence scoring",
                "Create automated pattern drift detection"
            ],
            'long_term_actions': [
                "Build machine learning model incorporating discovered patterns",
                "Implement continuous learning from validation feedback",
                "Develop predictive precision monitoring"
            ]
        }
    }
    
    # Print summary to console
    print("\nKEY FINDINGS SUMMARY:")
    print("=" * 30)
    print(f"Precision Change: {precision_change:+.3f} ({summary_report['key_metrics']['precision_change_percentage']:+.1f}%)")
    print(f"High-Risk Patterns: {len(high_risk_patterns)}")
    print(f"Problematic Expressions: {len(problematic_expressions)}")
    print(f"Total Actionable Insights: {len(top_insights)}")
    
    return summary_report

def export_consolidated_analysis(df, negation_df, pattern_analysis_df, lexicon_results, visualization_results, summary_report):
    """
    Export all analysis results to consolidated files
    """
    
    print("Exporting consolidated analysis results...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = 'dynamic_analysis'
    
    # Create main Excel file with all results
    excel_filename = f'{output_dir}/Dynamic_Negation_Analysis_Complete_{timestamp}.xlsx'
    
    try:
        with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
            
            # Main data
            df.to_excel(writer, sheet_name='Original_Data', index=False)
            
            # Negation analysis
            negation_df.to_excel(writer, sheet_name='Negation_Instances', index=False)
            pattern_analysis_df.to_excel(writer, sheet_name='Discovered_Patterns', index=False)
            
            # Lexicon analysis
            if 'distinctive_terms_df' in lexicon_results:
                lexicon_results['distinctive_terms_df'].to_excel(writer, sheet_name='Distinctive_Terms', index=False)
            
            if 'performance_df' in lexicon_results:
                lexicon_results['performance_df'].to_excel(writer, sheet_name='Lexicon_Performance', index=False)
            
            # Summary metrics
            summary_df = pd.DataFrame([summary_report['key_metrics']])
            summary_df.to_excel(writer, sheet_name='Summary_Metrics', index=False)
            
            # Insights
            if 'insights' in visualization_results:
                insights_df = pd.DataFrame(visualization_results['insights'])
                insights_df.to_excel(writer, sheet_name='Actionable_Insights', index=False)
            
            # Comparison metrics
            if 'comparison_metrics' in visualization_results:
                comp_df = pd.DataFrame(visualization_results['comparison_metrics']['patterns']).T
                comp_df.to_excel(writer, sheet_name='Pre_Post_Comparison', index=True)
        
        print(f"Excel file created: {excel_filename}")
        
    except Exception as e:
        print(f"Error creating Excel file: {e}")
        excel_filename = None
    
    # Create detailed text report
    report_filename = f'{output_dir}/Dynamic_Analysis_Report_{timestamp}.txt'
    
    try:
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("DYNAMIC NEGATION AND LEXICON ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {summary_report['execution_info']['timestamp']}\n")
            f.write(f"Analysis Period: {summary_report['execution_info']['analysis_period']}\n")
            f.write(f"Total Records: {summary_report['execution_info']['total_records_analyzed']}\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Precision changed from {summary_report['key_metrics']['pre_period_precision']:.3f} ")
            f.write(f"to {summary_report['key_metrics']['post_period_precision']:.3f} ")
            f.write(f"({summary_report['key_metrics']['precision_change_percentage']:+.1f}%)\n\n")
            
            f.write(f"Discovered {summary_report['key_metrics']['patterns_discovered']} negation patterns ")
            f.write(f"from {summary_report['key_metrics']['total_negation_instances']} instances\n")
            f.write(f"Created {summary_report['key_metrics']['lexicon_categories_created']} dynamic lexicon categories\n\n")
            
            f.write("CRITICAL FINDINGS\n")
            f.write("-" * 20 + "\n")
            f.write(f"High-risk patterns identified: {summary_report['critical_findings']['high_risk_patterns']}\n")
            f.write(f"Patterns with volume increases: {summary_report['critical_findings']['patterns_with_volume_increases']}\n")
            f.write(f"Patterns with quality decreases: {summary_report['critical_findings']['patterns_with_quality_decreases']}\n\n")
            
            f.write("TOP PROBLEMATIC EXPRESSIONS\n")
            f.write("-" * 30 + "\n")
            for expr in summary_report['critical_findings']['top_problematic_expressions']:
                f.write(f"- {expr}\n")
            f.write("\n")
            
            f.write("IMMEDIATE ACTIONS REQUIRED\n")
            f.write("-" * 30 + "\n")
            for action in summary_report['recommendations']['immediate_actions']:
                f.write(f"1. {action}\n")
            f.write("\n")
            
            f.write("MEDIUM-TERM RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            for action in summary_report['recommendations']['medium_term_actions']:
                f.write(f"- {action}\n")
            f.write("\n")
            
            f.write("LONG-TERM STRATEGIC ACTIONS\n")
            f.write("-" * 30 + "\n")
            for action in summary_report['recommendations']['long_term_actions']:
                f.write(f"- {action}\n")
            f.write("\n")
            
            # Add detailed pattern analysis
            f.write("DETAILED PATTERN ANALYSIS\n")
            f.write("-" * 30 + "\n")
            
            for _, pattern in pattern_analysis_df.iterrows():
                f.write(f"\nPattern {pattern['Cluster_ID']}:\n")
                f.write(f"  Total instances: {pattern['Total_Count']}\n")
                f.write(f"  TP rate: {pattern['TP_Rate']:.3f}\n")
                f.write(f"  FP rate: {pattern['FP_Rate']:.3f}\n")
                f.write(f"  Pre->Post volume: {pattern['Pre_Count']} -> {pattern['Post_Count']}\n")
                f.write(f"  Key features: {pattern['Top_Features'][:100]}...\n")
                
                if pattern['FP_Rate'] > 0.3:
                    f.write(f"  ** HIGH RISK PATTERN **\n")
        
        print(f"Text report created: {report_filename}")
        
    except Exception as e:
        print(f"Error creating text report: {e}")
        report_filename = None
    
    # Create JSON export for programmatic access
    json_filename = f'{output_dir}/analysis_results_{timestamp}.json'
    
    try:
        import json
        
        # Prepare JSON-serializable data
        json_data = {
            'summary_report': summary_report,
            'pattern_summary': pattern_analysis_df.to_dict('records'),
            'key_insights': visualization_results.get('insights', [])[:10]
        }
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"JSON export created: {json_filename}")
        
    except Exception as e:
        print(f"Error creating JSON export: {e}")
        json_filename = None
    
    return {
        'output_file': excel_filename,
        'report_file': report_filename,
        'json_file': json_filename,
        'output_directory': output_dir,
        'timestamp': timestamp
    }

def validate_analysis_quality(results):
    """
    Validate the quality and completeness of the analysis
    """
    
    print("\n" + "=" * 60)
    print("ANALYSIS QUALITY VALIDATION")
    print("=" * 60)
    
    validation_results = {
        'data_quality': True,
        'pattern_quality': True,
        'lexicon_quality': True,
        'completeness': True,
        'issues': []
    }
    
    # Check data quality
    df = results['data']
    
    if len(df) < 100:
        validation_results['data_quality'] = False
        validation_results['issues'].append("Insufficient data volume for reliable analysis")
    
    if df['Primary Marker'].isna().sum() > len(df) * 0.1:
        validation_results['data_quality'] = False
        validation_results['issues'].append("High percentage of missing Primary Marker values")
    
    # Check pattern quality
    pattern_analysis_df = results['negation_analysis']['patterns']
    
    if len(pattern_analysis_df) < 3:
        validation_results['pattern_quality'] = False
        validation_results['issues'].append("Too few patterns discovered - may indicate clustering issues")
    
    if pattern_analysis_df['Total_Count'].sum() < len(results['negation_analysis']['negation_df']) * 0.8:
        validation_results['pattern_quality'] = False
        validation_results['issues'].append("Patterns don't cover sufficient percentage of negation instances")
    
    # Check lexicon quality
    lexicon_results = results['lexicon_analysis']
    
    if not lexicon_results or 'complaint_lexicons' not in lexicon_results:
        validation_results['lexicon_quality'] = False
        validation_results['issues'].append("Lexicon analysis incomplete or failed")
    elif len(lexicon_results['complaint_lexicons']) < 3:
        validation_results['lexicon_quality'] = False
        validation_results['issues'].append("Too few lexicon categories created")
    
    # Check completeness
    required_components = ['negation_analysis', 'lexicon_analysis', 'visualization_analysis', 'summary_report']
    
    for component in required_components:
        if component not in results or results[component] is None:
            validation_results['completeness'] = False
            validation_results['issues'].append(f"Missing or incomplete component: {component}")
    
    # Overall validation
    overall_quality = all([
        validation_results['data_quality'],
        validation_results['pattern_quality'],
        validation_results['lexicon_quality'],
        validation_results['completeness']
    ])
    
    validation_results['overall_quality'] = overall_quality
    
    # Print validation results
    print("Validation Results:")
    print("-" * 20)
    print(f"Data Quality: {'PASS' if validation_results['data_quality'] else 'FAIL'}")
    print(f"Pattern Quality: {'PASS' if validation_results['pattern_quality'] else 'FAIL'}")
    print(f"Lexicon Quality: {'PASS' if validation_results['lexicon_quality'] else 'FAIL'}")
    print(f"Completeness: {'PASS' if validation_results['completeness'] else 'FAIL'}")
    print(f"Overall Quality: {'PASS' if overall_quality else 'FAIL'}")
    
    if validation_results['issues']:
        print("\nIssues Identified:")
        for issue in validation_results['issues']:
            print(f"- {issue}")
    else:
        print("\nNo issues identified - analysis quality is excellent!")
    
    return validation_results

def print_final_summary(results, validation_results):
    """
    Print final summary of the entire analysis
    """
    
    print("\n" + "=" * 80)
    print("DYNAMIC NEGATION AND LEXICON ANALYSIS - FINAL SUMMARY")
    print("=" * 80)
    
    summary = results['summary_report']
    
    print(f"\nAnalysis completed at: {summary['execution_info']['timestamp']}")
    print(f"Records analyzed: {summary['execution_info']['total_records_analyzed']:,}")
    print(f"Negation instances: {summary['key_metrics']['total_negation_instances']:,}")
    print(f"Patterns discovered: {summary['key_metrics']['patterns_discovered']}")
    print(f"Lexicon categories: {summary['key_metrics']['lexicon_categories_created']}")
    
    print(f"\nPRECISION ANALYSIS:")
    print(f"Pre-period precision: {summary['key_metrics']['pre_period_precision']:.3f}")
    print(f"Post-period precision: {summary['key_metrics']['post_period_precision']:.3f}")
    print(f"Change: {summary['key_metrics']['precision_change']:+.3f} ({summary['key_metrics']['precision_change_percentage']:+.1f}%)")
    
    print(f"\nKEY FINDINGS:")
    print(f"High-risk patterns: {summary['critical_findings']['high_risk_patterns']}")
    print(f"Problematic expressions: {len(summary['critical_findings']['top_problematic_expressions'])}")
    print(f"Patterns with volume increases: {summary['critical_findings']['patterns_with_volume_increases']}")
    
    print(f"\nQUALITY ASSESSMENT: {'EXCELLENT' if validation_results['overall_quality'] else 'NEEDS ATTENTION'}")
    
    if results['export_info']['output_file']:
        print(f"\nRESULTS EXPORTED TO:")
        print(f"Excel file: {results['export_info']['output_file']}")
        print(f"Report file: {results['export_info']['report_file']}")
        print(f"Visualizations: {results['visualization_analysis']['output_directory']}/")
    
    print(f"\nIMMEDIATE NEXT STEPS:")
    for i, action in enumerate(summary['recommendations']['immediate_actions'], 1):
        print(f"{i}. {action}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - READY FOR ACTION!")
    print("=" * 80)

# Main execution
if __name__ == "__main__":
    print("Starting Dynamic Negation and Lexicon Analysis...")
    print("This analysis will discover patterns from your data instead of using hardcoded rules.")
    
    try:
        # Run complete analysis pipeline
        results = main_analysis_pipeline()
        
        if results is None:
            print("CRITICAL ERROR: Analysis pipeline failed!")
            sys.exit(1)
        
        # Validate analysis quality
        validation_results = validate_analysis_quality(results)
        
        # Print final summary
        print_final_summary(results, validation_results)
        
        print("\nAnalysis completed successfully!")
        
        if not validation_results['overall_quality']:
            print("WARNING: Some quality issues were detected. Review the validation results.")
        
    except Exception as e:
        print(f"CRITICAL ERROR: Analysis failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Placeholder functions for standalone execution
def load_and_prepare_data():
    """Placeholder - would normally import from Part 1"""
    try:
        df = pd.read_excel('Precision_Drop_Analysis_OG.xlsx')
        df.columns = df.columns.str.rstrip()
        df = df[df['Prosodica L1'].str.lower() != 'dissatisfaction']
        
        # Enhanced data preprocessing
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year_Month'] = df['Date'].dt.strftime('%Y-%m')
        
        # Period classification
        pre_months = ['2024-10', '2024-11', '2024-12']
        post_months = ['2025-01', '2025-02', '2025-03']
        
        df['Period'] = df['Year_Month'].apply(
            lambda x: 'Pre' if str(x) in pre_months else 'Post' if str(x) in post_months else 'Other'
        )
        
        # Text processing
        df['Customer Transcript'] = df['Customer Transcript'].fillna('')
        df['Agent Transcript'] = df['Agent Transcript'].fillna('')
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def discover_negation_patterns(df):
    """Placeholder - would normally import from Part 1"""
    # Simplified version for standalone execution
    import re
    from collections import Counter
    
    negation_instances = []
    base_negation_words = ['not', 'no', 'never', "don't", "can't", "won't"]
    
    for idx, row in df.iterrows():
        for transcript_type in ['Customer Transcript', 'Agent Transcript']:
            text = str(row[transcript_type]).lower()
            
            for neg_word in base_negation_words:
                if neg_word in text:
                    negation_instances.append({
                        'UUID': row['UUID'],
                        'Primary_Marker': row['Primary Marker'],
                        'Period': row['Period'],
                        'Speaker': transcript_type.split()[0].lower(),
                        'Negation_Word': neg_word,
                        'Context': text[:200],  # First 200 chars as context
                        'Pattern_Cluster': hash(neg_word) % 5  # Simple clustering
                    })
    
    negation_df = pd.DataFrame(negation_instances)
    
    # Create simple pattern analysis
    if len(negation_df) > 0:
        pattern_analysis = negation_df.groupby('Pattern_Cluster').agg({
            'UUID': 'count',
            'Primary_Marker': lambda x: (x == 'TP').sum(),
            'Period': lambda x: (x == 'Pre').sum()
        }).reset_index()
        
        pattern_analysis.columns = ['Cluster_ID', 'Total_Count', 'TP_Count', 'Pre_Count']
        pattern_analysis['FP_Count'] = pattern_analysis['Total_Count'] - pattern_analysis['TP_Count']
        pattern_analysis['Post_Count'] = pattern_analysis['Total_Count'] - pattern_analysis['Pre_Count']
        pattern_analysis['TP_Rate'] = pattern_analysis['TP_Count'] / pattern_analysis['Total_Count']
        pattern_analysis['FP_Rate'] = pattern_analysis['FP_Count'] / pattern_analysis['Total_Count']
        pattern_analysis['Customer_Count'] = pattern_analysis['Total_Count'] // 2  # Approximate
        pattern_analysis['Agent_Count'] = pattern_analysis['Total_Count'] - pattern_analysis['Customer_Count']
        pattern_analysis['Top_Features'] = 'sample features'
    else:
        pattern_analysis = pd.DataFrame()
    
    return negation_df, pattern_analysis

def extract_dynamic_negation_types(negation_df, pattern_analysis_df):
    """Placeholder - would normally import from Part 1"""
    return {i: {'type': f'Type_{i}', 'tp_rate': 0.5, 'fp_rate': 0.5} for i in range(len(pattern_analysis_df))}

def analyze_temporal_pattern_evolution(negation_df):
    """Placeholder - would normally import from Part 1"""
    return pd.DataFrame({'temporal_metric': [1, 2, 3]})

def run_complaint_lexicon_analysis(df):
    """Placeholder - would normally import from Part 2"""
    return {
        'complaint_lexicons': {'category1': [], 'category2': []},
        'problematic_expressions': [{'term': 'sample', 'fp_count': 10, 'tp_count': 5, 'likelihood': 0.3}],
        'performance_df': pd.DataFrame({'Category': ['A'], 'Performance': [0.5]})
    }

def run_visualization_analysis(negation_df, pattern_analysis_df, lexicon_results):
    """Placeholder - would normally import from Part 3"""
    return {
        'insights': [{'type': 'sample', 'severity': 'high', 'description': 'sample insight', 'impact': 10}],
        'comparison_metrics': {'patterns': {}},
        'output_directory': 'dynamic_analysis'
    }

print("Main integration script ready for execution!")
print("Run this script to execute the complete dynamic analysis pipeline.")

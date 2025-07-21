# LIME Execution Script - Ready to Run
# This script combines everything and is ready to execute with your existing data

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import required libraries for LIME
try:
    import lime
    from lime.lime_text import LimeTextExplainer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    print("✓ LIME and ML libraries imported successfully")
except ImportError as e:
    print(f"❌ Missing required libraries: {e}")
    print("Please install: pip install lime-ml scikit-learn")
    exit()

def quick_lime_analysis_for_complaints(data_file_path):
    """
    Quick LIME analysis function that works with your existing data structure
    
    This function:
    1. Loads your data
    2. Identifies examples for all three insights
    3. Runs LIME analysis
    4. Exports results
    
    Parameters:
    data_file_path: str, path to your Excel file
    """
    
    print("STARTING QUICK LIME ANALYSIS FOR COMPLAINTS")
    print("="*60)
    
    # Step 1: Load and prepare data
    print("Step 1: Loading data...")
    try:
        df = pd.read_excel(data_file_path)
        print(f"✓ Data loaded: {df.shape}")
    except FileNotFoundError:
        print(f"❌ File not found: {data_file_path}")
        return None
    
    # Clean column names
    df.columns = df.columns.str.rstrip()
    
    # Basic data preparation
    df['Customer_Transcript_Clean'] = df['Customer Transcript'].fillna('').astype(str)
    df['Agent_Transcript_Clean'] = df['Agent Transcript'].fillna('').astype(str)
    df['Full_Transcript'] = df['Customer_Transcript_Clean'] + ' ' + df['Agent_Transcript_Clean']
    df['Is_Problematic'] = (df['Primary Marker'] == 'FP').astype(int)
    
    print(f"✓ Data prepared: {df['Is_Problematic'].sum()} problematic cases (FPs)")
    
    # Step 2: Identify examples for each insight
    print("\nStep 2: Identifying examples for each insight...")
    
    examples = {}
    
    # Insight 1: Context-Blind Negation
    negation_pattern = r'\b(not|no|never|dont|don\'t|wont|won\'t|cant|can\'t)\b'
    info_negation_pattern = r'\b(don\'t know|not sure|no idea|unclear|confused)\b'
    
    df['Has_Negation'] = df['Customer_Transcript_Clean'].str.lower().str.contains(negation_pattern, regex=True, na=False)
    df['Has_Info_Negation'] = df['Customer_Transcript_Clean'].str.lower().str.contains(info_negation_pattern, regex=True, na=False)
    
    negation_examples = df[
        (df['Is_Problematic'] == 1) & 
        (df['Has_Info_Negation'] == True)
    ].head(30)
    
    examples['negation'] = negation_examples
    print(f"✓ Found {len(negation_examples)} negation examples")
    
    # Insight 2: Agent Contamination
    agent_pattern = r'\b(let me explain|for example|suppose|if you|what if)\b'
    df['Has_Agent_Contamination'] = df['Agent_Transcript_Clean'].str.lower().str.contains(agent_pattern, regex=True, na=False)
    
    agent_examples = df[
        (df['Is_Problematic'] == 1) & 
        (df['Has_Agent_Contamination'] == True)
    ].head(30)
    
    examples['agent_contamination'] = agent_examples
    print(f"✓ Found {len(agent_examples)} agent contamination examples")
    
    # Insight 3: Qualifying Language
    qualifying_pattern = r'\b(maybe|might|please|thank|sorry|appreciate)\b'
    df['Has_Qualifying'] = df['Customer_Transcript_Clean'].str.lower().str.contains(qualifying_pattern, regex=True, na=False)
    
    qualifying_examples = df[
        (df['Is_Problematic'] == 0) & 
        (df['Has_Qualifying'] == True)
    ].head(30)
    
    examples['qualifying_language'] = qualifying_examples
    print(f"✓ Found {len(qualifying_examples)} qualifying language examples")
    
    # Step 3: Train models and generate LIME explanations
    print("\nStep 3: Training models and generating LIME explanations...")
    
    explanations = {}
    
    for insight_type in ['negation', 'agent_contamination', 'qualifying_language']:
        if len(examples[insight_type]) < 5:
            print(f"❌ Insufficient examples for {insight_type}")
            continue
        
        print(f"\nProcessing {insight_type}...")
        
        # Prepare data based on insight type
        if insight_type == 'negation':
            X = df['Customer_Transcript_Clean']
            y = df['Has_Negation']
            text_for_explanation = examples[insight_type]['Customer_Transcript_Clean']
        elif insight_type == 'agent_contamination':
            X = df['Agent_Transcript_Clean']
            y = df['Has_Agent_Contamination']
            text_for_explanation = examples[insight_type]['Agent_Transcript_Clean']
        else:  # qualifying_language
            X = df['Customer_Transcript_Clean']
            y = df['Has_Qualifying']
            text_for_explanation = examples[insight_type]['Customer_Transcript_Clean']
        
        # Filter valid data
        valid_mask = (X.str.len() > 10) & (X.notna())
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        if len(X_valid) < 20:
            print(f"❌ Insufficient valid data for {insight_type}")
            continue
        
        # Train model
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_valid, y_valid, test_size=0.2, random_state=42
            )
            
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ])
            
            pipeline.fit(X_train, y_train)
            accuracy = pipeline.score(X_test, y_test)
            print(f"✓ Model trained for {insight_type} (accuracy: {accuracy:.3f})")
            
            # Generate LIME explanations
            explainer = LimeTextExplainer(class_names=['No Pattern', 'Has Pattern'])
            
            insight_explanations = []
            
            for idx, text in enumerate(text_for_explanation.head(10)):  # Top 10 examples
                if len(str(text).strip()) < 10:
                    continue
                
                try:
                    explanation = explainer.explain_instance(
                        text, pipeline.predict_proba, num_features=8, num_samples=500
                    )
                    
                    insight_explanations.append({
                        'text': text,
                        'explanation': explanation,
                        'features': explanation.as_list()
                    })
                    
                except Exception as e:
                    print(f"❌ Error explaining example {idx}: {str(e)}")
                    continue
            
            explanations[insight_type] = insight_explanations
            print(f"✓ Generated {len(insight_explanations)} LIME explanations for {insight_type}")
            
        except Exception as e:
            print(f"❌ Error training model for {insight_type}: {str(e)}")
            continue
    
    # Step 4: Display results and export
    print("\nStep 4: Displaying results...")
    
    display_lime_results(explanations, examples)
    
    # Export results
    export_results(explanations, examples, df)
    
    return explanations, examples, df

def display_lime_results(explanations, examples):
    """Display LIME results in a readable format"""
    
    print("\n" + "="*80)
    print("LIME ANALYSIS RESULTS")
    print("="*80)
    
    for insight_type, insight_explanations in explanations.items():
        print(f"\n{insight_type.upper().replace('_', ' ')} ANALYSIS")
        print("-" * 60)
        
        if len(insight_explanations) == 0:
            print("No explanations generated")
            continue
        
        # Show top 3 examples
        for i, exp_data in enumerate(insight_explanations[:3]):
            print(f"\nExample {i+1}:")
            
            # Show text (truncated)
            text = exp_data['text']
            if len(text) > 300:
                text = text[:300] + "..."
            print(f"Text: {text}")
            
            # Show top features
            features = exp_data['features']
            print(f"Top LIME Features:")
            
            # Sort by absolute importance
            features_sorted = sorted(features, key=lambda x: abs(x[1]), reverse=True)
            
            for j, (feature, importance) in enumerate(features_sorted[:5]):
                direction = "SUPPORTS" if importance > 0 else "OPPOSES"
                print(f"  {j+1}. '{feature}' -> {importance:+.3f} ({direction})")
            
            print("-" * 40)
        
        # Show aggregate insights
        print(f"\nAGGREGATE INSIGHTS FOR {insight_type.upper()}:")
        
        # Collect all features and their importance
        all_features = {}
        for exp_data in insight_explanations:
            for feature, importance in exp_data['features']:
                if feature in all_features:
                    all_features[feature].append(importance)
                else:
                    all_features[feature] = [importance]
        
        # Calculate average importance
        avg_features = {
            feature: np.mean(importances) 
            for feature, importances in all_features.items()
        }
        
        # Top 5 most important features overall
        top_features = sorted(avg_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        print("Top 5 Most Important Features Overall:")
        for i, (feature, avg_importance) in enumerate(top_features, 1):
            direction = "SUPPORTS" if avg_importance > 0 else "OPPOSES"
            frequency = len(all_features[feature])
            print(f"  {i}. '{feature}' -> {avg_importance:+.3f} ({direction}) [appears in {frequency} explanations]")

def export_results(explanations, examples, df):
    """Export all results to Excel and HTML files"""
    
    from datetime import datetime
    import os
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nExporting results...")
    
    # 1. Export to Excel
    excel_filename = f'lime_complaints_analysis_{timestamp}.xlsx'
    
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        
        # Summary sheet
        summary_data = []
        for insight_type, insight_explanations in explanations.items():
            if len(insight_explanations) > 0:
                # Get top feature
                all_features = {}
                for exp_data in insight_explanations:
                    for feature, importance in exp_data['features']:
                        if feature in all_features:
                            all_features[feature].append(importance)
                        else:
                            all_features[feature] = [importance]
                
                if all_features:
                    avg_features = {f: np.mean(imp) for f, imp in all_features.items()}
                    top_feature = max(avg_features.items(), key=lambda x: abs(x[1]))
                    
                    summary_data.append({
                        'Insight': insight_type.replace('_', ' ').title(),
                        'Examples_Analyzed': len(insight_explanations),
                        'Top_Feature': top_feature[0],
                        'Top_Feature_Importance': top_feature[1],
                        'Unique_Features_Found': len(all_features)
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Examples sheets
        for insight_type, example_df in examples.items():
            if len(example_df) > 0:
                # Select key columns for export
                export_cols = ['UUID', 'Prosodica L1', 'Prosodica L2', 'Primary Marker']
                
                if insight_type == 'negation':
                    export_cols.extend(['Customer_Transcript_Clean'])
                elif insight_type == 'agent_contamination':
                    export_cols.extend(['Agent_Transcript_Clean', 'Customer_Transcript_Clean'])
                else:  # qualifying_language
                    export_cols.extend(['Customer_Transcript_Clean'])
                
                available_cols = [col for col in export_cols if col in example_df.columns]
                example_df[available_cols].head(20).to_excel(
                    writer, 
                    sheet_name=f'{insight_type.title()}_Examples', 
                    index=False
                )
        
        # Feature importance sheets
        for insight_type, insight_explanations in explanations.items():
            if len(insight_explanations) > 0:
                feature_data = []
                
                # Collect all features
                all_features = {}
                for exp_data in insight_explanations:
                    for feature, importance in exp_data['features']:
                        if feature in all_features:
                            all_features[feature].append(importance)
                        else:
                            all_features[feature] = [importance]
                
                # Create feature summary
                for feature, importances in all_features.items():
                    feature_data.append({
                        'Feature': feature,
                        'Avg_Importance': np.mean(importances),
                        'Min_Importance': np.min(importances),
                        'Max_Importance': np.max(importances),
                        'Frequency': len(importances),
                        'Direction': 'SUPPORTS' if np.mean(importances) > 0 else 'OPPOSES'
                    })
                
                if feature_data:
                    feature_df = pd.DataFrame(feature_data)
                    feature_df = feature_df.sort_values('Avg_Importance', key=abs, ascending=False)
                    feature_df.to_excel(
                        writer, 
                        sheet_name=f'{insight_type.title()}_Features', 
                        index=False
                    )
    
    print(f"✓ Excel results exported to: {excel_filename}")
    
    # 2. Export individual LIME explanations as HTML
    html_dir = f'lime_explanations_{timestamp}'
    os.makedirs(html_dir, exist_ok=True)
    
    html_count = 0
    for insight_type, insight_explanations in explanations.items():
        for i, exp_data in enumerate(insight_explanations[:5]):  # Top 5 per insight
            try:
                html_filename = os.path.join(html_dir, f'{insight_type}_example_{i+1}.html')
                exp_data['explanation'].save_to_file(html_filename)
                html_count += 1
            except Exception as e:
                print(f"❌ Error saving HTML for {insight_type} example {i+1}: {e}")
    
    print(f"✓ {html_count} HTML explanations exported to: {html_dir}/")
    
    # 3. Create actionable recommendations
    recommendations = create_actionable_recommendations(explanations)
    
    # Export recommendations
    recommendations_filename = f'lime_recommendations_{timestamp}.xlsx'
    recommendations.to_excel(recommendations_filename, index=False)
    print(f"✓ Recommendations exported to: {recommendations_filename}")
    
    return excel_filename, html_dir, recommendations_filename

def create_actionable_recommendations(explanations):
    """Create actionable recommendations based on LIME findings"""
    
    recommendations = []
    
    for insight_type, insight_explanations in explanations.items():
        if len(insight_explanations) == 0:
            continue
        
        # Extract top features
        all_features = {}
        for exp_data in insight_explanations:
            for feature, importance in exp_data['features']:
                if feature in all_features:
                    all_features[feature].append(importance)
                else:
                    all_features[feature] = [importance]
        
        if not all_features:
            continue
        
        avg_features = {f: np.mean(imp) for f, imp in all_features.items()}
        top_features = sorted(avg_features.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        if insight_type == 'negation':
            for i, (feature, importance) in enumerate(top_features):
                recommendations.append({
                    'Priority': f'HIGH-{i+1}',
                    'Insight': 'Context-Blind Negation',
                    'LIME_Feature': feature,
                    'Feature_Importance': importance,
                    'Action': f'Add context-aware rule for "{feature}"',
                    'Implementation': f'Use NEAR:5W operator: "{feature}" NOT NEAR:5W (question|information|help)',
                    'Expected_Impact': 'Reduce information-seeking negation false positives',
                    'Timeline': '1-2 weeks'
                })
        
        elif insight_type == 'agent_contamination':
            for i, (feature, importance) in enumerate(top_features):
                recommendations.append({
                    'Priority': f'HIGH-{i+1}',
                    'Insight': 'Agent Contamination',
                    'LIME_Feature': feature,
                    'Feature_Importance': importance,
                    'Action': f'Exclude agent pattern "{feature}"',
                    'Implementation': f'Add exclusion rule: complaint_query NOT NEAR:10W "{feature}"',
                    'Expected_Impact': 'Prevent agent explanations from triggering complaints',
                    'Timeline': '1-2 weeks'
                })
        
        elif insight_type == 'qualifying_language':
            for i, (feature, importance) in enumerate(top_features):
                recommendations.append({
                    'Priority': f'MEDIUM-{i+1}',
                    'Insight': 'Qualifying Language',
                    'LIME_Feature': feature,
                    'Feature_Importance': importance,
                    'Action': f'Enhance handling of "{feature}" in complaints',
                    'Implementation': f'Weight complaint keywords higher when "{feature}" is present',
                    'Expected_Impact': 'Better handle polite complaints with qualifying language',
                    'Timeline': '2-4 weeks'
                })
    
    recommendations_df = pd.DataFrame(recommendations)
    
    # Sort by priority
    priority_order = {'HIGH-1': 1, 'HIGH-2': 2, 'HIGH-3': 3, 'MEDIUM-1': 4, 'MEDIUM-2': 5, 'MEDIUM-3': 6}
    recommendations_df['Priority_Order'] = recommendations_df['Priority'].map(priority_order).fillna(99)
    recommendations_df = recommendations_df.sort_values('Priority_Order').drop('Priority_Order', axis=1)
    
    return recommendations_df

def create_monitoring_setup(explanations):
    """Create setup for ongoing LIME monitoring"""
    
    print("\nSETTING UP LIME MONITORING")
    print("-" * 40)
    
    monitoring_features = {}
    
    for insight_type, insight_explanations in explanations.items():
        if len(insight_explanations) == 0:
            continue
        
        # Get the most stable and important features for monitoring
        all_features = {}
        for exp_data in insight_explanations:
            for feature, importance in exp_data['features']:
                if feature in all_features:
                    all_features[feature].append(importance)
                else:
                    all_features[feature] = [importance]
        
        # Calculate stability (frequency) and importance
        stable_features = []
        for feature, importances in all_features.items():
            frequency = len(importances) / len(insight_explanations)
            avg_importance = np.mean(importances)
            stability_score = frequency * abs(avg_importance)
            
            if frequency >= 0.3:  # Appears in at least 30% of explanations
                stable_features.append({
                    'feature': feature,
                    'frequency': frequency,
                    'avg_importance': avg_importance,
                    'stability_score': stability_score
                })
        
        # Sort by stability score and take top 3
        stable_features.sort(key=lambda x: x['stability_score'], reverse=True)
        monitoring_features[insight_type] = stable_features[:3]
    
    # Create monitoring report
    monitoring_setup = []
    
    for insight_type, features in monitoring_features.items():
        for feature_data in features:
            monitoring_setup.append({
                'Insight': insight_type.replace('_', ' ').title(),
                'Feature_to_Monitor': feature_data['feature'],
                'Baseline_Importance': feature_data['avg_importance'],
                'Frequency_in_Explanations': feature_data['frequency'],
                'Monitoring_Alert': f"Alert if importance changes by >0.1 from baseline",
                'Review_Frequency': 'Monthly'
            })
    
    monitoring_df = pd.DataFrame(monitoring_setup)
    
    print("Key features to monitor:")
    for _, row in monitoring_df.iterrows():
        print(f"  {row['Insight']}: '{row['Feature_to_Monitor']}' (baseline: {row['Baseline_Importance']:+.3f})")
    
    return monitoring_df

# Main execution function
def main():
    """Main execution function - run this to perform the complete analysis"""
    
    print("LIME INTERPRETABILITY ANALYSIS FOR COMPLAINTS DETECTION")
    print("="*60)
    print("This analysis will:")
    print("1. Identify examples for Context-Blind Negation insight")
    print("2. Identify examples for Agent Contamination insight") 
    print("3. Identify examples for Qualifying Language insight")
    print("4. Generate LIME explanations for each insight")
    print("5. Export actionable recommendations")
    print("="*60)
    
    # Run the analysis
    try:
        data_file_path = 'Precision_Drop_Analysis_OG.xlsx'
        explanations, examples, df = quick_lime_analysis_for_complaints(data_file_path)
        
        if explanations:
            print("\n" + "="*60)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            # Create monitoring setup
            monitoring_df = create_monitoring_setup(explanations)
            
            print("\nFILES GENERATED:")
            print("1. Excel file with LIME analysis results")
            print("2. HTML files with individual LIME explanations")
            print("3. Excel file with actionable recommendations")
            print("4. Monitoring setup for ongoing tracking")
            
            return explanations, examples, df, monitoring_df
        else:
            print("❌ Analysis failed - no explanations generated")
            return None
            
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return None


if __name__ == "__main__":
    result = main()

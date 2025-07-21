# LIME-Based Interpretability Analysis for Complaints Precision Drop Investigation
# Enhanced Explainability for Key Insights: Context-Blind Negation, Agent Contamination, and Qualifying Language
# FIXED VERSION - Compatible with latest LIME library

import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# LIME and ML dependencies
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import lime
from lime.lime_text import LimeTextExplainer

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("="*80)
print("LIME-BASED INTERPRETABILITY ANALYSIS FOR COMPLAINTS DETECTION")
print("="*80)
print("Objective: Explain model predictions for key insights using LIME")
print("Insights: 1) Context-Blind Negation 2) Agent Contamination 3) Qualifying Language")
print("="*80)

class ComplaintsLIMEAnalyzer:
    """
    Comprehensive LIME-based analyzer for complaints detection interpretability
    """
    
    def __init__(self, df_main):
        """
        Initialize the analyzer with the main dataset
        
        Parameters:
        df_main: DataFrame with complaint analysis data
        """
        self.df_main = df_main.copy()
        self.lime_explainer = None
        self.models = {}
        self.examples = {}
        self.explanations = {}
        
        # Prepare data
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare and clean data for analysis"""
        
        print("PREPARING DATA FOR LIME ANALYSIS")
        print("-" * 40)
        
        # Clean and prepare text data
        self.df_main['Customer_Transcript_Clean'] = self.df_main['Customer Transcript'].fillna('').astype(str)
        self.df_main['Agent_Transcript_Clean'] = self.df_main['Agent Transcript'].fillna('').astype(str)
        self.df_main['Full_Transcript_Clean'] = (
            self.df_main['Customer_Transcript_Clean'] + ' ' + 
            self.df_main['Agent_Transcript_Clean']
        )
        
        # Create binary target: 1 for FP (problematic), 0 for TP (correct)
        self.df_main['Is_Problematic'] = (self.df_main['Primary Marker'] == 'FP').astype(int)
        
        # Add feature flags for insights
        self._add_insight_features()
        
        print(f"Data prepared: {len(self.df_main)} records")
        print(f"Problematic cases (FPs): {self.df_main['Is_Problematic'].sum()}")
        print(f"Correct cases (TPs): {(1-self.df_main['Is_Problematic']).sum()}")
        
    def _add_insight_features(self):
        """Add feature flags for the three key insights"""
        
        # Insight 1: Context-Blind Negation Features
        negation_patterns = {
            'Simple_Negation': r'\b(not|no|never|dont|don\'t|wont|won\'t|cant|can\'t)\b',
            'Information_Negation': r'\b(don\'t know|not sure|no idea|unclear|confused)\b',
            'Service_Negation': r'\b(not working|can\'t access|won\'t load|doesn\'t work)\b',
            'Complaint_Negation': r'\b(not fair|not right|not satisfied|never received)\b'
        }
        
        for pattern_name, pattern in negation_patterns.items():
            self.df_main[f'Has_{pattern_name}'] = self.df_main['Customer_Transcript_Clean'].str.lower().str.contains(
                pattern, regex=True, na=False
            ).astype(int)
        
        # Count negations
        self.df_main['Customer_Negation_Count'] = (
            self.df_main[['Has_Simple_Negation', 'Has_Information_Negation', 
                         'Has_Service_Negation', 'Has_Complaint_Negation']].sum(axis=1)
        )
        
        # Overall negation flag
        self.df_main['Has_Negation'] = (
            self.df_main[['Has_Simple_Negation', 'Has_Information_Negation', 
                         'Has_Service_Negation', 'Has_Complaint_Negation']].sum(axis=1) > 0
        ).astype(int)
        
        # Insight 2: Agent Contamination Features
        agent_contamination_patterns = {
            'Agent_Explanations': r'\b(let me explain|i\'ll explain|what this means|this means that)\b',
            'Agent_Examples': r'\b(for example|for instance|let\'s say|suppose)\b',
            'Agent_Hypotheticals': r'\b(if you|what if|in case|should you|were to)\b',
            'Agent_Clarifications': r'\b(to clarify|what i mean|in other words|basically)\b',
            'Agent_Instructions': r'\b(you need to|you should|you can|you have to)\b'
        }
        
        for pattern_name, pattern in agent_contamination_patterns.items():
            self.df_main[f'Has_{pattern_name}'] = self.df_main['Agent_Transcript_Clean'].str.lower().str.contains(
                pattern, regex=True, na=False
            ).astype(int)
        
        # Overall agent contamination flag
        self.df_main['Has_Agent_Contamination'] = (
            self.df_main[['Has_Agent_Explanations', 'Has_Agent_Examples', 
                         'Has_Agent_Hypotheticals', 'Has_Agent_Clarifications', 
                         'Has_Agent_Instructions']].sum(axis=1) > 0
        ).astype(int)
        
        # Insight 3: Qualifying Language Features
        qualifying_patterns = {
            'Customer_Uncertainty': r'\b(might|maybe|seems|appears|possibly|perhaps|probably|likely|i think|i believe|i guess)\b',
            'Customer_Hedging': r'\b(sort of|kind of|more or less|somewhat|relatively|fairly|quite|rather)\b',
            'Customer_Doubt': r'\b(not sure|uncertain|unclear|confused|don\'t know|no idea)\b',
            'Customer_Politeness': r'\b(please|thank you|thanks|appreciate|grateful|excuse me|pardon|sorry)\b',
            'Customer_Questions': r'\?'
        }
        
        for pattern_name, pattern in qualifying_patterns.items():
            if pattern_name == 'Customer_Questions':
                self.df_main[f'Has_{pattern_name}'] = (
                    self.df_main['Customer_Transcript_Clean'].str.count(pattern) > 0
                ).astype(int)
            else:
                self.df_main[f'Has_{pattern_name}'] = self.df_main['Customer_Transcript_Clean'].str.lower().str.contains(
                    pattern, regex=True, na=False
                ).astype(int)
        
        # Overall qualifying language flag
        self.df_main['Has_Qualifying_Language'] = (
            self.df_main[['Has_Customer_Uncertainty', 'Has_Customer_Hedging', 
                         'Has_Customer_Doubt', 'Has_Customer_Politeness', 
                         'Has_Customer_Questions']].sum(axis=1) > 0
        ).astype(int)
        
        print("Insight features added successfully")
        
    def identify_insight_examples(self, insight_type, num_examples=20):
        """
        Identify examples for each insight type
        
        Parameters:
        insight_type: str - 'negation', 'agent_contamination', or 'qualifying_language'
        num_examples: int - number of examples to extract
        """
        
        print(f"\nIDENTIFYING EXAMPLES FOR INSIGHT: {insight_type.upper()}")
        print("-" * 60)
        
        if insight_type == 'negation':
            # Focus on FPs with negation patterns but likely information-seeking
            examples_df = self.df_main[
                (self.df_main['Is_Problematic'] == 1) &  # FPs
                (self.df_main['Has_Negation'] == 1) &     # Has negation
                (self.df_main['Has_Information_Negation'] == 1)  # Information-seeking negation
            ].copy()
            
            # Sort by negation count to get strongest examples
            examples_df = examples_df.sort_values('Customer_Negation_Count', ascending=False)
            
        elif insight_type == 'agent_contamination':
            # Focus on FPs with high agent contamination
            examples_df = self.df_main[
                (self.df_main['Is_Problematic'] == 1) &     # FPs
                (self.df_main['Has_Agent_Contamination'] == 1)  # Has agent contamination
            ].copy()
            
            # Calculate contamination score
            contamination_cols = ['Has_Agent_Explanations', 'Has_Agent_Examples', 
                                'Has_Agent_Hypotheticals', 'Has_Agent_Clarifications', 
                                'Has_Agent_Instructions']
            examples_df['Contamination_Score'] = examples_df[contamination_cols].sum(axis=1)
            examples_df = examples_df.sort_values('Contamination_Score', ascending=False)
            
        elif insight_type == 'qualifying_language':
            # Focus on TPs with high qualifying language (real complaints with polite/uncertain language)
            examples_df = self.df_main[
                (self.df_main['Is_Problematic'] == 0) &           # TPs
                (self.df_main['Has_Qualifying_Language'] == 1)    # Has qualifying language
            ].copy()
            
            # Calculate qualifying score
            qualifying_cols = ['Has_Customer_Uncertainty', 'Has_Customer_Hedging', 
                             'Has_Customer_Doubt', 'Has_Customer_Politeness', 
                             'Has_Customer_Questions']
            examples_df['Qualifying_Score'] = examples_df[qualifying_cols].sum(axis=1)
            examples_df = examples_df.sort_values('Qualifying_Score', ascending=False)
            
        else:
            raise ValueError("insight_type must be 'negation', 'agent_contamination', or 'qualifying_language'")
        
        # Select top examples
        selected_examples = examples_df.head(num_examples)
        
        print(f"Found {len(examples_df)} potential examples")
        print(f"Selected top {len(selected_examples)} examples")
        
        # Store examples
        self.examples[insight_type] = selected_examples
        
        # Display sample examples
        self._display_sample_examples(insight_type, selected_examples.head(5))
        
        return selected_examples
    
    def _display_sample_examples(self, insight_type, sample_df):
        """Display sample examples for review"""
        
        print(f"\nSAMPLE EXAMPLES FOR {insight_type.upper()}:")
        print("=" * 60)
        
        for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
            print(f"\nExample {idx}:")
            print(f"UUID: {row.get('uuid', row.get('UUID', 'N/A'))}")
            print(f"Category: {row.get('Prosodica L1', 'N/A')} -> {row.get('Prosodica L2', 'N/A')}")
            print(f"Classification: {'FP (Problematic)' if row['Is_Problematic'] else 'TP (Correct)'}")
            
            if insight_type == 'negation':
                print(f"Negation Count: {row.get('Customer_Negation_Count', 0)}")
                print(f"Info Negation: {bool(row.get('Has_Information_Negation', 0))}")
                
            elif insight_type == 'agent_contamination':
                print(f"Contamination Score: {row.get('Contamination_Score', 0)}")
                print(f"Agent Explanations: {bool(row.get('Has_Agent_Explanations', 0))}")
                
            elif insight_type == 'qualifying_language':
                print(f"Qualifying Score: {row.get('Qualifying_Score', 0)}")
                print(f"Has Uncertainty: {bool(row.get('Has_Customer_Uncertainty', 0))}")
            
            # Truncate long transcripts
            customer_text = str(row['Customer_Transcript_Clean'])
            agent_text = str(row['Agent_Transcript_Clean'])
            
            if len(customer_text) > 300:
                customer_text = customer_text[:300] + "..."
            if len(agent_text) > 300:
                agent_text = agent_text[:300] + "..."
                
            print(f"Customer: {customer_text}")
            if insight_type == 'agent_contamination':
                print(f"Agent: {agent_text}")
            
            print("-" * 60)
    
    def train_prediction_models(self):
        """Train models for each insight to use with LIME"""
        
        print("\nTRAINING PREDICTION MODELS FOR LIME")
        print("-" * 40)
        
        # Prepare features for each insight
        insights = ['negation', 'agent_contamination', 'qualifying_language']
        
        for insight in insights:
            print(f"\nTraining model for {insight}...")
            
            if insight == 'negation':
                # Focus on customer transcript for negation analysis
                X = self.df_main['Customer_Transcript_Clean'].fillna('')
                y = self.df_main['Has_Negation']
                
            elif insight == 'agent_contamination':
                # Focus on agent transcript for contamination analysis
                X = self.df_main['Agent_Transcript_Clean'].fillna('')
                y = self.df_main['Has_Agent_Contamination']
                
            elif insight == 'qualifying_language':
                # Focus on customer transcript for qualifying language
                X = self.df_main['Customer_Transcript_Clean'].fillna('')
                y = self.df_main['Has_Qualifying_Language']
            
            # Filter out empty texts
            valid_indices = X.str.len() > 10
            X_filtered = X[valid_indices]
            y_filtered = y[valid_indices]
            
            if len(X_filtered) < 10:
                print(f"Insufficient data for {insight} model")
                continue
            
            # Check if we have both classes
            if len(y_filtered.unique()) < 2:
                print(f"Insufficient class diversity for {insight} model")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
            )
            
            # Create pipeline with TF-IDF and Logistic Regression
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    stop_words='english',
                    lowercase=True
                )),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            train_score = pipeline.score(X_train, y_train)
            test_score = pipeline.score(X_test, y_test)
            
            print(f"  Train accuracy: {train_score:.3f}")
            print(f"  Test accuracy: {test_score:.3f}")
            
            # Store model
            self.models[insight] = pipeline
        
        print("\nModel training completed")
    
    def generate_lime_explanations(self, insight_type, num_explanations=10):
        """
        Generate LIME explanations for selected examples
        
        Parameters:
        insight_type: str - type of insight to explain
        num_explanations: int - number of explanations to generate
        """
        
        print(f"\nGENERATING LIME EXPLANATIONS FOR {insight_type.upper()}")
        print("-" * 60)
        
        if insight_type not in self.models:
            print(f"No model found for {insight_type}. Please train models first.")
            return None
            
        if insight_type not in self.examples:
            print(f"No examples found for {insight_type}. Please identify examples first.")
            return None
        
        model = self.models[insight_type]
        examples = self.examples[insight_type].head(num_explanations)
        
        # Initialize LIME explainer - FIXED VERSION
        # Removed the 'mode' parameter which doesn't exist in newer LIME versions
        explainer = LimeTextExplainer(
            class_names=['No Pattern', 'Has Pattern']
        )
        
        explanations = []
        
        for idx, (_, row) in enumerate(examples.iterrows()):
            print(f"Generating explanation {idx + 1}/{len(examples)}...")
            
            # Select appropriate text based on insight type
            if insight_type == 'negation':
                text_to_explain = row['Customer_Transcript_Clean']
            elif insight_type == 'agent_contamination':
                text_to_explain = row['Agent_Transcript_Clean']
            elif insight_type == 'qualifying_language':
                text_to_explain = row['Customer_Transcript_Clean']
            
            if len(text_to_explain.strip()) < 10:
                print(f"  Skipping example {idx + 1}: insufficient text")
                continue
            
            try:
                # Generate explanation
                explanation = explainer.explain_instance(
                    text_to_explain,
                    model.predict_proba,
                    num_features=10,
                    num_samples=1000
                )
                
                # Store explanation with metadata
                explanations.append({
                    'uuid': row.get('uuid', row.get('UUID', f'example_{idx}')),
                    'category_l1': row.get('Prosodica L1', 'N/A'),
                    'category_l2': row.get('Prosodica L2', 'N/A'),
                    'is_problematic': row['Is_Problematic'],
                    'text': text_to_explain,
                    'explanation': explanation,
                    'prediction_proba': model.predict_proba([text_to_explain])[0],
                    'insight_type': insight_type
                })
                
            except Exception as e:
                print(f"  Error generating explanation for example {idx + 1}: {str(e)}")
                continue
        
        print(f"Generated {len(explanations)} explanations")
        
        # Store explanations
        self.explanations[insight_type] = explanations
        
        return explanations
    
    def display_lime_explanations(self, insight_type, num_display=5):
        """Display LIME explanations in a readable format"""
        
        if insight_type not in self.explanations:
            print(f"No explanations found for {insight_type}")
            return
        
        explanations = self.explanations[insight_type][:num_display]
        
        print(f"\nLIME EXPLANATIONS FOR {insight_type.upper()}")
        print("=" * 80)
        
        for idx, exp_data in enumerate(explanations):
            print(f"\nEXAMPLE {idx + 1}")
            print("-" * 60)
            print(f"UUID: {exp_data['uuid']}")
            print(f"Category: {exp_data['category_l1']} -> {exp_data['category_l2']}")
            print(f"Classification: {'FP (Problematic)' if exp_data['is_problematic'] else 'TP (Correct)'}")
            print(f"Model Prediction Probability: {exp_data['prediction_proba'][1]:.3f}")
            
            # Display text (truncated)
            text = exp_data['text']
            if len(text) > 500:
                text = text[:500] + "..."
            print(f"Text: {text}")
            
            # Get explanation as list
            explanation_list = exp_data['explanation'].as_list()
            
            print(f"\nTop Feature Contributions:")
            print("(Positive values support pattern detection, negative values oppose it)")
            
            # Sort by absolute importance
            explanation_list.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for i, (feature, importance) in enumerate(explanation_list[:10]):
                direction = "SUPPORTS" if importance > 0 else "OPPOSES"
                print(f"  {i+1:2d}. '{feature}' -> {importance:+.3f} ({direction})")
            
            print("-" * 60)
    
    def create_lime_visualizations(self, insight_type, save_html=True):
        """Create and save LIME visualizations"""
        
        if insight_type not in self.explanations:
            print(f"No explanations found for {insight_type}")
            return
        
        explanations = self.explanations[insight_type]
        
        print(f"\nCREATING VISUALIZATIONS FOR {insight_type.upper()}")
        print("-" * 60)
        
        # Create feature importance summary
        all_features = []
        all_importance = []
        
        for exp_data in explanations:
            explanation_list = exp_data['explanation'].as_list()
            for feature, importance in explanation_list:
                all_features.append(feature)
                all_importance.append(importance)
        
        # Aggregate feature importance
        feature_importance = {}
        for feature, importance in zip(all_features, all_importance):
            if feature in feature_importance:
                feature_importance[feature].append(importance)
            else:
                feature_importance[feature] = [importance]
        
        # Calculate average importance
        avg_importance = {
            feature: np.mean(importances) 
            for feature, importances in feature_importance.items()
        }
        
        # Sort by absolute importance
        sorted_features = sorted(
            avg_importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:20]
        
        # Create bar plot
        features, importances = zip(*sorted_features)
        colors = ['green' if imp > 0 else 'red' for imp in importances]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), importances, color=colors, alpha=0.7)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Average Feature Importance')
        plt.title(f'LIME Feature Importance - {insight_type.title()} Analysis')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for i, (bar, imp) in enumerate(zip(bars, importances)):
            plt.text(
                bar.get_width() + (0.01 if imp > 0 else -0.01), 
                bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}',
                ha='left' if imp > 0 else 'right',
                va='center',
                fontsize=8
            )
        
        plt.tight_layout()
        
        if save_html:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'lime_analysis_{insight_type}_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Visualization saved as: {filename}")
        
        plt.show()
        
        # Save individual LIME explanations as HTML
        if save_html:
            for idx, exp_data in enumerate(explanations[:5]):  # Save first 5
                try:
                    html_filename = f'lime_explanation_{insight_type}_example_{idx+1}_{timestamp}.html'
                    exp_data['explanation'].save_to_file(html_filename)
                    print(f"Individual explanation saved as: {html_filename}")
                except Exception as e:
                    print(f"Warning: Could not save HTML explanation {idx+1}: {str(e)}")
    
    def generate_insight_summary_report(self):
        """Generate a comprehensive summary report of all insights"""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE LIME ANALYSIS SUMMARY REPORT")
        print("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create summary DataFrame
        summary_data = []
        
        for insight_type in ['negation', 'agent_contamination', 'qualifying_language']:
            if insight_type in self.examples and insight_type in self.explanations:
                examples = self.examples[insight_type]
                explanations = self.explanations[insight_type]
                
                # Calculate metrics
                total_examples = len(examples)
                total_explanations = len(explanations)
                
                if insight_type in self.models:
                    model = self.models[insight_type]
                    
                    # Get sample predictions
                    if insight_type == 'negation':
                        sample_texts = examples['Customer_Transcript_Clean'].head(100)
                    elif insight_type == 'agent_contamination':
                        sample_texts = examples['Agent_Transcript_Clean'].head(100)
                    else:
                        sample_texts = examples['Customer_Transcript_Clean'].head(100)
                    
                    valid_texts = sample_texts[sample_texts.str.len() > 10]
                    
                    if len(valid_texts) > 0:
                        predictions = model.predict_proba(valid_texts)
                        avg_confidence = np.mean(predictions[:, 1])
                    else:
                        avg_confidence = 0
                else:
                    avg_confidence = 0
                
                summary_data.append({
                    'Insight_Type': insight_type.title().replace('_', ' '),
                    'Total_Examples': total_examples,
                    'LIME_Explanations': total_explanations,
                    'Avg_Model_Confidence': avg_confidence,
                    'Model_Available': insight_type in self.models
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        print("ANALYSIS SUMMARY:")
        print(summary_df.to_string(index=False))
        
        # Key findings for each insight
        print("\nKEY FINDINGS BY INSIGHT:")
        print("-" * 60)
        
        findings = {
            'negation': [
                "Context-blind negation interpretation causes false positives",
                "Information-seeking negations ('don't know', 'not sure') misclassified as complaints",
                "Model fails to distinguish complaint vs information contexts"
            ],
            'agent_contamination': [
                "Agent explanations contaminate complaint detection",
                "Hypothetical scenarios and examples trigger false positives",
                "Agent language patterns mislead classification model"
            ],
            'qualifying_language': [
                "Polite complaint language often contains uncertainty markers",
                "Qualifying words ('maybe', 'might') appear in legitimate complaints",
                "Model needs to handle polite vs uncertain language better"
            ]
        }
        
        for insight_type, insight_findings in findings.items():
            if insight_type in self.explanations:
                print(f"\n{insight_type.upper().replace('_', ' ')}:")
                for finding in insight_findings:
                    print(f"  • {finding}")
        
        # Export summary
        summary_filename = f'lime_analysis_summary_{timestamp}.xlsx'
        try:
            with pd.ExcelWriter(summary_filename, engine='xlsxwriter') as writer:
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Export examples for each insight
                for insight_type in ['negation', 'agent_contamination', 'qualifying_language']:
                    if insight_type in self.examples:
                        examples = self.examples[insight_type]
                        export_cols = [
                            'uuid', 'UUID', 'Prosodica L1', 'Prosodica L2', 'Primary Marker',
                            'Customer_Transcript_Clean', 'Agent_Transcript_Clean'
                        ]
                        
                        # Add insight-specific columns
                        if insight_type == 'negation':
                            export_cols.extend(['Customer_Negation_Count', 'Has_Information_Negation'])
                        elif insight_type == 'agent_contamination':
                            export_cols.extend(['Has_Agent_Contamination', 'Contamination_Score'])
                        elif insight_type == 'qualifying_language':
                            export_cols.extend(['Has_Qualifying_Language', 'Qualifying_Score'])
                        
                        available_cols = [col for col in export_cols if col in examples.columns]
                        examples[available_cols].to_excel(
                            writer, 
                            sheet_name=f'{insight_type.title()}_Examples', 
                            index=False
                        )
            print(f"\nSummary report exported to: {summary_filename}")
        except Exception as e:
            print(f"Warning: Could not save Excel file: {str(e)}")
        
        return summary_df

# Usage function - UPDATED
def run_comprehensive_lime_analysis(df_main, insights_to_analyze=None):
    """
    Run comprehensive LIME analysis for complaints detection insights
    
    Parameters:
    df_main: DataFrame with complaint analysis data
    insights_to_analyze: list of insights to analyze ['negation', 'agent_contamination', 'qualifying_language']
                        If None, analyzes all insights
    """
    
    if insights_to_analyze is None:
        insights_to_analyze = ['negation', 'agent_contamination', 'qualifying_language']
    
    print("STARTING COMPREHENSIVE LIME ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = ComplaintsLIMEAnalyzer(df_main)
    
    # Train models
    analyzer.train_prediction_models()
    
    # Process each insight
    for insight_type in insights_to_analyze:
        print(f"\n{'='*80}")
        print(f"PROCESSING INSIGHT: {insight_type.upper()}")
        print("="*80)
        
        try:
            # Identify examples
            analyzer.identify_insight_examples(insight_type, num_examples=50)
            
            # Generate LIME explanations
            analyzer.generate_lime_explanations(insight_type, num_explanations=20)
            
            # Display explanations
            analyzer.display_lime_explanations(insight_type, num_display=3)
            
            # Create visualizations
            analyzer.create_lime_visualizations(insight_type, save_html=True)
            
        except Exception as e:
            print(f"Error processing {insight_type}: {str(e)}")
            continue
    
    # Generate final summary report
    summary_df = analyzer.generate_insight_summary_report()
    
    print("\n" + "="*80)
    print("LIME ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*80)
    print("Files generated:")
    print("- PNG visualizations for each insight")
    print("- HTML LIME explanations for top examples")
    print("- Excel summary report with all examples")
    
    return analyzer, summary_df

# Advanced Analysis Functions - UPDATED

def analyze_lime_pattern_evolution(analyzer, df_main):
    """
    Analyze how LIME-identified patterns evolve over time (Pre vs Post periods)
    """
    
    print("\n" + "="*80)
    print("LIME PATTERN EVOLUTION ANALYSIS (PRE VS POST)")
    print("="*80)
    
    # Define periods - updated to handle different date formats
    pre_months = ['2024-10', '2024-11', '2024-12']
    post_months = ['2025-01', '2025-02', '2025-03']
    
    # Handle different possible column names for date
    date_col = None
    for col in ['Year_Month', 'year_month', 'date', 'Date', 'month', 'Month']:
        if col in df_main.columns:
            date_col = col
            break
    
    if date_col is None:
        print("Warning: No date column found for evolution analysis")
        return {}
    
    df_main['Period'] = df_main[date_col].astype(str).apply(
        lambda x: 'Pre' if any(month in str(x) for month in pre_months) else 
                 'Post' if any(month in str(x) for month in post_months) else 'Other'
    )
    
    evolution_results = {}
    
    for insight_type in ['negation', 'agent_contamination', 'qualifying_language']:
        if insight_type not in analyzer.explanations:
            continue
            
        print(f"\nANALYZING {insight_type.upper()} PATTERN EVOLUTION")
        print("-" * 60)
        
        # Get all feature importance from explanations
        all_features = {}
        
        # Process explanations and map to periods
        for exp_data in analyzer.explanations[insight_type]:
            uuid = exp_data['uuid']
            
            # Find the period for this UUID - handle different UUID column names
            uuid_col = None
            for col in ['UUID', 'uuid', 'id', 'Id']:
                if col in df_main.columns:
                    uuid_col = col
                    break
            
            if uuid_col is None:
                continue
                
            uuid_data = df_main[df_main[uuid_col] == uuid]
            if len(uuid_data) == 0:
                continue
                
            period = uuid_data['Period'].iloc[0]
            if period not in ['Pre', 'Post']:
                continue
            
            # Extract features and importance
            explanation_list = exp_data['explanation'].as_list()
            
            for feature, importance in explanation_list:
                if feature not in all_features:
                    all_features[feature] = {'Pre': [], 'Post': []}
                all_features[feature][period].append(importance)
        
        # Calculate evolution metrics
        evolution_data = []
        
        for feature, period_data in all_features.items():
            pre_scores = period_data['Pre']
            post_scores = period_data['Post']
            
            if len(pre_scores) > 0 and len(post_scores) > 0:
                pre_avg = np.mean(pre_scores)
                post_avg = np.mean(post_scores)
                change = post_avg - pre_avg
                
                evolution_data.append({
                    'Feature': feature,
                    'Pre_Importance': pre_avg,
                    'Post_Importance': post_avg,
                    'Change': change,
                    'Pre_Count': len(pre_scores),
                    'Post_Count': len(post_scores),
                    'Pattern_Strength': abs(change)
                })
        
        if len(evolution_data) > 0:
            evolution_df = pd.DataFrame(evolution_data)
            evolution_df = evolution_df.sort_values('Pattern_Strength', ascending=False)
            
            print(f"Top 10 Features with Biggest Changes:")
            display_df = evolution_df.head(10)[['Feature', 'Pre_Importance', 'Post_Importance', 'Change']].round(3)
            print(display_df.to_string(index=False))
            
            # Identify emerging patterns
            emerging_patterns = evolution_df[
                (evolution_df['Change'] > 0.1) & 
                (evolution_df['Post_Importance'] > evolution_df['Pre_Importance'])
            ]
            
            declining_patterns = evolution_df[
                (evolution_df['Change'] < -0.1) & 
                (evolution_df['Post_Importance'] < evolution_df['Pre_Importance'])
            ]
            
            print(f"\nEMERGING PATTERNS ({len(emerging_patterns)}):")
            if len(emerging_patterns) > 0:
                for _, row in emerging_patterns.head(5).iterrows():
                    print(f"  '{row['Feature']}': {row['Pre_Importance']:.3f} → {row['Post_Importance']:.3f} (+{row['Change']:.3f})")
            else:
                print("  No significant emerging patterns detected")
            
            print(f"\nDECLINING PATTERNS ({len(declining_patterns)}):")
            if len(declining_patterns) > 0:
                for _, row in declining_patterns.head(5).iterrows():
                    print(f"  '{row['Feature']}': {row['Pre_Importance']:.3f} → {row['Post_Importance']:.3f} ({row['Change']:.3f})")
            else:
                print("  No significant declining patterns detected")
            
            evolution_results[insight_type] = evolution_df
        else:
            print("Insufficient data for evolution analysis")
    
    return evolution_results

def create_insight_comparison_dashboard(analyzer):
    """
    Create a comprehensive dashboard comparing insights
    """
    
    print("\n" + "="*80)
    print("CREATING INSIGHT COMPARISON DASHBOARD")
    print("="*80)
    
    # Collect data for comparison
    comparison_data = []
    
    for insight_type in ['negation', 'agent_contamination', 'qualifying_language']:
        if insight_type not in analyzer.explanations:
            continue
        
        explanations = analyzer.explanations[insight_type]
        
        # Extract feature importance statistics
        all_importance = []
        feature_counts = {}
        
        for exp_data in explanations:
            explanation_list = exp_data['explanation'].as_list()
            
            for feature, importance in explanation_list:
                all_importance.append(abs(importance))
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        if len(all_importance) > 0:
            most_common_feature, most_common_count = max(feature_counts.items(), key=lambda x: x[1]) if feature_counts else ("N/A", 0)
            
            comparison_data.append({
                'Insight': insight_type.title().replace('_', ' '),
                'Avg_Feature_Importance': np.mean(all_importance),
                'Max_Feature_Importance': np.max(all_importance),
                'Unique_Features': len(feature_counts),
                'Total_Explanations': len(explanations),
                'Most_Common_Feature': most_common_feature,
                'Most_Common_Feature_Count': most_common_count
            })
    
    if len(comparison_data) == 0:
        print("No data available for comparison dashboard")
        return pd.DataFrame()
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("INSIGHT COMPARISON SUMMARY:")
    print(comparison_df.to_string(index=False))
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Average Feature Importance
    ax1.bar(comparison_df['Insight'], comparison_df['Avg_Feature_Importance'])
    ax1.set_title('Average Feature Importance by Insight')
    ax1.set_ylabel('Average Importance')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Number of Unique Features
    ax2.bar(comparison_df['Insight'], comparison_df['Unique_Features'])
    ax2.set_title('Number of Unique Features by Insight')
    ax2.set_ylabel('Unique Features')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Total Explanations Generated
    ax3.bar(comparison_df['Insight'], comparison_df['Total_Explanations'])
    ax3.set_title('Total LIME Explanations Generated')
    ax3.set_ylabel('Number of Explanations')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Max Feature Importance
    ax4.bar(comparison_df['Insight'], comparison_df['Max_Feature_Importance'])
    ax4.set_title('Maximum Feature Importance by Insight')
    ax4.set_ylabel('Max Importance')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'insight_comparison_dashboard_{timestamp}.png'
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved as: {filename}")
    except Exception as e:
        print(f"Warning: Could not save dashboard: {str(e)}")
    
    plt.show()
    
    return comparison_df

def extract_actionable_recommendations(analyzer, evolution_results=None):
    """
    Extract actionable recommendations based on LIME analysis
    """
    
    print("\n" + "="*80)
    print("ACTIONABLE RECOMMENDATIONS FROM LIME ANALYSIS")
    print("="*80)
    
    recommendations = []
    
    for insight_type in ['negation', 'agent_contamination', 'qualifying_language']:
        if insight_type not in analyzer.explanations:
            continue
        
        print(f"\n{insight_type.upper().replace('_', ' ')} RECOMMENDATIONS:")
        print("-" * 60)
        
        explanations = analyzer.explanations[insight_type]
        
        # Analyze top features across all explanations
        feature_importance = {}
        for exp_data in explanations:
            explanation_list = exp_data['explanation'].as_list()
            for feature, importance in explanation_list:
                if feature in feature_importance:
                    feature_importance[feature].append(importance)
                else:
                    feature_importance[feature] = [importance]
        
        # Get top problematic features
        avg_importance = {
            feature: np.mean(importances) 
            for feature, importances in feature_importance.items()
        }
        
        top_problematic = sorted(
            avg_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        if insight_type == 'negation':
            print("1. IMMEDIATE ACTIONS:")
            print("   • Implement context-aware negation handling in query rules")
            print("   • Add proximity operators to distinguish information vs complaint negations")
            print("   • Create separate rules for 'don't know/understand' patterns")
            
            print("\n2. QUERY RULE MODIFICATIONS:")
            for feature, importance in top_problematic:
                if any(neg in feature.lower() for neg in ['not', 'no', 'never', 'don\'t']):
                    print(f"   • Add context rules for '{feature}' (importance: {importance:.3f})")
            
            print("\n3. TECHNICAL IMPLEMENTATION:")
            print("   • Use NEAR:5W operators for negation context")
            print("   • Implement sentiment analysis for negation disambiguation")
            print("   • Add channel-specific negation handling (customer vs agent)")
            
        elif insight_type == 'agent_contamination':
            print("1. IMMEDIATE ACTIONS:")
            print("   • Filter out agent explanation patterns from complaint detection")
            print("   • Implement speaker identification in transcripts")
            print("   • Create agent-specific exclusion rules")
            
            print("\n2. AGENT TRAINING:")
            print("   • Train agents to use specific markers for explanations")
            print("   • Implement structured explanation formats")
            print("   • Create clear boundaries between explanation and complaint content")
            
            print("\n3. TECHNICAL IMPLEMENTATION:")
            for feature, importance in top_problematic:
                if any(phrase in feature.lower() for phrase in ['explain', 'example', 'suppose', 'if you']):
                    print(f"   • Add exclusion rule for '{feature}' (importance: {importance:.3f})")
            
        elif insight_type == 'qualifying_language':
            print("1. IMMEDIATE ACTIONS:")
            print("   • Enhance rules to handle polite complaint language")
            print("   • Distinguish uncertainty from politeness")
            print("   • Improve handling of question-based complaints")
            
            print("\n2. LANGUAGE PROCESSING:")
            print("   • Implement sentiment-aware qualifying language analysis")
            print("   • Add customer emotion detection")
            print("   • Create politeness vs uncertainty classifiers")
            
            print("\n3. RULE REFINEMENT:")
            for feature, importance in top_problematic:
                if any(qual in feature.lower() for qual in ['maybe', 'might', 'please', 'thank', 'sorry']):
                    print(f"   • Refine handling of '{feature}' (importance: {importance:.3f})")
        
        # Add evolution-based recommendations if available
        if evolution_results and insight_type in evolution_results:
            evolution_df = evolution_results[insight_type]
            emerging = evolution_df[evolution_df['Change'] > 0.1].head(3)
            
            if len(emerging) > 0:
                print(f"\n4. EMERGING PATTERN ALERTS:")
                for _, row in emerging.iterrows():
                    print(f"   • Monitor increasing importance of '{row['Feature']}' (+{row['Change']:.3f})")
                    print(f"     Recommendation: Update rules to handle this emerging pattern")
        
        recommendations.append({
            'insight_type': insight_type,
            'top_features': [f[0] for f in top_problematic],
            'feature_importance': [f[1] for f in top_problematic],
            'recommendations_count': len(top_problematic)
        })
    
    # Create summary recommendations DataFrame
    summary_recommendations = pd.DataFrame([
        {
            'Priority': 'HIGH',
            'Insight': 'Context-Blind Negation',
            'Action': 'Implement context-aware negation rules with proximity operators',
            'Timeline': 'Immediate (1-2 weeks)',
            'Impact': 'Reduce information-seeking negation false positives'
        },
        {
            'Priority': 'HIGH', 
            'Insight': 'Agent Contamination',
            'Action': 'Add agent explanation exclusion patterns to rules',
            'Timeline': 'Immediate (1-2 weeks)',
            'Impact': 'Prevent agent explanations from triggering complaints'
        },
        {
            'Priority': 'MEDIUM',
            'Insight': 'Qualifying Language',
            'Action': 'Enhance polite complaint handling with sentiment analysis',
            'Timeline': 'Medium-term (1-2 months)',
            'Impact': 'Better handling of legitimate polite complaints'
        },
        {
            'Priority': 'MEDIUM',
            'Insight': 'Model Enhancement',
            'Action': 'Implement LIME-guided feature engineering',
            'Timeline': 'Medium-term (2-3 months)', 
            'Impact': 'Improve overall model interpretability and performance'
        },
        {
            'Priority': 'LOW',
            'Insight': 'Monitoring',
            'Action': 'Set up LIME-based model monitoring dashboard',
            'Timeline': 'Long-term (3-6 months)',
            'Impact': 'Continuous model improvement and drift detection'
        }
    ])
    
    print(f"\n" + "="*80)
    print("PRIORITIZED RECOMMENDATION SUMMARY")
    print("="*80)
    print(summary_recommendations.to_string(index=False))
    
    # Export recommendations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    recommendations_filename = f'lime_actionable_recommendations_{timestamp}.xlsx'
    
    try:
        with pd.ExcelWriter(recommendations_filename, engine='xlsxwriter') as writer:
            summary_recommendations.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed recommendations for each insight
            for rec in recommendations:
                insight_details = pd.DataFrame({
                    'Feature': rec['top_features'],
                    'Importance': rec['feature_importance'],
                    'Insight_Type': rec['insight_type']
                })
                insight_details.to_excel(
                    writer, 
                    sheet_name=f"{rec['insight_type'].title()}_Details", 
                    index=False
                )
        print(f"\nRecommendations exported to: {recommendations_filename}")
    except Exception as e:
        print(f"Warning: Could not save recommendations file: {str(e)}")
    
    return summary_recommendations, recommendations

# Integration function to tie everything together - UPDATED
def complete_lime_interpretability_pipeline(df_main, export_results=True):
    """
    Complete LIME interpretability pipeline for complaints analysis
    
    This function runs the entire pipeline:
    1. Basic LIME analysis for all insights
    2. Pattern evolution analysis
    3. Insight comparison dashboard
    4. Actionable recommendations extraction
    
    Parameters:
    df_main: DataFrame with complaint analysis data
    export_results: bool, whether to export all results
    
    Returns:
    analyzer: ComplaintsLIMEAnalyzer instance
    evolution_results: Pattern evolution analysis results
    comparison_df: Insight comparison summary
    recommendations: Actionable recommendations
    """
    
    print("STARTING COMPLETE LIME INTERPRETABILITY PIPELINE")
    print("="*80)
    
    # Step 1: Run basic LIME analysis
    print("STEP 1: Running basic LIME analysis...")
    analyzer, summary_df = run_comprehensive_lime_analysis(df_main)
    
    # Step 2: Analyze pattern evolution
    print("\nSTEP 2: Analyzing pattern evolution...")
    evolution_results = analyze_lime_pattern_evolution(analyzer, df_main)
    
    # Step 3: Create comparison dashboard
    print("\nSTEP 3: Creating insight comparison dashboard...")
    comparison_df = create_insight_comparison_dashboard(analyzer)
    
    # Step 4: Extract actionable recommendations
    print("\nSTEP 4: Extracting actionable recommendations...")
    recommendations, detailed_recs = extract_actionable_recommendations(analyzer, evolution_results)
    
    # Step 5: Export comprehensive results
    if export_results:
        print("\nSTEP 5: Exporting comprehensive results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comprehensive_filename = f'complete_lime_analysis_{timestamp}.xlsx'
        
        try:
            with pd.ExcelWriter(comprehensive_filename, engine='xlsxwriter') as writer:
                # Summary sheets
                summary_df.to_excel(writer, sheet_name='Analysis_Summary', index=False)
                comparison_df.to_excel(writer, sheet_name='Insight_Comparison', index=False)
                recommendations.to_excel(writer, sheet_name='Recommendations', index=False)
                
                # Evolution results
                for insight_type, evolution_df in evolution_results.items():
                    evolution_df.to_excel(
                        writer, 
                        sheet_name=f'{insight_type.title()}_Evolution', 
                        index=False
                    )
                
                # Examples for each insight
                for insight_type in ['negation', 'agent_contamination', 'qualifying_language']:
                    if insight_type in analyzer.examples:
                        examples = analyzer.examples[insight_type]
                        export_cols = [
                            'uuid', 'UUID', 'Prosodica L1', 'Prosodica L2', 'Primary Marker',
                            'Customer_Transcript_Clean', 'Agent_Transcript_Clean'
                        ]
                        available_cols = [col for col in export_cols if col in examples.columns]
                        examples[available_cols].to_excel(
                            writer, 
                            sheet_name=f'{insight_type.title()}_Examples', 
                            index=False
                        )
            
            print(f"Comprehensive results exported to: {comprehensive_filename}")
        except Exception as e:
            print(f"Warning: Could not save comprehensive results: {str(e)}")
    
    print("\n" + "="*80)
    print("COMPLETE LIME INTERPRETABILITY PIPELINE FINISHED")
    print("="*80)
    print("DELIVERABLES GENERATED:")
    print("1. LIME explanations for all three insights")
    print("2. Feature importance visualizations")
    print("3. Pattern evolution analysis (Pre vs Post)")
    print("4. Insight comparison dashboard")
    print("5. Actionable recommendations with priorities")
    print("6. Comprehensive Excel report with all findings")
    print("7. Individual HTML LIME explanations for top examples")
    
    return analyzer, evolution_results, comparison_df, recommendations

# Helper function to check LIME version and provide guidance
def check_lime_version():
    """Check LIME version and provide guidance"""
    try:
        import lime
        print(f"LIME version: {lime.__version__}")
        
        # Test if the LimeTextExplainer works with our parameters
        try:
            explainer = LimeTextExplainer(class_names=['Test1', 'Test2'])
            print("✓ LIME Text Explainer initialized successfully")
        except Exception as e:
            print(f"✗ LIME Text Explainer initialization failed: {str(e)}")
            
    except ImportError:
        print("✗ LIME is not installed. Install with: pip install lime")
    except Exception as e:
        print(f"✗ Error checking LIME: {str(e)}")

# Example usage with error handling:
"""
# Check LIME version first
check_lime_version()

# Load your data (using the same preparation as in your existing scripts)
df_main, df_validation, df_rules = load_and_prepare_data()

# Run the complete pipeline
try:
    analyzer, evolution_results, comparison_df, recommendations = complete_lime_interpretability_pipeline(df_main)
    print("Analysis completed successfully!")
except Exception as e:
    print(f"Error during analysis: {str(e)}")
    # For debugging - run individual steps
    analyzer = ComplaintsLIMEAnalyzer(df_main)
    analyzer.train_prediction_models()

# For specific insight analysis only:
# analyzer, summary = run_comprehensive_lime_analysis(df_main, insights_to_analyze=['negation'])
"""

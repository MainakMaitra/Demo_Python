# Core Data Handling
pandas>=1.3.0
numpy>=1.21.0

# NLP
spacy>=3.5.0
transformers>=4.36.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
wordcloud>=1.8.0
plotly>=5.10.0
networkx>=2.6.3

# Optional: for Jupyter display consistency
ipython>=7.0.0

# Required spaCy model
# This is not installed via pip, so include as instruction:
# RUN: python -m spacy download en_core_web_sm

pip install -r requirements.txt
python -m spacy download en_core_web_sm







# ============================================================================
# PART 1: CONTEXTUAL NEGATION SCOPE DETECTION
# Advanced NLP implementation to detect negation patterns and their scope
# ============================================================================

import pandas as pd
import numpy as np
import spacy
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import networkx as nx
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CONTEXTUAL NEGATION SCOPE DETECTION")
print("Advanced NLP Analysis for Precision Drop Investigation")
print("=" * 80)

class ContextualNegationDetector:
    """
    Advanced negation detection system that identifies negation cues,
    their scope, and contextual meaning in complaint transcripts
    """
    
    def __init__(self):
        # Load spaCy model for dependency parsing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Define negation cues with contextual weights
        self.negation_cues = {
            'simple_negation': {
                'patterns': ['not', 'no', 'never', 'none', 'nothing', 'nobody', 'nowhere'],
                'weight': 1.0,
                'complaint_relevance': 0.7
            },
            'contracted_negation': {
                'patterns': ["don't", "won't", "can't", "isn't", "aren't", "wasn't", "weren't", 
                           "doesn't", "didn't", "haven't", "hasn't", "hadn't"],
                'weight': 1.2,
                'complaint_relevance': 0.8
            },
            'weak_negation': {
                'patterns': ['hardly', 'barely', 'scarcely', 'rarely', 'seldom'],
                'weight': 0.6,
                'complaint_relevance': 0.5
            },
            'emphatic_negation': {
                'patterns': ['absolutely not', 'definitely not', 'certainly not', 'never ever'],
                'weight': 2.0,
                'complaint_relevance': 0.9
            }
        }
        
        # Complaint-specific terms that change meaning when negated
        self.complaint_indicators = [
            'received', 'working', 'resolved', 'fixed', 'helped', 'satisfied', 'happy',
            'correct', 'right', 'fair', 'reasonable', 'clear', 'understand', 'explained'
        ]
        
        # Information-seeking terms that don't indicate complaints when negated
        self.information_indicators = [
            'know', 'understand', 'sure', 'clear', 'aware', 'familiar', 'remember'
        ]
    
    def extract_negation_patterns(self, text, speaker='customer'):
        """
        Extract negation patterns with their scope and context
        """
        if not self.nlp:
            return self._simple_negation_extraction(text, speaker)
        
        doc = self.nlp(text.lower())
        negation_instances = []
        
        for token in doc:
            if self._is_negation_cue(token.text):
                negation_info = {
                    'cue': token.text,
                    'position': token.i,
                    'scope_tokens': [],
                    'scope_text': '',
                    'context_window': '',
                    'negation_type': self._classify_negation_type(token.text),
                    'complaint_relevance': 0.0,
                    'information_relevance': 0.0,
                    'speaker': speaker
                }
                
                # Find scope using dependency parsing
                scope_tokens = self._find_negation_scope(token, doc)
                negation_info['scope_tokens'] = [t.text for t in sorted(set(scope_tokens), key=lambda x: x.i)]
                negation_info['scope_text'] = ' '.join([t.text for t in sorted(set(scope_tokens), key=lambda x: x.i)])
                
                # Extract context window (5 words before and after)
                start_idx = max(0, token.i - 5)
                end_idx = min(len(doc), token.i + 6)
                context_tokens = doc[start_idx:end_idx]
                negation_info['context_window'] = ' '.join([t.text for t in context_tokens])
                
                # Calculate relevance scores
                negation_info['complaint_relevance'] = self._calculate_complaint_relevance(scope_tokens)
                negation_info['information_relevance'] = self._calculate_information_relevance(scope_tokens)
                
                negation_instances.append(negation_info)
        
        return negation_instances
    
    def _is_negation_cue(self, token_text):
        """Check if token is a negation cue"""
        for neg_type, neg_info in self.negation_cues.items():
            if token_text in neg_info['patterns']:
                return True
        return False
    
    def _classify_negation_type(self, token_text):
        """Classify the type of negation"""
        for neg_type, neg_info in self.negation_cues.items():
            if token_text in neg_info['patterns']:
                return neg_type
        return 'unknown'
    
    def _find_negation_scope(self, neg_token, doc):
        """Find the scope of negation using dependency parsing"""
        scope_tokens = [neg_token]
        
        # Get children of negation token
        for child in neg_token.children:
            if child.dep_ in ['dobj', 'attr', 'acomp', 'advmod', 'prep']:
                scope_tokens.append(child)
                scope_tokens.extend(list(child.subtree))
        
        # Get head if negation modifies a verb
        if neg_token.head.pos_ == 'VERB':
            scope_tokens.append(neg_token.head)
            # Get direct objects and complements
            for child in neg_token.head.children:
                if child.dep_ in ['dobj', 'iobj', 'attr', 'acomp']:
                    scope_tokens.extend(list(child.subtree))
        
        return list(set(scope_tokens))
    
    def _calculate_complaint_relevance(self, scope_tokens):
        """Calculate how relevant the negated scope is to complaints"""
        scope_text = ' '.join([t.text for t in sorted(set(scope_tokens), key=lambda x: x.i)])
        score = 0.0
        
        for indicator in self.complaint_indicators:
            if indicator in scope_text:
                score += 0.2
        
        # Boost score for complaint-specific patterns
        complaint_patterns = [
            'not working', 'never received', 'not resolved', 'not satisfied',
            'not fair', 'not right', 'not correct', 'not helped'
        ]
        
        for pattern in complaint_patterns:
            if pattern in scope_text:
                score += 0.5
        
        return min(score, 1.0)
    
    def _calculate_information_relevance(self, scope_tokens):
        """Calculate how relevant the negated scope is to information seeking"""
        scope_text = ' '.join([t.text for t in sorted(set(scope_tokens), key=lambda x: x.i)])
        score = 0.0
        
        for indicator in self.information_indicators:
            if indicator in scope_text:
                score += 0.3
        
        # Boost score for information-specific patterns
        info_patterns = [
            "don't know", "don't understand", "not sure", "not clear",
            "not aware", "not familiar", "don't remember"
        ]
        
        for pattern in info_patterns:
            if pattern in scope_text:
                score += 0.6
        
        return min(score, 1.0)
    
    def _simple_negation_extraction(self, text, speaker):
        """Fallback method when spaCy is not available"""
        negation_instances = []
        text_lower = text.lower()
        
        # Simple pattern matching
        all_patterns = []
        for neg_type, neg_info in self.negation_cues.items():
            for pattern in neg_info['patterns']:
                all_patterns.append((pattern, neg_type))
        
        for pattern, neg_type in all_patterns:
            for match in re.finditer(r'\b' + re.escape(pattern) + r'\b', text_lower):
                start_pos = max(0, match.start() - 50)
                end_pos = min(len(text_lower), match.end() + 50)
                context = text_lower[start_pos:end_pos]
                
                negation_info = {
                    'cue': pattern,
                    'position': match.start(),
                    'scope_text': context[match.start()-start_pos:match.end()-start_pos+20],
                    'context_window': context,
                    'negation_type': neg_type,
                    'complaint_relevance': self._simple_complaint_relevance(context),
                    'information_relevance': self._simple_information_relevance(context),
                    'speaker': speaker
                }
                
                negation_instances.append(negation_info)
        
        return negation_instances
    
    def _simple_complaint_relevance(self, context):
        """Simple complaint relevance calculation"""
        complaint_words = ['problem', 'issue', 'wrong', 'error', 'broken', 'failed']
        score = sum(1 for word in complaint_words if word in context) * 0.2
        return min(score, 1.0)
    
    def _simple_information_relevance(self, context):
        """Simple information relevance calculation"""
        info_words = ['how', 'what', 'when', 'where', 'why', 'help', 'explain']
        score = sum(1 for word in info_words if word in context) * 0.2
        return min(score, 1.0)

def analyze_negation_patterns(df, detector):
    """
    Comprehensive analysis of negation patterns in the dataset
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE NEGATION PATTERN ANALYSIS")
    print("=" * 60)
    
    results = []
    
    # Process each transcript
    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"Processing transcript {idx+1}/{len(df)}...")
        
        # Extract negations from customer transcript
        customer_negations = detector.extract_negation_patterns(
            str(row['Customer Transcript']), 'customer'
        )
        
        # Extract negations from agent transcript  
        agent_negations = detector.extract_negation_patterns(
            str(row['Agent Transcript']), 'agent'
        )
        
        # Combine results
        all_negations = customer_negations + agent_negations
        
        for neg in all_negations:
            result = {
                'UUID': row['UUID'],
                'variable5': row['variable5'],
                'Primary_Marker': row['Primary Marker'],
                'Prosodica_L1': row['Prosodica L1'],
                'Prosodica_L2': row['Prosodica L2'],
                'Year_Month': row['Year_Month'],
                'Period': row['Period'],
                'Speaker': neg['speaker'],
                'Negation_Cue': neg['cue'],
                'Negation_Type': neg['negation_type'],
                'Scope_Text': neg.get('scope_text', ''),
                'Context_Window': neg['context_window'],
                'Complaint_Relevance': neg['complaint_relevance'],
                'Information_Relevance': neg['information_relevance'],
                'Is_Complaint_Negation': neg['complaint_relevance'] > 0.3,
                'Is_Information_Negation': neg['information_relevance'] > 0.3
            }
            results.append(result)
    
    return pd.DataFrame(results)

def create_negation_visualizations(negation_df, output_dir='negation_analysis'):
    """
    Create comprehensive visualizations for negation analysis
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("CREATING NEGATION VISUALIZATIONS")
    print("=" * 60)
    
    # 1. Negation Type Distribution by TP/FP
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution of negation types
    tp_negations = negation_df[negation_df['Primary_Marker'] == 'TP']
    fp_negations = negation_df[negation_df['Primary_Marker'] == 'FP']
    
    negation_type_counts_tp = tp_negations['Negation_Type'].value_counts()
    negation_type_counts_fp = fp_negations['Negation_Type'].value_counts()
    
    axes[0,0].bar(negation_type_counts_tp.index, negation_type_counts_tp.values, alpha=0.7, color='green')
    axes[0,0].set_title('Negation Types in True Positives')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    axes[0,1].bar(negation_type_counts_fp.index, negation_type_counts_fp.values, alpha=0.7, color='red')
    axes[0,1].set_title('Negation Types in False Positives')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Complaint vs Information Relevance
    complaint_by_marker = negation_df.groupby('Primary_Marker')['Complaint_Relevance'].mean()
    info_by_marker = negation_df.groupby('Primary_Marker')['Information_Relevance'].mean()
    
    x = ['TP', 'FP']
    axes[1,0].bar(x, [complaint_by_marker['TP'], complaint_by_marker['FP']], 
                  alpha=0.7, color=['green', 'red'])
    axes[1,0].set_title('Average Complaint Relevance Score')
    axes[1,0].set_ylabel('Relevance Score')
    
    axes[1,1].bar(x, [info_by_marker['TP'], info_by_marker['FP']], 
                  alpha=0.7, color=['green', 'red'])
    axes[1,1].set_title('Average Information Relevance Score')
    axes[1,1].set_ylabel('Relevance Score')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/negation_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Monthly Evolution of Negation Patterns
    monthly_negation = negation_df.groupby(['Year_Month', 'Primary_Marker']).agg({
        'Complaint_Relevance': 'mean',
        'Information_Relevance': 'mean',
        'UUID': 'count'
    }).reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Complaint Relevance Over Time', 'Information Relevance Over Time',
                       'Negation Volume Over Time', 'Relevance Score Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Complaint relevance trend
    tp_monthly = monthly_negation[monthly_negation['Primary_Marker'] == 'TP']
    fp_monthly = monthly_negation[monthly_negation['Primary_Marker'] == 'FP']
    
    fig.add_trace(
        go.Scatter(x=tp_monthly['Year_Month'], y=tp_monthly['Complaint_Relevance'],
                  mode='lines+markers', name='TP Complaint Rel', line=dict(color='green')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=fp_monthly['Year_Month'], y=fp_monthly['Complaint_Relevance'],
                  mode='lines+markers', name='FP Complaint Rel', line=dict(color='red')),
        row=1, col=1
    )
    
    # Information relevance trend
    fig.add_trace(
        go.Scatter(x=tp_monthly['Year_Month'], y=tp_monthly['Information_Relevance'],
                  mode='lines+markers', name='TP Info Rel', line=dict(color='darkgreen')),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=fp_monthly['Year_Month'], y=fp_monthly['Information_Relevance'],
                  mode='lines+markers', name='FP Info Rel', line=dict(color='darkred')),
        row=1, col=2
    )
    
    # Volume trend
    fig.add_trace(
        go.Bar(x=tp_monthly['Year_Month'], y=tp_monthly['UUID'],
               name='TP Volume', marker_color='green', opacity=0.7),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=fp_monthly['Year_Month'], y=fp_monthly['UUID'],
               name='FP Volume', marker_color='red', opacity=0.7),
        row=2, col=1
    )
    
    # Distribution comparison
    fig.add_trace(
        go.Histogram(x=tp_negations['Complaint_Relevance'], name='TP Complaint',
                    opacity=0.7, marker_color='green', nbinsx=20),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Histogram(x=fp_negations['Complaint_Relevance'], name='FP Complaint',
                    opacity=0.7, marker_color='red', nbinsx=20),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Negation Pattern Analysis Over Time")
    fig.write_html(f'{output_dir}/negation_temporal_analysis.html')
    
    # 3. Context Word Cloud
    print("Creating context word clouds...")
    
    # High complaint relevance contexts
    high_complaint_contexts = negation_df[
        negation_df['Complaint_Relevance'] > 0.5
    ]['Context_Window'].str.cat(sep=' ')
    
    # High information relevance contexts  
    high_info_contexts = negation_df[
        negation_df['Information_Relevance'] > 0.5
    ]['Context_Window'].str.cat(sep=' ')
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    if high_complaint_contexts:
        wordcloud_complaint = WordCloud(width=800, height=400, background_color='white').generate(high_complaint_contexts)
        axes[0].imshow(wordcloud_complaint, interpolation='bilinear')
        axes[0].set_title('High Complaint Relevance Contexts', fontsize=16)
        axes[0].axis('off')
    
    if high_info_contexts:
        wordcloud_info = WordCloud(width=800, height=400, background_color='white').generate(high_info_contexts)
        axes[1].imshow(wordcloud_info, interpolation='bilinear')
        axes[1].set_title('High Information Relevance Contexts', fontsize=16)
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/negation_context_wordclouds.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Negation analysis visualizations saved to {output_dir}/")

# Main execution function for Part 1
def run_contextual_negation_analysis(df):
    """
    Main function to run contextual negation scope detection
    """
    print("Initializing Contextual Negation Detector...")
    detector = ContextualNegationDetector()
    
    print("Analyzing negation patterns in transcripts...")
    negation_df = analyze_negation_patterns(df, detector)
    
    print("Creating visualizations...")
    create_negation_visualizations(negation_df)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("NEGATION ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"Total negation instances found: {len(negation_df)}")
    print(f"Customer negations: {len(negation_df[negation_df['Speaker'] == 'customer'])}")
    print(f"Agent negations: {len(negation_df[negation_df['Speaker'] == 'agent'])}")
    
    # Relevance comparison
    tp_complaint_rel = negation_df[negation_df['Primary_Marker'] == 'TP']['Complaint_Relevance'].mean()
    fp_complaint_rel = negation_df[negation_df['Primary_Marker'] == 'FP']['Complaint_Relevance'].mean()
    
    tp_info_rel = negation_df[negation_df['Primary_Marker'] == 'TP']['Information_Relevance'].mean()
    fp_info_rel = negation_df[negation_df['Primary_Marker'] == 'FP']['Information_Relevance'].mean()
    
    print(f"\nComplaint Relevance Scores:")
    print(f"  TPs: {tp_complaint_rel:.3f}")
    print(f"  FPs: {fp_complaint_rel:.3f}")
    print(f"  Difference: {tp_complaint_rel - fp_complaint_rel:+.3f}")
    
    print(f"\nInformation Relevance Scores:")
    print(f"  TPs: {tp_info_rel:.3f}")
    print(f"  FPs: {fp_info_rel:.3f}")
    print(f"  Difference: {tp_info_rel - fp_info_rel:+.3f}")
    
    return negation_df, detector

print("\nContextual Negation Scope Detection module ready!")
print("Call run_contextual_negation_analysis(df) to execute the analysis.")

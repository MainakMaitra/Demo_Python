# ============================================================================
# PART 2: COMPLAINT EXPRESSION LEXICON MAPPING
# Creating dynamic complaint lexicons and analyzing TP vs FP patterns
# ============================================================================

import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("COMPLAINT EXPRESSION LEXICON MAPPING")
print("Dynamic discovery of complaint vs non-complaint expressions")
print("=" * 80)

def extract_complaint_lexicons(df):
    """
    Dynamically extract complaint and non-complaint lexicons from the data
    """
    
    print("\n" + "=" * 60)
    print("EXTRACTING COMPLAINT LEXICONS FROM DATA")
    print("=" * 60)
    
    # Separate TP and FP transcripts
    tp_transcripts = df[df['Primary Marker'] == 'TP']
    fp_transcripts = df[df['Primary Marker'] == 'FP']
    
    print(f"TP transcripts: {len(tp_transcripts)}")
    print(f"FP transcripts: {len(fp_transcripts)}")
    
    # Combine customer and agent transcripts for analysis
    tp_texts = []
    fp_texts = []
    
    for _, row in tp_transcripts.iterrows():
        combined_text = str(row['Customer Transcript']) + " " + str(row['Agent Transcript'])
        tp_texts.append(combined_text.lower())
    
    for _, row in fp_transcripts.iterrows():
        combined_text = str(row['Customer Transcript']) + " " + str(row['Agent Transcript'])
        fp_texts.append(combined_text.lower())
    
    # Use TF-IDF to find distinctive terms
    print("\n1. Finding distinctive terms using TF-IDF...")
    
    # Create separate vectorizers for TP and FP
    vectorizer_tp = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 3),
        min_df=5
    )
    
    vectorizer_fp = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 3),
        min_df=5
    )
    
    try:
        # Fit TF-IDF on TP and FP separately
        tp_tfidf = vectorizer_tp.fit_transform(tp_texts)
        fp_tfidf = vectorizer_fp.fit_transform(fp_texts)
        
        # Get feature names and scores
        tp_features = vectorizer_tp.get_feature_names_out()
        fp_features = vectorizer_fp.get_feature_names_out()
        
        # Calculate mean TF-IDF scores
        tp_scores = np.mean(tp_tfidf.toarray(), axis=0)
        fp_scores = np.mean(fp_tfidf.toarray(), axis=0)
        
        # Create feature dictionaries
        tp_feature_dict = dict(zip(tp_features, tp_scores))
        fp_feature_dict = dict(zip(fp_features, fp_scores))
        
        # Find terms that appear in both and calculate discrimination
        common_features = set(tp_features) & set(fp_features)
        distinctive_terms = []
        
        for term in common_features:
            tp_score = tp_feature_dict[term]
            fp_score = fp_feature_dict[term]
            
            # Calculate discrimination ratio
            if fp_score > 0:
                discrimination_ratio = tp_score / fp_score
            else:
                discrimination_ratio = float('inf')
            
            distinctive_terms.append({
                'term': term,
                'tp_score': tp_score,
                'fp_score': fp_score,
                'discrimination_ratio': discrimination_ratio,
                'complaint_likelihood': tp_score / (tp_score + fp_score) if (tp_score + fp_score) > 0 else 0
            })
        
        # Sort by discrimination ratio
        distinctive_terms_df = pd.DataFrame(distinctive_terms)
        distinctive_terms_df = distinctive_terms_df.sort_values('discrimination_ratio', ascending=False)
        
        print(f"Found {len(distinctive_terms)} common terms for analysis")
        
    except Exception as e:
        print(f"TF-IDF analysis failed: {e}")
        return None, None
    
    return distinctive_terms_df, (tp_feature_dict, fp_feature_dict)

def create_dynamic_complaint_lexicons(distinctive_terms_df):
    """
    Create dynamic complaint lexicons based on discrimination analysis
    """
    
    print("\n" + "=" * 60)
    print("CREATING DYNAMIC COMPLAINT LEXICONS")
    print("=" * 60)
    
    # Define thresholds for categorization
    high_complaint_threshold = 2.0  # TP score 2x higher than FP
    low_complaint_threshold = 0.5   # FP score 2x higher than TP
    
    # Categorize terms
    complaint_lexicons = {
        'Strong_Complaint_Indicators': [],
        'Moderate_Complaint_Indicators': [],
        'Neutral_Terms': [],
        'Information_Seeking_Indicators': [],
        'Strong_Information_Indicators': []
    }
    
    for _, row in distinctive_terms_df.iterrows():
        term = row['term']
        ratio = row['discrimination_ratio']
        complaint_likelihood = row['complaint_likelihood']
        
        if ratio >= high_complaint_threshold and complaint_likelihood > 0.7:
            complaint_lexicons['Strong_Complaint_Indicators'].append({
                'term': term,
                'ratio': ratio,
                'likelihood': complaint_likelihood
            })
        elif ratio >= 1.2 and complaint_likelihood > 0.6:
            complaint_lexicons['Moderate_Complaint_Indicators'].append({
                'term': term,
                'ratio': ratio,
                'likelihood': complaint_likelihood
            })
        elif ratio <= low_complaint_threshold and complaint_likelihood < 0.3:
            complaint_lexicons['Strong_Information_Indicators'].append({
                'term': term,
                'ratio': ratio,
                'likelihood': complaint_likelihood
            })
        elif ratio <= 0.8 and complaint_likelihood < 0.4:
            complaint_lexicons['Information_Seeking_Indicators'].append({
                'term': term,
                'ratio': ratio,
                'likelihood': complaint_likelihood
            })
        else:
            complaint_lexicons['Neutral_Terms'].append({
                'term': term,
                'ratio': ratio,
                'likelihood': complaint_likelihood
            })
    
    # Print lexicon summary
    print("Dynamic Complaint Lexicons Created:")
    print("=" * 40)
    
    for category, terms in complaint_lexicons.items():
        print(f"\n{category} ({len(terms)} terms):")
        
        # Sort by likelihood and show top 10
        sorted_terms = sorted(terms, key=lambda x: x['likelihood'], reverse=True)[:10]
        
        for term_info in sorted_terms:
            print(f"  {term_info['term']}: likelihood={term_info['likelihood']:.3f}, ratio={term_info['ratio']:.2f}")
    
    return complaint_lexicons

def analyze_lexicon_patterns_by_period(df, complaint_lexicons):
    """
    Analyze how complaint lexicon usage changed from Pre to Post period
    """
    
    print("\n" + "=" * 60)
    print("LEXICON PATTERN ANALYSIS BY PERIOD")
    print("=" * 60)
    
    # Prepare period analysis
    periods = ['Pre', 'Post']
    markers = ['TP', 'FP']
    
    lexicon_analysis = {}
    
    for category, terms in complaint_lexicons.items():
        if not terms:  # Skip empty categories
            continue
            
        category_analysis = {}
        term_list = [term_info['term'] for term_info in terms]
        
        for period in periods:
            period_data = df[df['Period'] == period]
            
            for marker in markers:
                marker_data = period_data[period_data['Primary Marker'] == marker]
                
                if len(marker_data) == 0:
                    continue
                
                # Count term occurrences
                term_counts = []
                total_transcripts = len(marker_data)
                
                for _, row in marker_data.iterrows():
                    combined_text = str(row['Customer Transcript']).lower() + " " + str(row['Agent Transcript']).lower()
                    
                    transcript_term_count = 0
                    for term in term_list:
                        # Use word boundaries for accurate matching
                        pattern = r'\b' + re.escape(term) + r'\b'
                        matches = len(re.findall(pattern, combined_text))
                        transcript_term_count += matches
                    
                    term_counts.append(transcript_term_count)
                
                # Calculate statistics
                avg_terms_per_transcript = np.mean(term_counts) if term_counts else 0
                transcripts_with_terms = sum(1 for count in term_counts if count > 0)
                term_presence_rate = transcripts_with_terms / total_transcripts if total_transcripts > 0 else 0
                
                category_analysis[f'{period}_{marker}'] = {
                    'avg_terms': avg_terms_per_transcript,
                    'presence_rate': term_presence_rate,
                    'total_transcripts': total_transcripts,
                    'transcripts_with_terms': transcripts_with_terms
                }
        
        lexicon_analysis[category] = category_analysis
    
    # Print analysis results
    print("Lexicon Usage Analysis by Period:")
    print("=" * 40)
    
    for category, analysis in lexicon_analysis.items():
        print(f"\n{category}:")
        
        if 'Pre_TP' in analysis and 'Post_TP' in analysis:
            pre_tp = analysis['Pre_TP']
            post_tp = analysis['Post_TP']
            pre_fp = analysis.get('Pre_FP', {})
            post_fp = analysis.get('Post_FP', {})
            
            print(f"  TP Average Terms: {pre_tp['avg_terms']:.2f} -> {post_tp['avg_terms']:.2f} "
                  f"({post_tp['avg_terms'] - pre_tp['avg_terms']:+.2f})")
            
            if pre_fp and post_fp:
                print(f"  FP Average Terms: {pre_fp['avg_terms']:.2f} -> {post_fp['avg_terms']:.2f} "
                      f"({post_fp['avg_terms'] - pre_fp['avg_terms']:+.2f})")
                
                print(f"  TP Presence Rate: {pre_tp['presence_rate']:.3f} -> {post_tp['presence_rate']:.3f}")
                print(f"  FP Presence Rate: {pre_fp['presence_rate']:.3f} -> {post_fp['presence_rate']:.3f}")
                
                # Calculate risk metrics
                pre_risk = (pre_fp['avg_terms'] / max(pre_tp['avg_terms'], 0.001))
                post_risk = (post_fp['avg_terms'] / max(post_tp['avg_terms'], 0.001))
                
                print(f"  Risk Factor: {pre_risk:.3f} -> {post_risk:.3f} ({post_risk - pre_risk:+.3f})")
    
    return lexicon_analysis

def identify_problematic_expressions(df, complaint_lexicons):
    """
    Identify specific expressions that are causing classification problems
    """
    
    print("\n" + "=" * 60)
    print("IDENTIFYING PROBLEMATIC EXPRESSIONS")
    print("=" * 60)
    
    problematic_expressions = []
    
    # Focus on expressions that appear more in FPs than TPs
    info_indicators = complaint_lexicons.get('Strong_Information_Indicators', []) + \
                     complaint_lexicons.get('Information_Seeking_Indicators', [])
    
    for term_info in info_indicators:
        term = term_info['term']
        
        # Find contexts where this term appears in FPs
        fp_contexts = []
        tp_contexts = []
        
        fp_data = df[df['Primary Marker'] == 'FP']
        tp_data = df[df['Primary Marker'] == 'TP']
        
        # Extract contexts for FPs
        for _, row in fp_data.iterrows():
            combined_text = str(row['Customer Transcript']).lower() + " " + str(row['Agent Transcript']).lower()
            
            pattern = r'\b' + re.escape(term) + r'\b'
            for match in re.finditer(pattern, combined_text):
                start = max(0, match.start() - 50)
                end = min(len(combined_text), match.end() + 50)
                context = combined_text[start:end]
                
                fp_contexts.append({
                    'context': context,
                    'period': row['Period'],
                    'uuid': row['UUID']
                })
        
        # Extract contexts for TPs (for comparison)
        for _, row in tp_data.iterrows():
            combined_text = str(row['Customer Transcript']).lower() + " " + str(row['Agent Transcript']).lower()
            
            pattern = r'\b' + re.escape(term) + r'\b'
            for match in re.finditer(pattern, combined_text):
                start = max(0, match.start() - 50)
                end = min(len(combined_text), match.end() + 50)
                context = combined_text[start:end]
                
                tp_contexts.append({
                    'context': context,
                    'period': row['Period'],
                    'uuid': row['UUID']
                })
        
        if len(fp_contexts) > 0:
            problematic_expressions.append({
                'term': term,
                'fp_count': len(fp_contexts),
                'tp_count': len(tp_contexts),
                'fp_contexts': fp_contexts[:5],  # Sample contexts
                'tp_contexts': tp_contexts[:5],   # Sample contexts
                'likelihood': term_info['likelihood'],
                'ratio': term_info['ratio']
            })
    
    # Sort by FP frequency
    problematic_expressions.sort(key=lambda x: x['fp_count'], reverse=True)
    
    print("Most Problematic Expressions (appearing frequently in FPs):")
    print("=" * 55)
    
    for expr in problematic_expressions[:10]:
        print(f"\nTerm: '{expr['term']}'")
        print(f"  FP occurrences: {expr['fp_count']}")
        print(f"  TP occurrences: {expr['tp_count']}")
        print(f"  Complaint likelihood: {expr['likelihood']:.3f}")
        print(f"  Sample FP context: {expr['fp_contexts'][0]['context'][:100]}..." if expr['fp_contexts'] else "  No FP contexts")
        
        if expr['tp_contexts']:
            print(f"  Sample TP context: {expr['tp_contexts'][0]['context'][:100]}...")
    
    return problematic_expressions

def create_expression_performance_matrix(df, complaint_lexicons):
    """
    Create a performance matrix showing how each expression type performs
    across different periods and markers
    """
    
    print("\n" + "=" * 60)
    print("EXPRESSION PERFORMANCE MATRIX")
    print("=" * 60)
    
    # Create performance matrix
    performance_data = []
    
    for category, terms in complaint_lexicons.items():
        if not terms:
            continue
            
        term_list = [term_info['term'] for term_info in terms]
        
        for period in ['Pre', 'Post']:
            for marker in ['TP', 'FP']:
                period_marker_data = df[(df['Period'] == period) & (df['Primary Marker'] == marker)]
                
                if len(period_marker_data) == 0:
                    continue
                
                # Calculate metrics for this category
                total_matches = 0
                transcripts_with_matches = 0
                
                for _, row in period_marker_data.iterrows():
                    combined_text = str(row['Customer Transcript']).lower() + " " + str(row['Agent Transcript']).lower()
                    
                    transcript_matches = 0
                    for term in term_list:
                        pattern = r'\b' + re.escape(term) + r'\b'
                        matches = len(re.findall(pattern, combined_text))
                        transcript_matches += matches
                    
                    total_matches += transcript_matches
                    if transcript_matches > 0:
                        transcripts_with_matches += 1
                
                avg_matches = total_matches / len(period_marker_data) if len(period_marker_data) > 0 else 0
                presence_rate = transcripts_with_matches / len(period_marker_data) if len(period_marker_data) > 0 else 0
                
                performance_data.append({
                    'Category': category,
                    'Period': period,
                    'Marker': marker,
                    'Avg_Matches': avg_matches,
                    'Presence_Rate': presence_rate,
                    'Total_Transcripts': len(period_marker_data),
                    'Transcripts_With_Matches': transcripts_with_matches
                })
    
    performance_df = pd.DataFrame(performance_data)
    
    # Create pivot tables for easier analysis
    avg_matches_pivot = performance_df.pivot_table(
        index=['Category', 'Marker'], 
        columns='Period', 
        values='Avg_Matches', 
        fill_value=0
    )
    
    presence_rate_pivot = performance_df.pivot_table(
        index=['Category', 'Marker'], 
        columns='Period', 
        values='Presence_Rate', 
        fill_value=0
    )
    
    print("Average Matches per Transcript:")
    print(avg_matches_pivot.round(3))
    
    print("\nPresence Rate (% of transcripts containing terms):")
    print(presence_rate_pivot.round(3))
    
    return performance_df, avg_matches_pivot, presence_rate_pivot

def generate_lexicon_insights(lexicon_analysis, problematic_expressions):
    """
    Generate key insights from lexicon analysis
    """
    
    print("\n" + "=" * 60)
    print("KEY LEXICON INSIGHTS")
    print("=" * 60)
    
    insights = []
    
    # Analyze each category for insights
    for category, analysis in lexicon_analysis.items():
        if 'Pre_TP' not in analysis or 'Post_TP' not in analysis:
            continue
            
        pre_tp = analysis['Pre_TP']
        post_tp = analysis['Post_TP']
        pre_fp = analysis.get('Pre_FP', {})
        post_fp = analysis.get('Post_FP', {})
        
        if not pre_fp or not post_fp:
            continue
        
        # Calculate changes
        tp_change = post_tp['avg_terms'] - pre_tp['avg_terms']
        fp_change = post_fp['avg_terms'] - pre_fp['avg_terms']
        
        pre_risk = pre_fp['avg_terms'] / max(pre_tp['avg_terms'], 0.001)
        post_risk = post_fp['avg_terms'] / max(post_tp['avg_terms'], 0.001)
        risk_change = post_risk - pre_risk
        
        # Generate insights based on patterns
        if abs(tp_change) > 0.5 or abs(fp_change) > 0.5:
            insights.append({
                'category': category,
                'type': 'usage_change',
                'severity': 'high' if max(abs(tp_change), abs(fp_change)) > 1.0 else 'medium',
                'description': f"Significant usage change: TP {tp_change:+.2f}, FP {fp_change:+.2f}",
                'tp_change': tp_change,
                'fp_change': fp_change
            })
        
        if abs(risk_change) > 0.2:
            insights.append({
                'category': category,
                'type': 'risk_change',
                'severity': 'high' if abs(risk_change) > 0.5 else 'medium',
                'description': f"Risk factor changed by {risk_change:+.3f} (from {pre_risk:.3f} to {post_risk:.3f})",
                'risk_change': risk_change
            })
        
        # Check for problematic patterns
        if category in ['Strong_Information_Indicators', 'Information_Seeking_Indicators']:
            if post_fp['avg_terms'] > post_tp['avg_terms']:
                insights.append({
                    'category': category,
                    'type': 'misclassification_risk',
                    'severity': 'high',
                    'description': f"Information indicators appearing more in FPs than TPs",
                    'post_fp_avg': post_fp['avg_terms'],
                    'post_tp_avg': post_tp['avg_terms']
                })
    
    # Sort insights by severity
    high_severity = [i for i in insights if i['severity'] == 'high']
    medium_severity = [i for i in insights if i['severity'] == 'medium']
    
    print("HIGH SEVERITY INSIGHTS:")
    print("-" * 25)
    for insight in high_severity:
        print(f"  {insight['category']}: {insight['description']}")
    
    print("\nMEDIUM SEVERITY INSIGHTS:")
    print("-" * 25)
    for insight in medium_severity:
        print(f"  {insight['category']}: {insight['description']}")
    
    # Top problematic expressions summary
    print("\nTOP PROBLEMATIC EXPRESSIONS:")
    print("-" * 30)
    for expr in problematic_expressions[:5]:
        print(f"  '{expr['term']}': {expr['fp_count']} FP occurrences, likelihood={expr['likelihood']:.3f}")
    
    return insights

# Main execution function for Part 2
def run_complaint_lexicon_analysis(df):
    """
    Main function to run complaint lexicon mapping analysis
    """
    
    print("Starting Complaint Lexicon Mapping Analysis...")
    
    # Extract lexicons
    distinctive_terms_df, feature_dicts = extract_complaint_lexicons(df)
    
    if distinctive_terms_df is None:
        print("Failed to extract lexicons!")
        return None
    
    # Create dynamic lexicons
    complaint_lexicons = create_dynamic_complaint_lexicons(distinctive_terms_df)
    
    # Analyze patterns by period
    lexicon_analysis = analyze_lexicon_patterns_by_period(df, complaint_lexicons)
    
    # Identify problematic expressions
    problematic_expressions = identify_problematic_expressions(df, complaint_lexicons)
    
    # Create performance matrix
    performance_df, avg_matches_pivot, presence_rate_pivot = create_expression_performance_matrix(df, complaint_lexicons)
    
    # Generate insights
    insights = generate_lexicon_insights(lexicon_analysis, problematic_expressions)
    
    print("\n" + "=" * 80)
    print("PART 2 COMPLETED: Complaint Lexicon Mapping")
    print("Dynamic lexicons created and analyzed for TP vs FP patterns")
    print("=" * 80)
    
    return {
        'distinctive_terms_df': distinctive_terms_df,
        'complaint_lexicons': complaint_lexicons,
        'lexicon_analysis': lexicon_analysis,
        'problematic_expressions': problematic_expressions,
        'performance_df': performance_df,
        'insights': insights
    }

# Example usage
if __name__ == "__main__":
    # This would be called from the main analysis script
    # df = load_and_prepare_data()
    # results = run_complaint_lexicon_analysis(df)
    print("Complaint Lexicon Mapping module ready!")

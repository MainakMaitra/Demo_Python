# ============================================================================
# PART 3: ADVANCED VISUALIZATION AND ANALYTICS
# Comprehensive visualizations for dynamic negation and lexicon analysis
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ADVANCED VISUALIZATION AND ANALYTICS")
print("Creating comprehensive visualizations for insights")
print("=" * 80)

def create_dynamic_pattern_visualizations(negation_df, pattern_analysis_df, output_dir='dynamic_analysis'):
    """
    Create visualizations for dynamically discovered negation patterns
    """
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("CREATING DYNAMIC PATTERN VISUALIZATIONS")
    print("=" * 60)
    
    # 1. Pattern Distribution by TP/FP and Period
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Pattern Distribution by Marker', 'Pattern Evolution Over Time',
                       'Speaker Distribution by Pattern', 'Pattern Performance Matrix'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "heatmap"}]]
    )
    
    # Plot 1: Pattern distribution
    pattern_counts = pattern_analysis_df.groupby(['Total_Count']).agg({
        'TP_Count': 'sum',
        'FP_Count': 'sum'
    }).reset_index()
    
    fig.add_trace(
        go.Bar(x=pattern_analysis_df['Cluster_ID'], 
               y=pattern_analysis_df['TP_Count'],
               name='True Positives', marker_color='green'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=pattern_analysis_df['Cluster_ID'], 
               y=pattern_analysis_df['FP_Count'],
               name='False Positives', marker_color='red'),
        row=1, col=1
    )
    
    # Plot 2: Pattern evolution
    fig.add_trace(
        go.Scatter(x=pattern_analysis_df['Pre_Count'],
                  y=pattern_analysis_df['Post_Count'],
                  mode='markers+text',
                  text=pattern_analysis_df['Cluster_ID'],
                  textposition="middle right",
                  marker=dict(size=pattern_analysis_df['Total_Count']/10,
                            color=pattern_analysis_df['TP_Rate'],
                            colorscale='RdYlGn',
                            showscale=True),
                  name='Pattern Evolution'),
        row=1, col=2
    )
    
    # Add diagonal line for reference
    max_val = max(pattern_analysis_df['Pre_Count'].max(), pattern_analysis_df['Post_Count'].max())
    fig.add_trace(
        go.Scatter(x=[0, max_val], y=[0, max_val],
                  mode='lines', line=dict(dash='dash', color='gray'),
                  name='No Change Line'),
        row=1, col=2
    )
    
    # Plot 3: Speaker distribution
    customer_data = pattern_analysis_df['Customer_Count']
    agent_data = pattern_analysis_df['Agent_Count']
    
    fig.add_trace(
        go.Bar(x=pattern_analysis_df['Cluster_ID'], 
               y=customer_data,
               name='Customer', marker_color='blue'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=pattern_analysis_df['Cluster_ID'], 
               y=agent_data,
               name='Agent', marker_color='orange'),
        row=2, col=1
    )
    
    # Plot 4: Performance heatmap
    performance_matrix = pattern_analysis_df[['Cluster_ID', 'TP_Rate', 'FP_Rate']].set_index('Cluster_ID')
    
    fig.add_trace(
        go.Heatmap(z=performance_matrix.values.T,
                  x=performance_matrix.index,
                  y=['TP Rate', 'FP Rate'],
                  colorscale='RdYlGn',
                  showscale=True),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, 
                     title_text="Dynamic Negation Pattern Analysis")
    
    fig.write_html(f'{output_dir}/dynamic_pattern_analysis.html')
    
    # 2. Detailed Pattern Performance Chart
    fig2, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # TP vs FP rates by pattern
    axes[0,0].scatter(pattern_analysis_df['TP_Rate'], pattern_analysis_df['FP_Rate'],
                     s=pattern_analysis_df['Total_Count'], alpha=0.7)
    axes[0,0].set_xlabel('TP Rate')
    axes[0,0].set_ylabel('FP Rate')
    axes[0,0].set_title('Pattern Performance: TP vs FP Rates')
    axes[0,0].plot([0,1], [1,0], 'r--', alpha=0.5, label='Perfect Discrimination')
    axes[0,0].legend()
    
    # Volume changes Pre vs Post
    volume_change = pattern_analysis_df['Post_Count'] - pattern_analysis_df['Pre_Count']
    axes[0,1].bar(pattern_analysis_df['Cluster_ID'], volume_change, 
                 color=['red' if x < 0 else 'green' for x in volume_change])
    axes[0,1].set_xlabel('Pattern Cluster')
    axes[0,1].set_ylabel('Volume Change (Post - Pre)')
    axes[0,1].set_title('Pattern Volume Changes')
    axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Customer vs Agent distribution
    speaker_ratio = pattern_analysis_df['Customer_Count'] / (pattern_analysis_df['Customer_Count'] + pattern_analysis_df['Agent_Count'] + 1)
    axes[1,0].bar(pattern_analysis_df['Cluster_ID'], speaker_ratio)
    axes[1,0].set_xlabel('Pattern Cluster')
    axes[1,0].set_ylabel('Customer Ratio')
    axes[1,0].set_title('Customer vs Agent Pattern Usage')
    axes[1,0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Equal Usage')
    axes[1,0].legend()
    
    # Pattern quality score
    quality_score = pattern_analysis_df['TP_Rate'] - pattern_analysis_df['FP_Rate']
    axes[1,1].bar(pattern_analysis_df['Cluster_ID'], quality_score,
                 color=['red' if x < 0 else 'green' for x in quality_score])
    axes[1,1].set_xlabel('Pattern Cluster')
    axes[1,1].set_ylabel('Quality Score (TP Rate - FP Rate)')
    axes[1,1].set_title('Pattern Quality Assessment')
    axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pattern_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dynamic pattern visualizations saved to {output_dir}/")

def create_lexicon_visualizations(lexicon_results, output_dir='dynamic_analysis'):
    """
    Create visualizations for complaint lexicon analysis
    """
    
    print("\n" + "=" * 60)
    print("CREATING LEXICON VISUALIZATIONS")
    print("=" * 60)
    
    complaint_lexicons = lexicon_results['complaint_lexicons']
    performance_df = lexicon_results['performance_df']
    problematic_expressions = lexicon_results['problematic_expressions']
    
    # 1. Lexicon Category Distribution
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Lexicon Category Sizes', 'Expression Performance by Period',
                       'Problematic Expressions Analysis', 'Risk Factor Evolution'),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Plot 1: Category sizes
    category_sizes = [len(terms) for terms in complaint_lexicons.values()]
    category_names = list(complaint_lexicons.keys())
    
    fig.add_trace(
        go.Pie(labels=category_names, values=category_sizes,
               name="Lexicon Categories"),
        row=1, col=1
    )
    
    # Plot 2: Performance by period
    if len(performance_df) > 0:
        pre_data = performance_df[performance_df['Period'] == 'Pre']
        post_data = performance_df[performance_df['Period'] == 'Post']
        
        categories = performance_df['Category'].unique()
        
        pre_tp_avg = []
        post_tp_avg = []
        pre_fp_avg = []
        post_fp_avg = []
        
        for cat in categories:
            pre_tp = pre_data[(pre_data['Category'] == cat) & (pre_data['Marker'] == 'TP')]['Avg_Matches'].values
            post_tp = post_data[(post_data['Category'] == cat) & (post_data['Marker'] == 'TP')]['Avg_Matches'].values
            pre_fp = pre_data[(pre_data['Category'] == cat) & (pre_data['Marker'] == 'FP')]['Avg_Matches'].values
            post_fp = post_data[(post_data['Category'] == cat) & (post_data['Marker'] == 'FP')]['Avg_Matches'].values
            
            pre_tp_avg.append(pre_tp[0] if len(pre_tp) > 0 else 0)
            post_tp_avg.append(post_tp[0] if len(post_tp) > 0 else 0)
            pre_fp_avg.append(pre_fp[0] if len(pre_fp) > 0 else 0)
            post_fp_avg.append(post_fp[0] if len(post_fp) > 0 else 0)
        
        fig.add_trace(
            go.Bar(x=categories, y=pre_tp_avg, name='Pre TP', marker_color='darkgreen'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=categories, y=post_tp_avg, name='Post TP', marker_color='lightgreen'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=categories, y=pre_fp_avg, name='Pre FP', marker_color='darkred'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=categories, y=post_fp_avg, name='Post FP', marker_color='lightcoral'),
            row=1, col=2
        )
    
    # Plot 3: Problematic expressions
    if len(problematic_expressions) > 0:
        top_problematic = problematic_expressions[:10]
        
        fig.add_trace(
            go.Scatter(x=[expr['fp_count'] for expr in top_problematic],
                      y=[expr['tp_count'] for expr in top_problematic],
                      mode='markers+text',
                      text=[expr['term'][:10] + '...' if len(expr['term']) > 10 else expr['term'] 
                           for expr in top_problematic],
                      textposition="middle right",
                      marker=dict(size=[expr['likelihood']*50 for expr in top_problematic],
                                color=[expr['likelihood'] for expr in top_problematic],
                                colorscale='RdYlGn_r',
                                showscale=True),
                      name='Problematic Terms'),
            row=2, col=1
        )
    
    fig.update_layout(height=800, showlegend=True, 
                     title_text="Complaint Lexicon Analysis")
    
    fig.write_html(f'{output_dir}/lexicon_analysis.html')
    
    # 2. Word Cloud Visualizations
    fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Create word clouds for each lexicon category
    for idx, (category, terms) in enumerate(complaint_lexicons.items()):
        if idx >= 6 or not terms:  # Limit to 6 categories
            continue
            
        row = idx // 3
        col = idx % 3
        
        # Create text for word cloud
        term_text = ' '.join([term_info['term'] for term_info in terms[:50]])  # Top 50 terms
        
        if term_text.strip():
            wordcloud = WordCloud(width=400, height=300, 
                                background_color='white',
                                max_words=50).generate(term_text)
            
            axes[row, col].imshow(wordcloud, interpolation='bilinear')
            axes[row, col].set_title(category.replace('_', ' '), fontsize=12)
            axes[row, col].axis('off')
        else:
            axes[row, col].text(0.5, 0.5, 'No terms', ha='center', va='center', 
                              transform=axes[row, col].transAxes)
            axes[row, col].set_title(category.replace('_', ' '), fontsize=12)
            axes[row, col].axis('off')
    
    # Hide unused subplots
    for idx in range(len(complaint_lexicons), 6):
        row = idx // 3
        col = idx % 3
        axes[row, col].axis('off')
    
    plt.suptitle('Complaint Lexicon Word Clouds', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/lexicon_wordclouds.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Lexicon visualizations saved to {output_dir}/")

def create_integrated_analysis_dashboard(negation_df, pattern_analysis_df, lexicon_results, output_dir='dynamic_analysis'):
    """
    Create an integrated dashboard combining negation and lexicon analysis
    """
    
    print("\n" + "=" * 60)
    print("CREATING INTEGRATED ANALYSIS DASHBOARD")
    print("=" * 60)
    
    # Create comprehensive dashboard
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=('Negation Pattern Performance', 'Lexicon Category Impact', 
                       'Temporal Evolution', 'Speaker Distribution',
                       'Risk Factor Matrix', 'Problem Expression Frequency',
                       'Pattern-Lexicon Correlation', 'Precision Impact', 'Action Priority Matrix'),
        specs=[[{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "heatmap"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # 1. Negation Pattern Performance
    fig.add_trace(
        go.Scatter(x=pattern_analysis_df['TP_Rate'],
                  y=pattern_analysis_df['FP_Rate'],
                  mode='markers+text',
                  text=pattern_analysis_df['Cluster_ID'],
                  marker=dict(size=pattern_analysis_df['Total_Count']/20,
                            color=pattern_analysis_df['Total_Count'],
                            colorscale='Viridis'),
                  name='Patterns'),
        row=1, col=1
    )
    
    # 2. Lexicon Category Impact
    if 'performance_df' in lexicon_results and len(lexicon_results['performance_df']) > 0:
        perf_df = lexicon_results['performance_df']
        category_impact = perf_df.groupby('Category')['Avg_Matches'].sum().sort_values(ascending=False)
        
        fig.add_trace(
            go.Bar(x=category_impact.index, y=category_impact.values,
                  marker_color='lightblue'),
            row=1, col=2
        )
    
    # 3. Temporal Evolution
    pre_counts = pattern_analysis_df['Pre_Count']
    post_counts = pattern_analysis_df['Post_Count']
    
    fig.add_trace(
        go.Scatter(x=pre_counts, y=post_counts,
                  mode='markers+text',
                  text=pattern_analysis_df['Cluster_ID'],
                  marker=dict(color=pattern_analysis_df['TP_Rate'],
                            colorscale='RdYlGn', size=10),
                  name='Evolution'),
        row=1, col=3
    )
    
    # 4. Speaker Distribution
    customer_ratios = pattern_analysis_df['Customer_Count'] / (pattern_analysis_df['Customer_Count'] + pattern_analysis_df['Agent_Count'] + 1)
    
    fig.add_trace(
        go.Bar(x=pattern_analysis_df['Cluster_ID'], y=customer_ratios,
              marker_color='purple'),
        row=2, col=1
    )
    
    # 5. Risk Factor Matrix
    risk_matrix = np.array([pattern_analysis_df['TP_Rate'], pattern_analysis_df['FP_Rate']])
    
    fig.add_trace(
        go.Heatmap(z=risk_matrix,
                  x=pattern_analysis_df['Cluster_ID'],
                  y=['TP Rate', 'FP Rate'],
                  colorscale='RdYlGn'),
        row=2, col=2
    )
    
    # 6. Problem Expression Frequency
    if 'problematic_expressions' in lexicon_results and len(lexicon_results['problematic_expressions']) > 0:
        prob_expr = lexicon_results['problematic_expressions'][:10]
        
        fig.add_trace(
            go.Bar(x=[expr['term'][:10] for expr in prob_expr],
                  y=[expr['fp_count'] for expr in prob_expr],
                  marker_color='red'),
            row=2, col=3
        )
    
    # 7. Pattern Quality Scores
    quality_scores = pattern_analysis_df['TP_Rate'] - pattern_analysis_df['FP_Rate']
    
    fig.add_trace(
        go.Scatter(x=pattern_analysis_df['Total_Count'], y=quality_scores,
                  mode='markers+text',
                  text=pattern_analysis_df['Cluster_ID'],
                  marker=dict(color=quality_scores, colorscale='RdYlGn', size=10),
                  name='Quality'),
        row=3, col=1
    )
    
    # 8. Precision Impact
    volume_impact = pattern_analysis_df['Total_Count'] * abs(quality_scores)
    
    fig.add_trace(
        go.Scatter(x=quality_scores, y=volume_impact,
                  mode='markers+text',
                  text=pattern_analysis_df['Cluster_ID'],
                  marker=dict(size=15, color='orange'),
                  name='Impact'),
        row=3, col=2
    )
    
    # 9. Action Priority Matrix
    urgency = abs(pattern_analysis_df['Post_Count'] - pattern_analysis_df['Pre_Count'])
    importance = pattern_analysis_df['Total_Count']
    
    fig.add_trace(
        go.Scatter(x=urgency, y=importance,
                  mode='markers+text',
                  text=pattern_analysis_df['Cluster_ID'],
                  marker=dict(color=quality_scores, colorscale='RdYlGn', size=12),
                  name='Priority'),
        row=3, col=3
    )
    
    # Update layout
    fig.update_layout(height=1200, showlegend=False, 
                     title_text="Integrated Dynamic Analysis Dashboard")
    
    # Update axis labels
    fig.update_xaxes(title_text="TP Rate", row=1, col=1)
    fig.update_yaxes(title_text="FP Rate", row=1, col=1)
    
    fig.update_xaxes(title_text="Lexicon Categories", row=1, col=2)
    fig.update_yaxes(title_text="Total Impact", row=1, col=2)
    
    fig.update_xaxes(title_text="Pre Period Count", row=1, col=3)
    fig.update_yaxes(title_text="Post Period Count", row=1, col=3)
    
    fig.update_xaxes(title_text="Pattern Cluster", row=2, col=1)
    fig.update_yaxes(title_text="Customer Ratio", row=2, col=1)
    
    fig.update_xaxes(title_text="Expressions", row=2, col=3)
    fig.update_yaxes(title_text="FP Count", row=2, col=3)
    
    fig.update_xaxes(title_text="Pattern Volume", row=3, col=1)
    fig.update_yaxes(title_text="Quality Score", row=3, col=1)
    
    fig.update_xaxes(title_text="Quality Score", row=3, col=2)
    fig.update_yaxes(title_text="Volume Impact", row=3, col=2)
    
    fig.update_xaxes(title_text="Urgency (Volume Change)", row=3, col=3)
    fig.update_yaxes(title_text="Importance (Total Volume)", row=3, col=3)
    
    fig.write_html(f'{output_dir}/integrated_dashboard.html')
    
    print(f"Integrated dashboard saved to {output_dir}/")

def create_actionable_insights_report(negation_df, pattern_analysis_df, lexicon_results, output_dir='dynamic_analysis'):
    """
    Create a comprehensive actionable insights report
    """
    
    print("\n" + "=" * 60)
    print("CREATING ACTIONABLE INSIGHTS REPORT")
    print("=" * 60)
    
    insights = []
    
    # Analyze negation patterns for actionability
    high_risk_patterns = pattern_analysis_df[
        (pattern_analysis_df['FP_Rate'] > 0.5) & 
        (pattern_analysis_df['Total_Count'] > 20)
    ]
    
    for _, pattern in high_risk_patterns.iterrows():
        insights.append({
            'type': 'High Risk Negation Pattern',
            'severity': 'Critical',
            'pattern_id': pattern['Cluster_ID'],
            'description': f"Pattern {pattern['Cluster_ID']} has {pattern['FP_Rate']:.1%} FP rate with {pattern['Total_Count']} instances",
            'action': f"Review and refine rules for pattern features: {pattern['Top_Features'][:50]}...",
            'impact': pattern['Total_Count'] * pattern['FP_Rate']
        })
    
    # Analyze temporal changes
    significant_changes = pattern_analysis_df[
        abs(pattern_analysis_df['Post_Count'] - pattern_analysis_df['Pre_Count']) > 20
    ]
    
    for _, pattern in significant_changes.iterrows():
        change = pattern['Post_Count'] - pattern['Pre_Count']
        insights.append({
            'type': 'Volume Shift',
            'severity': 'High' if abs(change) > 50 else 'Medium',
            'pattern_id': pattern['Cluster_ID'],
            'description': f"Pattern {pattern['Cluster_ID']} volume changed by {change:+d} from Pre to Post",
            'action': 'Investigate root cause of volume shift and adjust classification thresholds',
            'impact': abs(change) * max(pattern['TP_Rate'], pattern['FP_Rate'])
        })
    
    # Analyze lexicon problems
    if 'problematic_expressions' in lexicon_results:
        top_problems = lexicon_results['problematic_expressions'][:5]
        
        for expr in top_problems:
            insights.append({
                'type': 'Problematic Expression',
                'severity': 'High' if expr['fp_count'] > 50 else 'Medium',
                'pattern_id': f"Expression: {expr['term']}",
                'description': f"'{expr['term']}' appears in {expr['fp_count']} FPs vs {expr['tp_count']} TPs",
                'action': f"Add context rules for '{expr['term']}' to distinguish complaint vs information usage",
                'impact': expr['fp_count'] * (1 - expr['likelihood'])
            })
    
    # Sort by impact
    insights.sort(key=lambda x: x['impact'], reverse=True)
    
    # Create report visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Impact vs Severity
    impact_values = [insight['impact'] for insight in insights[:10]]
    severity_colors = {'Critical': 'red', 'High': 'orange', 'Medium': 'yellow'}
    colors = [severity_colors.get(insight['severity'], 'gray') for insight in insights[:10]]
    
    axes[0,0].barh(range(len(impact_values)), impact_values, color=colors)
    axes[0,0].set_yticks(range(len(impact_values)))
    axes[0,0].set_yticklabels([f"{insight['type'][:15]}..." for insight in insights[:10]], fontsize=8)
    axes[0,0].set_xlabel('Impact Score')
    axes[0,0].set_title('Top 10 Issues by Impact')
    
    # 2. Issue Type Distribution
    issue_types = [insight['type'] for insight in insights]
    type_counts = pd.Series(issue_types).value_counts()
    
    axes[0,1].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
    axes[0,1].set_title('Issue Type Distribution')
    
    # 3. Severity Distribution
    severity_counts = pd.Series([insight['severity'] for insight in insights]).value_counts()
    severity_colors_list = [severity_colors.get(sev, 'gray') for sev in severity_counts.index]
    
    axes[1,0].bar(severity_counts.index, severity_counts.values, color=severity_colors_list)
    axes[1,0].set_xlabel('Severity Level')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('Issues by Severity')
    
    # 4. Impact vs Pattern Volume
    pattern_impacts = []
    pattern_volumes = []
    
    for insight in insights:
        if 'Pattern' in insight['type']:
            pattern_id = insight['pattern_id']
            if isinstance(pattern_id, str) and 'Pattern' in pattern_id:
                try:
                    pid = int(pattern_id.split()[-1])
                    pattern_data = pattern_analysis_df[pattern_analysis_df['Cluster_ID'] == pid]
                    if not pattern_data.empty:
                        pattern_volumes.append(pattern_data.iloc[0]['Total_Count'])
                        pattern_impacts.append(insight['impact'])
                except:
                    continue
    
    if pattern_volumes:
        axes[1,1].scatter(pattern_volumes, pattern_impacts, alpha=0.7)
        axes[1,1].set_xlabel('Pattern Volume')
        axes[1,1].set_ylabel('Impact Score')
        axes[1,1].set_title('Impact vs Pattern Volume')
    
    plt.tight_layout()
    plt.suptitle('Actionable Insights Analysis', fontsize=16, y=1.02)
    plt.savefig(f'{output_dir}/actionable_insights.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create text report
    with open(f'{output_dir}/actionable_insights_report.txt', 'w') as f:
        f.write("ACTIONABLE INSIGHTS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated from analysis of {len(negation_df)} negation instances\n")
        f.write(f"across {len(pattern_analysis_df)} discovered patterns\n\n")
        
        f.write("TOP PRIORITY ACTIONS:\n")
        f.write("-" * 25 + "\n")
        
        for i, insight in enumerate(insights[:10], 1):
            f.write(f"\n{i}. {insight['type']} - {insight['severity']}\n")
            f.write(f"   Issue: {insight['description']}\n")
            f.write(f"   Action: {insight['action']}\n")
            f.write(f"   Impact Score: {insight['impact']:.1f}\n")
        
        f.write(f"\n\nSUMMARY STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Issues Identified: {len(insights)}\n")
        f.write(f"Critical Issues: {len([i for i in insights if i['severity'] == 'Critical'])}\n")
        f.write(f"High Priority Issues: {len([i for i in insights if i['severity'] == 'High'])}\n")
        f.write(f"Medium Priority Issues: {len([i for i in insights if i['severity'] == 'Medium'])}\n")
        
        total_impact = sum(insight['impact'] for insight in insights)
        f.write(f"Total Impact Score: {total_impact:.1f}\n")
    
    print(f"Actionable insights report saved to {output_dir}/")
    
    return insights

def create_comparison_analysis(negation_df, pattern_analysis_df, lexicon_results, output_dir='dynamic_analysis'):
    """
    Create detailed Pre vs Post comparison analysis
    """
    
    print("\n" + "=" * 60)
    print("CREATING PRE VS POST COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Pre vs Post negation analysis
    pre_data = negation_df[negation_df['Period'] == 'Pre']
    post_data = negation_df[negation_df['Period'] == 'Post']
    
    comparison_metrics = {}
    
    # Overall metrics
    comparison_metrics['overall'] = {
        'pre_total': len(pre_data),
        'post_total': len(post_data),
        'pre_tp_rate': len(pre_data[pre_data['Primary_Marker'] == 'TP']) / len(pre_data) if len(pre_data) > 0 else 0,
        'post_tp_rate': len(post_data[post_data['Primary_Marker'] == 'TP']) / len(post_data) if len(post_data) > 0 else 0,
        'pre_fp_rate': len(pre_data[pre_data['Primary_Marker'] == 'FP']) / len(pre_data) if len(pre_data) > 0 else 0,
        'post_fp_rate': len(post_data[post_data['Primary_Marker'] == 'FP']) / len(post_data) if len(post_data) > 0 else 0
    }
    
    # Pattern-level metrics
    comparison_metrics['patterns'] = {}
    
    for _, pattern in pattern_analysis_df.iterrows():
        cluster_id = pattern['Cluster_ID']
        
        pre_pattern = pre_data[pre_data['Pattern_Cluster'] == cluster_id]
        post_pattern = post_data[post_data['Pattern_Cluster'] == cluster_id]
        
        comparison_metrics['patterns'][cluster_id] = {
            'pre_count': len(pre_pattern),
            'post_count': len(post_pattern),
            'volume_change': len(post_pattern) - len(pre_pattern),
            'volume_change_pct': ((len(post_pattern) - len(pre_pattern)) / max(len(pre_pattern), 1)) * 100,
            'pre_tp_rate': len(pre_pattern[pre_pattern['Primary_Marker'] == 'TP']) / max(len(pre_pattern), 1),
            'post_tp_rate': len(post_pattern[post_pattern['Primary_Marker'] == 'TP']) / max(len(post_pattern), 1),
            'tp_rate_change': (len(post_pattern[post_pattern['Primary_Marker'] == 'TP']) / max(len(post_pattern), 1)) - 
                             (len(pre_pattern[pre_pattern['Primary_Marker'] == 'TP']) / max(len(pre_pattern), 1))
        }
    
    # Create comparison visualization
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Volume Changes by Pattern', 'TP Rate Changes', 'FP Rate Changes',
                       'Pattern Quality Evolution', 'Speaker Distribution Changes', 'Risk Assessment'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}]]
    )
    
    # 1. Volume changes
    pattern_ids = list(comparison_metrics['patterns'].keys())
    volume_changes = [comparison_metrics['patterns'][pid]['volume_change'] for pid in pattern_ids]
    
    colors = ['red' if x < 0 else 'green' for x in volume_changes]
    
    fig.add_trace(
        go.Bar(x=pattern_ids, y=volume_changes, marker_color=colors),
        row=1, col=1
    )
    
    # 2. TP rate changes
    tp_changes = [comparison_metrics['patterns'][pid]['tp_rate_change'] for pid in pattern_ids]
    colors_tp = ['red' if x < 0 else 'green' for x in tp_changes]
    
    fig.add_trace(
        go.Bar(x=pattern_ids, y=tp_changes, marker_color=colors_tp),
        row=1, col=2
    )
    
    # 3. FP rate changes (inverse of TP)
    fp_changes = [-x for x in tp_changes]  # Inverse relationship
    colors_fp = ['green' if x < 0 else 'red' for x in fp_changes]
    
    fig.add_trace(
        go.Bar(x=pattern_ids, y=fp_changes, marker_color=colors_fp),
        row=1, col=3
    )
    
    # 4. Quality evolution
    pre_quality = [comparison_metrics['patterns'][pid]['pre_tp_rate'] for pid in pattern_ids]
    post_quality = [comparison_metrics['patterns'][pid]['post_tp_rate'] for pid in pattern_ids]
    
    fig.add_trace(
        go.Scatter(x=pre_quality, y=post_quality,
                  mode='markers+text',
                  text=pattern_ids,
                  marker=dict(size=10, color='blue'),
                  name='Quality Evolution'),
        row=2, col=1
    )
    
    # Add diagonal line
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1],
                  mode='lines', line=dict(dash='dash', color='gray'),
                  name='No Change'),
        row=2, col=1
    )
    
    # 5. Speaker distribution changes
    pre_customer_ratios = []
    post_customer_ratios = []
    
    for pid in pattern_ids:
        pre_pattern = pre_data[pre_data['Pattern_Cluster'] == pid]
        post_pattern = post_data[post_data['Pattern_Cluster'] == pid]
        
        pre_customer = len(pre_pattern[pre_pattern['Speaker'] == 'customer'])
        pre_total = len(pre_pattern)
        post_customer = len(post_pattern[post_pattern['Speaker'] == 'customer'])
        post_total = len(post_pattern)
        
        pre_customer_ratios.append(pre_customer / max(pre_total, 1))
        post_customer_ratios.append(post_customer / max(post_total, 1))
    
    ratio_changes = [post - pre for post, pre in zip(post_customer_ratios, pre_customer_ratios)]
    colors_ratio = ['red' if x < 0 else 'green' for x in ratio_changes]
    
    fig.add_trace(
        go.Bar(x=pattern_ids, y=ratio_changes, marker_color=colors_ratio),
        row=2, col=2
    )
    
    # 6. Risk assessment
    risk_scores = []
    urgency_scores = []
    
    for pid in pattern_ids:
        metrics = comparison_metrics['patterns'][pid]
        
        # Risk = high FP rate increase + high volume
        risk = abs(metrics['tp_rate_change']) * metrics['post_count']
        urgency = abs(metrics['volume_change'])
        
        risk_scores.append(risk)
        urgency_scores.append(urgency)
    
    fig.add_trace(
        go.Scatter(x=urgency_scores, y=risk_scores,
                  mode='markers+text',
                  text=pattern_ids,
                  marker=dict(size=15, color='red'),
                  name='Risk Matrix'),
        row=2, col=3
    )
    
    fig.update_layout(height=800, showlegend=True, 
                     title_text="Pre vs Post Comparison Analysis")
    
    fig.write_html(f'{output_dir}/pre_post_comparison.html')
    
    # Save detailed comparison data
    comparison_df = pd.DataFrame(comparison_metrics['patterns']).T
    comparison_df.to_csv(f'{output_dir}/comparison_metrics.csv')
    
    print(f"Pre vs Post comparison analysis saved to {output_dir}/")
    
    return comparison_metrics

# Main execution function for Part 3
def run_visualization_analysis(negation_df, pattern_analysis_df, lexicon_results):
    """
    Main function to run all visualization analyses
    """
    
    print("Starting Advanced Visualization Analysis...")
    
    # Create output directory
    import os
    output_dir = 'dynamic_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create all visualizations
    create_dynamic_pattern_visualizations(negation_df, pattern_analysis_df, output_dir)
    create_lexicon_visualizations(lexicon_results, output_dir)
    create_integrated_analysis_dashboard(negation_df, pattern_analysis_df, lexicon_results, output_dir)
    
    # Create insights and comparisons
    insights = create_actionable_insights_report(negation_df, pattern_analysis_df, lexicon_results, output_dir)
    comparison_metrics = create_comparison_analysis(negation_df, pattern_analysis_df, lexicon_results, output_dir)
    
    print("\n" + "=" * 80)
    print("PART 3 COMPLETED: Advanced Visualization and Analytics")
    print("All visualizations and insights saved to dynamic_analysis/")
    print("=" * 80)
    
    return {
        'insights': insights,
        'comparison_metrics': comparison_metrics,
        'output_directory': output_dir
    }

# Example usage
if __name__ == "__main__":
    print("Advanced Visualization module ready!")

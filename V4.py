# =============================================================================
# ADDITIONAL VISUALIZATION FUNCTIONS FOR DYNAMIC NEGATION ANALYSIS
# =============================================================================

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import os

def create_comprehensive_visualizations(negation_df, pattern_analysis_df, output_dir='dynamic_negation_visualizations'):
    """
    Create comprehensive visualizations for dynamic negation analysis
    Enhanced version without hardcoded patterns
    """
    
    print("\n" + "="*60)
    print("CREATING COMPREHENSIVE VISUALIZATIONS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if len(negation_df) == 0:
        print("No negation data to visualize!")
        return
    
    # 1. PATTERN DISTRIBUTION ANALYSIS
    create_pattern_distribution_viz(negation_df, pattern_analysis_df, output_dir)
    
    # 2. TEMPORAL ANALYSIS
    create_temporal_analysis_viz(negation_df, output_dir)
    
    # 3. SPEAKER ANALYSIS
    create_speaker_analysis_viz(negation_df, output_dir)
    
    # 4. CONTEXT WORD CLOUDS
    create_context_wordclouds(negation_df, pattern_analysis_df, output_dir)
    
    # 5. PERFORMANCE HEATMAPS
    create_performance_heatmaps(negation_df, output_dir)
    
    # 6. INTERACTIVE DASHBOARD
    create_interactive_dashboard(negation_df, pattern_analysis_df, output_dir)
    
    print(f"All visualizations saved to {output_dir}/")

def create_pattern_distribution_viz(negation_df, pattern_analysis_df, output_dir):
    """Create pattern distribution visualizations"""
    
    print("Creating pattern distribution visualizations...")
    
    # Enhanced version of the basic pattern visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Pattern Cluster Distribution by TP/FP
    if 'Pattern_Cluster' in negation_df.columns:
        tp_patterns = negation_df[negation_df['Primary_Marker'] == 'TP']['Pattern_Cluster'].value_counts()
        fp_patterns = negation_df[negation_df['Primary_Marker'] == 'FP']['Pattern_Cluster'].value_counts()
        
        # Align indices
        all_patterns = sorted(set(list(tp_patterns.index) + list(fp_patterns.index)))
        tp_counts = [tp_patterns.get(p, 0) for p in all_patterns]
        fp_counts = [fp_patterns.get(p, 0) for p in all_patterns]
        
        x = np.arange(len(all_patterns))
        width = 0.35
        
        axes[0,0].bar(x - width/2, tp_counts, width, label='TP', alpha=0.8, color='green')
        axes[0,0].bar(x + width/2, fp_counts, width, label='FP', alpha=0.8, color='red')
        axes[0,0].set_xlabel('Pattern Cluster')
        axes[0,0].set_ylabel('Count')
        axes[0,0].set_title('Pattern Distribution: TP vs FP')
        axes[0,0].legend()
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(all_patterns)
    
    # 2. Negation Word Frequency
    negation_word_counts = negation_df['Negation_Word'].value_counts().head(10)
    axes[0,1].barh(range(len(negation_word_counts)), negation_word_counts.values, color='skyblue')
    axes[0,1].set_yticks(range(len(negation_word_counts)))
    axes[0,1].set_yticklabels(negation_word_counts.index)
    axes[0,1].set_xlabel('Frequency')
    axes[0,1].set_title('Top 10 Negation Words')
    
    # 3. Speaker Distribution
    speaker_dist = negation_df['Speaker'].value_counts()
    axes[0,2].pie(speaker_dist.values, labels=speaker_dist.index, autopct='%1.1f%%', 
                  colors=['lightblue', 'lightcoral'])
    axes[0,2].set_title('Negations by Speaker')
    
    # 4. Period Comparison
    period_marker = pd.crosstab(negation_df['Period'], negation_df['Primary_Marker'])
    period_marker.plot(kind='bar', ax=axes[1,0], color=['green', 'red'], alpha=0.7)
    axes[1,0].set_title('Period vs Primary Marker')
    axes[1,0].set_xlabel('Period')
    axes[1,0].set_ylabel('Count')
    axes[1,0].legend(title='Primary Marker')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 5. Text Length Distribution
    if 'Text_Length' in negation_df.columns:
        tp_lengths = negation_df[negation_df['Primary_Marker'] == 'TP']['Text_Length']
        fp_lengths = negation_df[negation_df['Primary_Marker'] == 'FP']['Text_Length']
        
        axes[1,1].hist(tp_lengths, bins=30, alpha=0.7, label='TP', color='green', density=True)
        axes[1,1].hist(fp_lengths, bins=30, alpha=0.7, label='FP', color='red', density=True)
        axes[1,1].set_xlabel('Text Length')
        axes[1,1].set_ylabel('Density')
        axes[1,1].set_title('Text Length Distribution')
        axes[1,1].legend()
    
    # 6. Position in Text Analysis
    if 'Position_In_Text' in negation_df.columns:
        tp_positions = negation_df[negation_df['Primary_Marker'] == 'TP']['Position_In_Text']
        fp_positions = negation_df[negation_df['Primary_Marker'] == 'FP']['Position_In_Text']
        
        axes[1,2].hist(tp_positions, bins=20, alpha=0.7, label='TP', color='green', density=True)
        axes[1,2].hist(fp_positions, bins=20, alpha=0.7, label='FP', color='red', density=True)
        axes[1,2].set_xlabel('Position in Text (0=start, 1=end)')
        axes[1,2].set_ylabel('Density')
        axes[1,2].set_title('Negation Position Distribution')
        axes[1,2].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pattern_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_temporal_analysis_viz(negation_df, output_dir):
    """Create temporal analysis visualizations"""
    
    print("Creating temporal analysis...")
    
    # Monthly evolution analysis
    if 'Year_Month' in negation_df.columns:
        monthly_stats = negation_df.groupby(['Year_Month', 'Primary_Marker']).agg({
            'UUID': 'count',
            'Negation_Word': lambda x: len(x.unique())
        }).reset_index()
        monthly_stats.columns = ['Year_Month', 'Primary_Marker', 'Volume', 'Unique_Negations']
        
        # Create interactive plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Volume Over Time', 'Unique Negations Over Time',
                           'TP/FP Ratio Over Time', 'Cumulative Volume'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Volume trend
        tp_monthly = monthly_stats[monthly_stats['Primary_Marker'] == 'TP']
        fp_monthly = monthly_stats[monthly_stats['Primary_Marker'] == 'FP']
        
        fig.add_trace(
            go.Scatter(x=tp_monthly['Year_Month'], y=tp_monthly['Volume'],
                      mode='lines+markers', name='TP Volume', line=dict(color='green')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=fp_monthly['Year_Month'], y=fp_monthly['Volume'],
                      mode='lines+markers', name='FP Volume', line=dict(color='red')),
            row=1, col=1
        )
        
        # Unique negations
        fig.add_trace(
            go.Scatter(x=tp_monthly['Year_Month'], y=tp_monthly['Unique_Negations'],
                      mode='lines+markers', name='TP Unique', line=dict(color='darkgreen')),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=fp_monthly['Year_Month'], y=fp_monthly['Unique_Negations'],
                      mode='lines+markers', name='FP Unique', line=dict(color='darkred')),
            row=1, col=2
        )
        
        # TP/FP Ratio
        ratio_data = []
        for month in tp_monthly['Year_Month'].unique():
            tp_vol = tp_monthly[tp_monthly['Year_Month'] == month]['Volume'].sum()
            fp_vol = fp_monthly[fp_monthly['Year_Month'] == month]['Volume'].sum()
            ratio = tp_vol / (fp_vol + 1)  # Add 1 to avoid division by zero
            ratio_data.append({'Month': month, 'Ratio': ratio})
        
        ratio_df = pd.DataFrame(ratio_data)
        fig.add_trace(
            go.Scatter(x=ratio_df['Month'], y=ratio_df['Ratio'],
                      mode='lines+markers', name='TP/FP Ratio', line=dict(color='blue')),
            row=2, col=1
        )
        
        # Cumulative volume
        tp_cumsum = tp_monthly['Volume'].cumsum()
        fp_cumsum = fp_monthly['Volume'].cumsum()
        
        fig.add_trace(
            go.Scatter(x=tp_monthly['Year_Month'], y=tp_cumsum,
                      mode='lines', name='TP Cumulative', fill='tonexty', 
                      line=dict(color='green'), opacity=0.7),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=fp_monthly['Year_Month'], y=fp_cumsum,
                      mode='lines', name='FP Cumulative', 
                      line=dict(color='red'), opacity=0.7),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, title_text="Temporal Analysis of Negation Patterns")
        fig.write_html(f'{output_dir}/temporal_analysis.html')

def create_speaker_analysis_viz(negation_df, output_dir):
    """Create speaker-specific analysis"""
    
    print("Creating speaker analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Speaker vs Primary Marker
    speaker_marker = pd.crosstab(negation_df['Speaker'], negation_df['Primary_Marker'])
    speaker_marker_pct = speaker_marker.div(speaker_marker.sum(axis=1), axis=0) * 100
    
    speaker_marker_pct.plot(kind='bar', ax=axes[0,0], color=['green', 'red'], alpha=0.7)
    axes[0,0].set_title('Speaker Performance (% TP vs FP)')
    axes[0,0].set_ylabel('Percentage')
    axes[0,0].legend(title='Primary Marker')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Negation Words by Speaker
    customer_words = negation_df[negation_df['Speaker'] == 'customer']['Negation_Word'].value_counts().head(8)
    agent_words = negation_df[negation_df['Speaker'] == 'agent']['Negation_Word'].value_counts().head(8)
    
    axes[0,1].barh(range(len(customer_words)), customer_words.values, color='lightblue', alpha=0.7)
    axes[0,1].set_yticks(range(len(customer_words)))
    axes[0,1].set_yticklabels(customer_words.index)
    axes[0,1].set_title('Top Customer Negation Words')
    
    # 3. Agent negation words
    axes[1,0].barh(range(len(agent_words)), agent_words.values, color='lightcoral', alpha=0.7)
    axes[1,0].set_yticks(range(len(agent_words)))
    axes[1,0].set_yticklabels(agent_words.index)
    axes[1,0].set_title('Top Agent Negation Words')
    
    # 4. Speaker pattern over time
    if 'Year_Month' in negation_df.columns:
        speaker_time = negation_df.groupby(['Year_Month', 'Speaker']).size().unstack(fill_value=0)
        speaker_time.plot(kind='bar', ax=axes[1,1], alpha=0.7)
        axes[1,1].set_title('Speaker Activity Over Time')
        axes[1,1].set_ylabel('Negation Count')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].legend(title='Speaker')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/speaker_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_context_wordclouds(negation_df, pattern_analysis_df, output_dir):
    """Create context word clouds for different patterns"""
    
    print("Creating context word clouds...")
    
    if 'Pattern_Cluster' not in negation_df.columns:
        print("No pattern clusters found for word clouds")
        return
    
    # Get top patterns by volume
    top_patterns = pattern_analysis_df.nlargest(4, 'Total_Count')['Cluster_ID'].tolist()
    
    n_patterns = len(top_patterns)
    if n_patterns == 0:
        return
    
    fig, axes = plt.subplots(2, max(2, n_patterns//2 + n_patterns%2), figsize=(16, 8))
    if n_patterns == 1:
        axes = [axes]
    elif n_patterns <= 2:
        axes = axes.reshape(-1)
    
    for i, pattern_id in enumerate(top_patterns[:4]):  # Max 4 word clouds
        row = i // 2
        col = i % 2
        
        if n_patterns <= 2:
            ax = axes[i]
        else:
            ax = axes[row, col]
        
        # Get contexts for this pattern
        pattern_contexts = negation_df[negation_df['Pattern_Cluster'] == pattern_id]['Context'].str.cat(sep=' ')
        
        if pattern_contexts and len(pattern_contexts.strip()) > 0:
            # Clean text for word cloud
            import re
            clean_text = re.sub(r'[^\w\s]', ' ', pattern_contexts.lower())
            clean_text = ' '.join(clean_text.split())  # Remove extra whitespace
            
            if len(clean_text) > 10:  # Only create if sufficient text
                wordcloud = WordCloud(
                    width=400, height=300, 
                    background_color='white',
                    max_words=50,
                    relative_scaling=0.5
                ).generate(clean_text)
                
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f'Pattern {pattern_id} Context\n({len(negation_df[negation_df["Pattern_Cluster"] == pattern_id])} instances)')
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, f'Pattern {pattern_id}\nInsufficient text', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'Pattern {pattern_id}\nNo context available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # Hide empty subplots
    if n_patterns < 4:
        for i in range(n_patterns, 4):
            row = i // 2
            col = i % 2
            if n_patterns > 2:
                axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pattern_context_wordclouds.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_heatmaps(negation_df, output_dir):
    """Create performance heatmaps"""
    
    print("Creating performance heatmaps...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Period vs Negation Word Performance
    if 'Period' in negation_df.columns:
        period_word_perf = pd.crosstab(
            negation_df['Period'], 
            negation_df['Negation_Word'], 
            negation_df['Primary_Marker'], 
            aggfunc='count'
        ).fillna(0)
        
        if 'TP' in period_word_perf.columns:
            tp_counts = period_word_perf['TP']
            total_counts = period_word_perf.sum(axis=1, level=0)
            tp_rate_matrix = tp_counts.div(total_counts).fillna(0)
            
            # Select top negation words for readability
            top_words = negation_df['Negation_Word'].value_counts().head(10).index
            tp_rate_filtered = tp_rate_matrix[tp_rate_matrix.columns.intersection(top_words)]
            
            sns.heatmap(tp_rate_filtered, annot=True, fmt='.2f', cmap='RdYlGn', 
                       ax=axes[0], cbar_kws={'label': 'TP Rate'})
            axes[0].set_title('TP Rate: Period vs Negation Word')
            axes[0].set_ylabel('Period')
            axes[0].set_xlabel('Negation Word')
    
    # 2. Speaker vs Pattern Performance
    if 'Pattern_Cluster' in negation_df.columns:
        speaker_pattern_perf = pd.crosstab(
            negation_df['Speaker'], 
            negation_df['Pattern_Cluster'], 
            negation_df['Primary_Marker'], 
            aggfunc='count'
        ).fillna(0)
        
        if 'TP' in speaker_pattern_perf.columns:
            tp_counts = speaker_pattern_perf['TP']
            total_counts = speaker_pattern_perf.sum(axis=1, level=0)
            tp_rate_matrix = tp_counts.div(total_counts).fillna(0)
            
            sns.heatmap(tp_rate_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
                       ax=axes[1], cbar_kws={'label': 'TP Rate'})
            axes[1].set_title('TP Rate: Speaker vs Pattern')
            axes[1].set_ylabel('Speaker')
            axes[1].set_xlabel('Pattern Cluster')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_interactive_dashboard(negation_df, pattern_analysis_df, output_dir):
    """Create interactive dashboard"""
    
    print("Creating interactive dashboard...")
    
    # Create multi-tab dashboard
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Pattern Volume', 'TP vs FP Distribution', 
                       'Monthly Trends', 'Speaker Analysis',
                       'Performance Metrics', 'Pattern Quality'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # 1. Pattern volume
    if len(pattern_analysis_df) > 0:
        fig.add_trace(
            go.Bar(x=pattern_analysis_df['Cluster_ID'], 
                   y=pattern_analysis_df['Total_Count'],
                   name='Pattern Volume',
                   marker_color='blue'),
            row=1, col=1
        )
    
    # 2. TP vs FP scatter
    tp_data = negation_df[negation_df['Primary_Marker'] == 'TP']
    fp_data = negation_df[negation_df['Primary_Marker'] == 'FP']
    
    if 'Pattern_Cluster' in negation_df.columns:
        tp_pattern_counts = tp_data['Pattern_Cluster'].value_counts()
        fp_pattern_counts = fp_data['Pattern_Cluster'].value_counts()
        
        all_patterns = sorted(set(list(tp_pattern_counts.index) + list(fp_pattern_counts.index)))
        tp_counts = [tp_pattern_counts.get(p, 0) for p in all_patterns]
        fp_counts = [fp_pattern_counts.get(p, 0) for p in all_patterns]
        
        fig.add_trace(
            go.Scatter(x=tp_counts, y=fp_counts,
                      mode='markers+text',
                      text=[f'P{p}' for p in all_patterns],
                      textposition='top center',
                      name='Pattern Performance',
                      marker=dict(size=10, color='red', opacity=0.7)),
            row=1, col=2
        )
    
    # 3. Monthly trends
    if 'Year_Month' in negation_df.columns:
        monthly_tp = negation_df[negation_df['Primary_Marker'] == 'TP'].groupby('Year_Month').size()
        monthly_fp = negation_df[negation_df['Primary_Marker'] == 'FP'].groupby('Year_Month').size()
        
        fig.add_trace(
            go.Scatter(x=monthly_tp.index, y=monthly_tp.values,
                      mode='lines+markers', name='TP Trend',
                      line=dict(color='green')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=monthly_fp.index, y=monthly_fp.values,
                      mode='lines+markers', name='FP Trend',
                      line=dict(color='red')),
            row=2, col=1
        )
    
    # 4. Speaker analysis
    speaker_counts = negation_df['Speaker'].value_counts()
    fig.add_trace(
        go.Bar(x=speaker_counts.index, y=speaker_counts.values,
               name='Speaker Distribution',
               marker_color=['lightblue', 'lightcoral']),
        row=2, col=2
    )
    
    # 5. Performance metrics
    if len(pattern_analysis_df) > 0:
        fig.add_trace(
            go.Bar(x=pattern_analysis_df['Cluster_ID'],
                   y=pattern_analysis_df['TP_Rate'],
                   name='TP Rate',
                   marker_color='green'),
            row=3, col=1
        )
    
    # 6. Quality scores
    if len(pattern_analysis_df) > 0 and 'Quality_Score' in pattern_analysis_df.columns:
        colors = ['red' if x < 0 else 'green' for x in pattern_analysis_df['Quality_Score']]
        fig.add_trace(
            go.Scatter(x=pattern_analysis_df['Cluster_ID'],
                      y=pattern_analysis_df['Quality_Score'],
                      mode='markers',
                      name='Quality Score',
                      marker=dict(size=12, color=colors, opacity=0.8)),
            row=3, col=2
        )
    
    fig.update_layout(height=1200, showlegend=True, 
                      title_text="Dynamic Negation Analysis Dashboard")
    fig.write_html(f'{output_dir}/interactive_dashboard.html')

   
# Create all visualizations
create_comprehensive_visualizations(negation_df_clustered, pattern_analysis_df)

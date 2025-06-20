# =============================================================================
# FIXES FOR VISUALIZATION ISSUES
# Replace the problematic functions with these corrected versions
# =============================================================================

def create_performance_heatmaps_fixed(negation_df, output_dir):
    """Fixed performance heatmaps - handles data structure issues"""
    
    print("Creating performance heatmaps (FIXED)...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Debug: Check data structure
    print(f"Columns in negation_df: {list(negation_df.columns)}")
    print(f"Unique Primary_Marker values: {negation_df['Primary_Marker'].unique()}")
    print(f"Sample data shape: {negation_df.shape}")
    
    # 1. FIXED: Period vs Negation Word Performance
    if 'Period' in negation_df.columns and 'Negation_Word' in negation_df.columns:
        # Create a proper crosstab with correct aggregation
        period_word_data = []
        
        for period in negation_df['Period'].unique():
            for word in negation_df['Negation_Word'].value_counts().head(8).index:  # Top 8 words only
                subset = negation_df[(negation_df['Period'] == period) & (negation_df['Negation_Word'] == word)]
                if len(subset) > 0:
                    tp_count = len(subset[subset['Primary_Marker'] == 'TP'])
                    total_count = len(subset)
                    tp_rate = tp_count / total_count if total_count > 0 else 0
                    
                    period_word_data.append({
                        'Period': period,
                        'Negation_Word': word,
                        'TP_Rate': tp_rate,
                        'Total_Count': total_count
                    })
        
        if period_word_data:
            heatmap_df = pd.DataFrame(period_word_data)
            # Pivot to create heatmap matrix
            heatmap_matrix = heatmap_df.pivot(index='Period', columns='Negation_Word', values='TP_Rate').fillna(0)
            
            if not heatmap_matrix.empty:
                sns.heatmap(heatmap_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
                           ax=axes[0], cbar_kws={'label': 'TP Rate'}, vmin=0, vmax=1)
                axes[0].set_title('TP Rate: Period vs Negation Word')
                axes[0].set_ylabel('Period')
                axes[0].set_xlabel('Negation Word')
            else:
                axes[0].text(0.5, 0.5, 'No data for Period vs Word analysis', 
                            ha='center', va='center', transform=axes[0].transAxes)
        else:
            axes[0].text(0.5, 0.5, 'Insufficient data for heatmap', 
                        ha='center', va='center', transform=axes[0].transAxes)
    else:
        axes[0].text(0.5, 0.5, 'Missing Period or Negation_Word columns', 
                    ha='center', va='center', transform=axes[0].transAxes)
    
    # 2. FIXED: Speaker vs Pattern Performance
    if 'Pattern_Cluster' in negation_df.columns and 'Speaker' in negation_df.columns:
        speaker_pattern_data = []
        
        for speaker in negation_df['Speaker'].unique():
            for pattern in negation_df['Pattern_Cluster'].unique():
                subset = negation_df[(negation_df['Speaker'] == speaker) & (negation_df['Pattern_Cluster'] == pattern)]
                if len(subset) > 0:
                    tp_count = len(subset[subset['Primary_Marker'] == 'TP'])
                    total_count = len(subset)
                    tp_rate = tp_count / total_count if total_count > 0 else 0
                    
                    speaker_pattern_data.append({
                        'Speaker': speaker,
                        'Pattern_Cluster': pattern,
                        'TP_Rate': tp_rate,
                        'Total_Count': total_count
                    })
        
        if speaker_pattern_data:
            heatmap_df2 = pd.DataFrame(speaker_pattern_data)
            heatmap_matrix2 = heatmap_df2.pivot(index='Speaker', columns='Pattern_Cluster', values='TP_Rate').fillna(0)
            
            if not heatmap_matrix2.empty:
                sns.heatmap(heatmap_matrix2, annot=True, fmt='.2f', cmap='RdYlGn', 
                           ax=axes[1], cbar_kws={'label': 'TP Rate'}, vmin=0, vmax=1)
                axes[1].set_title('TP Rate: Speaker vs Pattern')
                axes[1].set_ylabel('Speaker')
                axes[1].set_xlabel('Pattern Cluster')
            else:
                axes[1].text(0.5, 0.5, 'No data for Speaker vs Pattern analysis', 
                            ha='center', va='center', transform=axes[1].transAxes)
        else:
            axes[1].text(0.5, 0.5, 'Insufficient data for heatmap', 
                        ha='center', va='center', transform=axes[1].transAxes)
    else:
        axes[1].text(0.5, 0.5, 'Missing Pattern_Cluster or Speaker columns', 
                    ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_heatmaps_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_context_wordclouds_fixed(negation_df, pattern_analysis_df, output_dir):
    """Fixed word clouds - handles missing patterns and context issues"""
    
    print("Creating context word clouds (FIXED)...")
    
    if 'Pattern_Cluster' not in negation_df.columns:
        print("No pattern clusters found for word clouds")
        return
    
    # Debug information
    print(f"Available pattern clusters: {sorted(negation_df['Pattern_Cluster'].unique())}")
    print(f"Pattern analysis clusters: {sorted(pattern_analysis_df['Cluster_ID'].unique()) if len(pattern_analysis_df) > 0 else 'None'}")
    
    # Get all available patterns (not just top ones)
    if len(pattern_analysis_df) > 0:
        available_patterns = sorted(pattern_analysis_df['Cluster_ID'].unique())
    else:
        available_patterns = sorted(negation_df['Pattern_Cluster'].unique())
    
    # Filter out patterns with insufficient data
    valid_patterns = []
    for pattern_id in available_patterns:
        pattern_data = negation_df[negation_df['Pattern_Cluster'] == pattern_id]
        if len(pattern_data) >= 5:  # Minimum 5 instances
            # Check if context data exists
            contexts = pattern_data['Context'].dropna()
            if len(contexts) > 0 and contexts.str.len().sum() > 100:  # At least 100 characters total
                valid_patterns.append(pattern_id)
            else:
                print(f"Pattern {pattern_id}: Insufficient context data")
        else:
            print(f"Pattern {pattern_id}: Only {len(pattern_data)} instances (need >= 5)")
    
    print(f"Valid patterns for word clouds: {valid_patterns}")
    
    if len(valid_patterns) == 0:
        print("No patterns have sufficient data for word clouds")
        return
    
    # Create subplot grid
    n_patterns = len(valid_patterns)
    n_cols = min(3, n_patterns)  # Max 3 columns
    n_rows = (n_patterns + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Handle single subplot case
    if n_patterns == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, pattern_id in enumerate(valid_patterns):
        ax = axes[i] if n_patterns > 1 else axes[0]
        
        # Get contexts for this pattern
        pattern_contexts = negation_df[negation_df['Pattern_Cluster'] == pattern_id]['Context'].dropna()
        
        if len(pattern_contexts) > 0:
            # Combine all contexts
            combined_text = ' '.join(pattern_contexts.astype(str))
            
            # Clean text for word cloud
            import re
            # Remove common stopwords and clean
            clean_text = re.sub(r'[^\w\s]', ' ', combined_text.lower())
            clean_text = ' '.join([word for word in clean_text.split() 
                                 if len(word) > 2 and word not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'end', 'few', 'got', 'let', 'put', 'say', 'she', 'too', 'use']])
            
            if len(clean_text) > 10:  # Only create if sufficient text
                try:
                    wordcloud = WordCloud(
                        width=400, height=300, 
                        background_color='white',
                        max_words=40,
                        relative_scaling=0.5,
                        colormap='viridis'
                    ).generate(clean_text)
                    
                    ax.imshow(wordcloud, interpolation='bilinear')
                    
                    # Get pattern info for title
                    pattern_count = len(negation_df[negation_df['Pattern_Cluster'] == pattern_id])
                    tp_count = len(negation_df[(negation_df['Pattern_Cluster'] == pattern_id) & (negation_df['Primary_Marker'] == 'TP')])
                    tp_rate = tp_count / pattern_count if pattern_count > 0 else 0
                    
                    ax.set_title(f'Pattern {pattern_id}\n{pattern_count} instances (TP: {tp_rate:.1%})', fontsize=10)
                    ax.axis('off')
                    
                except Exception as e:
                    print(f"Error creating wordcloud for pattern {pattern_id}: {e}")
                    ax.text(0.5, 0.5, f'Pattern {pattern_id}\nWordcloud error', 
                           ha='center', va='center', transform=ax.transAxes)
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
    for i in range(len(valid_patterns), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pattern_context_wordclouds_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created word clouds for {len(valid_patterns)} patterns")

def create_temporal_analysis_viz_fixed(negation_df, output_dir):
    """Fixed temporal analysis - addresses the unique negations calculation"""
    
    print("Creating temporal analysis (FIXED)...")
    
    # Monthly evolution analysis
    if 'Year_Month' not in negation_df.columns:
        print("No Year_Month column found")
        return
    
    # FIXED: Proper unique negation calculation
    monthly_stats = []
    
    for month in negation_df['Year_Month'].unique():
        for marker in ['TP', 'FP']:
            month_marker_data = negation_df[(negation_df['Year_Month'] == month) & (negation_df['Primary_Marker'] == marker)]
            
            if len(month_marker_data) > 0:
                # Count total instances
                volume = len(month_marker_data)
                
                # Count unique negation words (not total unique negations)
                unique_neg_words = month_marker_data['Negation_Word'].nunique()
                
                # Count unique UUIDs (unique conversations)
                unique_conversations = month_marker_data['UUID'].nunique()
                
                monthly_stats.append({
                    'Year_Month': month,
                    'Primary_Marker': marker,
                    'Volume': volume,
                    'Unique_Negation_Words': unique_neg_words,
                    'Unique_Conversations': unique_conversations
                })
    
    monthly_stats_df = pd.DataFrame(monthly_stats)
    
    if len(monthly_stats_df) == 0:
        print("No monthly statistics to plot")
        return
    
    # Create interactive plot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Volume Over Time', 'Unique Negation Words Over Time',
                       'Unique Conversations Over Time', 'TP/FP Ratio Over Time'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Volume trend
    tp_monthly = monthly_stats_df[monthly_stats_df['Primary_Marker'] == 'TP']
    fp_monthly = monthly_stats_df[monthly_stats_df['Primary_Marker'] == 'FP']
    
    fig.add_trace(
        go.Scatter(x=tp_monthly['Year_Month'], y=tp_monthly['Volume'],
                  mode='lines+markers', name='TP Volume', line=dict(color='green'),
                  hovertemplate='Month: %{x}<br>TP Volume: %{y}<extra></extra>'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=fp_monthly['Year_Month'], y=fp_monthly['Volume'],
                  mode='lines+markers', name='FP Volume', line=dict(color='red'),
                  hovertemplate='Month: %{x}<br>FP Volume: %{y}<extra></extra>'),
        row=1, col=1
    )
    
    # FIXED: Unique negation words (not misleading "unique negations")
    fig.add_trace(
        go.Scatter(x=tp_monthly['Year_Month'], y=tp_monthly['Unique_Negation_Words'],
                  mode='lines+markers', name='TP Unique Words', line=dict(color='darkgreen'),
                  hovertemplate='Month: %{x}<br>Unique Neg Words: %{y}<extra></extra>'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=fp_monthly['Year_Month'], y=fp_monthly['Unique_Negation_Words'],
                  mode='lines+markers', name='FP Unique Words', line=dict(color='darkred'),
                  hovertemplate='Month: %{x}<br>Unique Neg Words: %{y}<extra></extra>'),
        row=1, col=2
    )
    
    # NEW: Unique conversations
    fig.add_trace(
        go.Scatter(x=tp_monthly['Year_Month'], y=tp_monthly['Unique_Conversations'],
                  mode='lines+markers', name='TP Conversations', line=dict(color='blue'),
                  hovertemplate='Month: %{x}<br>Unique Conversations: %{y}<extra></extra>'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=fp_monthly['Year_Month'], y=fp_monthly['Unique_Conversations'],
                  mode='lines+markers', name='FP Conversations', line=dict(color='orange'),
                  hovertemplate='Month: %{x}<br>Unique Conversations: %{y}<extra></extra>'),
        row=2, col=1
    )
    
    # TP/FP Ratio calculation
    ratio_data = []
    for month in tp_monthly['Year_Month'].unique():
        tp_vol = tp_monthly[tp_monthly['Year_Month'] == month]['Volume'].sum()
        fp_vol = fp_monthly[fp_monthly['Year_Month'] == month]['Volume'].sum()
        if tp_vol > 0 or fp_vol > 0:
            ratio = tp_vol / (fp_vol + 1)  # Add 1 to avoid division by zero
            ratio_data.append({'Month': month, 'Ratio': ratio, 'TP_Vol': tp_vol, 'FP_Vol': fp_vol})
    
    if ratio_data:
        ratio_df = pd.DataFrame(ratio_data)
        fig.add_trace(
            go.Scatter(x=ratio_df['Month'], y=ratio_df['Ratio'],
                      mode='lines+markers', name='TP/FP Ratio', line=dict(color='purple'),
                      hovertemplate='Month: %{x}<br>TP/FP Ratio: %{y:.2f}<extra></extra>'),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=True, 
                      title_text="Temporal Analysis of Negation Patterns (FIXED)")
    fig.write_html(f'{output_dir}/temporal_analysis_fixed.html')
    
    # Print debugging info
    print("TEMPORAL ANALYSIS DEBUG INFO:")
    print(f"Total months analyzed: {len(monthly_stats_df['Year_Month'].unique())}")
    print(f"TP months: {len(tp_monthly)}")
    print(f"FP months: {len(fp_monthly)}")
    print("\nSample monthly stats:")
    print(monthly_stats_df.head(10))

# =============================================================================
# UPDATED MAIN FUNCTION - Replace the original one
# =============================================================================

def create_comprehensive_visualizations_fixed(negation_df, pattern_analysis_df, output_dir='dynamic_negation_visualizations_fixed'):
    """
    Fixed version of comprehensive visualizations
    """
    
    print("\n" + "="*60)
    print("CREATING COMPREHENSIVE VISUALIZATIONS (FIXED)")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if len(negation_df) == 0:
        print("No negation data to visualize!")
        return
    
    # Debug info
    print(f"Negation data shape: {negation_df.shape}")
    print(f"Columns: {list(negation_df.columns)}")
    print(f"Primary Marker values: {negation_df['Primary_Marker'].value_counts()}")
    if 'Pattern_Cluster' in negation_df.columns:
        print(f"Pattern clusters: {sorted(negation_df['Pattern_Cluster'].unique())}")
    
    # 1. PATTERN DISTRIBUTION ANALYSIS (original - working)
    create_pattern_distribution_viz(negation_df, pattern_analysis_df, output_dir)
    
    # 2. FIXED TEMPORAL ANALYSIS
    create_temporal_analysis_viz_fixed(negation_df, output_dir)
    
    # 3. SPEAKER ANALYSIS (original - working)
    create_speaker_analysis_viz(negation_df, output_dir)
    
    # 4. FIXED CONTEXT WORD CLOUDS
    create_context_wordclouds_fixed(negation_df, pattern_analysis_df, output_dir)
    
    # 5. FIXED PERFORMANCE HEATMAPS
    create_performance_heatmaps_fixed(negation_df, output_dir)
    
    # 6. INTERACTIVE DASHBOARD (original - working)
    create_interactive_dashboard(negation_df, pattern_analysis_df, output_dir)
    
    print(f"All FIXED visualizations saved to {output_dir}/")

# =============================================================================
# DIAGNOSTIC FUNCTION - Run this first to check your data
# =============================================================================

def diagnose_visualization_data(negation_df, pattern_analysis_df):
    """Diagnose data issues before running visualizations"""
    
    print("\n" + "="*60)
    print("DATA DIAGNOSTIC FOR VISUALIZATIONS")
    print("="*60)
    
    print("1. BASIC DATA CHECK:")
    print(f"   Negation DataFrame shape: {negation_df.shape}")
    print(f"   Pattern Analysis DataFrame shape: {pattern_analysis_df.shape}")
    
    print("\n2. REQUIRED COLUMNS CHECK:")
    required_cols = ['UUID', 'Primary_Marker', 'Speaker', 'Negation_Word', 'Context']
    missing_cols = [col for col in required_cols if col not in negation_df.columns]
    if missing_cols:
        print(f"   MISSING: {missing_cols}")
    else:
        print("   All required columns present")
    
    print("\n3. DATA DISTRIBUTION:")
    print(f"   Primary Marker: {dict(negation_df['Primary_Marker'].value_counts())}")
    print(f"   Speaker: {dict(negation_df['Speaker'].value_counts())}")
    
    if 'Pattern_Cluster' in negation_df.columns:
        print(f"   Pattern Clusters: {sorted(negation_df['Pattern_Cluster'].unique())}")
        print(f"   Pattern distribution: {dict(negation_df['Pattern_Cluster'].value_counts())}")
    
    print("\n4. CONTEXT DATA CHECK:")
    if 'Context' in negation_df.columns:
        non_empty_contexts = negation_df['Context'].dropna()
        print(f"   Non-empty contexts: {len(non_empty_contexts)}/{len(negation_df)}")
        if len(non_empty_contexts) > 0:
            avg_length = non_empty_contexts.str.len().mean()
            print(f"   Average context length: {avg_length:.1f} characters")
    
    print("\n5. TEMPORAL DATA CHECK:")
    if 'Year_Month' in negation_df.columns:
        print(f"   Available months: {sorted(negation_df['Year_Month'].unique())}")
        monthly_counts = negation_df['Year_Month'].value_counts().sort_index()
        print(f"   Monthly distribution: {dict(monthly_counts)}")
    
    print("\n6. RECOMMENDATIONS:")
    if len(negation_df) < 100:
        print("   WARNING: Low data volume may result in sparse visualizations")
    
    if 'Pattern_Cluster' in negation_df.columns:
        pattern_counts = negation_df['Pattern_Cluster'].value_counts()
        small_patterns = pattern_counts[pattern_counts < 5]
        if len(small_patterns) > 0:
            print(f"   NOTE: {len(small_patterns)} patterns have <5 instances (may skip word clouds)")
    
    return True

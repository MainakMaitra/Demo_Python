import pandas as pd

def calculate_insight_presence(df):
    total_records = len(df)

    insight_summary = []

    # 1. Negation-Heavy Contexts
    negation_mask = df['Customer_Negation_Count'] >= 5  # threshold as observed in analysis
    insight_summary.append({
        'Insight': 'Negation-Heavy Contexts',
        'TP_Volume': df[negation_mask & (df['Primary Marker'] == 'TP')].shape[0],
        'FP_Volume': df[negation_mask & (df['Primary Marker'] == 'FP')].shape[0],
        'Total_Records': total_records
    })

    # 2. Agent-Dominated Turns
    agent_dom_mask = df['Customer_Agent_Ratio'] < 0.9
    insight_summary.append({
        'Insight': 'Agent-Dominated Turns',
        'TP_Volume': df[agent_dom_mask & (df['Primary Marker'] == 'TP')].shape[0],
        'FP_Volume': df[agent_dom_mask & (df['Primary Marker'] == 'FP')].shape[0],
        'Total_Records': total_records
    })

    # 3. Single-Category Transcripts
    # assuming only Prosodica L1 and L2 used for labeling
    single_cat_mask = df.groupby('variable5')['Prosodica L1'].transform('nunique') == 1
    insight_summary.append({
        'Insight': 'Single-Category Transcripts',
        'TP_Volume': df[single_cat_mask & (df['Primary Marker'] == 'TP')].shape[0],
        'FP_Volume': df[single_cat_mask & (df['Primary Marker'] == 'FP')].shape[0],
        'Total_Records': total_records
    })

    # 4. High Politeness / Uncertainty
    polite_uncertain_mask = (
        (df['Customer Transcript'].str.count(r'\bplease\b|\bthank you\b|\bsorry\b|\bnot sure\b|\bdon\'t understand\b', flags=re.IGNORECASE)) >= 2
    )
    insight_summary.append({
        'Insight': 'High Politeness/Uncertainty',
        'TP_Volume': df[polite_uncertain_mask & (df['Primary Marker'] == 'TP')].shape[0],
        'FP_Volume': df[polite_uncertain_mask & (df['Primary Marker'] == 'FP')].shape[0],
        'Total_Records': total_records
    })

    # 5. Short Complaint Descriptions
    short_transcript_mask = df['Transcript_Length'] < 2500  # Based on TP avg being 6k+ and FP ~3.9k
    insight_summary.append({
        'Insight': 'Short Complaint Descriptions',
        'TP_Volume': df[short_transcript_mask & (df['Primary Marker'] == 'TP')].shape[0],
        'FP_Volume': df[short_transcript_mask & (df['Primary Marker'] == 'FP')].shape[0],
        'Total_Records': total_records
    })

    # Final summary with percentage
    summary_df = pd.DataFrame(insight_summary)
    summary_df['% Presence'] = ((summary_df['TP_Volume'] + summary_df['FP_Volume']) / total_records * 100).round(2)

    return summary_df

# Example usage:
# summary_df = calculate_insight_presence(df_main)
# print(summary_df)

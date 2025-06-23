import pandas as pd

def generate_performance_metrics(df):
    output = {}

    ## Primary: Overall Precision
    total_tp = df[df['Primary Marker'] == 'TP'].shape[0]
    total_fp = df[df['Primary Marker'] == 'FP'].shape[0]
    primary_precision = round(100 * total_tp / (total_tp + total_fp), 1)
    output['Primary_Precision'] = primary_precision
    output['Primary_Gap'] = round(primary_precision - 70.0, 1)
    output['Primary_Status'] = (
        "EXCEEDING" if primary_precision >= 70.0 else "FAILING"
    )

    ## Secondary: Category-Level Precision Check
    category_precision = (
        df.groupby('Prosodica L1')['Primary Marker']
        .value_counts()
        .unstack()
        .fillna(0)
    )
    category_precision['Precision'] = (
        category_precision['TP'] / (category_precision['TP'] + category_precision['FP'])
    )
    num_failing = (category_precision['Precision'] < 0.60).sum()
    total_categories = category_precision.shape[0]

    output['Secondary_Summary'] = f"{num_failing}/{total_categories} below 70%"
    output['Secondary_Gap'] = round(-100 * (num_failing / total_categories), 1)
    output['Secondary_Status'] = (
        "FAILING" if num_failing > 0 else "EXCEEDING"
    )

    ## Tertiary: Validation Agreement
    df_valid = df[df['Has_Secondary_Validation'] == True]
    if len(df_valid) > 0:
        agreement = round(100 * df_valid['Primary_Secondary_Agreement'].mean(), 1)
    else:
        agreement = 0.0

    output['Validation_Agreement'] = agreement
    output['Validation_Gap'] = round(agreement - 85.0, 1)
    output['Validation_Status'] = (
        "EXCEEDING" if agreement >= 85 else "BELOW"
    )

    return output, category_precision.reset_index()

# Usage:
# metrics_summary, cat_precision_df = generate_performance_metrics(df_main)
# pd.DataFrame.from_dict(metrics_summary, orient='index')


def get_monthly_precision_trend(df, date_column='Month'):
    df['Month'] = pd.to_datetime(df[date_column]).dt.to_period('M').astype(str)
    monthly = df.groupby('Month')['Primary Marker'].value_counts().unstack().fillna(0)
    monthly['Precision'] = (monthly['TP'] / (monthly['TP'] + monthly['FP']) * 100).round(1)
    return monthly.reset_index()[['Month', 'Precision']]

# Usage:
# monthly_trend = get_monthly_precision_trend(df_main)
# print(monthly_trend)

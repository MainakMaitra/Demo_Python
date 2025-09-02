# PQC Manager Monthly Analysis - Replicating Manager's Output Format
# This snippet generates the same table format as your manager's independent analysis

import warnings
import oracledb
import configparser
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# Database Connection Setup (same as original)
def connect_to_database():
    """Connect to Oracle database using existing config"""
    config = configparser.ConfigParser(interpolation=None)
    config.read("/var/run/secrets/user_credentials/PQC_CONFIG")

    # Get credentials
    hostname = config.get("pub_conn", "hostname")
    port = config.get("pub_conn", "port")
    servicename = config.get("pub_conn", "servicename")
    username = config.get("pub_conn", "username")
    password = config.get("pub_conn", "pwd")

    # Connect
    conn_str = f"{hostname}:{port}/{servicename}"
    connection = oracledb.connect(user=username, password=password, dsn=conn_str)
    return connection

# Connect to database
print("Connecting to database...")
conn = connect_to_database()
print("Connected successfully!")

# Define date range - Past 3 months (June, July, August 2025)
start_date = "2025-06-01"
end_date = "2025-08-31"
print(f"Analyzing data from {start_date} to {end_date}")

# Load case assignments data (same query as original)
query_assignments = f"""
SELECT * FROM pqc_case_assignments 
WHERE PQC_ASSIGNMENT_DATE >= '{start_date}' 
AND PQC_ASSIGNMENT_DATE <= '{end_date}'
"""
df_assignments = pd.read_sql(query_assignments, conn)
print(f"Assignments loaded: {len(df_assignments)} records")

# Clean column names
df_assignments.columns = df_assignments.columns.str.lower().str.strip()

# Convert dates to datetime
df_assignments["pqc_assignment_date"] = pd.to_datetime(df_assignments["pqc_assignment_date"])

print("\n" + "=" * 70)
print("PQC ASSIGNMENT ANALYSIS - MONTHLY BREAKDOWN BY MANAGER")
print("=" * 70)

# Create manager lookup to get both name and SSO
manager_lookup = df_assignments[['pqc_sso', 'pqc_name']].drop_duplicates()

# Calculate monthly averages for each manager
monthly_data = []

# Define monthly date ranges
months = {
    'June 2025': ('2025-06-01', '2025-06-30'),
    'July 2025': ('2025-07-01', '2025-07-31'),
    'August 2025': ('2025-08-01', '2025-08-31')
}

# Process each manager
for _, manager_row in manager_lookup.iterrows():
    manager_sso = manager_row['pqc_sso']
    manager_name = manager_row['pqc_name']
    
    # Get all assignments for this manager
    manager_data = df_assignments[df_assignments['pqc_sso'] == manager_sso]
    
    monthly_averages = {'pqc_sso': manager_sso, 'pqc_name': manager_name}
    
    # Calculate average for each month
    for month_name, (month_start, month_end) in months.items():
        month_assignments = manager_data[
            (manager_data['pqc_assignment_date'] >= month_start) & 
            (manager_data['pqc_assignment_date'] <= month_end)
        ]
        
        if len(month_assignments) > 0:
            # Group by date and count daily cases, then take average
            daily_cases = month_assignments.groupby('pqc_assignment_date').size()
            monthly_avg = daily_cases.mean()
            monthly_averages[month_name] = monthly_avg
        else:
            monthly_averages[month_name] = 0
    
    monthly_data.append(monthly_averages)

# Create DataFrame
results_df = pd.DataFrame(monthly_data)

# Sort by manager name for consistent output
results_df = results_df.sort_values('pqc_name').reset_index(drop=True)

# Display results in the same format as your manager's table
print(f"\n{'Pqc Name':<25} {'SSO':<15} {'June 2025':<12} {'July 2025':<12} {'August 2025':<12}")
print("=" * 80)

for _, row in results_df.iterrows():
    june_val = f"{row['June 2025']:.1f}" if row['June 2025'] > 0 else "0"
    july_val = f"{row['July 2025']:.1f}" if row['July 2025'] > 0 else "0"
    august_val = f"{row['August 2025']:.1f}" if row['August 2025'] > 0 else "0"
    
    print(f"{row['pqc_name']:<25} {row['pqc_sso']:<15} {june_val:<12} {july_val:<12} {august_val:<12}")

# Summary statistics
print(f"\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

total_managers = len(results_df)
print(f"Total Managers Analyzed: {total_managers}")

for month_name in ['June 2025', 'July 2025', 'August 2025']:
    month_data = results_df[results_df[month_name] > 0][month_name]
    active_managers = len(month_data)
    avg_cases = month_data.mean() if len(month_data) > 0 else 0
    
    print(f"\n{month_name}:")
    print(f"  - Active Managers: {active_managers}")
    print(f"  - Average Cases per Day: {avg_cases:.1f}")
    print(f"  - Min Cases per Day: {month_data.min():.1f}" if len(month_data) > 0 else "  - Min Cases per Day: 0")
    print(f"  - Max Cases per Day: {month_data.max():.1f}" if len(month_data) > 0 else "  - Max Cases per Day: 0")

# Close database connection
conn.close()
print(f"\nAnalysis complete. Database connection closed.")

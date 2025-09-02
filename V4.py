# Detailed manager breakdown by month
print("\n" + "="*60)
print("INDIVIDUAL MANAGER PRODUCTIVITY BY MONTH")
print("="*60)

manager_monthly_avg = {}
for manager in df_assignments['pqc_sso'].unique():
    manager_data = df_assignments[df_assignments['pqc_sso'] == manager]
    monthly_stats = {}
    
    for month, (start, end) in [('June', ('2025-06-01', '2025-06-30')),
                               ('July', ('2025-07-01', '2025-07-31')), 
                               ('August', ('2025-08-01', '2025-08-31'))]:
        month_cases = manager_data[
            (manager_data['pqc_assignment_date'] >= start) & 
            (manager_data['pqc_assignment_date'] <= end)
        ]
        if len(month_cases) > 0:
            daily_avg = month_cases.groupby('pqc_assignment_date').size().mean()
            monthly_stats[month] = round(daily_avg, 1)
        else:
            monthly_stats[month] = 0
    
    manager_monthly_avg[manager] = monthly_stats

# Print in table format similar to your manager's
print(f"{'Manager Name':<20} {'June':<8} {'July':<8} {'August':<8}")
print("-" * 50)
for manager, months in manager_monthly_avg.items():
    print(f"{manager:<20} {months['June']:<8} {months['July']:<8} {months['August']:<8}")

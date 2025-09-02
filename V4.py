# =============================================================================
# FINAL SUMMARY - ALL ANSWERS CONSOLIDATED
# =============================================================================

print("\n" + "=" * 80)
print("PQC DATA ANALYSIS - FINAL RESULTS SUMMARY")
print("Data Period: June 1 - August 31, 2025")
print("=" * 80)

print("\n QUALITY REVIEW STATS")
print("-" * 50)
print(f"4. Average cases per day per PQC manager: {overall_avg_cases_per_day:.2f}")
print(f"5. Managers working <5 cases/day: {percent_less_than_5:.1f}%")
print(f"6. Managers working 5-8 cases/day: {percent_5_to_8:.1f}%")
print(f"7. Managers working 9+ cases/day: {percent_9_plus:.1f}%")
print(f"\n9. Sendback rates by workload group:")
print(f"   • <5 cases/day: {less_than_5_sendback:.1f}%")
print(f"   • 5-8 cases/day: {five_to_8_sendback:.1f}%")
print(f"   • 9+ cases/day: {nine_plus_sendback:.1f}%")
print(f"\n10. Correction rates by workload group:")
print(f"   • <5 cases/day: {less_than_5_correction:.1f}%")
print(f"   • 5-8 cases/day: {five_to_8_correction:.1f}%")
print(f"   • 9+ cases/day: {nine_plus_correction:.1f}%")
print(f"\n11. Approval rates by workload group:")
print(f"   • <5 cases/day: {less_than_5_approval:.1f}%")
print(f"   • 5-8 cases/day: {five_to_8_approval:.1f}%")
print(f"   • 9+ cases/day: {nine_plus_approval:.1f}%")

print("\n GEOGRAPHY (INDIA VS DOMESTIC)")
print("-" * 50)
print(f"14. Daily case volume:")
print(f"   • India: {avg_india_daily:.1f} cases/day")
print(f"   • Domestic: {avg_domestic_daily:.1f} cases/day")
print(f"\n15. Sendback rates:")
print(f"   • India: {india_sendback_rate:.1f}%")
print(f"   • Domestic: {domestic_sendback_rate:.1f}%")
print(f"\n16. Correction rates:")
print(f"   • India: {india_correction_rate:.1f}%")
print(f"   • Domestic: {domestic_correction_rate:.1f}%")

print("\n GENERAL INFORMATION")
print("-" * 50)
print(f"19. Average outflow per day: {avg_outflow_per_day:.1f} cases")
print(f"20. Average inflow per day: {avg_inflow_per_day:.1f} cases")
print(f"22. Cases fixed by PQC: {cases_with_corrections:,}")
print(f"23. Cases sent back: {cases_sent_back:,}")
print(f"25. Rate of cases fixed only: {rate_fixed_only:.1f}%")
print(f"26. Cases requiring PQC response fix: {cases_requiring_pqc_fix:,}")
print(f"27. Rate of cases sent back: {rate_sent_back:.1f}%")

print("\n OPERATIONS - TOP CCR MANAGERS")
print("-" * 50)
print("30. Defect rates by CCR Manager (min 5 cases):")
top_defect_managers = ccr_manager_defects.nlargest(5, "defect_rate")
for _, row in top_defect_managers.iterrows():
    print(
        f"   • {row['ccr_manager']}: {row['defect_rate']:.1f}% ({int(row['total_cases'])} cases)"
    )

print(f"\n31. Sendback rates by CCR Manager (min 5 cases):")
top_sendback_managers = ccr_manager_sendbacks.nlargest(5, "sendback_rate")
for _, row in top_sendback_managers.iterrows():
    print(
        f"   • {row['ccr_manager']}: {row['sendback_rate']:.1f}% ({int(row['total_cases'])} cases)"
    )

print("\n KEY METRICS SUMMARY")
print("-" * 50)
overall_defect_rate = (df_aggr["case_key_lvl_defect_flag"] == "True").mean() * 100
overall_sendback_rate = (df_aggr["case_key_lvl_sendback_flag"] == "True").mean() * 100
overall_approval_rate = 100 - overall_defect_rate

print(f"• Total cases analyzed: {len(df_aggr):,}")
print(f"• Total PQC managers: {total_managers}")
print(f"• Overall defect rate: {overall_defect_rate:.1f}%")
print(f"• Overall sendback rate: {overall_sendback_rate:.1f}%")
print(f"• Overall approval rate: {overall_approval_rate:.1f}%")
print(f"• India managers: {len(india_managers)} ({', '.join(india_managers)})")
print(
    f"• India vs Domestic case split: {len(india_assignments):,} vs {len(domestic_assignments):,}"

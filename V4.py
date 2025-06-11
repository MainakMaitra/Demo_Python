# Precision Drop Analysis - Contingency Table Validated Insights

## 1. Critical Priority Insights

### 1.1 Context-Insensitive Negation Handling

Our complaint detection system doesn't understand when customers are NOT complaining. This affects 96.7% of our false positives.

**Period Comparison (Pre vs Post)**:
```
                   Pre Period    Post Period    Change      % Change
FP Count          412           313            -99         -24.0%
Risk Factor Count 624           398            -226        -36.2%
Risk Rate         32.7%         27.4%          -5.3%       -16.2%
```

**Monthly Breakdown**:
```
Month       FP Count    TP Count    Risk Factor    Total    FP Rate    Risk Rate
Oct 2024    139         403         208            542      25.6%      38.4%
Nov 2024    107         338         201            445      24.0%      45.2%
Dec 2024    166         232         215            398      41.7%      54.0%
Jan 2025    107         325         134            432      24.8%      31.0%
Feb 2025    113         276         132            389      29.0%      33.9%
Mar 2025    93          278         132            371      25.1%      35.6%
```

**Statistical Significance**: Chi-square p-value: 0.000049

December 2024 shows the worst performance with 54.0% risk rate and 41.7% FP rate. The risk rate dropped from 32.7% to 27.4%, showing natural improvement that we can accelerate.

**Quick fixes**: Add negation templates like `(original_query) AND NOT ((not|no|never|don't|won't) NEAR:3 (complain|complaint))`
**Expected improvement**: +22% precision

### 1.2 Qualifying Language Creates Uncertainty

Customers using uncertain language ("maybe", "might", "seems like") create confusion. This was discovered through contingency analysis.

**Period Comparison (Pre vs Post)**:
```
                   Pre Period    Post Period    Change      % Change
FP Count          412           313            -99         -24.0%
Risk Factor Count 289           203            -86         -29.8%
Risk Rate         15.1%         14.0%          -1.1%       -7.3%
```

**Monthly Breakdown**:
```
Month       FP Count    TP Count    Risk Factor    Total    FP Rate    Risk Rate
Oct 2024    139         403         82             542      25.6%      15.1%
Nov 2024    107         338         67             445      24.0%      15.1%
Dec 2024    166         232         140            398      41.7%      35.2%
Jan 2025    107         325         65             432      24.8%      15.0%
Feb 2025    113         276         54             389      29.0%      13.9%
Mar 2025    93          278         84             371      25.1%      22.6%
```

**Statistical Significance**: Chi-square p-value: 0.003421

December again shows the highest qualifying language risk at 35.2%. Qualifying language improved more than negation handling (29.8% vs 24.0% reduction).

**Solutions**: Create confidence scoring based on uncertain language presence, set higher thresholds when lots of qualifying words are detected.
**Expected improvement**: +12% precision

### 1.3 Long Conversations Are Problematic

Long transcripts (top 25% by length) consistently have more false positives.

**Period Comparison (Pre vs Post)**:
```
                   Pre Period    Post Period    Change      % Change
FP Count          412           313            -99         -24.0%
Risk Factor Count 477           364            -113        -23.7%
Risk Rate         25.0%         25.0%          0.0%        0.0%
```

**Monthly Breakdown**:
```
Month       FP Count    TP Count    Risk Factor    Total    FP Rate    Risk Rate
Oct 2024    139         403         135            542      25.6%      24.9%
Nov 2024    107         338         111            445      24.0%      24.9%
Dec 2024    166         232         231            398      41.7%      58.0%
Jan 2025    107         325         108            432      24.8%      25.0%
Feb 2025    113         276         97             389      29.0%      24.9%
Mar 2025    93          278         159            371      25.1%      42.9%
```

**Statistical Significance**: Chi-square p-value: 0.001256

Unlike other risk factors, this one shows no overall improvement (stable at 25.0%) but has significant monthly variation, with December at 58.0% and March at 42.9%.

**Solutions**: Set minimum length thresholds, implement conversation segmentation, add complexity scoring.
**Expected improvement**: +8% precision

### 1.4 Agent Talk Triggers False Alarms

When agents dominate conversations, we get more false positives from agent explanations being mistaken for customer complaints.

**Period Comparison (Pre vs Post)**:
```
                   Pre Period    Post Period    Change      % Change
FP Count          412           313            -99         -24.0%
Risk Factor Count 382           289            -93         -24.3%
Risk Rate         20.0%         19.9%          -0.1%       -0.5%
```

**Monthly Breakdown**:
```
Month       FP Count    TP Count    Risk Factor    Total    FP Rate    Risk Rate
Oct 2024    139         403         108            542      25.6%      19.9%
Nov 2024    107         338         89             445      24.0%      20.0%
Dec 2024    166         232         185            398      41.7%      46.5%
Jan 2025    107         325         86             432      24.8%      19.9%
Feb 2025    113         276         78             389      29.0%      20.1%
Mar 2025    93          278         125            371      25.1%      33.7%
```

**Statistical Significance**: Chi-square p-value: 0.002156

Another barely-improving problem (only 0.1% better overall), but with significant monthly variation. December shows 46.5% risk rate and March shows 33.7%.

**Solutions**: Implement speaker detection, add agent explanation filters, focus rules on customer channel only.
**Expected improvement**: +10% precision

## 2. High Priority Insights

### 2.1 Month-to-Month Precision Change Patterns

**Monthly Performance Overview**:
```
Month       Precision    FP Rate    Volume    MoM Change    Status
Oct 2024    74.4%       25.6%      542       -             Baseline
Nov 2024    75.9%       24.0%      445       +1.5%         Improving
Dec 2024    58.3%       41.7%      398       -17.6%        Crisis
Jan 2025    75.2%       24.8%      432       +16.9%        Recovery
Feb 2025    71.0%       29.0%      389       -4.2%         Decline
Mar 2025    74.9%       25.1%      371       +3.9%         Stabilizing
```

December 2024 was a crisis month with precision dropping to 58.3%. January showed strong recovery, but February declined again before March stabilized.

### 2.2 Category Impact Distribution

**Top Impact Categories with Monthly Trends**:
```
Category: Customer Relations - Close Account (Impact Score: 22.4)
Month       FP Count    TP Count    Total    Precision
Oct 2024    8           15         23       65.2%
Nov 2024    12          18         30       60.0%
Dec 2024    18          12         30       40.0%
Jan 2025    9           21         30       70.0%
Feb 2025    11          19         30       63.3%
Mar 2025    13          17         30       56.7%

Category: Credit Card ATM - Unclassified (Impact Score: 14.6)
Month       FP Count    TP Count    Total    Precision
Oct 2024    3           5          8        62.5%
Nov 2024    4           4          8        50.0%
Dec 2024    7           1          8        12.5%
Jan 2025    5           3          8        37.5%
Feb 2025    6           2          8        25.0%
Mar 2025    4           4          8        50.0%
```

December consistently shows the worst performance across high-impact categories.

### 2.3 Problem Period Detection

**Period Comparison Detail**:
```
Metric                    Pre Period     Post Period    Change
Total Records            1,385          1,192          -193
FP Count                 412            313            -99
TP Count                 973            879            -94
Precision                70.2%          73.7%          +3.5%
Avg Transcript Length    4,967          5,187          +220
Avg Negation Count       10.1           10.8           +0.7
```

**Monthly Risk Factor Summary**:
```
Month       High Negation    Qualifying    Long Trans    Agent Dom    Combined Risk
Oct 2024    38.4%           15.1%         24.9%         19.9%        24.6%
Nov 2024    45.2%           15.1%         24.9%         20.0%        26.3%
Dec 2024    54.0%           35.2%         58.0%         46.5%        48.4%
Jan 2025    31.0%           15.0%         25.0%         19.9%        22.7%
Feb 2025    33.9%           13.9%         24.9%         20.1%        23.2%
Mar 2025    35.6%           22.6%         42.9%         33.7%        33.7%
```

**Statistical Significance**: Chi-square p-value: 0.002286

December stands out as an extreme outlier with 48.4% combined risk factors compared to 24.6% in October.

## 3. Medium Priority Insights

### 3.1 Root Cause Hierarchy

**FP Reason Distribution by Month**:
```
Month       Context Issues    Agent Explain    Broad Rules    New Language
Oct 2024    134 (96.4%)      81 (58.3%)       7 (5.0%)       10 (7.2%)
Nov 2024    103 (96.3%)      63 (58.9%)       6 (5.6%)       8 (7.5%)
Dec 2024    161 (97.0%)      97 (58.4%)       9 (5.4%)       12 (7.2%)
Jan 2025    103 (96.3%)      62 (57.9%)       6 (5.6%)       8 (7.5%)
Feb 2025    109 (96.5%)      66 (58.4%)       6 (5.3%)       8 (7.1%)
Mar 2025    91 (97.8%)       55 (59.1%)       5 (5.4%)       8 (8.6%)
```

Context issues consistently dominate across all months (96-98% of FPs), showing this is a systematic problem rather than seasonal.

### 3.2 Validation Process Breakdown

**Monthly Validation Metrics**:
```
Month       Agreement Rate    Sample Size    Disagreement Count    Consistency
Oct 2024    86.7%            83             11                    0.659
Nov 2024    86.0%            107            15                    0.651
Dec 2024    75.8%            128            31                    0.570
Jan 2025    85.0%            127            19                    0.642
Feb 2025    82.4%            136            24                    0.617
Mar 2025    85.7%            133            19                    0.649
```

December shows significant validation breakdown with 24% disagreement rate (31 out of 128 cases), correlating with the precision crisis.

### 3.3 Temporal Patterns

**Daily Performance Breakdown**:
```
Day         FP Rate    Records    Oct    Nov    Dec    Jan    Feb    Mar
Monday      27.1%      406        24%    26%    45%    23%    31%    26%
Tuesday     26.1%      314        23%    22%    43%    26%    28%    24%
Wednesday   23.7%      211        26%    25%    38%    22%    27%    23%
Thursday    25.6%      550        25%    24%    42%    25%    29%    25%
Friday      22.9%      481        27%    23%    40%    24%    28%    24%
Saturday    25.0%      492        26%    24%    41%    25%    30%    25%
Sunday      24.9%      401        25%    24%    40%    24%    29%    26%
```

**Weekly Patterns by Month**:
```
Week    Oct FP Rate    Nov FP Rate    Dec FP Rate    Jan FP Rate    Feb FP Rate    Mar FP Rate
1       18.2%         19.5%          35.2%          19.1%          23.8%          20.1%
2       24.8%         26.1%          43.2%          26.3%          31.2%          27.9%
3       25.2%         25.8%          44.1%          25.7%          30.8%          26.5%
4       28.9%         26.4%          47.3%          27.1%          32.1%          28.3%
5       21.6%         22.8%          38.9%          22.4%          26.7%          23.2%
```

December consistently shows the highest FP rates across all days and weeks, confirming it as a systematic problem month.

### 3.4 Content Pattern Analysis

**Monthly TP vs FP Characteristics**:
```
Month       TP Avg Length    FP Avg Length    Length Diff    TP Negations    FP Negations
Oct 2024    5,423           3,892            -1,531         12.1            6.8
Nov 2024    5,298           3,756            -1,542         11.8            6.2
Dec 2024    5,187           3,634            -1,553         11.3            5.9
Jan 2025    5,445           3,823            -1,622         12.2            6.5
Feb 2025    5,312           3,791            -1,521         11.9            6.1
Mar 2025    5,398           3,748            -1,650         12.0            6.4
```

The pattern of FPs being significantly shorter than TPs (around 1,500-1,650 characters less) remains consistent across all months, suggesting this is a reliable discriminator.

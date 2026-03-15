# Mining Steam: Player Behavior, Reviews, and Recommendations

Steam Reviews Dataset 2021. 21 million reviews across ~300 games.

Three branches mining the Steam ecosystem from different angles:
* Behavioral & Structured Mining (Sagar)
* Text Mining
* Recommendation & Integration

## Dataset
* Primary: [Steam Reviews Dataset 2021](https://www.kaggle.com/datasets/najzeko/steam-reviews-2021) (~8GB CSV)
* Stratified sampled at 12% by game. ~2.5M rows saved as parquet.

## Behavioral Analysis

### Notebooks
* `01_sampling`: chunked reading of 8GB CSV, stratified sampling by app_id, parquet output
* `02_eda`: distributions, missing values, data quality, correlations
* `03_bias_analysis`: chi-squared tests, Mann-Whitney U, playtime-sentiment thresholds
* `04_clustering`: game-level aggregation, K-Means, 4 market segments
* `05_classification`: logistic regression baseline, SMOTE, Random Forest, cross validation

### Key Findings
* 87.5% positive reviews. Heavily imbalanced
* Playtime is right-skewed (median 31h, mean 147h). Log-transform needed
* Early access reviewers are harsher (82% vs 88%). Statistically significant but weak effect
* Free copy bias is negligible (89% vs 87%, Cramér's V: 0.008)
* Playtime-sentiment relationship is non-linear. Peak satisfaction at 5-100 hours
* 4 game market segments: polarizing AAA, mainstream, blockbusters, early access
* Behavioral features alone are weak predictors of recommendation (Random Forest F1 macro: 0.67)
* Playtime is the strongest behavioral signal (67% feature importance), purchase context barely matters

### Conclusion
Behavioral features capture real but weak patterns. Recommendation is driven more by the actual game experience than by how long someone played or how they acquired the game. This justifies combining behavioral signals with review text (Branch 2) for better predictions and recommendations (Branch 3).

## Setup
```
pip install -r requirements.txt
```

## Requirements
* pandas, numpy, matplotlib, seaborn
* scikit-learn, scipy, pyarrow
* imbalanced-learn (for SMOTE)
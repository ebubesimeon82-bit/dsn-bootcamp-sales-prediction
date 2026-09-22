# DSN Bootcamp Qualification Hackathon 2026 — ML Track

Predicting product-store sales for **DSN Mart**, a retail chain spread across Nigeria.
Built as part of the [DSN Bootcamp Qualification Hackathon 2026](https://www.kaggle.com/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track).

## Problem

Predict `total_sales` for a given product at a given store, using what's known about the product and the store it's sold in. This is a regression problem, scored with **RMSE** (lower is better).

## Result

| | |
|---|---|
| Cross-validated RMSE | ~1,070 |
| Public leaderboard score | 1069.22 |
| Baseline (predicting the average) | 1,698 |
| R² | ~0.60 |

## Key finding

Sales roughly follow **total_sales ≈ price × units sold**, where "units sold" settles at a steady level per store — about **2.4** in a corner shop up to about **26** in the flagship hypermarket. Store format explains most of the variation. Shelf visibility, product weight, fat content and store age have almost no effect once the store is known.

## Approach

1. **Clean** — fix inconsistent category spellings, fill missing store sizes ("Unknown", since it's missing for whole stores), treat `shelf_visibility == 0` as missing, and use one denoised price/weight/visibility value per product.
2. **Feature engineer** — price relative to category average, price per kg, visibility relative to store average, plus one-hot encoded store/product categoricals.
3. **Model on the right target** — instead of predicting `total_sales` directly, models predict `units = total_sales / price`, then multiply the prediction back by price. This bakes the price × units structure into the model.
4. **Compare models** — Linear Regression, Ridge, Random Forest, Gradient Boosting, LightGBM, XGBoost, CatBoost, and a simple "structural" price × store-average-units model, all scored with 5-fold cross-validation.
5. **Small product-level correction** — a heavily shrunken adjustment for how a specific product performs relative to its store's average, shrunk toward zero for products with few observations (most products only appear ~5 times).
6. **Ensemble** — final predictions average several models on the units target.

## Repo contents

| File | Description |
|---|---|
| `DSN_Hackathon_Solution.ipynb` | Full notebook: cleaning, EDA, feature engineering, modelling, predictions |
| `DSN_Hackathon_Report.docx` | Write-up of findings and recommendations |
| `submission.csv` | Final Kaggle submission |
| `README.md` | This file |

## Why the models barely differ

Every reasonably good model lands within a few RMSE points of the others (~1,070–1,088). The remaining error is mostly random variation in units sold that isn't explained by any available column — the noise floor of the data, not a modelling weakness. This was confirmed by testing category effects, store×category interactions, and blended ensembles, none of which improved results meaningfully.

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm xgboost catboost
```

Place `train.csv` and `test.csv` in the same folder as the notebook, then run all cells. This produces `submission.csv`, the figures, and a results summary.

## Author

Simeon Clinton — Computer Science student

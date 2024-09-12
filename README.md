# S4E7 Insurance Cross-Selling

This repository contains the code and documentation for the **Kaggle Playground Series S4E7 - Insurance Cross-Selling** challenge. The task involves predicting whether an existing insurance customer will purchase an additional insurance product based on various customer attributes.

- **Kaggle competition**: [Playground Series S4E7](https://www.kaggle.com/competitions/playground-series-s4e7/overview)
- **Blog post**: [Binary Classification of Insurance Cross-Selling](https://surajwate.com/blog/binary-classification-of-insurance-cross-selling/)
- **Score**: 0.87820
- **Public score**: 0.87862

## Overview

In this challenge, we are working on a **binary classification** task to predict whether customers will opt for additional insurance coverage based on various features such as age, region, driving license status, and vehicle information. The data used for this competition is **synthetic**, allowing for experimentation and exploration of machine learning techniques.

The repository contains scripts for:
- Data preprocessing and feature engineering.
- Logistic Regression, Random Forest, and XGBoost models.
- Hyperparameter tuning using `RandomizedSearchCV` (with a warning about resource usage).

## Project Structure

```
S4E7-Insurance-Cross-Selling/
├── docs/                   # Documentation and result notes
├── input/                  # Dataset and train/test folds
├── log/                    # Logs (e.g., for hyperparameter tuning)
├── notebooks/              # Jupyter notebooks for exploration and modeling
├── output/                 # Model outputs and scripts
├── src/                    # Source code for model training
└── requirements.txt        # Python dependencies
```

## Usage

### 1. Set up the environment
Create a virtual environment and install the required packages:
```bash
python -m venv env
source env/bin/activate  # or `.\env\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Prepare the data
Ensure that the necessary datasets (`train.csv`, `test.csv`, `train_folds.csv`) are placed in the `input/` folder. These files can be downloaded from the [Kaggle competition page](https://www.kaggle.com/competitions/playground-series-s4e7/data).

### 3. Run the models
You can run the different models available in the `src/` folder, such as Logistic Regression, Random Forest, and XGBoost.

For example, to run the Logistic Regression model:
```bash
python src/logistic_regression.py
```

### 4. Hyperparameter Optimization (Warning)
We attempted hyperparameter optimization for XGBoost using `xgboost_hpo.py`, but it caused a system crash due to high resource consumption. **Proceed with caution** if you run this script, and consider using a machine with sufficient resources (e.g., cloud services like Kaggle Notebooks or Google Colab).

## Model Performance

The final model, using **XGBoost**, achieved the following scores:

- **Validation AUC Score**: 0.87820
- **Public Leaderboard Score**: 0.87862

## References

- **Kaggle competition**: [Playground Series S4E7](https://www.kaggle.com/competitions/playground-series-s4e7/overview)
- **Blog post**: [Binary Classification of Insurance Cross-Selling](https://surajwate.com/blog/binary-classification-of-insurance-cross-selling/)
- **GitHub Repository**: [S4E7 Insurance Cross-Selling](https://github.com/surajwate/S4E7-Insurance-Cross-Selling)


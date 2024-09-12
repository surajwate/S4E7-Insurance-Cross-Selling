import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filename="./log/xgboost_hpo.log",
    filemode="a"
    )


def run(fold):
    # Load the data
    df = pd.read_csv("./input/train_folds.csv")

    # Split the data into training and testing sets
    train = df[df.kfold != fold].reset_index(drop=True)
    test = df[df.kfold == fold].reset_index(drop=True)

    # Split the data into X and y
    X_train = train.drop(["id", "Response", "kfold"], axis=1)
    X_test = test.drop(["id", "Response", "kfold"], axis=1)
    y_train = train.Response
    y_test = test.Response

    # Numerical and categorical columns
    num_cols = ['Age', 'Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
    cat_cols = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage']

    # One-hot encode the categorical columns
    ohe = OneHotEncoder()
    X_train_cat = ohe.fit_transform(X_train[cat_cols]).toarray()
    X_test_cat = ohe.transform(X_test[cat_cols]).toarray()

    # Standardize the numerical columns
    ss = StandardScaler()
    X_train_num = ss.fit_transform(X_train[num_cols])
    X_test_num = ss.transform(X_test[num_cols])

    # Combine the numerical and categorical columns
    X_train = np.hstack((X_train_num, X_train_cat))
    X_test = np.hstack((X_test_num, X_test_cat))

    # Define the hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
        "gamma": [0, 0.25, 0.5, 1.0],
        "subsample": [0.5, 0.75, 1.0],
        "colsample_bytree": [0.5, 0.75, 1.0]
    }

    # Initialize the model
    model = XGBClassifier(n_jobs=-1, random_state=42)

    # Perform Randomized Search
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=5,
        scoring="roc_auc",
        n_jobs=-1,
        cv=2,
        verbose=2,
        random_state=42
    )

    # Fit the model
    random_search.fit(X_train, y_train)

    # Get the best estimator
    best_model = random_search.best_estimator_

    # Predict probabilities
    preds = best_model.predict_proba(X_test)[:, 1]

    # Calculate the ROC AUC score
    auc = roc_auc_score(y_test, preds)

    logging.info(f"Fold={fold}, Best AUC={auc}, Best Params={random_search.best_params_}")
    print(f"Fold={fold}, Best AUC={auc}, Best Params={random_search.best_params_}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)

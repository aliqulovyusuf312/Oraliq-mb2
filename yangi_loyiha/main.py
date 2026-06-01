import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

from utils.config import (
    DATA_PROCESSED_PATH, TARGET, MODELS_PATH,
    REPORTS_PATH, FEATURES, RANDOM_STATE, TEST_SIZE
)
from utils.logger import get_logger

logger = get_logger('main')

def main():
    logger.info("Pipeline boshlandi")

    # 1. Data yuklash
    logger.info("Data yuklanmoqda...")
    df = pd.read_csv(DATA_PROCESSED_PATH)
    df['YEAR_MONTH'] = df['YEAR'] * 100 + df['MONTH']

    selected_features = [
        'RETAIL TRANSFERS', 'ITEM CODE', 'YEAR_MONTH',
        'SUPPLIER_encoded', 'WAREHOUSE SALES'
    ]

    X = df[selected_features]
    y = df[TARGET]

    # 2. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 3. Model o'rgatish
    logger.info("Model o'rgatilmoqda...")
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 4. Baholash
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    logger.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # 5. Model saqlash
    os.makedirs(MODELS_PATH, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_PATH, 'lightgbm.pkl'))
    logger.info("Model saqlandi!")

    # 6. Metrics saqlash
    os.makedirs(REPORTS_PATH, exist_ok=True)
    metrics = {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4),
        'model': 'LightGBM',
        'features': selected_features,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    with open(os.path.join(REPORTS_PATH, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    logger.info("Pipeline tugadi!")

if __name__ == '__main__':
    main()
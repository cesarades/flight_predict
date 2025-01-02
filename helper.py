from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import numpy as np

def dic_print(dic):
    for key, value in dic.items():
        print(f"{key}: {value:.2f}")


def evaluate(y_true, y_preds):
    r2 = r2_score(y_true, y_preds)
    mae = mean_absolute_error(y_true, y_preds)
    mape = mean_absolute_percentage_error(y_true, y_preds) * 100
    price_range = y_true.max() - y_true.min()
    avg = np.mean(y_true)
    return {"R² Score": r2, "MAE": mae, "Price Range": price_range, "MAPE": mape, "Average Price": avg}
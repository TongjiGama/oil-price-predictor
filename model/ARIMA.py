import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
def duqu(path, date_col="Date", val_col="Price"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    encs = ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"]
    df = None
    for e in encs:
        try:
            df = pd.read_csv(path, encoding=e)
            break
        except:
            pass
    if df is None:
        df = pd.read_csv(path)

    if date_col not in df.columns or val_col not in df.columns:
        raise ValueError("need Date + Price")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[date_col, val_col]).sort_values(date_col)

    s = pd.Series(df[val_col].values, index=pd.DatetimeIndex(df[date_col].values))
    if not s.index.is_unique:
        s = s.groupby(s.index).mean()
    s = s.sort_index()

    s = s.asfreq("B")
    s = s.ffill()
    s.index = pd.DatetimeIndex(s.index, freq="B")
    s.name = val_col
    return s

def qiefen(s, test_ratio=0.1):
    n = len(s)
    if n < 30:
        raise ValueError("data too short")
    nt = int(n * test_ratio)
    if nt < 1:
        nt = 1
    train = s.iloc[:-nt]
    test = s.iloc[-nt:]
    return train, test

def pinggu(true, pred):
    y = np.array(true, dtype=float)
    p = np.array(pred, dtype=float)
    mae = mean_absolute_error(y, p)
    rmse = np.sqrt(mean_squared_error(y, p))
    mape = np.mean(np.abs((y - p) / (np.abs(y) + 1e-8))) * 100.0
    return mae, rmse, mape

def yibu_gundong(train, test, order):
    hist = train.copy()
    preds = []

    fit = ARIMA(hist, order=order).fit()

    for i in range(len(test)):
        yhat = fit.forecast(steps=1)
        if hasattr(yhat, "iloc"):
            yhat = float(yhat.iloc[0])
        else:
            yhat = float(yhat[0])

        preds.append(yhat)

        idx = pd.date_range(start=test.index[i], periods=1, freq="B")
        new = pd.Series([float(test.iloc[i])], index=idx)
        try:
            fit = fit.append(new, refit=False)

        except:
            hist = pd.concat([hist, new])
            fit = ARIMA(hist, order=order).fit()

    return pd.Series(preds, index=test.index)
def main():
    import argparse

    pa = argparse.ArgumentParser()
    pa.add_argument("--csv", type=str, default="oil-price-predictor/data/RBRTEd.csv")
    pa.add_argument("--p", type=int, default=5)
    pa.add_argument("--d", type=int, default=1)
    pa.add_argument("--q", type=int, default=0)
    pa.add_argument("--test_ratio", type=float, default=0.1)
    pa.add_argument("--future", type=int, default=60)
    pa.add_argument("--freq", type=str, default="D")
    pa.add_argument("--out_csv", type=str, default="arima_out.csv")
    pa.add_argument("--out_png", type=str, default="arima.png")
    args = pa.parse_args()
    s = duqu(args.csv)
    train, test = qiefen(s, args.test_ratio)
    order = (args.p, args.d, args.q)
    pred = yibu_gundong(train, test, order)
    mae, rmse, mape = pinggu(test.values, pred.values)
    print("order =", order)
    print("MAE =", round(mae, 6), "RMSE =", round(rmse, 6), "MAPE% =", round(mape, 6))

    fit2 = ARIMA(pd.concat([train, test]), order=order).fit()
    fc2 = fit2.get_forecast(steps=args.future)
    fut = fc2.predicted_mean
    ci = fc2.conf_int()
    last = s.index.max()
    idx2 = pd.date_range(start=last, periods=args.future + 1, freq=args.freq)[1:]
    fut.index = idx2
    ci.index = idx2

    out = pd.DataFrame({"test_true": test.values, "test_pred": pred.values}, index=test.index)
    out2 = pd.DataFrame(
        {"future_pred": fut.values, "lower": ci.iloc[:, 0].values, "upper": ci.iloc[:, 1].values},
        index=idx2,
    )
    if args.out_csv:
        out.to_csv(args.out_csv, encoding="utf-8-sig")
        out2.to_csv(args.out_csv.replace(".csv", "_future.csv"), encoding="utf-8-sig")

    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train.values, label="train")
    plt.plot(test.index, test.values, label="test")
    plt.plot(pred.index, pred.values, label="pred")
    plt.plot(fut.index, fut.values, label="future")
    plt.fill_between(ci.index, ci.iloc[:, 0].values, ci.iloc[:, 1].values, alpha=0.2)
    plt.legend()
    plt.tight_layout()

    if args.out_png:
        plt.savefig(args.out_png, dpi=200)
    else:
        plt.show()
    plt.close()
if __name__ == "__main__":
    main()

from pathlib import Path

import pandas as pd
import yfinance as yf

from sv_pf.config import DATA_RAW, PERIODS


def download_sp500(path, start, end, ticker="^GSPC"):
    """Télécharge les cours journaliers du S&P 500 entre start et end, puis les enregistre en CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ajout du dernier jour
    end_incl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(ticker, start=start, end=end_incl, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"Aucune donnée pour {ticker} entre {start} et {end}.")

    # Les versions récentes de yfinance renvoient des colonnes à deux niveaux : on garde les noms de prix.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    df.to_csv(path)
    return path


def download_all_periods(periods=PERIODS, folder=DATA_RAW):
    """Télécharge chaque période de PERIODS dans un fichier séparé."""
    return [
        download_sp500(Path(folder) / f"sp500_{name}.csv", start, end)
        for name, (start, end) in periods.items()
    ]


if __name__ == "__main__":
    for p in download_all_periods():
        print(p)
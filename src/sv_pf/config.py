from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

PERIODS = {
    "1982-1989": ("1982-02-02", "1989-12-29"),
    "1995-2003": ("1995-05-16", "2003-04-24"),
    "2000-2008": ("2000-12-19", "2008-12-12"),
    "1987-2011": ("1987-03-31", "2011-01-13"),
    "2000-2020": ("2000-01-01", "2020-12-31"),
}
START_DATE, END_DATE = PERIODS["1982-1989"]
END_DATE = "2020-12-31"
N_PARTICLES = 1000
SEED = 42
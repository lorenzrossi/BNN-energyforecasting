"""Loading and preprocessing of the ENTSO-E hourly generation data for Italy."""

import calendar
from pathlib import Path

import pandas as pd

DEFAULT_YEARS = tuple(range(2016, 2022))

# ENTSO-E column name -> short name. Sources that are always "n/e" for Italy
# (lignite, oil shale, peat, marine, nuclear, offshore wind) are not loaded.
SOURCES = {
    "Biomass": "biomass",
    "Fossil Coal-derived gas": "coal_gas",
    "Fossil Gas": "gas",
    "Fossil Hard coal": "hard_coal",
    "Fossil Oil": "oil",
    "Geothermal": "geothermal",
    "Hydro Pumped Storage": "hydro_pumped",
    "Hydro Run-of-river and poundage": "hydro_river",
    "Hydro Water Reservoir": "hydro_reservoir",
    "Other": "other",
    "Solar": "solar",
    "Waste": "waste",
    "Wind Onshore": "wind",
}

HYDRO = ["hydro_pumped", "hydro_river", "hydro_reservoir"]
GAS = ["coal_gas", "gas"]


def download_from_drive(folder_id: str, output_dir="data") -> list[Path]:
    """Download the ITA<year>.csv files from a public Google Drive folder. Needs gdown."""
    import gdown

    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(url, output=str(output_dir), quiet=False, use_cookies=False)
    return sorted(Path(output_dir).glob("ITA*.csv"))


def read_year(path) -> pd.DataFrame:
    """Read one ENTSO-E export. Index is the naive local start time of each hour."""
    raw = pd.read_csv(path, na_values=["n/e", "N/A", "-"])
    raw.columns = [" ".join(c.split()) for c in raw.columns]
    keep = {f"{name} - Actual Aggregated [MW]": short for name, short in SOURCES.items()}
    missing = [c for c in keep if c not in raw.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = raw[list(keep)].rename(columns=keep).astype(float)
    # MTU looks like "01.01.2016 00:00 - 01.01.2016 01:00 (CET/CEST)"
    df.index = pd.to_datetime(raw["MTU"].str[:16], format="%d.%m.%Y %H:%M")
    df.index.name = "time"
    return df


def load_raw(data_dir="data", years=DEFAULT_YEARS) -> pd.DataFrame:
    """Concatenate the yearly files into one hourly frame with a complete index."""
    frames = []
    for year in years:
        path = Path(data_dir) / f"ITA{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. See README for how to get the data.")
        frames.append(read_year(path))
    df = pd.concat(frames).sort_index()
    # The autumn DST change produces one duplicated local hour per year; the
    # spring change leaves a gap. Keep the first duplicate and fill the gap.
    df = df[~df.index.duplicated(keep="first")]
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", name="time")
    return df.reindex(full)


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Linear interpolation in time, then forward/backward fill for the ends."""
    return df.interpolate(method="time").ffill().bfill()


def aggregate_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Sum hydro and gas sources and add the total generation."""
    out = df.copy()
    out["hydro_tot"] = out[HYDRO].sum(axis=1)
    out["gas_tot"] = out[GAS].sum(axis=1)
    out = out.drop(columns=HYDRO + GAS)
    out["total_aggregated"] = out.sum(axis=1)
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour of day, weekday and the dummies used as model inputs."""
    out = df.copy()
    idx = out.index
    out["hour"] = idx.hour
    out["weekday"] = idx.weekday
    out["saturday"] = (idx.weekday == 5).astype(int)
    out["sunday"] = (idx.weekday == 6).astype(int)
    out["weekend"] = out["saturday"] + 2 * out["sunday"]
    out["business_hour"] = ((idx.hour >= 8) & (idx.hour <= 18)).astype(int)
    return out


def load_dataset(data_dir="data", years=DEFAULT_YEARS) -> pd.DataFrame:
    """Full pipeline: read, fill gaps, aggregate, add calendar features."""
    df = load_raw(data_dir, years)
    df = fill_missing(df)
    df = aggregate_sources(df)
    return add_calendar_features(df)


def hours_in_years(years) -> int:
    """Number of hourly observations in the given calendar years."""
    return 24 * sum(366 if calendar.isleap(y) else 365 for y in years)

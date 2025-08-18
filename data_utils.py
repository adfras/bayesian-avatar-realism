# data_utils.py
import pandas as pd
import numpy as np
import re

def clean_names(df):
    df = df.copy()
    df.columns = [re.sub(r'[^0-9a-zA-Z]+','_', c).strip('_').lower() for c in df.columns]
    return df

def load_study2(path_csv):
    df = pd.read_csv(path_csv)
    df = clean_names(df)

    need = ["participant_code","avatar_type","disclosure_sentiment",
            "real_person_rating","enjoyment_rating"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Study2 missing column(s): {miss}")

    # harmonize
    df["avatar_type"] = df["avatar_type"].astype(str).str.lower()
    df["disclosure_sentiment"] = df["disclosure_sentiment"].astype(str).str.lower()

    # map avatar to three levels if present, else pass through
    # (Study-2 may have low/medium/high)
    m = {"low":"first","medium":"second","high":"third",
         "sync":"first","unreal":"third"}
    df["avatar_recoded"] = df["avatar_type"].map(m).fillna(df["avatar_type"])

    # contrast codes (3-level avatar is okay; medium -> 0.0)
    df["avatar_c"] = np.where(df["avatar_recoded"]=="first", -0.5,
                       np.where(df["avatar_recoded"]=="second", 0.0,
                       np.where(df["avatar_recoded"]=="third",  0.5, np.nan)))
    df["type_c"]   = np.where(df["disclosure_sentiment"]=="positive", 0.5, -0.5)

    # outcomes
    for col in ["real_person_rating","enjoyment_rating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.rename(columns={"real_person_rating":"realism",
                            "enjoyment_rating":"quality"})

    df["participant_id"] = df["participant_code"].astype(str)
    # drop rows missing any key field
    df = df.dropna(subset=["realism","quality","avatar_c","type_c","participant_id"])
    return df

def load_study3(path_csv):
    df = pd.read_csv(path_csv)
    df = clean_names(df)

    need = ["participant_code","avatar_type","disclosure_sentiment",
            "real_person_rating","enjoyment_rating"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Study3 missing column(s): {miss}")

    df["avatar_type"] = df["avatar_type"].astype(str).str.lower()
    df["disclosure_sentiment"] = df["disclosure_sentiment"].astype(str).str.lower()

    # Build object-dtype series to avoid string/NaN dtype promotion issues
    avatar_recoded = pd.Series(index=df.index, dtype="object")
    avatar_recoded.loc[df["avatar_type"].str.contains("sync",   na=False)] = "first"
    avatar_recoded.loc[df["avatar_type"].str.contains("unreal", na=False)] = "third"
    df["avatar_recoded"] = avatar_recoded  # may contain NaN; we drop later

    df["avatar_c"] = df["avatar_recoded"].map({"first": -0.5, "third": 0.5})
    df["type_c"]   = df["disclosure_sentiment"].map({"positive": 0.5, "negative": -0.5})

    # coerce outcomes to numeric and enforce 1..5
    for col in ["real_person_rating","enjoyment_rating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = df["real_person_rating"].between(1,5) & df["enjoyment_rating"].between(1,5)
    bad = int((~keep).sum())
    if bad:
        print(f"[load_study3] Dropping {bad} rows with invalid/missing ratings (must be 1..5).")
    df = df.loc[keep].copy()

    df["participant_id"] = df["participant_code"].astype(str)
    df = df.rename(columns={"real_person_rating":"realism",
                            "enjoyment_rating":"quality"})

    df = df.dropna(subset=["realism","quality","avatar_c","type_c","participant_id"])
    # Leave realism/quality as float; the model will convert to 0..K-1 classes.
    return df

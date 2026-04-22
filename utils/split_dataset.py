import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def _load_and_split_data(csv_path, val_size=0.1, test_size=0.1, random_state=42):
    df = pd.read_csv(csv_path)

    df = df.dropna(subset=["text_corrected"])
    df["text_corrected"] = df["text_corrected"].astype(str)

    df = _binarize_labels(df)

    tasks = ["humour", "sarcasm", "offensive", "motivational"]

    encoders = {}
    for task in tasks:
        enc = LabelEncoder()
        df[task] = enc.fit_transform(df[task].astype(str))
        encoders[task] = enc

    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["humour"]
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=train_val_df["humour"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        encoders,
    )


def _binarize_labels(df):
    # humour
    df["humour"] = df["humour"].replace(
        {
            "funny": "funny",
            "hilarious": "funny",
            "very_funny": "funny",
            "not_funny": "not_funny",
        }
    )

    # sarcasm
    df["sarcasm"] = df["sarcasm"].replace(
        {
            "general": "not_sarcastic",
            "not_sarcastic": "not_sarcastic",
            "twisted_meaning": "sarcastic",
            "very_twisted": "sarcastic",
        }
    )

    # offensive
    df["offensive"] = df["offensive"].replace(
        {
            "not_offensive": "not_offensive",
            "slight": "offensive",
            "very_offensive": "offensive",
            "hateful_offensive": "offensive",
        }
    )

    return df

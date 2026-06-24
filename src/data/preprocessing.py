import re
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

EMOJI_PATTERN = re.compile("["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F9FF"
    u"\U00002700-\U000027BF"
    "]+", flags=re.UNICODE)


def extract_company(text: str) -> str:
    match = re.search(r'@(\w+)', str(text))
    return match.group(1) if match else "unknown"


def extract_hour(timestamp: str) -> int:
    try:
        return pd.to_datetime(timestamp).hour
    except:
        return datetime.now().hour


def extract_is_weekend(timestamp: str) -> int:
    try:
        return int(pd.to_datetime(timestamp).weekday() >= 5)
    except:
        return int(datetime.now().weekday() >= 5)


def extract_raw_features(text):
    text = str(text)
    letters = [c for c in text if c.isalpha()]
    caps_ratio = sum(1 for c in letters if c.isupper()) / max(len(letters), 1)
    return {
        "caps_ratio": round(caps_ratio, 3),
        "has_emoji": int(bool(EMOJI_PATTERN.search(text))),
        "has_exclamation": int("!" in text),
        "has_question": int("?" in text),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "text_length": len(text)
    }


def clean_text(text: str) -> str:
    text = re.sub(r'@\w+', '', str(text))
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s!?]', '', text)
    return text.lower().strip()


def get_sentiment(text: str) -> str:
    score = sia.polarity_scores(str(text))["compound"]
    return "negative" if score <= -0.3 else "non-negative"


def add_time_features(df, datetime_col):
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df["hour"] = df[datetime_col].dt.hour
    df["day_of_week"] = df[datetime_col].dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    return df


def add_company_name(df, raw_data_path):
    df_original = pd.read_csv(raw_data_path)
    company_lookup = df_original[df_original["inbound"] == False][["tweet_id", "author_id"]].rename(
        columns={"author_id": "company_name"}
    )
    return df.merge(
        company_lookup,
        left_on="tweet_id_company",
        right_on="tweet_id",
        how="left"
    ).drop("tweet_id", axis=1)


def run_pipeline(merged_path, raw_data_path, output_dir):
    merged = pd.read_csv(merged_path)
    merged = add_time_features(merged, "created_at_customer")

    raw_features = merged["customer_text"].apply(extract_raw_features)
    merged = pd.concat([merged, pd.DataFrame(raw_features.tolist(), index=merged.index)], axis=1)

    merged["customer_text_clean"] = merged["customer_text"].apply(clean_text).fillna("").str.strip()
    merged["company_text_clean"] = merged["company_text"].apply(clean_text).fillna("").str.strip()
    merged["sentiment"] = merged["customer_text"].apply(get_sentiment)
    merged["response_time_capped"] = merged["response_time_mins"].clip(upper=500)
    merged["response_time_log"] = np.log1p(merged["response_time_capped"])
    merged = add_company_name(merged, raw_data_path)

    merged = merged.dropna(subset=["customer_text_clean"])
    merged = merged[merged["customer_text_clean"] != ""]

    os.makedirs(output_dir, exist_ok=True)
    merged.to_csv(f"{output_dir}/full_data.csv", index=False)

    train_df, temp_df = train_test_split(merged, test_size=0.3, random_state=42, stratify=merged["sentiment"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["sentiment"])

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    return train_df, val_df, test_df


if __name__ == "__main__":
    run_pipeline(
        merged_path="data/raw/merged_with_text.csv",
        raw_data_path="data/raw/twcs.csv",
        output_dir="data/processed"
    )
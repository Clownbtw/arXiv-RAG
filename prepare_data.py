"""Очистка arxiv_data.csv и расчёт эмбеддингов -> data/arxiv_clean.parquet, data/embeddings.npy

Запуск:  python prepare_data.py --raw arxiv_data.csv
Колонки исходного CSV: titles, summaries, terms.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def clean_text(text: str) -> str:
    text = re.sub(r"\$.*?\$", "", str(text))                  # формулы $...$
    text = re.sub(r"\\[a-zA-Z]+\{(.*?)\}", r"\1", text)       # \textbf{word} -> word
    return re.sub(r"\s+", " ", text).strip()


def clean(raw_path: str, categories: list[str], min_len: int = 50) -> pd.DataFrame:
    df = pd.read_csv(raw_path).rename(columns={"summaries": "abstract", "terms": "categories"})
    if categories:
        pattern = "|".join(re.escape(c) for c in categories)
        df = df[df["categories"].str.contains(pattern, na=False)]
    df = df.drop_duplicates(subset=["titles"])
    df = df.assign(abstract_clean=df["abstract"].map(clean_text),
                   title_clean=df["titles"].map(clean_text))
    df = df[df["abstract_clean"].str.len() > min_len]
    return df[["title_clean", "abstract_clean", "categories"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="arxiv_data.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--categories", nargs="+", default=["cs.CL", "cs.LG"])
    ap.add_argument("--limit", type=int, default=0, help="для быстрой проверки: обработать N строк")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    df = clean(args.raw, args.categories)
    if args.limit:
        df = df.head(args.limit)
    df.to_parquet(out / "arxiv_clean.parquet", index=False)
    print("Статей после очистки:", len(df))

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    texts = ("passage: " + df["title_clean"] + ". " + df["abstract_clean"]).tolist()
    emb = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    np.save(out / "embeddings.npy", emb.astype(np.float16))   # float16: вдвое меньше, качество то же
    print("Векторы:", emb.shape)


if __name__ == "__main__":
    main()

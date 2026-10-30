"""Оценка поиска на eval/eval_set.json: Hit@k и MRR для 'только E5' и 'E5 + reranking'.

eval_set.json: список {"question", "reference", "gold_title", "gold_ids"}.
Запуск:  python evaluate.py
"""
import json
import sys

import numpy as np
import pandas as pd

from rag import RAG

PATH = sys.argv[1] if len(sys.argv) > 1 else "eval/eval_set.json"


def rank_of(gold: set, ranked: list[dict]):
    for pos, c in enumerate(ranked, 1):
        if c["id"] in gold:
            return pos
    return None


def metrics(ranks: pd.Series) -> dict:
    out = {f"Hit@{k}": round(float((ranks.notna() & (ranks <= k)).mean()), 2) for k in (1, 3, 5, 10)}
    out["MRR"] = round(float(np.where(ranks.notna(), 1 / ranks.fillna(1), 0).mean()), 3)
    out["Recall@20"] = round(float(ranks.notna().mean()), 2)
    return out


def main():
    items = json.load(open(PATH, encoding="utf-8"))
    rag = RAG()
    rows = []
    for it in items:
        q_en = rag.translate(it["question"])
        base = rag.retrieve(q_en, 20)
        rr = rag.rerank(q_en, list(map(dict, base)), k=20)
        gold = set(it["gold_ids"])
        rows.append({"question": it["question"], "gold": it["gold_title"],
                     "rank_vec": rank_of(gold, base), "rank_rr": rank_of(gold, rr)})
    res = pd.DataFrame(rows)
    print(f"Вопросов: {len(res)}")
    print("Только E5:         ", metrics(res["rank_vec"]))
    print("E5 + reranking:    ", metrics(res["rank_rr"]))
    n3 = int((res["rank_rr"].notna() & (res["rank_rr"] <= 3)).sum())
    print(f"С reranking нужная статья в топ-3: {n3} из {len(res)}")
    res.to_csv("eval/results.csv", index=False)


if __name__ == "__main__":
    main()

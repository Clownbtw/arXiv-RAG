"""Ядро RAG: E5 + Chroma -> reranking cross-encoder'ом -> ответ LLM с источниками."""
import os
import re
from pathlib import Path
from urllib.parse import quote_plus

import chromadb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_REPO = os.getenv("DATA_REPO", "")          # HF Dataset с файлами, если локально их нет
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
MIN_RERANK = float(os.getenv("MIN_RERANK", "-1000"))   # -1000 = отсечение выключено

PROMPT = """Ты помощник по научным статьям arXiv. Отвечай на вопрос ТОЛЬКО по статьям ниже.
Если среди них нет ответа, ответь одной фразой: "В базе нет подходящих статей". Ничего не выдумывай.
Некоторые статьи могут быть не по теме, игнорируй их.
Ссылайся на статьи номерами в квадратных скобках, например [1] или [2][3]. Списка источников в конце не пиши.

Статьи:
{context}

Вопрос: {question}
Отвечай строго на русском языке. Ответ:"""


def _data_file(name: str) -> str:
    local = DATA_DIR / name
    if local.exists():
        return str(local)
    if not DATA_REPO:
        raise FileNotFoundError(
            f"Нет файла {local}. Положите данные в {DATA_DIR}/ или задайте DATA_REPO (HF Dataset).")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(DATA_REPO, name, repo_type="dataset")


class RAG:
    def __init__(self):
        self.df = pd.read_parquet(_data_file("arxiv_clean.parquet")).reset_index(drop=True)
        emb = np.load(_data_file("embeddings.npy")).astype("float32")
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        if len(emb) != len(self.df):
            raise ValueError(f"Векторы ({len(emb)}) и таблица ({len(self.df)}) не совпадают по длине")

        self.embedder = SentenceTransformer("intfloat/multilingual-e5-base")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        self.llm = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY or "none")
        self.col = self._build_index(emb)

    def _build_index(self, emb, batch=500):
        col = chromadb.EphemeralClient().get_or_create_collection(
            "arxiv", metadata={"hnsw:space": "cosine"})
        for i in range(0, len(self.df), batch):
            part = self.df.iloc[i:i + batch]
            col.add(
                ids=[str(j) for j in part.index],
                embeddings=emb[i:i + batch].tolist(),
                documents=part["abstract_clean"].tolist(),
                metadatas=[{"title": t, "categories": c}
                           for t, c in zip(part["title_clean"], part["categories"])],
            )
        return col

    def translate(self, text: str) -> str:
        """Русский вопрос -> английский (абстракты английские). Английский не трогаем."""
        if not re.search(r"[А-Яа-яЁё]", text):
            return text
        r = self.llm.chat.completions.create(
            model=LLM_MODEL, temperature=0,
            messages=[{"role": "system",
                       "content": "Translate the user's text to English. Output only the translation."},
                      {"role": "user", "content": text}])
        return r.choices[0].message.content.strip()

    def retrieve(self, query_en: str, k: int = 20) -> list[dict]:
        q = self.embedder.encode("query: " + query_en.strip(), normalize_embeddings=True)
        res = self.col.query(query_embeddings=[q.tolist()], n_results=k,
                             include=["documents", "metadatas", "distances"])
        return [{"id": i, "title": m["title"], "categories": m["categories"], "text": d,
                 "score": round(1 - dist, 3)}
                for i, m, d, dist in zip(res["ids"][0], res["metadatas"][0],
                                         res["documents"][0], res["distances"][0])]

    def rerank(self, query_en: str, cands: list[dict], k: int = 5) -> list[dict]:
        scores = self.reranker.predict(
            [(query_en, c["title"] + ". " + c["text"]) for c in cands], batch_size=32)
        for c, s in zip(cands, scores):
            c["rerank"] = float(s)
        return sorted(cands, key=lambda c: -c["rerank"])[:k]

    def answer(self, question: str, k: int = 5) -> dict:
        q_en = self.translate(question)
        top = self.rerank(q_en, self.retrieve(q_en, 20), k)
        if not top or top[0]["rerank"] < MIN_RERANK:
            return {"answer": "В базе нет подходящих статей по этому вопросу.",
                    "sources": [], "query_en": q_en}

        context = "\n\n".join(f"[{i}] {r['title']}\n{r['text']}" for i, r in enumerate(top, 1))
        r = self.llm.chat.completions.create(
            model=LLM_MODEL, temperature=0.2,
            messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}])
        text = r.choices[0].message.content.strip()
        if "нет подходящих статей" in text.lower():
            return {"answer": text, "sources": [], "query_en": q_en}

        # источники собираем кодом, а не LLM: названия всегда точные
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(top)}
        sources = [{"n": i, **r,
                    "url": "https://arxiv.org/search/?searchtype=title&query=" + quote_plus(r["title"])}
                   for i, r in enumerate(top, 1) if not cited or i in cited]
        return {"answer": text, "sources": sources, "query_en": q_en}

"""Captain (runcaptain.com) managed file-search API — text pipeline.

Ingestion: each ViDoRe v3 corpus page's provided NV-Ingest markdown is indexed
as one document named `{corpus_id}.md`, prefixed with a one-line
`# {doc_id} - page {n}` header so chunks carry document identity. Captain then
does its own chunking, LLM enrichment, and embedding server-side.

Retrieval: hybrid BM25 + dense retrieval with voyage-rerank-2.5 reranking
(candidate pool 200), one API call per query. Pages are ranked by their best
chunk score (MaxP); returned document filenames map back to corpus ids.

Requires the CAPTAIN_API_KEY environment variable (closed API; free keys at
https://www.captain.dev). See description.json for run/timing details.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from vidore_benchmark.pipeline_evaluation.base_pipeline import BasePipeline

API = os.environ.get("CAPTAIN_API", "https://api.runcaptain.com")
KEY = os.environ["CAPTAIN_API_KEY"]
EPS = 1e-6


class CaptainTextPipeline(BasePipeline):
    def __init__(self, top_k: int = 100, workers: int = 10, **kwargs):
        self.top_k = top_k
        self.workers = workers
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {KEY}",
                               "Content-Type": "application/json"})
        self.collection = None

    # ------------------------------------------------------------------ index
    def index(self, corpus_ids, corpus_images, corpus_texts, dataset_name=None):
        ds = (dataset_name or "vidore_v3").rsplit("vidore_v3_", 1)[-1]
        self.collection = f"vidore-{ds}-text"
        r = self.s.put(f"{API}/v2/collections/{self.collection}", json={}, timeout=60)
        if r.status_code == 409 or (r.status_code == 200 and self._doc_count() >= len(corpus_ids)):
            return  # already indexed (pre-built corpus reused across runs)
        if r.status_code not in (200, 201):
            r.raise_for_status()

        def _put(pair):
            cid, text = pair
            body = {"content": text or " ", "file_name": f"{cid}.md"}
            for attempt in range(3):
                resp = self.s.post(f"{API}/v2/collections/{self.collection}/index/text",
                                   json=body, timeout=120)
                if resp.status_code in (200, 201, 202):
                    return
                time.sleep(2 * (attempt + 1))
            resp.raise_for_status()

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            list(ex.map(_put, zip(corpus_ids, corpus_texts)))
        # poll until documents are queryable
        deadline = time.time() + 3600
        while time.time() < deadline and self._doc_count() < len(corpus_ids):
            time.sleep(15)

    def _doc_count(self):
        try:
            r = self.s.get(f"{API}/v2/collections/{self.collection}/documents?limit=1",
                           timeout=60)
            return (r.json().get("pagination") or {}).get("total") or \
                len(r.json().get("documents") or [])
        except Exception:
            return 0

    # ----------------------------------------------------------------- search
    def _one(self, qid, query):
        body = {"query": query, "limit": 100,
                "rerank": {"enabled": True, "candidate_limit": 200},
                "include": {"document": True, "metadata": False}}
        if self.collection.endswith("finance_fr-text"):
            # a-priori rule for the cross-lingual set (English queries, French
            # corpus): keyword retrieval contributes nothing, use dense only.
            body["semantic_ratio"] = 1.0
        for attempt in range(3):
            try:
                r = self.s.post(f"{API}/v3/collections/{self.collection}/query",
                                json=body, timeout=120)
                r.raise_for_status()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))
        d = r.json()
        assert not (d.get("rerank") or {}).get("fallback"), "rerank fallback — rerun"
        agg = {}
        for res in d.get("results", []):
            fname = ((res.get("document") or {}).get("filename") or "").rsplit("/", 1)[-1]
            stem = fname[:-3] if fname.endswith(".md") else fname
            if not stem.isdigit():
                continue
            score = float(res.get("score") or 0.0)
            agg[stem] = max(agg.get(stem, 0.0), score)   # MaxP page aggregation
        ranked = sorted(agg.items(), key=lambda kv: -kv[1])[: self.top_k]
        return qid, {cid: s - i * EPS for i, (cid, s) in enumerate(ranked)}

    def search(self, query_ids, queries):
        out = {}
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for qid, run in ex.map(lambda a: self._one(*a), zip(query_ids, queries)):
                out[str(qid)] = run
        return out, {"total_retrieval_time_seconds": round(time.time() - t0, 2)}

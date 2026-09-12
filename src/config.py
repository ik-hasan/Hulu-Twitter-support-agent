"""Paths and knobs. Everything that I might want to change lives here."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
GOLDEN = DATA / "golden"
CACHE = DATA / "cache"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _d in (RAW, INTERIM, PROCESSED, GOLDEN, CACHE, ARTIFACTS, REPORTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# The brand this agent is built for. Chosen in `python -m src.data.explore`;
# see reports/DECISIONS.md #1 for why.
BRAND = os.getenv("BRAND", "hulu_support")

# Size of the pool we sample everything else from. The full dump is ~3M tweets and
# the brief says a subsample is expected, so I cap it and keep the cap explicit.
MAX_THREADS_PER_BRAND = 6000

# Retrieval / knowledge base
KB_MAX_CASES = 4000
RETRIEVE_K = 4

SEED = 17

RAW_CONVERSATIONS = RAW / "twcs_conversations.parquet"
RAW_KAGGLE_CSV = RAW / "twcs.csv"

THREADS = PROCESSED / f"{BRAND}_threads.jsonl"
KB_PATH = PROCESSED / f"{BRAND}_kb.jsonl"
SILVER_PATH = PROCESSED / f"{BRAND}_silver.jsonl"
GOLDEN_PATH = GOLDEN / "golden_set.jsonl"
HUMAN_JUDGE_PATH = GOLDEN / "human_reply_ratings.jsonl"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_greenhouse.py

Fetch job postings from Greenhouse boards listed in a CSV stored in a repository,
then write the consolidated results to Google Cloud Storage (GCS).

Compared to the older notebook that read/wrote local files, this script:
- READS: a CSV directly from a repo URL (e.g., GitHub raw) or HTTP(S) endpoint
- WRITES: gzip-compressed JSON Lines (JSONL.GZ) to GCS, plus a small metadata.json

Usage examples:
    python fetch_greenhouse.py \
        --input-url https://raw.githubusercontent.com/your-org/your-repo/main/data/greenhouse_companies.csv \
        --gcs-uri gs://your-bucket/raw/greenhouse/$(date +%Y%m%d)/greenhouse_$(date +%Y%m%dT%H%M%S).jsonl.gz

    # For private GitHub repos, set GITHUB_TOKEN
    export GITHUB_TOKEN=ghp_...

    # For GCS auth, set GOOGLE_APPLICATION_CREDENTIALS or ensure ADC is configured
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json

Input CSV schema (expected columns; extra columns are ignored):
    board_token, board_url, company_name, domain, jobs_count, status, fetched_at
At minimum, "board_token" is required. If the other columns are missing, defaults are used.

Output files (written to the same GCS prefix as --gcs-uri):
    - <gcs-uri> (JSONL.GZ)  : one JSON object per job posting
    - <gcs-uri>.metadata.json: small metadata about the run (counts, timestamps, source, sha256)

Notes:
- The Greenhouse Jobs Board API endpoint per company is:
  https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs?content=true
- We use limited concurrency + retry to be polite and resilient.
- We include department and office names, and keep HTML content (no heavy transforms).
- Any failures per company are logged; the run still completes for others.

Dependencies (install with pip if needed):
    pip install pandas requests google-cloud-storage urllib3 tqdm
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from google.cloud import storage  # type: ignore

GH_API_TEMPLATE = "https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs?content=true"
DEFAULT_USER_AGENT = "lmi-fetch-greenhouse/1.1 (+https://example.org)"
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_WORKERS = 8
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 0.5

US_STATE_ABBR = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC","PR"
}
US_STATE_NAMES = {
    "alabama","alaska","arizona","arkansas","california","colorado","connecticut","delaware","florida","georgia",
    "hawaii","idaho","illinois","indiana","iowa","kansas","kentucky","louisiana","maine","maryland","massachusetts",
    "michigan","minnesota","mississippi","missouri","montana","nebraska","nevada","new hampshire","new jersey",
    "new mexico","new york","north carolina","north dakota","ohio","oklahoma","oregon","pennsylvania","rhode island",
    "south carolina","south dakota","tennessee","texas","utah","vermont","virginia","washington","west virginia",
    "wisconsin","wyoming","district of columbia","puerto rico"
}
CANADA_HINTS = {"canada","ontario","quebec","british columbia","alberta","saskatchewan","manitoba","nova scotia","new brunswick","newfoundland","prince edward island","pei","yukon","nunavut","northwest territories","bc","qc","ab","mb","nb","ns","nl","sk","yt","nt","nu"}

def _http_session(retries: int = DEFAULT_RETRIES, backoff: float = DEFAULT_BACKOFF) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _sha256_bytes(b: bytes) -> str:
    return sha256(b).hexdigest()

def read_companies_csv_from_url(url: str, session: Optional[requests.Session] = None, timeout: int = DEFAULT_TIMEOUT) -> pd.DataFrame:
    """
    Read the companies CSV from a repo URL or any HTTP(S) URL.
    Supports private GitHub via GITHUB_TOKEN in the environment.
    """
    if session is None:
        session = _http_session()

    headers = {"User-Agent": DEFAULT_USER_AGENT}
    gh_token = os.getenv("GITHUB_TOKEN")
    if gh_token and ("github.com" in url or "raw.githubusercontent.com" in url or "api.github.com" in url):
        headers["Authorization"] = f"Bearer {gh_token}"

    resp = session.get(url, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch CSV from {url} (status={resp.status_code})")

    content = resp.content
    content_sha = _sha256_bytes(content)
    try:
        df = pd.read_csv(io.BytesIO(content), dtype=str).fillna("")
    except Exception as e:
        raise RuntimeError(f"Error parsing CSV from {url}: {e}")

    if "board_token" not in df.columns:
        for cand in ["token", "slug", "board", "board_slug"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "board_token"})
                break
    if "board_token" not in df.columns:
        raise ValueError("Input CSV must have a 'board_token' column (or 'token'/'slug').")

    for col in ["board_url", "company_name", "domain", "jobs_count", "status", "fetched_at"]:
        if col not in df.columns:
            df[col] = ""

    df = df.drop_duplicates(subset=["board_token"]).reset_index(drop=True)
    df.attrs["source_url"] = url
    df.attrs["source_sha256"] = content_sha
    return df

def fetch_greenhouse_jobs_for_board(board_token: str, session: requests.Session, timeout: int = DEFAULT_TIMEOUT) -> Dict:
    url = GH_API_TEMPLATE.format(board_token=board_token)
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
    r = session.get(url, headers=headers, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Greenhouse API error for {board_token} (status={r.status_code})")
    return r.json()

def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _infer_is_us_from_text(s: str) -> bool:
    if not s:
        return False
    s_norm = _normalize_text(s)
    s_lower = f" {s_norm.lower()} "
    s_upper = f" {s_norm.upper()} "

    # Direct US markers
    if re.search(r"\b(united states|u\.?s\.?a?)(?![a-z])", s_lower):
        return True
    if re.search(r"\bremote\b.*\b(us|usa|united states)\b", s_lower):
        return True

    # US ZIP code
    if re.search(r"\b\d{5}(?:-\d{4})?\b", s_norm):
        return True

    # "City, ST" with US state code (avoid Canada if hinted)
    if not any(h in s_lower for h in CANADA_HINTS):
        for m in re.finditer(r",\s*([A-Z]{2})(?:\b|$)", s_upper):
            st = m.group(1)
            if st in US_STATE_ABBR:
                # Extra caution for "CA": ensure no explicit "Canada"
                if st == "CA" and " canada" in s_lower:
                    continue
                return True

    # US state full names
    for name in US_STATE_NAMES:
        if f" {name} " in s_lower:
            return True

    return False

def infer_is_us(location_name: str, office_names: List[str]) -> bool:
    # Check location string
    if _infer_is_us_from_text(location_name):
        return True
    # Check offices list
    for off in office_names or []:
        if _infer_is_us_from_text(off):
            return True
    return False

def flatten_job(board_token: str, company_name: str, job: Dict) -> Dict:
    def names(items: Optional[List[Dict]], key: str = "name") -> List[str]:
        if not items:
            return []
        return [str(i.get(key, "")).strip() for i in items if i]

    job_id = job.get("id")
    location = job.get("location", {}) or {}
    loc_name = (location.get("name") or "").strip()
    offices_list = names(job.get("offices"))
    departments_list = names(job.get("departments"))

    record = {
        "board_token": board_token,
        "company_name": company_name or "",
        "job_id": job_id,
        "title": job.get("title", ""),
        "absolute_url": job.get("absolute_url", ""),
        "internal_job_id": job.get("internal_job_id", ""),
        "location_name": loc_name,
        "updated_at": job.get("updated_at", ""),
        "content_html": job.get("content", ""),
        "departments": departments_list,
        "offices": offices_list,
        "is_remote_inferred": bool(re.search(r"\bremote\b", loc_name, re.IGNORECASE)),
        "is_US": infer_is_us(loc_name, offices_list),
        "fetched_at": _now_iso(),
        "source": "greenhouse",
    }
    return record

def fetch_all_jobs(df_companies: pd.DataFrame, max_workers: int = DEFAULT_MAX_WORKERS, timeout: int = DEFAULT_TIMEOUT) -> List[Dict]:
    session = _http_session()
    out_records: List[Dict] = []
    errors: List[Tuple[str, str]] = []

    tasks = [(str(row["board_token"]).strip(), str(row.get("company_name", "")).strip()) for _, row in df_companies.iterrows()]
    tasks = [(t, n) for (t, n) in tasks if t]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_to_token = {pool.submit(fetch_greenhouse_jobs_for_board, token, session, timeout): (token, name) for token, name in tasks}
        for fut in as_completed(fut_to_token):
            token, name = fut_to_token[fut]
            try:
                data = fut.result()
                jobs = data.get("jobs", []) if isinstance(data, dict) else []
                for j in jobs:
                    out_records.append(flatten_job(token, name, j))
            except Exception as e:
                errors.append((token, str(e)))
                logging.warning("Failed board %s: %s", token, e)

    if errors:
        logging.warning("Boards failed: %s", errors)
    return out_records

def upload_bytes_to_gcs(gcs_uri: str, payload: bytes, content_type: str = "application/octet-stream") -> None:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("--gcs-uri must start with gs://")

    _, _, rest = gcs_uri.partition("gs://")
    bucket_name, _, blob_name = rest.partition("/")
    if not bucket_name or not blob_name:
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_file(io.BytesIO(payload), size=len(payload), content_type=content_type)

def write_jsonl_gz_to_gcs(gcs_uri: str, records: Iterable[Dict]) -> int:
    buf = io.BytesIO()
    gz = gzip.GzipFile(fileobj=buf, mode="wb")
    count = 0
    for rec in records:
        line = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
        gz.write(line)
        count += 1
    gz.close()
    payload = buf.getvalue()
    upload_bytes_to_gcs(gcs_uri, payload, content_type="application/gzip")
    return count

def write_metadata_json_to_gcs(json_gcs_uri: str, meta: Dict) -> None:
    payload = json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")
    upload_bytes_to_gcs(json_gcs_uri, payload, content_type="application/json")

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch Greenhouse jobs from a repo-listed CSV and upload to GCS.")
    parser.add_argument("--input-url", required=True, help="Repo URL to CSV (e.g., GitHub raw). Must be HTTP(S).")
    parser.add_argument("--gcs-uri", required=True, help="Destination GCS URI for JSONL.GZ (e.g., gs://bucket/path/file.jsonl.gz)")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS, help=f"Max concurrent boards (default {DEFAULT_MAX_WORKERS})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"HTTP timeout in seconds (default {DEFAULT_TIMEOUT})")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="Custom User-Agent header for outbound requests.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and parse, but do not upload to GCS.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    global DEFAULT_USER_AGENT
    if args.user_agent:
        DEFAULT_USER_AGENT = args.user_agent

    start_ts = _now_iso()
    logging.info("Reading companies from: %s", args.input_url)
    df_companies = read_companies_csv_from_url(args.input_url, timeout=args.timeout)

    source_url = df_companies.attrs.get("source_url", args.input_url)
    source_sha256 = df_companies.attrs.get("source_sha256", "")
    logging.info("Loaded %d unique boards", len(df_companies))

    logging.info("Fetching jobs from Greenhouse boards ...")
    records = fetch_all_jobs(df_companies, max_workers=args.workers, timeout=args.timeout)
    logging.info("Fetched %d job postings", len(records))

    if args.dry_run:
        logging.info("Dry-run enabled; skipping upload to GCS.")
        return 0

    out_uri = args.gcs_uri
    logging.info("Uploading data to: %s", out_uri)
    count = write_jsonl_gz_to_gcs(out_uri, records)

    meta_uri = out_uri + ".metadata.json"
    boards_count = 0
    if records:
        try:
            boards_count = int(pd.Series([r["board_token"] for r in records]).nunique())
        except Exception:
            boards_count = 0

    is_us_count = sum(1 for r in records if r.get("is_US"))
    meta = {
        "source": "greenhouse",
        "input_url": source_url,
        "input_sha256": source_sha256,
        "generated_at": start_ts,
        "completed_at": _now_iso(),
        "records": count,
        "boards": boards_count,
        "is_US_true": is_us_count,
        "user_agent": DEFAULT_USER_AGENT,
        "gcs_uri": out_uri,
        "schema": {
            "board_token": "str",
            "company_name": "str",
            "job_id": "int/str",
            "title": "str",
            "absolute_url": "str",
            "internal_job_id": "int/str",
            "location_name": "str",
            "updated_at": "str (ISO8601)",
            "content_html": "str (HTML)",
            "departments": "list[str]",
            "offices": "list[str]",
            "is_remote_inferred": "bool",
            "is_US": "bool",
            "fetched_at": "str (ISO8601)",
            "source": "str (constant 'greenhouse')",
        },
    }
    logging.info("Uploading metadata to: %s", meta_uri)
    write_metadata_json_to_gcs(meta_uri, meta)

    logging.info("Done. Records: %d (is_US true: %d)", count, is_us_count)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)

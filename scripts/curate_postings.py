#!/usr/bin/env python
import argparse
import os
import re
import sys
from pathlib import Path
from typing import Tuple, Optional
import html as html_lib
import pandas as pd
from bs4 import BeautifulSoup


def file_exists(path: str) -> bool:
    """Check if a file exists, supporting local paths and gs://."""
    if path.startswith("gs://"):
        try:
            import gcsfs
        except ImportError as exc:
            raise RuntimeError(
                "Path uses gs:// but gcsfs is not installed. "
                "Install with `pip install gcsfs`."
            ) from exc
        fs = gcsfs.GCSFileSystem()
        return fs.exists(path)
    else:
        return os.path.exists(path)


def read_active_if_exists(active_path: str) -> pd.DataFrame:
    if not file_exists(active_path):
        return pd.DataFrame()
    return pd.read_parquet(active_path)


def read_raw_snapshot(raw_path: str) -> pd.DataFrame:
    """
    Read raw snapshot from parquet, jsonl, jsonl.gz, or json.gz.
    """
    if raw_path.endswith(".parquet"):
        return pd.read_parquet(raw_path)

    # Handle json / jsonl
    if (
        raw_path.endswith(".jsonl")
        or raw_path.endswith(".json")
        or raw_path.endswith(".jsonl.gz")
        or raw_path.endswith(".json.gz")
    ):
        return pd.read_json(raw_path, lines=True, compression="infer")

    raise ValueError(f"Unsupported raw format for {raw_path}")


def html_to_text(html: Optional[str]) -> str:
    """Parse (possibly escaped) HTML to a normalized plain-text string."""
    if html is None or pd.isna(html):
        return ""
    unescaped = html_lib.unescape(str(html))
    soup = BeautifulSoup(unescaped, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def curate_for_day(
    raw_df: pd.DataFrame,
    active_prev: pd.DataFrame,
    id_col: str,
    run_date: str,
    html_col: Optional[str] = "content",
    text_col: str = "content_text",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process one day:
    - Update active set (all columns from raw; plus first_seen_date, last_seen_date).
    - Produce closed postings for this day:
        * includes all non-HTML columns from active_prev
        * adds parsed text column (text_col)
        * does NOT include the original HTML column.
    """
    if id_col not in raw_df.columns:
        raise KeyError(f"ID column '{id_col}' not found in raw snapshot")

    # Drop duplicate postings within the same day
    raw_today = raw_df.drop_duplicates(subset=[id_col], keep="last").copy()

    # Ensure tracking columns exist on previous active
    if not active_prev.empty:
        if "first_seen_date" not in active_prev.columns:
            raise KeyError("Missing 'first_seen_date' in active postings")
        if "last_seen_date" not in active_prev.columns:
            raise KeyError("Missing 'last_seen_date' in active postings")

    ids_today = set(raw_today[id_col].astype(str))
    ids_prev = set(active_prev[id_col].astype(str)) if not active_prev.empty else set()

    # Closed postings: in previous active but not in today's raw
    closed_ids = ids_prev - ids_today
    if closed_ids:
        closed_mask = active_prev[id_col].astype(str).isin(closed_ids)
        closed_df = active_prev.loc[closed_mask].copy()
        # closed_date = last day we saw it (stored in last_seen_date)
        closed_df["closed_date"] = closed_df["last_seen_date"]

        # Parse HTML to text only for closed postings, then drop HTML column
        if html_col is not None and html_col in closed_df.columns:
            closed_df[text_col] = closed_df[html_col].map(html_to_text)
            closed_df = closed_df.drop(columns=[html_col])
    else:
        closed_df = pd.DataFrame()

    # Build new active set from today's raw, carrying first_seen_date where available
    if active_prev.empty:
        active_new = raw_today.copy()
        active_new["first_seen_date"] = run_date
        active_new["last_seen_date"] = run_date
    else:
        # Only need id + first_seen_date from previous active
        prev_meta = active_prev[[id_col, "first_seen_date"]].drop_duplicates(subset=[id_col])

        active_new = raw_today.merge(
            prev_meta,
            on=id_col,
            how="left",
            suffixes=("", "_prev"),
        )

        # If we have a previous first_seen_date, keep it; otherwise set to today
        active_new["first_seen_date"] = active_new["first_seen_date"].fillna(run_date)
        # Last seen is always today for currently-observed postings
        active_new["last_seen_date"] = run_date

    return active_new, closed_df


def main():
    parser = argparse.ArgumentParser(description="Curate Greenhouse postings for one day.")
    parser.add_argument(
        "--run-date",
        required=True,
        help="Run date in YYYY-MM-DD (used for first/last seen dates and closed partition).",
    )
    parser.add_argument(
        "--raw-path",
        required=True,
        help="Path to today's raw snapshot (parquet, jsonl, jsonl.gz, json.gz).",
    )
    parser.add_argument(
        "--active-path",
        required=True,
        help="Path to active postings parquet (will be created/overwritten).",
    )
    parser.add_argument(
        "--closed-dir",
        required=True,
        help=(
            "Root directory for closed postings, e.g. 'closed'. "
            "Closed postings will be written to "
            "<closed-dir>/<run-date>/closed_postings.parquet"
        ),
    )
    parser.add_argument(
        "--id-column",
        required=True,
        help="Column name in raw snapshot that uniquely identifies a posting (e.g. 'job_id').",
    )
    parser.add_argument(
        "--html-column",
        default="content_html",
        help="Name of the HTML content column in the snapshots (default: 'content'). "
             "Set to empty string to disable HTML parsing.",
    )
    parser.add_argument(
        "--text-column",
        default="content",
        help="Name of the parsed text column added to closed postings (default: 'content_text').",
    )

    args = parser.parse_args()

    run_date = args.run_date
    raw_path = args.raw_path
    active_path = args.active_path
    closed_dir = args.closed_dir.rstrip("/")
    id_col = args.id_column
    html_col = args.html_column or None
    text_col = args.text_column

    raw_df = read_raw_snapshot(raw_path)

    raw_df["posting_id"] = raw_df["board_token"].astype(str) + ":" + raw_df["job_id"].astype(str)

    active_prev = read_active_if_exists(active_path)

    raw_today = raw_df.drop_duplicates(subset=[id_col], keep="last")
    num_today = len(raw_today)

    if active_prev.empty:
        prev_ids = set()
    else:
        prev_ids = set(active_prev[id_col].astype(str))

    ids_today = set(raw_today[id_col].astype(str))
    num_new = len(ids_today - prev_ids)


    if len(raw_df) <= 1000:
        print(f"[curator] Suspiciously small snapshot ({len(raw_df)} rows). "
            "Skipping curation for this day.")
        sys.exit(0)

    # Run curator
    active_new, closed_today = curate_for_day(
        raw_df=raw_df,
        active_prev=active_prev,
        id_col=id_col,
        run_date=run_date,
        html_col=html_col,
        text_col=text_col,
    )
    
    num_closed = len(closed_today)

    print(
        f"[curate_postings] {run_date}: {num_today} unique postings in snapshot,"
        f" {num_new} new, {num_closed} closed."
    )
    print(
        f"[curate_postings] {run_date}: {len(active_new)} active postings after update."
    )


    # Write active postings
    active_new.to_parquet(active_path, index=False)

    # Write closed postings for this day, if any
    if not closed_today.empty:
        year, month, day = run_date.split("-")
        closed_path = f"{closed_dir}/{year}/{month}/{day}/closed_postings.parquet"
        if not closed_path.startswith("gs://"):
            Path(closed_path).parent.mkdir(parents=True, exist_ok=True)
        closed_today.to_parquet(closed_path, index=False)



if __name__ == "__main__":
    main()

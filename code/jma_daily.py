"""
最近の日別データ（気象庁「過去の気象データ」）を月単位で取得・キャッシュし、preprocess.py と同じ規則で数値化する。

- 検証（check_realtime_features.py）と答え合わせ（score_predictions.py）で使う。
- 取得結果は data/raw/recent/{hirosaki|aomori}_YYYYMM.csv に保存し、必要な日がそろっていれば再取得しない。
- 当月分は月の途中までしか載っていないため、必要な日が欠けているときだけ取り直す。
"""
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from download_jma_data import STATIONS, SLEEP_SEC, fetch_month
from preprocess import AOMORI_COLS, HIROSAKI_COLS, parse_daily

CACHE = Path(__file__).resolve().parent.parent / "data" / "raw" / "recent"
COLS = {"hirosaki": HIROSAKI_COLS, "aomori": AOMORI_COLS}


def _months(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def load_recent(name: str, start: date, end: date, allow_fetch: bool = True) -> pd.DataFrame:
    """start〜end を含む月の日別データを返す（index=日付）。取れない月は飛ばす。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    info = STATIONS[name]
    frames = []
    for y, m in _months(start, end):
        path = CACHE / f"{name}_{y}{m:02d}.csv"
        raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str) if path.exists() else None
        # この月で必要な最終日（昨日まで。今日の日別値はまだ確定していない）
        need_last = min(end, date.today() - pd.Timedelta(days=1).to_pytimedelta())
        have_last = None
        if raw is not None and len(raw):
            have_last = date(y, m, int(raw.iloc[-1, 2]))
        month_need = (y, m) <= (need_last.year, need_last.month)
        stale = raw is None or (month_need and have_last is not None
                                and (have_last.year, have_last.month) == (need_last.year, need_last.month)
                                and have_last < need_last)
        if stale and allow_fetch and month_need:
            try:
                df = fetch_month(info["page"], info["prec_no"], info["block_no"], y, m)
                time.sleep(SLEEP_SEC)
                if df is not None:
                    df.to_csv(path, index=False, encoding="utf-8-sig")
                    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
            except requests.RequestException as e:
                print(f"[{name}] {y}-{m:02d} の日別データ取得に失敗: {e}")
        if raw is not None and len(raw):
            frames.append(parse_daily(raw, COLS[name]))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    return out[(out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))]

"""
気象庁「過去の気象データ・ダウンロード」ページから、日別データを取得するスクリプト。

対象地点:
  - 弘前 (prec_no=31, block_no=0166) : アメダス地点。気温・降水量・湿度・風・日照時間・雪を取得。
  - 青森 (prec_no=31, block_no=47575) : 気象台。天気概況(昼/夜)・気温などを取得。
    弘前は天気カテゴリを観測していないため、天気概況は青森のデータで代用する。

取得期間: 過去5年分（デフォルトでは今日から5年前の月初め〜先月末）

出力:
  data/raw/hirosaki_YYYYMM.csv
  data/raw/aomori_YYYYMM.csv
  各地点を1本のCSVに結合したもの (hirosaki_daily.csv / aomori_daily.csv) も出力する。

注意:
  気象庁のサーバーに配慮し、リクエストの間に待機時間を入れている。
  実行には数分かかる。
"""

import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.data.jma.go.jp/obd/stats/etrn/view/{page}.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (research use; JMA daily data download script)"}
SLEEP_SEC = 1.5  # 気象庁サーバーへの配慮のための待機時間

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATIONS = {
    "hirosaki": {"page": "daily_a1", "prec_no": 31, "block_no": "0166"},
    "aomori": {"page": "daily_s1", "prec_no": 31, "block_no": "47575"},
}


def month_range(start: date, end: date):
    """start〜end (両端含む) の各月の1日を列挙する。"""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield date(y, m, 1)
        m += 1
        if m > 12:
            m = 1
            y += 1


def fetch_month(page: str, prec_no: int, block_no: str, year: int, month: int) -> pd.DataFrame | None:
    """指定地点・年月の日別データを1つのDataFrameとして取得する。"""
    params = {
        "prec_no": prec_no,
        "block_no": block_no,
        "year": year,
        "month": month,
        "day": 1,
        "view": "",
    }
    url = BASE_URL.format(page=page)
    resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    html = resp.text

    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S)
    data_rows = []
    for r in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        # 先頭列が日付(1〜31の数字)である行だけをデータ行とみなす
        if cells and cells[0].isdigit():
            data_rows.append(cells)

    if not data_rows:
        return None

    df = pd.DataFrame(data_rows)
    df.insert(0, "year", year)
    df.insert(1, "month", month)
    df = df.rename(columns={0: "day"})
    return df


def download_station(name: str, page: str, prec_no: int, block_no: str, start: date, end: date):
    monthly_frames = []
    for d in month_range(start, end):
        print(f"[{name}] {d.year}-{d.month:02d} を取得中...")
        try:
            df = fetch_month(page, prec_no, block_no, d.year, d.month)
        except requests.RequestException as e:
            print(f"  -> 取得失敗: {e}")
            df = None
        if df is not None:
            monthly_frames.append(df)
            df.to_csv(OUT_DIR / f"{name}_{d.year}{d.month:02d}.csv", index=False, encoding="utf-8-sig")
        time.sleep(SLEEP_SEC)

    if monthly_frames:
        combined = pd.concat(monthly_frames, ignore_index=True)
        combined.to_csv(OUT_DIR / f"{name}_daily.csv", index=False, encoding="utf-8-sig")
        print(f"[{name}] 結合ファイルを保存しました: {OUT_DIR / f'{name}_daily.csv'} ({len(combined)}行)")
    else:
        print(f"[{name}] データを取得できませんでした。")


def main():
    today = date.today()
    # 過去5年分。今月はまだ月末まで揃っていない可能性があるため先月末までとする。
    end_year, end_month = today.year, today.month - 1
    if end_month == 0:
        end_year -= 1
        end_month = 12
    end = date(end_year, end_month, 1)
    start = date(end.year - 5, end.month, 1)

    print(f"取得期間: {start.year}-{start.month:02d} 〜 {end.year}-{end.month:02d}")

    for name, info in STATIONS.items():
        download_station(name, info["page"], info["prec_no"], info["block_no"], start, end)


if __name__ == "__main__":
    main()

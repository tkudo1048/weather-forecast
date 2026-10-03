"""
今夜の予測: 今日（または --date の日）の10分データから特徴量を作り、models/model.joblib で
翌日の天気（晴/曇/雨/雪）の確率を出して logs/predictions.csv に1行追記する。

記録する条件（本番扱い）。どれか1つでも満たさなければ「暫定」として表示だけし、記録しない。
- 観測日の最終観測時刻 last_obs が 23:00 以降（観測が1日の終わり近くまで届いている）。
  last_obs は弘前と青森の最終観測時刻の早いほう。どちらかの地点の当日データが1件も取れない場合は
  None となり、記録しない（その地点の特徴量も欠測になる）。
  最高・最低気温や最小湿度、最大風速などは途中までの観測でも欠測にならず静かにずれるため、
  欠測数ではなく「23:00以降まで観測が届いているか」で判定する。
- 欠測した特徴量が0個。
- 実行日（date.today()）が観測日と同じ（0時過ぎの実行や過去日の --date 指定は暫定扱い）。
  ただし --cutoff で打ち切り時刻を指定した再現実行は、この条件を問わない。
  --cutoff 付きで記録した行は note に「cutoff再現 HH:MM」を必ず入れ、cutoff 列も埋める。
  score_predictions.py はこの行を本番と別の group（cutoff_replay）で集計する。
- 同じ対象日（翌日）の本番行がすでにあれば追記しない。
- --force: 上記の条件を満たさなくても記録する（既存行は置き換え）。条件を満たさない行には provisional=1 を付ける。
- 実行: プロジェクト直下で `PYTHONPATH=code python code/predict_tonight.py`
  夜の実行の再現: `... predict_tonight.py --date 2026-10-02 --cutoff 23:30`
"""
import argparse
from datetime import date, datetime, time, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fetch_realtime import build_features, log

ROOT = Path(__file__).resolve().parent.parent
LOG_CSV = ROOT / "logs" / "predictions.csv"
MIN_LAST_OBS = time(23, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat(), help="観測日 YYYY-MM-DD（既定: 今日）")
    ap.add_argument("--cutoff", default=None, help="観測の打ち切り時刻 HH:MM（夜の実行の再現用）")
    ap.add_argument("--force", action="store_true", help="条件を満たさなくても記録する（provisional=1）。同じ対象日の既存行は置き換える")
    a = ap.parse_args()
    run_at = datetime.now()
    obs_day = date.fromisoformat(a.date)
    target = obs_day + timedelta(days=1)
    cutoff = None
    if a.cutoff:
        hh, mm = map(int, a.cutoff.split(":"))
        cutoff = datetime.combine(obs_day, time(0)) + timedelta(hours=hh, minutes=mm)

    old = pd.read_csv(LOG_CSV, dtype=str) if LOG_CSV.exists() else pd.DataFrame()
    if len(old) and (old["target_date"] == target.isoformat()).any() and not a.force:
        print(f"対象日 {target} の予測は記録済みのため追記しません（--force で上書き）")
        return

    bundle = joblib.load(ROOT / "models" / "model.joblib")
    feats, last_obs, _ = build_features(obs_day, cutoff=cutoff)
    X = pd.DataFrame([feats[bundle["features"]].values], columns=bundle["features"])
    proba = bundle["pipeline"].predict_proba(X)[0]
    classes = bundle["classes"]
    pred = classes[int(np.argmax(proba))]
    n_missing = int(feats.isna().sum())

    reasons = []
    if last_obs is None or pd.Timestamp(last_obs) < pd.Timestamp(datetime.combine(obs_day, MIN_LAST_OBS)):
        reasons.append(f"最終観測 {last_obs} が {obs_day} 23:00 より前")
    if n_missing > 0:
        reasons.append(f"欠測特徴量 {n_missing}個")
    if cutoff is None and run_at.date() != obs_day:
        reasons.append(f"実行日 {run_at.date()} が観測日 {obs_day} と違う（日付またぎ・過去日）")
    provisional = bool(reasons)

    print(f"観測日 {obs_day}（最終観測 {last_obs}）→ 対象日 {target} の予測")
    for c, p in zip(classes, proba):
        print(f"  {c}: {p:.1%}")
    print(f"予測クラス: {pred} / 欠測特徴量: {n_missing}個")

    if provisional and not a.force:
        print("暫定・記録しない（理由: " + " / ".join(reasons) + "）")
        log.info("暫定のため記録せず: target=%s pred=%s 理由=%s", target, pred, reasons)
        return

    row = {"run_at": run_at.isoformat(timespec="seconds"), "target_date": target.isoformat(),
           **{f"p_{c}": round(float(p), 4) for c, p in zip(classes, proba)},
           "pred": pred, "last_obs_time": str(last_obs) if last_obs is not None else "",
           "n_missing_features": n_missing, "provisional": int(provisional),
           "cutoff": a.cutoff or "",
           "note": " / ".join(([f"cutoff再現 {a.cutoff}"] if a.cutoff else []) + reasons)}
    if len(old) and a.force:
        old = old[old["target_date"] != target.isoformat()]
    out = pd.concat([old, pd.DataFrame([row]).astype(str)], ignore_index=True)
    LOG_CSV.parent.mkdir(exist_ok=True)
    out.to_csv(LOG_CSV, index=False, encoding="utf-8")
    log.info("予測を記録: %s", row)
    print("記録しました" + ("（provisional=1: " + " / ".join(reasons) + "）" if provisional else ""))


if __name__ == "__main__":
    main()

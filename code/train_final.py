"""
本番用モデルの学習: 特徴量セット(c)（夜に作れる18特徴量）で、全期間（学習+評価の全1852行）を使って
ロジスティック回帰を学習し、models/model.joblib に保存する。

- 手法・前処理は評価時（ablation_evening_features.py, 評価364行で Macro F1 0.484）と同じ。
  全期間で学習し直すので、このモデル自体の評価値は存在しない（評価は上記の値と、運用後の score_predictions.py で見る）。
- 保存内容: pipeline, 特徴量名の順序, クラス順（predict_proba の列順）, 学習期間。
- 実行: プロジェクト直下で `PYTHONPATH=code python code/train_final.py`
"""
from datetime import datetime
from pathlib import Path

import joblib

from compare_models import load_data, build_models, TARGET_COL
from fetch_realtime import FEATURES

ROOT = Path(__file__).resolve().parent.parent


def main():
    df = load_data()
    lr = build_models()["ロジスティック回帰"]  # SimpleImputer(中央値)+StandardScaler+LogReg(balanced)
    lr.fit(df[FEATURES], df[TARGET_COL])
    bundle = {
        "pipeline": lr,
        "features": list(FEATURES),
        "classes": list(lr.classes_),
        "train_period": (str(df.date.min().date()), str(df.date.max().date())),
        "n_rows": len(df),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    out = ROOT / "models" / "model.joblib"
    out.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, out)
    print(f"学習 {len(df)}行 {bundle['train_period'][0]}〜{bundle['train_period'][1]} / 特徴量 {len(FEATURES)}個")
    print(f"クラス順: {bundle['classes']}")
    print(f"保存: {out}")


if __name__ == "__main__":
    main()

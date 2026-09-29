"""
勾配ブースティング（sklearn GradientBoostingClassifier）のハイパーパラメータチューニングと、
ロジスティック回帰（train_model.py / compare_models.py と同一設定）との再比較。

- 探索は学習データ（1,488行）内の TimeSeriesSplit(n_splits=5) による交差検証のみで行う。
  （各foldで「過去で学習→直後の期間で検証」。シャッフルなし）
- 評価データ（364行）は、最良パラメータ決定後の最終評価にのみ使う（リーク防止）。
- 実行方法: プロジェクト直下で `PYTHONPATH=code python code/tune_gradient_boosting.py`
- 探索指標は Macro F1（クラス不均衡を考慮し、4クラスを平等に扱うため）。
"""
import time
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from compare_models import (load_data, time_series_split, build_models, evaluate,
                            FEATURE_COLS, TARGET_COL, CLASSES)

PARAM_GRID = {
    "clf__n_estimators": [50, 100, 200, 400],
    "clf__max_depth": [1, 2, 3],
    "clf__learning_rate": [0.01, 0.03, 0.1],
    "clf__min_samples_leaf": [1, 20, 50],
    "clf__subsample": [0.6, 0.8, 1.0],
}


def main():
    df = load_data()
    train_df, eval_df = time_series_split(df, eval_days=365)
    X_tr, y_tr = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_ev, y_ev = eval_df[FEATURE_COLS], eval_df[TARGET_COL]

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(random_state=42)),
    ])
    cv = TimeSeriesSplit(n_splits=5)
    gs = GridSearchCV(pipe, PARAM_GRID, cv=cv, scoring="f1_macro",
                      n_jobs=-1, return_train_score=True)
    t0 = time.time()
    gs.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    n = len(gs.cv_results_["params"])
    print(f"探索組み合わせ数: {n}, fold数: 5, 所要時間: {elapsed:.1f}秒")
    print("最良パラメータ:", gs.best_params_)
    i = gs.best_index_
    print(f"CV Macro F1: {gs.best_score_:.3f} (±{gs.cv_results_['std_test_score'][i]:.3f}), "
          f"CV学習側 Macro F1: {gs.cv_results_['mean_train_score'][i]:.3f}")

    res = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score")
    cols = [c for c in res.columns if c.startswith("param_")] + ["mean_test_score", "std_test_score", "mean_train_score"]
    print(res[cols].head(10).to_string(index=False))
    res[cols].to_csv("research/gb_tuning_cv_results.csv", index=False)

    # ロジスティック回帰のCVスコア（同じ分割）も参考に算出
    lr = build_models()["ロジスティック回帰"]
    from sklearn.model_selection import cross_val_score
    lr_cv = cross_val_score(lr, X_tr, y_tr, cv=cv, scoring="f1_macro")
    print(f"ロジスティック回帰 CV Macro F1: {lr_cv.mean():.3f} (±{lr_cv.std():.3f})")
    gb_folds = [gs.cv_results_[f"split{k}_test_score"][i] for k in range(5)]
    print("fold別 GB:", [round(v, 3) for v in gb_folds], " LR:", [round(v, 3) for v in lr_cv])

    models = {"ロジスティック回帰": lr, "勾配ブースティング(チューニング後)": gs.best_estimator_}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        tra, trf, _ = evaluate(y_tr, m.predict(X_tr))
        eva, evf, cf = evaluate(y_ev, m.predict(X_ev))
        print(f"\n===== {name} =====")
        print(f"学習 Acc={tra:.3f} F1={trf:.3f} / 評価 Acc={eva:.3f} F1={evf:.3f} / "
              f"ギャップ Acc={tra-eva:.3f} F1={trf-evf:.3f}")
        print("クラス別F1:", {c: round(v, 3) for c, v in cf.items()})


if __name__ == "__main__":
    main()

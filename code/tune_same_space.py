"""
Optuna版と同じ探索空間（max_features含む）を離散化し、GridSearchCV（全探索）と
RandomizedSearchCV（Optunaと同予算150回）で勾配ブースティングをチューニングして公平に比較する。

目的: 「Optunaが良く見えたのは探索アルゴリズムの差か、探索空間の差か」を切り分ける。
- 探索は学習データ内の TimeSeriesSplit(n_splits=5) のみ。評価データ（直近365日）は最終評価だけに使う。
- 実行方法: プロジェクト直下で `PYTHONPATH=code python code/tune_same_space.py`
"""
import time
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV

from compare_models import load_data, time_series_split, evaluate, FEATURE_COLS, TARGET_COL
from tune_gradient_boosting_optuna import make_pipe

SPACE = {
    "clf__n_estimators": [100, 200, 300, 450, 600],
    "clf__max_depth": [2, 3, 4, 5],
    "clf__learning_rate": [0.01, 0.02, 0.05, 0.1],
    "clf__min_samples_leaf": [10, 30, 70],
    "clf__subsample": [0.6, 0.8, 1.0],
    "clf__max_features": ["sqrt", "log2", None],
}
N_ITER = 150


def main():
    df = load_data()
    train_df, eval_df = time_series_split(df, eval_days=365)
    X_tr, y_tr = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_ev, y_ev = eval_df[FEATURE_COLS], eval_df[TARGET_COL]
    cv = TimeSeriesSplit(n_splits=5)

    searches = {
        "GridSearchCV(全2160通り)": GridSearchCV(make_pipe(), SPACE, cv=cv, scoring="f1_macro", n_jobs=-1),
        "RandomizedSearchCV(150回)": RandomizedSearchCV(make_pipe(), SPACE, n_iter=N_ITER, cv=cv,
                                                        scoring="f1_macro", n_jobs=-1, random_state=42),
    }
    rows = []
    for name, s in searches.items():
        t0 = time.time()
        s.fit(X_tr, y_tr)
        el = time.time() - t0
        i = s.best_index_
        cv_mean, cv_std = s.cv_results_["mean_test_score"][i], s.cv_results_["std_test_score"][i]
        m = s.best_estimator_
        tra, trf, _ = evaluate(y_tr, m.predict(X_tr))
        eva, evf, _ = evaluate(y_ev, m.predict(X_ev))
        params = {k.replace("clf__", ""): v for k, v in s.best_params_.items()}
        res = pd.DataFrame(s.cv_results_)
        top = res.sort_values("mean_test_score", ascending=False)["mean_test_score"]
        print(f"[{name}] {el:.1f}秒 best={params} CV={cv_mean:.3f}(±{cv_std:.3f}) "
              f"上位10平均={top.head(10).mean():.3f} 評価Acc={eva:.3f} F1={evf:.3f} ギャップF1={trf-evf:.3f}")
        rows.append(dict(method=name, n_candidates=len(res), best_params=str(params),
                         cv_f1=round(cv_mean, 3), cv_std=round(cv_std, 3),
                         train_acc=round(tra, 3), train_f1=round(trf, 3),
                         eval_acc=round(eva, 3), eval_f1=round(evf, 3),
                         gap_acc=round(tra - eva, 3), gap_f1=round(trf - evf, 3), seconds=round(el, 1)))
        if name.startswith("Grid"):
            # 全探索のCV分布（最大値選択の楽観バイアスの目安）
            print(res["mean_test_score"].describe().round(3).to_string())
            res.to_csv("research/same_space_grid_cv.csv", index=False)

    rows.append(dict(method="Optuna TPE(150回, 連続空間) ※既存結果", cv_f1=0.498, eval_acc=0.503, eval_f1=0.514))
    rows.append(dict(method="ロジスティック回帰 ※既存結果", eval_acc=0.486, eval_f1=0.494))
    out = pd.DataFrame(rows)
    out.to_csv("research/same_space_results.csv", index=False)
    print(out.drop(columns=["best_params"]).to_string(index=False))


if __name__ == "__main__":
    main()

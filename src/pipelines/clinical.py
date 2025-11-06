from autogluon.tabular import TabularDataset, TabularPredictor
import os
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')  
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from collections import Counter
import math
from typing import List, Sequence, Optional, Tuple

data_dir = "/home/emxie/scratch/CHIMERA_minimal_baseline/chimera-bladder-brs-submission/src/data"
resources_dir = "/home/emxie/scratch/CHIMERA_minimal_baseline/chimera-bladder-brs-submission/resources"

# Collect all clinical data into a list
clinical_records = []
for patient_id in os.listdir(data_dir):
    patient_path = os.path.join(data_dir, patient_id)
    if os.path.isdir(patient_path):
        cd_file = os.path.join(patient_path, f"{patient_id}_CD.json")
        if os.path.exists(cd_file):
            with open(cd_file, "r") as f:
                clinical_data = json.load(f)
                # Ensure we retain patient ID for plotting and tracking
                if isinstance(clinical_data, dict):
                    clinical_data["PATIENT_ID"] = patient_id
                clinical_records.append(clinical_data)

# Convert to DataFrame
clinical_df = pd.DataFrame(clinical_records)
if "PATIENT_ID" not in clinical_df.columns:
    clinical_df["PATIENT_ID"] = [str(i) for i in range(len(clinical_df))]
clinical_df["PATIENT_ID"] = clinical_df["PATIENT_ID"].astype(str)
clinical_df = clinical_df.set_index("PATIENT_ID", drop=False)

# Keep only rows that have a BRS label
clinical_df = clinical_df.dropna(subset=["BRS"]).reset_index(drop=True)

# Make a binary label: BRS3 vs BRS1/2
# Clean any stray whitespace / case just in case
clinical_df["BRS"] = clinical_df["BRS"].astype(str).str.strip().str.upper()
clinical_df["BRS_binary"] = np.where(clinical_df["BRS"] == "BRS3", "BRS3", "BRS1_2")

clinical_df = clinical_df.drop(columns=["BRS"])  # drop the multiclass label

# Create directories in resources if they don't exist
models_dir = os.path.join(resources_dir, "models")
results_dir = os.path.join(resources_dir, "results", "clinical_cv")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# =====================
# Plotting utilities
# =====================
def _safe_series(x: pd.Series):
    """Ensure 1D numeric array and drop NaNs for density/hist plots."""
    return pd.to_numeric(x, errors="coerce").dropna().values

def _proportion_diff(train_counts: Counter, test_counts: Counter) -> float:
    """Max absolute difference in category proportions (0..1)."""
    cats = set(train_counts) | set(test_counts)
    n_tr = sum(train_counts.values()) or 1
    n_te = sum(test_counts.values()) or 1
    diffs = []
    for c in cats:
        p_tr = train_counts.get(c, 0) / n_tr
        p_te = test_counts.get(c, 0) / n_te
        diffs.append(abs(p_tr - p_te))
    return float(max(diffs)) if diffs else 0.0

def _js_divergence_numeric(train_vals: np.ndarray, test_vals: np.ndarray, bins: int = 30) -> float:
    """Jensen–Shannon divergence between train/test histograms (0 = identical)."""
    if len(train_vals) == 0 or len(test_vals) == 0:
        return 0.0
    lo = np.nanmin([np.min(train_vals), np.min(test_vals)])
    hi = np.nanmax([np.max(train_vals), np.max(test_vals)])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0
    tr_hist, edges = np.histogram(train_vals, bins=bins, range=(lo, hi), density=True)
    te_hist, _     = np.histogram(test_vals, bins=edges, density=True)
    # avoid zeros
    tr = tr_hist + 1e-12
    te = te_hist + 1e-12
    tr /= tr.sum()
    te /= te.sum()
    m = 0.5 * (tr + te)
    js = 0.5 * (np.sum(tr * np.log(tr/m)) + np.sum(te * np.log(te/m)))
    return float(max(js, 0.0))

def plot_categorical_distributions(
    df: pd.DataFrame,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    categorical_cols: List[str],
    out_path: str,
    fold_title: str,
    roc_auc: Optional[float] = None,
    n_cols: int = 2,
):
    """Bar charts of category counts for train vs. test."""
    cols = [c for c in categorical_cols if c in df.columns]
    if not cols:
        return

    n = len(cols)
    n_cols = max(1, n_cols)
    n_rows = math.ceil(n / n_cols)

    fig_h = max(3 * n_rows, 4)
    fig_w = max(6 * n_cols, 8)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle(
        f"Categorical Feature Distributions (Train vs. Test)\n{fold_title}"
        + (f" (ROC-AUC: {roc_auc:.3f})" if roc_auc is not None else ""),
        fontsize=16, fontweight="bold", y=1.02
    )

    for i, col in enumerate(cols):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]

        tr = df.loc[train_ids, col].astype("object")
        te = df.loc[test_ids,  col].astype("object")
        tr_counts = Counter(tr.fillna("NA"))
        te_counts = Counter(te.fillna("NA"))
        cats = sorted(set(tr_counts) | set(te_counts), key=lambda x: str(x))

        # heights
        tr_heights = [tr_counts.get(k, 0) for k in cats]
        te_heights = [te_counts.get(k, 0) for k in cats]

        x = np.arange(len(cats))
        width = 0.42

        ax.bar(x - width/2, tr_heights, width, label="train", alpha=0.8)
        ax.bar(x + width/2, te_heights, width, label="test",  alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in cats], rotation=25, ha="right")
        ax.set_ylabel("count")

        max_prop_diff = _proportion_diff(tr_counts, te_counts)
        warn = " \u26A0\uFE0F" if max_prop_diff >= 0.15 else ""
        ax.set_title(f"{col} (max \u0394prop={max_prop_diff:.2f}){warn}", fontsize=11)

        if r == 0 and c == 0:
            ax.legend(frameon=False)

    # hide empty axes
    for j in range(n, n_rows*n_cols):
        r, c = divmod(j, n_cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_numeric_distributions(
    df: pd.DataFrame,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    numeric_cols: List[str],
    out_path: str,
    fold_title: str,
    roc_auc: Optional[float] = None,
    n_cols: int = 4,
):
    """Overlaid hist+density for train vs. test numerics; title shows JS divergence."""
    cols = [c for c in numeric_cols if c in df.columns]
    if not cols:
        return

    n = len(cols)
    n_cols = max(1, n_cols)
    n_rows = math.ceil(n / n_cols)

    fig_h = max(2.8 * n_rows, 5)
    fig_w = max(4.2 * n_cols, 8)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle(
        f"Numerical Feature Distributions (Train vs. Test)\n{fold_title}"
        + (f" (ROC-AUC: {roc_auc:.3f})" if roc_auc is not None else ""),
        fontsize=16, fontweight="bold", y=1.02
    )

    for i, col in enumerate(cols):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]

        tr_vals = _safe_series(df.loc[train_ids, col])
        te_vals = _safe_series(df.loc[test_ids,  col])

        # common range
        vals = np.concatenate([tr_vals, te_vals]) if len(tr_vals) and len(te_vals) else tr_vals
        if len(vals) == 0:
            ax.set_title(f"{col}: no data")
            ax.axis("off")
            continue

        lo, hi = np.nanmin(vals), np.nanmax(vals)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            ax.set_title(f"{col}: constant")
            ax.axis("off")
            continue

        bins = int(np.clip(np.sqrt(len(vals)), 15, 50))  # robust bin rule

        # hist overlays
        ax.hist(tr_vals, bins=bins, range=(lo, hi), alpha=0.35, density=True, label="train")
        ax.hist(te_vals, bins=bins, range=(lo, hi), alpha=0.35, density=True, label="test")

        js = _js_divergence_numeric(tr_vals, te_vals, bins=bins)
        warn = " \u26A0\uFE0F" if js >= 0.10 else ""
        ax.set_title(f"{col} (JS={js:.2f}){warn}", fontsize=11)

        if r == 0 and c == 0:
            ax.legend(frameon=False)

    # hide empty axes
    for j in range(n, n_rows*n_cols):
        r, c = divmod(j, n_cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_cindex_by_model_and_fold(
    cindex_by_model: dict,
    out_path: str,
    title: Optional[str] = None,
    n_cols: int = 2,
):
    """Create grouped bar plots of train/test ROC-AUC per fold for each model.

    Parameters
    ----------
    cindex_by_model: dict
        Mapping: model name -> {"train": List[float], "test": List[float]}
    out_path: str
        Path to save the resulting figure (PNG).
    title: Optional[str]
        Optional figure-level title.
    n_cols: int
        Number of subplot columns.
    """
    if not cindex_by_model:
        return

    models = list(cindex_by_model.keys())
    n_models = len(models)
    n_cols = max(1, n_cols)
    n_rows = int(math.ceil(n_models / n_cols))

    # Determine max number of folds
    max_folds = 0
    for m in models:
        tr = cindex_by_model[m].get("train", [])
        te = cindex_by_model[m].get("test", [])
        max_folds = max(max_folds, len(tr), len(te))
    if max_folds == 0:
        return

    fig_h = max(3.2 * n_rows, 4)
    fig_w = max(8.5 * n_cols, 14)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    # Grouped bar positions: widen spacing between each fold's train/test pair
    group_gap = 1.6  # >1 increases spacing between pairs while keeping pair tight
    x_idx = np.arange(1, max_folds + 1)
    x = (x_idx - 1) * group_gap
    width = 0.42
    sep = width * 0.65

    for i, model in enumerate(models):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]

        tr_vals = np.array(cindex_by_model[model].get("train", []), dtype=float)
        te_vals = np.array(cindex_by_model[model].get("test", []), dtype=float)

        # Pad to max_folds with NaNs for alignment
        if tr_vals.size < max_folds:
            tr_vals = np.pad(tr_vals, (0, max_folds - tr_vals.size), constant_values=np.nan)
        if te_vals.size < max_folds:
            te_vals = np.pad(te_vals, (0, max_folds - te_vals.size), constant_values=np.nan)

        # Plot bars; NaNs are skipped by using masked arrays
        tr_masked = np.ma.masked_invalid(tr_vals)
        te_masked = np.ma.masked_invalid(te_vals)

        bars_tr = ax.bar(
            x - sep, tr_masked, width,
            label="Train", color="#4C78A8", alpha=0.9,
            edgecolor="#1b2a41", linewidth=0.8
        )
        bars_te = ax.bar(
            x + sep, te_masked, width,
            label="Test", color="#F58518", alpha=0.9,
            edgecolor="#5a2a00", linewidth=0.8
        )

        # Title without "ROC-AUC"
        ax.set_title(f"{model}")
        ax.set_xlabel("Fold")
        ax.set_ylabel("ROC-AUC")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x_idx])

        # Dynamic y-limit with headroom, clamp to [0, 1.05]
        finite_vals = np.concatenate([tr_masked.compressed(), te_masked.compressed()]) if (tr_masked.count() + te_masked.count()) else np.array([0.0])
        ymax = float(np.nanmax(finite_vals)) if finite_vals.size else 0.0
        upper = min(max(0.7, ymax + 0.08), 1.05)
        ax.set_ylim(0.0, upper)

        # Horizontal gridlines for readability (light)
        ax.yaxis.grid(True, linestyle="--", color="#bbbbbb", alpha=0.5)
        ax.set_axisbelow(True)

        # Legend on every subplot with light background
        ax.legend(
            frameon=True, fancybox=True, framealpha=0.65,
            edgecolor="#cccccc", facecolor="#ffffff",
            loc="best"
        )

        # Add numeric labels on top of each bar, inside if near top
        def _annotate(bars):
            for bar in bars:
                h = bar.get_height()
                if not np.isfinite(h):
                    continue
                xmid = bar.get_x() + bar.get_width()/2
                dy = 0.012
                ytext = min(h + dy, upper - 0.01)
                va = "bottom"
                ax.text(xmid, ytext, f"{h:.2f}", ha="center", va=va, fontsize=8)
        _annotate(bars_tr)
        _annotate(bars_te)

    # Hide any unused axes
    for j in range(n_models, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# =====================
# 10-fold CV with AutoGluon models and ensembling
# =====================

cindex_by_model = {}
per_fold_rows = []

label_col = "BRS_binary"
positive_label = "BRS3"

if clinical_df.shape[0] >= 10:
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_labels = (clinical_df[label_col] == positive_label).astype(int).values
    ids = clinical_df.index.to_numpy()

    TIME_LIMIT_PER_FOLD = 300  # seconds

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(ids, y_labels), start=1):
        train_ids = ids[tr_idx]
        test_ids = ids[te_idx]

        train_df = clinical_df.loc[train_ids]
        test_df = clinical_df.loc[test_ids]

        fold_path = os.path.join(models_dir, f"cv_fold_{fold_idx}")
        os.makedirs(fold_path, exist_ok=True)

        hyperparameters = {
            'XGB': {},   # XGBoost
            'GBM': {},   # LightGBM
            'CAT': {},   # CatBoost
            'RF': {},    # RandomForest
        }

        predictor = TabularPredictor(
            label=label_col,
            eval_metric="roc_auc",
            path=fold_path,
        ).fit(
            train_data=TabularDataset(train_df),
            presets="medium_quality_faster_train",
            hyperparameters=hyperparameters,
            time_limit=TIME_LIMIT_PER_FOLD,
            verbosity=1,
            num_stack_levels=0,
            num_bag_folds=0,
            num_bag_sets=0,
        )

        # Identify trained model names per type (robust to AutoGluon naming)
        model_names = predictor.model_names()
        # Skip meta-ensembles
        model_names = [m for m in model_names if not m.startswith('WeightedEnsemble')]
        family_patterns = {
            'XGB': ['XGBoost', 'XGB'],
            'GBM': ['LightGBM', 'GBM'],
            'CAT': ['CatBoost', 'CAT'],
            'RF':  ['RandomForest', 'RF'],
        }
        type_to_name = {}
        for m in model_names:
            upper_m = m.upper()
            for fam_key, pats in family_patterns.items():
                if fam_key in type_to_name:
                    continue
                for p in pats:
                    if p.upper() in upper_m:
                        type_to_name[fam_key] = m
                        break
        # Optional: fallback to best model of a family if multiple names exist is unnecessary here

        # Collect probabilities for ensembling
        prob_train_by_model = {}
        prob_test_by_model = {}

        y_tr = (train_df[label_col] == positive_label).astype(int).values
        y_te = (test_df[label_col] == positive_label).astype(int).values

        for mkey, mname in type_to_name.items():
            try:
                p_tr = predictor.predict_proba(train_df, model=mname)
                p_te = predictor.predict_proba(test_df, model=mname)
                # Select positive class probability
                pcol = positive_label if positive_label in p_tr.columns else p_tr.columns[-1]
                prob_train = p_tr[pcol].values
                prob_test = p_te[pcol].values
                auc_tr = float(roc_auc_score(y_tr, prob_train))
                auc_te = float(roc_auc_score(y_te, prob_test))
                cindex_by_model.setdefault(mkey, {"train": [], "test": []})
                cindex_by_model[mkey]["train"].append(auc_tr)
                cindex_by_model[mkey]["test"].append(auc_te)
                per_fold_rows.append({"fold": fold_idx, "model": mkey, "split": "train", "cindex": auc_tr})
                per_fold_rows.append({"fold": fold_idx, "model": mkey, "split": "test",  "cindex": auc_te})
                prob_train_by_model[mkey] = prob_train
                prob_test_by_model[mkey] = prob_test
            except Exception as _e:
                print(f"Warning: failed model {mkey} on fold {fold_idx}: {_e}")

        # Ensembles
        def _safe_auc(name: str, prob_tr: np.ndarray, prob_te: np.ndarray):
            try:
                auc_tr = float(roc_auc_score(y_tr, prob_tr))
                auc_te = float(roc_auc_score(y_te, prob_te))
                cindex_by_model.setdefault(name, {"train": [], "test": []})
                cindex_by_model[name]["train"].append(auc_tr)
                cindex_by_model[name]["test"].append(auc_te)
                per_fold_rows.append({"fold": fold_idx, "model": name, "split": "train", "cindex": auc_tr})
                per_fold_rows.append({"fold": fold_idx, "model": name, "split": "test",  "cindex": auc_te})
            except Exception as _e:
                print(f"Warning: failed ensemble {name} on fold {fold_idx}: {_e}")

        # Boosting-only ensemble (XGB, GBM, CAT)
        boost_keys = [k for k in ['XGB', 'GBM', 'CAT'] if k in prob_test_by_model]
        if len(boost_keys) >= 2:
            tr_stack = np.vstack([prob_train_by_model[k] for k in boost_keys])
            te_stack = np.vstack([prob_test_by_model[k] for k in boost_keys])
            _safe_auc("Ensemble_AvgBoost", tr_stack.mean(axis=0), te_stack.mean(axis=0))
            _safe_auc("Ensemble_MedianBoost", np.median(tr_stack, axis=0), np.median(te_stack, axis=0))

        # All-models average
        all_keys = list(prob_test_by_model.keys())
        if len(all_keys) >= 2:
            tr_stack_all = np.vstack([prob_train_by_model[k] for k in all_keys])
            te_stack_all = np.vstack([prob_test_by_model[k] for k in all_keys])
            _safe_auc("Ensemble_AvgAll", tr_stack_all.mean(axis=0), te_stack_all.mean(axis=0))

        # Per-fold plots: feature distributions
        non_label_cols = [c for c in clinical_df.columns if c not in {label_col, "PATIENT_ID"}]
        num_cols = clinical_df[non_label_cols].select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in non_label_cols if c not in num_cols]
        fold_title = f"Fold {fold_idx}: N_train={len(train_df)}, N_test={len(test_df)}"
        plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        try:
            plot_categorical_distributions(
                clinical_df, train_ids, test_ids, cat_cols,
                out_path=os.path.join(plots_dir, f"fold_{fold_idx}_categorical.png"),
                fold_title=fold_title,
                roc_auc=None,
            )
        except Exception as _e:
            print(f"Warning: categorical plot failed on fold {fold_idx}: {_e}")
        try:
            plot_numeric_distributions(
                clinical_df, train_ids, test_ids, num_cols,
                out_path=os.path.join(plots_dir, f"fold_{fold_idx}_numeric.png"),
                fold_title=fold_title,
                roc_auc=None,
            )
        except Exception as _e:
            print(f"Warning: numeric plot failed on fold {fold_idx}: {_e}")

    # Save per-fold ROC-AUC metrics
    try:
        df_per_fold = pd.DataFrame(per_fold_rows)
        per_fold_csv = os.path.join(results_dir, 'per_fold_cindex.csv')
        df_per_fold.to_csv(per_fold_csv, index=False)
        print(f"Saved per-fold ROC-AUC metrics to: {per_fold_csv}")
    except Exception as _e:
        print(f"Warning: failed to save per-fold metrics: {_e}")

    # Plot ROC-AUC grouped bars across models and folds
    try:
        bar_out = os.path.join(results_dir, 'cindex_by_model_and_fold.png')
        plot_cindex_by_model_and_fold(
            cindex_by_model,
            out_path=bar_out,
            title="10-fold CV: Train/Test ROC-AUC per Model",
            n_cols=2,
        )
        print(f"Saved ROC-AUC bar plots to: {bar_out}")
    except Exception as _e:
        print(f"Warning: failed to plot ROC-AUC by model and fold: {_e}")

    # Extended summary metrics (mean, std, min, quartiles, max, count) per model and split
    try:
        stats_rows = []

        def _compute_stats(values):
            series = pd.Series(values, dtype=float)
            series = series[np.isfinite(series)]
            if series.empty:
                return None
            return {
                'mean': float(series.mean()),
                'std': float(series.std(ddof=1)) if series.size > 1 else 0.0,
                'min': float(series.min()),
                '25%': float(series.quantile(0.25)),
                '50%': float(series.median()),
                '75%': float(series.quantile(0.75)),
                'max': float(series.max()),
                'n': int(series.size),
            }

        for model_name, splits in cindex_by_model.items():
            for split_name in ['train', 'test']:
                vals = splits.get(split_name, [])
                stats = _compute_stats([v for v in vals if np.isfinite(v)])
                if stats is None:
                    continue
                row = {'model': model_name, 'split': split_name}
                row.update(stats)
                stats_rows.append(row)

        df_extended = pd.DataFrame(stats_rows)
        extended_csv = os.path.join(results_dir, 'summary_cindex_extended.csv')
        df_extended.to_csv(extended_csv, index=False)
        print(f"Saved detailed summary metrics to: {extended_csv}")
    except Exception as _e:
        print(f"Warning: failed to save detailed summary metrics: {_e}")
else:
    print("Not enough samples for 10-fold CV; skipping CV section.")

# Train final RF model on full dataset
print("\n" + "="*60)
print("TRAINING FINAL RF MODEL ON FULL DATASET")
print("="*60)

try:
    # Define save path for the final RF model
    final_rf_dir = os.path.join(models_dir, "rf_final")

    # Only train RandomForest within AutoGluon
    rf_hyperparams = {
        'RF': {
            # Reasonable defaults; AutoGluon will also tune based on presets
        }
    }

    predictor = TabularPredictor(
        label="BRS_binary",
        eval_metric="f1",
        path=final_rf_dir
    ).fit(
        TabularDataset(clinical_df),
        presets="high_quality_fast_inference_only_refit",
        hyperparameters=rf_hyperparams,
        time_limit=600,
        verbosity=2
    )

    # Save lightweight metadata for inference
    metadata = {
        "label": "BRS_binary",
        "positive_class": "BRS3"
    }
    with open(os.path.join(final_rf_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f)

    print(f"Final RF model saved to: {final_rf_dir}")
except Exception as e:
    print(f"Error training final RF model: {e}")

# =====================
# Across-fold ensembles on full dataset
# =====================
try:
    folds_available = []
    for k in range(1, 11):
        fpath = os.path.join(models_dir, f"cv_fold_{k}")
        if os.path.isdir(fpath):
            folds_available.append((k, fpath))

    if folds_available:
        # Prepare container of predictions per family across folds
        preds_family_full = { 'XGB': [], 'GBM': [], 'CAT': [], 'RF': [] }
        # Track chosen model names per fold for reproducibility
        family_model_by_fold = {}
        # Load full dataset once
        full_df = clinical_df.copy()
        y_full = (full_df[label_col] == positive_label).astype(int).values

        for fold_idx, fold_path in folds_available:
            try:
                p = TabularPredictor.load(fold_path)
                mnames = [m for m in p.model_names() if not m.startswith('WeightedEnsemble')]
                # map to families (robust)
                family_patterns = {
                    'XGB': ['XGBoost', 'XGB'],
                    'GBM': ['LightGBM', 'GBM'],
                    'CAT': ['CatBoost', 'CAT'],
                    'RF':  ['RandomForest', 'RF'],
                }
                fam_to_model = {}
                for m in mnames:
                    up = m.upper()
                    for fam, pats in family_patterns.items():
                        if fam in fam_to_model:
                            continue
                        for pat in pats:
                            if pat.upper() in up:
                                fam_to_model[fam] = m
                                break
                # record chosen models for this fold (only families we found)
                family_model_by_fold[str(fold_idx)] = {k: v for k, v in fam_to_model.items() if k in ['XGB', 'GBM', 'CAT', 'RF']}
                # get probs for each detected family
                for fam, mname in fam_to_model.items():
                    try:
                        probs = p.predict_proba(full_df, model=mname)
                        pcol = positive_label if positive_label in probs.columns else probs.columns[-1]
                        preds_family_full[fam].append(probs[pcol].values.astype(float))
                    except Exception as _e:
                        print(f"Warn: fold {fold_idx} prediction failed for {fam}: {_e}")
            except Exception as _e:
                print(f"Warn: could not load predictor for fold {fold_idx}: {_e}")

        # Build requested ensembles
        ens_results = []

        def _safe_auc(name: str, arrs: List[np.ndarray]):
            arrs = [a for a in arrs if isinstance(a, np.ndarray) and a.shape[0] == y_full.shape[0]]
            if not arrs:
                print(f"Skipping ensemble {name}: no valid predictions")
                return None
            avg = np.mean(np.stack(arrs, axis=0), axis=0)
            auc = float(roc_auc_score(y_full, avg))
            ens_results.append({ 'ensemble': name, 'roc_auc': auc, 'n_models': len(arrs) })
            print(f"{name}: ROC AUC={auc:.4f} using {len(arrs)} model predictions")
            return avg

        # CAT 10-fold ensemble (average CAT predictions across available folds)
        cat_avg = _safe_auc('CAT_10fold_ensemble', preds_family_full['CAT'])

        # AvgBoost 4x10 (average XGB, GBM, CAT, RF across all folds)
        all_model_preds = preds_family_full['XGB'] + preds_family_full['GBM'] + preds_family_full['CAT'] + preds_family_full['RF']
        _ = _safe_auc('Ensemble_AvgBoost_4x10', all_model_preds)

        # Ensemble CAT + RF (20 models in total if all folds present)
        cat_rf_preds = preds_family_full['CAT'] + preds_family_full['RF']
        _ = _safe_auc('Ensemble_CAT_RF_20', cat_rf_preds)

        # Save detailed config/weights for Ensemble_CAT_RF_20
        try:
            # Build model list entries from available folds
            model_entries = []
            for fold_idx, fold_path in folds_available:
                rel_path = os.path.join('models', os.path.basename(fold_path))
                mapping = family_model_by_fold.get(str(fold_idx), {})
                for fam in ['CAT', 'RF']:
                    mname = mapping.get(fam)
                    if not mname:
                        continue
                    model_entries.append({
                        'fold': int(fold_idx),
                        'family': fam,
                        'model_name': mname,
                        'fold_rel_path': rel_path,
                    })

            n_models = len(model_entries)
            if n_models > 0:
                weight = 1.0 / float(n_models)
                for e in model_entries:
                    e['weight'] = weight

                payload = {
                    'ensemble_name': 'Ensemble_CAT_RF_20',
                    'label': label_col,
                    'positive_class': positive_label,
                    'aggregation': 'weighted_mean',
                    'models': model_entries,
                }
                cfg_path = os.path.join(models_dir, 'ensemble_CAT_RF_20.json')
                with open(cfg_path, 'w') as f:
                    json.dump(payload, f, indent=2)
                print(f"Saved Ensemble_CAT_RF_20 config to: {cfg_path}")
            else:
                print("No CAT/RF models found across folds; skipping save of Ensemble_CAT_RF_20 config")
        except Exception as _e:
            print(f"Warning: failed to save Ensemble_CAT_RF_20 config: {_e}")

        # Save ensemble metrics
        df_ens = pd.DataFrame(ens_results)
        out_ens_csv = os.path.join(results_dir, 'across_fold_ensembles.csv')
        df_ens.to_csv(out_ens_csv, index=False)
        print(f"Saved across-fold ensemble metrics to: {out_ens_csv}")
    else:
        print("No cv_fold_* predictors found; skipping across-fold ensembles.")
except Exception as _e:
    print(f"Warning: across-fold ensemble computation failed: {_e}")

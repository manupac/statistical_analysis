"""
Compute inter-phoneme distance matrices, Mantel tests, bootstrap CIs,
and nearest-centroid phoneme-identification results.
"""

import os
import warnings
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = "data/pipeline"
OUT_DIR = "results/interphoneme"
TABLE_DIR = os.path.join(OUT_DIR, "tables")
FIGURE_DIR = os.path.join(OUT_DIR, "figures")

WHISPER_LAYER = 20
XLSR_LAYER = 20
N_PERMUTATIONS = 9999
N_BOOTSTRAP = 2000
SEED = 42
DPI = 150

ORAL_VOWELS = ["a", "e", "i", "o", "u", "y", "ø", "œ", "ɑ", "ə", "ɛ"]
ORAL_VOWELS_SET = set(ORAL_VOWELS)

FOCAL_PAIRS = [("e", "ɛ"), ("e", "ø"), ("y", "u")]

MODELS = {
    "whisper": {"npz": "features_whisper_pca.npz"},
    "xlsr": {"npz": "features_xlsr_pca.npz"},
}


def load_acoustic():
    df_raw = pd.read_csv(os.path.join(DATA_DIR, "features_acoustic_norm.csv"))
    keep_mask = ~df_raw["phoneme"].str.contains("ding", na=False).values
    df = df_raw[keep_mask].reset_index(drop=True)

    l1_map = {
        "fr": "L1", "fra": "L1", "l1": "L1", "french": "L1", "native": "L1",
        "ru": "L2", "rus": "L2", "l2": "L2", "russian": "L2",
    }
    df["L1_norm"] = df["L1"].astype(str).str.lower().str.strip().map(l1_map)

    return df, keep_mask


def load_neural_pca50(model: str, layer: int) -> np.ndarray:
    npz_path = os.path.join(DATA_DIR, MODELS[model]["npz"])
    npz = np.load(npz_path)
    key = f"hiddens_layer{layer}_pca50"

    if key not in npz:
        raise KeyError(f"{key} not found in {npz_path}")

    print(f"    {model} layer {layer}: loaded {key}, shape={npz[key].shape}")
    return npz[key]


def compute_acoustic_centroids(df):
    centroids = {}

    for phoneme in ORAL_VOWELS:
        sub = df[df["phoneme"] == phoneme][["F1_lob_mid", "F2_lob_mid"]].dropna()
        if len(sub):
            centroids[phoneme] = sub.values.mean(axis=0)

    present = [p for p in ORAL_VOWELS if p in centroids]
    return centroids, present


def pooled_within_covariance(df, phonemes):
    ss = np.zeros((2, 2))
    total_df = 0

    for phoneme in phonemes:
        sub = df[df["phoneme"] == phoneme][["F1_lob_mid", "F2_lob_mid"]].dropna().values
        if len(sub) < 2:
            continue

        ss += np.cov(sub.T, ddof=1) * (len(sub) - 1)
        total_df += len(sub) - 1

    if total_df == 0:
        return np.eye(2)

    return ss / total_df


def build_acoustic_distance_matrices(df, phonemes):
    centroids, _ = compute_acoustic_centroids(df)
    present = [p for p in phonemes if p in centroids]

    centroid_matrix = np.array([centroids[p] for p in present])

    d_euclidean = squareform(pdist(centroid_matrix, metric="euclidean"))

    sigma = pooled_within_covariance(df, present)
    sigma_inv = np.linalg.pinv(sigma)

    n = len(present)
    d_mahalanobis = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            diff = centroid_matrix[i] - centroid_matrix[j]
            distance = np.sqrt(max(0.0, diff @ sigma_inv @ diff))
            d_mahalanobis[i, j] = d_mahalanobis[j, i] = distance

    return (
        pd.DataFrame(d_euclidean, index=present, columns=present),
        pd.DataFrame(d_mahalanobis, index=present, columns=present),
    )


def cosine_distance(a, b) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0 or nb == 0:
        return np.nan

    return float(1.0 - np.dot(a, b) / (na * nb))


def build_neural_distance_matrix(df, projection, phonemes):
    centroids = {}

    for phoneme in phonemes:
        mask = df["phoneme"].values == phoneme
        if mask.sum():
            centroids[phoneme] = projection[mask].mean(axis=0)

    present = [p for p in phonemes if p in centroids]
    n = len(present)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance = cosine_distance(centroids[present[i]], centroids[present[j]])
            distance_matrix[i, j] = distance_matrix[j, i] = distance

    return pd.DataFrame(distance_matrix, index=present, columns=present)


def upper_triangle(matrix):
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def mantel_test(matrix_a, matrix_b, rng):
    from scipy.stats import rankdata

    shared = sorted(set(matrix_a.index) & set(matrix_b.index))
    if len(shared) < 3:
        return np.nan, np.nan, len(shared)

    a = matrix_a.loc[shared, shared].values
    b = matrix_b.loc[shared, shared].values

    tri_a = upper_triangle(a)
    tri_b = upper_triangle(b)

    r_obs, _ = spearmanr(tri_a, tri_b)

    n = len(shared)
    row_idx, col_idx = np.triu_indices(n, k=1)

    ranked_a = rankdata(tri_a).astype(np.float64)
    ranked_a -= ranked_a.mean()
    norm_a = np.linalg.norm(ranked_a)

    if norm_a > 0:
        ranked_a /= norm_a

    flat_b = b[row_idx, col_idx]
    rank_lookup = rankdata(flat_b).astype(np.float64)

    rank_matrix_b = np.zeros((n, n), dtype=np.float64)
    rank_matrix_b[row_idx, col_idx] = rank_lookup
    rank_matrix_b[col_idx, row_idx] = rank_lookup

    mean_rank = rank_lookup.mean()
    count_extreme = 0

    for _ in range(N_PERMUTATIONS):
        perm = rng.permutation(n)
        ranked_b_perm = rank_matrix_b[perm[row_idx], perm[col_idx]]
        ranked_b_perm -= mean_rank

        norm_b = np.linalg.norm(ranked_b_perm)
        if norm_b == 0:
            continue

        ranked_b_perm /= norm_b
        r_perm = np.dot(ranked_a, ranked_b_perm)

        if abs(r_perm) >= abs(r_obs):
            count_extreme += 1

    p_value = (count_extreme + 1) / (N_PERMUTATIONS + 1)
    return float(r_obs), float(p_value), len(shared)


def pair_distances(df, proj_whisper, proj_xlsr, p, q):
    phonemes = df["phoneme"].values
    p_mask = phonemes == p
    q_mask = phonemes == q

    ac_cols = ["F1_lob_mid", "F2_lob_mid"]

    p_ac = df.loc[p_mask, ac_cols].dropna().values
    q_ac = df.loc[q_mask, ac_cols].dropna().values

    if len(p_ac) and len(q_ac):
        acoustic = float(np.linalg.norm(p_ac.mean(axis=0) - q_ac.mean(axis=0)))
    else:
        acoustic = np.nan

    def neural_distance(projection):
        if projection is None or not p_mask.sum() or not q_mask.sum():
            return np.nan
        return cosine_distance(projection[p_mask].mean(axis=0), projection[q_mask].mean(axis=0))

    return acoustic, neural_distance(proj_whisper), neural_distance(proj_xlsr)


def bootstrap_pair_distance(df, proj_whisper, proj_xlsr, rng):
    speakers = df["speaker"].unique()
    speaker_col = df["speaker"].values
    rows = []

    for p, q in FOCAL_PAIRS:
        if p not in set(df["phoneme"]) or q not in set(df["phoneme"]):
            print(f"  Skipping {p}-{q}: missing phoneme")
            continue

        observed = pair_distances(df, proj_whisper, proj_xlsr, p, q)

        boot_acoustic = []
        boot_whisper = []
        boot_xlsr = []

        for _ in range(N_BOOTSTRAP):
            sampled_speakers = rng.choice(speakers, size=len(speakers), replace=True)
            token_idx = np.concatenate([
                np.where(speaker_col == speaker)[0]
                for speaker in sampled_speakers
            ])

            df_boot = df.iloc[token_idx].reset_index(drop=True)
            wh_boot = proj_whisper[token_idx]
            xl_boot = proj_xlsr[token_idx]

            if not ((df_boot["phoneme"] == p).any() and (df_boot["phoneme"] == q).any()):
                continue

            ac_d, wh_d, xl_d = pair_distances(df_boot, wh_boot, xl_boot, p, q)

            if not np.isnan(ac_d):
                boot_acoustic.append(ac_d)
            if not np.isnan(wh_d):
                boot_whisper.append(wh_d)
            if not np.isnan(xl_d):
                boot_xlsr.append(xl_d)

        for measure, obs, values in [
            ("acoustic_euclidean", observed[0], boot_acoustic),
            (f"whisper_layer{WHISPER_LAYER}_cosine", observed[1], boot_whisper),
            (f"xlsr_layer{XLSR_LAYER}_cosine", observed[2], boot_xlsr),
        ]:
            if len(values) >= 10:
                ci_low = np.percentile(values, 2.5)
                ci_high = np.percentile(values, 97.5)
            else:
                ci_low = np.nan
                ci_high = np.nan

            rows.append({
                "pair": f"{p}-{q}",
                "measure": measure,
                "observed": round(float(obs), 6) if not np.isnan(obs) else np.nan,
                "ci_lower_95": round(float(ci_low), 6) if not np.isnan(ci_low) else np.nan,
                "ci_upper_95": round(float(ci_high), 6) if not np.isnan(ci_high) else np.nan,
                "n_bootstrap_valid": len(values),
            })

    return pd.DataFrame(rows)


def loso_classify_acoustic(df, phonemes, mahalanobis=False):
    predictions = {}

    for speaker in df["speaker"].unique():
        test_mask = df["speaker"] == speaker
        train_mask = ~test_mask & df["phoneme"].isin(phonemes)

        train = df[train_mask]
        centroids = {}

        for phoneme in phonemes:
            sub = train[train["phoneme"] == phoneme][["F1_lob_mid", "F2_lob_mid"]].dropna()
            if len(sub):
                centroids[phoneme] = sub.values.mean(axis=0)

        if not centroids:
            continue

        labels = list(centroids.keys())
        centroid_matrix = np.array([centroids[p] for p in labels])

        if mahalanobis:
            sigma_train = pooled_within_covariance(train, labels)
            sigma_inv_train = np.linalg.pinv(sigma_train)

        test = df[test_mask & df["phoneme"].isin(phonemes)]
        test_values = test[["F1_lob_mid", "F2_lob_mid"]].values

        for i, idx in enumerate(test.index):
            value = test_values[i]

            if np.isnan(value).any():
                predictions[idx] = None
                continue

            if mahalanobis:
                diffs = centroid_matrix - value
                distances = np.array([
                    np.sqrt(max(0.0, diff @ sigma_inv_train @ diff))
                    for diff in diffs
                ])
            else:
                distances = np.linalg.norm(centroid_matrix - value, axis=1)

            predictions[idx] = labels[int(np.argmin(distances))]

    return predictions


def loso_classify_neural(df, projection, phonemes):
    predictions = {}

    for speaker in df["speaker"].unique():
        test_mask = (df["speaker"] == speaker).values
        train_mask = (~test_mask) & df["phoneme"].isin(phonemes).values

        centroids = {}

        for phoneme in phonemes:
            ph_train = train_mask & (df["phoneme"].values == phoneme)
            if ph_train.sum():
                centroids[phoneme] = projection[ph_train].mean(axis=0)

        if not centroids:
            continue

        labels = list(centroids.keys())
        centroid_matrix = np.array([centroids[p] for p in labels])

        centroid_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
        centroid_norms = np.where(centroid_norms == 0, 1.0, centroid_norms)
        centroid_matrix = centroid_matrix / centroid_norms

        test_indices = np.where(test_mask & df["phoneme"].isin(phonemes).values)[0]

        for pos in test_indices:
            value = projection[pos]
            norm = np.linalg.norm(value)
            value = value / norm if norm > 0 else value

            similarities = centroid_matrix @ value
            predictions[df.index[pos]] = labels[int(np.argmax(similarities))]

    return predictions


def evaluate_classifier(df, predictions, phonemes, label):
    true_labels = []
    pred_labels = []
    token_results = {}

    l1_correct = 0
    l1_total = 0
    l2_correct = 0
    l2_total = 0

    for idx, pred in predictions.items():
        if pred is None:
            continue

        true = df.loc[idx, "phoneme"]
        if true not in phonemes:
            continue

        l1_status = df.loc[idx, "L1_norm"]
        correct = pred == true

        true_labels.append(true)
        pred_labels.append(pred)
        token_results[idx] = {
            "true": true,
            "pred": pred,
            "correct": correct,
            "L1_norm": l1_status,
        }

        if l1_status == "L1":
            l1_correct += int(correct)
            l1_total += 1
        elif l1_status == "L2":
            l2_correct += int(correct)
            l2_total += 1

    n_total = len(true_labels)
    n_correct = sum(t == p for t, p in zip(true_labels, pred_labels))

    accuracy_row = {
        "classifier": label,
        "n_tokens": n_total,
        "accuracy": round(n_correct / n_total, 4) if n_total else np.nan,
        "accuracy_L1": round(l1_correct / l1_total, 4) if l1_total else np.nan,
        "accuracy_L2": round(l2_correct / l2_total, 4) if l2_total else np.nan,
    }

    f1_rows = []

    for phoneme in phonemes:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == phoneme and p == phoneme)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != phoneme and p == phoneme)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == phoneme and p != phoneme)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        f1_rows.append({
            "classifier": label,
            "phoneme": phoneme,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for t in true_labels if t == phoneme),
        })

    confusion = pd.DataFrame(0, index=phonemes, columns=phonemes)

    for true, pred in zip(true_labels, pred_labels):
        if true in phonemes and pred in phonemes:
            confusion.loc[true, pred] += 1

    return accuracy_row, f1_rows, confusion, token_results


def mcnemar_test_pair(results_a, results_b, label_a, label_b):
    from scipy.stats import chi2 as chi2_dist

    shared_idx = set(results_a.keys()) & set(results_b.keys())

    if len(shared_idx) < 10:
        return {
            "classifier_A": label_a,
            "classifier_B": label_b,
            "n_shared": len(shared_idx),
            "b": np.nan,
            "c": np.nan,
            "chi2": np.nan,
            "p_value": np.nan,
        }

    b = 0
    c = 0

    for idx in shared_idx:
        ca = results_a[idx]["correct"]
        cb = results_b[idx]["correct"]

        if not ca and cb:
            b += 1
        if ca and not cb:
            c += 1

    if b + c == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
        p_value = float(chi2_dist.sf(chi2, df=1))

    return {
        "classifier_A": label_a,
        "classifier_B": label_b,
        "n_shared": len(shared_idx),
        "b": b,
        "c": c,
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
    }


def plot_distance_matrix(matrix_df, title, path):
    labels = list(matrix_df.index)
    matrix = matrix_df.values

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")

    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=11)

    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_bootstrap_ci(bootstrap_df, path):
    pairs = bootstrap_df["pair"].unique()
    measures = bootstrap_df["measure"].unique()

    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4), sharey=False)

    if len(pairs) == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")

    for ax, pair in zip(axes, pairs):
        sub = bootstrap_df[bootstrap_df["pair"] == pair]

        for i, measure in enumerate(measures):
            row = sub[sub["measure"] == measure]

            if row.empty:
                continue

            obs = row["observed"].values[0]
            low = row["ci_lower_95"].values[0]
            high = row["ci_upper_95"].values[0]

            if np.isnan(low) or np.isnan(high):
                yerr = None
            else:
                yerr = [[obs - low], [high - obs]]

            ax.errorbar(
                i,
                obs,
                yerr=yerr,
                fmt="o",
                color=cmap(i),
                capsize=4,
                label=measure,
            )

        ax.set_title(f"/{pair}/", fontsize=11)
        ax.set_xticks(range(len(measures)))
        ax.set_xticklabels(
            [m.replace("_cosine", "").replace("_euclidean", "") for m in measures],
            rotation=30,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel("Distance")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    handles = [
        plt.Line2D([0], [0], marker="o", color=cmap(i), linestyle="", label=m)
        for i, m in enumerate(measures)
    ]

    fig.legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle("Bootstrap 95% CI on inter-phoneme distances", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_confusion(confusion_df, title, path):
    matrix = confusion_df.values.astype(float)

    row_sum = matrix.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1, row_sum)
    matrix_norm = matrix / row_sum

    labels = list(confusion_df.index)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Recall")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix_norm[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if value > 0.6 else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    rng = np.random.default_rng(SEED)

    print("Loading data ...")
    df, keep_mask = load_acoustic()

    vowel_mask = df["phoneme"].isin(ORAL_VOWELS_SET).values
    df_vowels = df[vowel_mask].reset_index(drop=True)

    print(f"  Tokens after artifact removal: {len(df):,}")
    print(f"  Oral vowel tokens: {len(df_vowels):,}")

    print("Loading neural projections ...")
    proj_wh_raw = load_neural_pca50("whisper", WHISPER_LAYER)
    proj_xl_raw = load_neural_pca50("xlsr", XLSR_LAYER)

    proj_wh = proj_wh_raw[keep_mask][vowel_mask]
    proj_xl = proj_xl_raw[keep_mask][vowel_mask]

    present_phonemes = [p for p in ORAL_VOWELS if p in set(df_vowels["phoneme"].unique())]
    print(f"  Phonemes present: {present_phonemes}")

    print("Building distance matrices ...")
    d_euclidean, d_mahalanobis = build_acoustic_distance_matrices(df_vowels, present_phonemes)
    d_whisper = build_neural_distance_matrix(df_vowels, proj_wh, present_phonemes)
    d_xlsr = build_neural_distance_matrix(df_vowels, proj_xl, present_phonemes)

    matrices = [
        (d_euclidean, "distance_matrix_acoustic_euclidean", "Acoustic distance — Euclidean"),
        (d_mahalanobis, "distance_matrix_acoustic_mahalanobis", "Acoustic distance — Mahalanobis"),
        (d_whisper, f"distance_matrix_whisper_layer{WHISPER_LAYER}", f"Whisper layer {WHISPER_LAYER} — cosine distance"),
        (d_xlsr, f"distance_matrix_xlsr_layer{XLSR_LAYER}", f"XLS-R layer {XLSR_LAYER} — cosine distance"),
    ]

    for matrix, name, title in matrices:
        csv_path = os.path.join(TABLE_DIR, f"{name}.csv")
        fig_name = name.replace("distance_matrix_", "distmat_") + ".png"
        fig_path = os.path.join(FIGURE_DIR, fig_name)

        matrix.to_csv(csv_path)
        plot_distance_matrix(matrix, title, fig_path)

    print("Running Mantel tests ...")
    mantel_pairs = [
        ("Acoustic (Euclidean) vs Whisper", d_euclidean, d_whisper),
        ("Acoustic (Euclidean) vs XLS-R", d_euclidean, d_xlsr),
        ("Whisper vs XLS-R", d_whisper, d_xlsr),
    ]

    mantel_rows = []

    for label, matrix_a, matrix_b in mantel_pairs:
        r, p_value, n_phonemes = mantel_test(matrix_a, matrix_b, rng)
        mantel_rows.append({
            "pair": label,
            "mantel_r": round(r, 6),
            "p_value": round(p_value, 6),
            "n_phonemes": n_phonemes,
            "n_permutations": N_PERMUTATIONS,
        })
        print(f"  {label}: r={r:.4f}, p={p_value:.4f}")

    pd.DataFrame(mantel_rows).to_csv(
        os.path.join(TABLE_DIR, "mantel_results.csv"),
        index=False,
    )

    print("Computing bootstrap CIs ...")
    bootstrap_df = bootstrap_pair_distance(df_vowels, proj_wh, proj_xl, rng)
    bootstrap_df.to_csv(
        os.path.join(TABLE_DIR, "bootstrap_ci_distances.csv"),
        index=False,
    )
    plot_bootstrap_ci(
        bootstrap_df,
        os.path.join(FIGURE_DIR, "bootstrap_ci_distances.png"),
    )

    print("Running nearest-centroid classifiers ...")
    df_vowels_indexed = df[vowel_mask].copy()

    classifier_results = {}
    accuracy_rows = []
    f1_rows = []
    confusion_matrices = {}

    def run_eval(predictions, label):
        accuracy, f1, confusion, token_results = evaluate_classifier(
            df_vowels_indexed,
            predictions,
            present_phonemes,
            label,
        )
        accuracy_rows.append(accuracy)
        f1_rows.extend(f1)
        confusion_matrices[label] = confusion
        classifier_results[label] = token_results

        print(
            f"  {label}: accuracy={accuracy['accuracy']:.4f}, "
            f"L1={accuracy['accuracy_L1']:.4f}, L2={accuracy['accuracy_L2']:.4f}"
        )

    preds_ac_euclidean = loso_classify_acoustic(
        df_vowels_indexed,
        present_phonemes,
        mahalanobis=False,
    )
    run_eval(preds_ac_euclidean, "acoustic_euclidean")

    preds_ac_mahalanobis = loso_classify_acoustic(
        df_vowels_indexed,
        present_phonemes,
        mahalanobis=True,
    )
    run_eval(preds_ac_mahalanobis, "acoustic_mahalanobis")

    preds_whisper = loso_classify_neural(
        df_vowels_indexed,
        proj_wh,
        present_phonemes,
    )
    run_eval(preds_whisper, f"whisper_layer{WHISPER_LAYER}")

    preds_xlsr = loso_classify_neural(
        df_vowels_indexed,
        proj_xl,
        present_phonemes,
    )
    run_eval(preds_xlsr, f"xlsr_layer{XLSR_LAYER}")

    pd.DataFrame(accuracy_rows).to_csv(
        os.path.join(TABLE_DIR, "classifier_accuracy.csv"),
        index=False,
    )
    pd.DataFrame(f1_rows).to_csv(
        os.path.join(TABLE_DIR, "classifier_f1_perclass.csv"),
        index=False,
    )

    for label, confusion in confusion_matrices.items():
        plot_confusion(
            confusion,
            f"Confusion matrix — {label}",
            os.path.join(FIGURE_DIR, f"confusion_{label}.png"),
        )

    print("Running McNemar tests ...")
    mcnemar_rows = []

    for label_a, label_b in combinations(classifier_results.keys(), 2):
        mcnemar_rows.append(
            mcnemar_test_pair(
                classifier_results[label_a],
                classifier_results[label_b],
                label_a,
                label_b,
            )
        )

    pd.DataFrame(mcnemar_rows).to_csv(
        os.path.join(TABLE_DIR, "mcnemar_results.csv"),
        index=False,
    )

    print("Done.")
    print(f"Tables saved to {TABLE_DIR}")
    print(f"Figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
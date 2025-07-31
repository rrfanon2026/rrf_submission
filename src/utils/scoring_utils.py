import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, fbeta_score, matthews_corrcoef

def f_beta_score(p, r, beta=0.5):
    b2 = beta * beta
    return 0.0 if (p + r) == 0 else (1 + b2) * p * r / (b2 * p + r)

def compute_weighted_cumulative_scores(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Computes cumulative weighted scores over questions for each founder.

    Args:
        X (np.ndarray): Binary prediction matrix of shape (Q, F)
        W (np.ndarray): Weight vector of shape (Q,)

    Returns:
        np.ndarray: Cumulative score matrix of shape (Q, F)
    """
    Q, F = X.shape
    S = np.zeros((Q, F))
    running = np.zeros(F)

    for q in range(Q):
        running += W[q] * X[q, :]
        S[q, :] = running.copy()

    return S

def compute_weights(weighting, exponent, prec_array, ratio_array):
    """
    Compute question weights based on precision scores and weighting method.

    Args:
        weighting (str): 'adaboost' or numeric string ('0', '1')
        exponent (float or str): exponent value or 'adaboost'
        prec_array (np.ndarray): precision scores per question
        ratio_array (np.ndarray): normalized precision ratios

    Returns:
        tuple: (weight array W, label)
    """
    if weighting == 'adaboost':
        epsilon_q = 1 - prec_array
        epsilon_q = np.clip(epsilon_q, 1e-6, 0.999)
        W = 0.5 * np.log((1 - epsilon_q) / epsilon_q)
        W = np.clip(W, -2, 2)
        W = W - W.mean() + 1
        label = 'adaboost'
    else:
        W = ratio_array ** exponent
        W = W / W.mean()
        label = exponent
    return W, label

# def select_best_hyperparams(X_train, y_train, train_ids, weighting, exponents,
#                             score_thresholds, Q, success_values, n_splits, outer_seed):
#     prec_array = np.zeros(Q, dtype=float)
#     for qi in range(Q):
#         preds_q = X_train[qi, :]
#         tp = ((preds_q == 1) & (y_train == 1)).sum()
#         fp = ((preds_q == 1) & (y_train == 0)).sum()
#         prec_array[qi] = tp / (tp + fp) if tp + fp > 0 else 0.0

#     mean_prec_train = prec_array.mean()
#     ratio_array = (prec_array / mean_prec_train) if mean_prec_train > 0 else np.ones(Q)

#     best_combo = None
#     best_mean_f05 = -1.0
#     inner_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=outer_seed)
#     weighting_options = ['adaboost'] if weighting == 'adaboost' else exponents

#     for w in weighting_options:
#         W, label = compute_weights(weighting, w, prec_array, ratio_array)
#         scores_train = compute_weighted_cumulative_scores(X_train, W)
#         for n_q in range(1, Q + 1):
#             for t in score_thresholds:
#                 f05_vals = []
#                 for _, (_, iv) in enumerate(inner_skf.split(train_ids, y_train)):
#                     val_ids = train_ids[iv]
#                     pos = np.where(np.isin(train_ids, val_ids))[0]
#                     preds = (scores_train[n_q - 1, pos] >= t).astype(int)
#                     yv = success_values[val_ids]
#                     tp_i = ((preds == 1) & (yv == 1)).sum()
#                     fp_i = ((preds == 1) & (yv == 0)).sum()
#                     fn_i = ((preds == 0) & (yv == 1)).sum()
#                     p = tp_i / (tp_i + fp_i) if tp_i + fp_i > 0 else 0.0
#                     r = tp_i / (tp_i + fn_i) if tp_i + fn_i > 0 else 0.0
#                     f05_vals.append(f_beta_score(p, r))

#                 f05_score = np.mean(f05_vals)
#                 if f05_score > best_mean_f05:
#                     best_mean_f05 = f05_score
#                     best_combo = (label, n_q, t)

#     return best_combo, prec_array, ratio_array, best_mean_f05

def select_best_hyperparams(
    X_train, y_train, train_ids, weighting, exponents,
    score_thresholds, Q, success_values, n_splits, outer_seed,
    optimise_for="f0.5"  # Options: "precision", "f0.5", "f1", "f2"
):
    prec_array = np.zeros(Q, dtype=float)
    for qi in range(Q):
        preds_q = X_train[qi, :]
        tp = ((preds_q == 1) & (y_train == 1)).sum()
        fp = ((preds_q == 1) & (y_train == 0)).sum()
        prec_array[qi] = tp / (tp + fp) if tp + fp > 0 else 0.0

    mean_prec_train = prec_array.mean()
    ratio_array = (prec_array / mean_prec_train) if mean_prec_train > 0 else np.ones(Q)

    best_combo = None
    best_score = -1.0
    inner_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=outer_seed)
    weighting_options = ['adaboost'] if weighting == 'adaboost' else exponents

    for w in weighting_options:
        W, label = compute_weights(weighting, w, prec_array, ratio_array)
        scores_train = compute_weighted_cumulative_scores(X_train, W)

        for n_q in range(1, Q + 1):
            for t in score_thresholds:
                fold_scores = []

                for _, (_, iv) in enumerate(inner_skf.split(train_ids, y_train)):
                    val_ids = train_ids[iv]
                    pos = np.where(np.isin(train_ids, val_ids))[0]
                    preds = (scores_train[n_q - 1, pos] >= t).astype(int)
                    yv = success_values[val_ids]

                    tp_i = ((preds == 1) & (yv == 1)).sum()
                    fp_i = ((preds == 1) & (yv == 0)).sum()
                    fn_i = ((preds == 0) & (yv == 1)).sum()

                    p = tp_i / (tp_i + fp_i) if tp_i + fp_i > 0 else 0.0
                    r = tp_i / (tp_i + fn_i) if tp_i + fn_i > 0 else 0.0

                    if optimise_for == "precision":
                        fold_scores.append(p)
                    elif optimise_for == "f1":
                        fold_scores.append(fbeta_score(yv, preds, beta=1.0, zero_division=0))
                    elif optimise_for == "f2":
                        fold_scores.append(fbeta_score(yv, preds, beta=2.0, zero_division=0))
                    elif optimise_for == "f0.5":
                        fold_scores.append(f_beta_score(p, r))  # custom function
                    else:
                        raise ValueError(f"Unknown optimise_for metric: {optimise_for}")

                mean_score = np.mean(fold_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_combo = (label, n_q, t)

    return best_combo, prec_array, ratio_array, best_score

def evaluate_test_fold(X, W, n_q, threshold, test_ids, success_values):
    scores_full = compute_weighted_cumulative_scores(X, W)
    preds_test = (scores_full[n_q - 1, test_ids] >= threshold).astype(int)
    y_test = success_values[test_ids]

    tp = ((preds_test == 1) & (y_test == 1)).sum()
    fp = ((preds_test == 1) & (y_test == 0)).sum()
    tn = ((preds_test == 0) & (y_test == 0)).sum()
    fn = ((preds_test == 0) & (y_test == 1)).sum()
    p_out = tp / (tp + fp) if tp + fp > 0 else 0.0
    r_out = tp / (tp + fn) if tp + fn > 0 else 0.0
    f05_out = f_beta_score(p_out, r_out)
    f05_out_sk = fbeta_score(y_test, preds_test, beta=0.5, zero_division=0)
    f1_out = f1_score(y_test, preds_test, zero_division=0)
    f2_out = fbeta_score(y_test, preds_test, beta=2.0, zero_division=0)
    mcc_out = matthews_corrcoef(y_test, preds_test)    

    return preds_test, y_test, tp, fp, tn, fn, p_out, r_out, f05_out, f05_out_sk, f1_out, f2_out, mcc_out

def select_best_centroid_params(X, y_train, train_ix, weighting, param_grid, score_thresholds, Q, success_values, n_splits, random_state):
    """
    Run inner CV for centroid-style (clustering) grid search and return best params.
    """
    best_score = -np.inf
    best_combo = None

    # Recompute precision array using only train_ix
    tp_arr = np.zeros(Q)
    fp_arr = np.zeros(Q)

    for q in range(Q):
        preds_q = X[q, train_ix]
        labels_q = success_values[train_ix]
        tp_arr[q] = ((preds_q == 1) & (labels_q == 1)).sum()
        fp_arr[q] = ((preds_q == 1) & (labels_q == 0)).sum()

    prec_array = tp_arr / (tp_arr + fp_arr + 1e-8)
    ratio_array = prec_array / prec_array.mean()

    inner_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for idx, (exponent, pct) in enumerate(param_grid, 1):
        if idx == 1 or idx == len(param_grid) or idx % 100 == 0:
            print(f"  Inner combo {idx}/{len(param_grid)} (e={exponent}, pct={pct})")

        W, _ = compute_weights(weighting, exponent, prec_array, ratio_array)
        S_train = compute_weighted_cumulative_scores(X[:, train_ix], W)

        T = len(score_thresholds)
        sum_p, cnt_p = {}, {}

        for _, val_ix in inner_skf.split(train_ix, y_train):
            val_f = train_ix[val_ix]
            y_val = success_values[val_f]
            mask_cols = np.isin(train_ix, val_f)
            sub = S_train[:, mask_cols]

            for i, t in enumerate(score_thresholds):
                preds_q_t = (sub >= t).astype(int)
                for j in range(Q):
                    preds = (preds_q_t[:j + 1].sum(axis=0) > 0).astype(int)
                    tp = ((preds == 1) & (y_val == 1)).sum()
                    fp = ((preds == 1) & (y_val == 0)).sum()
                    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
                    key = (i, j)
                    sum_p[key] = sum_p.get(key, 0.0) + prec
                    cnt_p[key] = cnt_p.get(key, 0) + 1

        mat = np.zeros((T, Q))
        for i in range(T):
            for j in range(Q):
                if (i, j) in sum_p:
                    mat[i, j] = sum_p[(i, j)] / cnt_p[(i, j)]
                else:
                    mat[i, j] = 0.0

        cutoff = np.percentile(mat, pct)
        mask = mat >= cutoff
        pts = np.argwhere(mask)
        i_star, j_star = pts.mean(axis=0)
        t_sel = score_thresholds[int(round(i_star))]
        n_q_sel = int(round(j_star)) + 1

        f05s = []
        for _, val_ix in inner_skf.split(train_ix, y_train):
            val_f = train_ix[val_ix]
            y_val = success_values[val_f]
            mask_cols = np.isin(train_ix, val_f)
            preds = (S_train[n_q_sel - 1, mask_cols] >= t_sel).astype(int)

            tp = ((preds == 1) & (y_val == 1)).sum()
            fp = ((preds == 1) & (y_val == 0)).sum()
            fn = ((preds == 0) & (y_val == 1)).sum()
            p = tp / (tp + fp) if tp + fp > 0 else 0.0
            r = tp / (tp + fn) if tp + fn > 0 else 0.0
            f05 = f_beta_score(p, r)
            f05s.append(f05)

        avg_f05 = np.mean(f05s)

        if avg_f05 > best_score:
            best_score = avg_f05
            best_combo = (exponent, pct, n_q_sel, t_sel)

    return best_combo, prec_array, ratio_array, best_score

def balanced_accuracy(tp, tn, fp, fn):
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (tpr + tnr)

def evaluate_votes(votes, y_true, threshold):
    from sklearn.metrics import (roc_auc_score,average_precision_score, precision_score, recall_score,f1_score,fbeta_score,accuracy_score,confusion_matrix,roc_curve,auc,precision_recall_curve)
    """Compute all metrics for one vector of soft votes and a hard threshold."""
    roc   = roc_auc_score(y_true, votes) if len(np.unique(y_true))>1 else np.nan
    pr    = average_precision_score(y_true, votes) if len(np.unique(y_true))>1 else np.nan
    yhat  = (votes>=threshold).astype(int)
    prec  = precision_score(y_true, yhat, zero_division=0)
    rec   = recall_score(y_true, yhat, zero_division=0)
    f1    = f1_score(y_true, yhat, zero_division=0)
    f05   = fbeta_score(y_true, yhat, beta=0.5, zero_division=0)
    acc   = accuracy_score(y_true, yhat)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()
    ba    = balanced_accuracy(tp, tn, fp, fn)
    return {
        "ROC-AUC": roc,
        "PR-AUC":  pr,
        "Prec":    prec,
        "Rec":     rec,
        "F1":      f1,
        "F0_5":    f05,
        "Acc":     acc,
        "Bal_Acc": ba
    }
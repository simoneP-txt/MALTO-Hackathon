# ═══════════════════════════════════════════════════════════════
# submissions.py — Train top 5 configs on full train.csv, predict test.csv
# Generates submission2.csv → submission6.csv + specifics JSON
# ═══════════════════════════════════════════════════════════════

import os, random, warnings, json
from collections import Counter

import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")

SEED = 359956
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ═══════════════════════════════════════════════════════════════
# TOP 5 CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════
CONFIGS = [
    {
        "sub_num": 2,
        "trial": 1148,
        "f1_10fold": 0.9596,
        "svc_C": 6.9463,
        "tfidf_max_features": 7680,
        "tfidf_ngram_max": 1,
        "tfidf_sublinear": True,
        "tfidf_min_df": 3,
        "hgb_max_iter": 76,
        "hgb_max_depth": 11,
        "hgb_learning_rate": 0.0993,
        "hgb_max_leaf_nodes": 9,
        "hgb_min_samples_leaf": 6,
        "hgb_l2_regularization": 1.4705,
        "p5_impchi_k_per_class": 65,
    },
    {
        "sub_num": 3,
        "trial": 1178,
        "f1_10fold": 0.9573,
        "svc_C": 6.4141,
        "tfidf_max_features": 4863,
        "tfidf_ngram_max": 1,
        "tfidf_sublinear": True,
        "tfidf_min_df": 3,
        "hgb_max_iter": 59,
        "hgb_max_depth": 11,
        "hgb_learning_rate": 0.1403,
        "hgb_max_leaf_nodes": 7,
        "hgb_min_samples_leaf": 7,
        "hgb_l2_regularization": 1.2439,
        "p5_impchi_k_per_class": 65,
    },
    {
        "sub_num": 4,
        "trial": 1179,
        "f1_10fold": 0.9567,
        "svc_C": 6.4099,
        "tfidf_max_features": 5272,
        "tfidf_ngram_max": 1,
        "tfidf_sublinear": True,
        "tfidf_min_df": 3,
        "hgb_max_iter": 58,
        "hgb_max_depth": 11,
        "hgb_learning_rate": 0.1417,
        "hgb_max_leaf_nodes": 9,
        "hgb_min_samples_leaf": 8,
        "hgb_l2_regularization": 0.8649,
        "p5_impchi_k_per_class": 64,
    },
    {
        "sub_num": 5,
        "trial": 832,
        "f1_10fold": 0.9567,
        "svc_C": 11.6207,
        "tfidf_max_features": 3511,
        "tfidf_ngram_max": 1,
        "tfidf_sublinear": True,
        "tfidf_min_df": 3,
        "hgb_max_iter": 123,
        "hgb_max_depth": 10,
        "hgb_learning_rate": 0.0749,
        "hgb_max_leaf_nodes": 10,
        "hgb_min_samples_leaf": 4,
        "hgb_l2_regularization": 7.6574,
        "p5_impchi_k_per_class": 43,
    },
    {
        "sub_num": 6,
        "trial": 1199,
        "f1_10fold": 0.9566,
        "svc_C": 6.8716,
        "tfidf_max_features": 4931,
        "tfidf_ngram_max": 1,
        "tfidf_sublinear": True,
        "tfidf_min_df": 3,
        "hgb_max_iter": 69,
        "hgb_max_depth": 11,
        "hgb_learning_rate": 0.1741,
        "hgb_max_leaf_nodes": 9,
        "hgb_min_samples_leaf": 7,
        "hgb_l2_regularization": 0.9366,
        "p5_impchi_k_per_class": 65,
    },
]

# ═══════════════════════════════════════════════════════════════
# NLP SETUP
# ═══════════════════════════════════════════════════════════════
for res in ["punkt", "punkt_tab", "wordnet", "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng"]:
    nltk.download(res, quiet=True)

lemmatizer = WordNetLemmatizer()


def clean_text_none(text):
    text = str(text).lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def extract_stylometric(text):
    text = str(text)
    sentences = sent_tokenize(text)
    n_sentences = max(len(sentences), 1)
    sent_lengths = [len(word_tokenize(s)) for s in sentences]
    words = word_tokenize(text)
    alpha_words = [w for w in words if w.isalpha()]
    n_words = max(len(alpha_words), 1)
    word_lengths = [len(w) for w in alpha_words] if alpha_words else [0]
    n_chars = max(len(text), 1)

    n_commas = text.count(",")
    n_periods = text.count(".")
    n_exclaim = text.count("!")
    n_question = text.count("?")
    n_semicolon = text.count(";")
    n_colon = text.count(":")
    n_dash = text.count("\u2014") + text.count("-")
    n_parens = text.count("(") + text.count(")")
    n_quotes = text.count('"') + text.count("'") + text.count("\u201c") + text.count("\u201d")

    try:
        tags = pos_tag(alpha_words[:200])
        tag_counts = Counter(t for _, t in tags)
        n_tagged = max(sum(tag_counts.values()), 1)
        pct_noun = (tag_counts.get("NN", 0) + tag_counts.get("NNS", 0) +
                    tag_counts.get("NNP", 0) + tag_counts.get("NNPS", 0)) / n_tagged
        pct_verb = (tag_counts.get("VB", 0) + tag_counts.get("VBD", 0) +
                    tag_counts.get("VBG", 0) + tag_counts.get("VBN", 0) +
                    tag_counts.get("VBP", 0) + tag_counts.get("VBZ", 0)) / n_tagged
        pct_adj = (tag_counts.get("JJ", 0) + tag_counts.get("JJR", 0) +
                   tag_counts.get("JJS", 0)) / n_tagged
        pct_adv = (tag_counts.get("RB", 0) + tag_counts.get("RBR", 0) +
                   tag_counts.get("RBS", 0)) / n_tagged
    except:
        pct_noun = pct_verb = pct_adj = pct_adv = 0.0

    unique_words = set(w.lower() for w in alpha_words)
    ttr = len(unique_words) / n_words
    pct_short = sum(1 for w in alpha_words if len(w) <= 3) / n_words
    pct_long = sum(1 for w in alpha_words if len(w) >= 8) / n_words

    return {
        "n_chars": n_chars, "n_words": n_words, "n_sentences": n_sentences,
        "avg_word_len": np.mean(word_lengths), "std_word_len": np.std(word_lengths),
        "avg_sent_len": np.mean(sent_lengths),
        "std_sent_len": np.std(sent_lengths) if len(sent_lengths) > 1 else 0,
        "max_sent_len": max(sent_lengths), "min_sent_len": min(sent_lengths),
        "comma_rate": n_commas / n_words, "period_rate": n_periods / n_words,
        "exclaim_rate": n_exclaim / n_words, "question_rate": n_question / n_words,
        "semicolon_rate": n_semicolon / n_words, "colon_rate": n_colon / n_words,
        "dash_rate": n_dash / n_words, "paren_rate": n_parens / n_words,
        "quote_rate": n_quotes / n_words,
        "total_punct_rate": (n_commas + n_periods + n_exclaim + n_question +
                             n_semicolon + n_colon) / n_words,
        "pct_noun": pct_noun, "pct_verb": pct_verb,
        "pct_adj": pct_adj, "pct_adv": pct_adv,
        "noun_verb_ratio": pct_noun / max(pct_verb, 0.001),
        "ttr": ttr, "pct_short_words": pct_short, "pct_long_words": pct_long,
        "words_per_sentence": n_words / n_sentences,
        "chars_per_word": n_chars / n_words,
    }


# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════
print("Loading data …")
df_train = pd.read_csv("train.csv").dropna(subset=["TEXT", "LABEL"]).reset_index(drop=True)
df_test = pd.read_csv("test.csv")

print(f"Train: {len(df_train)} | Test: {len(df_test)}")

# ── Preprocess train ──────────────────────────────────────────
print("Preprocessing train …")
df_train["text_clean"] = df_train["TEXT"].apply(clean_text_none)
train_texts = df_train["text_clean"].values
train_raw = df_train["TEXT"].values
train_labels = df_train["LABEL"].values

print("Extracting train stylometric features …")
train_stylo = pd.DataFrame(df_train["TEXT"].apply(extract_stylometric).tolist()).values.astype(float)

# ── Preprocess test ───────────────────────────────────────────
print("Preprocessing test …")
df_test["text_clean"] = df_test["TEXT"].apply(clean_text_none)
test_texts = df_test["text_clean"].values
test_raw = df_test["TEXT"].values

print("Extracting test stylometric features …")
test_stylo = pd.DataFrame(df_test["TEXT"].apply(extract_stylometric).tolist()).values.astype(float)

# ── Test IDs ──────────────────────────────────────────────────
if df_test.columns[0] in ["", "Unnamed: 0"]:
    test_ids = df_test.iloc[:, 0].values
else:
    test_ids = df_test.index.values

os.makedirs("submissions", exist_ok=True)
print("Setup complete.\n")


# ═══════════════════════════════════════════════════════════════
# BUILD ImpCHI features
# ═══════════════════════════════════════════════════════════════
def build_impchi(X_tr_text, X_test_text, y_tr, k_per_class):
    tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2),
                            sublinear_tf=True, min_df=2,
                            strip_accents="unicode", analyzer="word")
    X_tr_full = tfidf.fit_transform(X_tr_text)
    X_test_full = tfidf.transform(X_test_text)

    classes = np.unique(y_tr)
    selected_indices = set()
    for cls in classes:
        y_binary = (y_tr == cls).astype(int)
        scores, _ = chi2(X_tr_full, y_binary)
        top_k_idx = np.argsort(scores)[-k_per_class:]
        selected_indices.update(top_k_idx)

    selected = sorted(selected_indices)
    return X_tr_full[:, selected], X_test_full[:, selected]


# ═══════════════════════════════════════════════════════════════
# LOOP OVER 5 CONFIGS
# ═══════════════════════════════════════════════════════════════
for cfg in CONFIGS:
    sub_num = cfg["sub_num"]
    print(f"{'='*60}")
    print(f"  SUBMISSION {sub_num} (Trial {cfg['trial']}, CV F1={cfg['f1_10fold']:.4f})")
    print(f"{'='*60}")

    # ── TF-IDF ────────────────────────────────────────────────
    tfidf = TfidfVectorizer(
        max_features=cfg["tfidf_max_features"],
        ngram_range=(1, cfg["tfidf_ngram_max"]),
        sublinear_tf=cfg["tfidf_sublinear"],
        min_df=cfg["tfidf_min_df"],
        strip_accents="unicode", analyzer="word"
    )
    X_tr_tfidf = tfidf.fit_transform(train_texts)
    X_te_tfidf = tfidf.transform(test_texts)

    def make_svc():
        return LinearSVC(C=cfg["svc_C"], max_iter=3000,
                         class_weight="balanced", random_state=SEED)

    # ── Phase 1: Human vs AI ─────────────────────────────────
    print("  Phase 1: Human vs AI …")
    m1 = make_svc()
    m1.fit(X_tr_tfidf, (train_labels != 0).astype(int))

    n_test = len(df_test)
    final_preds = np.full(n_test, -1, dtype=int)

    pred_p1 = m1.predict(X_te_tfidf)
    final_preds[pred_p1 == 0] = 0
    ai_idx = np.where(pred_p1 == 1)[0]
    print(f"    Human: {(pred_p1==0).sum()}, AI: {(pred_p1==1).sum()}")

    # ── Phase 2: Label 4 vs (1,2,3,5) ────────────────────────
    print("  Phase 2: Label 4 vs rest …")
    ai_mask = train_labels != 0
    m2 = make_svc()
    m2.fit(X_tr_tfidf[ai_mask], (train_labels[ai_mask] == 4).astype(int))

    pred_p2 = m2.predict(X_te_tfidf[ai_idx])
    is_4 = pred_p2 == 1
    final_preds[ai_idx[is_4]] = 4
    remaining = ai_idx[~is_4]
    print(f"    Label 4: {is_4.sum()}, remaining: {(~is_4).sum()}")

    # ── Phase 3: Label 3 vs (1,2,5) ──────────────────────────
    print("  Phase 3: Label 3 vs rest …")
    mask_1235 = np.isin(train_labels, [1, 2, 3, 5])
    m3 = make_svc()
    m3.fit(X_tr_tfidf[mask_1235], (train_labels[mask_1235] == 3).astype(int))

    pred_p3 = m3.predict(X_te_tfidf[remaining])
    is_3 = pred_p3 == 1
    final_preds[remaining[is_3]] = 3
    remaining = remaining[~is_3]
    print(f"    Label 3: {is_3.sum()}, remaining: {(~is_3).sum()}")

    # ── Phase 4: Label 5 vs (1,2) ────────────────────────────
    print("  Phase 4: Label 5 vs rest …")
    mask_125 = np.isin(train_labels, [1, 2, 5])
    m4 = make_svc()
    m4.fit(X_tr_tfidf[mask_125], (train_labels[mask_125] == 5).astype(int))

    pred_p4 = m4.predict(X_te_tfidf[remaining])
    is_5 = pred_p4 == 1
    final_preds[remaining[is_5]] = 5
    remaining = remaining[~is_5]
    print(f"    Label 5: {is_5.sum()}, remaining: {(~is_5).sum()}")

    # ── Phase 5: Label 1 vs 2 — ImpCHI + Stylo + HistGB ─────
    print("  Phase 5: Label 1 vs 2 (ImpCHI+Stylo → HistGB) …")

    mask_12 = np.isin(train_labels, [1, 2])
    tr_12_texts = train_texts[mask_12]
    tr_12_labels = train_labels[mask_12]

    # ImpCHI
    X_tr_chi, X_te_chi = build_impchi(
        tr_12_texts, test_texts[remaining],
        tr_12_labels, k_per_class=cfg["p5_impchi_k_per_class"])

    # Stylo
    scaler = StandardScaler()
    X_tr_stylo = csr_matrix(scaler.fit_transform(train_stylo[mask_12]))
    X_te_stylo = csr_matrix(scaler.transform(test_stylo[remaining]))

    # Concat
    X_tr_p5 = hstack([X_tr_chi, X_tr_stylo]).toarray()
    X_te_p5 = hstack([X_te_chi, X_te_stylo]).toarray()

    m5 = HistGradientBoostingClassifier(
        max_iter=cfg["hgb_max_iter"],
        max_depth=cfg["hgb_max_depth"],
        learning_rate=cfg["hgb_learning_rate"],
        max_leaf_nodes=cfg["hgb_max_leaf_nodes"],
        min_samples_leaf=cfg["hgb_min_samples_leaf"],
        l2_regularization=cfg["hgb_l2_regularization"],
        class_weight="balanced",
        random_state=SEED
    )
    m5.fit(X_tr_p5, tr_12_labels)
    pred_p5 = m5.predict(X_te_p5)
    final_preds[remaining] = pred_p5
    print(f"    Label 1: {(pred_p5==1).sum()}, Label 2: {(pred_p5==2).sum()}")

    # Fallback
    if (final_preds == -1).any():
        n_fallback = (final_preds == -1).sum()
        print(f"    WARNING: {n_fallback} unassigned, defaulting to 1")
        final_preds[final_preds == -1] = 1

    # ── Save submission CSV ───────────────────────────────────
    sub_df = pd.DataFrame({"ID": test_ids, "LABEL": final_preds})
    csv_path = f"submissions/submission{sub_num}.csv"
    sub_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    print(f"  Distribution: {dict(sorted(Counter(final_preds).items()))}")

    # ── Save specifics JSON ───────────────────────────────────
    specifics = {
        f"submission{sub_num}": {
            "pipeline": "5-phase hierarchical binary classification",
            "cv_f1_macro_10fold": cfg["f1_10fold"],
            "n_folds": 10,
            "optuna_trial": cfg["trial"],
            "phases_1_to_4": {
                "model": "LinearSVC",
                "params": {
                    "C": cfg["svc_C"],
                    "max_iter": 3000,
                    "class_weight": "balanced"
                },
                "tfidf": {
                    "max_features": cfg["tfidf_max_features"],
                    "ngram_range": [1, cfg["tfidf_ngram_max"]],
                    "sublinear_tf": cfg["tfidf_sublinear"],
                    "min_df": cfg["tfidf_min_df"]
                },
                "phase_1": "Human (0) vs AI (1,2,3,4,5)",
                "phase_2": "Label 4 vs (1,2,3,5)",
                "phase_3": "Label 3 vs (1,2,5)",
                "phase_4": "Label 5 vs (1,2)"
            },
            "phase_5": {
                "task": "Label 1 vs Label 2",
                "features": "ImpCHI (Improved Chi-squared, top-K per class) + stylometric (30 features)",
                "impchi_k_per_class": cfg["p5_impchi_k_per_class"],
                "model": "HistGradientBoostingClassifier",
                "params": {
                    "max_iter": cfg["hgb_max_iter"],
                    "max_depth": cfg["hgb_max_depth"],
                    "learning_rate": cfg["hgb_learning_rate"],
                    "max_leaf_nodes": cfg["hgb_max_leaf_nodes"],
                    "min_samples_leaf": cfg["hgb_min_samples_leaf"],
                    "l2_regularization": cfg["hgb_l2_regularization"],
                    "class_weight": "balanced"
                }
            }
        }
    }

    json_path = f"submissions/submission{sub_num}_specifics.json"
    with open(json_path, "w") as fp:
        json.dump(specifics, fp, indent=2)
    print(f"  Saved: {json_path}\n")

print("=" * 60)
print("  ALL SUBMISSIONS GENERATED!")
print("=" * 60)

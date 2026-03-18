"""
Creates submission spec files .json in submissions
"""

import json, os

OUTPUT_DIR = "submissions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ═══════════════════════════════════════════════════════════════
# # PART 1: submission7-9 from tuning/tuning_top5.json
# # ═══════════════════════════════════════════════════════════════
# INPUT_TUNING = "tuning/tuning_top5.json"
# START_SUB_TUNING = 7
# N_TUNING = 3

# if os.path.exists(INPUT_TUNING):
#     with open(INPUT_TUNING, "r") as fp:
#         top5_tuning = json.load(fp)
#     print(f"Loaded {len(top5_tuning)} configs from {INPUT_TUNING}")

#     for i in range(min(N_TUNING, len(top5_tuning))):
#         r = top5_tuning[i]
#         p = r["params"]
#         sub_num = START_SUB_TUNING + i
#         sub_key = f"submission{sub_num}"

#         spec = {
#             sub_key: {
#                 "pipeline": "5-phase hierarchical binary classification",
#                 "cv_f1_e2e_3fold": r.get("f1_e2e_3fold"),
#                 "cv_f1_p5_loocv": r.get("f1_p5_loocv"),
#                 "cv_f1_optuna": r.get("f1_optuna"),
#                 "n_folds": 3,
#                 "optuna_trial": r.get("trial"),
#                 "phases_1_to_4": {
#                     "model": "LinearSVC" if p["p14_model"] == "linearsvc" else "RandomForestClassifier",
#                     "params": {
#                         "C": p.get("svc_C"),
#                         "max_iter": 3000,
#                         "class_weight": p["p14_class_weight"],
#                     } if p["p14_model"] == "linearsvc" else {
#                         "n_estimators": p.get("rf_n_estimators"),
#                         "max_depth": p.get("rf_max_depth"),
#                         "min_samples_split": p.get("rf_min_samples_split"),
#                         "min_samples_leaf": p.get("rf_min_samples_leaf"),
#                         "class_weight": p["p14_class_weight"],
#                     },
#                     "tfidf": {
#                         "max_features": p["tfidf_max_features"],
#                         "ngram_range": [1, p["tfidf_ngram_max"]],
#                         "sublinear_tf": p["tfidf_sublinear"],
#                         "min_df": p["tfidf_min_df"],
#                     },
#                     "phase_1": "Human (0) vs AI (1,2,3,4,5)",
#                     "phase_2": "Label 4 vs (1,2,3,5)",
#                     "phase_3": "Label 3 vs (1,2,5)",
#                     "phase_4": "Label 5 vs (1,2)",
#                 },
#                 "phase_5": {
#                     "task": "Label 1 vs Label 2",
#                     "features": "ImpCHI + stylometric (30 features)",
#                     "impchi_k_per_class": p["impchi_k_per_class"],
#                     "impchi_max_features": p["impchi_max_features"],
#                     "impchi_ngram_max": p["impchi_ngram_max"],
#                     "impchi_min_df": p["impchi_min_df"],
#                     "model": "HistGradientBoostingClassifier",
#                     "params": {
#                         "max_iter": p["hgb_max_iter"],
#                         "max_depth": p["hgb_max_depth"],
#                         "learning_rate": p["hgb_learning_rate"],
#                         "max_leaf_nodes": p["hgb_max_leaf_nodes"],
#                         "min_samples_leaf": p["hgb_min_samples_leaf"],
#                         "l2_regularization": p["hgb_l2_regularization"],
#                         "class_weight": p["hgb_class_weight"],
#                     },
#                 },
#             }
#         }

#         fpath = f"{OUTPUT_DIR}/{sub_key}_specifics.json"
#         with open(fpath, "w") as fp:
#             json.dump(spec, fp, indent=2)
#         print(f"  Created: {fpath}")
# else:
#     print(f"  {INPUT_TUNING} not found, skipping submission7-9")

# ═══════════════════════════════════════════════════════════════
# PART 2: submission10-11 — CUSTOM
#   Phases 1-4: from submission4_specifics.json
#   Phase 5:    from phase5_tuning/top5_loocv.json
# ═══════════════════════════════════════════════════════════════
INPUT_SUB4 = "submissions/submission4_specifics.json"
INPUT_P5_LOOCV = "phase5_tuning/top5_loocv.json"
START_SUB_CUSTOM = 10
N_CUSTOM = 2

print(f"\n{'='*60}")
print("  CUSTOM SUBMISSIONS (phases 1-4 from sub4 + phase 5 from LOOCV)")
print(f"{'='*60}")

if not os.path.exists(INPUT_SUB4):
    print(f"  {INPUT_SUB4} not found, skipping")
elif not os.path.exists(INPUT_P5_LOOCV):
    print(f"  {INPUT_P5_LOOCV} not found, skipping")
else:
    with open(INPUT_SUB4, "r") as fp:
        sub4_raw = json.load(fp)
    sub4_key = list(sub4_raw.keys())[0]
    sub4 = sub4_raw[sub4_key]
    phases_14 = sub4["phases_1_to_4"]
    print(f"  Phases 1-4 from: {INPUT_SUB4}")

    with open(INPUT_P5_LOOCV, "r") as fp:
        loocv_top = json.load(fp)
    print(f"  Phase 5 from: {INPUT_P5_LOOCV} ({len(loocv_top)} configs)")

    for i in range(min(N_CUSTOM, len(loocv_top))):
        r = loocv_top[i]
        p = r["params"]
        sub_num = START_SUB_CUSTOM + i
        sub_key = f"submission{sub_num}"

        spec = {
            sub_key: {
                "pipeline": "5-phase hierarchical binary classification (CUSTOM)",
                "notes": "Phases 1-4 from submission4, Phase 5 from phase5_tuning LOOCV",
                "cv_f1_p5_loocv": r.get("f1_loocv"),
                "cv_f1_p5_10fold": r.get("f1_10fold"),
                "phase5_optuna_trial": r.get("trial"),
                "phases_1_to_4": phases_14,
                "phase_5": {
                    "task": "Label 1 vs Label 2",
                    "features": f"ImpCHI + stylometric ({p['stylo_subset']})",
                    "stylo_subset": p["stylo_subset"],
                    "impchi_k_per_class": p["impchi_k"],
                    "impchi_max_features": p["impchi_max_feat"],
                    "impchi_ngram_max": p["impchi_ngram_max"],
                    "impchi_min_df": p["impchi_min_df"],
                    "impchi_sublinear": p.get("impchi_sublinear", True),
                    "model": "HistGradientBoostingClassifier",
                    "params": {
                        "max_iter": p["hgb_max_iter"],
                        "max_depth": p["hgb_max_depth"],
                        "learning_rate": p["hgb_lr"],
                        "max_leaf_nodes": p["hgb_max_leaf_nodes"],
                        "min_samples_leaf": p["hgb_min_samples_leaf"],
                        "l2_regularization": p["hgb_l2"],
                        "class_weight": p["hgb_class_weight"],
                    },
                },
            }
        }

        fpath = f"{OUTPUT_DIR}/{sub_key}_specifics.json"
        with open(fpath, "w") as fp:
            json.dump(spec, fp, indent=2)
        print(f"  Created: {fpath}")
        print(f"    Phase 5 trial {r.get('trial')}, LOOCV F1={r.get('f1_loocv', 'N/A'):.4f}, "
              f"stylo={p['stylo_subset']}")

print(f"\nDone!")

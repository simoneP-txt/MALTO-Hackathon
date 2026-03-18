"""
Creates submissions\submission*_specifics.json
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

# # ═══════════════════════════════════════════════════════════════
# # PART 2: submission10-11 — CUSTOM
# # ═══════════════════════════════════════════════════════════════
# INPUT_SUB4 = "submissions/submission4_specifics.json"
# INPUT_P5_LOOCV = "phase5_tuning/top5_loocv.json"
# START_SUB_CUSTOM = 10
# N_CUSTOM = 2

# print(f"\n{'='*60}")
# print(" CUSTOM SUBMISSIONS")
# print(f"{'='*60}")

# if not os.path.exists(INPUT_SUB4):
#     print(f"  {INPUT_SUB4} not found, skipping")
# elif not os.path.exists(INPUT_P5_LOOCV):
#     print(f"  {INPUT_P5_LOOCV} not found, skipping")
# else:
#     with open(INPUT_SUB4, "r") as fp:
#         sub4_raw = json.load(fp)
#     sub4_key = list(sub4_raw.keys())[0]
#     sub4 = sub4_raw[sub4_key]
#     phases_14 = sub4["phases_1_to_4"]
#     print(f"  Phases 1-4 from: {INPUT_SUB4}")

#     with open(INPUT_P5_LOOCV, "r") as fp:
#         loocv_top = json.load(fp)
#     print(f"  Phase 5 from: {INPUT_P5_LOOCV} ({len(loocv_top)} configs)")

#     for i in range(min(N_CUSTOM, len(loocv_top))):
#         r = loocv_top[i]
#         p = r["params"]
#         sub_num = START_SUB_CUSTOM + i
#         sub_key = f"submission{sub_num}"

#         spec = {
#             sub_key: {
#                 "pipeline": "5-phase hierarchical binary classification (CUSTOM)",
#                 "notes": "Phases 1-4 from submission4, Phase 5 from phase5_tuning LOOCV",
#                 "cv_f1_p5_loocv": r.get("f1_loocv"),
#                 "cv_f1_p5_10fold": r.get("f1_10fold"),
#                 "phase5_optuna_trial": r.get("trial"),
#                 "phases_1_to_4": phases_14,
#                 "phase_5": {
#                     "task": "Label 1 vs Label 2",
#                     "features": f"ImpCHI + stylometric ({p['stylo_subset']})",
#                     "stylo_subset": p["stylo_subset"],
#                     "impchi_k_per_class": p["impchi_k"],
#                     "impchi_max_features": p["impchi_max_feat"],
#                     "impchi_ngram_max": p["impchi_ngram_max"],
#                     "impchi_min_df": p["impchi_min_df"],
#                     "impchi_sublinear": p.get("impchi_sublinear", True),
#                     "model": "HistGradientBoostingClassifier",
#                     "params": {
#                         "max_iter": p["hgb_max_iter"],
#                         "max_depth": p["hgb_max_depth"],
#                         "learning_rate": p["hgb_lr"],
#                         "max_leaf_nodes": p["hgb_max_leaf_nodes"],
#                         "min_samples_leaf": p["hgb_min_samples_leaf"],
#                         "l2_regularization": p["hgb_l2"],
#                         "class_weight": p["hgb_class_weight"],
#                     },
#                 },
#             }
#         }

#         fpath = f"{OUTPUT_DIR}/{sub_key}_specifics.json"
#         with open(fpath, "w") as fp:
#             json.dump(spec, fp, indent=2)
#         print(f"  Created: {fpath}")
#         print(f"    Phase 5 trial {r.get('trial')}, LOOCV F1={r.get('f1_loocv', 'N/A'):.4f}, "
#               f"stylo={p['stylo_subset']}")

# print(f"\nDone!")


# # ═══════════════════════════════════════════════════════════════
# # PART 3: submission12-14 from tuning/tuning_top5.json  (top 3)
# #         submission15-18 from tuning/tuning_top5_2.json (top 4)
# # ═══════════════════════════════════════════════════════════════

# def build_spec_from_tuning(rank_entry, sub_num):
#     """Convert an Optuna tuning result into submission*_specifics.json format."""
#     p = rank_entry["params"]
#     sub_key = f"submission{sub_num}"

#     mode = p.get("p14_mode", "shared")
#     voting_p14 = p.get("p14_voting", False)

#     # ── Phases 1-4 params ─────────────────────────────────────
#     params_14 = {"max_iter": 3000, "class_weight": "balanced"}

#     if mode == "per_phase":
#         for i in range(1, 5):
#             params_14[f"C_p{i}"] = p[f"svc_C_p{i}"]
#     else:
#         params_14["C"] = p["svc_C"]

#     if voting_p14:
#         for view in ["v2", "v3"]:
#             shared_key = f"svc_C_{view}"
#             per_phase_key = f"svc_C_{view}_p1"
#             if per_phase_key in p:
#                 for i in range(1, 5):
#                     params_14[f"C_{view}_p{i}"] = p[f"svc_C_{view}_p{i}"]
#             elif shared_key in p:
#                 params_14[f"C_{view}"] = p[shared_key]

#     phases_14 = {
#         "model": "LinearSVC",
#         "mode": mode,
#         "voting": voting_p14,
#         "params": params_14,
#         "tfidf": {
#             "max_features": p["tfidf_max_feat"],
#             "ngram_range": [1, p["tfidf_ngram_max"]],
#             "sublinear_tf": p["tfidf_sublinear"],
#             "min_df": p["tfidf_min_df"],
#         },
#         "phase_1": "Human (0) vs AI (1,2,3,4,5)",
#         "phase_2": "Label 4 vs (1,2,3,5)",
#         "phase_3": "Label 3 vs (1,2,5)",
#         "phase_4": "Label 5 vs (1,2)",
#     }

#     # ── Phase 5 ───────────────────────────────────────────────
#     model_type = p.get("p5_model_type", "histgb")
#     model_name = (
#         "HistGradientBoostingClassifier" if model_type == "histgb"
#         else "LGBMClassifier"
#     )
#     p5_cw = p.get("p5_class_weight", "balanced")
#     voting_p5 = p.get("p5_voting", False)
#     stylo_sub = p.get("p5_stylo_subset", "all_30")

#     p5_params = {
#         "max_iter": p["p5_max_iter"],
#         "max_depth": p["p5_max_depth"],
#         "learning_rate": p["p5_lr"],
#         "max_leaf_nodes": p["p5_max_leaf_nodes"],
#         "min_samples_leaf": p["p5_min_samples_leaf"],
#         "l2_regularization": p["p5_l2"],
#         "class_weight": p5_cw,
#     }

#     phase5 = {
#         "task": "Label 1 vs Label 2",
#         "model": model_name,
#         "voting": voting_p5,
#         "features": f"ImpCHI + stylometric ({stylo_sub})",
#         "stylo_subset": stylo_sub,
#         "impchi_k_per_class": p.get("p5_impchi_k", 64),
#         "impchi_max_features": p.get("p5_impchi_max_feat", 15000),
#         "impchi_ngram_range": [
#             p.get("p5_impchi_ngram_min", 1),
#             p.get("p5_impchi_ngram_max", 2),
#         ],
#         "impchi_min_df": p.get("p5_impchi_min_df", 2),
#         "impchi_sublinear": p.get("p5_impchi_sublinear", True),
#         "params": p5_params,
#     }

#     if voting_p5:
#         phase5["v2_params"] = {
#             "learning_rate": p["p5_v2_lr"],
#             "l2_regularization": p["p5_v2_l2"],
#         }
#         phase5["v3_params"] = {
#             "learning_rate": p["p5_v3_lr"],
#             "l2_regularization": p["p5_v3_l2"],
#         }

#     spec = {
#         sub_key: {
#             "pipeline": "5-phase hierarchical binary classification",
#             "cv_f1_optuna": rank_entry["f1_mean"],
#             "optuna_trial": rank_entry["trial"],
#             "phases_1_to_4": phases_14,
#             "phase_5": phase5,
#         }
#     }
#     return spec


# INPUT_TOP5_1 = "tuning/tuning_top5.json"
# INPUT_TOP5_2 = "tuning/tuning_top5_2.json"

# sub_num = 12  # starting submission number

# print(f"\n{'='*60}")
# print(" SUBMISSIONS 12-18 (from tuning results)")
# print(f"{'='*60}")

# # ── Top 3 from tuning_top5.json -> submissions 12, 13, 14 ─────
# if os.path.exists(INPUT_TOP5_1):
#     with open(INPUT_TOP5_1, "r") as fp:
#         top5_1 = json.load(fp)
#     print(f"\nLoaded {len(top5_1)} configs from {INPUT_TOP5_1}")

#     for i in range(min(3, len(top5_1))):
#         r = top5_1[i]
#         spec = build_spec_from_tuning(r, sub_num)
#         fpath = f"{OUTPUT_DIR}/submission{sub_num}_specifics.json"
#         with open(fpath, "w") as fp:
#             json.dump(spec, fp, indent=2)
#         print(f"  Created: {fpath}  (trial {r['trial']}, F1={r['f1_mean']:.6f})")
#         sub_num += 1
# else:
#     print(f"  {INPUT_TOP5_1} not found, skipping")

# # ── Top 4 from tuning_top5_2.json -> submissions 15, 16, 17, 18
# if os.path.exists(INPUT_TOP5_2):
#     with open(INPUT_TOP5_2, "r") as fp:
#         top5_2 = json.load(fp)
#     print(f"\nLoaded {len(top5_2)} configs from {INPUT_TOP5_2}")

#     for i in range(min(4, len(top5_2))):
#         r = top5_2[i]
#         spec = build_spec_from_tuning(r, sub_num)
#         fpath = f"{OUTPUT_DIR}/submission{sub_num}_specifics.json"
#         with open(fpath, "w") as fp:
#             json.dump(spec, fp, indent=2)
#         print(f"  Created: {fpath}  (trial {r['trial']}, F1={r['f1_mean']:.6f})")
#         sub_num += 1
# else:
#     print(f"  {INPUT_TOP5_2} not found, skipping")

# print(f"\nDone! Created submissions 12 to {sub_num - 1}")


# ═══════════════════════════════════════════════════════════════
# PART 4: Ensemble submissions
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(" ENSEMBLE SUBMISSIONS")
print(f"{'='*60}")

# submission19: vote sub2 + sub4, if phases 1-4 disagree -> sub4 wins
spec19 = {
    "submission19": {
        "pipeline": "ensemble",
        "ensemble_type": "priority_vote",
        "priority": "submission4",
        "members": ["submission2", "submission4"],
    }
}
fpath = f"{OUTPUT_DIR}/submission19_specifics.json"
with open(fpath, "w") as fp:
    json.dump(spec19, fp, indent=2)
print(f"  Created: {fpath}  (sub2+sub4, priority=sub4)")

# submission20: majority vote sub2 + sub4 + sub5
spec20 = {
    "submission20": {
        "pipeline": "ensemble",
        "ensemble_type": "majority_vote",
        "members": ["submission2", "submission4", "submission5"],
    }
}
fpath = f"{OUTPUT_DIR}/submission20_specifics.json"
with open(fpath, "w") as fp:
    json.dump(spec20, fp, indent=2)
print(f"  Created: {fpath}  (sub2+sub4+sub5, majority)")

print(f"\nDone!")

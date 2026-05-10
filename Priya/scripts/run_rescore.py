"""Re-score Mistral and Gemma on StereoSet/CrowS-Pairs with the text-based fallback fix."""

import sys
import os
import time
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model_scoring import BiasEvaluatorModel, score_stereoset, score_crows
from data_preprocessing import load_all_preprocessed
from evaluation_pipeline import (
    ensure_dir, summarize_all, demographic_disparity_tests,
    intersectional_analysis, cross_model_tests,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

MODELS_TO_RESCORE = {
    "mistralai/Mistral-7B-Instruct-v0.2": "mistral-7b",
    "google/gemma-2-9b-it": "gemma2-9b",
}
BENCHMARKS_TO_RESCORE = ["stereoset", "crows_pairs"]


def run():
    print("Loading datasets...", flush=True)
    data = load_all_preprocessed()

    for model_hf, model_label in MODELS_TO_RESCORE.items():
        print(f"\n{'='*60}", flush=True)
        print(f"RE-SCORING: {model_label} ({model_hf})", flush=True)
        print(f"{'='*60}", flush=True)

        results_path = os.path.join(OUTPUT_DIR, model_label, "item_level_results.jsonl")
        df = pd.read_json(results_path, lines=True)
        print(f"  Loaded {len(df)} existing results", flush=True)

        model = BiasEvaluatorModel(model_hf)

        for bench in BENCHMARKS_TO_RESCORE:
            bench_mask = df["benchmark"] == bench
            bench_items = df[bench_mask].copy()
            item_ids = set(bench_items["item_id"].astype(str))

            bench_data = data[bench]
            bench_data["_id_str"] = bench_data["item_id"].astype(str)
            to_score = bench_data[bench_data["_id_str"].isin(item_ids)].drop(columns=["_id_str"])

            print(f"\n  >>> {bench}: re-scoring {len(to_score)} items", flush=True)

            new_rows = []
            for idx, (_, row) in enumerate(to_score.iterrows()):
                rowd = row.to_dict()
                t1 = time.time()
                try:
                    if bench == "stereoset":
                        scores = score_stereoset(model, rowd, "baseline")
                    else:
                        scores = score_crows(model, rowd, "baseline")

                    new_rows.append({
                        "item_id": str(rowd["item_id"]),
                        **scores,
                    })
                    print(f"    [{idx+1}/{len(to_score)}] scored in {time.time()-t1:.1f}s", flush=True)
                except Exception as e:
                    print(f"    [{idx+1}/{len(to_score)}] ERROR: {str(e)[:80]}", flush=True)
                    time.sleep(1)

            new_scores_df = pd.DataFrame(new_rows)
            new_scores_df["item_id"] = new_scores_df["item_id"].astype(str)
            bench_items = bench_items.copy()
            bench_items["item_id"] = bench_items["item_id"].astype(str)

            score_cols = [c for c in new_scores_df.columns if c != "item_id"]
            bench_items = bench_items.drop(columns=score_cols, errors="ignore")
            bench_items = bench_items.merge(new_scores_df, on="item_id", how="left")

            df = pd.concat([df[~bench_mask], bench_items], ignore_index=True)
            print(f"    Updated {len(new_rows)} rows", flush=True)

        df.to_json(results_path, orient="records", lines=True)
        print(f"\n  Saved {len(df)} total results", flush=True)

        model_dir = os.path.join(OUTPUT_DIR, model_label)
        summarize_all(df).to_csv(os.path.join(model_dir, "summary_metrics.csv"), index=False)
        demographic_disparity_tests(df).to_csv(os.path.join(model_dir, "demographic_disparity_tests.csv"), index=False)
        intersectional_analysis(df).to_csv(os.path.join(model_dir, "intersectional_analysis.csv"), index=False)
        print(f"  Recomputed summaries for {model_label}", flush=True)

    print("\nRecomputing combined outputs...", flush=True)
    all_dfs = []
    for label in ["mistral-7b", "llama3-8b", "gemma2-9b"]:
        p = os.path.join(OUTPUT_DIR, label, "item_level_results.jsonl")
        all_dfs.append(pd.read_json(p, lines=True))
    combined = pd.concat(all_dfs, ignore_index=True)

    combined_dir = os.path.join(OUTPUT_DIR, "combined")
    combined.to_json(os.path.join(combined_dir, "all_item_level_results.jsonl"), orient="records", lines=True)
    cross_model_tests(combined).to_csv(os.path.join(combined_dir, "cross_model_tests.csv"), index=False)
    summarize_all(combined).to_csv(os.path.join(combined_dir, "summary_all_models.csv"), index=False)
    print("Saved combined outputs", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("DONE — Re-scoring complete", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    run()

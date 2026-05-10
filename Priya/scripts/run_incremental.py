"""
Incremental Evaluation - Add More Samples to Existing Results
============================================================

Add 15 more samples per dataset per model to existing evaluation outputs.

This script:
1. Loads existing item_level_results.jsonl for each model
2. Samples 15 NEW items (excluding already-scored IDs) from each benchmark
3. Scores the new items using the same methodology as the original evaluation
4. Appends results to the existing JSONL files
5. Recomputes all summary statistics and cross-model comparisons

Usage: python scripts/run_incremental.py

Prerequisites: Must have existing evaluation results in outputs/ directory
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd

# Add the src directory to Python path for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import DECODING_CONFIGS, RANDOM_SEED
from data_preprocessing import load_all_preprocessed
from model_scoring import BiasEvaluatorModel, DecodeConfig, score_stereoset, score_crows, score_bbq
from evaluation_pipeline import (
    ensure_dir, summarize_all, demographic_disparity_tests,
    intersectional_analysis, cross_model_tests,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Number of additional samples to collect per dataset per model
NEW_PER_DATASET = 15

# Models to run incremental evaluation on (must match existing results)
MODELS_TO_RUN = {
    "mistralai/Mistral-7B-Instruct-v0.2": "mistral-7b",     # Mistral 7B model
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3-8b",     # LLaMA 3 8B model
    "google/gemma-2-9b-it": "gemma2-9b",                    # Gemma 2 9B model
}

# Evaluation configuration (should match original evaluation settings)
MITIGATIONS_TO_RUN = ["baseline"]        # Bias mitigation strategies
DECODINGS_TO_RUN = ["deterministic"]     # Decoding methods

# Output directory (updated path structure)
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_existing_results(model_label):
    """
    Load existing evaluation results for a specific model.
    
    Args:
        model_label (str): Model identifier (e.g., 'mistral-7b')
        
    Returns:
        pd.DataFrame: Existing results or empty DataFrame if no results exist
    """
    path = os.path.join(BASE_OUTPUT_DIR, model_label, "item_level_results.jsonl")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def get_scored_ids(existing_df, bench_name):
    """
    Extract item IDs that have already been scored for a specific benchmark.
    
    Args:
        existing_df (pd.DataFrame): Existing evaluation results
        bench_name (str): Benchmark name ('stereoset', 'crows_pairs', or 'bbq')
        
    Returns:
        set: Set of item IDs that have already been evaluated
    """
    if existing_df.empty:
        return set()
    bench_rows = existing_df[existing_df["benchmark"] == bench_name]
    return set(bench_rows["item_id"].astype(str).tolist())


def sample_new(full_df, scored_ids, n, seed):
    """
    Sample new items that haven't been scored yet.
    
    Args:
        full_df (pd.DataFrame): Complete dataset for the benchmark
        scored_ids (set): Set of item IDs that have already been scored
        n (int): Number of new samples to collect
        seed (int): Random seed for reproducible sampling
        
    Returns:
        pd.DataFrame: New items to score (up to n items)
    """
    full_df = full_df.copy()
    full_df["_id_str"] = full_df["item_id"].astype(str)  # Convert IDs to strings for matching
    # Filter out already-scored items
    available = full_df[~full_df["_id_str"].isin(scored_ids)].drop(columns=["_id_str"])
    
    if len(available) == 0:
        return pd.DataFrame()  # No new items available
    if len(available) <= n:
        return available       # Return all available if fewer than requested
    return available.sample(n=n, random_state=seed)  # Random sample


def score_items(model, mitigation, decoding_name, bench_name, df):
    """
    Score new evaluation items using the same methodology as original evaluation.
    
    This function is identical to the scoring function in run_sample.py to ensure
    consistency between original and incremental evaluations.
    
    Args:
        model (BiasEvaluatorModel): Model instance for evaluation
        mitigation (str): Bias mitigation strategy
        decoding_name (str): Decoding configuration name
        bench_name (str): Benchmark name
        df (pd.DataFrame): Items to score
        
    Returns:
        list: Results for each scored item
    """
    decode_cfg = DecodeConfig(**DECODING_CONFIGS[decoding_name])
    results = []

    for idx, (_, row) in enumerate(df.iterrows()):
        rowd = row.to_dict()
        t1 = time.time()  # Start timing

        # Base metadata for all benchmarks
        base = {
            "model": model.model_name,
            "mitigation": mitigation,
            "decoding": decoding_name,
            "benchmark": bench_name,
            "item_id": rowd["item_id"],
            "bias_type": rowd.get("bias_type", "unknown"),
            "intersection_group": rowd.get("intersection_group", "unknown"),
        }

        try:
            # Route to appropriate scoring function based on benchmark
            if bench_name == "stereoset":
                scores = score_stereoset(model, rowd, mitigation)
                results.append({**base, **scores})
            elif bench_name == "crows_pairs":
                scores = score_crows(model, rowd, mitigation)
                results.append({**base, **scores})
            elif bench_name == "bbq":
                scores = score_bbq(model, rowd, mitigation, decode_cfg)
                # BBQ requires additional metadata
                results.append({
                    **base,
                    "category": rowd.get("category"),
                    "context_condition": rowd.get("context_condition"),
                    "gold_label": rowd.get("gold_label"),
                    "unknown_label": rowd.get("unknown_label"),
                    "stereotyped_label": rowd.get("stereotyped_label"),
                    **scores,
                })

            # Log progress with timing
            elapsed = time.time() - t1
            print(f"    [{idx+1}/{len(df)}] scored in {elapsed:.1f}s", flush=True)
        except Exception as e:
            # Log errors but continue with remaining items
            print(f"    [{idx+1}/{len(df)}] ERROR: {str(e)[:80]}", flush=True)
            time.sleep(1)  # Brief pause on error

    return results


def save_model_outputs(model_label, full_df):
    """
    Save updated model results and recompute all statistical analyses.
    
    This function overwrites the existing results files with the expanded dataset
    and regenerates all derived analyses to reflect the new sample size.
    
    Args:
        model_label (str): Model identifier for output directory
        full_df (pd.DataFrame): Complete results including original + new items
    """
    model_dir = os.path.join(BASE_OUTPUT_DIR, model_label)
    ensure_dir(model_dir)

    # Save updated raw results (overwrites existing file)
    full_df.to_json(
        os.path.join(model_dir, "item_level_results.jsonl"),
        orient="records", lines=True,
    )
    print(f"  Saved item_level_results.jsonl ({len(full_df)} rows)", flush=True)

    # Regenerate all statistical summaries with expanded data
    summary_df = summarize_all(full_df)
    summary_df.to_csv(os.path.join(model_dir, "summary_metrics.csv"), index=False)
    print(f"  Saved summary_metrics.csv ({len(summary_df)} rows)", flush=True)

    # Recompute demographic disparity tests with larger sample
    disparity_df = demographic_disparity_tests(full_df)
    disparity_df.to_csv(os.path.join(model_dir, "demographic_disparity_tests.csv"), index=False)
    print(f"  Saved demographic_disparity_tests.csv ({len(disparity_df)} rows)", flush=True)

    # Recompute intersectional analysis with expanded data
    intersection_df = intersectional_analysis(full_df)
    intersection_df.to_csv(os.path.join(model_dir, "intersectional_analysis.csv"), index=False)
    print(f"  Saved intersectional_analysis.csv ({len(intersection_df)} rows)", flush=True)


# ============================================================================
# MAIN INCREMENTAL EVALUATION PIPELINE
# ============================================================================

def run():
    """
    Execute incremental evaluation pipeline to expand existing results.
    
    This pipeline:
    1. Loads existing results and full datasets
    2. Identifies items not yet scored for each model/benchmark combination
    3. Samples additional items (avoiding duplicates)
    4. Scores new items using existing methodology
    5. Merges with existing results and recomputes all analyses
    """
    print("=" * 60, flush=True)
    print("INCREMENTAL RUN — Adding 15 new samples per dataset per model", flush=True)
    print("=" * 60, flush=True)

    # Load complete datasets (same as original evaluation)
    print("\nLoading datasets...", flush=True)
    data = load_all_preprocessed()
    for name, df in data.items():
        print(f"  {name}: {len(df)} total items available", flush=True)

    all_model_dfs = []  # Store final results for cross-model analysis
    seed_offset = 0     # Ensure different random seeds for each model

    # Process each model individually
    for model_[HF_TOKEN_REDACTED], model_label in MODELS_TO_RUN.items():
        print(f"\n{'=' * 60}", flush=True)
        print(f"MODEL: {model_label} ({model_[HF_TOKEN_REDACTED]})", flush=True)
        print(f"{'=' * 60}", flush=True)

        # Load existing results for this model
        existing_df = load_existing_results(model_label)
        print(f"  Existing results: {len(existing_df)} rows", flush=True)

        # Identify new items to score for each benchmark
        new_samples = {}
        for bench_name, full_df in data.items():
            scored_ids = get_scored_ids(existing_df, bench_name)  # Already scored items
            seed_offset += 1  # Different seed for each benchmark/model combination
            new_df = sample_new(full_df, scored_ids, NEW_PER_DATASET, RANDOM_SEED + seed_offset)
            new_samples[bench_name] = new_df
            print(f"  {bench_name}: {len(scored_ids)} already scored, {len(new_df)} new to score", flush=True)

        # Check if any new items need scoring
        total_new = sum(len(df) for df in new_samples.values())
        if total_new == 0:
            print("  No new items to score, skipping.", flush=True)
            all_model_dfs.append(existing_df)  # Use existing results as-is
            continue

        # Initialize model for scoring new items
        model = BiasEvaluatorModel(model_[HF_TOKEN_REDACTED])
        new_results = []

        # Score new items using all configured mitigation/decoding combinations
        for mitigation in MITIGATIONS_TO_RUN:
            for decoding_name in DECODINGS_TO_RUN:
                for bench_name, df in new_samples.items():
                    if len(df) == 0:  # Skip benchmarks with no new items
                        continue
                    print(f"\n  >>> {bench_name} | {mitigation} | {decoding_name} ({len(df)} new items)", flush=True)
                    # Score new items using same methodology as original evaluation
                    results = score_items(model, mitigation, decoding_name, bench_name, df)
                    new_results.extend(results)

        # Merge new results with existing results
        new_df = pd.DataFrame(new_results)
        merged_df = pd.concat([existing_df, new_df], ignore_index=True)
        all_model_dfs.append(merged_df)

        # Save updated results and regenerate all analyses
        print(f"\n  Saving {model_label} outputs ({len(existing_df)} old + {len(new_df)} new = {len(merged_df)} total)...", flush=True)
        save_model_outputs(model_label, merged_df)

    # Regenerate cross-model analysis with expanded datasets
    if len(all_model_dfs) > 1:
        print(f"\n{'=' * 60}", flush=True)
        print("UPDATING CROSS-MODEL ANALYSIS", flush=True)
        print(f"{'=' * 60}", flush=True)
        
        combined_dir = os.path.join(BASE_OUTPUT_DIR, "combined")
        ensure_dir(combined_dir)
        combined_df = pd.concat(all_model_dfs, ignore_index=True)

        # Save combined raw results
        combined_df.to_json(
            os.path.join(combined_dir, "all_item_level_results.jsonl"),
            orient="records", lines=True,
        )

        # Recompute cross-model statistical comparisons with expanded data
        cross_df = cross_model_tests(combined_df)
        cross_df.to_csv(os.path.join(combined_dir, "cross_model_tests.csv"), index=False)
        print(f"\nSaved combined cross_model_tests.csv ({len(cross_df)} rows)", flush=True)

        # Recompute overall summary with expanded sample sizes
        summary_df = summarize_all(combined_df)
        summary_df.to_csv(os.path.join(combined_dir, "summary_all_models.csv"), index=False)
        print(f"Saved combined summary_all_models.csv ({len(summary_df)} rows)", flush=True)

    # Display final summary
    print("\n" + "=" * 60, flush=True)
    print("DONE — Updated results saved to outputs/", flush=True)
    for _, label in MODELS_TO_RUN.items():
        existing = load_existing_results(label)
        print(f"  outputs/{label}/ ({len(existing)} total items)", flush=True)
    print(f"  outputs/combined/", flush=True)
    print("=" * 60, flush=True)


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Entry point for incremental evaluation.
    
    This script extends existing evaluation results by adding more samples
    from each benchmark while avoiding duplication. Useful for:
    - Increasing statistical power of existing analyses
    - Adding more data points without re-running full evaluation
    - Gradual expansion of evaluation scope
    
    Usage:
        python scripts/run_incremental.py
        
    Prerequisites:
        - Existing evaluation results in outputs/ directory
        - Same model access and configuration as original evaluation
    """
    run()

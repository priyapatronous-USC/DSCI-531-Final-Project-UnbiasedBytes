"""
Bias Evaluation Pipeline for Large Language Models (LLMs)
========================================================

This script runs a comprehensive bias evaluation pipeline that tests multiple LLMs
for demographic bias using three established bias evaluation benchmarks:
- StereoSet: Tests stereotype preferences in sentence completion
- CrowS-Pairs: Tests stereotype preferences in minimal pair sentences  
- BBQ (Bias Benchmark for QA): Tests bias in question-answering scenarios

The pipeline evaluates models across different demographic groups (gender, race)
and produces detailed statistical analysis including disparity tests and 
intersectional bias analysis.

Output structure:
    bias_eval/outputs/
        mistral-7b/       — per-model item results + summary + disparity + intersectional
        llama3-8b/
        gemma2-9b/
        combined/         — cross-model comparison tests

Usage:
    python run_sample.py

Dependencies:
    - HuggingFace Transformers library for model access
    - Statistical packages (numpy, pandas, scipy) for analysis
    - Custom bias_eval module with preprocessing and scoring functions
"""

import sys
import os
import time
import numpy as np
import pandas as pd

# Add the src directory to Python path so we can import custom modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Import configuration settings and core pipeline functions
from config import DECODING_CONFIGS, RANDOM_SEED
from data_preprocessing import load_all_preprocessed
from model_scoring import BiasEvaluatorModel, DecodeConfig, score_stereoset, score_crows, score_bbq
from evaluation_pipeline import (
    ensure_dir, summarize_all, demographic_disparity_tests,
    intersectional_analysis, cross_model_tests,
)

# Set random seed for reproducible sampling and analysis
np.random.seed(RANDOM_SEED)

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Sample sizes for each benchmark (reduced for faster testing)
# These are smaller subsets of the full datasets to enable quick evaluation
SAMPLE_SIZES = {
    "stereoset": 15,      # StereoSet stereotype preference test items
    "crows_pairs": 15,    # CrowS-Pairs minimal pair sentences  
    "bbq": 20,            # BBQ question-answering bias test items
}

# Dictionary mapping HuggingFace model identifiers to shorter labels for output
# These are the three LLMs we'll evaluate for bias
MODELS_TO_RUN = {
    "mistralai/Mistral-7B-Instruct-v0.2": "mistral-7b",      # Mistral AI's 7B parameter model
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3-8b",      # Meta's LLaMA 3 8B parameter model  
    "google/gemma-2-9b-it": "gemma2-9b",                     # Google's Gemma 2 9B parameter model
}

# Bias mitigation strategies to test (currently only baseline/no mitigation)
# Future versions could include prompt-based debiasing techniques
MITIGATIONS_TO_RUN = ["baseline"]

# Decoding strategies for text generation (deterministic = temperature 0)
# Deterministic decoding ensures reproducible results across runs
DECODINGS_TO_RUN = ["deterministic"]

# Base directory where all evaluation results will be saved
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "sample_run")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def sample_df(df, n, seed=RANDOM_SEED):
    """
    Randomly sample n rows from a DataFrame for testing purposes.
    
    This function creates smaller, manageable datasets for quicker evaluation
    while maintaining randomness for representative sampling.
    
    Args:
        df (pd.DataFrame): The input DataFrame to sample from
        n (int): Number of rows to sample
        seed (int): Random seed for reproducible sampling
        
    Returns:
        pd.DataFrame: Sampled DataFrame with at most n rows
        
    Note:
        If the DataFrame has fewer than n rows, returns the entire DataFrame
    """
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def score_items(model, mitigation, decoding_name, bench_name, df):
    """
    Score a batch of bias evaluation items using a specified model and configuration.
    
    This is the core evaluation function that processes individual test items through
    the model and collects bias metrics. Each item is scored according to the 
    specific benchmark protocol (StereoSet, CrowS-Pairs, or BBQ).
    
    Args:
        model (BiasEvaluatorModel): The language model to evaluate
        mitigation (str): Bias mitigation strategy (e.g., 'baseline', 'neutral_framing')
        decoding_name (str): Decoding configuration name (e.g., 'deterministic') 
        bench_name (str): Benchmark name ('stereoset', 'crows_pairs', or 'bbq')
        df (pd.DataFrame): DataFrame containing test items to score
        
    Returns:
        list: List of dictionaries containing results for each item, including:
            - Basic metadata (model, mitigation, decoding, benchmark, item_id)
            - Demographic information (bias_type, intersection_group)
            - Benchmark-specific scores and metrics
            - Additional metadata for BBQ benchmark (category, labels, etc.)
    """
    # Create decoding configuration from predefined settings
    decode_cfg = DecodeConfig(**DECODING_CONFIGS[decoding_name])
    results = []

    # Process each test item individually
    for idx, (_, row) in enumerate(df.iterrows()):
        rowd = row.to_dict()
        t1 = time.time()  # Start timing for performance monitoring

        # Create base metadata structure shared across all benchmarks
        base = {
            "model": model.model_name,
            "mitigation": mitigation,
            "decoding": decoding_name,
            "benchmark": bench_name,
            "item_id": rowd["item_id"],
            "bias_type": rowd.get("bias_type", "unknown"),          # e.g., 'gender', 'race'
            "intersection_group": rowd.get("intersection_group", "unknown"),  # intersectional demographics
        }

        try:
            # Route to appropriate scoring function based on benchmark type
            if bench_name == "stereoset":
                # StereoSet: Tests stereotype preference in sentence completion
                scores = score_stereoset(model, rowd, mitigation)
                results.append({**base, **scores})
                
            elif bench_name == "crows_pairs":
                # CrowS-Pairs: Tests stereotype preference between minimal pairs
                scores = score_crows(model, rowd, mitigation)
                results.append({**base, **scores})
                
            elif bench_name == "bbq":
                # BBQ: Tests bias in question-answering with additional metadata
                scores = score_bbq(model, rowd, mitigation, decode_cfg)
                results.append({
                    **base,
                    # BBQ-specific metadata for deeper analysis
                    "category": rowd.get("category"),                    # demographic category
                    "context_condition": rowd.get("context_condition"), # ambiguous vs disambiguated
                    "gold_label": rowd.get("gold_label"),               # correct answer
                    "unknown_label": rowd.get("unknown_label"),         # "unknown" option
                    "stereotyped_label": rowd.get("stereotyped_label"), # stereotypical answer
                    **scores,
                })

            # Log timing information for performance monitoring
            elapsed = time.time() - t1
            print(f"    [{idx+1}/{len(df)}] scored in {elapsed:.1f}s")
            
        except Exception as e:
            # Log errors but continue processing remaining items
            print(f"    [{idx+1}/{len(df)}] ERROR: {str(e)[:80]}")
            time.sleep(1)  # Brief pause to avoid overwhelming API if there are connection issues

    return results


def save_model_outputs(model_label, item_results_df):
    """
    Save comprehensive analysis outputs for a single model's evaluation results.
    
    This function takes the raw item-level results and generates multiple analysis
    files including summary statistics, statistical significance tests, and 
    intersectional bias analysis. All files are saved in a model-specific directory.
    
    Args:
        model_label (str): Short name for the model (e.g., 'mistral-7b')
        item_results_df (pd.DataFrame): DataFrame containing item-level evaluation results
        
    Returns:
        pd.DataFrame: The original item_results_df (for chaining or further processing)
        
    Generated Files:
        - item_level_results.jsonl: Raw results for each test item (JSON Lines format)
        - summary_metrics.csv: Aggregated metrics by benchmark and bias type
        - demographic_disparity_tests.csv: Statistical tests for bias across groups
        - intersectional_analysis.csv: Analysis of intersectional bias patterns
    """
    # Create model-specific output directory
    model_dir = os.path.join(BASE_OUTPUT_DIR, model_label)
    ensure_dir(model_dir)

    # Save raw item-level results in JSON Lines format for easy loading
    item_results_df.to_json(
        os.path.join(model_dir, "item_level_results.jsonl"),
        orient="records", lines=True,
    )
    print(f"  Saved item_level_results.jsonl ({len(item_results_df)} rows)")

    # Generate and save summary statistics aggregated by benchmark and bias type
    summary_df = summarize_all(item_results_df)
    summary_df.to_csv(os.path.join(model_dir, "summary_metrics.csv"), index=False)
    print(f"  Saved summary_metrics.csv ({len(summary_df)} rows)")

    # Conduct statistical tests for demographic disparities in model performance
    disparity_df = demographic_disparity_tests(item_results_df)
    disparity_df.to_csv(os.path.join(model_dir, "demographic_disparity_tests.csv"), index=False)
    print(f"  Saved demographic_disparity_tests.csv ({len(disparity_df)} rows)")

    # Analyze intersectional bias patterns (e.g., bias affecting multiple demographic dimensions)
    intersection_df = intersectional_analysis(item_results_df)
    intersection_df.to_csv(os.path.join(model_dir, "intersectional_analysis.csv"), index=False)
    print(f"  Saved intersectional_analysis.csv ({len(intersection_df)} rows)")

    return item_results_df


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def run():
    """
    Execute the complete bias evaluation pipeline.
    
    This is the main orchestration function that:
    1. Loads and samples datasets from three bias evaluation benchmarks
    2. Evaluates each specified LLM on all benchmarks with different configurations
    3. Generates individual model analysis reports 
    4. Creates cross-model comparison analysis
    5. Saves all results in structured output directories
    
    The pipeline is designed to be comprehensive yet efficient, providing both
    detailed item-level results and high-level statistical summaries.
    """
    # Ensure output directory exists
    ensure_dir(BASE_OUTPUT_DIR)
    
    # Print pipeline header
    print("=" * 60)
    print("BIAS EVALUATION PIPELINE — API RUN")
    print("=" * 60)

    # ========================================================================
    # STEP 1: Load and Sample Datasets
    # ========================================================================
    print("\nLoading and preprocessing datasets...")
    data = load_all_preprocessed()  # Load StereoSet, CrowS-Pairs, and BBQ datasets

    # Create smaller samples for faster evaluation while maintaining representativeness
    sampled = {}
    for name, df in data.items():
        n = SAMPLE_SIZES.get(name, 30)  # Get predefined sample size or default to 30
        sdf = sample_df(df, n)
        sampled[name] = sdf
        print(f"  {name}: {len(sdf)} items (sampled from {len(df)})")

    # Store results from all models for cross-model analysis
    all_item_results = []

    # ========================================================================
    # STEP 2: Evaluate Each Model
    # ========================================================================
    for model_[HF_TOKEN_REDACTED], model_label in MODELS_TO_RUN.items():
        print(f"\n{'=' * 60}")
        print(f"MODEL: {model_label} ({model_[HF_TOKEN_REDACTED]})")
        print(f"{'=' * 60}")

        # Initialize the model wrapper for bias evaluation
        model = BiasEvaluatorModel(model_[HF_TOKEN_REDACTED])

        # Collect all results for this model across all configurations
        model_results = []

        # Nested loops to test all combinations of mitigation strategies, 
        # decoding methods, and benchmarks
        for mitigation in MITIGATIONS_TO_RUN:
            for decoding_name in DECODINGS_TO_RUN:
                for bench_name, df in sampled.items():
                    print(f"\n  >>> {bench_name} | {mitigation} | {decoding_name} ({len(df)} items)")
                    
                    # Score all items in this benchmark with current configuration
                    results = score_items(model, mitigation, decoding_name, bench_name, df)
                    model_results.extend(results)

        # Convert results to DataFrame for analysis and storage
        model_df = pd.DataFrame(model_results)
        all_item_results.append(model_df)

        # Generate and save comprehensive analysis for this model
        print(f"\n  Saving {model_label} outputs...")
        save_model_outputs(model_label, model_df)

    # ========================================================================
    # STEP 3: Cross-Model Analysis (if multiple models evaluated)
    # ========================================================================
    if len(all_item_results) > 1:
        print(f"\n{'=' * 60}")
        print("GENERATING CROSS-MODEL ANALYSIS")
        print(f"{'=' * 60}")
        
        # Create combined directory for cross-model comparisons
        combined_dir = os.path.join(BASE_OUTPUT_DIR, "combined")
        ensure_dir(combined_dir)
        
        # Combine all model results into single dataset
        combined_df = pd.concat(all_item_results, ignore_index=True)

        # Save combined raw results
        combined_df.to_json(
            os.path.join(combined_dir, "all_item_level_results.jsonl"),
            orient="records", lines=True,
        )

        # Generate statistical comparison tests between models
        cross_df = cross_model_tests(combined_df)
        cross_df.to_csv(os.path.join(combined_dir, "cross_model_tests.csv"), index=False)
        print(f"\nSaved combined cross_model_tests.csv ({len(cross_df)} rows)")

        # Generate overall summary across all models
        summary_df = summarize_all(combined_df)
        summary_df.to_csv(os.path.join(combined_dir, "summary_all_models.csv"), index=False)
        print(f"Saved combined summary_all_models.csv ({len(summary_df)} rows)")

    # ========================================================================
    # STEP 4: Pipeline Completion Summary  
    # ========================================================================
    print("\n" + "=" * 60)
    print("DONE — Results saved to bias_eval/outputs/")
    for _, label in MODELS_TO_RUN.items():
        print(f"  bias_eval/outputs/{label}/")
    if len(all_item_results) > 1:
        print(f"  bias_eval/outputs/combined/")
    print("=" * 60)


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Entry point for the bias evaluation pipeline.
    
    When run as a standalone script, this will execute the complete evaluation
    pipeline including dataset loading, model evaluation, statistical analysis,
    and report generation.
    
    Example usage:
        python run_sample.py
        
    The script will automatically:
    - Load the three bias evaluation datasets
    - Sample smaller subsets for faster evaluation  
    - Evaluate all configured models on all benchmarks
    - Generate comprehensive statistical reports
    - Save results in structured output directories
    """
    run()

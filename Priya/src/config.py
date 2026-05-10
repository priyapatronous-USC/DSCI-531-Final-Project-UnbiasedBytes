"""
Configuration file for LLM bias evaluation pipeline.

This module contains all the configuration parameters used throughout the
bias evaluation pipeline, including model specifications, decoding parameters,
mitigation strategies, and statistical analysis settings.
"""

# List of HuggingFace model identifiers to evaluate for bias
MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",     # Meta's LLaMA 3 8B parameter model
    "mistralai/Mistral-7B-Instruct-v0.2",     # Mistral AI's 7B parameter model
    "google/gemma-2-9b-it",                    # Google's Gemma 2 9B parameter model
]

# Decoding configuration parameters for text generation
DECODING_CONFIGS = {
    # Deterministic decoding with temperature=0 for reproducible results
    "deterministic": {
        "temperature": 0.0,    # No randomness in token selection
        "top_p": 1.0,          # Consider all tokens in probability mass
        "num_samples": 1,      # Generate single output per prompt
    },
    # Stochastic decoding with controlled randomness for diversity
    "stochastic": {
        "temperature": 0.7,    # Moderate randomness in token selection
        "top_p": 0.9,          # Nucleus sampling with 90% probability mass
        "num_samples": 10,     # Generate multiple samples per prompt
    },
}

# Bias mitigation strategies to test (currently only baseline implemented)
MITIGATION_CONDITIONS = [
    "baseline",           # No mitigation - standard model prompting
    "neutral_framing",    # Neutral prompt framing (future implementation)
    "self_debias",        # Self-debiasing techniques (future implementation)
    "chain_of_thought",   # CoT prompting for bias reduction (future implementation)
]

# Statistical analysis parameters
BOOTSTRAP_SAMPLES = 2000    # Number of bootstrap samples for confidence intervals
RANDOM_SEED = 42           # Fixed seed for reproducible random sampling

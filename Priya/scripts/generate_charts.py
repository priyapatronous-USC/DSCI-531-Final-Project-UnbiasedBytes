"""
Generate publication-ready charts for bias evaluation results.

Creates two main visualizations:
1. Multi-panel bar chart showing key bias metrics across models
2. Statistical significance heatmap for cross-model comparisons

Usage: python generate_charts.py (requires completed evaluation data)
"""

import sys
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
import numpy as np

# Add src module to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Directory setup - check for both regular outputs and sample_run
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Project root
SAMPLE_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "sample_run")
REGULAR_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Use sample_run directory if it exists and has data, otherwise use regular outputs
if (os.path.exists(SAMPLE_OUTPUT_DIR) and 
    any(os.path.exists(os.path.join(SAMPLE_OUTPUT_DIR, d)) for d in ["mistral-7b", "llama3-8b", "gemma2-9b"])):
    OUTPUT_DIR = SAMPLE_OUTPUT_DIR
    print("Using sample_run output directory")
else:
    OUTPUT_DIR = REGULAR_OUTPUT_DIR
    print("Using regular output directory")

CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)  # Create charts directory if it doesn't exist

# Model configuration for visualization
# Family-keyed display labels and colors — full HuggingFace model IDs are
# auto-detected from the data below. This keeps the script robust to model
# variant changes (e.g. Mistral-7B-v0.1 vs Mistral-7B-Instruct-v0.2,
# LLaMA-3-8B vs LLaMA-3-8B-Instruct, etc.).
FAMILY_DISPLAY = [
    ("llama",   "LLaMA-3-8B",  "#3B82F6"),
    ("mistral", "Mistral-7B",  "#EF4444"),
    ("gemma",   "Gemma-2-9B",  "#10B981"),
]

# Load evaluation results data
summary = pd.read_csv(os.path.join(OUTPUT_DIR, "combined", "summary_all_models.csv"))

def _detect_family_models(model_ids):
    """Map each family substring to the actual model ID present in the data."""
    detected = {}
    for family, _, _ in FAMILY_DISPLAY:
        match = next((m for m in model_ids if family in m.lower()), None)
        if match is not None:
            detected[family] = match
    return detected

_present_models = sorted(summary["model"].dropna().unique().tolist())
_family_to_id = _detect_family_models(_present_models)

MODEL_LABELS = {
    _family_to_id[fam]: label
    for fam, label, _ in FAMILY_DISPLAY
    if fam in _family_to_id
}
MODEL_ORDER = [_family_to_id[fam] for fam, _, _ in FAMILY_DISPLAY if fam in _family_to_id]
COLORS = {label: color for fam, label, color in FAMILY_DISPLAY if fam in _family_to_id}

_missing = [label for fam, label, _ in FAMILY_DISPLAY if fam not in _family_to_id]
if _missing:
    print(f"Warning: no data found for {_missing} — those bars will be omitted.")
print(f"Charting models: {[(MODEL_LABELS[m], m) for m in MODEL_ORDER]}")

summary["model_label"] = summary["model"].map(MODEL_LABELS)  # Add short labels for plotting

all_items = pd.read_json(
    os.path.join(OUTPUT_DIR, "combined", "all_item_level_results.jsonl"), lines=True
)
all_items["model_label"] = all_items["model"].map(MODEL_LABELS)  # Add short labels for plotting


# ============================================================================
# CHART 1: Multi-Panel Bar Chart - Key Bias Metrics by Model
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle("Bias Evaluation: Key Metrics Across Models", fontsize=15, fontweight="bold", y=1.02)

# Panel A: StereoSet Stereotype Preference Rate by demographic group
ax = axes[0]
ss = summary[(summary["benchmark"] == "stereoset")].copy()  # Filter StereoSet data
x = np.arange(2)  # Two demographic groups: gender, race
width = 0.25  # Bar width for grouped bars

# Plot bars for each model
for i, model in enumerate(MODEL_ORDER):
    label = MODEL_LABELS[model]
    # Extract data for gender and race bias types
    row_g = ss[(ss["model"] == model) & (ss["bias_type"] == "gender")]
    row_r = ss[(ss["model"] == model) & (ss["bias_type"] == "race")]
    vals = [
        row_g["spr"].values[0] if len(row_g) else 0,  # Gender SPR
        row_r["spr"].values[0] if len(row_r) else 0,  # Race SPR
    ]
    # Create grouped bars with model-specific colors
    bars = ax.bar(x + i * width, vals, width, label=label, color=COLORS[label], edgecolor="white")
    # Add value labels on top of bars
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# Add reference line for no-bias threshold (50%)
ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(1.85, 51, "No bias (50%)", fontsize=7.5, color="gray")
# Configure axis labels and formatting
ax.set_xticks(x + width)
ax.set_xticklabels(["Gender", "Race"])
ax.set_ylabel("Stereotype Preference Rate (%)")
ax.set_title("StereoSet SPR", fontweight="bold")
ax.set_ylim(0, 110)
ax.legend(fontsize=8)

# Panel B: CrowS-Pairs Stereotype Preference Rate by demographic group
ax = axes[1]
cp = summary[(summary["benchmark"] == "crows_pairs")].copy()  # Filter CrowS-Pairs data

# Plot bars for each model (same structure as Panel A)
for i, model in enumerate(MODEL_ORDER):
    label = MODEL_LABELS[model]
    # Extract data for gender and race bias types
    row_g = cp[(cp["model"] == model) & (cp["bias_type"] == "gender")]
    row_r = cp[(cp["model"] == model) & (cp["bias_type"] == "race")]
    vals = [
        row_g["spr"].values[0] if len(row_g) else 0,  # Gender SPR
        row_r["spr"].values[0] if len(row_r) else 0,  # Race SPR
    ]
    # Create grouped bars with model-specific colors
    bars = ax.bar(x + i * width, vals, width, label=label, color=COLORS[label], edgecolor="white")
    # Add value labels on top of bars
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# Add reference line for no-bias threshold (50%)
ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
# Configure axis labels and formatting  
ax.set_xticks(x + width)
ax.set_xticklabels(["Gender", "Race"])
ax.set_ylabel("Stereotype Preference Rate (%)")
ax.set_title("CrowS-Pairs SPR", fontweight="bold")
ax.set_ylim(0, 110)
ax.legend(fontsize=8)

# Panel C: BBQ Question-Answering Metrics
ax = axes[2]
bbq = summary[(summary["benchmark"] == "bbq")].copy()  # Filter BBQ data

# Aggregate BBQ metrics by model (average across demographic groups)
bbq_agg = bbq.groupby("model").agg(
    accuracy=("accuracy_supported", "mean"),      # Accuracy on disambiguated questions
    bias_amb=("bias_ambiguous", "mean"),          # Bias rate on ambiguous questions
    unknown_rate=("unknown_rate_ambiguous", "mean"),  # Unknown rate on ambiguous questions
).reindex(MODEL_ORDER)  # Maintain consistent model ordering

# Define metrics and their display labels
metrics = ["accuracy", "bias_amb", "unknown_rate"]
metric_labels = ["Accuracy\n(disambig.)", "Bias Rate\n(ambig.)", "Unknown Rate\n(ambig.)"]
x3 = np.arange(3)  # Three metrics to display

# Plot bars for each model across the three BBQ metrics
for i, model in enumerate(MODEL_ORDER):
    label = MODEL_LABELS[model]
    vals = [bbq_agg.loc[model, m] * 100 for m in metrics]  # Convert to percentages
    # Create grouped bars with model-specific colors
    bars = ax.bar(x3 + i * width, vals, width, label=label, color=COLORS[label], edgecolor="white")
    # Add value labels on top of bars
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

# Configure axis labels and formatting
ax.set_xticks(x3 + width)
ax.set_xticklabels(metric_labels, fontsize=9)
ax.set_ylabel("Percentage (%)")
ax.set_title("BBQ Question Answering", fontweight="bold")
ax.set_ylim(0, 115)
ax.legend(fontsize=8)

# Save the multi-panel chart
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "bar_chart_key_metrics.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved bar_chart_key_metrics.png")



# ============================================================================
# CHART 2: Statistical Significance Heatmap - Cross-Model Comparisons
# ============================================================================

# Load cross-model statistical test results
cross = pd.read_csv(os.path.join(OUTPUT_DIR, "combined", "cross_model_tests.csv"))

# Define benchmark and demographic combinations for heatmap
benchmarks = ["stereoset", "crows_pairs", "bbq"]
bias_types = ["gender", "race"]
combos = [(b, bt) for b in benchmarks for bt in bias_types]  # All benchmark-demographic pairs
combo_labels = [f"{b.replace('_','-')}\n({bt})" for b, bt in combos]  # Format labels for display

# Define model pairs for comparison — built from auto-detected model IDs so
# this is robust to model variant changes.
_pair_specs = [
    ("LLaMA vs Gemma",   "llama",   "gemma"),
    ("LLaMA vs Mistral", "llama",   "mistral"),
    ("Gemma vs Mistral", "gemma",   "mistral"),
]
pairs = [
    (label, _family_to_id[fa], _family_to_id[fb])
    for label, fa, fb in _pair_specs
    if fa in _family_to_id and fb in _family_to_id
]

# Initialize matrices to store p-values and effect sizes
matrix = np.full((len(pairs), len(combos)), np.nan)  # P-values matrix
effect_matrix = np.full((len(pairs), len(combos)), np.nan)  # Effect sizes matrix

# Populate matrices with statistical test results
for j, (bench, bt) in enumerate(combos):  # Loop through benchmark-demographic combinations
    for i, (pair_label, ma, mb) in enumerate(pairs):  # Loop through model pairs
        # Filter data for current benchmark and bias type
        row = cross[(cross["benchmark"] == bench) & (cross["bias_type"] == bt)]
        # Find matching model pair (handle both A-B and B-A orderings)
        match = row[
            ((row["model_a"] == ma) & (row["model_b"] == mb)) |
            ((row["model_a"] == mb) & (row["model_b"] == ma))
        ]
        # Extract p-value and effect size if data exists
        if len(match) > 0:
            p_val = match.iloc[0]["p"]
            es = match.iloc[0]["effect_size"]
            if pd.notna(p_val):
                matrix[i, j] = p_val  # Store p-value
            if pd.notna(es):
                effect_matrix[i, j] = abs(es)  # Store absolute effect size

# Create and configure the heatmap
fig, ax = plt.subplots(figsize=(12, 4.5))
fig.suptitle("Cross-Model Statistical Comparisons (p-values & Effect Sizes)",
             fontsize=14, fontweight="bold", y=1.04)

# Set up color mapping for p-values (red = significant, green = non-significant)
cmap = plt.cm.RdYlGn_r  # Reversed red-yellow-green colormap
norm = matplotlib.colors.LogNorm(vmin=1e-16, vmax=1.0)  # Logarithmic scale for p-values
im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

# Add text annotations for each cell (p-values, effect sizes, significance markers)
for i in range(len(pairs)):
    for j in range(len(combos)):
        p = matrix[i, j]
        es = effect_matrix[i, j]
        if np.isnan(p):
            # No data available for this combination
            ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="gray")
        else:
            # Determine significance level markers
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            es_str = f"|d|={es:.2f}" if not np.isnan(es) else ""  # Effect size string
            color = "white" if p < 0.01 else "black"  # Text color based on background
            # Display p-value, effect size, and significance
            ax.text(j, i, f"p={p:.1e}\n{es_str} {sig}", ha="center", va="center",
                    fontsize=7.5, color=color, fontweight="bold" if sig else "normal")

# Configure axis labels and layout
ax.set_xticks(range(len(combos)))
ax.set_xticklabels(combo_labels, fontsize=9)  # Benchmark-demographic combinations
ax.set_yticks(range(len(pairs)))
ax.set_yticklabels([p[0] for p in pairs], fontsize=10)  # Model pair labels
cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="p-value (log scale)")

# Save the heatmap
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "heatmap_cross_model.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved heatmap_cross_model.png")

print("\nAll charts saved to bias_eval/outputs/charts/")

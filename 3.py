import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# Step 1: Create a dummy dataset with model performance metrics
models = ["GPT-3", "LLaMA-2", "BART", "T5", "GPT-2"]
criteria = ["Perplexity", "BLEU", "ROUGE", "Inference Time", "Model Size"]

# Hypothetical performance metrics
data = np.array([
    [20, 0.85, 0.80, 0.5, 6],  # GPT-3
    [18, 0.83, 0.78, 0.6, 5],  # LLaMA-2
    [22, 0.82, 0.75, 0.4, 4],  # BART
    [25, 0.80, 0.72, 0.7, 3],  # T5
    [30, 0.75, 0.70, 0.8, 7]   # GPT-2
])

df_models = pd.DataFrame(data, columns=criteria, index=models)

# Step 2: Normalize the Decision Matrix
norm_data = data / np.sqrt((data**2).sum(axis=0))

# Step 3: Define Weights (Equal weights for now)
weights = np.array([0.25, 0.25, 0.20, 0.15, 0.15])

# Step 4: Identify Benefit (+) and Cost (-) Criteria
benefit_criteria = ["BLEU", "ROUGE"]  # Higher is better
cost_criteria = ["Perplexity", "Inference Time", "Model Size"]  # Lower is better

# Step 5: Compute Ideal Best & Worst Solutions
ideal_best = np.max(norm_data, axis=0)  # Best for benefit criteria
ideal_worst = np.min(norm_data, axis=0)  # Worst for benefit criteria

for i, crit in enumerate(criteria):
    if crit in cost_criteria:
        ideal_best[i], ideal_worst[i] = ideal_worst[i], ideal_best[i]  # Swap for cost criteria

# Step 6: Compute Separation Measures
dist_best = np.sqrt(((norm_data - ideal_best) ** 2).sum(axis=1))
dist_worst = np.sqrt(((norm_data - ideal_worst) ** 2).sum(axis=1))

# Step 7: Compute TOPSIS Score
topsis_score = dist_worst / (dist_best + dist_worst)

# Step 8: Rank Models
df_models["TOPSIS Score"] = topsis_score
df_models["Rank"] = df_models["TOPSIS Score"].rank(ascending=False)

# Sorting models by Rank
df_models_sorted = df_models.sort_values(by="Rank")

# --- 1. Bar Chart for TOPSIS Scores ---
plt.figure(figsize=(10, 5))
sns.barplot(x=df_models_sorted.index, y=df_models_sorted["TOPSIS Score"], palette="viridis")
plt.xlabel("Pretrained Models", fontsize=12)
plt.ylabel("TOPSIS Score", fontsize=12)
plt.title("TOPSIS Score Comparison of Text Generation Models", fontsize=14)
plt.ylim(0, 1)
plt.show()

# --- 2. Scatter Plot (BLEU vs. Perplexity) ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_models["Perplexity"], y=df_models["BLEU"], hue=df_models.index, s=150, palette="coolwarm", edgecolor="black")
plt.xlabel("Perplexity (Lower is Better)", fontsize=12)
plt.ylabel("BLEU Score (Higher is Better)", fontsize=12)
plt.title("Perplexity vs BLEU Score for Text Generation Models", fontsize=14)
plt.legend(title="Models")
plt.show()

# --- 3. Radar Chart for Model Metrics ---
# Data Preparation
categories = criteria  # Metrics
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]  # Angles for radar chart
angles += angles[:1]  # Close the circle

# Normalize values for radar chart (scale 0-1)
normalized_values = (df_models.iloc[:, :-2] - df_models.iloc[:, :-2].min()) / (df_models.iloc[:, :-2].max() - df_models.iloc[:, :-2].min())

# Plot radar chart
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Plot each model
for idx, model in enumerate(df_models.index):
    values = normalized_values.loc[model].tolist()
    values += values[:1]  # Close the circle
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
    ax.fill(angles, values, alpha=0.1)

# Labels and Design
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
plt.title("Performance Comparison of Models (Radar Chart)", fontsize=14)
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.show()

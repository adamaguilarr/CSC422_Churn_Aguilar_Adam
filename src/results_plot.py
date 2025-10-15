import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_path = Path("results/metrics.csv")
df = pd.read_csv(results_path)

# Drop duplicate baseline if needed
df = df.drop_duplicates(subset=["model"], keep="last")

# Plot accuracy and F1 side by side
fig, ax = plt.subplots(figsize=(7,4))
df.plot(
    x="model",
    y=["accuracy", "f1"],
    kind="bar",
    ax=ax,
    color=["skyblue", "orange"],
    edgecolor="black"
)
ax.set_title("Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.set_ylim(0, 1)
ax.legend(["Accuracy", "F1 Score"])
plt.xticks(rotation=15)
fig.tight_layout()

# Save and show
plt.savefig("results/model_performance.png", dpi=150)
plt.show()

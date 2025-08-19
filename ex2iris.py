# Load Dataset
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target,
iris.target_names)
# Pair Plot
sns.pairplot(df, hue="species", diag_kind="hist")
plt.show()
# Box Plots
for col in iris.feature_names:
sns.boxplot(x="species", y=col, data=df)
plt.title(f"Boxplot: {col}")
plt.show()
# Violin Plots
for col in iris.feature_names:
sns.violinplot(x="species", y=col, data=df)
plt.title(f"Violin Plot: {col}")
plt.show()
# Correlation Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True,
cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

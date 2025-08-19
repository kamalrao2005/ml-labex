# Load the Iris dataset
from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# Apply transformations to a skewed feature
import numpy as np
from scipy.stats import boxcox
log_transformed = np.log1p(df['sepal width (cm)'])
sqrt_transformed = np.sqrt(df['sepal width (cm)'])
boxcox_transformed, _ = boxcox(df['sepal width (cm)'] + 1e-
6)
# Check normality using Shapiro-Wilk test
from scipy.stats import shapiro
print(shapiro(df['sepal width (cm)'].sample(100)))
print(shapiro(pd.Series(log_transformed).sample(100)))
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(log_transformed, kde=True)
plt.title('Log-transformed Sepal Width')
plt.show()

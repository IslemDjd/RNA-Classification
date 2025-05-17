# ----------------------------
# 1. Imports
# ----------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# ----------------------------
# 2. Chargement et préparation des données
# ----------------------------
df = pd.read_csv("Dataset/breastCancer.csv")
df['bare_nucleoli'] = pd.to_numeric(df['bare_nucleoli'], errors='coerce')
df.dropna(inplace=True)
df.drop(columns='id', inplace=True)

X = df.drop(columns='class')
y = df['class'].map({2: 0, 4: 1})  # 2 = bénin, 4 = malin

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------
# 3. PMC (Perceptron Multi-Couche)
# ----------------------------
model = Sequential([
    Input(shape=(9,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=0)

accuracy_pmc = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"\n🔷 Accuracy PMC : {accuracy_pmc:.4f}")

# ----------------------------
# 4. RBF (Radial Basis Function)
# ----------------------------
n_centers = 10
kmeans = KMeans(n_clusters=n_centers, random_state=42).fit(X_train)
centers = kmeans.cluster_centers_

def rbf(X, centers, sigma=1.0):
    dists = cdist(X, centers, 'euclidean')
    return np.exp(-(dists ** 2) / (2 * sigma ** 2))

X_train_rbf = rbf(X_train, centers, sigma=1.0)
X_test_rbf = rbf(X_test, centers, sigma=1.0)

clf = LogisticRegression()
clf.fit(X_train_rbf, y_train)

accuracy_rbf = clf.score(X_test_rbf, y_test)
print(f"🔶 Accuracy RBF : {accuracy_rbf:.4f}")

# ----------------------------
# 5. Visualisation : bar chart des accuracies
# ----------------------------
models = ['PMC (MLP)', 'RBF']
accuracies = [accuracy_pmc, accuracy_rbf]

plt.figure(figsize=(6, 5))
bars = plt.bar(models, accuracies, color=['skyblue', 'salmon'])
plt.title("Comparaison des performances : PMC vs RBF")
plt.ylabel("Accuracy")
plt.ylim(0, 1.1)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ----------------------------
# 6. Visualisation 2D des centres RBF (PCA)
# ----------------------------
pca = PCA(n_components=2)
X_train_2D = pca.fit_transform(X_train)
centers_2D = pca.transform(centers)

plt.figure(figsize=(8, 6))
plt.scatter(X_train_2D[:, 0], X_train_2D[:, 1], c=y_train, cmap='coolwarm', alpha=0.5, label='Données')
plt.scatter(centers_2D[:, 0], centers_2D[:, 1], c='black', s=100, marker='X', label='Centres RBF')
plt.title("Visualisation 2D des données + centres RBF")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

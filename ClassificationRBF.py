import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Chargement et nettoyage
df = pd.read_csv("Dataset/breastCancer.csv")
df['bare_nucleoli'] = pd.to_numeric(df['bare_nucleoli'], errors='coerce')
df.dropna(inplace=True)
df.drop(columns='id', inplace=True)
X = df.drop(columns='class')
y = df['class'].map({2: 0, 4: 1})

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# --- Construction du RBF ---
# 1. Choisir le nombre de centres (neurones dans la couche RBF)
n_centers = 10

# 2. Trouver les centres avec KMeans
kmeans = KMeans(n_clusters=n_centers, random_state=42)
kmeans.fit(X_train)
centers = kmeans.cluster_centers_

# 3. Calculer la matrice RBF (gaussienne) pour chaque échantillon
def rbf(X, centers, sigma=1.0):
    dists = cdist(X, centers, 'euclidean')
    return np.exp(-(dists ** 2) / (2 * sigma ** 2))

# 4. Construire les matrices RBF
X_train_rbf = rbf(X_train, centers, sigma=1.0)
X_test_rbf = rbf(X_test, centers, sigma=1.0)

# 5. Couche de sortie = régression logistique (ou perceptron simple)
clf = LogisticRegression()
clf.fit(X_train_rbf, y_train)

# 6. Prédictions
y_pred = clf.predict(X_test_rbf)

# 7. Évaluation
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
print("Rapport de classification :\n", classification_report(y_test, y_pred))


# Réduction à 2 dimensions pour visualisation
pca = PCA(n_components=2)
X_train_2D = pca.fit_transform(X_train)
centers_2D = pca.transform(centers)

# Visualisation
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

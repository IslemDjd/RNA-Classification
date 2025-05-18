import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from minisom import MiniSom
from matplotlib.patches import Patch

# Chargement et préparation
df = pd.read_csv("Dataset/breastCancer.csv")
df['bare_nucleoli'] = pd.to_numeric(df['bare_nucleoli'], errors='coerce')
df.dropna(inplace=True)
df.drop(columns='id', inplace=True)

X = df.drop(columns='class')
y = df['class'].map({2: 0, 4: 1})  # 0 = bénin, 1 = malin

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

som = MiniSom(
    x=18,               # largeur de la grille
    y=18,               # hauteur de la grille → 400 neurones
    input_len=9,        # 9 attributs d’entrée
    sigma=0.6,          # rayon d’influence modéré
    learning_rate=0.2, # apprentissage stable
    random_seed=42
)
som.random_weights_init(X_scaled)
som.train(X_scaled, num_iteration=3000, verbose=True)




# =============================
# 📊 1. U-Matrix (Unified Distance Matrix)
# =============================
plt.figure(figsize=(10, 10))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # Transposé pour bonne orientation
plt.colorbar(label='Distance entre neurones')
plt.title("U-Matrix – Carte de distance SOM")
plt.tight_layout()
plt.show()

...

# =============================
# 🎨 2. Affichage des étiquettes
# =============================
plt.figure(figsize=(10, 10))
for i, x in enumerate(X_scaled):
    w = som.winner(x)
    label = y.iloc[i]
    color = 'blue' if label == 0 else 'red'
    marker = 'o' if label == 0 else 's'
    plt.plot(w[0]+0.5, w[1]+0.5, marker, markerfacecolor=color,
             markeredgecolor='k', markersize=8, alpha=0.8)

plt.title("SOM – Affichage des étiquettes (bleu=bénin, rouge=malin)")
legend_elements = [
    Patch(facecolor='blue', edgecolor='k', label='Bénin (0)'),
    Patch(facecolor='red', edgecolor='k', label='Malin (1)')
]
plt.legend(handles=legend_elements, loc='upper right')
plt.xlim([0, 15])
plt.ylim([0, 15])
plt.grid(True)
plt.tight_layout()
plt.show()

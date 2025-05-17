import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
df = pd.read_csv("Dataset/breastCancer.csv")

# Nettoyage
df['bare_nucleoli'] = pd.to_numeric(df['bare_nucleoli'], errors='coerce')
df.dropna(inplace=True)
df.drop(columns='id', inplace=True)

# Séparation des données
X = df.drop(columns='class')
y = df['class'].map({2: 0, 4: 1})  # 2 = bénin, 4 = malin

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Construction du PMC
model = Sequential([
    Input(shape=(9,)),             # au lieu de Dense(..., input_dim=9)
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

# Évaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy sur test : {accuracy:.4f}")

# Prédictions et rapport
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
print("Rapport de classification :\n", classification_report(y_test, y_pred))




# Courbes de perte et précision
plt.figure(figsize=(12, 5))

# Courbe de perte
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Entraînement')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Courbe de perte")
plt.xlabel("Épochs")
plt.ylabel("Perte")
plt.legend()

# Courbe d'accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Entraînement')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Courbe de précision")
plt.xlabel("Épochs")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# Matrice de confusion
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Matrice de confusion")
plt.xlabel("Prédictions")
plt.ylabel("Vraies valeurs")
plt.show()

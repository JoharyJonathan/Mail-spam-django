import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Charger les données à partir du fichier CSV
data = pd.read_csv("E:\IA\mail-spam\train-model\spam_ham_dataset.csv")

# Séparer les features (X) et les labels (y)
X_train = data['text']  # Les messages
y_train = data['label_num']  # Les labels (0 ou 1)

# Vectoriser les données textuelles
vectorizer = TfidfVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)

# Entraîner le modèle
model = LogisticRegression()
model.fit(X_train_transformed, y_train)

# Calculer la précision du modèle
accuracy = model.score(X_train_transformed, y_train)

# Afficher le pourcentage de précision
print(f'Précision du modèle : {accuracy * 100:.2f}%')

# Sauvegarder le modèle et le vectorizer
joblib.dump(model, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
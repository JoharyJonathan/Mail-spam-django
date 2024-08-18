import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Charger les données à partir du fichier CSV
data = pd.read_csv("E:/IA/mail-spam/train-model/MAILIAGASY.csv", sep=',', names=["label", "text"])

# Remplacer les labels textuels par des valeurs numériques
data['label_num'] = data['label'].apply(lambda x: 1 if x.strip().lower() == 'spam' else 0)

# Remplacer les valeurs NaN dans la colonne 'text' par des chaînes vides
data['text'] = data['text'].fillna("")

# Filtrer les documents vides
data = data[data['text'].str.strip() != '']

# Séparer les features (X) et les labels (y)
X_train = data['text']
y_train = data['label_num']

# Vectoriser les données textuelles avec des paramètres plus permissifs
vectorizer = TfidfVectorizer(stop_words=None, min_df=1)
X_train_transformed = vectorizer.fit_transform(X_train)

# Entraîner le modèle
model = LogisticRegression()
model.fit(X_train_transformed, y_train)

# Calculer la précision du modèle
accuracy = model.score(X_train_transformed, y_train)

# Afficher le pourcentage de précision
print(f'Précision du modèle : {accuracy * 100:.2f}%')

# Sauvegarder le modèle et le vectorizer
joblib.dump(model, 'modelgasy.joblib')
joblib.dump(vectorizer, 'vectorizergasy.joblib')
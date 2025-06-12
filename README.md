# 📸 Pneumonia Detection via Transfer Learning (CNN)

Ce projet propose un **prototype fonctionnel** (Proof of Concept) d'un système de classification binaire de radiographies thoraciques visant à détecter des cas de **pneumonie**, en utilisant le **transfert learning** appliqué à un modèle pré-entraîné de vision par ordinateur (CNN).

## 🧠 Objectifs

- Réutiliser un **modèle CNN pré-entraîné** pour classifier les radios thoraciques (saines vs. atteintes de pneumonie).
- Comprendre et mettre en œuvre le **transfert learning** dans un contexte médical réel.
- Initier une démarche **MLOps** via l'outil **MLflow** pour assurer la traçabilité des expérimentations.
- Fournir un **notebook complet, lisible et structuré**, documentant l'ensemble du processus.

## 🗂 Contenu du projet

Le projet est entièrement réalisé dans un **notebook Jupyter** disponible dans ce dépôt.

### Étapes couvertes :

- 📁 **Préparation et exploration du dataset** : *Chest X-Ray Pneumonia Dataset*
- 🧩 **Chargement et adaptation d’un modèle pré-entraîné** (e.g., ResNet, DenseNet)
- 🔁 **Entraînement du modèle** sur les données adaptées
- 📊 **Évaluation rigoureuse** : matrice de confusion, courbes ROC/AUC, précision, rappel, F1-score...
- 🧪 **Suivi d’expérimentations avec MLflow** : paramètres, métriques, version des modèles

## 📦 Dataset

- **Nom** : Chest X-Ray Pneumonia
- **Source** : [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Format** : Images JPEG (réparties en train/test/val)
- **Tâche** : Classification binaire — Pneumonia vs. Normal

## 🛠 Technologies utilisées

- Python 3
- TensorFlow / Keras
- scikit-learn
- MLflow
- Jupyter Notebook
- Matplotlib / Seaborn

## 🚀 MLOps & Suivi d’expérimentations

Une première introduction à la culture **MLOps** est proposée via **MLflow** :
- Suivi des hyperparamètres et métriques
- Versioning de modèles
- Visualisation des performances

## 🧪 Reproductibilité

Le projet est structuré pour être facilement reproductible :
- Code clair et commenté
- Instructions intégrées dans le notebook
- Utilisation de chemins relatifs et bonnes pratiques

## ✅ Résultats attendus

> L’objectif n’est pas d’atteindre une performance optimale, mais de **démontrer une démarche claire et rigoureuse** :
- Compréhension du transfert learning
- Adaptation correcte du modèle à la classification binaire
- Expérimentations tracées et évaluées
- Interprétation des résultats




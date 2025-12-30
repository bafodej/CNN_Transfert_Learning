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

### 📥 Installation du dataset

1. Installez Kaggle CLI :
```bash
pip install kaggle
```

2. Configurez vos identifiants Kaggle :
   - Allez sur [Kaggle Account Settings](https://www.kaggle.com/settings)
   - Cliquez sur "Create New API Token"
   - Placez le fichier `kaggle.json` dans `~/.kaggle/`

3. Téléchargez le dataset :
```bash
cd CNN_Transfert_Learning
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
rm chest-xray-pneumonia.zip
```

La structure du dataset sera :
```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

## 🛠 Technologies utilisées

- Python 3
- TensorFlow / Keras
- scikit-learn
- MLflow
- Jupyter Notebook
- Matplotlib / Seaborn

## 🚀 Démarrage rapide

1. Clonez le dépôt :
```bash
git clone <votre-repo-url>
cd CNN_Transfert_Learning
```

2. Créez et activez l'environnement virtuel :
```bash
python3 -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

4. Téléchargez le dataset (voir section Dataset ci-dessous)

5. Lancez Jupyter Notebook :
```bash
jupyter notebook
```

6. Ouvrez `Pneumonia_Detection_Transfert_Learning.ipynb` et exécutez les cellules

## 🧪 Reproductibilité

Le projet est structuré pour être facilement reproductible :
- Code clair et commenté
- Instructions intégrées dans le notebook
- Utilisation de chemins relatifs et bonnes pratiques

## 📁 Structure du projet

```
CNN_Transfert_Learning/
├── Pneumonia_Detection_Transfert_Learning.ipynb  # Notebook principal
├── README.md                                      # Documentation
├── requirements.txt                               # Dépendances Python
├── .gitignore                                     # Fichiers ignorés par Git
├── data/                                          # Dataset (non versionné)
│   └── chest_xray/
│       ├── train/
│       ├── test/
│       └── val/
├── models/                                        # Modèles sauvegardés (non versionné)
└── src/                                           # Scripts Python (optionnel)
```

## ✅ Résultats attendus

> L'objectif n'est pas d'atteindre une performance optimale, mais de **démontrer une démarche claire et rigoureuse** :
- Compréhension du transfert learning
- Adaptation correcte du modèle à la classification binaire
- Expérimentations tracées et évaluées
- Interprétation des résultats

## 📝 Contenu du notebook

Le notebook complet couvre les étapes suivantes :

1. **Configuration et imports** - Setup de l'environnement
2. **Exploration du dataset** - Statistiques et visualisations
3. **Préparation des données** - Chargement, normalisation, split
4. **Modèle CNN Baseline** - Construction et compilation d'un CNN simple
5. **Transfer Learning VGG16** - Adaptation d'un modèle pré-entraîné
6. **Entraînement** - Avec callbacks (EarlyStopping, ReduceLROnPlateau)
7. **Évaluation** - Métriques complètes (accuracy, precision, recall, F1, AUC)
8. **Visualisations** - Courbes d'entraînement, matrices de confusion, ROC curves
9. **Comparaison** - Tableau comparatif des performances
10. **Conclusions** - Analyse et perspectives d'amélioration




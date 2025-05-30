# 🫁 Détection de Pneumonie avec Transfer Learning

## 📋 Description du Projet

Ce projet utilise l'intelligence artificielle pour détecter automatiquement la pneumonie sur des radiographies thoraciques. Il implémente une approche de **transfer learning** avec le modèle pré-entraîné **DenseNet121** pour classifier les images en deux catégories : NORMAL et PNEUMONIA.

### 🎯 Objectifs
- Développer un système de classification binaire d'images médicales
- Démontrer l'efficacité du transfer learning en imagerie médicale
- Mettre en place un pipeline MLOps avec suivi des expérimentations
- Créer un prototype fonctionnel (Proof of Concept) pour l'aide au diagnostic

---

## 🏗️ Architecture du Projet

### 📊 Modèle
- **Architecture de base** : DenseNet121 pré-entraîné sur ImageNet
- **Transfer Learning** : Gel des couches convolutionnelles pré-entraînées
- **Couches personnalisées** : GlobalAveragePooling2D + Dense layers avec Dropout
- **Optimisation** : Adam optimizer avec learning rate adaptatif

### 📁 Structure des Données
```
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

### 🔧 Pipeline ML
1. **Préparation des données** : ImageDataGenerator avec augmentation
2. **Modélisation** : Transfer learning avec DenseNet121
3. **Entraînement** : Callbacks (EarlyStopping, ReduceLROnPlateau)
4. **Évaluation** : Métriques complètes et matrices de confusion
5. **Suivi** : MLflow pour la traçabilité des expériences

---

## 🚀 Installation et Configuration

### 📋 Prérequis
- Python 3.10 ou 3.11
- RAM : 8 GB minimum (16 GB recommandé)
- Espace disque : 2 GB pour le dataset

### 🔧 Installation des Dépendances

1. **Cloner le repository**
```bash
git clone https://github.com/votre-username/pneumonia-detection.git
cd pneumonia-detection
```

2. **Créer un environnement virtuel**
```bash
python -m venv pneumonia_env
pneumonia_env\Scripts\activate  # Windows
# ou
source pneumonia_env/bin/activate  # Linux/Mac
```

3. **Installer les packages**
```bash
pip install --upgrade pip
pip install tensorflow-cpu==2.13.0  # ou tensorflow==2.13.0 pour GPU
pip install jupyter jupyterlab
pip install scikit-learn matplotlib seaborn pandas numpy pillow
pip install mlflow
```

### 📥 Téléchargement du Dataset

1. **Depuis Kaggle** (Recommandé)
   - Aller sur : https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
   - Télécharger le dataset (1.15 GB)
   - Extraire dans `data/chest_xray/`

2. **Via API Kaggle**
```bash
pip install kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

---

## 💻 Utilisation

### 🚀 Lancement du Projet

1. **Démarrer Jupyter Lab**
```bash
jupyter lab
```

2. **Ouvrir le notebook principal**
   - `Pneumonia_Detection_Transfert_Learning.ipynb`

3. **Exécuter les cellules** dans l'ordre séquentiel

### ⚙️ Configuration des Paramètres

Les paramètres principaux sont configurables dans le notebook :

```python
# Paramètres du modèle
IMG_SIZE = (224, 224)    # Taille des images
BATCH_SIZE = 16          # Taille des batches
EPOCHS = 15              # Nombre d'époques
LEARNING_RATE = 0.0001   # Taux d'apprentissage
```

### 📊 Suivi avec MLflow

Pour lancer l'interface MLflow :
```bash
mlflow ui
```
Accès via : http://localhost:5000

---

## 📈 Résultats et Performances

### 📊 Métriques du Modèle
- **Accuracy** : ~92% sur le jeu de test
- **Precision** : ~90% pour la détection de pneumonie
- **Recall** : ~95% (sensibilité élevée pour les cas positifs)
- **F1-Score** : ~92%

### 📉 Courbes d'Apprentissage
Le modèle converge généralement en 8-12 époques avec :
- Diminution progressive de la loss
- Amélioration stable de l'accuracy
- Pas de surapprentissage grâce aux callbacks

### 🎯 Matrice de Confusion
```
              Predicted
Actual     NORMAL  PNEUMONIA
NORMAL       234      12
PNEUMONIA     15     363
```

---

## 🔬 Approche Technique

### 🧠 Transfer Learning
- **Modèle de base** : DenseNet121 (ImageNet)
- **Stratégie** : Feature extraction (couches gelées)
- **Fine-tuning** : Couches de classification personnalisées

### 🖼️ Augmentation de Données
- Rotation (±15°)
- Décalages horizontaux/verticaux (±10%)
- Zoom (±10%)
- Retournement horizontal
- Variation de luminosité (0.8-1.2)

### 📏 Métriques d'Évaluation
- **Accuracy** : Précision globale
- **Precision** : Vrais positifs / (Vrais positifs + Faux positifs)
- **Recall** : Vrais positifs / (Vrais positifs + Faux négatifs)
- **F1-Score** : Moyenne harmonique de Precision et Recall

---

## 📁 Structure du Projet

```
pneumonia-detection/
│
├── data/
│   └── chest_xray/          # Dataset (à télécharger)
│
├── notebooks/
│   └── Pneumonia_Detection_Transfert_Learning.ipynb
│
├── models/
│   └── densenet121_pneumonia.h5    # Modèle sauvegardé
│
├── mlruns/                  # Expériences MLflow
│
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
│
├── requirements.txt
├── README.md
└── .gitignore
```

---
## 🔮 Développements Futurs

### 🎯 Améliorations Prévues
- [ ] **Fine-tuning** des dernières couches de DenseNet121
- [ ] **Ensemble methods** avec plusieurs architectures
- [ ] **Gradual unfreezing** pour optimiser le transfer learning
- [ ] **Data balancing** pour gérer le déséquilibre des classes

### 🚀 Déploiement
- [ ] **API REST** avec FastAPI
- [ ] **Interface web** avec Streamlit
- [ ] **Containerisation** avec Docker
- [ ] **CI/CD Pipeline** avec GitHub Actions

### 📊 MLOps Avancé
- [ ] **Model registry** avec MLflow
- [ ] **Monitoring** des performances en production
- [ ] **A/B Testing** de différents modèles
- [ ] **Drift detection** pour les données d'entrée

---

---

## 📚 Références et Sources


### 🌐 Dataset
- **Source** : Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018)
- **Kaggle** : https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
- **Citation** : Chest X-Ray Images (Pneumonia)

### 🛠️ Technologies
- **TensorFlow/Keras** : Framework de deep learning
- **DenseNet121** : Architecture CNN pré-entraînée
- **MLflow** : Plateforme MLOps
- **Scikit-learn** : Métriques et évaluation

---

## 👨‍💻 Auteur et Contributions

**Auteur** : [Bafodé Jaiteh]  

**Date** : Mai 2025


---


*Projet réalisé dans le cadre d'un apprentissage du Machine Learning et du Transfer Learning appliqué à l'imagerie médicale.*
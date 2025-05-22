# Détection de Pneumonie par Transfer Learning 🫁
## 📋 Description du Projet
Ce projet développe un Proof of Concept (POC) pour la détection automatisée de pneumonie à partir de radiographies thoraciques, en utilisant les techniques de Transfer Learning avec des modèles de vision par ordinateur pré-entraînés.

## Contexte Médical

- Problématique : Aide au diagnostic médical pour détecter la pneumonie sur des radiographies thoraciques
- Enjeu : Réduire le temps de diagnostic et améliorer la précision du dépistage
- Approche : Classification binaire (Normal vs Pneumonie) avec CNN pré-entraîné
- Perspective : Initiation à une démarche MLOps pour la traçabilité des expérimentations

  ## 🎯 Objectifs du Projet

 - Explorer et analyser le dataset Chest X-Ray Pneumonia
 - Implémenter un modèle de Transfer Learning (CNN pré-entraîné)
 - Adapter le modèle à la classification binaire médicale
 - Évaluer les performances avec des métriques adaptées au contexte médical
 - Tracer les expérimentations avec MLflow (initiation MLOps)
 - Documenter la démarche de manière claire et reproductible

## 📊 Dataset
Source : Chest X-Ray Images (Pneumonia) - Kaggle

- Total d'images : ~5,856 images
- Classes :

- Normal : ~1,559 images
- Pneumonie : ~4,277 images


- Format : Images JPEG en niveaux de gris
- Splits : Train / Validation / Test pré-définis

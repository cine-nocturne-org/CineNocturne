# CineNocturne 🎬

## Description

CineNocturne est une application de recommandation de films personnalisée. Elle permet aux utilisateurs de :
	•	Rechercher et noter des films 🎯
	•	Recevoir des recommandations personnalisées basées sur leurs goûts 🍿
	•	Explorer des films par genre et par plateforme 📺

Tout cela dans une interface Streamlit fluide et esthétique, avec gestion des utilisateurs et monitoring MLOps via MLflow.

⸻

## Fonctionnalités
	1.	Connexion sécurisée 🔐
Gestion simple des utilisateurs via variables d’environnement.
	3.	Recommandations personnalisées 🎲
	•	Basées sur les notes précédentes
	•	Top 10 des films similaires
	4.	Suggestions aléatoires ⚡
	•	Par genre et plateforme
	•	Limité à 10 films pour rapidité
	5.	Plateformes disponibles 📺
	•	Recherche par titre pour connaître toutes les plateformes de diffusion
	6.	Optimisations 🚀
	•	Caching des appels API pour accélérer le temps de traitement
	•	Centralisation des affichages pour éviter les duplications
	•	Compatible MLflow pour tracking et monitoring

⸻

## Installation
	1.	Cloner le dépôt

```git clone https://github.com/PixelLouve/CineNocturne.git
cd CineNocturne

	2.	Créer un environnement virtuel

```python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

	3.	Installer les dépendances

```pip install -r requirements.txt

	4.	Configurer les variables d’environnement

```# Exemple pour Linux / Mac
export API_TOKEN="ton_token_api"
export USER_LOU="motdepasse"
# Ajouter d'autres utilisateurs si nécessaire

	5.	Lancer l’application

```streamlit run E3_E4_API_app/reco_app_v2.py


⸻

## Structure du projet

CineNocturne/
│
├─ E3_E4_API_app/
│  ├─ reco_app_v2.py       # Application principale Streamlit
│  ├─ config.py            # Configurations et constantes
│  └─ logo_cinenocturne.png
│
├─ requirements.txt
├─ .env                    # Variables d'environnement (API_TOKEN, USERS)
└─ README.md


⸻

## Utilisation
	1.	Connexion
	•	Entrez votre nom d’utilisateur et mot de passe.
	2.	Onglet 1 : Recommandations perso
	•	Cherchez un film que vous avez vu
	•	Donnez-lui une note
	•	Recevez vos recommandations personnalisées
	3.	Onglet 2 : Suggestions aléatoires
	•	Choisissez un genre et une ou des plateformes
	•	Découvrez des films aléatoires
	4.	Onglet 3 : Plateformes disponibles
	•	Recherchez un film pour connaître toutes les plateformes où il est disponible

⸻

## Notes techniques
	•	Caching : @st.cache_data pour réduire les appels API répétitifs.
	•	Affichage centralisé : display_movie() pour uniformiser l’affichage des films.
	•	MLflow : intégration pour tracker les notes et recommandations.
	•	Optimisations : structure multithreadable et réutilisation des composants pour accélérer l’appli.

⸻

### Auteur
Nyx Valen (Lou) – Développeuse passionnée et de cinéma nocturne. 🌙🖤
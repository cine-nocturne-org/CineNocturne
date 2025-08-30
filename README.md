# CineNocturne 🎬

> *La nuit tombe, l’écran s’illumine : CineNocturne murmure les films qui te ressemblent.* 🌙🖤

---

## ✨ Description

CineNocturne est une application **Streamlit** de recommandation de films personnalisée. Elle permet de :

* **Rechercher et noter** des films 🎯
* **Recevoir des recommandations** adaptées à tes goûts 🍿
* **Explorer par genre** et **par plateforme** 📺

Le tout avec une interface fluide, une gestion simple des utilisateurs, et un **monitoring MLOps via MLflow**.

---

## 🗺️ Sommaire

* [Fonctionnalités](#-fonctionnalités)
* [Prérequis](#-prérequis)
* [Installation](#-installation)
* [Configuration](#-configuration)
* [Lancement](#-lancement)
* [Structure du projet](#-structure-du-projet)
* [Utilisation](#-utilisation)
* [Notes techniques](#-notes-techniques)
* [Dépannage rapide](#-dépannage-rapide)
* [Auteur](#-auteur)

---

## 🚀 Fonctionnalités

1. **Connexion sécurisée** 🔐
   Gestion simple des utilisateurs via variables d’environnement.
2. **Recommandations personnalisées** 🎲

   * Basées sur les notes précédentes
   * Top 10 des films similaires
3. **Suggestions aléatoires** ⚡

   * Par genre et par plateforme
   * Limité à 10 films pour la rapidité
4. **Plateformes disponibles** 📺

   * Recherche par titre pour connaître les plateformes de diffusion
5. **Optimisations** 🚀

   * Caching des appels API pour accélérer le temps de traitement
   * Centralisation des affichages pour éviter les duplications
   * Intégration **MLflow** pour le tracking et le monitoring

---

## 🧰 Prérequis

* **Python 3.9+**
* **pip** ou **uv/pipx** (au choix)
* **virtualenv** / `venv`

---

## 📦 Installation

1. **Cloner le dépôt**

```bash
git clone https://github.com/PixelLouve/CineNocturne.git
cd CineNocturne
```

2. **Créer et activer l’environnement virtuel**

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate
```

```powershell
# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

Tu peux configurer l’appli via **variables d’environnement** (recommandé avec un fichier `.env`).

### Variables attendues

* `API_URL` : URL de l’API backend (ex. `https://cinenocturne.onrender.com/`)
* `API_TOKEN` : jeton d’accès à l’API
* `USERS` ou paires utilisateur/mot de passe (ex. `USER_LOU`) selon ton implémentation

### Exemple `.env`

Crée un fichier `.env` à la racine du projet :

```env
# Backend API
API_URL="https://cinenocturne.onrender.com/"
API_TOKEN="ton_token_api"

# Utilisateurs (exemples)
USER_LOU="motdepasse"
# USER_ANOTHER="motdepasse2"
```

> 💡 Sur Linux/macOS, tu peux aussi exporter à la volée :

```bash
export API_TOKEN="ton_token_api"
export USER_LOU="motdepasse"
```

> 💡 Sur Windows PowerShell :

```powershell
$env:API_TOKEN = "ton_token_api"
$env:USER_LOU  = "motdepasse"
```

---

## ▶️ Lancement

```bash
streamlit run E3_E4_API_app/reco_app_v2.py
```

Une URL locale sera affichée dans le terminal (ex. `http://localhost:8501`).

---

## 🗂️ Structure du projet

```
CineNocturne/
│
├─ E3_E4_API_app/
│  ├─ reco_app_v2.py       # Application principale Streamlit
│  ├─ config.py            # Configurations et constantes
│  └─ logo_cinenocturne.png
│
├─ requirements.txt
├─ .env                    # Variables d'environnement (API_URL, API_TOKEN, USERS)
└─ README.md
```

---

## 🕹️ Utilisation

1. **Connexion**
   Entre ton nom d’utilisateur et ton mot de passe.
2. **Onglet 1 : Recommandations perso**

   * Cherche un film que tu as vu
   * Donne-lui une note
   * Reçois des recommandations personnalisées
3. **Onglet 2 : Suggestions aléatoires**

   * Choisis un genre et une ou plusieurs plateformes
   * Découvre des films aléatoires
4. **Onglet 3 : Plateformes disponibles**

   * Recherche un film pour connaître toutes les plateformes où il est disponible

---

## 🧪 Notes techniques

* **Caching** : `@st.cache_data` pour réduire les appels API répétitifs.
* **Affichage centralisé** : `display_movie()` pour uniformiser la présentation des films.
* **MLflow** : intégration pour tracker notes et recommandations.
* **Optimisations** : composants réutilisables et structure compatible exécution multithread.

---

## 🩹 Dépannage rapide

* **Erreur de définition d’API (ex. 502 sur `/openapi.json`)** :

  * Vérifie que `API_URL` pointe vers un backend **déployé et accessible**.
  * Si tu utilises Render/railway/etc., patiente après le *cold start* du service et réessaie.
* **Problèmes d’environnement** :

  * Assure-toi que l’environnement virtuel est **activé** et que `python -V` ≥ 3.9.
* **Variables non lues** :

  * Confirme la présence du fichier `.env` à la racine et l’appel à `load_dotenv()` dans l’app.

---

## ✍️ Auteur

**Nyx Valen (Lou)** – Développeuse passionnée… et de cinéma nocturne. 🌙🖤

---

> *Que la nuit te soit douce, et tes recommandations percutantes.*

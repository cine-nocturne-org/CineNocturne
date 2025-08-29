import streamlit as st
import requests
from urllib.parse import quote
import traceback
from requests.auth import HTTPBasicAuth
import os
from dotenv import load_dotenv
import mlflow
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "E3_E4_API_app"))
import config


# -----------------------------
# Configuration générale
# -----------------------------
load_dotenv()
API_URL = "https://cinenocturne.onrender.com/"
API_TOKEN = os.getenv("API_TOKEN")

st.set_page_config(page_title="Recommandation de Films", page_icon="🍿")

# Dictionnaire utilisateurs (login -> mdp)
USERS = {k.replace("USER_", "").lower(): v for k, v in os.environ.items() if k.startswith("USER_")}

# -----------------------------
# Fonctions de connexion
# -----------------------------
def login_page():
    """Affiche la page de connexion"""
    st.title("🔐 Connexion")
    st.markdown("---")

    with st.form("login_form"):
        st.subheader("Veuillez vous connecter pour accéder à l'application")
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submit_button = st.form_submit_button("Se connecter")

        if submit_button:
            if username and password:
                if USERS.get(username.lower()) == password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.api_token = API_TOKEN  # Stockage du token
                    st.success("Connexion réussie ! 🎉")
                    st.rerun()
                else:
                    st.error("❌ Identifiants incorrects")
            else:
                st.warning("⚠️ Veuillez remplir tous les champs")


def logout():
    """Déconnecte l'utilisateur"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.api_token = None
    st.rerun()

# -----------------------------
# Fonctions utilitaires API
# -----------------------------
def api_get(endpoint: str, params: dict = None):
    headers = {"Authorization": f"Bearer {st.session_state.api_token}"}
    return requests.get(f"{API_URL}{endpoint}", headers=headers, params=params, timeout=5)

def api_post(endpoint: str, payload: dict):
    headers = {"Authorization": f"Bearer {st.session_state.api_token}"}
    return requests.post(f"{API_URL}{endpoint}", headers=headers, json=payload, timeout=5)

# -----------------------------
# Application principale
# ----------------------------
def main_app():
    # Header avec bouton déconnexion
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🎮 Recommandation de Films Personnalisée")
    with col2:
        st.write(f"👋 Connecté en tant que: **{st.session_state.username}**")
        if st.button("🚪 Se déconnecter"):
            logout()

    USERNAME = st.session_state.username

    # ------------------------------
    # Onglets
    # ------------------------------
    tab1, tab2, tab3 = st.tabs([
        "✨​ Recommandations perso",
        "🎲 Suggestions aléatoires",
        "📺​ Plateformes disponibles"
        ])


    # ------------------------------
    # Onglet 1 : Film vu + reco perso
    # ------------------------------
    with tab1:
        st.subheader("✨ Noter un film que vous avez vu")

        # Entrée titre
        film_input = st.text_input("Entrez le titre du film :")
        if st.button("Chercher", key="btn_search"):
            st.session_state["fuzzy_matches"] = None
            st.session_state["chosen_film"] = None
            if film_input:
                try:
                    response = api_get(f"fuzzy_match/{film_input}", params={"top_k": 10})
                    if response.status_code == 200:
                        matches = response.json().get("matches", [])
                        st.session_state["fuzzy_matches"] = matches if matches else []
                        if not matches:
                            st.warning("⚠️ Aucun film trouvé avec ce titre.")
                    else:
                        st.error("❌ Erreur lors de la recherche.")
                except requests.exceptions.RequestException:
                    st.error("❌ Erreur de connexion avec le serveur")

        # Sélection du film
        if st.session_state.get("fuzzy_matches"):
            matches_info = []
            for match in st.session_state["fuzzy_matches"]:
                details_resp = api_get(f"movie-details/{match['title']}")
                poster_url = None
                movie_id = match.get("movie_id")
                if details_resp.status_code == 200:
                    details = details_resp.json()
                    poster_url = details.get("poster_url")
                    movie_id = details.get("movie_id", movie_id)
                matches_info.append({
                    "title": match["title"],
                    "poster": poster_url,
                    "movie_id": movie_id
                })

            st.markdown("### Sélectionnez le film correct :")
            rows, cols_per_row = 2, 5
            for row in range(rows):
                row_matches = matches_info[row*cols_per_row : (row+1)*cols_per_row]
                if not row_matches:
                    continue
                cols = st.columns(len(row_matches))
                for col_idx, match in enumerate(row_matches):
                    with cols[col_idx]:
                        if match.get("poster"):
                            st.image(match["poster"], width="stretch")
                        st.caption(match.get("title", "Titre inconnu"))
                        unique_key = f"select_{match['movie_id']}_{row}_{col_idx}"
                        if st.session_state.get("chosen_film") == match["title"]:
                            st.button("✅ Sélectionné", key=unique_key, disabled=True)
                        else:
                            if st.button("Sélectionner", key=unique_key):
                                st.session_state["chosen_film"] = match["title"]

        # Notation du film choisi
        chosen_film = st.session_state.get("chosen_film")
        if chosen_film:
            st.success(f"🎬 Film sélectionné : {chosen_film}")
            note_input = st.number_input("Note du film (0.0 - 10.0)", min_value=0.0, max_value=10.0, value=5.0, step=0.1, format="%.1f", key="note_input")
            if st.button("Valider la note"):
                payload = {"title": chosen_film, "rating": note_input}
                update_resp = api_post("update_rating", payload)
                if update_resp.status_code == 200:
                    st.success(f"✅ La note {note_input} a été enregistrée pour '{chosen_film}' !")
                else:
                    detail = update_resp.json().get("detail", "Erreur inconnue")
                    st.error(f"❌ Échec : {detail}")

        # Recommandations personnalisées
        if chosen_film:
            st.subheader("🔍 Obtenir une recommandation personnalisée")
            if st.button("Me recommander un film", key="btn_reco"):
                try:
                    # --- Démarrage d'un run MLflow côté Streamlit ---
                    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)  # par défaut dossier local
                    with mlflow.start_run(run_name=f"streamlit_reco_{chosen_film}", nested=True):
                        mlflow.log_param("user", st.session_state.username)
                        mlflow.log_param("film_input", chosen_film)

                        # Appel à ton API
                        response = api_get(f"recommend_xgb_personalized/{chosen_film}", params={"top_k": 5})
                        
                        if response.status_code == 200:
                            recos = response.json()
                            mlflow.log_text(str([r["title"] for r in recos]), "top_recommended_titles.txt")

                            if recos:
                                st.success("🎯 Recommandations trouvées !")
                                for reco in recos:
                                    mlflow.log_metric(f"pred_score_{reco['title']}", reco.get("pred_score", 0))
                                    cols = st.columns([1, 3])
                                    with cols[0]:
                                        if reco.get("poster_url"):
                                            st.image(reco["poster_url"], width="stretch")
                                    with cols[1]:
                                        reco_title = reco.get("title", "Titre inconnu")
                                        reco_year = reco.get("releaseYear")
                                        reco_genres = reco.get("genres", [])
                                        reco_platforms = reco.get("platforms", [])
                                        reco_synopsis = reco.get("synopsis", "Pas de synopsis disponible.")
                                        score_pct = int(reco.get("pred_score", 0) * 100)

                                        st.markdown(f"### 🎬 {reco_title} ({reco_year})")
                                        st.markdown(f"**Ce film est susceptible de vous plaire à {score_pct}%**")
                                        st.write(f"**Genres :** {', '.join(reco_genres) if reco_genres else 'N/A'}")
                                        st.write(f"**Plateformes disponibles :** {', '.join(reco_platforms) if reco_platforms else 'Indisponible'}")
                                        st.write(reco_synopsis)
                            else:
                                st.info("Aucune recommandation trouvée pour ce film")
                        else:
                            st.error(response.json().get("detail", "Erreur inconnue"))

                except requests.exceptions.RequestException:
                    st.error("❌ Erreur de connexion avec le serveur")
                except Exception as e:
                    st.error(f"❌ Erreur MLflow : {e}")
                    st.text(traceback.format_exc())


    # ------------------------------
    # Onglet 2 : Suggestions aléatoires
    # ------------------------------
    with tab2:
        st.subheader("🎲 Suggestions aléatoires par genre")
        try:
            genre_response = api_get("genres/")
            if genre_response.status_code == 200:
                genre_list = genre_response.json()
                with st.form("random_movies_form"):
                    selected_genre = st.selectbox("Choisissez un genre", genre_list)
                    selected_platforms = st.multiselect("Choisissez les plateformes", ["netflix", "prime", "hulu", "hbo", "apple"])
                    submitted = st.form_submit_button("Afficher des films aléatoires")

                if "already_seen_movies" not in st.session_state:
                    st.session_state["already_seen_movies"] = set()
                if "current_movies" not in st.session_state:
                    st.session_state["current_movies"] = []

                def fetch_random_movies():
                    params = {"genre": selected_genre, "platforms": selected_platforms, "limit": 20}
                    response = api_get("random_movies/", params=params)
                    if response.status_code == 200:
                        movies = response.json()
                        fresh_movies = [
                            m for m in movies
                            if m["title"] not in st.session_state["already_seen_movies"]
                            and m.get("poster_url") and m.get("synopsis")
                        ][:10]
                        for m in fresh_movies:
                            st.session_state["already_seen_movies"].add(m["title"])
                        st.session_state["current_movies"] = fresh_movies

                if submitted:
                    st.session_state["already_seen_movies"].clear()
                    fetch_random_movies()

                if st.session_state["current_movies"]:
                    if st.button("🔄 Proposer d'autres films"):
                        fetch_random_movies()

                    for movie in st.session_state["current_movies"]:
                        poster = movie.get("poster_url")
                        synopsis = movie.get("synopsis")
                        year = movie.get("releaseYear", "N/A")
                        if not poster or not synopsis:
                            continue
                        cols = st.columns([1, 3])
                        with cols[0]:
                            st.image(poster, width="stretch")
                        with cols[1]:
                            title = movie.get("title", "Titre inconnu")
                            raw_genres = movie.get("genres", [])
                            genres = raw_genres if isinstance(raw_genres, list) else [g.strip() for g in raw_genres.split(",")]
                            st.markdown(f"### 🎬 {title} ({year})")
                            st.write(f"**Genres :** {', '.join(genres) if genres else 'N/A'}")
                            st.write(synopsis)

        except Exception as e:
            st.error(f"❌ Impossible de récupérer les genres : {e}")
            st.text(traceback.format_exc())


    # ------------------------------
    # Onglet 3 : Plateformes dispo
    # ------------------------------
    with tab3:
        st.subheader("📺 Plateformes disponibles pour un film")
        film_details_title = st.text_input("Titre du film :", key="details_title")
        if st.button("🔍 Chercher correspondances", key="btn_fuzzy"):
            if film_details_title:
                try:
                    fuzzy_resp = api_get(f"fuzzy_match/{film_details_title}")
                    if fuzzy_resp.status_code == 200:
                        matches = fuzzy_resp.json().get("matches", [])
                        st.session_state["fuzzy_matches"] = matches if matches else []
                        if not matches:
                            st.warning("⚠️ Aucun film trouvé avec ce titre.")
                    else:
                        st.error(fuzzy_resp.json().get("detail", "Erreur lors du fuzzy match."))
                except requests.exceptions.RequestException:
                    st.error("❌ Erreur de connexion avec le serveur")
            else:
                st.warning("⚠️ Veuillez entrer un titre de film")

        if "fuzzy_matches" in st.session_state:
            chosen_movie = st.selectbox("Films correspondants :", [m["title"] for m in st.session_state["fuzzy_matches"]], key="chosen_movie_details")
            if st.button("✅ Confirmer ce film"):
                try:
                    response = api_get(f"movie-details/{chosen_movie}")
                    if response.status_code == 200:
                        details = response.json()
                        st.success("✅ Détails du film trouvés !")
                        col1, col2 = st.columns([1,2])
                        with col1:
                            if details.get("poster_url"):
                                st.image(details["poster_url"], width="stretch")
                        with col2:
                            st.markdown(f"### 🎬 {details['title']} ({details['releaseYear']})")
                            st.write(f"**Genres :** {details['genres']}")
                            st.write(f"**Note :** {details['rating']}")
                            st.write(f"**Plateformes disponibles :** {', '.join(details['platforms'])}")
                            st.write(details['synopsis'])
                    else:
                        st.error(response.json().get("detail", "Film non trouvé"))
                except requests.exceptions.RequestException:
                    st.error("❌ Erreur de connexion avec le serveur")


# -----------------------------
# Point d'entrée principal
# -----------------------------
def main():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'api_token' not in st.session_state:
        st.session_state.api_token = None

    if st.session_state.authenticated:
        main_app()
    else:
        login_page()

if __name__ == "__main__":
    main()



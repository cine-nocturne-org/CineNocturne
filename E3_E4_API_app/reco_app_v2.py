# -----------------------------
# Imports
# -----------------------------
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

import requests
import streamlit as st
from dotenv import load_dotenv
import mlflow

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
# Fonctions utilitaires API
# -----------------------------
@st.cache_data(show_spinner=False)
def cached_api_get(endpoint: str, api_token: str, params: dict = None):
    headers = {"Authorization": f"Bearer {api_token}"}
    return requests.get(f"{API_URL}{endpoint}", headers=headers, params=params)


@st.cache_data(show_spinner=False)
def cached_api_post(endpoint: str, payload: dict):
    headers = {"Authorization": f"Bearer {st.session_state.api_token}"}
    return requests.post(f"{API_URL}{endpoint}", headers=headers, json=payload)

# -----------------------------
# Connexion / Déconnexion
# -----------------------------
def login_page():
    render_header()
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
                    st.session_state.api_token = API_TOKEN
                    st.success("Connexion réussie ! 🎉")
                    st.rerun()
                else:
                    st.error("❌ Identifiants incorrects")
            else:
                st.warning("⚠️ Veuillez remplir tous les champs")

def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.api_token = None
    st.rerun()

# -----------------------------
# Film selector optimisé (posters en parallèle)
# -----------------------------
def fetch_movie_details(match, api_token):
    details_resp = cached_api_get(f"movie-details/{match['title']}", api_token)
    poster_url = None
    movie_id = match.get("movie_id")
    if details_resp.status_code == 200:
        details = details_resp.json()
        poster_url = details.get("poster_url")
        movie_id = details.get("movie_id", movie_id)
    return {
        "title": match["title"],
        "poster": poster_url,
        "movie_id": movie_id
    }

def film_selector(matches: List[Dict], state_key_prefix: str):
    matches_info = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_movie_details, m, st.session_state.api_token) for m in matches]
        for future in as_completed(futures):
            matches_info.append(future.result())

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
                unique_key = f"{state_key_prefix}_{match['movie_id']}_{row}_{col_idx}"
                if st.session_state.get(f"{state_key_prefix}_chosen") == match["title"]:
                    st.button("✅ Sélectionné", key=unique_key, disabled=True)
                else:
                    if st.button("Sélectionner", key=unique_key):
                        st.session_state[f"{state_key_prefix}_chosen"] = match["title"]

# -----------------------------
# Header
# -----------------------------
def render_header():
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <div style="display: flex; align-items: center;">
                <img src='https://github.com/PixelLouve/CineNocturne/blob/main/E3_E4_API_app/logo_cinenocturne.png?raw=true' width='120'>
                <h1 style="margin-left: 15px; margin-top: 5px;">🍿 Recommandation de Films Personnalisée</h1>
            </div>
        """,
        unsafe_allow_html=True
    )
    if st.session_state.get("username"):
        col1, col2 = st.columns([8, 2])
        with col2:
            st.markdown(
                f"""
                <div style="display:flex; justify-content:flex-end; align-items:center; gap:10px;">
                    <span>👋 Connecté en tant que: <b>{st.session_state.username}</b></span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.button("🚪 Se déconnecter", on_click=logout)

# -----------------------------
# Recommandations cache
# -----------------------------
@st.cache_data(show_spinner=False)
def cached_recommendations(chosen_film: str, username: str):
    response = cached_api_get(f"recommend_xgb_personalized/{chosen_film}", params={"top_k": 20})
    if response.status_code == 200:
        return response.json()
    return []

# -----------------------------
# Main app
# -----------------------------
def main_app():
    render_header()
    tab1, tab2, tab3 = st.tabs([
        "✨​ Recommandations perso",
        "🎲 Suggestions aléatoires",
        "📺​ Plateformes disponibles"
    ])

    # -----------------------------
    # Onglet 1 : Film vu + reco perso
    # -----------------------------
    with tab1:
        st.subheader("✨ Noter un film que vous avez vu")
        if st.button("🔄 Nouvelle recherche", key="btn_reset_tab1"):
            st.session_state["fuzzy_matches_tab1"] = None
            st.session_state["tab1_chosen"] = None
            st.session_state["already_recommended"] = set()
            st.session_state["last_recos"] = []

        film_input = st.text_input("Entrez le titre du film :")
        if st.button("Chercher", key="btn_tab1"):
            st.session_state["fuzzy_matches_tab1"] = None
            st.session_state["tab1_chosen"] = None
            st.session_state["already_recommended"] = set()
            if film_input:
                try:
                    response = cached_api_get(f"fuzzy_match/{film_input}", params={"top_k": 10})
                    if response.status_code == 200:
                        matches = response.json().get("matches", [])
                        st.session_state["fuzzy_matches_tab1"] = matches if matches else []
                        if not matches:
                            st.warning("⚠️ Aucun film trouvé avec ce titre.")
                    else:
                        st.error("❌ Erreur lors de la recherche.")
                except requests.exceptions.RequestException:
                    st.error("❌ Erreur de connexion avec le serveur")

        if st.session_state.get("fuzzy_matches_tab1"):
            film_selector(st.session_state["fuzzy_matches_tab1"], "tab1")

        chosen_film = st.session_state.get("tab1_chosen")
        if chosen_film:
            st.success(f"🎬 Film sélectionné : {chosen_film}")

            note_str: str = st.text_input(
                "Note du film (0 à 10, utilisez ',' ou '.' pour les décimales)",
                value="", key="note_input_str"
            )

            if st.button("Valider la note"):
                if not note_str.strip():
                    st.warning("⚠️ Veuillez saisir une note")
                else:
                    note_str = note_str.replace(",", ".")
                    try:
                        note_float = float(note_str)
                        if 0 <= note_float <= 10:
                            payload = {"title": chosen_film, "rating": note_float}
                            update_resp = cached_api_post("update_rating", payload)
                            if update_resp.status_code == 200:
                                st.success(f"✅ La note {note_float} a été enregistrée pour '{chosen_film}' !")
                            else:
                                detail = update_resp.json().get("detail", "Erreur inconnue")
                                st.error(f"❌ Échec : {detail}")
                        else:
                            st.warning("⚠️ La note doit être comprise entre 0 et 10")
                    except ValueError:
                        st.warning("⚠️ Veuillez saisir un nombre valide")

            if "already_recommended" not in st.session_state:
                st.session_state["already_recommended"] = set()
            if "last_recos" not in st.session_state:
                st.session_state["last_recos"] = []

            def fetch_recommendations():
                try:
                    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
                    with mlflow.start_run(run_name=f"streamlit_reco_{chosen_film}", nested=True):
                        mlflow.log_param("user", st.session_state.username)
                        mlflow.log_param("film_input", chosen_film)

                        recos = cached_recommendations(chosen_film, st.session_state.username)
                        new_recos = [r for r in recos if r["title"] not in st.session_state["already_recommended"]][:10]

                        if new_recos:
                            for r in new_recos:
                                st.session_state["already_recommended"].add(r["title"])
                            st.session_state["last_recos"] = new_recos
                        else:
                            st.info("Aucune nouvelle recommandation disponible pour ce film")
                            st.session_state["last_recos"] = []

                except Exception as e:
                    st.error(f"❌ Erreur : {e}")
                    st.session_state["last_recos"] = []

            if st.button("🎯 Me proposer des recommandations"):
                fetch_recommendations()

            if st.session_state["last_recos"]:
                st.subheader("🎯 Recommandations précédentes")
                for reco in st.session_state["last_recos"]:
                    cols = st.columns([1,3])
                    with cols[0]:
                        if reco.get("poster_url"):
                            st.image(reco["poster_url"], width="stretch")
                    with cols[1]:
                        st.markdown(f"### 🎬 {reco['title']} ({reco.get('releaseYear','N/A')})")
                        st.write(f"**Genres :** {', '.join(reco.get('genres',[]))}")
                        st.write(f"**Plateformes :** {', '.join(reco.get('platforms',[]))}")
                        st.write(reco.get("synopsis","Pas de synopsis disponible"))

                if st.button("🔄 Me proposer d'autres recommandations"):
                    fetch_recommendations()

    # -----------------------------
    # Onglet 2 : Suggestions aléatoires
    # -----------------------------
    with tab2:
        st.subheader("🎲 Suggestions aléatoires par genre")
        try:
            genre_response = cached_api_get("genres/")
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
                    response = cached_api_get("random_movies/", params=params)
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
                            st.markdown(f"### 🎬 {title}")
                            st.write(f"**Genres :** {', '.join(genres) if genres else 'N/A'}")
                            st.write(synopsis)
        except Exception as e:
            st.error(f"❌ Impossible de récupérer les genres : {e}")
            st.text(traceback.format_exc())

    # -----------------------------
    # Onglet 3 : Plateformes dispo
    # -----------------------------
    with tab3:
        st.subheader("📺 Plateformes disponibles pour un film")
        if st.button("🔄 Nouvelle recherche", key="btn_reset_tab3"):
            st.session_state["fuzzy_matches_tab3"] = None
            st.session_state["tab3_chosen"] = None
            st.session_state["details_title"] = ""

        film_details_title = st.text_input("Titre du film :", key="details_title")
        if st.button("🔍 Chercher correspondances", key="btn_tab3"):
            if film_details_title:
                try:
                    fuzzy_resp = cached_api_get(f"fuzzy_match/{film_details_title}")
                    if fuzzy_resp.status_code == 200:
                        matches = fuzzy_resp.json().get("matches", [])
                        st.session_state["fuzzy_matches_tab3"] = matches if matches else []
                        if not matches:
                            st.warning("⚠️ Aucun film trouvé avec ce titre.")
                    else:
                        st.error(fuzzy_resp.json().get("detail", "Erreur lors du fuzzy match."))
                except requests.exceptions.RequestException:
                    st.error("❌ Erreur de connexion avec le serveur")
            else:
                st.warning("⚠️ Veuillez entrer un titre de film")

        if st.session_state.get("fuzzy_matches_tab3") and not st.session_state.get("tab3_chosen"):
            film_selector(st.session_state["fuzzy_matches_tab3"], "tab3")

        chosen_film = st.session_state.get("tab3_chosen")
        if chosen_film:
            st.success(f"🎬 Film sélectionné : {chosen_film}")
            try:
                response = cached_api_get(f"movie-details/{chosen_film}")
                if response.status_code == 200:
                    details = response.json()
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if details.get("poster_url"):
                            st.image(details["poster_url"], width="stretch")
                    with col2:
                        st.markdown(f"### 🎬 {details['title']} ({details.get('releaseYear','N/A')})")
                        st.write(f"**Genres :** {details.get('genres','N/A')}")
                        st.write(f"**Note :** {details.get('rating','N/A')}")
                        st.write(f"**Plateformes disponibles :** {', '.join(details.get('platforms',[]))}")
                        st.write(details.get('synopsis','Pas de synopsis disponible'))
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


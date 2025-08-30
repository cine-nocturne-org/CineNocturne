# -----------------------------
# Imports
# -----------------------------
import os
import sys
import traceback
from typing import List, Dict, Any, Optional

import requests
import streamlit as st
from dotenv import load_dotenv
import mlflow

sys.path.append(os.path.join(os.path.dirname(__file__), "E3_E4_API_app"))
import config

# -----------------------------
# Config
# -----------------------------
load_dotenv()
API_URL: str = "https://cinenocturne.onrender.com/"
API_TOKEN: Optional[str] = os.getenv("API_TOKEN")

st.set_page_config(page_title="Recommandation de Films", page_icon="🍿")

USERS: Dict[str, str] = {
    k.replace("USER_", "").lower(): v
    for k, v in os.environ.items()
    if k.startswith("USER_")
}

# -----------------------------
# Session HTTP persistante
# -----------------------------
session = requests.Session()

def _get_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {st.session_state.get('api_token', '')}"}

def api_get(endpoint: str, params: Optional[dict] = None) -> requests.Response:
    return session.get(f"{API_URL}{endpoint}", headers=_get_headers(), params=params)

def api_post(endpoint: str, payload: dict) -> requests.Response:
    return session.post(f"{API_URL}{endpoint}", headers=_get_headers(), json=payload)

# -----------------------------
# Cache des appels API
# -----------------------------
@st.cache_data(ttl=3600)
def cached_api_json(endpoint: str, params: Optional[dict] = None) -> Optional[Dict[str, Any]]:
    try:
        resp = api_get(endpoint, params)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None

# -----------------------------
# Connexion / Déconnexion
# -----------------------------
def login_page():
    render_header()
    st.title("🔐 Connexion")
    st.markdown("---")
    with st.form("login_form"):
        st.subheader("Veuillez vous connecter pour accéder à l'application")
        username: str = st.text_input("Nom d'utilisateur")
        password: str = st.text_input("Mot de passe", type="password")
        submit_button: bool = st.form_submit_button("Se connecter")
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
# Sélection de films
# -----------------------------
def film_selector(matches: List[dict], state_key_prefix: str):
    matches_info: List[Dict[str, Any]] = []
    for match in matches:
        details = cached_api_json(f"movie-details/{match['title']}")
        if details:
            matches_info.append({
                "title": details.get("title", match["title"]),
                "poster": details.get("poster_url"),
                "movie_id": details.get("movie_id", match.get("movie_id"))
            })
    st.markdown("### Sélectionnez le film correct :")
    cols = st.columns(min(len(matches_info), 5))
    for idx, match in enumerate(matches_info):
        with cols[idx % 5]:
            if match.get("poster"):
                st.image(match["poster"], use_container_width=True)
            st.caption(match.get("title", "Titre inconnu"))
            unique_key = f"{state_key_prefix}_{match['movie_id']}_{idx}"
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
                <img src='https://github.com/PixelLouve/CineNocturne/blob/main/E3_E4_API_app/logo_cinenocturne.png?raw=true' width='250'>
                <h1 style="margin-left: 5px; margin-top: 5px;">🍿 Recommandation de Films Personnalisée</h1>
            </div>
        """,
        unsafe_allow_html=True
    )
    if st.session_state.get("username"):
        col1, col2 = st.columns([8, 2])
        with col2:
            st.markdown(
                f"<span>👋 Connecté en tant que: <b>{st.session_state.username}</b></span>",
                unsafe_allow_html=True
            )
            st.button("🚪 Se déconnecter", on_click=logout)

# -----------------------------
# Main App
# -----------------------------
def main_app():
    render_header()
    tab1, tab2, tab3 = st.tabs([
        "✨​ Recommandations perso",
        "🎲 Suggestions aléatoires",
        "📺​ Plateformes disponibles"
    ])

    # -----------------------------
    # Onglet 1
    # -----------------------------
    with tab1:
        st.subheader("✨ Noter un film que vous avez vu")
        if "fuzzy_matches_tab1" not in st.session_state:
            st.session_state["fuzzy_matches_tab1"] = None
        if "tab1_chosen" not in st.session_state:
            st.session_state["tab1_chosen"] = None
        if "already_recommended" not in st.session_state:
            st.session_state["already_recommended"] = set()
        if "last_recos" not in st.session_state:
            st.session_state["last_recos"] = []

        # Nouvelle recherche
        if st.button("🔄 Nouvelle recherche", key="btn_reset_tab1"):
            st.session_state["fuzzy_matches_tab1"] = None
            st.session_state["tab1_chosen"] = None
            st.session_state["already_recommended"].clear()
            st.session_state["last_recos"] = []

        # Recherche film
        film_input = st.text_input("Entrez le titre du film :")
        if st.button("Chercher", key="btn_tab1"):
            if film_input:
                try:
                    matches = cached_api_json(f"fuzzy_match/{film_input}", params={"top_k": 10})
                    st.session_state["fuzzy_matches_tab1"] = matches.get("matches", []) if matches else []
                    if not matches:
                        st.warning("⚠️ Aucun film trouvé avec ce titre.")
                except Exception:
                    st.error("❌ Erreur de connexion avec le serveur")

        # Sélection film
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
                if note_str.strip():
                    try:
                        note_float = float(note_str.replace(",", "."))
                        if 0 <= note_float <= 10:
                            payload = {"title": chosen_film, "rating": note_float}
                            resp = api_post("update_rating", payload)
                            if resp.status_code == 200:
                                st.success(f"✅ La note {note_float} a été enregistrée pour '{chosen_film}' !")
                            else:
                                st.error(f"❌ Échec : {resp.json().get('detail','Erreur inconnue')}")
                        else:
                            st.warning("⚠️ La note doit être comprise entre 0 et 10")
                    except ValueError:
                        st.warning("⚠️ Veuillez saisir un nombre valide")
                else:
                    st.warning("⚠️ Veuillez saisir une note")

            # Recommandations
            def fetch_recommendations():
                try:
                    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
                    with mlflow.start_run(run_name=f"streamlit_reco_{chosen_film}", nested=True):
                        mlflow.log_param("user", st.session_state.username)
                        mlflow.log_param("film_input", chosen_film)
                        recos_resp = cached_api_json(f"recommend_xgb_personalized/{chosen_film}", params={"top_k": 20})
                        if recos_resp:
                            new_recos = [r for r in recos_resp if r["title"] not in st.session_state["already_recommended"]][:10]
                            for r in new_recos:
                                st.session_state["already_recommended"].add(r["title"])
                            st.session_state["last_recos"] = new_recos
                        else:
                            st.info("Aucune nouvelle recommandation disponible")
                            st.session_state["last_recos"] = []
                except Exception as e:
                    st.error(f"❌ Erreur : {e}")
                    st.session_state["last_recos"] = []

            if st.button("🎯 Me proposer des recommandations"):
                fetch_recommendations()

            if st.session_state["last_recos"]:
                st.subheader("🎯 Recommandations précédentes")
                for reco in st.session_state["last_recos"]:
                    cols = st.columns([1, 3])
                    with cols[0]:
                        if reco.get("poster_url"):
                            st.image(reco["poster_url"], use_container_width=True)
                    with cols[1]:
                        st.markdown(f"### 🎬 {reco['title']} ({reco.get('releaseYear','N/A')})")
                        st.write(f"**Genres :** {', '.join(reco.get('genres',[]))}")
                        st.write(f"**Plateformes :** {', '.join(reco.get('platforms',[]))}")
                        st.write(reco.get("synopsis","Pas de synopsis disponible"))

                if st.button("🔄 Me proposer d'autres recommandations"):
                    fetch_recommendations()

    # -----------------------------
    # Onglet 2 et 3 restent similaires...
    # -----------------------------
# -----------------------------
# Onglet 2 : Suggestions aléatoires
# -----------------------------
with tab2:
    st.subheader("🎲 Suggestions aléatoires par genre")
    try:
        genre_list = cached_api_json("genres/")
        genre_list = genre_list if genre_list else []
        with st.form("random_movies_form"):
            selected_genre = st.selectbox("Choisissez un genre", genre_list)
            selected_platforms = st.multiselect(
                "Choisissez les plateformes",
                ["netflix", "prime", "hulu", "hbo", "apple"]
            )
            submitted = st.form_submit_button("Afficher des films aléatoires")

        if "already_seen_movies" not in st.session_state:
            st.session_state["already_seen_movies"] = set()
        if "current_movies" not in st.session_state:
            st.session_state["current_movies"] = []

        def fetch_random_movies():
            params = {"genre": selected_genre, "platforms": selected_platforms, "limit": 20}
            movies = cached_api_json("random_movies/", params=params)
            if movies:
                fresh_movies = [
                    m for m in movies
                    if m["title"] not in st.session_state["already_seen_movies"]
                    and m.get("poster_url") and m.get("synopsis")
                ][:10]
                for m in fresh_movies:
                    st.session_state["already_seen_movies"].add(m["title"])
                st.session_state["current_movies"] = fresh_movies
            else:
                st.session_state["current_movies"] = []

        if submitted:
            st.session_state["already_seen_movies"].clear()
            fetch_random_movies()

        if st.session_state["current_movies"]:
            if st.button("🔄 Proposer d'autres films"):
                fetch_random_movies()

            for movie in st.session_state["current_movies"]:
                cols = st.columns([1, 3])
                with cols[0]:
                    st.image(movie.get("poster_url"), use_container_width=True)
                with cols[1]:
                    title = movie.get("title", "Titre inconnu")
                    raw_genres = movie.get("genres", [])
                    genres = raw_genres if isinstance(raw_genres, list) else [g.strip() for g in raw_genres.split(",")]
                    st.markdown(f"### 🎬 {title}")
                    st.write(f"**Genres :** {', '.join(genres) if genres else 'N/A'}")
                    st.write(movie.get("synopsis", "Pas de synopsis disponible"))

    except Exception as e:
        st.error(f"❌ Impossible de récupérer les genres : {e}")
        st.text(traceback.format_exc())

# -----------------------------
# Onglet 3 : Plateformes disponibles
# -----------------------------
with tab3:
    st.subheader("📺 Plateformes disponibles pour un film")

    if "fuzzy_matches_tab3" not in st.session_state:
        st.session_state["fuzzy_matches_tab3"] = None
    if "tab3_chosen" not in st.session_state:
        st.session_state["tab3_chosen"] = None
    if "details_title" not in st.session_state:
        st.session_state["details_title"] = ""

    if st.button("🔄 Nouvelle recherche", key="btn_reset_tab3"):
        st.session_state["fuzzy_matches_tab3"] = None
        st.session_state["tab3_chosen"] = None
        st.session_state["details_title"] = ""

    film_details_title = st.text_input("Titre du film :", key="details_title")

    if st.button("🔍 Chercher correspondances", key="btn_tab3"):
        if film_details_title:
            try:
                matches_resp = cached_api_json(f"fuzzy_match/{film_details_title}")
                matches = matches_resp.get("matches", []) if matches_resp else []
                st.session_state["fuzzy_matches_tab3"] = matches
                if not matches:
                    st.warning("⚠️ Aucun film trouvé avec ce titre.")
            except Exception:
                st.error("❌ Erreur de connexion avec le serveur")
        else:
            st.warning("⚠️ Veuillez entrer un titre de film")

    if st.session_state.get("fuzzy_matches_tab3") and not st.session_state.get("tab3_chosen"):
        film_selector(st.session_state["fuzzy_matches_tab3"], "tab3")

    chosen_film = st.session_state.get("tab3_chosen")
    if chosen_film:
        st.success(f"🎬 Film sélectionné : {chosen_film}")
        details = cached_api_json(f"movie-details/{chosen_film}")
        if details:
            col1, col2 = st.columns([1, 2])
            with col1:
                if details.get("poster_url"):
                    st.image(details["poster_url"], use_container_width=True)
            with col2:
                st.markdown(f"### 🎬 {details.get('title','N/A')} ({details.get('releaseYear','N/A')})")
                st.write(f"**Genres :** {details.get('genres', 'N/A')}")
                st.write(f"**Note :** {details.get('rating', 'N/A')}")
                st.write(f"**Plateformes disponibles :** {', '.join(details.get('platforms', []))}")
                st.write(details.get('synopsis','Pas de synopsis disponible'))
        else:
            st.error("❌ Film non trouvé ou erreur serveur")



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

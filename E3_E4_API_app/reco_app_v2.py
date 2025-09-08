import streamlit as st
import requests
import traceback
import os
from dotenv import load_dotenv
import mlflow
import sys
import tempfile
import json


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
# Utilitaires API (aucun timeout)
# -----------------------------
def _headers():
    return {"Authorization": f"Bearer {st.session_state.api_token}"} if st.session_state.get("api_token") else {}

def api_get(endpoint: str, params: dict = None):
    return requests.get(f"{API_URL}{endpoint.lstrip('/')}", headers=_headers(), params=params)

def api_post(endpoint: str, payload: dict):
    return requests.post(f"{API_URL}{endpoint.lstrip('/')}", headers=_headers(), json=payload)

# -----------------------------
# MLflow (nested run côté UI)
# -----------------------------
def start_ui_run(input_title: str, user: str):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("louve_movies_monitoring")
    return mlflow.start_run(run_name=f"ui_reco_{input_title}", nested=True)

# -----------------------------
# Application principale
# -----------------------------
def main_app():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🎮 Recommandation de Films Personnalisée")
    with col2:
        st.write(f"👋 Connecté en tant que: **{st.session_state.username}**")
        if st.button("🚪 Se déconnecter"):
            logout()

    # Onglets
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

        # Recherche
        film_input = st.text_input("Entrez le titre du film :")
        if st.button("Chercher", key="btn_search"):
            st.session_state["fuzzy_matches_1"] = None
            st.session_state["chosen_film"] = None
            if film_input:
                try:
                    response = api_get(f"fuzzy_match/{film_input}")
                    if response.status_code == 200:
                        matches = response.json().get("matches", [])
                        st.session_state["fuzzy_matches_1"] = matches if matches else []
                        if not matches:
                            st.warning("⚠️ Aucun film trouvé avec ce titre.")
                    else:
                        st.error("❌ Erreur lors de la recherche.")
                except requests.exceptions.RequestException:
                    st.error("❌ Erreur de connexion avec le serveur")

        # Sélection du film
        if st.session_state.get("fuzzy_matches_1"):
            matches_info = []
            for match in st.session_state["fuzzy_matches_1"]:
                poster_url = None
                movie_id = match.get("movie_id")
                try:
                    details_resp = api_get(f"movie-details/{match['title']}")
                    if details_resp.status_code == 200:
                        details = details_resp.json()
                        poster_url = details.get("poster_url")
                        movie_id = details.get("movie_id", movie_id)
                except requests.exceptions.RequestException:
                    pass

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
                            st.image(match["poster"], use_container_width=True)
                        st.caption(match.get("title", "Titre inconnu"))
                        unique_key = f"select_{match['movie_id']}_{row}_{col_idx}"
                        if st.session_state.get("chosen_film") == match["title"]:
                            st.button("✅ Sélectionné", key=unique_key, disabled=True)
                        else:
                            if st.button("Sélectionner", key=unique_key):
                                st.session_state["chosen_film"] = match["title"]

        # Notation
        chosen_film = st.session_state.get("chosen_film")
        if chosen_film:
            st.success(f"🎬 Film sélectionné : {chosen_film}")
            note_input = st.number_input("Note du film (0.0 - 10.0)", min_value=0.0, max_value=10.0,
                                         value=5.0, step=0.1, format="%.1f", key="note_input")
            if st.button("Valider la note"):
                payload = {"title": chosen_film, "rating": note_input}
                update_resp = api_post("update_rating", payload)
                if update_resp.status_code == 200:
                    st.success(f"✅ La note {note_input} a été enregistrée pour '{chosen_film}' !")
                else:
                    detail = update_resp.json().get("detail", "Erreur inconnue")
                    st.error(f"❌ Échec : {detail}")

        # Recommandations personnalisées (avec run_id + feedback)
        if chosen_film:
            st.subheader("🔍 Obtenir une recommandation personnalisée")
            if st.button("Me recommander des films", key="btn_reco"):
                try:
                    with start_ui_run(input_title=chosen_film, user=st.session_state.username):
                        mlflow.set_tags({"source": "streamlit", "stage": "inference_ui"})
                        mlflow.log_param("user", st.session_state.username)
                        mlflow.log_param("film_input", chosen_film)

                        # Appel API
                        response = api_get(f"recommend_xgb_personalized/{chosen_film}", params={"top_k": 5})
                        if response.status_code != 200:
                            st.error(response.json().get("detail", "Erreur inconnue"))
                        else:
                            payload = response.json()
                            st.session_state["last_run_id"] = payload.get("run_id")
                            recos = payload.get("recommendations", [])
                            mlflow.log_param("n_recos", int(len(recos)))

                            # Log des scores (UI) dans le nested run
                            for i, reco in enumerate(recos, start=1):
                                mlflow.log_metric("pred_score", float(reco.get("pred_score", 0.0)), step=i)

                            # Artifact JSON des recos UI
                            # with tempfile.TemporaryDirectory() as tmp:
                            #     path_json = os.path.join(tmp, "streamlit_recommendations.json")
                            #     with open(path_json, "w", encoding="utf-8") as f:
                            #         json.dump(recos, f, ensure_ascii=False, indent=2)
                            #     mlflow.log_artifact(path_json)

                            if recos:
                                st.success("🎯 Recommandations trouvées !")
                                st.session_state["current_recos"] = recos
                            else:
                                st.info("Aucune recommandation trouvée pour ce film")
                except requests.exceptions.RequestException:
                    st.error("❌ Erreur de connexion avec le serveur")
                except Exception as e:
                    st.error(f"❌ Erreur MLflow : {e}")
                    st.text(traceback.format_exc())

            # Affichage + Feedback
            recos = st.session_state.get("current_recos", [])
            run_id = st.session_state.get("last_run_id")

            def send_feedback(reco_title: str, pred_label: int, pred_score: float, liked: int):
                if not run_id:
                    st.error("Run MLflow introuvable. Relance une recommandation.")
                    return
                try:
                    fb = {
                        "run_id": run_id,
                        "user_name": st.session_state.username,
                        "input_title": st.session_state.get("chosen_film"),
                        "reco_title": reco_title,
                        "pred_label": int(pred_label),
                        "pred_score": float(pred_score),
                        "liked": int(liked)
                    }
                    resp = api_post("log_feedback", fb)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.toast(f"✅ Feedback pris en compte. Accuracy: {data.get('online_accuracy'):.2%} (n={data.get('count')})")
                    else:
                        st.error(resp.json().get("detail", "Erreur lors du feedback"))
                except requests.exceptions.RequestException:
                    st.error("❌ Erreur de connexion avec le serveur")

            if recos:
                for reco in recos:
                    cols = st.columns([1, 3])
                    with cols[0]:
                        if reco.get("poster_url"):
                            st.image(reco["poster_url"], use_container_width=True)
                    with cols[1]:
                        reco_title = reco.get("title", "Titre inconnu")
                        raw_genres = reco.get("genres", [])
                        genres = raw_genres if isinstance(raw_genres, list) else [
                            g.strip() for g in str(raw_genres).split(",")
                        ]
                        reco_synopsis = reco.get("synopsis", "Pas de synopsis disponible.")
                        score = float(reco.get("pred_score", 0.0))
                        pred_label = int(score >= 0.5)

                        st.markdown(f"### 🎬 {reco_title}")
                        st.markdown(f"**Ce film est susceptible de vous plaire à {int(score*100)}%**")
                        st.write(f"**Genres :** {', '.join(genres) if genres else 'N/A'}")
                        st.write(reco_synopsis)

                        b1, b2 = st.columns(2)
                        with b1:
                            if st.button("👍 Ça m'intéresse", key=f"like_{reco_title}_{hash(score)}", disabled=(run_id is None)):
                                send_feedback(reco_title, pred_label, score, liked=1)
                        with b2:
                            if st.button("👎 Pas pour moi", key=f"dislike_{reco_title}_{hash(score)}", disabled=(run_id is None)):
                                send_feedback(reco_title, pred_label, score, liked=0)

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
                    else:
                        st.error(response.json().get("detail", "Erreur lors de la récupération des films."))

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
                            st.image(poster, use_container_width=True)
                        with cols[1]:
                            title = movie.get("title", "Titre inconnu")
                            raw_genres = movie.get("genres", [])
                            genres = raw_genres if isinstance(raw_genres, list) else [g.strip() for g in str(raw_genres).split(",")]
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
                        st.session_state["fuzzy_matches_3"] = matches if matches else []
                        if not matches:
                            st.warning("⚠️ Aucun film trouvé avec ce titre.")
                    else:
                        st.error(fuzzy_resp.json().get("detail", "Erreur lors du fuzzy match."))
                except requests.exceptions.RequestException:
                    st.error("❌ Erreur de connexion avec le serveur")
            else:
                st.warning("⚠️ Veuillez entrer un titre de film")

        if st.session_state.get("fuzzy_matches_3"):
            chosen_movie = st.selectbox("Films correspondants :", [m["title"] for m in st.session_state["fuzzy_matches_3"]], key="chosen_movie_details")
            if st.button("✅ Confirmer ce film"):
                try:
                    response = api_get(f"movie-details/{chosen_movie}")
                    if response.status_code == 200:
                        details = response.json()
                        st.success("✅ Détails du film trouvés !")
                        col1, col2 = st.columns([1,2])
                        with col1:
                            if details.get("poster_url"):
                                st.image(details["poster_url"], use_container_width=True)
                        with col2:
                            st.markdown(f"### 🎬 {details['title']} ({details.get('releaseYear', 'N/A')})")
                            st.write(f"**Genres :** {details.get('genres', 'N/A')}")
                            st.write(f"**Note :** {details.get('rating', 'N/A')}")
                            plats = details.get("platforms", [])
                            st.write(f"**Plateformes disponibles :** {', '.join(plats) if plats else 'N/A'}")
                            st.write(details.get("synopsis", "Pas de synopsis disponible."))
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
    if 'current_recos' not in st.session_state:
        st.session_state["current_recos"] = []
    if 'last_run_id' not in st.session_state:
        st.session_state["last_run_id"] = None

    if st.session_state.authenticated:
        main_app()
    else:
        login_page()

if __name__ == "__main__":
    main()








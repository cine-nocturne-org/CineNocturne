# -----------------------------
# Import
# -----------------------------
import os
import sys
import traceback

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
# Fonctions de connexion
# -----------------------------
def login_page():
    """Affiche la page de connexion"""
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
    return requests.get(f"{API_URL}{endpoint}", headers=headers, params=params)

def api_post(endpoint: str, payload: dict):
    headers = {"Authorization": f"Bearer {st.session_state.api_token}"}
    return requests.post(f"{API_URL}{endpoint}", headers=headers, json=payload)
    
# -----------------------------
# Fonctions selection avec poster
# -----------------------------
def film_selector(matches: list, state_key_prefix: str):
    matches_info = []
    for match in matches:
        movie_id = match.get("movie_id")
        details_resp = api_get(f"movie-details/{match['title']}")
        poster_url = None
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
                unique_key = f"{state_key_prefix}_{match['movie_id']}_{row}_{col_idx}"
                if st.session_state.get(f"{state_key_prefix}_chosen") == match["title"]:
                    st.button("✅ Sélectionné", key=unique_key, disabled=True)
                else:
                    if st.button("Sélectionner", key=unique_key):
                        st.session_state[f"{state_key_prefix}_chosen"] = match["title"]
                        

# -----------------------------
# Header commun
# -----------------------------
def render_header():
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <!-- Logo + titre -->
            <div style="display: flex; align-items: center;">
                <img src='https://github.com/PixelLouve/CineNocturne/blob/main/E3_E4_API_app/logo_cinenocturne.png?raw=true' width='120'>
                <h1 style="margin-left: 15px; margin-top: 5px;">🍿 Recommandation de Films Personnalisée</h1>
            </div>
        """,
        unsafe_allow_html=True
    )

# Si l'utilisateur est connecté → afficher pseudo + bouton
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
# Header avec logo et titres
# -----------------------------
def main_app():
    render_header()

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
    
        # -----------------------------
        # Bouton Nouvelle recherche
        # -----------------------------
        if st.button("🔄 Nouvelle recherche", key="btn_reset_tab1"):
            st.session_state["fuzzy_matches_tab1"] = None
            st.session_state["tab1_chosen"] = None
            st.session_state["already_recommended"] = set()
            st.session_state["last_recos"] = []
    
        # -----------------------------
        # Entrée titre film
        # -----------------------------
        film_input = st.text_input("Entrez le titre du film :")
        if st.button("Chercher", key="btn_tab1"):
            st.session_state["fuzzy_matches_tab1"] = None
            st.session_state["tab1_chosen"] = None
            st.session_state["already_recommended"] = set()
            if film_input:
                try:
                    response = api_get(f"fuzzy_match/{film_input}", params={"top_k": 10})
                    if response.status_code == 200:
                        matches = response.json().get("matches", [])
                        st.session_state["fuzzy_matches_tab1"] = matches if matches else []
                        if not matches:
                            st.warning("⚠️ Aucun film trouvé avec ce titre.")
                    else:
                        st.error("❌ Erreur lors de la recherche.")
                except requests.exceptions.RequestException:
                    st.error("❌ Erreur de connexion avec le serveur")
    
        # -----------------------------
        # Sélection du film
        # -----------------------------
        if st.session_state.get("fuzzy_matches_tab1"):
            film_selector(st.session_state["fuzzy_matches_tab1"], "tab1")
    
        chosen_film = st.session_state.get("tab1_chosen")
        if chosen_film:
            st.success(f"🎬 Film sélectionné : {chosen_film}")
    
            # -----------------------------
            # Saisie de la note
            # -----------------------------
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
                            update_resp = api_post("update_rating", payload)
                            if update_resp.status_code == 200:
                                st.success(f"✅ La note {note_float} a été enregistrée pour '{chosen_film}' !")
                            else:
                                detail = update_resp.json().get("detail", "Erreur inconnue")
                                st.error(f"❌ Échec : {detail}")
                        else:
                            st.warning("⚠️ La note doit être comprise entre 0 et 10")
                    except ValueError:
                        st.warning("⚠️ Veuillez saisir un nombre valide")
    
            # -----------------------------
            # Gestion des recommandations
            # -----------------------------
            if "already_recommended" not in st.session_state:
                st.session_state["already_recommended"] = set()
            if "last_recos" not in st.session_state:
                st.session_state["last_recos"] = []
    
            def fetch_recommendations():
                """
                Récupère les recommandations depuis l'API et logge params + metrics dans MLflow.
                Inclut movie_rating, pred_score, score_diff et score_final pour chaque film.
                """
                try:
                    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
                    chosen_film = st.session_state.get("tab1_chosen")
                    if not chosen_film:
                        st.warning("⚠️ Aucun film sélectionné")
                        return
            
                    run_name = f"streamlit_reco_{st.session_state.username}_{chosen_film}"
                    with mlflow.start_run(run_name=run_name) as run:
            
                        mlflow.log_param("user", st.session_state.username)
                        mlflow.log_param("film_input", chosen_film)
            
                        response = api_get(f"recommend_xgb_personalized/{chosen_film}", params={"top_k": 20})
                        if response.status_code == 200:
                            recos = response.json()
                            new_recos = [r for r in recos if r["title"] not in st.session_state.get("already_recommended", set())][:10]
            
                            for idx, r in enumerate(new_recos):
                                title = r["title"]
                                st.session_state["already_recommended"].add(title)
            
                                # Log des metrics spécifiques par film
                                mlflow.log_metric(f"{title}_movie_rating", r.get("movie_rating", 0))
                                mlflow.log_metric(f"{title}_pred_score", r.get("pred_score", 0))
                                mlflow.log_metric(f"{title}_score_diff", r.get("score_diff", 0))
                                mlflow.log_metric(f"{title}_final_score", r.get("final_score", 0))
            
                            st.session_state["last_recos"] = new_recos
            
                        else:
                            st.error(response.json().get("detail", "Erreur inconnue"))
                            st.session_state["last_recos"] = []
            
                except Exception as e:
                    st.error(f"❌ Erreur : {e}")
                    st.session_state["last_recos"] = []
                    mlflow.log_metric("error_flag", 1)

    
            # -----------------------------
            # Bouton pour obtenir les recommandations
            # -----------------------------
            if st.button("🎯 Me proposer des recommandations"):
                fetch_recommendations()
    
            # -----------------------------
            # Affichage des recommandations
            # -----------------------------
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
    
                # -----------------------------
                # Bouton pour obtenir d'autres recommandations
                # -----------------------------
                if st.button("🔄 Me proposer d'autres recommandations"):
                    fetch_recommendations()


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
                            st.markdown(f"### 🎬 {title}")
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
    
        # Bouton pour réinitialiser la recherche
        if st.button("🔄 Nouvelle recherche", key="btn_reset_tab3"):
            st.session_state["fuzzy_matches_tab3"] = None
            st.session_state["tab3_chosen"] = None
            st.session_state["details_title"] = ""

    
        film_details_title = st.text_input("Titre du film :", key="details_title")
    
        if st.button("🔍 Chercher correspondances", key="btn_tab3"):
            if film_details_title:
                try:
                    fuzzy_resp = api_get(f"fuzzy_match/{film_details_title}")
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
    
        # Sélection du film
        if st.session_state.get("fuzzy_matches_tab3") and not st.session_state.get("tab3_chosen"):
            film_selector(st.session_state["fuzzy_matches_tab3"], "tab3")
        
        # Affichage détails du film choisi
        chosen_film = st.session_state.get("tab3_chosen")
        if chosen_film:
            st.success(f"🎬 Film sélectionné : {chosen_film}")
            try:
                response = api_get(f"movie-details/{chosen_film}")
                if response.status_code == 200:
                    details = response.json()
                    col1, col2 = st.columns([1, 2])
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


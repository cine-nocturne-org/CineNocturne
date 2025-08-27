import streamlit as st
import requests
from urllib.parse import quote

from requests.auth import HTTPBasicAuth

API_URL = "https://cinenocturne.onrender.com/"
# cmd lancemennt : streamlit run reco_app_v2.py

# Configuration de la page
st.set_page_config(page_title="Recommandation de Films", page_icon="🍿​")

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
                # Test de connexion avec l'API
                try:
                    test_response = requests.get(f"{API_URL}/genres/", auth=HTTPBasicAuth(username, password), timeout=5)
                    if test_response.status_code == 200:
                        # Connexion réussie - stocker les infos dans la session
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.password = password
                        st.success("Connexion réussie ! 🎉")
                        st.rerun()
                    else:
                        st.error("❌ Identifiants incorrects")
                except requests.exceptions.RequestException:
                    st.error("❌ Impossible de se connecter au serveur")
            else:
                st.warning("⚠️ Veuillez remplir tous les champs")

def logout():
    """Déconnecte l'utilisateur"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.password = None
    st.rerun()

def main_app():
    """Application principale après connexion"""
    # Header avec bouton de déconnexion
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🎮 Recommandation de Films Personnalisée")
    with col2:
        st.write(f"👋 Connecté en tant que: **{st.session_state.username}**")
        if st.button("🚪 Se déconnecter"):
            logout()

    # Récupération des identifiants depuis la session
    USERNAME = st.session_state.username
    PASSWORD = st.session_state.password

    # ------------------------------
    # Onglets
    # ------------------------------
    tab1, tab2, tab3 = st.tabs(["✨​ Recommandations perso", "🎲 Suggestions aléatoires", "📺​ Plateformes disponibles"])


    # ------------------------------
    # Onglet 1 : Film vu + reco perso
    # ------------------------------
    # ------------------------------
    # Section "Film vu / Notation"
    # ------------------------------
    with tab1:
        st.subheader("✨​​ Noter un film que vous avez vu")
    
        # 1️⃣ Entrée titre + bouton Chercher
        film_input = st.text_input("Entrez le titre du film :")
    
        if st.button("Chercher", key="btn_search"):
            st.session_state["fuzzy_matches"] = None
            st.session_state["chosen_film"] = None
            if film_input:
                try:
                    response = requests.get(
                        f"{API_URL}/fuzzy_match/{film_input}",
                        params={"top_k": 10},
                        auth=HTTPBasicAuth(USERNAME, PASSWORD)
                    )
                    if response.status_code == 200:
                        matches = response.json().get("matches", [])
                        if matches:
                            st.session_state["fuzzy_matches"] = matches
                        else:
                            st.warning("⚠️ Aucun film trouvé avec ce titre.")
                    else:
                        st.error("❌ Erreur lors de la recherche.")
                except requests.exceptions.RequestException:
                    st.error("❌ Erreur de connexion avec le serveur")
    
        # 2️⃣ Sélection du film
        if st.session_state.get("fuzzy_matches"):
            matches_info = []
            for match in st.session_state["fuzzy_matches"]:
                details_resp = requests.get(
                    f"{API_URL}/movie-details/{match['title']}",
                    auth=HTTPBasicAuth(USERNAME, PASSWORD)
                )
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
                            st.image(match["poster"], width=120)
                        st.caption(match.get("title", "Titre inconnu"))
                        unique_key = f"select_{match['movie_id']}_{row}_{col_idx}"
                        if st.session_state.get("chosen_film") == match["title"]:
                            st.button("✅ Sélectionné", key=unique_key, disabled=True)
                        else:
                            if st.button("Sélectionner", key=unique_key):
                                st.session_state["chosen_film"] = match["title"]
    
        # 3️⃣ Notation du film choisi uniquement
        chosen_film = st.session_state.get("chosen_film")
        if chosen_film:
            st.success(f"🎬 Film sélectionné : {chosen_film}")
            note_input = st.number_input(
                "Note du film (0.0 - 10.0)",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.1,
                format="%.1f",
                key="note_input"
            )
            if st.button("Valider la note"):
                payload = {"title": chosen_film, "rating": note_input}
                update_resp = requests.post(
                    f"{API_URL}/update_rating",
                    json=payload,
                    auth=HTTPBasicAuth(USERNAME, PASSWORD)
                )
                if update_resp.status_code == 200:
                    st.success(f"✅ La note {note_input} a été enregistrée pour '{chosen_film}' !")
                else:
                    detail = update_resp.json().get("detail", "Erreur inconnue")
                    st.error(f"❌ Échec : {detail}")
    
        # 4️⃣ Recommandations personnalisées (infos seulement sur les films recommandés)
        if chosen_film:
            st.subheader("🔍 Obtenir une recommandation personnalisée")
            if st.button("Me recommander un film", key="btn_reco"):
                try:
                    response = requests.get(
                        f"{API_URL}/recommend_xgb_personalized/{chosen_film}",
                        params={"top_k": 5},
                        auth=HTTPBasicAuth(USERNAME, PASSWORD)
                    )
                    if response.status_code == 200:
                        recos = response.json()
                        if recos:
                            st.success("🎯 Recommandations trouvées !")
                            for reco in recos:
                                cols = st.columns([1, 3])
                                with cols[0]:
                                    if reco.get("poster_url"):
                                        st.image(reco["poster_url"], width=150)
                                with cols[1]:
                                    # Variables spécifiques à la reco
                                    reco_title = reco.get("title", "Titre inconnu")
                                    reco_year = reco.get("releaseYear") or reco.get("release_year") or "N/A"
                                    reco_genres_raw = reco.get("genres", [])
                                    if isinstance(reco_genres_raw, str):
                                        reco_genres = [g.strip() for g in reco_genres_raw.split(",")]
                                    elif isinstance(reco_genres_raw, list):
                                        reco_genres = reco_genres_raw
                                    else:
                                        reco_genres = []
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







    # ------------------------------
    # Onglet 2 : Suggestions aléatoires
    # ------------------------------
    # 4. Suggestions aléatoires par genre et plateformeswith tab2:
    with tab2:
        st.subheader("🎲 Suggestions aléatoires par genre")
        try:
            genre_response = requests.get(f"{API_URL}/genres/", auth=HTTPBasicAuth(USERNAME, PASSWORD))
            if genre_response.status_code == 200:
                genre_list = genre_response.json()
    
                # --- Formulaire sélection genre et plateformes ---
                with st.form("random_movies_form"):
                    selected_genre = st.selectbox("Choisissez un genre", genre_list)
                    selected_platforms = st.multiselect(
                        "Choisissez les plateformes", 
                        ["netflix", "prime", "hulu", "hbo", "apple"]
                    )
                    submitted = st.form_submit_button("Afficher des films aléatoires")
    
                # Init session state
                if "already_seen_movies" not in st.session_state:
                    st.session_state["already_seen_movies"] = set()
                if "current_movies" not in st.session_state:
                    st.session_state["current_movies"] = []
    
                def fetch_random_movies():
                    """Récupère de nouveaux films sans doublons"""
                    params = {
                        "genre": selected_genre,
                        "platforms": selected_platforms,
                        "limit": 20  # en demander un peu plus pour éviter doublons
                    }
                    response = requests.get(
                        f"{API_URL}/random_movies/", 
                        params=params, 
                        auth=HTTPBasicAuth(USERNAME, PASSWORD)
                    )
                    if response.status_code == 200:
                        movies = response.json()
                        # Filtrer ceux déjà vus
                        fresh_movies = [
                            m for m in movies if m["title"] not in st.session_state["already_seen_movies"]
                        ]
                        # En garder 10 max
                        fresh_movies = fresh_movies[:10]
                        # Mémoriser
                        for m in fresh_movies:
                            st.session_state["already_seen_movies"].add(m["title"])
                        st.session_state["current_movies"] = fresh_movies
    
                # --- Premier affichage ---
                if submitted:
                    st.session_state["already_seen_movies"].clear()
                    fetch_random_movies()
    
                # --- Bouton pour nouvelles suggestions ---
                if st.session_state["current_movies"]:
                    if st.button("🔄 Proposer d'autres films"):
                        fetch_random_movies()
    
                    # Affichage
                    for movie in st.session_state["current_movies"]:
                        cols = st.columns([1, 3])
                        with cols[0]:
                            if movie.get("poster_url"):
                                st.image(movie["poster_url"], use_container_width=True)
                        with cols[1]:
                            # Récup infos propres
                            title = movie.get("title", "Titre inconnu")
                            year = movie.get("releaseYear") or movie.get("release_year") or "N/A"
    
                            raw_genres = movie.get("genres", [])
                            if isinstance(raw_genres, str):
                                genres = [g.strip() for g in raw_genres.split(",")]
                            elif isinstance(raw_genres, list):
                                genres = raw_genres
                            else:
                                genres = []
    
                            st.markdown(f"### 🎬 {title} ({year})")
                            st.write(f"**Genres :** {', '.join(genres) if genres else 'N/A'}")
                            st.write(movie.get('synopsis', 'Pas de synopsis disponible.'))
    
        except Exception as e:
            st.error(f"❌ Impossible de se connecter pour récupérer les genres : {e}")

    # ------------------------------
    # Onglet 3 : Plateformes dispo
    # ------------------------------
    # 5. Plateformes disponibles pour un film
    with tab3:
        st.subheader("📺 Plateformes disponibles pour un film")
        film_details_title = st.text_input("Titre du film :", key="details_title_tab3")
    
        # Bouton de recherche
        if st.button("🔍 Chercher correspondances", key="btn_fuzzy_tab3"):
            if film_details_title:
                try:
                    fuzzy_resp = requests.get(
                        f"{API_URL}/fuzzy_match/{film_details_title}",
                        auth=HTTPBasicAuth(USERNAME, PASSWORD)
                    )
    
                    if fuzzy_resp.status_code == 200:
                        matches = fuzzy_resp.json().get("matches", [])
                        if matches:
                            st.session_state["fuzzy_matches_platforms"] = matches
                            st.session_state["chosen_platform_movie"] = None  # reset film choisi
                        else:
                            st.warning("⚠️ Aucun film trouvé avec ce titre.")
                    else:
                        st.error(fuzzy_resp.json().get("detail", "Erreur lors du fuzzy match."))
                except requests.exceptions.RequestException:
                    st.error("❌ Erreur de connexion avec le serveur")
            else:
                st.warning("⚠️ Veuillez entrer un titre de film")
    
        # Affichage des correspondances pour sélection
        if st.session_state.get("fuzzy_matches_platforms"):
            matches_info_platforms = []
            for match in st.session_state["fuzzy_matches_platforms"]:
                details_resp = requests.get(
                    f"{API_URL}/movie-details/{match['movie_id']}",
                    auth=HTTPBasicAuth(USERNAME, PASSWORD)
                )
                poster_url = "https://via.placeholder.com/120x180?text=Pas+de+poster"
                release_year = "N/A"
                if details_resp.status_code == 200:
                    details = details_resp.json()
                    poster_url = details.get("poster_url") or poster_url
                    release_year = details.get("releaseYear") or details.get("release_year") or release_year
                matches_info_platforms.append({
                    "title": match["title"],
                    "poster": poster_url,
                    "movie_id": match["movie_id"],
                    "releaseYear": release_year
                })
        
            st.markdown("### Sélectionnez le film correct :")
            rows, cols_per_row = 2, 5
            for row in range(rows):
                row_matches = matches_info_platforms[row*cols_per_row : (row+1)*cols_per_row]
                if not row_matches:
                    continue
                cols = st.columns(len(row_matches))
                for col_idx, match in enumerate(row_matches):
                    with cols[col_idx]:
                        st.image(match["poster"], width=120)
                        st.caption(match.get('title', 'Titre inconnu'))
                        unique_key = f"select_platform_{match['movie_id']}_{row}_{col_idx}"
                        if st.session_state.get("chosen_platform_movie") == match["movie_id"]:
                            st.button("✅ Sélectionné", key=unique_key, disabled=True)
                        else:
                            if st.button("Sélectionner", key=unique_key):
                                st.session_state["chosen_platform_movie"] = match["movie_id"]

    
        # Affichage fiche complète si film sélectionné
        chosen_movie_id = st.session_state.get("chosen_platform_movie")
        if chosen_movie_id:
            try:
                response = requests.get(
                    f"{API_URL}/movie-details/{chosen_movie_id}",
                    auth=HTTPBasicAuth(USERNAME, PASSWORD)
                )
                if response.status_code == 200:
                    details = response.json()
                    st.success("✅ Détails du film trouvés !")
                    st.image(details.get("poster_url") or "https://via.placeholder.com/150x225?text=Pas+de+poster", width=150)
    
                    # Genres robustes
                    raw_genres = details.get("genres", [])
                    if isinstance(raw_genres, str):
                        genres = [g.strip() for g in raw_genres.split(",")]
                    elif isinstance(raw_genres, list):
                        genres = raw_genres
                    else:
                        genres = []
    
                    # Année avec fallback
                    year = details.get("releaseYear") or details.get("release_year") or "N/A"
    
                    st.markdown(f"### 🎬 {details['title']} ({year})")
                    st.write(f"**Genres :** {', '.join(genres) if genres else 'N/A'}")
                    st.write(f"**Note :** {details.get('rating', 'N/A')}")
                    st.write(f"**Plateformes disponibles :** {', '.join(details.get('platforms', [])) if details.get('platforms') else 'Indisponible'}")
                    st.write(details.get('synopsis', 'Pas de synopsis disponible.'))
                else:
                    st.error(response.json().get("detail", "Film non trouvé"))
            except requests.exceptions.RequestException:
                st.error("❌ Erreur de connexion avec le serveur")




# Point d'entrée principal
def main():
    """Point d'entrée principal de l'application"""
    # Initialiser l'état de session si nécessaire
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'password' not in st.session_state:
        st.session_state.password = None

    # Afficher la page appropriée
    if st.session_state.authenticated:
        main_app()
    else:
        login_page()

if __name__ == "__main__":
    main()








import streamlit as st
import requests
from urllib.parse import quote

from requests.auth import HTTPBasicAuth

API_URL = "https://cinenocturne.onrender.com/docs"
# cmd lancemennt : streamlit run reco_app_v2.py

# Configuration de la page
st.set_page_config(page_title="Recommandation de Films", page_icon="🎮")

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
    tab1, tab2, tab3 = st.tabs(["⭐ Recommandations perso", "🎲 Suggestions aléatoires", "📄 Plateformes dispo"])


    # ------------------------------
    # Onglet 1 : Film vu + reco perso
    # ------------------------------
    # ------------------------------
    # Section "Film vu / Notation"
    # ------------------------------
    with tab1:
        st.subheader("📝 Noter un film que vous avez vu")

        film_input = st.text_input("Entrez le titre du film :")
        note_input = st.number_input(
            "Note du film (0.0 - 10.0)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1,
            format="%.1f"
        )

        if film_input:
            try:
                # 1️⃣ Fuzzy match
                response = requests.get(
                    f"{API_URL}/fuzzy_match/{film_input}",
                    params={"top_k": 10},
                    auth=HTTPBasicAuth(USERNAME, PASSWORD)
                )

                if response.status_code == 200:
                    data = response.json()
                    matches = data.get("matches", [film_input])

                    # 2️⃣ Récupération posters et titres
                    matches_info = []
                    for match in matches:  # match est déjà un dict
                        details_resp = requests.get(
                            f"{API_URL}/movie-details/{match['title']}",
                            auth=HTTPBasicAuth(USERNAME, PASSWORD)
                        )
                        poster_url = None
                        movie_id = match.get("movie_id")  # par défaut depuis fuzzy_match
                        if details_resp.status_code == 200:
                            details = details_resp.json()
                            poster_url = details.get("poster_url")
                            movie_id = details.get("movie_id", movie_id)
                        matches_info.append({
                            "title": match["title"],
                            "poster": poster_url,
                            "movie_id": movie_id
                        })

                    # 🔥 Suppression des doublons par movie_id
                    unique_movies: dict[str, dict] = {}
                    for movie in matches_info:
                        movie_key = str(movie["movie_id"])  # toujours hashable
                        if movie_key not in unique_movies:
                            unique_movies[movie_key] = movie



                    # 3️⃣ Affichage grille 2x5 pour les fuzzy matches
                    rows = 2
                    cols_per_row = 5
                    chosen_film = st.session_state.get("chosen_film", None)

                    for row in range(rows):
                        start_idx = row * cols_per_row
                        end_idx = start_idx + cols_per_row
                        row_matches = matches_info[start_idx:end_idx]

                        if not row_matches:
                            continue

                        cols = st.columns(len(row_matches))
                        for col_idx, match in enumerate(row_matches):
                            with cols[col_idx]:
                                # Affichage poster si disponible
                                if match.get("poster"):
                                    st.image(match["poster"], width=120)
                                
                                st.caption(match.get("title", "Titre inconnu"))

                                # Clé unique sécurisée
                                title_str = match.get("title") or f"unknown_{row}_{col_idx}"
                                safe_title = title_str.replace(" ", "_").replace(":", "_").replace("-", "_")
                                unique_key = f"select_{safe_title}_{row}_{col_idx}"

                                if st.button("Sélectionner", key=unique_key):
                                    st.session_state["chosen_film"] = match.get("title")


                    # 4️⃣ Validation après sélection
                    if chosen_film:
                        st.success(f"🎬 Film sélectionné : {chosen_film}")
                        if st.button("Valider le film vu et la note"):
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


                else:
                    st.warning("Aucune correspondance trouvée pour ce titre.")
            except requests.exceptions.RequestException:
                st.error("❌ Erreur de connexion avec le serveur")




        # 3. Recommandation personnalisée
        st.subheader("🔍 Obtenir une recommandation personnalisée")

        if st.button("Me recommander un film", key="btn_reco"):
            chosen_film = st.session_state.get("chosen_film", None)
            if chosen_film:
                try:
                    # On appelle le endpoint XGB ultra-rapide
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
                                # Créer deux colonnes : gauche = poster, droite = texte
                                cols = st.columns([1, 3])  # ratio 1:3
                                with cols[0]:
                                    if reco.get("poster_url"):
                                        st.image(reco["poster_url"], width=150)
                                with cols[1]:
                                    st.markdown(f"### 🎥 {reco['title']}")
                                    score_pct = int(reco.get("pred_score", 0) * 100)
                                    st.markdown(f"**Ce film est susceptible de vous plaire à {score_pct}%**")
                                    st.write(f"**Genres :** {reco.get('genres')}")
                                    st.write(reco.get('synopsis'))
                        else:
                            st.info("Aucune recommandation trouvée pour ce film")
                    else:
                        st.error(response.json().get("detail", "Erreur inconnue"))
                except requests.exceptions.RequestException:
                    st.error("❌ Erreur de connexion avec le serveur")
            else:
                st.warning("⚠️ Veuillez entrer un titre de film pour les recommandations")



    # ------------------------------
    # Onglet 2 : Suggestions aléatoires
    # ------------------------------
    # 4. Suggestions aléatoires par genre et plateformes
    with tab2:
        st.subheader("🎲 Suggestions aléatoires par genre")
        try:
            genre_response = requests.get(f"{API_URL}/genres/", auth=HTTPBasicAuth(USERNAME, PASSWORD))
            if genre_response.status_code == 200:
                genre_list = genre_response.json()
                selected_genre = st.selectbox("Choisissez un genre", genre_list)
                selected_platforms = st.multiselect("Choisissez les plateformes", ["netflix", "prime", "hulu", "hbo", "apple"], key="random_platforms")

                if st.button("Afficher des films aléatoires"):
                    params = {"genre": selected_genre, "platforms": selected_platforms, "limit": 10}
                    random_response = requests.get(f"{API_URL}/random_movies/", params=params, auth=HTTPBasicAuth(USERNAME, PASSWORD))
                    if random_response.status_code == 200:
                        movies = random_response.json()
                        for movie in movies:
                            if movie.get("poster_url"):
                                st.image(movie["poster_url"], width=150)
                            st.markdown(f"### 🎮 {movie['title']} ({movie['platform']})")
                            st.write(f"**Genres :** {movie['genres']}")
                            st.write(movie['synopsis'])
        except:
            st.error("❌ Impossible de se connecter pour récupérer les genres")


    # ------------------------------
    # Onglet 3 : Plateformes dispo
    # ------------------------------
    # 5. Plateformes disponibles pour un film
    with tab3:
        st.subheader("📄 Plateformes disponibles pour un film")
        film_details_title = st.text_input("Titre du film :", key="details_title")
        if st.button("🔍 Chercher correspondances", key="btn_fuzzy"):
            if film_details_title:
                try:
                    fuzzy_resp = requests.get(
                        f"{API_URL}/fuzzy_match/{film_details_title}",
                        auth=HTTPBasicAuth(USERNAME, PASSWORD)
                    )

                    if fuzzy_resp.status_code == 200:
                        matches = fuzzy_resp.json().get("matches", [])
                        if matches:
                            st.session_state["fuzzy_matches"] = matches
                        else:
                            st.warning("⚠️ Aucun film trouvé avec ce titre.")
                    else:
                        st.error(fuzzy_resp.json().get("detail", "Erreur lors du fuzzy match."))

                except requests.exceptions.RequestException:
                    st.error("❌ Erreur de connexion avec le serveur")
            else:
                st.warning("⚠️ Veuillez entrer un titre de film")

    # --- Étape 2 : choix du film ---
    if "fuzzy_matches" in st.session_state:
        chosen_movie = st.selectbox(
            "Films correspondants :", 
            st.session_state["fuzzy_matches"], 
            key="chosen_movie_details"
        )

        # --- Étape 3 : confirmation ---
        if st.button("✅ Confirmer ce film"):
            try:
                response = requests.get(
                    f"{API_URL}/movie-details/{chosen_movie}",
                    auth=HTTPBasicAuth(USERNAME, PASSWORD)
                )
                if response.status_code == 200:
                    details = response.json()
                    st.success("✅ Détails du film trouvés !")
                    if details.get("poster_url"):
                        st.image(details["poster_url"], width=150)
                    st.markdown(f"### 🎬 {details['title']} ({details['releaseYear']})")
                    st.write(f"**Genres :** {details['genres']}")
                    st.write(f"**Note :** {details['rating']}")
                    st.write(f"**Plateformes disponibles :** {', '.join(details['platforms'])}")
                    st.write(details['synopsis'])
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
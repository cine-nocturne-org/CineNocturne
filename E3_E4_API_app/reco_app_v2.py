import pandas as pd
import streamlit as st
import requests
import traceback
import os
from dotenv import load_dotenv
import mlflow
import sys
import tempfile
import json
import math


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

# --- Helpers reco/pagination ---
def ensure_session_defaults():
    for k, v in {
        "current_recos": [],
        "reco_pool": [],
        "reco_shown_titles": [],
        "reco_page": 0,
        "page_size": 5,            # <- taille d’une page (contrôlée par le select_slider)
        "fuzzy_matches_1": None,
        "chosen_film": None,
        "last_run_id": None,
        "film_input": "",
    }.items():
        st.session_state.setdefault(k, v)

def reset_reco(full=False):
    st.session_state["current_recos"] = []
    st.session_state["reco_pool"] = []
    st.session_state["reco_shown_titles"] = []
    st.session_state["reco_page"] = 0
    st.session_state["last_run_id"] = None
    if full:
        st.session_state["chosen_film"] = None
        st.session_state["fuzzy_matches_1"] = None
        # ⚠️ NE PAS toucher à st.session_state["film_input"] ici (sinon StreamlitAPIException)

def page_from_pool():
    pool = st.session_state.get("reco_pool", [])
    shown = set(st.session_state.get("reco_shown_titles", []))
    size = st.session_state.get("page_size", 5)
    batch = [r for r in pool if r.get("title") not in shown][:size]
    st.session_state["current_recos"] = batch
    st.session_state["reco_shown_titles"].extend([r["title"] for r in batch])
    st.session_state["reco_page"] += 1
    return len(batch) > 0

def parse_genres(raw):
    if isinstance(raw, list):
        return raw
    if not raw:
        return []
    return [g.strip() for g in str(raw).replace("|", ",").split(",") if g.strip()]

# --- Callbacks sûrs pour modifier film_input ---
def reset_search_all():
    """Réinitialise tout + vide le champ de recherche (autorisé via callback)."""
    reset_reco(full=True)
    st.session_state["film_input"] = ""

def reset_only_reco():
    """Réinitialise uniquement la pagination/des recos (garde la recherche)."""
    reset_reco(full=False)


# -----------------------------
# Application principale
# -----------------------------
def main_app():
    ensure_session_defaults()
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🍿 Recommandation de Films Personnalisée")
    with col2:
        st.write(f"👋 Bonjour: **{st.session_state.username}**")
        if st.button("🚪 Se déconnecter"):
            logout()

    # Onglets
    tab1, tab2, tab3, tab4 = st.tabs([
    "✨​ Recommandations perso",
    "🎲 Suggestions aléatoires",
    "📺​ Plateformes disponibles",
    "📈 Mon tableau de bord"
    ])


    # ------------------------------
    # Onglet 1 : Film vu + reco perso
    # ------------------------------
    with tab1:
        st.subheader("✨ Noter un film que vous avez vu")

        # === Taille de page ===
        st.select_slider(
            "Films par page",
            options=[3, 5, 8, 10],
            value=st.session_state.get("page_size", 5),
            key="page_size",
            help="Nombre de recommandations affichées à la fois."
        )

        # === Recherche ===
        film_input = st.text_input("Entrez le titre du film :", key="film_input")
        c_search, c_reset_search = st.columns([1, 1])
        with c_search:
            if st.button("Chercher", key="btn_search"):
                reset_reco(full=True)
                st.session_state["fuzzy_matches_1"] = None
                st.session_state["chosen_film"] = None
                if film_input:
                    try:
                        with st.spinner("🔎 Recherche des correspondances…"):
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
        with c_reset_search:
            st.button("🧹 Réinitialiser la recherche", key="btn_reset_search", on_click=reset_search_all)

        # === Sélection du film ===
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
            cols_per_row = 5
            rows = math.ceil(len(matches_info) / cols_per_row)
            
            # CSS une seule fois (petite carte propre)
            st.markdown("""
            <style>
            .card { border:1px solid #eee; border-radius:12px; padding:8px; height:100%; }
            </style>
            """, unsafe_allow_html=True)
            
            for row in range(rows):
                row_matches = matches_info[row*cols_per_row : (row+1)*cols_per_row]
                if not row_matches:
                    continue
            
                cols = st.columns(cols_per_row, gap="small")
            
                # Centre la rangée si < 5 items
                offset = (cols_per_row - len(row_matches)) // 2
            
                for i, match in enumerate(row_matches):
                    with cols[offset + i]:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
            
                        if match.get("poster"):
                            st.image(match["poster"], width='stretch)
                        st.caption(match.get("title", "Titre inconnu"))
            
                        key = f"select_{match.get('movie_id','na')}_{row}_{i}"
                        is_selected = (st.session_state.get("chosen_film") == match.get("title"))
                        label = "✅ Sélectionné" if is_selected else "Sélectionner"
                        if st.button(label, key=key, width='stretch, disabled=is_selected):
                            reset_only_reco()
                            st.session_state["chosen_film"] = match.get("title")
            
                        st.markdown("</div>", unsafe_allow_html=True)


        # === Notation du film sélectionné ===
        chosen_film = st.session_state.get("chosen_film")
        if chosen_film:
            st.success(f"🎬 Film sélectionné : {chosen_film}")
            note_input = st.number_input(
                "Note du film (0.0 - 10.0)",
                min_value=0.0, max_value=10.0, value=5.0, step=0.1, format="%.1f",
                key="note_input"
            )
            if st.button("Valider la note"):
                payload = {
                    "title": chosen_film,
                    "rating": note_input,
                    "user_name": st.session_state.username,  # <- ajouté
                }
                update_resp = api_post("update_rating", payload)

                if update_resp.status_code == 200:
                    st.success(f"✅ La note {note_input} a été enregistrée pour '{chosen_film}' !")
                else:
                    detail = update_resp.json().get("detail", "Erreur inconnue")
                    st.error(f"❌ Échec : {detail}")

        # === Recommandations personnalisées (avec run_id + feedback) ===
        if chosen_film:
            st.subheader("🔍 Obtenir une recommandation personnalisée")
            if st.button("Me recommander des films", key="btn_reco"):
                try:
                    with start_ui_run(input_title=chosen_film, user=st.session_state.username):
                        mlflow.set_tags({"source": "streamlit", "stage": "inference_ui"})
                        mlflow.log_param("user", st.session_state.username)
                        mlflow.log_param("film_input", chosen_film)

                        with st.spinner("🧠 Calcul des recommandations…"):
                            # Appel API : on demande large puis on pagine côté UI
                            response = api_get(
                                f"recommend_xgb_personalized/{chosen_film}",
                                params={"top_k": 50}
                            )

                        if response.status_code != 200:
                            st.error(response.json().get("detail", "Erreur inconnue"))
                        else:
                            payload = response.json()
                            st.session_state["last_run_id"] = payload.get("run_id")
                            recos = payload.get("recommendations", [])

                            # dédoublonnage
                            seen, pool = set(), []
                            for r in recos:
                                t = r.get("title")
                                if t and t not in seen:
                                    seen.add(t)
                                    pool.append(r)

                            st.session_state["reco_pool"] = pool
                            st.session_state["reco_shown_titles"] = []
                            st.session_state["reco_page"] = 0

                            if page_from_pool():
                                st.success("🎯 Recommandations trouvées !")
                            else:
                                st.info("Aucune recommandation trouvée pour ce film")

                except requests.exceptions.RequestException:
                    st.error("❌ Erreur de connexion avec le serveur")
                except Exception as e:
                    st.error(f"❌ Erreur MLflow : {e}")
                    st.text(traceback.format_exc())

            # --- Affichage + Feedback (page courante) ---
            recos = st.session_state.get("current_recos", [])
            run_id = st.session_state.get("last_run_id") or "no_run"

            def send_feedback(reco_title: str, pred_label: int, pred_score: float, liked: int):
                if st.session_state.get("last_run_id") is None:
                    st.error("Run MLflow introuvable. Relance une recommandation.")
                    return
                try:
                    fb = {
                        "run_id": st.session_state["last_run_id"],
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
                for i, reco in enumerate(recos, start=1):
                    cols = st.columns([1, 3])
                    with cols[0]:
                        if reco.get("poster_url"):
                            st.image(reco["poster_url"], width='stretch)
                    with cols[1]:
                        reco_title = reco.get("title", "Titre inconnu")
                        raw_genres = reco.get("genres", [])
                        genres = parse_genres(raw_genres)
                        reco_synopsis = reco.get("synopsis", "Pas de synopsis disponible.")
                        score = float(reco.get("pred_score", 0.0))
                        pred_label = int(score >= 0.5)

                        st.markdown(f"### 🎬 {reco_title}")
                        st.markdown(f"**Ce film est susceptible de vous plaire à {int(score*100)}%**")
                        st.write(f"**Genres :** {', '.join(genres) if genres else 'N/A'}")
                        st.write(reco_synopsis)

                        b1, b2 = st.columns(2)
                        with b1:
                            if st.button(
                                "👍 Ça m'intéresse",
                                key=f"like_{reco_title}_{i}_{run_id}",
                                disabled=(st.session_state.get('last_run_id') is None)
                            ):
                                send_feedback(reco_title, pred_label, score, liked=1)
                        with b2:
                            if st.button(
                                "👎 Pas pour moi",
                                key=f"dislike_{reco_title}_{i}_{run_id}",
                                disabled=(st.session_state.get('last_run_id') is None)
                            ):
                                send_feedback(reco_title, pred_label, score, liked=0)

                # --- Pagination / Reset (après la liste) ---
                left = max(0, len(st.session_state.get("reco_pool", [])) - len(st.session_state.get("reco_shown_titles", [])))
                c_more, c_reset = st.columns([1, 1])

                with c_more:
                    if st.button(
                        f"🔁 Proposer d'autres recommandations ({left} restants)",
                        key=f"btn_more_reco_{st.session_state.get('reco_page', 0)}",
                        disabled=not bool(st.session_state.get('reco_pool')) or left == 0
                    ):
                        if not page_from_pool():
                            st.info("Plus de recommandations disponibles.")

                with c_reset:
                    st.button("🧹 Réinitialiser la recherche", key="btn_reset_reco", on_click=reset_search_all)

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
                            st.image(poster, width='stretch)
                        with cols[1]:
                            title = movie.get("title", "Titre inconnu")
                            raw_genres = movie.get("genres", [])
                            genres = parse_genres(raw_genres)
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
                                st.image(details["poster_url"], width='stretch)
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

    # ------------------------------
    # Onglet 4 : Dashboard perso
    # ------------------------------
    with tab4:
        st.subheader("📈 Mon tableau de bord")
        user = st.session_state.get("username")
        if not user:
            st.info("Connecte-toi pour voir ton tableau de bord.")
        else:
            with st.spinner("Chargement de tes statistiques…"):
                resp = api_get(f"user_stats/{user}")
    
            if resp.status_code != 200:
                # Essaie JSON sinon texte brut
                try:
                    err = resp.json()
                    msg = err.get("detail") or err.get("message") or str(err)
                except Exception:
                    msg = f"Erreur {resp.status_code} — {resp.text[:300]}"
                st.error(msg)
                st.stop()
    
            try:
                stats = resp.json()
            except Exception:
                st.error("Réponse invalide du serveur (non-JSON).")
                st.stop()
    
            # --- KPIs
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("👍 Likes", f"{stats['likes']}")
            c2.metric("👎 Dislikes", f"{stats['dislikes']}")
            c3.metric("Taux de like", f"{stats['like_rate']*100:.0f}%")
            c4.metric("Accuracy modèle", f"{stats['accuracy']*100:.0f}%")
    
            # --- Confusion matrix light
            conf = stats["confusion"]
            st.caption("Confusion (sur les recos où tu as donné un avis)")
            st.table(pd.DataFrame({
                "Prédit 👍": [conf["tp"], conf["fp"]],
                "Prédit 👎": [conf["fn"], conf["tn"]],
            }, index=["Réel 👍", "Réel 👎"]))
    
            # --- Genres préférés (bar chart)
            st.markdown("### 🎭 Genres préférés (selon mes notes > 5)")
            THRESH = 5.0
            
            r = api_get(f"user_ratings/{user}", params={"limit": 5000})
            if r.status_code != 200:
                try:
                    msg = r.json().get("detail", "Impossible de récupérer l'historique des notes.")
                except Exception:
                    msg = f"Erreur {r.status_code} — {r.text[:300]}"
                st.warning(msg)
            else:
                ratings = r.json().get("ratings", [])
                if not ratings:
                    st.info("Tu n'as pas encore noté de films.")
                else:
                    df_r = pd.DataFrame(ratings)
            
                    # Garder la DERNIÈRE note par film (l’API historise les notes)
                    if "ts" in df_r:
                        df_r = df_r.sort_values("ts")
                    df_last = df_r.groupby("title", as_index=False).tail(1).copy()
            
                    # Nettoyage + filtre > 5
                    df_last["rating"] = pd.to_numeric(df_last["rating"], errors="coerce")
                    df_last = df_last[df_last["rating"] > THRESH]
            
                    # Parse genres -> liste puis explode
                    df_last["genres_list"] = df_last["genres"].apply(parse_genres)
                    expl = df_last.explode("genres_list").dropna(subset=["genres_list"])
            
                    if expl.empty:
                        st.info("Aucun genre favori détecté (notes > 5) pour l’instant.")
                    else:
                        # Compter le NOMBRE DE FILMS par genre (dernière note > 5)
                        # (un film multi-genres compte pour chacun de ses genres)
                        counts = expl.groupby("genres_list")["title"].nunique().sort_values(ascending=False)
            
                        # Bar chart
                        st.bar_chart(counts)
            
                        # Top 5 en texte
                        top5 = counts.head(5)
                        st.caption(
                            "Top genres (par nombre de films notés > 5) : " +
                            ", ".join(f"{g} ({n})" for g, n in top5.items())
                        )

    
            # --- Likes par année (line chart)
            st.markdown("### 📅 Likes par année de sortie")
            if stats["by_year"]:
                df_y = pd.DataFrame(stats["by_year"]).sort_values("year").set_index("year")
                st.line_chart(df_y["likes"])
            else:
                st.info("Aucune donnée par année pour l’instant.")
    
            # --- Historique récent des feedbacks
            st.markdown("### 🕒 Derniers avis")
            recent = pd.DataFrame(stats["recent"])
            if not recent.empty:
                recent_display = recent.copy()
                recent_display["avis"] = recent_display["liked"].map({1: "👍", 0: "👎"})
                recent_display = recent_display[["ts","reco_title","avis","pred_score","genres","release_year"]]
                recent_display = recent_display.rename(columns={
                    "ts": "horodatage",
                    "reco_title": "film",
                    "pred_score": "score_modèle",
                    "genres": "genres",
                    "release_year": "année",
                })
                st.dataframe(recent_display, width='stretch, hide_index=True)
                csv_fb = recent_display.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Exporter l'historique (CSV)", csv_fb, file_name="historique_feedback.csv", mime="text/csv")
            else:
                st.info("Pas encore d’historique de feedback. Like/dislike quelques recos !")
    
            # --- 🆕 Historique des films notés (via /user_ratings)
            st.markdown("### 📝 Mes films notés")
            r = api_get(f"user_ratings/{user}", params={"limit": 500})
            if r.status_code != 200:
                try:
                    msg = r.json().get("detail", "Impossible de récupérer l'historique des notes.")
                except Exception:
                    msg = f"Erreur {r.status_code} — {r.text[:300]}"
                st.warning(msg)
            else:
                ratings = r.json().get("ratings", [])
                if ratings:
                    df_r = pd.DataFrame(ratings)
                    df_r = df_r.rename(columns={
                        "ts": "horodatage",
                        "title": "film",
                        "rating": "note",
                        "genres": "genres",
                        "release_year": "année",
                    })
                    # tri + arrondi
                    if "horodatage" in df_r:
                        df_r = df_r.sort_values("horodatage", ascending=False)
                    if "note" in df_r:
                        df_r["note"] = pd.to_numeric(df_r["note"], errors="coerce").round(1)
                    st.dataframe(df_r[["horodatage","film","note","genres","année","poster_url"]], width='stretch, hide_index=True)
    
                    csv_notes = df_r.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Exporter mes notes (CSV)", csv_notes, file_name="mes_notes.csv", mime="text/csv")
                else:
                    st.info("Tu n'as pas encore noté de film.")



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









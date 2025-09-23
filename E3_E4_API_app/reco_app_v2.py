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
import plotly.express as px

sys.path.append(os.path.join(os.path.dirname(__file__), "E3_E4_API_app"))
import config

# -----------------------------
# Configuration générale
# -----------------------------
load_dotenv()
API_URL = "https://cinenocturne.onrender.com/"
API_TOKEN = os.getenv("API_TOKEN")

st.set_page_config(page_title="CinéNocturne", page_icon="🍿")
st.markdown("""
<style>
div.stButton > button { 
  white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)

# Dictionnaire utilisateurs (login -> mdp)
USERS = {k.replace("USER_", "").lower(): v for k, v in os.environ.items() if k.startswith("USER_")}

BANNER_URL = "https://github.com/cine-nocturne-org/CineNocturne/blob/main/E3_E4_API_app/Banniere_CineNocturne.jpg?raw=true"

def show_banner():
    st.markdown("""
    <style>
      .banner img { border-radius: 12px; box-shadow: 0 6px 24px rgba(0,0,0,.25); }
      .block-container { padding-top: 1rem; }
    </style>
    """, unsafe_allow_html=True)
    st.image(BANNER_URL, use_container_width=True)

# -----------------------------
# Fonctions de connexion
# -----------------------------
def login_page():
    show_banner()
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

def full_reset_ui_state():
    keys_to_clear = [
        # reco & sélection
        "current_recos", "reco_pool", "reco_shown_titles", "reco_page",
        "fuzzy_matches_1", "chosen_film", "chosen_film_details",
        "last_run_id", "has_rated_current", "film_input",
        # onglet 2
        "already_seen_movies", "current_movies",
        # onglet 3
        "fuzzy_matches_3", "details_title",
        # inputs divers
        "note_text",
    ]
    for k in keys_to_clear:
        st.session_state.pop(k, None)
    try:
        st.cache_data.clear()
    except Exception:
        pass
    # ⚠️ Ne pas toucher aux clés "watchlist_<user>"

def logout():
    """Déconnecte l'utilisateur et remet l'UI à zéro (état local uniquement)."""
    full_reset_ui_state()
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.api_token = None
    st.rerun()

# -----------------------------
# Utilitaires API
# -----------------------------
def _headers():
    return {"Authorization": f"Bearer {st.session_state.api_token}"} if st.session_state.get("api_token") else {}

def api_get(endpoint: str, params: dict = None):
    return requests.get(f"{API_URL}{endpoint.lstrip('/')}", headers=_headers(), params=params)

def api_post(endpoint: str, payload: dict):
    return requests.post(f"{API_URL}{endpoint.lstrip('/')}", headers=_headers(), json=payload)

PLAT_LABELS = {
    "netflix": "Netflix",
    "prime": "Prime Video",
    "hulu": "Hulu",
    "hbo": "HBO Max",
    "apple": "Apple TV+",
    "canal": "Canal+",
    "disney": "Disney+",
    "paramount": "Paramount+",
    "crunchyroll": "crunchyroll"
}

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
        "page_size": 5,
        "fuzzy_matches_1": None,
        "chosen_film": None,
        "last_run_id": None,
        "film_input": "",
        "saga_recos": [],
    }.items():
        st.session_state.setdefault(k, v)

def reset_reco(full=False):
    st.session_state["current_recos"] = []
    st.session_state["reco_pool"] = []
    st.session_state["reco_shown_titles"] = []
    st.session_state["reco_page"] = 0
    st.session_state["last_run_id"] = None
    st.session_state["saga_recos"] = []
    if full:
        st.session_state["chosen_film"] = None
        st.session_state["fuzzy_matches_1"] = None

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

def reset_search_all():
    reset_reco(full=True)
    st.session_state["film_input"] = ""

def reset_only_reco():
    reset_reco(full=False)

def fetch_movie_details(title: str) -> dict:
    try:
        r = api_get(f"movie-details/{title}")
        return r.json() if r.status_code == 200 else {}
    except requests.exceptions.RequestException:
        return {}

# --- Watchlist (par utilisateur) ---
def _wl_key():
    user = st.session_state.get("username") or "_anon"
    return f"watchlist_{user}"

def get_watchlist():
    st.session_state.setdefault(_wl_key(), [])
    return st.session_state[_wl_key()]

def save_watchlist(lst):
    st.session_state[_wl_key()] = lst

def is_in_watchlist(title: str) -> bool:
    return any((it.get("title") == title) for it in get_watchlist())

def add_to_watchlist(item: dict):
    """item: dict avec au minimum title; idéalement poster_url, genres, releaseYear, platforms, synopsis"""
    if not item or not item.get("title"):
        return
    wl = get_watchlist()
    if not any(it.get("title") == item["title"] for it in wl):
        wl.append(item)
        save_watchlist(wl)
        st.toast(f"➕ Ajouté à la watchlist : {item['title']}")

def remove_from_watchlist(title: str):
    wl = [it for it in get_watchlist() if it.get("title") != title]
    save_watchlist(wl)
    st.toast(f"🗑️ Retiré de la watchlist : {title}")

# -----------------------------
# Application principale
# -----------------------------
def main_app():
    ensure_session_defaults()
    show_banner()
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🍿 CinéNocturne")
    with col2:
        st.write(f"👋 Bonjour: **{st.session_state.username}**")
        if st.button("🚪 Se déconnecter"):
            logout()

    # Onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "✨​ Recommandations perso",
        "🎲 Suggestions aléatoires",
        "📺​ Plateformes disponibles",
        "📈 Mon Profil"
    ])

    # ------------------------------
    # Onglet 1 : Film vu + reco perso
    # ------------------------------
    with tab1:
        st.subheader("🎞️​ Dernier film vu :")
    
        st.session_state.setdefault("has_rated_current", False)
        st.session_state.setdefault("chosen_film_details", {})
    
        def fetch_movie_details(title: str) -> dict:
            try:
                r = api_get(f"movie-details/{title}")
                return r.json() if r.status_code == 200 else {}
            except requests.exceptions.RequestException:
                return {}
    
        @st.cache_data(ttl=3600)
        def _fetch_details_map(titles):
            out = {}
            for t in titles:
                try:
                    r = api_get(f"movie-details/{t}")
                    if r.status_code == 200:
                        j = r.json() or {}
                        out[t] = {
                            "poster_url": j.get("poster_url"),
                            "platforms": j.get("platforms", []),
                            "genres": j.get("genres"),
                            "releaseYear": j.get("releaseYear"),
                            "synopsis": j.get("synopsis"),
                            "rating": j.get("rating"),
                        }
                    else:
                        out[t] = {}
                except Exception:
                    out[t] = {}
            return out
    
        def _pretty_plats(plats):
            if not plats:
                return []
            try:
                return [PLAT_LABELS.get((p or "").lower(), p) for p in plats]
            except Exception:
                return plats
    
        # === Recherche ===
        film_input = st.text_input("Entrez le titre du film :", key="film_input")
        c_search, c_reset_search = st.columns([1, 1])
    
        with c_search:
            if st.button("Chercher", key="btn_search"):
                reset_reco(full=True)
                st.session_state["fuzzy_matches_1"] = None
                st.session_state["chosen_film"] = None
                st.session_state["chosen_film_details"] = {}
                st.session_state["has_rated_current"] = False
    
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
    
        # === Sélection du film (grille d’options)
        if st.session_state.get("fuzzy_matches_1") and not st.session_state.get("chosen_film"):
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
                offset = (cols_per_row - len(row_matches)) // 2
    
                for i, match in enumerate(row_matches):
                    with cols[offset + i]:
                        with st.container(border=True):
                            if match.get("poster"):
                                st.image(match["poster"], width="stretch")
                            st.caption(match.get("title", "Titre inconnu"))
    
                            key = f"select_{match.get('movie_id','na')}_{row}_{i}"
                            is_selected = (st.session_state.get("chosen_film") == match.get("title"))
                            label = "✅ Sélectionné" if is_selected else "Sélectionner"
    
                            if st.button(label, key=key, width="stretch", disabled=is_selected):
                                reset_only_reco()
                                st.session_state["chosen_film"] = match.get("title")
                                st.session_state["chosen_film_details"] = fetch_movie_details(match.get("title"))
                                st.session_state["has_rated_current"] = False
                                st.session_state["fuzzy_matches_1"] = None
                                st.rerun()
    
        # === Fiche du film sélectionné ===
        chosen_film = st.session_state.get("chosen_film")
        chosen_details = st.session_state.get("chosen_film_details", {}) if chosen_film else {}
    
        if chosen_film:
            st.markdown("---")
            with st.container(border=True):
                colA, colB = st.columns([1, 2])
                with colA:
                    if chosen_details.get("poster_url"):
                        st.image(chosen_details["poster_url"], width="stretch")
                with colB:
                    title = chosen_details.get("title", chosen_film)
                    year = chosen_details.get("releaseYear", "N/A")
                    genres = chosen_details.get("genres", [])
                    synopsis = chosen_details.get("synopsis", "Pas de synopsis disponible.")
                    plats = _pretty_plats(chosen_details.get("platforms", []))
    
                    if isinstance(genres, list):
                        genres_str = ", ".join(genres)
                    else:
                        genres_str = str(genres) if genres else "N/A"
    
                    st.markdown(f"### 🎬 {title} ({year})")
                    if plats:
                        st.caption("Disponible sur : " + " · ".join(plats))
                    st.write(f"**Genres :** {genres_str}")
                    st.write(synopsis)
    
            # === Notation du film ===
            st.success(f"🎬 Film sélectionné : {chosen_film}")
            raw_note = st.text_input(
                "Notez votre film (0.0 – 10.0) pour obtenir des recommandations",
                placeholder="ex : 7.5",
                key="note_text"
            )
            if st.button("Valider la note", key="btn_save_rating"):
                if not raw_note.strip():
                    st.error("Entre une note avant de valider.")
                else:
                    try:
                        val = float(raw_note.replace(",", "."))
                        if 0.0 <= val <= 10.0:
                            payload = {
                                "title": chosen_film,
                                "rating": round(val, 1),
                                "user_name": st.session_state.username,
                            }
                            update_resp = api_post("update_rating", payload)
                            if update_resp.status_code == 200:
                                st.success(f"✅ La note {round(val,1)} a été enregistrée pour '{chosen_film}' !")
                                st.session_state["has_rated_current"] = True
                            else:
                                st.error(update_resp.json().get("detail", "Erreur inconnue"))
                        else:
                            st.warning("La note doit être comprise **entre 0 et 10**.")
                    except ValueError:
                        st.warning("Format invalide. Exemple valide : 7.5")
    
            # === Recommandations personnalisées ===
            st.subheader("🔍 Obtenir des recommandations personnalisées")
            btn_disabled = not st.session_state.get("has_rated_current", False)
            if btn_disabled:
                st.info("📝 Note ce film pour débloquer les recommandations.")
            if st.button("Me recommander des films", key="btn_reco", disabled=btn_disabled):
                try:
                    with start_ui_run(input_title=chosen_film, user=st.session_state.username):
                        mlflow.set_tags({"source": "streamlit", "stage": "inference_ui"})
                        mlflow.log_param("user", st.session_state.username)
                        mlflow.log_param("film_input", chosen_film)
    
                        with st.spinner("🧠 Calcul des recommandations…"):
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
    
                            # ✅ NEW: enregistrer la liste "saga"
                            st.session_state["saga_recos"] = payload.get("saga_recommendations", [])
    
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
    
            # --- Affichage + Feedback (page courante) + Watchlist ---
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
                titles_to_fetch = [
                    r.get("title") for r in recos
                    if r.get("title") and (not r.get("poster_url") or r.get("platforms") in (None, [], ""))
                ]
                details_map = _fetch_details_map(sorted(set(titles_to_fetch))) if titles_to_fetch else {}
    
                for i, reco in enumerate(recos, start=1):
                    cols = st.columns([1, 3])
                    with cols[0]:
                        poster_url = reco.get("poster_url") or details_map.get(reco.get("title"), {}).get("poster_url")
                        if poster_url:
                            st.image(poster_url, width="stretch")
                    with cols[1]:
                        reco_title = reco.get("title", "Titre inconnu")
                        raw_genres = reco.get("genres", [])
                        genres = parse_genres(raw_genres)
                        reco_synopsis = reco.get("synopsis") or details_map.get(reco_title, {}).get("synopsis") or "Pas de synopsis disponible."
                        score = float(reco.get("pred_score", 0.0))
                        pred_label = int(score >= 0.5)
    
                        plats = reco.get("platforms")
                        if not plats:
                            plats = details_map.get(reco_title, {}).get("platforms", [])
                        plats = _pretty_plats(plats)
    
                        st.markdown(f"### 🎬 {reco_title}")
                        st.write(f"**Probabilité d'aimer :** {int(score*100)}%")
                        if plats:
                            st.caption("Disponible sur : " + " · ".join(plats))
                        st.write(f"**Genres :** {', '.join(genres) if genres else 'N/A'}")
                        st.write(reco_synopsis)
    
                        b1, b2, b3 = st.columns(3)
                        with b1:
                            if st.button(
                                "👍 Ça m'intéresse",
                                key=f"like_{reco_title}_{i}_{run_id}",
                                disabled=(st.session_state.get('last_run_id') is None)
                            ):
                                send_feedback(reco_title, pred_label, score, liked=1)
                                # ➕ Ajout automatique à la watchlist
                                add_to_watchlist({
                                    "title": reco_title,
                                    "poster_url": poster_url,
                                    "genres": genres,
                                    "releaseYear": reco.get("releaseYear") or details_map.get(reco_title, {}).get("releaseYear"),
                                    "platforms": plats,
                                    "synopsis": reco_synopsis,
                                })
                        with b2:
                            if st.button(
                                "👎 Pas pour moi",
                                key=f"dislike_{reco_title}_{i}_{run_id}",
                                disabled=(st.session_state.get('last_run_id') is None)
                            ):
                                send_feedback(reco_title, pred_label, score, liked=0)
                        with b3:
                            # Bouton dédié Watchlist (utile si on ne veut pas liker tout en sauvegardant)
                            in_wl = is_in_watchlist(reco_title)
                            if st.button(
                                "✅ Dans la watchlist" if in_wl else "➕ Watchlist",
                                key=f"wl_reco_{reco_title}_{i}",
                                disabled=in_wl
                            ):
                                add_to_watchlist({
                                    "title": reco_title,
                                    "poster_url": poster_url,
                                    "genres": genres,
                                    "releaseYear": reco.get("releaseYear") or details_map.get(reco_title, {}).get("releaseYear"),
                                    "platforms": plats,
                                    "synopsis": reco_synopsis,
                                })
    
            # === NEW: Section "Films de la même saga" (affichée en plus)
            saga_list = st.session_state.get("saga_recos", []) or []
            if saga_list:
                st.markdown("## 🔗 Films de la même saga")
    
                saga_titles_to_fetch = [
                    r.get("title") for r in saga_list
                    if r.get("title") and (not r.get("poster_url") or r.get("platforms") in (None, [], ""))
                ]
                saga_details_map = _fetch_details_map(sorted(set(saga_titles_to_fetch))) if saga_titles_to_fetch else {}
    
                for i, r in enumerate(saga_list, start=1):
                    t = r.get("title", "Titre inconnu")
                    poster_url = r.get("poster_url") or saga_details_map.get(t, {}).get("poster_url")
                    synopsis = r.get("synopsis") or saga_details_map.get(t, {}).get("synopsis") or "Pas de synopsis disponible."
                    year = r.get("releaseYear") or saga_details_map.get(t, {}).get("releaseYear")
    
                    raw_genres = r.get("genres", [])
                    genres = parse_genres(raw_genres)
    
                    plats = r.get("platforms") or saga_details_map.get(t, {}).get("platforms", [])
                    plats = _pretty_plats(plats)
    
                    cols = st.columns([1, 3])
                    with cols[0]:
                        if poster_url:
                            st.image(poster_url, width="stretch")
                    with cols[1]:
                        st.markdown(f"### 🎬 {t} ({year or 'N/A'})")
                        if plats:
                            st.caption("Disponible sur : " + " · ".join(plats))
                        st.write(f"**Genres :** {', '.join(genres) if genres else 'N/A'}")
                        st.write(synopsis)
    
                        in_wl = is_in_watchlist(t)
                        if st.button("✅ Dans la watchlist" if in_wl else "➕ Watchlist",
                                     key=f"wl_saga_{t}_{i}", disabled=in_wl):
                            add_to_watchlist({
                                "title": t,
                                "poster_url": poster_url,
                                "genres": genres,
                                "releaseYear": year,
                                "platforms": plats,
                                "synopsis": synopsis,
                            })
    
            # --- Pagination / Reset ---
            left = max(0, len(st.session_state.get("reco_pool", [])) - len(st.session_state.get("reco_shown_titles", [])))
            c_more, c_reset = st.columns([1, 1])
    
            with c_more:
                if st.button(
                    f"🔁 Proposer d'autres recommandations ({left} restants)",
                    key="btn_more_reco",
                    disabled=not bool(st.session_state.get('reco_pool')) or left == 0
                ):
                    if not page_from_pool():
                        st.info("Plus de recommandations disponibles.")
                    st.rerun()   # 👈 pour éviter le double clic
    
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
                    selected_platforms = st.multiselect(
                        "Choisissez les plateformes",
                        ["netflix", "prime", "hulu", "hbo", "apple", "canal", "disney", "paramount", "crunchyroll"]
                    )
                    submitted = st.form_submit_button("Afficher des films aléatoires")

                if "already_seen_movies" not in st.session_state:
                    st.session_state["already_seen_movies"] = set()
                if "current_movies" not in st.session_state:
                    st.session_state["current_movies"] = []

                def fetch_random_movies():
                    if not selected_platforms:
                        st.warning("Sélectionne au moins une plateforme.")
                        return

                    params = {"genre": selected_genre, "platforms": selected_platforms, "limit": 20}
                    response = api_get("random_movies/", params=params)
                    if response.status_code == 200:
                        movies = response.json()  # un item par (titre, plateforme)

                        # --- Agrégation par titre -> plateformes multiples
                        agg = {}
                        for m in movies:
                            t = m.get("title")
                            if not t:
                                continue

                            p_raw = (m.get("platform") or "").lower()
                            plat = PLAT_LABELS.get(p_raw, m.get("platform"))

                            if t not in agg:
                                agg[t] = {
                                    "title": t,
                                    "synopsis": m.get("synopsis"),
                                    "poster_url": m.get("poster_url"),
                                    "genres": m.get("genres"),
                                    "releaseYear": m.get("releaseYear"),
                                    "platforms": set(),
                                }

                            for k in ("synopsis", "poster_url", "genres", "releaseYear"):
                                if not agg[t].get(k) and m.get(k):
                                    agg[t][k] = m.get(k)

                            if plat:
                                agg[t]["platforms"].add(plat)

                        movies_agg = []
                        for e in agg.values():
                            e["platforms"] = sorted(e["platforms"])
                            movies_agg.append(e)

                        # Filtre et limite (10) en évitant les doublons déjà vus
                        fresh_movies = [
                            m for m in movies_agg
                            if m["title"] not in st.session_state["already_seen_movies"]
                               and m.get("poster_url") and m.get("synopsis")
                        ][:10]

                        for m in fresh_movies:
                            st.session_state["already_seen_movies"].add(m["title"])
                        st.session_state["current_movies"] = fresh_movies
                    else:
                        try:
                            st.error(response.json().get("detail", "Erreur lors de la récupération des films."))
                        except Exception:
                            st.error(f"Erreur lors de la récupération des films. (HTTP {response.status_code})")

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
                            st.image(poster, width='stretch')

                        with cols[1]:
                            title = movie.get("title", "Titre inconnu")
                            raw_genres = movie.get("genres", [])
                            genres = parse_genres(raw_genres)

                            st.markdown(f"### 🎬 {title} ({year})")

                            plats = movie.get("platforms") or []
                            if plats:
                                st.caption("Disponible sur : " + " · ".join(plats))

                            st.write(f"**Genres :** {', '.join(genres) if genres else 'N/A'}")
                            st.write(synopsis)

                            # ➕ Watchlist depuis les suggestions
                            in_wl = is_in_watchlist(title)
                            if st.button(
                                "✅ Dans la watchlist" if in_wl else "➕ Ajouter à la watchlist",
                                key=f"add_wl_sugg_{title}",
                                disabled=in_wl
                            ):
                                add_to_watchlist({
                                    "title": title,
                                    "poster_url": poster,
                                    "genres": genres,
                                    "releaseYear": year,
                                    "platforms": plats,
                                    "synopsis": synopsis,
                                })
            else:
                try:
                    st.error(genre_response.json().get("detail", "Impossible de récupérer la liste des genres."))
                except Exception:
                    st.error(f"Impossible de récupérer la liste des genres. (HTTP {genre_response.status_code})")
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
                                st.image(details["poster_url"], width='stretch')
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
    # Utils dashboard (robustes)
    # ------------------------------
    def fetch_user_stats(user: str):
        try:
            r = api_get(f"user_stats/{user}")
            if r.status_code != 200:
                try:
                    return None, r.json().get("detail", f"Erreur {r.status_code}")
                except Exception:
                    return None, f"Erreur {r.status_code} — {r.text[:300]}"
            return r.json(), None
        except requests.exceptions.RequestException:
            return None, "Erreur de connexion avec le serveur."
        except Exception as e:
            return None, f"Erreur inattendue: {e}"

    def fetch_user_ratings(user: str, limit: int = 500):
        try:
            r = api_get(f"user_ratings/{user}", params={"limit": limit})
            if r.status_code != 200:
                try:
                    return None, r.json().get("detail", f"Erreur {r.status_code}")
                except Exception:
                    return None, f"Erreur {r.status_code} — {r.text[:300]}"
            payload = r.json() or {}
            return payload.get("ratings", []), None
        except requests.exceptions.RequestException:
            return None, "Erreur de connexion avec le serveur."
        except Exception as e:
            return None, f"Erreur inattendue: {e}"

    # ------------------------------
    # Onglet 4 : Dashboard perso + Watchlist
    # ------------------------------
    with tab4:
        user = st.session_state.get("username")
        if not user:
            st.info("Connecte-toi pour voir ton tableau de bord.")
            st.stop()

        # Sous-onglets
        sub_tab1, sub_tab2 = st.tabs(["📊 Tableau de bord", "📌 Watchlist"])

        with sub_tab1:
            # ---------- Helpers locaux ----------
            def _parse_g(s):
                if s is None or (isinstance(s, float) and pd.isna(s)):
                    return []
                return [g.strip() for g in str(s).replace("|", ",").split(",") if g.strip()]

            def _normalize_genre_fr(g: str) -> str:
                g0 = (g or "").strip().lower()
                mapping = {"science fiction": "science-fiction", "mystere": "mystère", "comedie": "comédie"}
                return mapping.get(g0, g0)

            def hero_title_for_genre(genre: str) -> str:
                g = (genre or "").strip().lower()
                fr_map = {
                    "horreur": "🕯️ Ambiance frisson",
                    "comédie": "😂 Bonne humeur garantie",
                    "action": "💥 Décharge d’adrénaline",
                    "thriller": "🕵️ Suspense maximal",
                    "drame": "🎭 Émotions fortes",
                    "romance": "💞 Âme romantique",
                    "animation": "🎨 Âme animée",
                    "science-fiction": "🚀 Rêves de SF",
                    "fantasy": "🧙 Cap sur l’imaginaire",
                    "documentaire": "🎓 Esprit curieux",
                    "crime": "🕶️ Amateur de polars",
                    "famille": "👨‍👩‍👧 Esprit famille",
                    "western": "🤠 Esprit Far West",
                    "mystère": "🧩 Chasseur d’énigmes",
                    "guerre": "⚔️ Chroniques de guerre",
                    "musique": "🎵 Cinéphile mélomane",
                    "histoire": "🏺 Passion histoire",
                    "aventure": "🗺️ Goût de l’aventure",
                    "biopic": "👤 Vies d’exception",
                    "sport": "🏅 Esprit sportif",
                    "noir": "🌑 Noir c’est noir",
                }
                if not g:
                    return "⭐ Découvrons tes goûts"
                return fr_map.get(g, f"⭐ Ton ambiance : {genre}")

            @st.cache_data(ttl=3600)
            def _fetch_release_year_map(titles):
                out = {}
                for t in titles:
                    try:
                        r = api_get(f"movie-details/{t}")
                        out[t] = r.json().get("releaseYear") if r.status_code == 200 else None
                    except Exception:
                        out[t] = None
                return out

            # ---------- Données API ----------
            with st.spinner("Chargement de tes statistiques…"):
                stats, err = fetch_user_stats(user)

            if err:
                st.error(err)
                st.stop()

            stats = stats or {}
            total       = int(stats.get("total") or 0)
            likes       = int(stats.get("likes") or 0)
            dislikes    = int(stats.get("dislikes") or 0)
            like_rate   = float(stats.get("like_rate") or 0.0)
            accuracy    = float(stats.get("accuracy") or 0.0)  # non affiché
            confusion   = stats.get("confusion") or {"tp":0,"tn":0,"fp":0,"fn":0}
            top_genres  = stats.get("top_genres") or []
            by_year     = stats.get("by_year") or []
            recent_list = stats.get("recent") or []

            # --- Récupère toutes les notations utilisateur ---
            ratings, rerr = fetch_user_ratings(user, limit=5000)
            if rerr:
                st.warning(rerr)
                ratings = []

            df_r = pd.DataFrame(ratings or [])
            df_last = pd.DataFrame()
            if not df_r.empty:
                if "ts" in df_r.columns:
                    df_r["ts"] = pd.to_datetime(df_r["ts"], errors="coerce")
                    df_r = df_r.sort_values("ts")
                df_last = df_r.groupby("title", as_index=False).tail(1).copy()
                df_last["rating"] = pd.to_numeric(df_last.get("rating"), errors="coerce")

                if "release_year" in df_last.columns:
                    df_last["release_year"] = pd.to_numeric(df_last["release_year"], errors="coerce")
                else:
                    df_last["release_year"] = pd.NA

                missing_year_titles = df_last.loc[df_last["release_year"].isna(), "title"].dropna().unique().tolist()
                if missing_year_titles:
                    ry_map = _fetch_release_year_map(sorted(set(missing_year_titles)))
                    df_last.loc[df_last["release_year"].isna(), "release_year"] = df_last.loc[
                        df_last["release_year"].isna(), "title"
                    ].map(ry_map)
                    df_last["release_year"] = pd.to_numeric(df_last["release_year"], errors="coerce")

            # --- Genre favori calculé (notes > 5) pour le TITRE dynamique ---
            pref_genre_calc = "N/A"
            if not df_last.empty:
                liked_df = df_last[df_last["rating"] > 5.0].copy()
                liked_df["genres_list"] = liked_df["genres"].apply(_parse_g)
                expl = liked_df.explode("genres_list").dropna(subset=["genres_list"])
                if not expl.empty:
                    expl["genres_norm"] = expl["genres_list"].map(_normalize_genre_fr)
                    counts_title = expl.groupby("genres_norm")["title"].nunique().sort_values(ascending=False)
                    if not counts_title.empty:
                        pref_genre_calc = str(counts_title.index[0])

            if pref_genre_calc == "N/A" and top_genres:
                pref_genre_calc = top_genres[0].get("genre", "N/A")

            def smart_capitalize(s: str) -> str:
                if not isinstance(s, str) or not s:
                    return s
                return s[0].upper() + s[1:]

            pref_genre_display = pref_genre_calc if pref_genre_calc in (None, "", "N/A") else smart_capitalize(pref_genre_calc)

            st.markdown(f"## {hero_title_for_genre(pref_genre_calc)}")

            # KPIs sans accuracy
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("👍 Likes", f"{likes}")
            c2.metric("👎 Dislikes", f"{dislikes}")
            c3.metric("Taux de like", f"{like_rate*100:.0f}%")
            c4.metric("Genre préféré", pref_genre_display)

            # Confusion
            st.caption("Confusion (sur les recos où tu as donné un avis)")
            try:
                cm_df = pd.DataFrame({
                    "Prédit 👍": [int(confusion.get("tp",0)), int(confusion.get("fp",0))],
                    "Prédit 👎": [int(confusion.get("fn",0)), int(confusion.get("tn",0))],
                }, index=["Réel 👍", "Réel 👎"])
                st.table(cm_df)
            except Exception:
                st.info("Confusion non disponible.")

            # Pie genres préférés
            st.markdown("### 🎭 Genres préférés")
            THRESH = 5.0
            if df_last.empty:
                st.info("Tu n'as pas encore noté de films.")
            else:
                liked = df_last[df_last["rating"] > THRESH].copy()
                liked["genres_list"] = liked["genres"].apply(_parse_g)

                expl = liked.explode("genres_list").dropna(subset=["genres_list"])
                expl["genres_list"] = expl["genres_list"].astype(str).str.strip()
                expl = expl[expl["genres_list"] != ""]
                expl["genres_norm"] = expl["genres_list"].map(_normalize_genre_fr)

                counts = (
                    expl.groupby("genres_norm")["title"]
                        .nunique()
                        .sort_values(ascending=False)
                )

                if counts.empty:
                    st.info("Aucun genre favori détecté (notes > 5).")
                else:
                    df_counts = counts.reset_index()
                    df_counts.columns = ["genre", "films_distincts"]

                    fig = px.pie(
                        df_counts, values="films_distincts", names="genre",
                    )
                    fig.update_traces(
                        textposition="inside",
                        textinfo="percent+label",
                        hovertemplate="<b>%{label}</b><br>Films distincts: %{value}<br>Part: %{percent}<extra></extra>",
                    )
                    fig.update_layout(
                        margin=dict(t=0, b=0, l=0, r=0),
                        legend_title_text="Genres",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    top5 = counts.head(5)
                    st.caption("Top genres : " + ", ".join(f"{g} ({n})" for g, n in top5.items()))

            # Courbe par année
            st.markdown("### 📅 Films notés par année de sortie")
            try:
                if df_last.empty or "release_year" not in df_last.columns:
                    st.info("Aucune donnée par année pour l’instant.")
                else:
                    df_year = df_last.dropna(subset=["release_year"]).copy()
                    df_year["release_year"] = pd.to_numeric(df_year["release_year"], errors="coerce")
                    df_year = df_year.dropna(subset=["release_year"])
                    if df_year.empty:
                        st.info("Aucune donnée par année pour l’instant.")
                    else:
                        counts_by_year = df_year.groupby("release_year")["title"].nunique().sort_index()
                        st.line_chart(counts_by_year)
                        st.caption(f"Total films notés (distincts) : {int(counts_by_year.sum())}")
            except Exception:
                st.info("Aucune donnée par année pour l’instant.")

            # Historique notations
            st.markdown("### 📝 Historique de mes notations")
            df_hist = pd.DataFrame(ratings or [])
            if df_hist.empty:
                st.info("Tu n'as pas encore noté de films.")
            else:
                if "ts" in df_hist.columns:
                    df_hist["ts"] = pd.to_datetime(df_hist["ts"], errors="coerce")
                    df_hist = df_hist.sort_values("ts")
                df_hist = df_hist.groupby("title", as_index=False).tail(1).copy()

                def _fmt_genres(s):
                    if s is None or (isinstance(s, float) and pd.isna(s)):
                        return ""
                    return ", ".join([g.strip() for g in str(s).replace("|", ",").split(",") if g.strip()])
                if "genres" in df_hist.columns:
                    df_hist["genres"] = df_hist["genres"].apply(_fmt_genres)

                if "movie_rating" not in df_hist.columns:
                    @st.cache_data(ttl=3600)
                    def _fetch_movie_ratings_map(titles):
                        out = {}
                        for t in titles:
                            try:
                                r = api_get(f"movie-details/{t}")
                                out[t] = r.json().get("rating") if r.status_code == 200 else None
                            except Exception:
                                out[t] = None
                        return out
                    uniq_titles = sorted(df_hist["title"].dropna().unique().tolist())
                    rating_map = _fetch_movie_ratings_map(uniq_titles)
                    df_hist["movie_rating"] = df_hist["title"].map(rating_map)

                for c in ("release_year", "rating", "movie_rating"):
                    if c in df_hist.columns:
                        df_hist[c] = pd.to_numeric(df_hist[c], errors="coerce")

                cols_order = ["ts", "title", "release_year", "genres", "rating", "movie_rating"]
                present = [c for c in cols_order if c in df_hist.columns]
                df_display = df_hist[present].rename(columns={
                    "ts": "horodatage",
                    "title": "film",
                    "release_year": "année",
                    "genres": "genres",
                    "rating": "ma_note",
                    "movie_rating": "note_globale"
                })

                st.dataframe(df_display, width='stretch', hide_index=True)

                csv_hist = df_display.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Exporter mes notations (CSV)",
                    csv_hist,
                    file_name="historique_notations.csv",
                    mime="text/csv"
                )

        with sub_tab2:
            st.subheader("📌 Ma Watchlist")
            wl = get_watchlist()
            if not wl:
                st.info("Ta watchlist est vide. Ajoute des films depuis les recommandations ou les suggestions.")
            else:
                wl_sorted = sorted(wl, key=lambda x: (x.get("title") or "").lower())
                for idx, it in enumerate(wl_sorted, start=1):
                    cols = st.columns([1, 3, 1])
                    with cols[0]:
                        if it.get("poster_url"):
                            st.image(it["poster_url"], width="stretch")
                    with cols[1]:
                        title = it.get("title", "Titre inconnu")
                        year = it.get("releaseYear", "N/A")
                        g = it.get("genres") or []
                        if not isinstance(g, list):
                            g = [str(g)]
                        synopsis = it.get("synopsis") or "Pas de synopsis disponible."
                        plats = it.get("platforms") or []
                        st.markdown(f"### 🎬 {title} ({year})")
                        if plats:
                            st.caption("Disponible sur : " + " · ".join(plats))
                        st.write(f"**Genres :** {', '.join(g) if g else 'N/A'}")
                        st.write(synopsis)
                    with cols[2]:
                        if st.button("Retirer", key=f"rm_wl_{idx}"):
                            remove_from_watchlist(it.get("title"))
                            st.rerun()

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


# api.py
# ============================================================
# 🎬 Louve Movies API — Recommandations + Monitoring MLflow
# Corrigé : TF-IDF recalculée côté API avec le vectorizer du train
# ============================================================

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials, APIKeyHeader
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import unicodedata
import random
import io
from datetime import datetime
import os
import csv
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import logging
import tempfile
import json
import re

# -------- Config locale du projet --------
# (le module E3_E4_API_app.config doit définir MLFLOW_TRACKING_URI)
from E3_E4_API_app import config

# ======================
# 🔧 Logging
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("louve_api")

# ======================
# 🚀 FastAPI
# ======================
app = FastAPI(title="🎬 Louve Movies API")

# ======================
# 🧪 MLflow
# ======================
mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment("louve_movies_monitoring")

# ======================
# 🗄️ Base de données
# ======================
DATABASE_URL = "mysql+pymysql://louve:%40Marley080922@mysql-louve.alwaysdata.net/louve_movies"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def ensure_column_exists(table: str, column: str, coldef: str):
    """Crée la colonne si absente (idempotent)."""
    with engine.begin() as conn:
        exists = conn.execute(text("""
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = :table
              AND COLUMN_NAME = :column
        """), {"table": table, "column": column}).scalar()
        if not exists:
            conn.execute(text(f"ALTER TABLE `{table}` ADD COLUMN `{column}` {coldef}"))

def ensure_movies_schema():
    """Crée/ajoute sans casser: colonnes requises + index FULLTEXT."""
    required_cols = {
        "title": "VARCHAR(255)",
        "original_title": "VARCHAR(255)",
        "release_year": "INT",
        "genres": "VARCHAR(255)",
        "synopsis": "TEXT",
        "rating": "FLOAT",
        "vote_count": "INT",
        "original_language": "VARCHAR(10)",
        "poster_url": "VARCHAR(255)",
        "key_words": "TEXT",
        "user_rating": "FLOAT",
        "platforms_flatrate": "VARCHAR(1024)",
        "platforms_rent": "VARCHAR(1024)",
        "platforms_buy": "VARCHAR(1024)",
        "platform_link": "VARCHAR(512)",
    }

    with engine.begin() as conn:
        # crée la table si absente (squelette minimal)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS movies (
                movie_id INT PRIMARY KEY
            ) ENGINE=InnoDB
        """))

        # colonnes existantes
        existing = conn.execute(text("""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'movies'
        """)).fetchall()
        existing_cols = {r[0] for r in existing}

        # ajouter manquantes
        for col, ddl in required_cols.items():
            if col not in existing_cols:
                conn.execute(text(f"ALTER TABLE movies ADD COLUMN `{col}` {ddl}"))

        # FULLTEXT (sur title)
        has_idx = conn.execute(text("""
            SELECT COUNT(1) FROM INFORMATION_SCHEMA.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'movies'
              AND INDEX_NAME = 'idx_movies_title_fulltext'
        """)).scalar()
        if not has_idx:
            try:
                conn.execute(text("CREATE FULLTEXT INDEX idx_movies_title_fulltext ON movies(title)"))
            except SQLAlchemyError:
                pass  # non bloquant (selon version/permissions)

# --- migrations sûres avant de charger en mémoire ---
ensure_movies_schema()
ensure_column_exists("movies", "user_rating", "FLOAT")  # (redondant mais OK)

# ======================
# 🔒 Auth (Basic ou Bearer)
# ======================
load_dotenv()
USERNAME: str = os.getenv("API_USERNAME")
PASSWORD: str = os.getenv("API_PASSWORD")
API_TOKEN: str = os.getenv("API_TOKEN")

security_basic = HTTPBasic(auto_error=False)
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def verify_credentials(
    credentials: HTTPBasicCredentials = Depends(security_basic),
    api_key: str = Depends(api_key_header)
):
    # Mode 1 : Basic Auth
    if credentials and credentials.username == USERNAME and credentials.password == PASSWORD:
        return True
    # Mode 2 : Bearer Token
    if api_key and api_key == f"Bearer {API_TOKEN}":
        return True
    raise HTTPException(status_code=401, detail="Non autorisé")

# ======================
# 🧠 Helpers
# ======================
STOPWORDS = {"the","of","and","a","an","la","le","les","de","des","du","et"}

def normalize_text(text: str) -> str:
    text = text.strip()
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    return text.lower()

def _preprocess_text(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    return re.sub(r"[^a-z0-9\s]", " ", t)

def _split_genres(s: str):
    if not s:
        return []
    parts = []
    for token in str(s).split("|"):
        parts.extend([g.strip() for g in token.split(",") if g.strip()])
    return parts

def safe_mlb_transform(_mlb, genre_lists):
    try:
        return _mlb.transform(genre_lists)
    except ValueError:
        known = set(_mlb.classes_)
        cleaned = [[g for g in gs if g in known] for gs in genre_lists]
        return _mlb.transform(cleaned)

def mlflow_start_inference_run(input_title: str, top_k: int):
    try:
        mlflow.set_experiment("louve_movies_monitoring")
        if mlflow.active_run() is not None:
            mlflow.end_run()
        return mlflow.start_run(run_name=f"recommend_{input_title}")
    except Exception as e:
        logger.warning(f"MLflow indisponible: {e}")
        class DummyRun:
            info = type("I", (), {"run_id": ""})()
            def __enter__(self): return self
            def __exit__(self, *args): return False
        return DummyRun()

# ======================
# 🤖 Chargement modèles & données (CORRIGÉ)
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Modèles / objets (du train)
xgb_model   = joblib.load(os.path.join(MODEL_DIR, "xgb_classifier_model.joblib"))
vectorizer  = joblib.load(os.path.join(MODEL_DIR, "reco_vectorizer.joblib"))  # ✅ nouveau
mlb         = joblib.load(os.path.join(MODEL_DIR, "mlb_model.joblib"))
scaler_year = joblib.load(os.path.join(MODEL_DIR, "scaler_year.joblib"))
svd_full    = joblib.load(os.path.join(MODEL_DIR, "svd_model.joblib"))
# ❌ on ne charge plus tfidf_matrix_full.joblib ni nn_full.joblib

# Films depuis BDD (mémoire)
with engine.connect() as conn:
    rows = conn.execute(
        text("""
            SELECT movie_id, title, genres, release_year, synopsis, poster_url, rating, user_rating
            FROM movies
            WHERE synopsis IS NOT NULL
        """)
    ).mappings().all()

rows = rows or []

movies = [dict(r) for r in rows]
titles = [r["title"] for r in rows]
years  = [r["release_year"] if r["release_year"] is not None else 2000 for r in rows]
genres_list_all = [_split_genres(r["genres"]) for r in rows]
syn_clean = [_preprocess_text(r["synopsis"]) for r in rows]

# --- TF-IDF recalculée sur la BDD actuelle avec le vectorizer du train ---
tfidf_matrix = vectorizer.transform(syn_clean)

# --- SVD -> vecteurs compacts (mêmes composantes qu’à l’entraînement) ---
svd_vecs = svd_full.transform(tfidf_matrix)

# --- Encodages genres / années ---
genres_encoded_matrix = safe_mlb_transform(mlb, genres_list_all)
years_scaled = scaler_year.transform(np.array(years).reshape(-1, 1))

# --- kNN recalculé (aligné sur la TF-IDF actuelle) ---
nn = NearestNeighbors(metric="cosine", algorithm="brute")
nn.fit(tfidf_matrix)

# --- Garde-fous d’alignement : on tronque toutes les structures au minimum commun ---
n = min(
    len(titles),
    getattr(tfidf_matrix, "shape", [0])[0],
    getattr(genres_encoded_matrix, "shape", [0])[0],
    getattr(years_scaled, "shape", [0])[0],
    getattr(svd_vecs, "shape", [0])[0],
)

if n == 0:
    logger.error("Catalogue vide ou artefacts incompatibles (n=0).")
    titles = []
    movies = []
    genres_list_all = []
    years = []
else:
    if len(titles) > n:
        titles = titles[:n]
        movies = movies[:n]
        genres_list_all = genres_list_all[:n]
        years = years[:n]
    if getattr(tfidf_matrix, "shape", [0])[0] > n:
        tfidf_matrix = tfidf_matrix[:n]
    if getattr(genres_encoded_matrix, "shape", [0])[0] > n:
        genres_encoded_matrix = genres_encoded_matrix[:n]
    if getattr(years_scaled, "shape", [0])[0] > n:
        years_scaled = years_scaled[:n]
    if getattr(svd_vecs, "shape", [0])[0] > n:
        svd_vecs = svd_vecs[:n]

movies_dict = {m["title"]: m for m in movies}

# ======================
# 📦 Pydantic Models
# ======================
class RatingUpdate(BaseModel):
    title: str
    rating: float
    user_name: Optional[str] = None  # NEW: identifiant de l'utilisateur qui note

class FeedbackPayload(BaseModel):
    run_id: str
    user_name: str
    input_title: str
    reco_title: str
    pred_label: int
    pred_score: float
    liked: int  # 1 si l'utilisateur aime / 0 sinon

# ======================
# 🔧 Endpoints
# ======================

# -- Mettre à jour la note utilisateur d’un film (+ historisation)
@app.post("/update_rating", include_in_schema=False, dependencies=[Depends(verify_credentials)])
async def update_rating(payload: RatingUpdate):
    title = payload.title
    rating = payload.rating
    user_name = payload.user_name or "anonymous"

    if rating < 0 or rating > 10:
        raise HTTPException(status_code=400, detail="La note doit être comprise entre 0 et 10.")

    try:
        with engine.begin() as conn:
            # S'assurer de l'existence de la colonne user_rating
            check_col = conn.execute(text("""
                SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'movies' AND COLUMN_NAME = 'user_rating'
            """)).fetchone()
            if not check_col:
                conn.execute(text("ALTER TABLE movies ADD COLUMN user_rating FLOAT"))

            result = conn.execute(
                text("UPDATE movies SET user_rating = :rating WHERE title = :title"),
                {"rating": rating, "title": title}
            )
            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail=f"Film '{title}' non trouvé.")

            # Historiser la note
            conn.execute(
                text("""
                    INSERT INTO user_ratings (user_name, title, rating)
                    VALUES (:u, :t, :r)
                """),
                {"u": user_name, "t": title, "r": float(rating)}
            )

        # NEW: refresh cache en mémoire pour les recos (après commit)
        if title in movies_dict:
            movies_dict[title]["user_rating"] = float(rating)

        return {"message": f"La note {rating} a été enregistrée pour le film '{title}'."}

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Erreur SQLAlchemy : {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")

# -- Recommandations XGB personnalisées (publique pour l’UI)
@app.get("/recommend_xgb_personalized/{title}")
async def recommend_xgb_personalized(title: str, top_k: int = 5):
    if not titles:
        raise HTTPException(status_code=500, detail="Catalogue indisponible.")

    # 1) Trouver l'index du film d'entrée
    try:
        idx = next(i for i, t in enumerate(titles) if t.lower() == title.lower())
    except StopIteration:
        raise HTTPException(status_code=404, detail="Film non trouvé")

    # 2) Démarrer un run MLflow (en mode dégradé si indisponible)
    with mlflow_start_inference_run(input_title=title, top_k=top_k) as run:
        try:
            info = getattr(run, "info", None)
            rid = (info.get("run_id") if isinstance(info, dict) else getattr(info, "run_id", None))
            if not rid:
                rid = getattr(run, "run_id", "")
            run_id = str(rid or "")
        except Exception:
            run_id = ""

        # Tags + paramètres de base (protégés)
        try:
            mlflow.set_tags({
                "stage": "inference",
                "component": "xgb_recommender",
                "source": "api",
            })
            mlflow.log_param("input_title", title)
            mlflow.log_param("top_k", int(top_k))
        except Exception:
            pass

        # 3) Notes du film d'entrée
        input_movie = movies_dict.get(title, {})
        input_user_rating = float(input_movie.get("user_rating") or 0.0)
        input_movie_rating = float(input_movie.get("rating") or 5.0)
        try:
            mlflow.log_metric("user_rating", input_user_rating)
            mlflow.log_metric("movie_rating", input_movie_rating)
        except Exception:
            pass

        # 4) Candidats par similarité TF-IDF + filtre genres
        N = getattr(tfidf_matrix, "shape", [0])[0]
        if idx >= N:
            raise HTTPException(status_code=500, detail="Catalogue ML et BDD désalignés (TF-IDF).")

        chosen_genres = set(genres_list_all[idx])
        vec = tfidf_matrix[idx].reshape(1, -1)
        cosine_sim = cosine_similarity(vec, tfidf_matrix).flatten()
        cosine_sim[idx] = -1.0  # exclure le film lui-même

        candidate_indices = np.argsort(cosine_sim)[-50:][::-1]
        candidate_indices = [i for i in candidate_indices if i < N and (chosen_genres & set(genres_list_all[i]))]
        try:
            mlflow.log_metric("n_candidates", int(len(candidate_indices)))
        except Exception:
            pass

        if not candidate_indices:
            return {"run_id": run_id, "recommendations": []}

        # 5) Features pour XGB
        features_list, candidate_titles = [], []
        for i in candidate_indices:
            # SVD déjà calculé et aligné
            svd_vec = svd_vecs[i].reshape(1, -1)

            # distances kNN sur la matrice kNN recalc
            dists, _ = nn.kneighbors(tfidf_matrix[i], n_neighbors=6)
            sims = 1 - dists[0][1:] if dists.shape[1] > 1 else np.array([0.0])

            feature_vec = np.hstack([
                svd_vec.ravel(),
                genres_encoded_matrix[i],
                years_scaled[i],
                sims.mean(), sims.max(), sims.min(), sims.std()
            ])
            features_list.append(feature_vec)
            candidate_titles.append(titles[i])

        if not features_list:
            try:
                mlflow.log_metric("n_candidates", 0)
            except Exception:
                pass
            return {"run_id": run_id, "recommendations": []}

        X = np.array(features_list)
        proba = xgb_model.predict_proba(X)[:, 1]

        # 6) Normalisation robuste 0–1
        mn, mx = float(proba.min()), float(proba.max())
        proba_scaled = (proba - mn) / (mx - mn) if mx > mn else np.ones_like(proba)

        # 7) Top-K + construction des recos + logging par rang
        top_idx = np.argsort(proba_scaled)[::-1][:top_k]

        recos, rows_for_table = [], []
        for rank, idx_top in enumerate(top_idx, start=1):
            mv_title = candidate_titles[idx_top]
            movie = movies_dict.get(mv_title, {})
            user_rating = float(movie.get("user_rating") or 0.0)
            movie_rating = float(movie.get("rating") or 5.0)

            final_score = (
                0.6 * float(proba_scaled[idx_top]) +
                0.25 * (user_rating / 10.0) +
                0.15 * (movie_rating / 10.0)
            )
            pred_label = int(final_score >= 0.5)

            # Metrics par rang (protégées)
            try:
                mlflow.log_metric("pred_score", float(proba_scaled[idx_top]), step=rank)
                mlflow.log_metric("final_score", float(final_score), step=rank)
            except Exception:
                pass

            rows_for_table.append({
                "rank": rank,
                "title": mv_title,
                "pred_score_scaled": float(proba_scaled[idx_top]),
                "user_rating": user_rating,
                "movie_rating": movie_rating,
                "final_score": float(final_score),
                "pred_label": pred_label,
            })

            recos.append({
                "title": movie.get("title", mv_title),
                "poster_url": movie.get("poster_url"),
                "genres": movie.get("genres"),
                "synopsis": movie.get("synopsis"),
                "pred_score": float(final_score),
                "pred_label": pred_label,
            })

        try:
            mlflow.log_metric("max_pred_score", float(proba_scaled.max()))
            mlflow.log_metric("min_pred_score", float(proba_scaled.min()))
        except Exception:
            pass

        # 8) Artifacts (format JSON + tableau) — best effort
        try:
            df_recos = pd.DataFrame(rows_for_table)
            if hasattr(mlflow, "log_table"):
                mlflow.log_table(df_recos, artifact_file="recommendations.parquet")
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    path_csv = os.path.join(tmpdir, "recommendations.csv")
                    df_recos.to_csv(path_csv, index=False)
                    mlflow.log_artifact(path_csv)
            mlflow.log_dict([r["title"] for r in recos], artifact_file="top_titles.json")
        except Exception:
            pass

        return {"run_id": run_id, "recommendations": recos}

# -- Enregistrer le feedback utilisateur (like/dislike) + accuracy online
@app.post("/log_feedback", include_in_schema=False, dependencies=[Depends(verify_credentials)])
async def log_feedback(payload: FeedbackPayload):
    try:
        # 1) Sauvegarder le feedback
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO feedback (user_name, input_title, reco_title, pred_label, pred_score, liked, run_id)
                VALUES (:u, :it, :rt, :pl, :ps, :lk, :rid)
            """), {
                "u": payload.user_name,
                "it": payload.input_title,
                "rt": payload.reco_title,
                "pl": int(payload.pred_label),
                "ps": float(payload.pred_score),
                "lk": int(payload.liked),
                "rid": payload.run_id
            })

            # 2) Recalcul accuracy online (sur ce run)
            row = conn.execute(text("""
                SELECT SUM(CASE WHEN pred_label = liked THEN 1 ELSE 0 END) AS correct,
                       COUNT(*) AS total
                FROM feedback
                WHERE run_id = :rid
            """), {"rid": payload.run_id}).fetchone()

        correct, total = (row[0] or 0), (row[1] or 0)
        online_accuracy = float(correct) / float(total) if total > 0 else 0.0

        # 3) Log sans rouvrir le run (évite "already active")
        try:
            client = MlflowClient()
            client.log_metric(run_id=payload.run_id, key="online_accuracy", value=online_accuracy, step=total)
            client.log_metric(run_id=payload.run_id, key="feedback_count", value=total, step=total)
        except Exception as e:
            logger.warning(f"MLflowClient log_metric failed: {e}")

        return {"message": "Feedback enregistré", "online_accuracy": online_accuracy, "count": total}

    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Erreur SQLAlchemy : {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")

# -- Fuzzy match titres
@app.get("/fuzzy_match/{title}", include_in_schema=False, dependencies=[Depends(verify_credentials)])
async def fuzzy_match(
    title: str,
    limit_sql: int = 400, 
    top_k: int = 10,
    score_cutoff: int = 60
):
    q = title.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Titre vide.")

    # Normalisation & tokens
    q_norm = normalize_text(q).replace("&", " and ")   
    tokens = [t for t in re.findall(r"[a-z0-9]+", q_norm) if t and t not in STOPWORDS]
    if not tokens:
        tokens = [t for t in re.findall(r"[a-z0-9]+", q_norm)]

    # Requêtes très courtes => augmente le rappel
    if len(tokens) == 1:
        limit_sql = max(limit_sql, 800)
        score_cutoff = min(score_cutoff, 50)

    # FULLTEXT en BOOLEAN MODE avec préfixes (+tok*)
    bool_query = " ".join(f"+{t}*" for t in tokens if len(t) >= 2)

    with engine.connect() as conn:
        rows = []

        # 1) FULLTEXT BOOLEAN MODE
        if bool_query:
            try:
                rows = conn.execute(text("""
                    SELECT movie_id, title
                    FROM movies
                    WHERE MATCH(title) AGAINST(:bq IN BOOLEAN MODE)
                    LIMIT :lim
                """), {"bq": bool_query, "lim": int(limit_sql)}).fetchall()
            except SQLAlchemyError:
                rows = []

        # 2) FULLTEXT NATURAL LANGUAGE (fallback)
        if not rows:
            try:
                rows = conn.execute(text("""
                    SELECT movie_id, title
                    FROM movies
                    WHERE MATCH(title) AGAINST(:q IN NATURAL LANGUAGE MODE)
                    LIMIT :lim
                """), {"q": q, "lim": int(limit_sql)}).fetchall()
            except SQLAlchemyError:
                rows = []

        # 3) LIKE "normalisé" (retire espaces/ponctuation, & -> and)
        if not rows:
            norm_q = re.sub(r"[^a-z0-9]+", "", q_norm)
            try:
                rows = conn.execute(text("""
                    SELECT movie_id, title
                    FROM movies
                    WHERE
                        REPLACE(
                        REPLACE(
                        REPLACE(
                        REPLACE(
                        REPLACE(
                        REPLACE(
                        REPLACE(
                            LOWER(title),
                        ' ', ''), '-', ''), '''', ''), ':', ''), '.', ''), ',', ''), '&', 'and'
                        ) LIKE :norm_like
                    LIMIT :lim
                """), {"norm_like": f"%{norm_q}%", "lim": int(limit_sql)}).fetchall()
            except SQLAlchemyError:
                rows = []

        # 4) LIKE souple avec jokers entre tokens: %hung%games%songbirds%
        if not rows and tokens:
            like_chain = "%" + "%".join(tokens) + "%"
            try:
                rows = conn.execute(text("""
                    SELECT movie_id, title
                    FROM movies
                    WHERE LOWER(title) LIKE :like_chain
                    LIMIT :lim
                """), {"like_chain": like_chain, "lim": int(limit_sql)}).fetchall()
            except SQLAlchemyError:
                rows = []

    if not rows:
        raise HTTPException(status_code=404, detail="Aucune correspondance trouvée.")

    candidates = [r[1] for r in rows]
    id_by_title = {r[1]: r[0] for r in rows}

    # Rerank avec RapidFuzz
    from rapidfuzz import process, fuzz as rf_fuzz
    matches = process.extract(
        query=q,
        choices=candidates,
        scorer=rf_fuzz.WRatio,
        limit=top_k * 3,
        score_cutoff=int(score_cutoff)
    )

    # Si peu de résultats, on complète via partial_ratio (plus permissif)
    if len(matches) < top_k:
        extra = process.extract(
            query=q,
            choices=candidates,
            scorer=rf_fuzz.partial_ratio,
            limit=top_k * 3,
            score_cutoff=max(45, score_cutoff - 10)
        )
        # fusion garde le meilleur score par titre
        best = {}
        for t, s, *_ in matches + extra:
            best[t] = max(best.get(t, 0), int(s))
        matches = sorted(best.items(), key=lambda x: x[1], reverse=True)

    out, seen = [], set()
    for item in matches:
        # item = (title, score, idx) ou (title, score)
        t = item[0]
        s = int(item[1])
        if t in seen:
            continue
        seen.add(t)
        out.append({"title": t, "score": s, "movie_id": id_by_title.get(t)})
        if len(out) >= top_k:
            break

    if not out:
        raise HTTPException(status_code=404, detail="Aucune correspondance fiable trouvée.")
    return {"matches": out}

# -- Genres uniques (sur la BDD)
@app.get("/genres/", include_in_schema=False, dependencies=[Depends(verify_credentials)])
async def get_unique_genres():
    try:
        query = "SELECT genres FROM movies"
        all_genres = set()
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchall()
            for row in result:
                raw = row[0] or ""
                parts = [g.strip() for token in raw.split("|") for g in token.split(",")]
                parts = [g for g in parts if g]
                all_genres.update(parts)
        return sorted(all_genres)
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Erreur SQLAlchemy : {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")

# -- Détails d’un film + plateformes
@app.get("/movie-details/{title}", dependencies=[Depends(verify_credentials)])
async def get_movie_details(title: str):
    try:
        query = """
        SELECT 
            m.movie_id,
            m.title AS movie_title,
            m.genres AS movie_genres,
            m.release_year,
            m.rating AS movie_rating,
            m.synopsis,
            m.poster_url,
            p.platform_name
        FROM movies m
        LEFT JOIN (
            SELECT 'Netflix' AS platform_name, title FROM netflix
            UNION
            SELECT 'Prime' AS platform_name, title FROM prime
            UNION
            SELECT 'Hulu' AS platform_name, title FROM hulu
            UNION
            SELECT 'HBO Max' AS platform_name, title FROM hbo
            UNION
            SELECT 'Apple' AS platform_name, title FROM apple
        ) p
        ON m.title = p.title
        WHERE LOWER(m.title) = LOWER(:title)
        """
        with engine.connect() as conn:
            result = conn.execute(text(query), {"title": title}).fetchall()

        if not result:
            raise HTTPException(status_code=404, detail="Film non trouvé.")

        first = result[0]
        platforms = list({row[7] for row in result if row[7]})
        return {
            "movie_id": first[0],
            "title": first[1],
            "genres": first[2],
            "releaseYear": first[3],
            "rating": first[4],
            "synopsis": first[5],
            "poster_url": first[6],
            "platforms": platforms
        }
    
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Erreur SQLAlchemy : {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")

# -- Films aléatoires par genre et plateformes
PLATFORM_TABLES = {
    "netflix": "netflix",
    "prime": "prime",
    "hulu": "hulu",
    "hbo": "hbo",
    "apple": "apple"
}

@app.get("/random_movies/", dependencies=[Depends(verify_credentials)])
async def get_random_movies(genre: str, platforms: List[str] = Query(...), limit: int = 10):
    try:
        selected_platforms = [p.lower() for p in platforms if p.lower() in PLATFORM_TABLES]
        if not selected_platforms:
            raise HTTPException(status_code=400, detail="Aucune plateforme valide sélectionnée.")

        out = []
        with engine.connect() as conn:
            for platform in selected_platforms:
                query = f"""
                SELECT m.title, m.synopsis, m.poster_url, m.genres, '{platform}' AS platform, m.release_year
                FROM movies m
                JOIN {platform} p ON m.title = p.title
                WHERE FIND_IN_SET(:genre, REPLACE(m.genres, '|', ','))  -- gère '|' et ','
                """
                result = conn.execute(text(query), {"genre": genre}).fetchall()
                for row in result:
                    out.append({
                        "title": row[0],
                        "synopsis": row[1],
                        "poster_url": row[2],
                        "genres": row[3],
                        "platform": row[4],
                        "releaseYear": row[5]
                    })

        if not out:
            raise HTTPException(status_code=404, detail="Aucun film trouvé pour ce genre et ces plateformes.")

        # Echantillonner en évitant les entrées incomplètes
        filtered = [m for m in out if m.get("poster_url") and m.get("synopsis")]
        if not filtered:
            raise HTTPException(status_code=404, detail="Aucun film exploitable (poster/synopsis manquants).")

        return random.sample(filtered, min(limit, len(filtered)))

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Erreur SQLAlchemy : {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")

# -- Export CSV complet movie + plateformes (left join)
@app.get("/download-movie-details/", dependencies=[Depends(verify_credentials)])
async def download_movie_details():
    try:
        query = """
        SELECT 
            m.movie_id,
            m.title AS movie_title,
            m.genres AS movie_genres,
            m.release_year,
            m.rating AS movie_rating,
            m.synopsis,
            m.poster_url,
            COALESCE(p.platform_name, 'Not available') AS platform_name
        FROM movies m
        LEFT JOIN (
            SELECT 'Netflix' AS platform_name, title FROM netflix
            UNION
            SELECT 'Prime' AS platform_name, title FROM prime
            UNION
            SELECT 'Hulu' AS platform_name, title FROM hulu
            UNION
            SELECT 'HBO Max' AS platform_name, title FROM hbo
            UNION
            SELECT 'Apple' AS platform_name, title FROM apple
        ) p
        ON m.title = p.title
        ORDER BY m.title;
        """
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="Aucune donnée à télécharger.")

        csv_file = io.StringIO()
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([col for col in result.keys()])
        csv_writer.writerows(rows)
        csv_file.seek(0)

        return StreamingResponse(
            csv_file,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=movie_details.csv"}
        )
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Erreur SQLAlchemy : {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")

# -- Stats utilisateur : historique likes/dislikes, genres préférés, accuracy, etc.
@app.get("/user_stats/{user_name}", include_in_schema=False, dependencies=[Depends(verify_credentials)])
async def user_stats(user_name: str, limit_recent: int = 50):
    try:
        query = """
        SELECT f.ts, f.input_title, f.reco_title, f.pred_label, f.pred_score, f.liked,
               m.genres, m.release_year
        FROM feedback f
        LEFT JOIN movies m ON m.title = f.reco_title
        WHERE f.user_name = :u
        ORDER BY f.ts DESC
        """
        with engine.connect() as conn:
            rows = conn.execute(text(query), {"u": user_name}).fetchall()

        # Si aucun historique pour cet utilisateur, renvoyer un squelette vide
        if not rows:
            return {
                "user": user_name,
                "total": 0,
                "likes": 0,
                "dislikes": 0,
                "like_rate": 0.0,
                "accuracy": 0.0,
                "confusion": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
                "top_genres": [],
                "by_year": [],
                "recent": []
            }

        cols = ["ts","input_title","reco_title","pred_label","pred_score","liked","genres","release_year"]
        df = pd.DataFrame(rows, columns=cols)
        df["liked"] = df["liked"].astype(int)
        df["pred_label"] = df["pred_label"].astype(int)

        n = len(df)
        likes = int((df["liked"] == 1).sum())
        dislikes = int((df["liked"] == 0).sum())
        like_rate = float(likes / n) if n else 0.0
        accuracy = float((df["pred_label"] == df["liked"]).mean()) if n else 0.0

        tp = int(((df["pred_label"] == 1) & (df["liked"] == 1)).sum())
        tn = int(((df["pred_label"] == 0) & (df["liked"] == 0)).sum())
        fp = int(((df["pred_label"] == 1) & (df["liked"] == 0)).sum())
        fn = int(((df["pred_label"] == 0) & (df["liked"] == 1)).sum())

        # Genres préférés (sum des likes par genre)
        def split_genres(s):
            if not s:
                return []
            parts = []
            for token in str(s).split("|"):
                parts.extend([g.strip() for g in token.split(",")])
            return [g for g in parts if g]

        df["genres_list"] = df["genres"].apply(split_genres)
        expl = df.explode("genres_list")
        if not expl.empty:
            likes_by_genre = expl.groupby("genres_list")["liked"].sum().sort_values(ascending=False)
            top_genres = [{"genre": k, "likes": int(v)} for k, v in likes_by_genre.head(10).items()]
        else:
            top_genres = []

        # Likes par année de sortie
        by_year = df.groupby("release_year")["liked"].sum().sort_index().reset_index()
        by_year = [
            {"year": int(y) if pd.notna(y) else None, "likes": int(v)}
            for y, v in zip(by_year["release_year"], by_year["liked"])
        ]

        # Récents
        df_recent = df[["ts","reco_title","pred_score","pred_label","liked","genres","release_year"]].head(limit_recent).copy()
        df_recent["ts"] = df_recent["ts"].astype(str)
        recent = df_recent.to_dict(orient="records")

        return {
            "user": user_name,
            "total": n,
            "likes": likes,
            "dislikes": dislikes,
            "like_rate": like_rate,
            "accuracy": accuracy,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "top_genres": top_genres,
            "by_year": by_year,
            "recent": recent
        }

    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Erreur SQLAlchemy : {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")

# -- NEW: Historique des notes utilisateur (films notés via /update_rating)
@app.get("/user_ratings/{user_name}", include_in_schema=False, dependencies=[Depends(verify_credentials)])
async def get_user_ratings(user_name: str, limit: int = 200):
    try:
        query = """
        SELECT ur.ts, ur.title, ur.rating, m.genres, m.release_year, m.poster_url
        FROM user_ratings ur
        LEFT JOIN movies m ON m.title = ur.title
        WHERE ur.user_name = :u
        ORDER BY ur.ts DESC
        LIMIT :lim
        """
        with engine.connect() as conn:
            rows = conn.execute(text(query), {"u": user_name, "lim": int(limit)}).mappings().all()
        return {"ratings": [dict(r) for r in rows]}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Erreur SQLAlchemy : {str(e)}")

# -- DEBUG / SANTÉ DU CATALOGUE
@app.get("/_debug_catalog", include_in_schema=False)
async def _debug_catalog():
    try:
        return {
            "n_titles": len(titles),
            "tfidf_rows": int(getattr(tfidf_matrix, "shape", [0])[0]),
            "svd_rows": int(getattr(svd_vecs, "shape", [0])[0]),
            "genres_rows": int(getattr(genres_encoded_matrix, "shape", [0])[0]),
            "years_rows": int(getattr(years_scaled, "shape", [0])[0]),
            "sample_titles": titles[:5],
        }
    except Exception as e:
        return {"error": str(e)}

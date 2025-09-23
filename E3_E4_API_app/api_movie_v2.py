# api.py
# ============================================================
# 🎬 Louve Movies API — Recommandations + Monitoring MLflow
# ============================================================
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials, APIKeyHeader
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from rapidfuzz import process, fuzz
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
import traceback

# -------- Config locale du projet --------
# (le module E3_E4_API_app.config doit définir MLFLOW_TRACKING_URI)
from E3_E4_API_app import config

STOPWORDS = {"the","of","and","a","an","la","le","les","de","des","du","et"}

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
app = FastAPI(title="🎬 API CinéNocturne")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    error_message = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.error(f"Unhandled error: {error_message}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error": str(exc)},
    )
# ======================
# 🧪 MLflow
# ======================
mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment("louve_movies_monitoring")

# ======================
# 🗄️ Base de données
# ======================
DATABASE_URL = "mysql+pymysql://louve:%40Marley080922@mysql-louve.alwaysdata.net/louve_movies"
engine = create_engine(DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600)

# Crée les tables si absentes
with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_name VARCHAR(255),
            input_title VARCHAR(512),
            reco_title VARCHAR(512),
            pred_label TINYINT,
            pred_score FLOAT,
            liked TINYINT,
            run_id VARCHAR(64)
        )
    """))
    # NEW: historique des notes utilisateurs
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS user_ratings (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_name VARCHAR(255),
            title VARCHAR(512),
            rating FLOAT
        )
    """))

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
# 🤖 Chargement modèles & données
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Modèles / objets
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_classifier_model.joblib"))
scaler_proba = joblib.load(os.path.join(MODEL_DIR, "scaler_proba.joblib"))
mlb = joblib.load(os.path.join(MODEL_DIR, "mlb_model.joblib"))
scaler_year = joblib.load(os.path.join(MODEL_DIR, "scaler_year.joblib"))
nn_full = joblib.load(os.path.join(MODEL_DIR, "nn_full.joblib"))
svd_full = joblib.load(os.path.join(MODEL_DIR, "svd_model.joblib"))
tfidf_matrix = joblib.load(os.path.join(MODEL_DIR, "tfidf_matrix_full.joblib"))

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

# Films depuis BDD (mémoire)
with engine.connect() as conn:
    rows = conn.execute(
        text("""
            SELECT movie_id, title, genres, release_year, synopsis, poster_url, rating, user_rating
            FROM movies
        """)
    ).mappings().all()
    
if not rows:
    rows = []


movies = []
titles = []
genres_list_all = []
years = []

for r in rows:
    raw = r["genres"] or ""
    parts = []
    for token in raw.split("|"):
        parts.extend([g.strip() for g in token.split(",") if g.strip()])
    genres_list = parts
    movies.append(dict(r))
    titles.append(r["title"])
    genres_list_all.append(genres_list)
    years.append(r["release_year"] if r["release_year"] else 2000)

def safe_mlb_transform(mlb, genre_lists):
    try:
        return mlb.transform(genre_lists)
    except ValueError:
        known = set(mlb.classes_)
        cleaned = [[g for g in gs if g in known] for gs in genre_lists]
        return mlb.transform(cleaned)

# Encodage genres & années
genres_encoded_matrix = safe_mlb_transform(mlb, genres_list_all)
years_scaled = scaler_year.transform(np.array([[y] for y in years]))

# Accès rapide par titre (sensible à la casse telle que BDD)
movies_dict = {movie["title"]: movie for movie in movies}

# ======================
# 🧠 Helpers
# ======================
def normalize_text(text: str) -> str:
    text = text.strip()
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    return text.lower()

def normalize_for_match(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("utf-8")
    s = s.lower().replace("&", " and ")
    # leet → lettres (permet "m3gan" ~ "megan")
    leet_map = str.maketrans({
        "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"
    })
    s = s.translate(leet_map)
    return s

# === Helpers "saga" / suites de films ===
# (repose sur re et rapidfuzz.fuzz déjà importés)
_ROMAN = r"(?:\b[ivxlcdm]+\b)"
_SAGA_HINTS = r"(?:part|episode|chapitre|chapter|vol\.?|volume|saga|collection)"

def _base_title(s: str) -> str:
    """Titre 'racine' pour comparer les suites: retire parenthèses, sous-titres, numéros, etc."""
    if not s:
        return ""
    s = normalize_for_match(s)
    s = re.sub(r"\(.*?\)", " ", s)                 # retire parenthèses
    s = re.split(r":|-", s)[0]                      # coupe aux sous-titres
    s = re.sub(fr"\b{_SAGA_HINTS}\b.*", " ", s)    # retire "part/episode/vol..."
    s = re.sub(fr"\b\d+\b", " ", s)                # nombres arabes (2, 3, 4…)
    s = re.sub(fr"{_ROMAN}", " ", s)               # nombres romains (II, III…)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def title_saga_similarity(a: str, b: str) -> float:
    """Similarité 0..1 entre les 'bases' de titres (robuste aux variantes)."""
    a0, b0 = _base_title(a), _base_title(b)
    if not a0 or not b0:
        return 0.0
    return fuzz.token_set_ratio(a0, b0) / 100.0


@contextmanager
def mlflow_start_inference_run(input_title: str, top_k: int):
    """Démarre un run MLflow si possible. Sinon, no-op avec run_id='no-mlflow'."""
    try:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("louve_movies_monitoring")
        if mlflow.active_run() is not None:
            mlflow.end_run()
        run = mlflow.start_run(run_name=f"recommend_{input_title}")
        ok = True
    except Exception as e:
        logger.warning("MLflow indisponible: %s (mode no-mlflow)", e)
        class _DummyRun:
            run_id = "no-mlflow"
        run, ok = _DummyRun(), False

    try:
        yield run, ok
    finally:
        try:
            if ok:
                mlflow.end_run()
        except Exception:
            pass

def build_platform_union_sql() -> str:
    """
    Construit une sous-requête UNION ALL : (platform_name, movie_id, imdb_id)
    à partir de PLATFORM_TABLES.
    """
    parts = []
    for label, tbl in PLATFORM_TABLES.values():
        parts.append(f"SELECT '{label}' AS platform_name, movie_id, imdb_id FROM `{tbl}`")
    return "\nUNION ALL\n".join(parts)


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


# -- Recommandations XGB personnalisées
@app.get("/recommend_xgb_personalized/{title}")
async def recommend_xgb_personalized(title: str, top_k: int = 5):
    # 1. Index du film d'entrée
    try:
        idx = next(i for i, t in enumerate(titles) if t.lower() == title.lower())
    except StopIteration:
        raise HTTPException(status_code=404, detail="Film non trouvé")

    exact_title = titles[idx]

    # --- NEW: détecter les films de la même "saga" (liste séparée)
    SAGA_SIM_THRESHOLD = 0.90  # 70%
    saga_candidate_idxs = [
        j for j, tj in enumerate(titles)
        if j != idx and title_saga_similarity(exact_title, tj) >= SAGA_SIM_THRESHOLD
    ]

    with mlflow_start_inference_run(input_title=exact_title, top_k=top_k) as (run, ok):
        try:
            run_id = str(getattr(run, "info", None).run_id if ok else "no-mlflow")
        except Exception:
            run_id = "no-mlflow"

        # Logging helpers
        def _log_metric(k, v, step=None):
            if ok:
                try: mlflow.log_metric(k, v, step=step)
                except Exception: pass

        def _log_param(k, v):
            if ok:
                try: mlflow.log_param(k, v)
                except Exception: pass

        def _log_table_df(df):
            if ok:
                try:
                    if hasattr(mlflow, "log_table"):
                        mlflow.log_table(df, artifact_file="recommendations.parquet")
                    else:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            p = os.path.join(tmpdir, "recommendations.csv")
                            df.to_csv(p, index=False)
                            mlflow.log_artifact(p)
                except Exception: pass

        _log_param("input_title", exact_title)
        _log_param("top_k", top_k)

        # 2. Données du film d’entrée
        input_movie = movies_dict.get(exact_title, {})
        input_user_rating = float(input_movie.get("user_rating") or 0.0)
        input_movie_rating = float(input_movie.get("rating") or 5.0)

        _log_metric("user_rating", input_user_rating)
        _log_metric("movie_rating", input_movie_rating)

        # 3. Candidats
        chosen_genres = set(genres_list_all[idx])
        vec = tfidf_matrix[idx].reshape(1, -1)
        cosine_sim = cosine_similarity(vec, tfidf_matrix).flatten()
        cosine_sim[idx] = -1.0

        candidate_indices = np.argsort(cosine_sim)[-50:][::-1]
        candidate_indices = [i for i in candidate_indices if chosen_genres & set(genres_list_all[i])]

        _log_metric("n_candidates", len(candidate_indices))

        if not candidate_indices:
            return {"run_id": run_id, "recommendations": []}

        # 4. Features
        features_list, candidate_titles = [], []
        for i in candidate_indices:
            svd_vec = svd_full.transform(tfidf_matrix[i].reshape(1, -1))
            nn_distances, _ = nn_full.kneighbors(tfidf_matrix[i].reshape(1, -1))
            sims = 1 - nn_distances[0][1:] if nn_distances.shape[1] > 1 else np.array([0.0])
            feature_vec = np.hstack([
                svd_vec[0],
                genres_encoded_matrix[i],
                years_scaled[i],
                sims.mean(), sims.max(), sims.min(), sims.std()
            ])
            features_list.append(feature_vec)
            candidate_titles.append(titles[i])

        if not features_list:
            _log_metric("n_candidates", 0)
            return {"run_id": run_id, "recommendations": []}

        X = np.array(features_list)
        proba = xgb_model.predict_proba(X)[:, 1]

        # 5. ⚖️ Normalisation via scaler_proba.joblib
        try:
            scaler_proba = joblib.load(os.path.join(MODEL_DIR, "scaler_proba.joblib"))
            proba_scaled = scaler_proba.transform(proba.reshape(-1, 1)).flatten()
        except Exception:
            # fallback: min-max local
            mn, mx = proba.min(), proba.max()
            proba_scaled = (proba - mn) / (mx - mn) if mx > mn else np.ones_like(proba)

        _log_metric("raw_proba_mean", float(proba.mean()))
        _log_metric("raw_proba_std", float(proba.std()))

        # 6. Top-K + scoring
        top_idx = np.argsort(proba_scaled)[::-1][:top_k]
        recos, rows_for_table = [], []

        for rank, idx_top in enumerate(top_idx, start=1):
            mv_title = candidate_titles[idx_top]
            movie = movies_dict.get(mv_title, {})
            user_rating = float(movie.get("user_rating") or 0.0)
            movie_rating = float(movie.get("rating") or 5.0)

            # ✅ Nouveau score mieux équilibré
            final_score = (
                0.75 * float(proba_scaled[idx_top]) +
                0.15 * (user_rating / 10.0) +
                0.10 * (movie_rating / 10.0)
            )
            pred_label = int(final_score >= 0.5)

            _log_metric("raw_proba", float(proba[idx_top]), step=rank)
            _log_metric("pred_score", float(proba_scaled[idx_top]), step=rank)
            _log_metric("final_score", float(final_score), step=rank)

            rows_for_table.append({
                "rank": rank,
                "title": mv_title,
                "pred_score_scaled": float(proba_scaled[idx_top]),
                "user_rating": user_rating,
                "movie_rating": movie_rating,
                "final_score": final_score,
                "pred_label": pred_label,
            })

            recos.append({
                "title": movie.get("title", mv_title),
                "poster_url": movie.get("poster_url"),
                "genres": movie.get("genres"),
                "synopsis": movie.get("synopsis"),
                "pred_score": final_score,
                "pred_label": pred_label,
            })

        _log_table_df(pd.DataFrame(rows_for_table))

        # === NEW: Section séparée "saga_recommendations"
        main_reco_titles = {r["title"] for r in recos}
        
        # titres de la même saga, hors film d'entrée et hors recos principales
        saga_titles = [
            titles[j] for j in saga_candidate_idxs
            if titles[j].lower() != exact_title.lower() and titles[j] not in main_reco_titles
        ]
        
        # (optionnel) limite douce
        MAX_SAGA = 12
        saga_titles = saga_titles[:MAX_SAGA]
        
        saga_recos = []
        for t in saga_titles:
            m = movies_dict.get(t, {})
            saga_recos.append({
                "title": m.get("title", t),
                "poster_url": m.get("poster_url"),
                "genres": m.get("genres"),
                "synopsis": m.get("synopsis"),
                "releaseYear": m.get("release_year"),
                "rating": m.get("rating"),
                "saga_boost": True
            })


        return {
            "run_id": run_id,
            "recommendations": recos,          # <= top_k classique
            "saga_recommendations": saga_recos # <= AJOUTÉ EN PLUS (ne réduit pas top_k)
        }


# -- Enregistrer le feedback utilisateur (like/dislike) + accuracy online
@app.post("/log_feedback", include_in_schema=False, dependencies=[Depends(verify_credentials)])
async def log_feedback(payload: FeedbackPayload):
    try:
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

            row = conn.execute(text("""
                SELECT SUM(CASE WHEN pred_label = liked THEN 1 ELSE 0 END) AS correct,
                       COUNT(*) AS total
                FROM feedback
                WHERE run_id = :rid
            """), {"rid": payload.run_id}).fetchone()

        correct, total = (row[0] or 0), (row[1] or 0)
        online_accuracy = float(correct) / float(total) if total > 0 else 0.0

        # 🔒 Log MLflow seulement si on a un vrai run_id
        if payload.run_id and payload.run_id != "no-mlflow":
            try:
                client = MlflowClient()
                client.log_metric(run_id=payload.run_id, key="online_accuracy", value=online_accuracy, step=total)
                client.log_metric(run_id=payload.run_id, key="feedback_count", value=total, step=total)
            except Exception as e:
                logger.warning(f"MLflow logging skipped: {e}")

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

        # 1) FULLTEXT BOOLEAN MODE (title + original_title)
        if bool_query:
            try:
                rows = conn.execute(text("""
                    SELECT movie_id, title
                    FROM movies
                    WHERE MATCH(title, original_title) AGAINST(:bq IN BOOLEAN MODE)
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
                    WHERE MATCH(title, original_title) AGAINST(:q IN NATURAL LANGUAGE MODE)
                    LIMIT :lim
                """), {"q": q, "lim": int(limit_sql)}).fetchall()
            except SQLAlchemyError:
                rows = []

        # 3) LIKE "normalisé" (sur title + original_title)
        if not rows:
            norm_q = re.sub(r"[^a-z0-9]+", "", q_norm)
            try:
                rows = conn.execute(text("""
                    SELECT movie_id, title
                    FROM movies
                    WHERE
                        REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
                            LOWER(CONCAT_WS(' ', title, COALESCE(original_title,''))),
                        ' ', ''), '-', ''), '''', ''), ':', ''), '.', ''), ',', ''), '&', 'and'
                        ) LIKE :norm_like
                    LIMIT :lim
                """), {"norm_like": f"%{norm_q}%", "lim": int(limit_sql)}).fetchall()
            except SQLAlchemyError:
                rows = []

        # 4) LIKE souple avec jokers (sur title + original_title)
        if not rows and tokens:
            like_chain = "%" + "%".join(tokens) + "%"
            try:
                rows = conn.execute(text("""
                    SELECT movie_id, title
                    FROM movies
                    WHERE LOWER(CONCAT_WS(' ', title, COALESCE(original_title,''))) LIKE :like_chain
                    LIMIT :lim
                """), {"like_chain": like_chain, "lim": int(limit_sql)}).fetchall()
            except SQLAlchemyError:
                rows = []

    if not rows:
        raise HTTPException(status_code=404, detail="Aucune correspondance trouvée.")

    candidates = [r[1] for r in rows]
    id_by_title = {r[1]: r[0] for r in rows}

    # Rerank avec RapidFuzz
    matches = process.extract(
        query=q,
        choices=candidates,
        scorer=fuzz.WRatio,
        limit=top_k * 3,
        score_cutoff=int(score_cutoff)
    )

    # Si peu de résultats, on complète via partial_ratio (plus permissif)
    if len(matches) < top_k:
        extra = process.extract(
            query=q,
            choices=candidates,
            scorer=fuzz.partial_ratio,
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
        # ⚡ reconnect à chaque appel
        with engine.begin() as conn:
            result = conn.execute(text(query)).fetchall()
            for row in result:
                raw = row[0] or ""
                parts = [g.strip() for token in raw.split("|") for g in token.split(",")]
                all_genres.update([g for g in parts if g])
        return sorted(all_genres)
    except SQLAlchemyError as e:
        # Si erreur 2006 -> reconnect
        if "2006" in str(e):
            engine.dispose()  # force reset de toutes les connexions
        raise HTTPException(status_code=500, detail=f"Erreur SQLAlchemy : {str(e)}")

# -- Détails d’un film + plateformes
@app.get("/movie-details/{title}", dependencies=[Depends(verify_credentials)])
async def get_movie_details(title: str):
    try:
        platforms_union = build_platform_union_sql()
        query = f"""
        WITH p AS (
            {platforms_union}
        )
        SELECT 
            m.movie_id,
            m.title AS movie_title,
            m.genres AS movie_genres,
            m.release_year,
            m.rating AS movie_rating,
            m.synopsis,
            m.poster_url,
            pu.platform_name
        FROM movies m
        LEFT JOIN p pu
          ON (pu.movie_id = m.movie_id)
          OR (pu.imdb_id IS NOT NULL AND pu.imdb_id <> '' AND pu.imdb_id = m.imdb_id)
        WHERE LOWER(m.title) = LOWER(:title)
        """
        with engine.connect() as conn:
            result = conn.execute(text(query), {"title": title}).fetchall()

        if not result:
            raise HTTPException(status_code=404, detail="Film non trouvé.")

        first = result[0]
        platforms = sorted({row[7] for row in result if row[7]})
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

# -- Films aléatoires par genre et plateformes + mapping tables
PLATFORM_TABLES = {
    "netflix":      ("Netflix", "netflix"),
    "prime":        ("Prime Video", "prime"),
    "apple":        ("Apple TV+", "apple"),
    "canal":        ("Canal+", "canal"),
    "disney":       ("Disney+", "disney"),
    "paramount":    ("Paramount+", "paramount"),
    "hbo":          ("HBO Max", "hbo"),
    "crunchyroll":  ("Crunchyroll", "crunchyroll"),
}


@app.get("/random_movies/", dependencies=[Depends(verify_credentials)])
async def get_random_movies(genre: str, platforms: List[str] = Query(...), limit: int = 10):
    try:
        # valider les plateformes demandées
        selected = []
        for p in platforms:
            key = p.lower().strip()
            if key in PLATFORM_TABLES:
                selected.append(key)
        if not selected:
            raise HTTPException(status_code=400, detail="Aucune plateforme valide sélectionnée.")

        out = []
        with engine.connect() as conn:
            for key in selected:
                label, tbl = PLATFORM_TABLES[key]
                query = f"""
                SELECT 
                    m.title, m.synopsis, m.poster_url, m.genres, :label AS platform, m.release_year
                FROM movies m
                JOIN `{tbl}` p
                  ON (p.movie_id = m.movie_id)
                  OR (p.imdb_id IS NOT NULL AND p.imdb_id <> '' AND p.imdb_id = m.imdb_id)
                WHERE FIND_IN_SET(:genre, REPLACE(m.genres, '|', ','))
                """
                result = conn.execute(text(query), {"genre": genre, "label": label}).fetchall()
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

        # filtrer les films exploitables et échantillonner
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
        platforms_union = build_platform_union_sql()
        query = f"""
        WITH p AS (
            {platforms_union}
        )
        SELECT 
            m.movie_id,
            m.title AS movie_title,
            m.genres AS movie_genres,
            m.release_year,
            m.rating AS movie_rating,
            m.synopsis,
            m.poster_url,
            COALESCE(pp.platform_name, 'Not available') AS platform_name
        FROM movies m
        LEFT JOIN p pp
          ON (pp.movie_id = m.movie_id)
          OR (pp.imdb_id IS NOT NULL AND pp.imdb_id <> '' AND pp.imdb_id = m.imdb_id)
        ORDER BY m.title
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

        # Squelette vide si aucun historique
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

        # 🔧 PATCH: robustifier la conversion (gère NULL / chaînes vides)
        df["liked"] = pd.to_numeric(df["liked"], errors="coerce").fillna(0).astype(int)
        df["pred_label"] = pd.to_numeric(df["pred_label"], errors="coerce").fillna(0).astype(int)

        n = len(df)
        likes = int((df["liked"] == 1).sum())
        dislikes = int((df["liked"] == 0).sum())
        like_rate = float(likes / n) if n else 0.0
        accuracy = float((df["pred_label"] == df["liked"]).mean()) if n else 0.0

        tp = int(((df["pred_label"] == 1) & (df["liked"] == 1)).sum())
        tn = int(((df["pred_label"] == 0) & (df["liked"] == 0)).sum())
        fp = int(((df["pred_label"] == 1) & (df["liked"] == 0)).sum())
        fn = int(((df["pred_label"] == 0) & (df["liked"] == 1)).sum())

        # Genres préférés (sum des likes par genre) — tolérant aux formats "|" et ","
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

        # Likes par année de sortie (conserve les NaN au groupby si besoin)
        by_year_df = df.groupby("release_year", dropna=False)["liked"].sum().sort_index().reset_index()
        by_year = [
            {"year": int(y) if pd.notna(y) else None, "likes": int(v)}
            for y, v in zip(by_year_df["release_year"], by_year_df["liked"])
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
        SELECT 
            ur.ts,
            ur.title,
            ur.rating,                -- note donnée par l'utilisateur
            m.genres,
            m.release_year,
            m.poster_url,
            m.rating AS movie_rating  -- ✅ note "globale" du film (depuis la table movies)
        FROM user_ratings ur
        LEFT JOIN movies m ON m.title = ur.title
        WHERE ur.user_name = :u
        ORDER BY ur.ts DESC
        LIMIT :lim
        """
        with engine.connect() as conn:
            rows = conn.execute(
                text(query), {"u": user_name, "lim": int(limit)}
            ).mappings().all()

        # (optionnel) nettoyage des types
        out = []
        for r in rows:
            d = dict(r)
            if d.get("rating") is not None:
                d["rating"] = float(d["rating"])
            if d.get("movie_rating") is not None:
                d["movie_rating"] = float(d["movie_rating"])
            if d.get("release_year") is not None:
                d["release_year"] = int(d["release_year"])
            out.append(d)

        return {"ratings": out}

    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Erreur SQLAlchemy : {str(e)}")





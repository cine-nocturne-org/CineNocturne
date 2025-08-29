from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz
import unicodedata
import random
import io
from datetime import datetime
import os
import csv
import mlflow
from E3_E4_API_app import config
import logging
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials, APIKeyHeader

# Configuration du logger (mettre ça en début de fichier)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


app = FastAPI(title="🎬 Louve Movies API")

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment("louve_movies_monitoring")

# ------------------------------
# Configuration BDD
# ------------------------------
DATABASE_URL = "mysql+pymysql://louve:%40Marley080922@mysql-louve.alwaysdata.net/louve_movies"
engine = create_engine(DATABASE_URL)

# ------------------------------
# Chargement des modèles ML
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_classifier_model.joblib"))
mlb = joblib.load(os.path.join(MODEL_DIR, "mlb_model.joblib"))
scaler_year = joblib.load(os.path.join(MODEL_DIR, "scaler_year.joblib"))
nn_full = joblib.load(os.path.join(MODEL_DIR, "nn_full.joblib"))
reco_vectorizer = joblib.load(os.path.join(MODEL_DIR, "reco_vectorizer.joblib"))
svd_full = joblib.load(os.path.join(MODEL_DIR, "svd_model.joblib"))
tfidf_matrix = joblib.load(os.path.join(MODEL_DIR, "tfidf_matrix_full.joblib"))
movie_index_df = pd.read_csv(os.path.join(MODEL_DIR, "movie_index.csv"))

# ------------------------------
# Chargement des films depuis BDD
# ------------------------------
with engine.connect() as conn:
    rows = conn.execute(
        text("SELECT movie_id, title, genres, release_year, synopsis, poster_url FROM movies")
    ).mappings().all()

movies = []
titles = []
genres_list_all = []
years = []

for r in rows:
    genres_list = r["genres"].split("|") if r["genres"] else []
    movies.append(dict(r))
    titles.append(r["title"])
    genres_list_all.append(genres_list)
    years.append(r["release_year"] if r["release_year"] else 2000)

# Encodage genres
genres_encoded_matrix = mlb.transform(genres_list_all)
years_scaled = scaler_year.transform(np.array([[y] for y in years]))

# Dict pour accès rapide par titre
movies_dict = {movie["title"]: movie for movie in movies}

# ------------------------------
# Fonction d'interprétation des scores
# ------------------------------
def interpret_score(score: float) -> str:
    pct = int(score * 100)
    if score >= 0.85:
        return f"🎯 Très forte recommandation ({pct}%)"
    elif score >= 0.7:
        return f"👍 Forte similarité ({pct}%)"
    elif score >= 0.5:
        return f"🤔 Moyennement similaire ({pct}%)"
    else:
        return f"⚠️ Faible similarité ({pct}%)"

# ------------------------------
# Tables plateformes
# ------------------------------
PLATFORM_TABLES = {
    "netflix": "netflix",
    "prime": "prime",
    "hulu": "hulu",
    "hbo": "hbo",
    "apple": "apple"
}

# ------------------------------
# Authentification HTTP Basic / Bearer
# ------------------------------
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

# ------------------------------
# Pydantic model
# ------------------------------
class RatingUpdate(BaseModel):
    title: str
    rating: float

# ------------------------------
# Route: Mise à jour note utilisateur
# ------------------------------
@app.post("/update_rating", dependencies=[Depends(verify_credentials)])
async def update_rating(payload: RatingUpdate):
    title = payload.title
    rating = payload.rating

    if rating < 0 or rating > 10:
        raise HTTPException(status_code=400, detail="La note doit être comprise entre 0 et 10.")

    try:
        with engine.begin() as conn:
            # Vérifie si la colonne user_rating existe
            check_col = conn.execute(text("""
                SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'movies' AND COLUMN_NAME = 'user_rating'
            """)).fetchone()
            if not check_col:
                conn.execute(text("ALTER TABLE movies ADD COLUMN user_rating FLOAT"))

            # Mise à jour
            result = conn.execute(
                text("UPDATE movies SET user_rating = :rating WHERE title = :title"),
                {"rating": rating, "title": title}
            )
            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail=f"Film '{title}' non trouvé.")

        return {"message": f"La note {rating} a été enregistrée pour le film '{title}'."}

    except HTTPException:  # ⚡ On laisse passer le 404
        raise
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Erreur SQLAlchemy : {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")


# ------------------------------
# Route: Recommandation XGB personnalisée
# ------------------------------
@app.get("/recommend_xgb_personalized/{title}")
async def recommend_xgb_personalized(title: str, top_k: int = 5):
    # Chercher le film
    try:
        idx = next(i for i, t in enumerate(titles) if t.lower() == title.lower())
    except StopIteration:
        raise HTTPException(status_code=404, detail="Film non trouvé")

    # --- Début MLflow run ---
    with mlflow.start_run(run_name=f"recommend_{title}") as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")
    
        chosen_genres = set(genres_list_all[idx])

        vec = tfidf_matrix[idx].reshape(1, -1)
        cosine_sim = cosine_similarity(vec, tfidf_matrix).flatten()
        cosine_sim[idx] = -1  # exclure le film lui-même

        candidate_indices = np.argsort(cosine_sim)[-50:][::-1]
        candidate_indices = [i for i in candidate_indices if chosen_genres & set(genres_list_all[i])]

        features_list = []
        candidate_titles = []

        for i in candidate_indices:
            svd_vec = svd_full.transform(tfidf_matrix[i])
            nn_distances, _ = nn_full.kneighbors(tfidf_matrix[i])
            neighbor_mean = (1 - nn_distances[0][1:]).mean()
            neighbor_max = (1 - nn_distances[0][1:]).max()
            neighbor_min = (1 - nn_distances[0][1:]).min()
            neighbor_std = (1 - nn_distances[0][1:]).std()

            feature_vec = np.hstack([
                svd_vec[0],
                genres_encoded_matrix[i],
                years_scaled[i],
                neighbor_mean,
                neighbor_max,
                neighbor_min,
                neighbor_std
            ])
            features_list.append(feature_vec)
            candidate_titles.append(titles[i])

        if not features_list:
            return []

        all_features = np.array(features_list)
        pred_scores = xgb_model.predict_proba(all_features)[:, 1]

        # Normalisation
        min_score, max_score = pred_scores.min(), pred_scores.max()
        pred_scores_scaled = (pred_scores - min_score) / (max_score - min_score) if max_score > min_score else np.ones_like(pred_scores)

        top_indices = np.argsort(pred_scores_scaled)[::-1][:top_k]

        top_recos_list = []
        for idx_top in top_indices:
            movie = movies_dict[candidate_titles[idx_top]]
            user_rating = movie.get("user_rating") or 0.0
            movie_rating = movie.get("rating") or 5.0
            score_final = 0.6 * pred_scores_scaled[idx_top] + 0.25 * (user_rating / 10) + 0.15 * (movie_rating / 10)

            # ⚡ Log dans MLflow
            mlflow.log_metric(f"pred_score_{movie['title']}", float(pred_scores_scaled[idx_top]))
            mlflow.log_metric(f"user_rating_{movie['title']}", float(user_rating))
            mlflow.log_metric(f"movie_rating_{movie['title']}", float(movie_rating))
            
            # Log de la différence entre prédiction et note utilisateur
            score_diff = abs(pred_scores_scaled[idx_top] - (user_rating / 10))
            mlflow.log_metric(f"score_diff_{movie['title']}", score_diff)

            top_recos_list.append({
                "title": movie["title"],
                "poster_url": movie.get("poster_url"),
                "releaseYear": movie.get("release_year"),
                "genres": [g.strip() for g in movie.get("genres", "").replace(",", "|").split("|") if g.strip()],
                "synopsis": movie.get("synopsis"),
                "platforms": [],  # ⚡ dispo seulement dans movie-details
                "pred_score": float(score_final)
            })
        

        # --- Log dans MLflow ---
        mlflow.log_param("input_title", title)
        mlflow.log_param("top_k", top_k)
        mlflow.log_metric("max_score", float(pred_scores_scaled.max()))
        mlflow.log_metric("min_score", float(pred_scores_scaled.min()))
        mlflow.log_text(str([r["title"] for r in top_recos_list]), "top_recommended_titles.txt")

        # ⚡ Ici on log chaque score de recommandation
        for i, reco in enumerate(top_recos_list):
            mlflow.log_metric(f"pred_score_{i}", reco["pred_score"])

    return top_recos_list
    
# ------------------------------
# Route: Fuzzy match
# ------------------------------
def normalize_text(text: str) -> str:
    text = text.strip()
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    return text.lower()

@app.get("/fuzzy_match/{title}", dependencies=[Depends(verify_credentials)])
async def fuzzy_match(title: str):
    title_input = normalize_text(title.strip())
    choices = [(normalize_text(c), c) for c in movie_index_df["title"].tolist()]
    movies_dict_normalized = {normalize_text(k): v for k, v in movies_dict.items()}

    matches = process.extract(
        query=title_input,
        choices=[c[0] for c in choices],
        scorer=fuzz.WRatio,
        limit=20,
        score_cutoff=70
    )

    if not matches:
        raise HTTPException(status_code=404, detail="Aucune correspondance ≥70% trouvée.")

    seen_titles = set()
    filtered = []

    for match in matches:
        norm_title, score, idx = match
        original_title = choices[idx][1]
        if original_title in seen_titles:
            continue
        partial = fuzz.partial_ratio(title_input, norm_title)
        if partial >= 60:
            seen_titles.add(original_title)
            filtered.append({
                "title": original_title,
                "score": round(score),
                "movie_id": movies_dict_normalized.get(norm_title, {}).get("movie_id")
            })
        if len(filtered) >= 10:
            break

    if not filtered:
        raise HTTPException(status_code=404, detail="Aucune correspondance fiable trouvée.")

    return {"matches": filtered}


# ------------------------------
# Route: Genres uniques
# ------------------------------
@app.get("/genres/", dependencies=[Depends(verify_credentials)])
async def get_unique_genres():
    try:
        query = "SELECT genres FROM movies"
        all_genres = set()
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchall()
            for row in result:
                genres = row[0].split(",")
                all_genres.update([g.strip() for g in genres])
        return sorted(all_genres)
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Erreur SQLAlchemy : {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")

# ------------------------------
# Route: Détails d'un film
# ------------------------------
@app.get("/movie-details/{title}", dependencies=[Depends(verify_credentials)])
async def get_movie_details(title: str):
    try:
        query = f"""
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
            raise HTTPException(status_code=404, detail="Film non trouvé.")  # ✅ avant d'accéder à row[7]

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
    
    except HTTPException:  # ✅ ne pas avaler ton 404
        raise
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Erreur SQLAlchemy : {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")

# ------------------------------
# Route: Films aléatoires par genre et plateforme
# ------------------------------
@app.get("/random_movies/", dependencies=[Depends(verify_credentials)])
async def get_random_movies(genre: str, platforms: List[str] = Query(...), limit: int = 10):
    try:
        selected_platforms = [p.lower() for p in platforms if p.lower() in PLATFORM_TABLES]
        if not selected_platforms:
            raise HTTPException(status_code=400, detail="Aucune plateforme valide sélectionnée.")

        movies = []
        with engine.connect() as conn:
            for platform in selected_platforms:
                query = f"""
                SELECT m.title, m.synopsis, m.poster_url, m.genres, '{platform}' AS platform
                FROM movies m
                JOIN {platform} p ON m.title = p.title
                WHERE FIND_IN_SET(:genre, m.genres)
                """
                result = conn.execute(text(query), {"genre": genre}).fetchall()
                for row in result:
                    movies.append({
                        "title": row[0],
                        "synopsis": row[1],
                        "poster_url": row[2],
                        "genres": row[3],
                        "platform": row[4]
                    })

        if not movies:
            raise HTTPException(status_code=404, detail="Aucun film trouvé pour ce genre et ces plateformes.")

        return random.sample(movies, min(limit, len(movies)))

    except HTTPException:  # ⚡ protégé (400 & 404)
        raise
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Erreur SQLAlchemy : {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")


# ------------------------------
# Route: Téléchargement CSV
# ------------------------------
@app.get("/download-movie-details/", dependencies=[Depends(verify_credentials)])
async def download_movie_details():
    try:
        query = f"""
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

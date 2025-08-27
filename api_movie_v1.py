from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from typing import List
import random
from fastapi.responses import StreamingResponse
import csv
import io
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()

# Chargement du système de recommandation
try:
    reco_vectorizer = joblib.load("model/reco_vectorizer.joblib")
    tfidf_matrix = joblib.load("model/tfidf_matrix.joblib")
    movie_index_df = pd.read_csv("model/movie_index.csv")
except Exception as e:
    print(f"⚠️ Erreur lors du chargement du système de recommandation : {e}")
    reco_vectorizer, tfidf_matrix, movie_index_df = None, None, None


# Fonction pour afficher un score interprété
def interpret_score(score: float) -> str:
    pct = int(score * 100)
    if score >= 0.85:
        return f"🎯 Très forte recommandation ({pct}%)"
    elif score >= 0.7: return f"👍 Forte similarité ({pct}%)"
    elif score >= 0.5:
        return f"🤔 Moyennement similaire ({pct}%)"
    else:
        return f"⚠️ Faible similarité ({pct}%)"


# Configuration de la base de données
DATABASE_URL = "mysql+pymysql://root:MW080922@127.0.0.1/movies"
engine = create_engine(DATABASE_URL)

# Liste centralisée des plateformes
PLATFORM_TABLES = {
    "netflix": "Netflix",
    "prime": "Prime",
    "hulu": "Hulu",
    "hbo": "HBO Max",
    "apple": "Apple"
}

# Définir des identifiants statiques pour l'authentification
USERNAME = "user"
PASSWORD = "password"

# Configuration de HTTPBasic pour l'authentification
security = HTTPBasic()

# Fonction de vérification des identifiants
def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != USERNAME or credentials.password != PASSWORD:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

@app.get("/download-movie-details/", dependencies=[Depends(verify_credentials)])
async def download_movie_details():
    """
    Télécharge un fichier CSV avec les détails des films et les plateformes disponibles.
    """
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
    FROM 
        movies m
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

    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="Aucune donnée à télécharger.")

        # Crée un fichier CSV en mémoire
        csv_file = io.StringIO()
        csv_writer = csv.writer(csv_file)
        # Ajouter les en-têtes
        csv_writer.writerow([col for col in result.keys()])
        # Ajouter les données
        csv_writer.writerows(rows)

        # Revenir au début du fichier en mémoire
        csv_file.seek(0)

        # Retourner une réponse téléchargeable
        response = StreamingResponse(
            csv_file,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=movie_details.csv"}
        )
        return response

    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Erreur SQLAlchemy : {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")

        
@app.get("/genres/", dependencies=[Depends(verify_credentials)])
async def get_unique_genres():
    """
    Récupère tous les genres uniques de la base de données.
    """
    try:
        query = "SELECT genres FROM movies"
        all_genres = set()

        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchall()
            for row in result:
                genres = row[0].split(",")
                genres = [genre.strip() for genre in genres]
                all_genres.update(genres)

        return sorted(all_genres)

    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Erreur SQLAlchemy : {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")


@app.get("/movie-details/{title}", dependencies=[Depends(verify_credentials)])
async def get_movie_details(title: str):
    """
    Récupère les détails d'un film par son titre.
    """
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
        FROM 
            movies m
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
        WHERE m.title = :title
        ORDER BY m.title
        """

        with engine.connect() as conn:
            result = conn.execute(text(query), {"title": title}).fetchall()

        if not result:
            raise HTTPException(status_code=404, detail="Film non trouvé dans la base de données ou sur les plateformes.")

        # Structurer les résultats
        movie_info = {
            "movie_id": result[0][0],
            "title": result[0][1],
            "genres": result[0][2],
            "releaseYear": result[0][3],
            "rating": result[0][4],
            "synopsis": result[0][5],
            "poster_url": result[0][6],
            "platforms": list(set(row[7] for row in result if row[7]))  # Éliminer les doublons
        }

        return movie_info

    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Erreur SQLAlchemy : {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")


@app.get("/random_movies/", dependencies=[Depends(verify_credentials)])
async def get_random_movies(genre: str, platforms: List[str] = Query(...), limit: int = 10):
    """
    Récupère des films aléatoires filtrés par genre et plateformes sélectionnées.
    """
    try:
        selected_platforms = [platform for platform in platforms if platform in PLATFORM_TABLES]
        if not selected_platforms:
            raise HTTPException(status_code=400, detail="Aucune plateforme valide sélectionnée.")

        movies = []

        with engine.connect() as conn:
            for platform in selected_platforms:
                query = f"""
                SELECT movies.title, movies.synopsis, movies.poster_url, movies.genres, '{platform}' AS platform
                FROM movies
                JOIN {platform} ON movies.title = {platform}.title
                WHERE FIND_IN_SET(:genre, {platform}.genres)
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

        random_movies = random.sample(movies, min(limit, len(movies)))
        return random_movies

    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Erreur SQLAlchemy : {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")
    


@app.get("/recommend/{title}", dependencies=[Depends(verify_credentials)])
async def recommend_movies(title: str, platforms: List[str] = Query(default=[]), top_k: int = 5):
    """
    Recommande des films similaires à un film donné, avec filtrage facultatif par plateformes.
    """
    if not reco_vectorizer or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="Système de recommandation non initialisé.")

    title = title.strip().lower()
    matches = movie_index_df[movie_index_df["title"].str.lower() == title]

    if matches.empty:
        raise HTTPException(status_code=404, detail="Film non trouvé dans la base de recommandations.")

    idx = matches.index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    movie_index_df["similarity"] = cosine_sim
    filtered_df = movie_index_df.drop(index=idx)

    # 🎛️ Filtrage par plateformes
    if platforms:
        valid_platforms = [p.lower() for p in platforms if p.lower() in PLATFORM_TABLES]
        if not valid_platforms:
            raise HTTPException(status_code=400, detail="Aucune plateforme valide sélectionnée.")
        
        try:
            union_query = " UNION ".join([f"SELECT title FROM {p}" for p in valid_platforms])
            with engine.connect() as conn:
                result = conn.execute(text(union_query))
                allowed_titles = set(row[0].lower() for row in result.fetchall())
            filtered_df = filtered_df[filtered_df["title"].str.lower().isin(allowed_titles)]
        except SQLAlchemyError as e:
            raise HTTPException(status_code=500, detail=f"Erreur lors du filtrage plateforme : {str(e)}")

    top_recos = filtered_df.sort_values(by="similarity", ascending=False).head(top_k)

    return [
        {
            "title": row["title"],
            "similarity_score": round(row["similarity"], 4),
            "match_confidence": interpret_score(row["similarity"])
        }
        for _, row in top_recos.iterrows()
    ]
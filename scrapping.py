import requests
import pandas as pd
from sqlalchemy import create_engine, Integer, String, Text, Float
from time import sleep
from tqdm import tqdm

# Configuration de l'API
api_key = '6450515c5357c7e0d49ac4972810e9f4'
base_url = "https://api.themoviedb.org/3/discover/movie"
genre_url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={api_key}"

def fetch_genres() -> dict[int, str]:
    """Récupère la liste des genres avec leur ID."""
    response = requests.get(genre_url)
    if response.status_code == 200:
        return {genre['id']: genre['name'] for genre in response.json().get('genres', [])}
    print(f"Erreur lors de la récupération des genres : {response.status_code}")
    return {}

def fetch_keywords(movie_id: int, session: requests.Session) -> str | None:
    """Récupère les mots-clés pour un film donné."""
    keyword_url = f"https://api.themoviedb.org/3/movie/{movie_id}/keywords?api_key={api_key}"
    response = session.get(keyword_url)
    if response.status_code == 200:
        keywords = response.json().get('keywords', [])
        return ', '.join([kw['name'] for kw in keywords])
    return None

def fetch_movies_by_genre(genre_id: int, genre_name: str) -> list[dict]:
    """Scrape les films pour un genre spécifique avec une barre de progression."""
    all_movies = []
    page, total_pages = 1, 1

    with requests.Session() as session:
        # Première requête pour connaître total_pages
        url = f"{base_url}?api_key={api_key}&page=1&with_genres={genre_id}"
        response = session.get(url)

        if response.status_code == 200:
            data = response.json()
            total_pages = min(data.get('total_pages', 1), 500)

            # Barre de progression sur les pages
            with tqdm(total=total_pages, desc=f"🎬 {genre_name}", unit="page", leave=True) as pbar:
                while page <= total_pages:
                    url = f"{base_url}?api_key={api_key}&page={page}&with_genres={genre_id}"
                    response = session.get(url)

                    if response.status_code == 200:
                        data = response.json()

                        for movie in data['results']:
                            release_date = movie.get('release_date')
                            release_year = pd.to_datetime(release_date).year if release_date else None
                            poster_path = movie.get('poster_path')

                            keywords = fetch_keywords(movie['id'], session)

                            all_movies.append({
                                'movie_id': movie['id'],
                                'title': movie['title'],
                                'release_year': release_year,
                                'genres': genre_name,
                                'synopsis': movie.get('overview', 'No summary available'),
                                'rating': round(movie.get('vote_average', 0), 1),
                                'vote_count': movie.get('vote_count', 0),
                                'original_language': movie.get('original_language', 'Unknown'),
                                'poster_url': f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None,
                                'key_words': keywords
                            })

                        page += 1
                        pbar.update(1)
                        sleep(0.5)
                    else:
                        print(f"⚠️ Erreur pour {genre_name}, page {page}: {response.status_code}")
                        break
        else:
            print(f"❌ Impossible de charger les pages pour {genre_name} (code {response.status_code})")

    return all_movies

def scrape_all_genres() -> pd.DataFrame:
    """Scrape les films pour tous les genres disponibles."""
    genre_mapping = fetch_genres()
    all_movies = []

    for genre_id, genre_name in genre_mapping.items():
        movies = fetch_movies_by_genre(genre_id, genre_name)
        all_movies.extend(movies)

    df_movies = pd.DataFrame(all_movies).drop_duplicates(subset=['movie_id'])
    df_movies.to_csv("movies_data.csv", index=False)
    print("\n✅ Scraping terminé. Données sauvegardées dans 'movies_data.csv'.")
    return df_movies

def save_to_sql(df: pd.DataFrame, table_name: str, engine):
    """Sauvegarde le DataFrame dans une base de données SQL."""
    df.to_sql(table_name, con=engine, if_exists='replace', index=False, dtype={
        'movie_id': Integer(),
        'release_year': Integer(),
        'title': String(255),
        'genres': String(255),
        'synopsis': Text,
        'rating': Float,
        'original_language': String(10),
        'poster_url': String(255),
        'key_words': Text
    })
    print(f"📥 Table `{table_name}` mise à jour avec succès dans la base de données.")

# Exécution principale
if __name__ == "__main__":
    DATABASE_URL = "mysql+pymysql://louve:%40Marley080922@mysql-louve.alwaysdata.net/louve_movies"
    engine = create_engine(DATABASE_URL)

    try:
        df_movies = scrape_all_genres()
        if not df_movies.empty:
            save_to_sql(df_movies, 'movies', engine)
            print("🎉 Données insérées dans la base SQL avec succès.")
        else:
            print("⚠️ Aucune donnée à insérer.")
    except KeyboardInterrupt:
        print("\n⛔️ Script interrompu manuellement.")

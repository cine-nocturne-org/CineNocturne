# E1_BDD/scrapping_movies_fr_with_providers.py
import os
import time
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy import Integer, String, Text, Float
from time import sleep
from tqdm import tqdm
from dotenv import load_dotenv

# =========================
# Config
# =========================
load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY", "")                 # clé TMDB (via secrets/env)
COUNTRY = os.getenv("TMDB_COUNTRY", "FR")               # pays pour watch/providers
LANG    = os.getenv("TMDB_LANGUAGE", "fr-FR")           # langue de scraping
DATABASE_URL = os.getenv("MYSQL_URL",                   # URL MySQL (via secrets/env)
                         "mysql+pymysql://louve:%40Marley080922@mysql-louve.alwaysdata.net/louve_movies")

# Si = "1", on re-calcule aussi les providers pour les films déjà présents
FETCH_PROVIDERS_FOR_EXISTING = os.getenv("FETCH_PROVIDERS_FOR_EXISTING", "0") == "1"

# Pause entre pages (pour ménager l'API)
REQUEST_SLEEP = float(os.getenv("REQUEST_SLEEP", "0.5"))

BASE_URL      = "https://api.themoviedb.org/3"
DISCOVER_URL  = f"{BASE_URL}/discover/movie"
GENRE_URL     = f"{BASE_URL}/genre/movie/list"
KEYWORDS_URL  = f"{BASE_URL}/movie/{{movie_id}}/keywords"
PROVIDERS_URL = f"{BASE_URL}/movie/{{movie_id}}/watch/providers"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# =========================
# DDL non destructif (préserve user_rating + index)
# =========================
DDL_MOVIES = """
CREATE TABLE IF NOT EXISTS movies (
  movie_id INT PRIMARY KEY,
  title VARCHAR(255),
  original_title VARCHAR(255),
  release_year INT,
  genres VARCHAR(255),
  synopsis TEXT,
  rating FLOAT,
  vote_count INT,
  original_language VARCHAR(10),
  poster_url VARCHAR(255),
  key_words TEXT,
  user_rating FLOAT,
  platforms_flatrate VARCHAR(1024),
  platforms_rent VARCHAR(1024),
  platforms_buy VARCHAR(1024),
  platform_link VARCHAR(512)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

DDL_FULLTEXT = """
CREATE FULLTEXT INDEX idx_movies_title_fulltext ON movies(title);
"""

def ensure_schema(engine):
    with engine.begin() as conn:
        # 1) créer la table si absente
        conn.execute(text(DDL_MOVIES))

        # 2) migrer colonnes manquantes (ajout sans casser l'existant)
        required_cols = {
            "original_title": "VARCHAR(255)",
            "vote_count": "INT",
            "key_words": "TEXT",
            "user_rating": "FLOAT",
            "platforms_flatrate": "VARCHAR(1024)",
            "platforms_rent": "VARCHAR(1024)",
            "platforms_buy": "VARCHAR(1024)",
            "platform_link": "VARCHAR(512)",
        }

        # récupérer les colonnes existantes
        existing = conn.execute(text("""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'movies'
        """)).fetchall()
        existing_cols = {row[0] for row in existing}

        # ajouter les colonnes manquantes
        for col, ddl in required_cols.items():
            if col not in existing_cols:
                conn.execute(text(f"ALTER TABLE movies ADD COLUMN {col} {ddl}"))

        # 3) FULLTEXT si absent
        has_idx = conn.execute(text("""
            SELECT COUNT(1) FROM INFORMATION_SCHEMA.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'movies'
              AND INDEX_NAME = 'idx_movies_title_fulltext'
        """)).scalar()
        if not has_idx:
            try:
                conn.execute(text(DDL_FULLTEXT))
            except Exception:
                # non bloquant (selon version/permissions)
                pass

# =========================
# Utils
# =========================
def _get_with_retry(session_or_module, url, params=None, tries=3, wait=1.0):
    """GET avec retry simple (ex: 429 rate-limit ou coupures)"""
    last = None
    for i in range(tries):
        try:
            r = session_or_module.get(url, params=params, timeout=30)
            if r.status_code != 429:
                return r
            # 429 : rate limit
            delay = wait * (2 ** i)
            print(f"⚠️ Rate limit TMDB (429), retry dans {delay:.1f}s...")
            time.sleep(delay)
            last = r
        except requests.RequestException as e:
            delay = wait * (2 ** i)
            print(f"⚠️ Erreur réseau {e.__class__.__name__}: {e}. Retry dans {delay:.1f}s...")
            time.sleep(delay)
    return last

def safe_year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        return pd.to_datetime(date_str, errors="coerce").year
    except Exception:
        return None

def join_unique(items):
    dedup, seen = [], set()
    for x in items:
        x = (x or "").strip()
        if x and x not in seen:
            seen.add(x)
            dedup.append(x)
    return ", ".join(dedup) if dedup else None

# =========================
# TMDB: genres, keywords, providers
# =========================
def fetch_genres() -> dict[int, str]:
    r = _get_with_retry(requests, GENRE_URL, params={"api_key": API_KEY, "language": LANG})
    if r and r.status_code == 200:
        return {g['id']: g['name'] for g in r.json().get('genres', [])}
    code = r.status_code if r else "?"
    body = r.text[:200] if r and hasattr(r, "text") else ""
    print(f"Erreur genres : {code} -> {body}")
    return {}

def fetch_keywords(movie_id: int, session: requests.Session) -> str | None:
    r = _get_with_retry(session, KEYWORDS_URL.format(movie_id=movie_id), params={"api_key": API_KEY})
    if r and r.status_code == 200:
        keywords = r.json().get('keywords', [])
        return join_unique([kw.get('name') for kw in keywords])
    return None

def fetch_providers(movie_id: int, country: str = COUNTRY, session: requests.Session | None = None):
    own_session = False
    if session is None:
        session = requests.Session()
        own_session = True
    try:
        r = _get_with_retry(session, PROVIDERS_URL.format(movie_id=movie_id), params={"api_key": API_KEY})
        if not (r and r.status_code == 200):
            return {"flatrate": [], "rent": [], "buy": [], "link": None}
        data = r.json().get("results", {})
        region = data.get(country, {})
        return {
            "flatrate": [p.get("provider_name") for p in (region.get("flatrate") or [])],
            "rent":     [p.get("provider_name") for p in (region.get("rent") or [])],
            "buy":      [p.get("provider_name") for p in (region.get("buy") or [])],
            "link":     region.get("link")
        }
    finally:
        if own_session:
            session.close()

# =========================
# Scraping
# =========================
def fetch_movies_by_genre(genre_id: int, genre_name: str) -> list[dict]:
    all_movies = []
    page, total_pages = 1, 1
    with requests.Session() as session:
        # Première requête pour connaître total_pages
        params = {"api_key": API_KEY, "page": 1, "with_genres": genre_id, "language": LANG}
        r = _get_with_retry(session, DISCOVER_URL, params=params)
        if not (r and r.status_code == 200):
            body = r.text[:200] if r and hasattr(r, "text") else ""
            print(f"❌ discover {genre_name} (p1) : {getattr(r, 'status_code', '?')} -> {body}")
            return all_movies

        data = r.json()
        total_pages = min(data.get('total_pages', 1), 500)

        with tqdm(total=total_pages, desc=f"🎬 {genre_name}", unit="page", dynamic_ncols=True, leave=False) as pbar:
            while page <= total_pages:
                params["page"] = page
                r = _get_with_retry(session, DISCOVER_URL, params=params)
                if not (r and r.status_code == 200):
                    print(f"⚠️ discover {genre_name}, page {page}: {getattr(r, 'status_code', '?')}")
                    break

                data = r.json()
                for movie in data.get('results', []):
                    mid = movie.get('id')
                    release_year = safe_year(movie.get('release_date'))
                    poster_path = movie.get('poster_path')
                    keywords = fetch_keywords(mid, session) if mid else None

                    all_movies.append({
                        'movie_id': mid,
                        'title': movie.get('title'),                     # FR si dispo
                        'original_title': movie.get('original_title'),   # VO
                        'release_year': release_year,
                        'genres': genre_name,  # fusion plus tard
                        'synopsis': movie.get('overview') or 'Résumé non disponible',
                        'rating': round((movie.get('vote_average') or 0.0), 1),
                        'vote_count': movie.get('vote_count') or 0,
                        'original_language': movie.get('original_language') or 'Unknown',
                        'poster_url': f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None,
                        'key_words': keywords
                    })
                page += 1
                pbar.update(1)
                sleep(REQUEST_SLEEP)
    return all_movies

def scrape_all_genres() -> pd.DataFrame:
    genre_mapping = fetch_genres()
    all_rows = []
    for gid, gname in genre_mapping.items():
        all_rows.extend(fetch_movies_by_genre(gid, gname))

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    # 🔁 Fusionner les genres par movie_id (union unique, séparateur '|')
    def _merge_genres(s: pd.Series) -> str | None:
        vals = set()
        for g in s.dropna().astype(str):
            for tok in g.replace(",", "|").split("|"):
                t = tok.strip()
                if t:
                    vals.add(t)
        return "|".join(sorted(vals)) if vals else None

    if "genres" not in df.columns:
        df["genres"] = None

    agg = {
        "title": "first",
        "original_title": "first",
        "release_year": "first",
        "genres": _merge_genres,
        "synopsis": lambda s: next((x for x in s if x and x != "Résumé non disponible"),
                                   "Résumé non disponible"),
        "rating": "max",
        "vote_count": "max",
        "original_language": "first",
        "poster_url": lambda s: next((x for x in s if x), None),
        "key_words": lambda s: join_unique([w for x in s.dropna().astype(str) for w in x.split(",")]),
    }

    df_merged = df.groupby("movie_id", as_index=False).agg(agg)
    return df_merged

# =========================
# Sauvegarde NON destructrice + providers
# =========================
def get_existing_movie_ids(engine) -> set[int]:
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT movie_id FROM movies")).fetchall()
        return set(int(r[0]) for r in rows) if rows else set()
    except Exception:
        return set()

def enrich_with_providers(df: pd.DataFrame, only_new: bool = True) -> pd.DataFrame:
    if df.empty:
        return df

    existing_ids = get_existing_movie_ids(engine) if only_new else set()
    ids_to_fetch = df['movie_id'].tolist() if not only_new else [mid for mid in df['movie_id'] if mid not in existing_ids]

    # colonnes cibles
    for col in ['platforms_flatrate', 'platforms_rent', 'platforms_buy', 'platform_link']:
        if col not in df.columns:
            df[col] = None

    if not ids_to_fetch:
        return df

    with requests.Session() as session:
        for mid in tqdm(ids_to_fetch, desc=f"🔎 Providers {COUNTRY}", unit="movie", leave=True):
            prov = fetch_providers(mid, COUNTRY, session=session)
            df.loc[df['movie_id'] == mid, 'platforms_flatrate'] = join_unique(prov.get("flatrate", []))
            df.loc[df['movie_id'] == mid, 'platforms_rent']     = join_unique(prov.get("rent", []))
            df.loc[df['movie_id'] == mid, 'platforms_buy']      = join_unique(prov.get("buy", []))
            df.loc[df['movie_id'] == mid, 'platform_link']      = prov.get("link")
            time.sleep(0.05)  # ménage l'API

    return df

def upsert_movies(df: pd.DataFrame, engine):
    """
    - charge df dans une table staging (remplaçable),
    - UPSERT dans movies sans toucher user_rating,
    - ne détruit ni colonnes ni index.
    """
    ensure_schema(engine)

    staging = 'movies_staging'
    df.to_sql(
        staging, con=engine, if_exists='replace', index=False, dtype={
            'movie_id': Integer(),
            'title': String(255),
            'original_title': String(255),
            'release_year': Integer(),
            'genres': String(255),
            'synopsis': Text(),
            'rating': Float(),
            'vote_count': Integer(),
            'original_language': String(10),
            'poster_url': String(255),
            'key_words': Text(),
            'platforms_flatrate': String(1024),
            'platforms_rent': String(1024),
            'platforms_buy': String(1024),
            'platform_link': String(512),
        }
    )

    upsert_sql = """
    INSERT INTO movies (
        movie_id, title, original_title, release_year, genres, synopsis, rating,
        vote_count, original_language, poster_url, key_words,
        platforms_flatrate, platforms_rent, platforms_buy, platform_link
    )
    SELECT
        s.movie_id, s.title, s.original_title, s.release_year, s.genres, s.synopsis, s.rating,
        s.vote_count, s.original_language, s.poster_url, s.key_words,
        s.platforms_flatrate, s.platforms_rent, s.platforms_buy, s.platform_link
    FROM movies_staging s
    ON DUPLICATE KEY UPDATE
        title = IFNULL(VALUES(title), movies.title),
        original_title = IFNULL(VALUES(original_title), movies.original_title),
        release_year = IFNULL(VALUES(release_year), movies.release_year),
        genres = IFNULL(VALUES(genres), movies.genres),
        synopsis = IFNULL(NULLIF(VALUES(synopsis), ''), movies.synopsis),
        rating = IFNULL(VALUES(rating), movies.rating),
        vote_count = IFNULL(VALUES(vote_count), movies.vote_count),
        original_language = IFNULL(VALUES(original_language), movies.original_language),
        poster_url = IFNULL(NULLIF(VALUES(poster_url), ''), movies.poster_url),
        key_words = IFNULL(NULLIF(VALUES(key_words), ''), movies.key_words),
        platforms_flatrate = IFNULL(NULLIF(VALUES(platforms_flatrate), ''), movies.platforms_flatrate),
        platforms_rent = IFNULL(NULLIF(VALUES(platforms_rent), ''), movies.platforms_rent),
        platforms_buy = IFNULL(NULLIF(VALUES(platforms_buy), ''), movies.platforms_buy),
        platform_link = IFNULL(NULLIF(VALUES(platform_link), ''), movies.platform_link);
    """
    with engine.begin() as conn:
        conn.execute(text(upsert_sql))
        conn.execute(text(f"DROP TABLE {staging};"))

    ensure_schema(engine)
    print("✅ UPSERT terminé : `movies` complétée, `user_rating` et index conservés.")

# =========================
# Main
# =========================
if __name__ == "__main__":
    if not API_KEY.strip():
        raise SystemExit("⚠️ Configure TMDB_API_KEY (env/.env).")

    ensure_schema(engine)

    print(f"📥 Scraping TMDB (lang={LANG}) ...")
    df_movies = scrape_all_genres()

    if df_movies.empty:
        print("⚠️ Aucune donnée à insérer.")
    else:
        print(f"🔗 Providers pour pays={COUNTRY} (only_new={not FETCH_PROVIDERS_FOR_EXISTING}) ...")
        df_movies = enrich_with_providers(df_movies, only_new=(not FETCH_PROVIDERS_FOR_EXISTING))

        print("💾 UPSERT vers MySQL ...")
        upsert_movies(df_movies, engine)

        with engine.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM movies")).scalar()
        print(f"🎉 Terminé. Total films en base: {total}")

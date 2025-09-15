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
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrapping TMDB movies")
    parser.add_argument(
        "--genre",
        type=int,
        help="ID du genre à scrapper (si absent, scrappe tous les genres)"
    )
    return parser.parse_args()

# =========================
# Config
# =========================
load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY", "")
COUNTRY = os.getenv("TMDB_COUNTRY", "FR")
LANG    = os.getenv("TMDB_LANGUAGE", "fr-FR")
DATABASE_URL = os.getenv("MYSQL_URL")

REQUEST_SLEEP = float(os.getenv("REQUEST_SLEEP", "0.5"))

BASE_URL         = "https://api.themoviedb.org/3"
DISCOVER_URL     = f"{BASE_URL}/discover/movie"
GENRE_URL        = f"{BASE_URL}/genre/movie/list"
KEYWORDS_URL     = f"{BASE_URL}/movie/{{movie_id}}/keywords"
DETAILS_URL      = f"{BASE_URL}/movie/{{movie_id}}"
IMAGES_URL       = f"{BASE_URL}/movie/{{movie_id}}/images"
TRANSLATIONS_URL = f"{BASE_URL}/movie/{{movie_id}}/translations"
PROVIDERS_URL    = f"{BASE_URL}/movie/{{movie_id}}/watch/providers"

TRANSLATE_IF_MISSING = os.getenv("TRANSLATE_IF_MISSING", "0").lower() in {"1", "true", "yes", "on"}

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# =========================
# Utils
# =========================
def _get_with_retry(session_or_module, url, params=None, tries=3, wait=1.0):
    last = None
    for i in range(tries):
        try:
            r = session_or_module.get(url, params=params, timeout=30)
            if r.status_code != 429:
                return r
            time.sleep(wait * (2 ** i))
            last = r
        except requests.RequestException:
            time.sleep(wait * (2 ** i))
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
    for x in (items or []):
        x = (x or "").strip()
        if x and x not in seen:
            seen.add(x)
            dedup.append(x)
    return ", ".join(dedup) if dedup else None

# =========================
# TMDB helpers
# =========================
def fetch_genres() -> dict[int, str]:
    r = _get_with_retry(requests, GENRE_URL, params={"api_key": API_KEY, "language": LANG})
    if r and r.status_code == 200:
        return {g['id']: g['name'] for g in r.json().get('genres', [])}
    return {}

def fetch_keywords(movie_id: int, session: requests.Session) -> str | None:
    r = _get_with_retry(session, KEYWORDS_URL.format(movie_id=movie_id), params={"api_key": API_KEY})
    if r and r.status_code == 200:
        return join_unique([kw.get('name') for kw in r.json().get('keywords', [])])
    return None

def fetch_details_lang(movie_id: int, lang: str, session: requests.Session) -> dict:
    r = _get_with_retry(session, DETAILS_URL.format(movie_id=movie_id),
                        params={"api_key": API_KEY, "language": lang})
    if r and r.status_code == 200:
        return r.json() or {}
    return {}

def fetch_overview_lang(movie_id: int, lang: str, session: requests.Session) -> str | None:
    dt = fetch_details_lang(movie_id, lang, session)
    ov = (dt.get("overview") or "").strip()
    return ov or None

def fetch_overview_any_language(movie_id: int, session: requests.Session, prefer: list[str] | None = None) -> tuple[str | None, str | None]:
    r = _get_with_retry(session, TRANSLATIONS_URL.format(movie_id=movie_id), params={"api_key": API_KEY})
    if not (r and r.status_code == 200):
        return None, None
    translations = (r.json() or {}).get("translations") or []
    by_lang = {}
    for t in translations:
        lang_code = (t.get("iso_639_1") or "").strip()
        data = t.get("data") or {}
        ov = (data.get("overview") or "").strip()
        if lang_code and ov:
            by_lang[lang_code] = ov
    if not by_lang:
        return None, None
    if prefer:
        for lc in prefer:
            if lc in by_lang:
                return by_lang[lc], lc
    lang_code, ov = next(iter(by_lang.items()))
    return ov, lang_code

def translate_to_fr_if_needed(text: str, src_lang_code: str | None) -> str:
    if not text:
        return text
    if src_lang_code == "fr":
        return text
    if not TRANSLATE_IF_MISSING:
        return text
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target="fr").translate(text)
    except Exception:
        return text

def select_best_poster(posters: list[dict]) -> str | None:
    if not posters:
        return None
    posters_sorted = sorted(
        posters,
        key=lambda p: (
            int(p.get("vote_count") or 0),
            float(p.get("vote_average") or 0.0),
            int(p.get("width") or 0)
        ),
        reverse=True
    )
    return posters_sorted[0].get("file_path")

def fetch_any_poster_url(movie_id: int, session: requests.Session) -> str | None:
    r = _get_with_retry(session, IMAGES_URL.format(movie_id=movie_id),
                        params={"api_key": API_KEY, "include_image_language": "fr,en,null"})
    if not (r and r.status_code == 200):
        return None
    posters = (r.json() or {}).get("posters") or []
    file_path = select_best_poster(posters)
    return f"https://image.tmdb.org/t/p/w500{file_path}" if file_path else None

# =========================
# Scraping
# =========================
def fetch_movies_by_genre(genre_id: int, genre_name: str) -> list[dict]:
    all_movies = []
    page, total_pages = 1, 1
    with requests.Session() as session:
        params = {
            "api_key": API_KEY,
            "page": 1,
            "with_genres": genre_id,
            "language": LANG,
            "watch_region": COUNTRY,
        }
        r = _get_with_retry(session, DISCOVER_URL, params=params)
        if not (r and r.status_code == 200):
            return all_movies
        data = r.json()
        total_pages = min(data.get('total_pages', 1), 500)
        with tqdm(total=total_pages, desc=f"🎬 {genre_name}", unit="page", leave=False) as pbar:
            while page <= total_pages:
                params["page"] = page
                r = _get_with_retry(session, DISCOVER_URL, params=params)
                if not (r and r.status_code == 200):
                    break
                data = r.json()
                for movie in data.get('results', []):
                    mid = movie.get('id')
                    release_year = safe_year(movie.get('release_date'))
                    poster_path = movie.get('poster_path')
                    keywords = fetch_keywords(mid, session) if mid else None
                    title_fr = movie.get('title')
                    orig_title = (movie.get('original_title') or "").strip()

                    # ---- Résumé ----
                    synopsis_final = (movie.get('overview') or '').strip()
                    src_lang = "fr" if synopsis_final else None
                    if not synopsis_final and mid:
                        ov_en = fetch_overview_lang(mid, "en-US", session)
                        if ov_en:
                            synopsis_final, src_lang = ov_en, "en"
                    if not synopsis_final and mid:
                        preferred = ["es","de","it","pt","ru","ja","zh","ko","ar","tr","nl","sv","pl","no","da","fi"]
                        ov_any, lang_code = fetch_overview_any_language(mid, session, prefer=preferred)
                        if ov_any:
                            synopsis_final, src_lang = ov_any, lang_code
                    synopsis_fr = translate_to_fr_if_needed(synopsis_final, src_lang) if synopsis_final else "Résumé non disponible"

                    # ---- Poster ----
                    if poster_path:
                        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                    else:
                        poster_url = fetch_any_poster_url(mid, session)

                    all_movies.append({
                        'movie_id': mid,
                        'title': title_fr,
                        'original_title': orig_title,
                        'release_year': release_year,
                        'genres': genre_name,
                        'synopsis': synopsis_fr,
                        'rating': round((movie.get('vote_average') or 0.0), 1),
                        'vote_count': movie.get('vote_count') or 0,
                        'original_language': movie.get('original_language') or 'Unknown',
                        'poster_url': poster_url,
                        'key_words': keywords
                    })
                page += 1
                pbar.update(1)
                sleep(REQUEST_SLEEP)
    return all_movies

def scrape_all_genres(genre_ids: list[int] | None = None) -> pd.DataFrame:
    genre_mapping = fetch_genres()
    if genre_ids is None:
        genre_ids = list(genre_mapping.keys())
    all_rows = []
    for gid in genre_ids:
        gname = genre_mapping.get(gid, f"Unknown-{gid}")
        all_rows.extend(fetch_movies_by_genre(gid, gname))
    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    def _merge_genres(s: pd.Series) -> str | None:
        vals = set()
        for g in s.dropna().astype(str):
            for tok in g.replace(",", "|").split("|"):
                t = tok.strip()
                if t:
                    vals.add(t)
        return "|".join(sorted(vals)) if vals else None
    agg = {
        "title": "first",
        "original_title": "first",
        "release_year": "first",
        "genres": _merge_genres,
        "synopsis": lambda s: next((x for x in s if x and x != "Résumé non disponible"), "Résumé non disponible"),
        "rating": "max",
        "vote_count": "max",
        "original_language": "first",
        "poster_url": lambda s: next((x for x in s if x), None),
        "key_words": lambda s: join_unique([w for x in s.dropna().astype(str) for w in x.split(",")]),
    }
    return df.groupby("movie_id", as_index=False).agg(agg)

# =========================
# Providers
# =========================
def fetch_providers(movie_id: int, country: str = COUNTRY, session: requests.Session | None = None):
    own_session = False
    if session is None:
        session = requests.Session()
        own_session = True
    try:
        r = _get_with_retry(session, PROVIDERS_URL.format(movie_id=movie_id), params={"api_key": API_KEY})
        if not (r and r.status_code == 200):
            return {"flatrate": [], "rent": [], "buy": [], "link": None}
        data = r.json().get("results", {}) or {}
        region = data.get(country, {}) or {}
        return {
            "flatrate": [p.get("provider_name") for p in (region.get("flatrate") or [])],
            "rent":     [p.get("provider_name") for p in (region.get("rent") or [])],
            "buy":      [p.get("provider_name") for p in (region.get("buy") or [])],
            "link":     region.get("link")
        }
    finally:
        if own_session:
            session.close()

def enrich_with_providers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ['platforms_flatrate', 'platforms_rent', 'platforms_buy', 'platform_link']:
        if col not in df.columns:
            df[col] = None
    ids_in_batch = set(df['movie_id'].dropna().astype(int).tolist())
    print(f"📺 Fetch providers pour {len(ids_in_batch)} films (région {COUNTRY})...")
    with requests.Session() as session:
        for mid in tqdm(ids_in_batch, desc=f"🔎 Providers {COUNTRY}", unit="movie", leave=True):
            prov = fetch_providers(mid, COUNTRY, session=session)
            df.loc[df['movie_id'] == mid, 'platforms_flatrate'] = join_unique(prov.get("flatrate"))
            df.loc[df['movie_id'] == mid, 'platforms_rent'] = join_unique(prov.get("rent"))
            df.loc[df['movie_id'] == mid, 'platforms_buy'] = join_unique(prov.get("buy"))
            df.loc[df['movie_id'] == mid, 'platform_link'] = prov.get("link")
            time.sleep(0.05)
    return df

# =========================
# UPSERT
# =========================
def upsert_movies(df: pd.DataFrame, engine):
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
    print("✅ UPSERT terminé.")

# =========================
# Main
# =========================
if __name__ == "__main__":
    args = parse_args()
    if not API_KEY.strip():
        raise SystemExit("⚠️ Configure TMDB_API_KEY (env/.env).")
    if args.genre:
        print(f"📥 Scraping uniquement le genre {args.genre} (lang={LANG}) ...")
        df_movies = scrape_all_genres([args.genre])
    else:
        print(f"📥 Scraping TOUS les genres (lang={LANG}) ...")
        df_movies = scrape_all_genres()
    if df_movies.empty:
        print("⚠️ Aucune donnée à insérer.")
    else:
        print(f"🔗 Providers pour pays={COUNTRY} ...")
        df_movies = enrich_with_providers(df_movies)
        print("💾 UPSERT vers MySQL ...")
        upsert_movies(df_movies, engine)
        with engine.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM movies")).scalar()
            missing_poster = conn.execute(text(
                "SELECT COUNT(*) FROM movies WHERE poster_url IS NULL OR TRIM(poster_url)=''"
            )).scalar()
            missing_synopsis = conn.execute(text(
                "SELECT COUNT(*) FROM movies WHERE synopsis IS NULL OR TRIM(synopsis)='' OR synopsis='Résumé non disponible'"
            )).scalar()
        print(f"🎉 Terminé. Total films en base: {total}")
        print(f"🖼️ Posters manquants: {missing_poster} | 📝 Résumés manquants: {missing_synopsis}")

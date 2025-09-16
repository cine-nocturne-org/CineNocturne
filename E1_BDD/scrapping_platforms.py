# E1_BDD/scrapping_platforms.py
import os
import re
import time
import random
import argparse
from typing import Optional, Dict, List, Iterable, Tuple

import requests
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# =========================
# Config & ENV
# =========================
load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY", "")
COUNTRY = os.getenv("TMDB_COUNTRY", "FR").upper()
LANG    = os.getenv("TMDB_LANGUAGE", "fr-FR")
DATABASE_URL = os.getenv("MYSQL_URL")

if not API_KEY or not DATABASE_URL:
    raise SystemExit("⛔ Configurez TMDB_API_KEY et MYSQL_URL (env/.env).")

BASE_URL = "https://api.themoviedb.org/3"
DISCOVER_MOVIE_URL = f"{BASE_URL}/discover/movie"
EXTERNAL_IDS_URL   = f"{BASE_URL}/movie/{{movie_id}}/external_ids"

REQUEST_SLEEP = float(os.getenv("REQUEST_SLEEP", "0.25"))
MAX_PAGES     = 500
TIMEOUT       = 25
BATCH_SIZE    = 300  # upsert par paquets

# ============
# Plateformes (slug -> provider_id TMDB) — ajoute librement
# ============
TARGET_PLATFORM_IDS: Dict[str, int] = {
    "netflix":     8,
    "disney":      337,   # Disney Plus
    "prime":       119,   # Amazon Prime Video
    "apple":       350,   # Apple TV+
    "canal":       381,   # Canal+
    "paramount":   531,   # Paramount Plus
    "hbo":         1899,  # HBO Max / Max
    "crunchyroll": 283,
    # exemples FR utiles si tu veux:
    # "arte": 234,
    # "francetv": 236,
}

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# =========================
# HTTP helper with backoff
# =========================
def get_with_backoff(url: str, params: dict, max_tries=6, base_sleep=0.6) -> Optional[requests.Response]:
    for attempt in range(1, max_tries + 1):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
        except requests.RequestException:
            r = None

        if r and r.status_code == 200:
            return r

        if r and r.status_code in (429, 500, 502, 503, 504):
            wait = base_sleep * (2 ** (attempt - 1)) + random.random() * 0.5
            ra = r.headers.get("Retry-After") if r is not None else None
            if ra:
                try:
                    wait = max(wait, float(ra))
                except Exception:
                    pass
            time.sleep(wait)
            continue

        if r is not None:
            # laisser lever si autre code
            r.raise_for_status()
        break
    return None

# =========================
# DB helpers
# =========================
def exec_sql(sql: str, params: dict | None = None):
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})

def fetchval(sql: str, params: dict | None = None):
    with engine.connect() as conn:
        return conn.execute(text(sql), params or {}).scalar()

def col_exists(conn, table: str, col: str) -> bool:
    q = """
    SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t AND COLUMN_NAME = :c
    """
    return conn.execute(text(q), {"t": table, "c": col}).scalar() > 0

def idx_exists(conn, table: str, idx: str) -> bool:
    q = """
    SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t AND INDEX_NAME = :i
    """
    return conn.execute(text(q), {"t": table, "i": idx}).scalar() > 0

def fk_exists(conn, fk: str) -> bool:
    q = """
    SELECT COUNT(*) FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS
    WHERE CONSTRAINT_SCHEMA = DATABASE() AND CONSTRAINT_NAME = :fk
    """
    return conn.execute(text(q), {"fk": fk}).scalar() > 0

# =========================
# Movies schema
# =========================
def ensure_movies_schema():
    with engine.begin() as conn:
        # table movies au minimum
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS movies (
                movie_id INT PRIMARY KEY
            ) ENGINE=InnoDB
        """))
        # imdb_id + index si absents
        if not col_exists(conn, "movies", "imdb_id"):
            conn.execute(text("ALTER TABLE movies ADD COLUMN imdb_id VARCHAR(15) NULL"))
        if not idx_exists(conn, "movies", "idx_movies_imdb"):
            conn.execute(text("CREATE INDEX idx_movies_imdb ON movies(imdb_id)"))

# =========================
# Shape idempotent des tables plateformes (n'ajoute que le manquant)
# =========================
def _platform_add_column_if_missing(conn, tbl: str, col: str, ddl: str):
    if not col_exists(conn, tbl, col):
        conn.execute(text(f"ALTER TABLE `{tbl}` ADD COLUMN {col} {ddl}"))

def _platform_add_unique_on_imdb(conn, tbl: str):
    # on crée un index UNIQUE (sans toucher à une PK existante) pour activer ON DUPLICATE KEY
    if not idx_exists(conn, tbl, f"ux_{tbl}_imdb"):
        try:
            conn.execute(text(f"CREATE UNIQUE INDEX ux_{tbl}_imdb ON `{tbl}`(imdb_id)"))
        except SQLAlchemyError:
            pass  # si imdb_id n'existe pas encore, ce sera rejoué après ajout de la colonne

def ensure_platform_shape(tbl: str):
    """
    Idempotent : ne supprime rien, n’écrase rien.
    - crée la table si absente (schéma minimal)
    - ajoute UNIQUEMENT les colonnes manquantes
    - ajoute un UNIQUE INDEX sur imdb_id (sans modifier la PK)
    - index movie_id si manquant
    - FK vers movies(movie_id) si absente
    """
    with engine.begin() as conn:
        # 1) crée si absente (schéma non destructif)
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS `{tbl}` (
                imdb_id      VARCHAR(15) NULL,
                movie_id     INT NULL,
                title        VARCHAR(512) NULL,
                release_year INT NULL,
                rating       FLOAT NULL,
                region       CHAR(2) NULL,
                genres       VARCHAR(255) NULL
            ) ENGINE=InnoDB
        """))

        # 2) ajoute colonnes manquantes
        _platform_add_column_if_missing(conn, tbl, "imdb_id",      "VARCHAR(15) NULL")
        _platform_add_column_if_missing(conn, tbl, "movie_id",     "INT NULL")
        _platform_add_column_if_missing(conn, tbl, "title",        "VARCHAR(512) NULL")
        _platform_add_column_if_missing(conn, tbl, "release_year", "INT NULL")
        _platform_add_column_if_missing(conn, tbl, "rating",       "FLOAT NULL")
        _platform_add_column_if_missing(conn, tbl, "region",       "CHAR(2) NULL")
        _platform_add_column_if_missing(conn, tbl, "genres",       "VARCHAR(255) NULL")

        # 3) index unique imdb pour l'UPSERT
        _platform_add_unique_on_imdb(conn, tbl)

        # 4) index movie_id
        if not idx_exists(conn, tbl, f"idx_{tbl}_movie"):
            conn.execute(text(f"CREATE INDEX idx_{tbl}_movie ON `{tbl}`(movie_id)"))

        # 5) FK vers movies(movie_id)
        fk_name = f"fk_{tbl}_movie"
        if not fk_exists(conn, fk_name):
            try:
                conn.execute(text(f"""
                    ALTER TABLE `{tbl}`
                    ADD CONSTRAINT {fk_name}
                    FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
                    ON UPDATE CASCADE ON DELETE SET NULL
                """))
            except SQLAlchemyError:
                pass

# =========================
# EXTERNAL IDS (IMDB) — cache simple en BDD
# =========================
def ensure_cache_table():
    exec_sql("""
    CREATE TABLE IF NOT EXISTS tmdb_external_ids_cache (
        tmdb_id INT PRIMARY KEY,
        imdb_id VARCHAR(15) NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB
    """)

def cache_get_imdb(tmdb_id: int) -> Optional[str]:
    return fetchval("SELECT imdb_id FROM tmdb_external_ids_cache WHERE tmdb_id = :t", {"t": tmdb_id})

def cache_set_imdb(tmdb_id: int, imdb_id: Optional[str]):
    exec_sql("""
    INSERT INTO tmdb_external_ids_cache (tmdb_id, imdb_id)
    VALUES (:t, :i)
    ON DUPLICATE KEY UPDATE imdb_id = VALUES(imdb_id)
    """, {"t": tmdb_id, "i": imdb_id})

def fetch_external_imdb_id(tmdb_id: int) -> Optional[str]:
    cached = cache_get_imdb(tmdb_id)
    if cached is not None:
        return cached or None
    r = get_with_backoff(EXTERNAL_IDS_URL.format(movie_id=tmdb_id), params={"api_key": API_KEY})
    imdb = None
    if r and r.status_code == 200:
        raw = r.json() or {}
        val = (raw.get("imdb_id") or "").strip()
        if re.fullmatch(r"tt\d+", val or ""):
            imdb = val
    cache_set_imdb(tmdb_id, imdb)
    return imdb

# =========================
# Discover movies for a provider (by ID)
# =========================
def safe_year(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    s = str(date_str)
    if len(s) >= 4 and s[:4].isdigit():
        try:
            return int(s[:4])
        except Exception:
            return None
    return None

def discover_provider_movies(provider_id: int, country: str, lang: str) -> List[dict]:
    """Tous les films (flatrate) disponibles pour un provider donné (toutes pages)."""
    out = []
    params = {
        "api_key": API_KEY,
        "with_watch_providers": provider_id,
        "watch_region": country,
        "with_watch_monetization_types": "flatrate",  # abonnement ONLY
        "language": lang,
        "sort_by": "popularity.desc",
        "page": 1
    }
    r = get_with_backoff(DISCOVER_MOVIE_URL, params)
    if not (r and r.status_code == 200):
        return out
    data = r.json() or {}
    total_pages = min(int(data.get("total_pages") or 1), MAX_PAGES)

    def collect(js):
        for mv in (js or []):
            tmdb_id = mv.get("id")
            if not tmdb_id:
                continue
            out.append({
                "tmdb_id": int(tmdb_id),
                "title": (mv.get("title") or mv.get("original_title") or "").strip(),
                "release_year": safe_year(mv.get("release_date")),
                "rating": float(mv.get("vote_average") or 0.0)
            })

    collect(data.get("results"))
    time.sleep(REQUEST_SLEEP)
    for page in range(2, total_pages + 1):
        params["page"] = page
        r = get_with_backoff(DISCOVER_MOVIE_URL, params)
        if not (r and r.status_code == 200):
            break
        collect((r.json() or {}).get("results"))
        time.sleep(REQUEST_SLEEP)
    return out

# =========================
# Upsert direct (sans staging)
# =========================
def chunked(iterable: Iterable, n: int) -> Iterable[List]:
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf

def upsert_platform_rows(tbl: str, rows: List[Dict]):
    """
    rows = list of dicts {imdb_id, movie_id, title, release_year, rating, region}
    genres sera backfill après map movie_id
    """
    if not rows:
        return
    sql = f"""
    INSERT INTO `{tbl}` (imdb_id, movie_id, title, release_year, rating, region)
    VALUES (:imdb_id, :movie_id, :title, :release_year, :rating, :region)
    ON DUPLICATE KEY UPDATE
        title = COALESCE(VALUES(title), `{tbl}`.title),
        release_year = COALESCE(VALUES(release_year), `{tbl}`.release_year),
        rating = COALESCE(VALUES(rating), `{tbl}`.rating),
        region = COALESCE(VALUES(region), `{tbl}`.region)
    """
    with engine.begin() as conn:
        conn.execute(text(sql), rows)

def map_movie_ids_from_imdb(tbl: str):
    exec_sql(f"""
        UPDATE `{tbl}` p
        JOIN movies m ON m.imdb_id = p.imdb_id
        SET p.movie_id = m.movie_id
        WHERE p.movie_id IS NULL
          AND p.imdb_id IS NOT NULL
    """)

def backfill_genres_from_movies(tbl: str):
    exec_sql(f"""
        UPDATE `{tbl}` p
        JOIN movies m ON m.movie_id = p.movie_id
        SET p.genres = m.genres
        WHERE p.movie_id IS NOT NULL
          AND (p.genres IS NULL OR p.genres = '')
    """)

# =========================
# Full run (UNE plateforme → table directe)
# =========================
def run_platform(slug: str):
    if slug not in TARGET_PLATFORM_IDS:
        print(f"• WARNING: plateforme inconnue '{slug}' — ignorée")
        return

    ensure_movies_schema()
    ensure_cache_table()
    ensure_platform_shape(slug)

    pid = TARGET_PLATFORM_IDS[slug]
    print(f"\n▶ {slug} (provider_id={pid}) — {COUNTRY} flatrate")

    # 1) discover tous les films
    movies = discover_provider_movies(pid, COUNTRY, LANG)
    print(f"  - Découvert: {len(movies)} films")

    # 2) fetch imdb_id (obligatoire) + upsert direct en batch
    to_upsert: List[Dict] = []
    done = 0
    for mv in movies:
        tmdb_id = mv["tmdb_id"]
        imdb = fetch_external_imdb_id(tmdb_id)
        if not imdb:  # imdb obligatoire → on skip si absent
            continue
        to_upsert.append({
            "imdb_id": imdb,
            "movie_id": None,           # sera mappé après via movies.imdb_id
            "title": mv["title"],
            "release_year": mv["release_year"],
            "rating": mv["rating"],
            "region": COUNTRY
        })
        if len(to_upsert) >= BATCH_SIZE:
            upsert_platform_rows(slug, to_upsert)
            done += len(to_upsert)
            print(f"    upsert: {done}/{len(movies)} (imdb ok)")
            to_upsert.clear()

    if to_upsert:
        upsert_platform_rows(slug, to_upsert)
        done += len(to_upsert)
        print(f"    upsert: {done}/{len(movies)} (final)")

    # 3) mapping + genres depuis movies
    map_movie_ids_from_imdb(slug)
    backfill_genres_from_movies(slug)

    # 4) stats
    total     = fetchval(f"SELECT COUNT(*) FROM `{slug}`")
    with_imdb = fetchval(f"SELECT COUNT(*) FROM `{slug}` WHERE imdb_id IS NOT NULL AND TRIM(imdb_id)<>''")
    mapped    = fetchval(f"SELECT COUNT(*) FROM `{slug}` WHERE movie_id IS NOT NULL")
    with_gen  = fetchval(f"SELECT COUNT(*) FROM `{slug}` WHERE genres IS NOT NULL AND TRIM(genres)<>''")
    print(f"  → {slug}: rows={total}, with_imdb_id={with_imdb}, mapped_to_movies={mapped}, with_genres={with_gen}")
    print("✅ Done.\n")

# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser(description="Scrape streaming catalogs (FR flatrate) via TMDB — direct, sans staging.")
    ap.add_argument("--platform", default="all", help="slug plateforme (ex: netflix) ou 'all'")
    return ap.parse_args()

def parse_platform_arg(pstr: str) -> List[str]:
    if not pstr or pstr.strip().lower() == "all":
        return list(TARGET_PLATFORM_IDS.keys())
    s = pstr.strip().lower()
    return [s] if s in TARGET_PLATFORM_IDS else []

if __name__ == "__main__":
    args = parse_args()
    slugs = parse_platform_arg(args.platform)
    if not slugs:
        print("ℹ️ Aucune plateforme valide fournie. Slugs possibles:", ", ".join(TARGET_PLATFORM_IDS.keys()))
    for slug in slugs:
        try:
            run_platform(slug)
        except SQLAlchemyError as e:
            print(f"⛔ SQLAlchemy error on {slug}:", e)
        except Exception as e:
            print(f"⛔ Unexpected error on {slug}:", e)

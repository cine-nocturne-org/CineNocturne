# E1_BDD/manage_streaming_catalogs.py
import os
import re
import time
import math
import random
import argparse
import requests
import pandas as pd
from typing import Optional, Dict, List
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# =========================
# Config
# =========================
load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY", "")
COUNTRY = os.getenv("TMDB_COUNTRY", "FR").upper()
LANG    = os.getenv("TMDB_LANGUAGE", "fr-FR")
DATABASE_URL = os.getenv("MYSQL_URL")

if not API_KEY or not DATABASE_URL:
    raise SystemExit("⛔ Configure TMDB_API_KEY et MYSQL_URL in .env/env.")

BASE_URL = "https://api.themoviedb.org/3"
DISCOVER_MOVIE_URL = f"{BASE_URL}/discover/movie"
PROVIDERS_LIST_MOVIE_URL = f"{BASE_URL}/watch/providers/movie"
EXTERNAL_IDS_URL = f"{BASE_URL}/movie/{{movie_id}}/external_ids"

# Politesse + limites
REQUEST_SLEEP = float(os.getenv("REQUEST_SLEEP", "0.25"))
MAX_PAGES = 500
TIMEOUT = 25

# Plateformes cibles (clé = nom de table, valeur = nom TMDB du provider)
TARGET_PLATFORMS = {
    "netflix": "Netflix",
    "prime":   "Amazon Prime Video",
    "hulu":    "Hulu",
    "hbo":     "Max",        # HBO Max = "Max"
    "apple":   "Apple TV+",
}

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# =========================
# HTTP helpers (429/backoff)
# =========================
def get_with_backoff(url: str, params: dict, max_tries=6, base_sleep=0.5) -> Optional[requests.Response]:
    for attempt in range(1, max_tries + 1):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
        except requests.RequestException:
            r = None
        if r and r.status_code == 200:
            return r
        if r and r.status_code in (429, 500, 502, 503, 504):
            wait = base_sleep * (2 ** (attempt - 1)) + random.random() * 0.5
            if r is not None:
                ra = r.headers.get("Retry-After")
                if ra:
                    try:
                        wait = max(wait, float(ra))
                    except Exception:
                        pass
            time.sleep(wait)
            continue
        # autres codes → stop
        if r is not None:
            try:
                r.raise_for_status()
            except Exception:
                pass
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
# Movies schema + external_ids cache
# =========================
def ensure_movies_schema():
    with engine.begin() as conn:
        if not col_exists(conn, "movies", "imdb_id"):
            conn.execute(text("ALTER TABLE movies ADD COLUMN imdb_id VARCHAR(15) NULL"))
        if not idx_exists(conn, "movies", "idx_movies_imdb"):
            conn.execute(text("CREATE INDEX idx_movies_imdb ON movies(imdb_id)"))

def ensure_cache_table():
    # cache durable des external_ids pour éviter de re-pinger TMDB
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
# Provider discovery
# =========================
def fetch_providers_map(country: str) -> Dict[str, int]:
    r = get_with_backoff(PROVIDERS_LIST_MOVIE_URL, params={"api_key": API_KEY, "watch_region": country})
    out = {}
    if r and r.status_code == 200:
        for p in (r.json() or {}).get("results", []):
            name = (p.get("provider_name") or "").strip().lower()
            pid = p.get("provider_id")
            if name and pid:
                out[name] = int(pid)
    return out

def safe_year(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    try:
        y = pd.to_datetime(date_str, errors="coerce").year
        return int(y) if pd.notna(y) else None
    except Exception:
        return None

def discover_provider_movies(provider_id: int, country: str, lang: str) -> List[dict]:
    out = []
    params = {
        "api_key": API_KEY,
        "with_watch_providers": provider_id,
        "watch_region": country,
        "with_watch_monetization_types": "flatrate",  # abonnement uniquement
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
# Platform tables lifecycle
# =========================
PLATFORM_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS `{tbl}` (
    id INT AUTO_INCREMENT PRIMARY KEY,
    imdb_id VARCHAR(15) NULL,
    movie_id INT NULL,                 -- TMDB id, FK vers movies(movie_id)
    title VARCHAR(512) NULL,
    release_year INT NULL,
    rating FLOAT NULL,
    region CHAR(2) NULL
) ENGINE=InnoDB
"""

def reset_streaming_tables():
    """Drop & recreate un schéma propre pour toutes les tables de plateformes."""
    ensure_movies_schema()
    for tbl in TARGET_PLATFORMS.keys():
        exec_sql(f"DROP TABLE IF EXISTS `{tbl}`")
        exec_sql(PLATFORM_SCHEMA_SQL.replace("{tbl}", tbl))
        # indexes + FK
        exec_sql(f"CREATE INDEX idx_{tbl}_imdb ON `{tbl}`(imdb_id)")
        exec_sql(f"CREATE INDEX idx_{tbl}_movie ON `{tbl}`(movie_id)")
        exec_sql(f"""
            ALTER TABLE `{tbl}`
            ADD CONSTRAINT fk_{tbl}_movie
            FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
            ON UPDATE CASCADE ON DELETE SET NULL
        """)
    print("✅ Tables streaming réinitialisées (schéma clean).")

def ensure_platform_staging(tbl: str):
    exec_sql(f"DROP TABLE IF EXISTS `{tbl}_new`")
    exec_sql(PLATFORM_SCHEMA_SQL.replace("{tbl}", f"{tbl}_new"))
    exec_sql(f"CREATE INDEX idx_{tbl}_new_imdb ON `{tbl}_new`(imdb_id)")
    exec_sql(f"CREATE INDEX idx_{tbl}_new_movie ON `{tbl}_new`(movie_id)")

def insert_rows_staging(tbl: str, rows: List[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["imdb_id","movie_id","title","release_year","rating","region"])
    df.to_sql(f"{tbl}_new", con=engine, if_exists="append", index=False)

def clean_invalid_imdb(tbl: str):
    exec_sql(f"""
        UPDATE `{tbl}_new`
        SET imdb_id = NULL
        WHERE imdb_id IS NOT NULL AND imdb_id NOT REGEXP '^tt[0-9]+$'
    """)

def map_movie_ids_from_imdb(tbl: str):
    exec_sql(f"""
        UPDATE `{tbl}_new` n
        JOIN movies m ON m.imdb_id = n.imdb_id
        SET n.movie_id = m.movie_id
        WHERE n.movie_id IS NULL AND n.imdb_id IS NOT NULL
    """)

def swap_in_platform(tbl: str):
    # Retire FK pour éviter conflits, swap, puis recrée FK
    with engine.begin() as conn:
        # Supprime FK existante si présente
        fk_name = f"fk_{tbl}_movie"
        if fk_exists(conn, fk_name):
            conn.execute(text(f"ALTER TABLE `{tbl}` DROP FOREIGN KEY {fk_name}"))

        # swap atomique
        conn.execute(text(f"RENAME TABLE `{tbl}` TO `{tbl}_old`, `{tbl}_new` TO `{tbl}`"))

        # Réindexation (au cas où)
        if not idx_exists(conn, tbl, f"idx_{tbl}_imdb"):
            conn.execute(text(f"CREATE INDEX idx_{tbl}_imdb ON `{tbl}`(imdb_id)"))
        if not idx_exists(conn, tbl, f"idx_{tbl}_movie"):
            conn.execute(text(f"CREATE INDEX idx_{tbl}_movie ON `{tbl}`(movie_id)"))

        # Recrée FK
        if not fk_exists(conn, fk_name):
            conn.execute(text(f"""
                ALTER TABLE `{tbl}`
                ADD CONSTRAINT {fk_name}
                FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
                ON UPDATE CASCADE ON DELETE SET NULL
            """))

        # Drop old
        conn.execute(text(f"DROP TABLE IF EXISTS `{tbl}_old`"))

def run_scrape_and_replace():
    ensure_movies_schema()
    ensure_cache_table()

    providers_map = fetch_providers_map(COUNTRY)
    if not providers_map:
        raise SystemExit(f"⛔ Impossible de récupérer la liste des providers TMDB pour {COUNTRY}.")

    for tbl, provider_display in TARGET_PLATFORMS.items():
        provider_id = providers_map.get(provider_display.lower())
        if not provider_id:
            print(f"• {provider_display} indisponible en {COUNTRY} — skip {tbl}.")
            continue

        print(f"\n▶ {provider_display} (id={provider_id}) — {COUNTRY} flatrate")
        ensure_platform_staging(tbl)

        # Discover
        movies = discover_provider_movies(provider_id, COUNTRY, LANG)
        print(f"  - Découvert: {len(movies)} films")

        # External IDs + build rows
        rows = []
        for i, mv in enumerate(movies, start=1):
            tmdb_id = mv["tmdb_id"]
            imdb = fetch_external_imdb_id(tmdb_id)
            rows.append({
                "imdb_id": imdb,
                "movie_id": tmdb_id,
                "title": mv["title"],
                "release_year": mv["release_year"],
                "rating": mv["rating"],
                "region": COUNTRY
            })
            if i % 100 == 0:
                print(f"    external_ids: {i}/{len(movies)}")
            time.sleep(REQUEST_SLEEP)

        insert_rows_staging(tbl, rows)
        clean_invalid_imdb(tbl)
        map_movie_ids_from_imdb(tbl)
        swap_in_platform(tbl)

        total = fetchval(f"SELECT COUNT(*) FROM `{tbl}`")
        mapped = fetchval(f"SELECT COUNT(*) FROM `{tbl}` WHERE movie_id IS NOT NULL")
        with_imdb = fetchval(f"SELECT COUNT(*) FROM `{tbl}` WHERE imdb_id IS NOT NULL AND TRIM(imdb_id)<>''")
        print(f"  → {tbl}: rows={total}, with_imdb_id={with_imdb}, mapped_to_movies={mapped}")

    print("\n✅ Scrape & replace terminé.")

# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser(description="Reset & scrape streaming catalogs (flatrate) from TMDB")
    ap.add_argument("--reset", action="store_true", help="Réinitialise les tables streaming (drop & recreate)")
    ap.add_argument("--scrape", action="store_true", help="Scrape & remplace les tables via staging+swap")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        if args.reset:
            reset_streaming_tables()
        if args.scrape:
            run_scrape_and_replace()
        if not (args.reset or args.scrape):
            print("ℹ️ Rien à faire. Utilise --reset et/ou --scrape.")
    except SQLAlchemyError as e:
        print("⛔ SQLAlchemy error:", e)
    except Exception as e:
        print("⛔ Unexpected error:", e)

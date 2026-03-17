"""
fetch_wiki_species_text.py

Fetches Wikipedia article text for species and produces a geolocated
(text, lat, lon) dataset.

Input files (pick one observation source):
  matched_sinr_goodsentinel2_deduped.csv  [RECOMMENDED]
      Output of 00a_match_sinr_data.py.
      Columns: fp, datetime, year_x, month_x, day_x, col, row,
               observation_uuid, observer_id, latitude, longitude,
               taxon_id, observed_on, year_y, month_y, day_y, col-row
      Best choice: already spatially deduped and Sentinel-matched.

  geo_prior_train.csv
      Raw iNaturalist observations (LE-SINR training file).
      Columns: observation_uuid, observer_id, latitude, longitude,
               positional_accuracy, taxon_id, quality_grade, observed_on
      Largest set; use for maximum species coverage.

  sampled_geo_prior_data.csv
      ~100k sample from 00b_assignlatlon.py. Same columns as geo_prior_train.
      Use for fast development runs.

  sinr_with_mercator.csv
      SINR obs with added Mercator tile col/row.
      Columns: same as geo_prior_train + col, row

  common_sinr_goodsentinel2_locyear.csv
      SINR × Sentinel matched by location AND year (pre-dedup).
      Columns: similar to matched_sinr_goodsentinel2_deduped.csv

NOTE: None of these files contain species names. Scientific names must be
supplied via --name_csv pointing to the iNaturalist taxonomy file, which is
a standard LE-SINR/SINR dependency:
  https://www.inaturalist.org/taxa/inaturalist-taxonomy.dwca  (DwCA zip)
  or the pre-extracted taxa.csv from that archive.
  Relevant columns: id (= taxon_id), name (scientific name), rank

If --name_csv is not provided the script falls back to the iNaturalist
public API (one request per taxon_id) which is much slower.

API strategy
------------
Primary  : MediaWiki Action API  action=parse  (current, not deprecated)
             Step 1 — fetch section list:
               action=parse&page={title}&prop=sections
             Step 2 — fetch each section's wikitext:
               action=parse&page={title}&prop=wikitext&section={idx}
           Endpoint: https://en.wikipedia.org/w/api.php

Fallback : TextExtracts  prop=extracts  (used by LE-SINR; officially deprecated)
             action=query&prop=extracts&explaintext=1&exsectionformat=wiki
               &redirects=1&titles={title}
           Enable with --use_extracts.  Fewer API calls (1 per species) but
           the extension may be removed from Wikipedia in future.

Rate limits: Wikipedia asks for ≤200 req/s.  Default --delay 1.0 is safe
             for a single process.  On a cluster use --delay 0.2.

Usage
-----
# Recommended: matched+deduped obs, names from iNat taxonomy CSV
python fetch_wiki_species_text.py \
    --obs      data/matched_sinr_goodsentinel2_deduped.csv \
    --name_csv data/taxa.csv \
    --out_dir  output/

# Larger species coverage using raw geo_prior obs:
python fetch_wiki_species_text.py \
    --obs      data/geo_prior_train.csv \
    --name_csv data/taxa.csv \
    --out_dir  output/

# Fast dev run: 200 species, deprecated extracts API (1 req/species):
python fetch_wiki_species_text.py \
    --obs          data/sampled_geo_prior_data.csv \
    --name_csv     data/taxa.csv \
    --use_extracts \
    --limit        200 \
    --out_dir      output/

# Without taxa.csv: fall back to iNat API for names (slow):
python fetch_wiki_species_text.py \
    --obs      data/matched_sinr_goodsentinel2_deduped.csv \
    --out_dir  output/

# Resume an interrupted run:
python fetch_wiki_species_text.py \
    --obs      data/matched_sinr_goodsentinel2_deduped.csv \
    --name_csv data/taxa.csv \
    --resume \
    --out_dir  output/

# Only keep range + habitat sections:
python fetch_wiki_species_text.py \
    --obs            data/matched_sinr_goodsentinel2_deduped.csv \
    --name_csv       data/taxa.csv \
    --section_filter range,habitat \
    --out_dir        output/

Output
------
output/wiki_species_text_raw.jsonl      one JSON object per species (all sections)
output/geolocated_text_dataset.csv      flat (taxon_id, section_name, text, lat, lon)
output/geolocated_text_dataset.parquet  same, columnar format
output/fetch_errors.csv                 species that could not be fetched
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIKI_API   = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "WildSAT-species-range-dataset/1.0 (research; contact via github)"

SKIP_SECTION_RE = re.compile(
    r"(references?|external\s+links?|further\s+reading|bibliography"
    r"|see\s+also|notes?|footnotes?|sources?|citation)",
    re.IGNORECASE,
)
RANGE_RE   = re.compile(r"(range|distribution|geographic|geography)", re.IGNORECASE)
HABITAT_RE = re.compile(r"(habitat|ecolog|environment|niche|biome)", re.IGNORECASE)


def classify_section(name: str) -> str:
    if RANGE_RE.search(name):   return "range"
    if HABITAT_RE.search(name): return "habitat"
    if name.lower() in ("", "lead", "summary", "overview", "introduction"): return "lead"
    return "other"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def _get(params: dict, retries: int = 3, backoff: float = 5.0) -> dict:
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    for attempt in range(retries):
        try:
            r = SESSION.get(WIKI_API, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (attempt + 1))


# ---------------------------------------------------------------------------
# Wikitext cleaning
# ---------------------------------------------------------------------------

def clean_wikitext(text: str) -> str:
    """Strip wikitext markup, return plain prose."""
    text = re.sub(r"\{\{[^}]*\}\}", " ", text)                          # templates
    text = re.sub(r"\[\[(File|Image):[^\]]*\]\]", "", text, re.I)       # media links
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)       # wiki links
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)          # ext links with label
    text = re.sub(r"\[https?://\S+\]", "", text)                         # bare ext links
    text = re.sub(r"'{2,3}", "", text)                                   # bold/italic
    text = re.sub(r"^={1,6}[^=]+=+\s*$", "", text, flags=re.MULTILINE) # headings
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Wikipedia fetch — action=parse  (primary, non-deprecated)
# ---------------------------------------------------------------------------

def get_sections_parse_api(title: str) -> list[dict]:
    """
    Two-step fetch:
      1. action=parse&prop=sections  → section list with index numbers
      2. action=parse&prop=wikitext&section=N  → per-section wikitext
    Returns list of {section_index, section_name, section_type, text}.
    """
    toc = _get({"action": "parse", "page": title, "prop": "sections", "redirects": True})
    if "error" in toc:
        raise ValueError(toc["error"].get("info", "unknown API error"))

    sections_meta = toc.get("parse", {}).get("sections", [])
    results = []

    # Lead section (index 0)
    try:
        lead_raw = _get({"action": "parse", "page": title, "prop": "wikitext", "section": 0})
        lead_text = clean_wikitext(lead_raw.get("parse", {}).get("wikitext", ""))
        if len(lead_text) > 50:
            results.append({"section_index": 0, "section_name": "Lead",
                             "section_type": "lead", "text": lead_text})
    except Exception:
        pass

    # Named sections
    for sec in sections_meta:
        sec_name = sec.get("line", sec.get("anchor", ""))
        sec_idx  = sec.get("index", sec.get("number", ""))
        if SKIP_SECTION_RE.search(sec_name):
            continue
        try:
            raw = _get({"action": "parse", "page": title,
                        "prop": "wikitext", "section": sec_idx})
            text = clean_wikitext(raw.get("parse", {}).get("wikitext", ""))
            if len(text) >= 30:
                results.append({"section_index": sec_idx, "section_name": sec_name,
                                 "section_type": classify_section(sec_name), "text": text})
        except Exception:
            continue

    return results


# ---------------------------------------------------------------------------
# Wikipedia fetch — prop=extracts  (deprecated, LE-SINR compatible)
# ---------------------------------------------------------------------------

def get_sections_extracts_api(title: str) -> list[dict]:
    """
    Single call using the TextExtracts extension — same URL as LE-SINR paper:
      action=query&prop=extracts&redirects=1&formatversion=2&titles={title}
    Splits the plain-text response on == Section == headers.
    """
    data = _get({
        "action":          "query",
        "prop":            "extracts",
        "redirects":       True,
        "titles":          title,
        "explaintext":     True,
        "exsectionformat": "wiki",   # headers rendered as == Title ==
    })
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        raise ValueError(f"Page not found: '{title}'")
    full_text = pages[0].get("extract", "")
    if not full_text:
        raise ValueError(f"Empty extract for '{title}'")

    header_re = re.compile(r"^(={2,4})\s*(.+?)\s*\1\s*$", re.MULTILINE)
    chunks = header_re.split(full_text)

    results = []
    lead = chunks[0].strip()
    if len(lead) > 50:
        results.append({"section_index": 0, "section_name": "Lead",
                         "section_type": "lead", "text": clean_wikitext(lead)})

    i, sec_idx = 1, 1
    while i + 2 <= len(chunks):
        sec_name = chunks[i + 1].strip()
        sec_text = chunks[i + 2].strip()
        i += 3
        if SKIP_SECTION_RE.search(sec_name):
            sec_idx += 1
            continue
        text = clean_wikitext(sec_text)
        if len(text) >= 30:
            results.append({"section_index": sec_idx, "section_name": sec_name,
                             "section_type": classify_section(sec_name), "text": text})
        sec_idx += 1

    return results


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Known column-name variants across the five CSV files.
# geo_prior_train / sinr_with_mercator use:  latitude, longitude, taxon_id
# matched_sinr_goodsentinel2_deduped uses:   latitude, longitude, taxon_id  (same)
_LAT_COLS  = ("latitude", "lat", "y")
_LON_COLS  = ("longitude", "lon", "lng", "x")
_TID_COLS  = ("taxon_id", "taxonid", "id", "species_id")


def _pick(col_lower: dict, *candidates) -> str | None:
    for c in candidates:
        if c in col_lower:
            return col_lower[c]
    return None


def load_obs_csv(path: str) -> pd.DataFrame:
    """
    Load any of the five available observation CSVs.
    Both confirmed schemas use: taxon_id, latitude, longitude
    Returns DataFrame with columns: taxon_id (int), lat (float), lon (float).
    """
    print(f"  Reading {Path(path).name} ...")
    df = pd.read_csv(path, low_memory=False)
    cl = {c.lower().strip(): c for c in df.columns}

    tid = _pick(cl, *_TID_COLS)
    lat = _pick(cl, *_LAT_COLS)
    lon = _pick(cl, *_LON_COLS)

    missing = [k for k, v in {"taxon_id": tid, "lat": lat, "lon": lon}.items() if v is None]
    if missing:
        raise ValueError(f"Cannot find {missing} in columns: {list(df.columns)}")

    out = df[[tid, lat, lon]].rename(columns={tid: "taxon_id", lat: "lat", lon: "lon"}).copy()
    out["taxon_id"] = pd.to_numeric(out["taxon_id"], errors="coerce").astype("Int64")
    out["lat"]      = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"]      = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["taxon_id", "lat", "lon"])
    out["taxon_id"] = out["taxon_id"].astype(int)
    print(f"    {len(out):,} valid rows, {out['taxon_id'].nunique():,} unique species")
    return out


def load_name_lookup_from_taxonomy(path: str) -> dict[int, str]:
    """
    Build taxon_id → scientific_name from the iNaturalist taxonomy CSV.

    The standard file is taxa.csv extracted from the DwCA archive at:
      https://www.inaturalist.org/taxa/inaturalist-taxonomy.dwca

    Relevant columns (may vary slightly by export version):
      id          → taxon_id  (integer)
      name        → scientific name (binomial for species)
      rank        → 'species', 'genus', etc.  (we keep all ranks)

    Also handles the simpler two-column format used by some SINR releases:
      taxon_id, name
    """
    print(f"  Reading taxonomy from {Path(path).name} ...")
    df = pd.read_csv(path, low_memory=False)
    cl = {c.lower().strip(): c for c in df.columns}

    # Try iNat DwCA taxa.csv format first (id, scientificName or name)
    id_col   = _pick(cl, "id", "taxon_id", "taxonid")
    name_col = _pick(cl, "scientificname", "name", "scientific_name",
                     "taxon_name", "species_name")

    if id_col is None or name_col is None:
        raise ValueError(
            f"Cannot find id/name columns in taxonomy file. "
            f"Columns present: {list(df.columns)}\n"
            f"Expected a file with (id/taxon_id) and (name/scientificName) columns.\n"
            f"Download from: https://www.inaturalist.org/taxa/inaturalist-taxonomy.dwca"
        )

    sub = df[[id_col, name_col]].dropna()
    sub[id_col] = pd.to_numeric(sub[id_col], errors="coerce").astype("Int64")
    sub = sub.dropna(subset=[id_col])
    lookup = {int(r[id_col]): str(r[name_col]) for _, r in sub.iterrows()}
    print(f"    {len(lookup):,} taxon names loaded")
    return lookup


def lookup_name_inat_api(taxon_id: int, session: requests.Session,
                          cache: dict, delay: float = 0.5) -> str | None:
    """
    Fall back: query the iNaturalist public API for a single taxon name.
    Caches results in `cache` dict to avoid duplicate requests.
    Rate limit: iNat asks for ≤60 req/min; delay=0.5 gives ~120/min so
    use delay≥1.0 to be safe.
    """
    if taxon_id in cache:
        return cache[taxon_id]
    try:
        r = session.get(
            f"https://api.inaturalist.org/v1/taxa/{taxon_id}",
            timeout=10,
            headers={"User-Agent": USER_AGENT},
        )
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        name = results[0].get("name") if results else None
        cache[taxon_id] = name
        time.sleep(delay)
        return name
    except Exception:
        cache[taxon_id] = None
        return None


def aggregate_locations(obs: pd.DataFrame, method: str) -> pd.DataFrame:
    """One representative (lat, lon) per taxon_id."""
    if method == "mean":
        loc = obs.groupby("taxon_id")[["lat", "lon"]].mean().reset_index()
    elif method == "first":
        loc = (obs.sort_values("taxon_id")
                  .groupby("taxon_id").first()
                  .reset_index()[["taxon_id", "lat", "lon"]])
    else:  # dedup — keep one row per unique location per species
        loc = obs[["taxon_id", "lat", "lon"]].drop_duplicates()
    return loc


# ---------------------------------------------------------------------------
# Section filter helper
# ---------------------------------------------------------------------------

def passes_filter(sec: dict, section_filter: list[str] | None) -> bool:
    if not section_filter:
        return True
    name_lc = sec["section_name"].lower()
    return sec["section_type"] in section_filter or any(f in name_lc for f in section_filter)


# ---------------------------------------------------------------------------
# JSONL → flat CSV
# ---------------------------------------------------------------------------

def jsonl_to_csv(jsonl_path: str, csv_path: str, parq_path: str) -> pd.DataFrame:
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for sec in obj.get("sections", []):
                rows.append({
                    "taxon_id":     obj["taxon_id"],
                    "species_name": obj.get("species_name", ""),
                    "section_name": sec["section_name"],
                    "section_type": sec["section_type"],
                    "text":         sec["text"],
                    "lat":          obj["lat"],
                    "lon":          obj["lon"],
                })
    result = pd.DataFrame(rows)
    result.to_csv(csv_path, index=False)
    print(f"Saved CSV     → {csv_path}  ({os.path.getsize(csv_path)/1e6:.1f} MB)")
    try:
        result.to_parquet(parq_path, index=False)
        print(f"Saved Parquet → {parq_path}  ({os.path.getsize(parq_path)/1e6:.1f} MB)")
    except ImportError:
        print("(pyarrow not installed — skipping Parquet)")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch Wikipedia text per species and build a geolocated (text, lat, lon) dataset."
    )
    parser.add_argument("--obs", required=True,
                        help="Observation CSV. Any of: matched_sinr_goodsentinel2_deduped.csv, "
                             "geo_prior_train.csv, sampled_geo_prior_data.csv, "
                             "sinr_with_mercator.csv, common_sinr_goodsentinel2_locyear.csv")
    parser.add_argument("--name_csv", default=None,
                        help="iNaturalist taxonomy CSV with taxon id + scientific name. "
                             "Recommended: taxa.csv from inaturalist-taxonomy.dwca. "
                             "If omitted, names are fetched one-by-one from the iNat API (slow).")
    parser.add_argument("--out_dir",        default=".")
    parser.add_argument("--agg",            default="mean", choices=["mean", "first", "dedup"],
                        help="How to get one location per species (default: mean)")
    parser.add_argument("--delay",          type=float, default=1.0,
                        help="Seconds between Wikipedia API requests (default: 1.0)")
    parser.add_argument("--use_extracts",   action="store_true",
                        help="Use deprecated prop=extracts API — matches LE-SINR exactly, "
                             "1 request per species instead of N+1")
    parser.add_argument("--section_filter", default=None,
                        help="Comma-separated section types to keep, e.g. range,habitat. "
                             "Default: keep all sections.")
    parser.add_argument("--resume",         action="store_true",
                        help="Skip taxon_ids already written to output JSONL")
    parser.add_argument("--limit",          type=int, default=None,
                        help="Stop after this many species (for testing)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    section_filter = (
        [s.strip().lower() for s in args.section_filter.split(",")]
        if args.section_filter else None
    )

    jsonl_path = os.path.join(args.out_dir, "wiki_species_text_raw.jsonl")
    csv_path   = os.path.join(args.out_dir, "geolocated_text_dataset.csv")
    parq_path  = os.path.join(args.out_dir, "geolocated_text_dataset.parquet")
    err_path   = os.path.join(args.out_dir, "fetch_errors.csv")

    # --- Load observations → one (lat, lon) per taxon_id ---
    print("Loading observations ...")
    obs    = load_obs_csv(args.obs)
    loc_df = aggregate_locations(obs, args.agg)

    # --- Build taxon_id → scientific name lookup ---
    name_lookup: dict[int, str] = {}
    if args.name_csv:
        name_lookup = load_name_lookup_from_taxonomy(args.name_csv)
    else:
        print("  No --name_csv provided. Will query iNat API for missing names (slow).")
        print("  Tip: download taxa.csv from https://www.inaturalist.org/taxa/inaturalist-taxonomy.dwca")

    # Pre-populate species_name column; blanks will be resolved via iNat API at fetch time
    loc_df["species_name"] = loc_df["taxon_id"].map(name_lookup).fillna("")

    n_missing_names = (loc_df["species_name"] == "").sum()
    if n_missing_names:
        print(f"  {n_missing_names:,} species have no name in taxonomy file "
              f"— will fall back to iNat API for these.")

    if args.limit:
        loc_df = loc_df.head(args.limit)
        print(f"  Limiting to {args.limit} species.")

    # --- Resume ---
    done_ids: set[int] = set()
    if args.resume and os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                try:
                    done_ids.add(int(json.loads(line)["taxon_id"]))
                except Exception:
                    pass
        print(f"  Resuming: {len(done_ids):,} species already done, skipping.")

    fetch_fn  = get_sections_extracts_api if args.use_extracts else get_sections_parse_api
    inat_cache: dict[int, str | None] = {}
    print(f"  API: {'prop=extracts (deprecated, LE-SINR compatible)' if args.use_extracts else 'action=parse (current)'}")
    print(f"  Species to fetch: {len(loc_df) - len(done_ids):,}\n")

    errors     = []
    n_fetched  = 0
    n_sections = 0

    todo = [row for _, row in loc_df.iterrows() if int(row["taxon_id"]) not in done_ids]

    with open(jsonl_path, "a", encoding="utf-8") as out_f:
        pbar = tqdm(todo, unit="species", dynamic_ncols=True)
        for row in pbar:
            tid  = int(row["taxon_id"])
            lat  = float(row["lat"])
            lon  = float(row["lon"])
            name = str(row["species_name"]).strip()

            # Resolve name via iNat API if still missing
            if not name and not args.name_csv:
                name = lookup_name_inat_api(tid, SESSION, inat_cache, delay=args.delay) or ""

            if not name:
                errors.append({"taxon_id": tid, "search_term": "", "error": "no species name found"})
                pbar.set_postfix(fetched=n_fetched, errors=len(errors), last="NO NAME")
                continue

            # Show what's being fetched so names/ids can be sanity-checked
            tqdm.write(f"  {tid:>8}  {name}")

            try:
                sections = fetch_fn(name)
            except Exception as e:
                errors.append({"taxon_id": tid, "search_term": name, "error": str(e)})
                tqdm.write(f"           ERROR: {e}")
                time.sleep(args.delay)
                pbar.set_postfix(fetched=n_fetched, errors=len(errors), last=f"ERR:{name[:20]}")
                continue

            if section_filter:
                sections = [s for s in sections if passes_filter(s, section_filter)]

            if not sections:
                errors.append({"taxon_id": tid, "search_term": name,
                                "error": "no sections after filter"})
                tqdm.write(f"           no matching sections")
            else:
                out_f.write(json.dumps({
                    "taxon_id": tid, "species_name": name,
                    "lat": lat, "lon": lon, "sections": sections,
                }, ensure_ascii=False) + "\n")
                out_f.flush()
                n_fetched  += 1
                n_sections += len(sections)
                sec_types = ", ".join(sorted({s["section_type"] for s in sections}))
                tqdm.write(f"           {len(sections)} sections [{sec_types}]  ({lat:.2f}, {lon:.2f})")

            pbar.set_postfix(fetched=n_fetched, errors=len(errors), last=name[:25])
            time.sleep(args.delay)

    print(f"\nFetch complete: {n_fetched:,} species, {n_sections:,} sections, "
          f"{len(errors):,} errors.")

    if errors:
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print(f"Errors → {err_path}")

    print("\nBuilding final dataset ...")
    result = jsonl_to_csv(jsonl_path, csv_path, parq_path)

    print(f"\n=== Dataset Summary ===")
    print(f"(text, lat, lon) pairs : {len(result):,}")
    print(f"Unique species         : {result['taxon_id'].nunique():,}")
    if "section_type" in result.columns:
        print("\nSection type breakdown:")
        print(result["section_type"].value_counts().to_string())
    print(f"\nLat range : [{result['lat'].min():.2f}, {result['lat'].max():.2f}]")
    print(f"Lon range : [{result['lon'].min():.2f}, {result['lon'].max():.2f}]")


if __name__ == "__main__":
    main()
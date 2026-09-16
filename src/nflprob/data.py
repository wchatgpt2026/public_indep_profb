from __future__ import annotations

import io
import time
from collections.abc import Iterable
from typing import Any

import pandas as pd
import requests

from .features import build_pregame_features

NFLVERSE_REPO = "nflverse/nflverse-data"
PBP_RELEASE_TAG = "pbp"
GITHUB_API = "https://api.github.com"


def _request(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    attempts: int = 4,
    timeout: int = 120,
) -> requests.Response:
    """GET with bounded exponential retry for transient GitHub/CDN failures."""
    last_error: Exception | None = None
    request_headers = {
        "User-Agent": "public-indep-profb/0.1",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if headers:
        request_headers.update(headers)

    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                headers=request_headers,
                timeout=timeout,
                allow_redirects=True,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 8))

    raise ConnectionError(f"Failed after {attempts} attempts: {url}: {last_error}") from last_error


def _release_assets() -> dict[str, dict[str, Any]]:
    """Return PBP release assets keyed by filename using GitHub's API."""
    release = _request(
        f"{GITHUB_API}/repos/{NFLVERSE_REPO}/releases/tags/{PBP_RELEASE_TAG}"
    ).json()
    assets_url = release["assets_url"]
    assets: dict[str, dict[str, Any]] = {}
    page = 1
    while True:
        batch = _request(f"{assets_url}?per_page=100&page={page}").json()
        for asset in batch:
            assets[str(asset["name"])] = asset
        if len(batch) < 100:
            break
        page += 1
    return assets


def _download_asset(asset: dict[str, Any]) -> bytes:
    """Download a release asset via the API, then browser URL as a fallback."""
    api_url = str(asset["url"])
    try:
        response = _request(
            api_url,
            headers={"Accept": "application/octet-stream"},
        )
        content_type = response.headers.get("content-type", "").lower()
        if "application/json" not in content_type and response.content:
            return response.content
    except ConnectionError:
        pass

    # A cache-busting query avoids stale signed redirect responses occasionally
    # served by GitHub's release-asset path.
    browser_url = str(asset["browser_download_url"])
    separator = "&" if "?" in browser_url else "?"
    response = _request(
        f"{browser_url}{separator}cachebust={time.time_ns()}",
        headers={"Accept": "application/octet-stream"},
    )
    return response.content


def _fallback_pbp(season: int) -> pd.DataFrame:
    """Load one PBP season directly from nflverse release assets."""
    assets = _release_assets()
    candidates = [
        (f"play_by_play_{season}.parquet", "parquet"),
        (f"play_by_play_{season}.csv.gz", "csv.gz"),
        (f"play_by_play_{season}.csv", "csv"),
    ]
    errors: list[str] = []
    for filename, kind in candidates:
        asset = assets.get(filename)
        if asset is None:
            continue
        try:
            content = _download_asset(asset)
            buffer = io.BytesIO(content)
            if kind == "parquet":
                return pd.read_parquet(buffer)
            if kind == "csv.gz":
                return pd.read_csv(buffer, compression="gzip", low_memory=False)
            return pd.read_csv(buffer, low_memory=False)
        except Exception as exc:  # noqa: BLE001 - deliberately try the next published format
            errors.append(f"{filename}: {exc}")
    detail = "; ".join(errors) if errors else "no matching release assets found"
    raise ConnectionError(f"Unable to load nflverse PBP for {season}: {detail}")


def _load_pbp_season(nfl: Any, season: int, attempts: int = 3) -> pd.DataFrame:
    """Use nflreadpy first, retry transient failures, then use release API fallback."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return nfl.load_pbp(season).to_pandas()
        except ConnectionError as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 4))
    try:
        return _fallback_pbp(season)
    except ConnectionError as fallback_error:
        raise ConnectionError(
            f"nflreadpy and direct nflverse fallback both failed for {season}. "
            f"nflreadpy error: {last_error}; fallback error: {fallback_error}"
        ) from fallback_error


def _load_schedules(nfl: Any, seasons: list[int], attempts: int = 3) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return nfl.load_schedules(seasons).to_pandas()
        except ConnectionError as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 4))
    raise ConnectionError(f"Failed to load nflverse schedules after {attempts} attempts") from last_error


def load_nflverse(seasons: Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load nflverse schedule and PBP with retries and release-asset fallback."""
    try:
        import nflreadpy as nfl
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install data dependencies with: pip install 'nflprob[data]'") from exc

    seasons = [int(s) for s in seasons]
    if not seasons:
        raise ValueError("at least one season is required")

    schedule = _load_schedules(nfl, seasons)
    pbp_frames = [_load_pbp_season(nfl, season) for season in seasons]
    pbp = pd.concat(pbp_frames, ignore_index=True, sort=False)
    return schedule, pbp


def build_nflverse_dataset(
    seasons: Iterable[int],
    half_life_games: float = 6.0,
) -> pd.DataFrame:
    schedule, pbp = load_nflverse(seasons)
    return build_pregame_features(schedule, pbp, half_life_games=half_life_games)

# src/data_collector.py
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, List, Dict
import os
import json
import re

logger = logging.getLogger(__name__)

class NBADataCollector:
    """Collects NBA data from ESPN and Odds API with robust normalization"""
    
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        # ESPN JSON API — same pattern as the working NCAA scraper
        self.espn_api_url    = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        self.espn_injury_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
        self.odds_api_key = os.getenv('ODDS_API_KEY', '')  # Optional — only used for odds enrichment
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Comprehensive team name normalization mapping
        self.team_name_mapping = {
            # ESPN variations to standard names
            'LA Clippers': 'LA Clippers',
            'L.A. Clippers': 'LA Clippers',
            'Los Angeles Clippers': 'LA Clippers',
            'Clippers': 'LA Clippers',
            
            'LA Lakers': 'Los Angeles Lakers',
            'L.A. Lakers': 'Los Angeles Lakers',
            'Los Angeles Lakers': 'Los Angeles Lakers',
            'Lakers': 'Los Angeles Lakers',
            
            'NY Knicks': 'New York Knicks',
            'N.Y. Knicks': 'New York Knicks',
            'New York Knicks': 'New York Knicks',
            'Knicks': 'New York Knicks',
            
            'SA Spurs': 'San Antonio Spurs',
            'S.A. Spurs': 'San Antonio Spurs',
            'San Antonio Spurs': 'San Antonio Spurs',
            'Spurs': 'San Antonio Spurs',
            
            'GS Warriors': 'Golden State Warriors',
            'G.S. Warriors': 'Golden State Warriors',
            'Golden State Warriors': 'Golden State Warriors',
            'Warriors': 'Golden State Warriors',
            'Golden St.': 'Golden State Warriors',
            
            'OKC Thunder': 'Oklahoma City Thunder',
            'Oklahoma City Thunder': 'Oklahoma City Thunder',
            'Thunder': 'Oklahoma City Thunder',
            
            'PHX Suns': 'Phoenix Suns',
            'Phoenix Suns': 'Phoenix Suns',
            'Suns': 'Phoenix Suns',
            
            'Utah Jazz': 'Utah Jazz',
            'Jazz': 'Utah Jazz',
            'UTAH': 'Utah Jazz',
            
            'Portland Trail Blazers': 'Portland Trail Blazers',
            'Trail Blazers': 'Portland Trail Blazers',
            'Blazers': 'Portland Trail Blazers',
            'POR': 'Portland Trail Blazers',
            
            'Denver Nuggets': 'Denver Nuggets',
            'Nuggets': 'Denver Nuggets',
            'DEN': 'Denver Nuggets',
            
            'Minnesota Timberwolves': 'Minnesota Timberwolves',
            'Timberwolves': 'Minnesota Timberwolves',
            'Wolves': 'Minnesota Timberwolves',
            'MIN': 'Minnesota Timberwolves',
            
            'Milwaukee Bucks': 'Milwaukee Bucks',
            'Bucks': 'Milwaukee Bucks',
            'MIL': 'Milwaukee Bucks',
            
            'Chicago Bulls': 'Chicago Bulls',
            'Bulls': 'Chicago Bulls',
            'CHI': 'Chicago Bulls',
            
            'Cleveland Cavaliers': 'Cleveland Cavaliers',
            'Cavaliers': 'Cleveland Cavaliers',
            'Cavs': 'Cleveland Cavaliers',
            'CLE': 'Cleveland Cavaliers',
            
            'Boston Celtics': 'Boston Celtics',
            'Celtics': 'Boston Celtics',
            'BOS': 'Boston Celtics',
            
            'Brooklyn Nets': 'Brooklyn Nets',
            'Nets': 'Brooklyn Nets',
            'BKN': 'Brooklyn Nets',
            
            'Philadelphia 76ers': 'Philadelphia 76ers',
            '76ers': 'Philadelphia 76ers',
            'Sixers': 'Philadelphia 76ers',
            'PHI': 'Philadelphia 76ers',
            
            'Toronto Raptors': 'Toronto Raptors',
            'Raptors': 'Toronto Raptors',
            'TOR': 'Toronto Raptors',
            
            'Miami Heat': 'Miami Heat',
            'Heat': 'Miami Heat',
            'MIA': 'Miami Heat',
            
            'Orlando Magic': 'Orlando Magic',
            'Magic': 'Orlando Magic',
            'ORL': 'Orlando Magic',
            
            'Atlanta Hawks': 'Atlanta Hawks',
            'Hawks': 'Atlanta Hawks',
            'ATL': 'Atlanta Hawks',
            
            'Charlotte Hornets': 'Charlotte Hornets',
            'Hornets': 'Charlotte Hornets',
            'CHA': 'Charlotte Hornets',
            
            'Washington Wizards': 'Washington Wizards',
            'Wizards': 'Washington Wizards',
            'WAS': 'Washington Wizards',
            
            'Detroit Pistons': 'Detroit Pistons',
            'Pistons': 'Detroit Pistons',
            'DET': 'Detroit Pistons',
            
            'Indiana Pacers': 'Indiana Pacers',
            'Pacers': 'Indiana Pacers',
            'IND': 'Indiana Pacers',
            
            'Memphis Grizzlies': 'Memphis Grizzlies',
            'Grizzlies': 'Memphis Grizzlies',
            'MEM': 'Memphis Grizzlies',
            
            'New Orleans Pelicans': 'New Orleans Pelicans',
            'Pelicans': 'New Orleans Pelicans',
            'NO': 'New Orleans Pelicans',
            'NOP': 'New Orleans Pelicans',
            
            'Houston Rockets': 'Houston Rockets',
            'Rockets': 'Houston Rockets',
            'HOU': 'Houston Rockets',
            
            'Dallas Mavericks': 'Dallas Mavericks',
            'Mavericks': 'Dallas Mavericks',
            'Mavs': 'Dallas Mavericks',
            'DAL': 'Dallas Mavericks',
            
            'Sacramento Kings': 'Sacramento Kings',
            'Kings': 'Sacramento Kings',
            'SAC': 'Sacramento Kings',
            
            'Phoenix Suns': 'Phoenix Suns',  # Duplicate but keeping for completeness
            
            # Odds API formats
            'LA Clippers': 'LA Clippers',
            'Los Angeles Lakers': 'Los Angeles Lakers',
            'New York Knicks': 'New York Knicks',
            'San Antonio Spurs': 'San Antonio Spurs',
            'Golden State Warriors': 'Golden State Warriors',
            'Oklahoma City Thunder': 'Oklahoma City Thunder'
        }
        
        # Feature scaling parameters (for normalization)
        self.scaling_params = {
            'points': {'mean': 110.5, 'std': 12.3},  # NBA averages
            'pace': {'mean': 100.0, 'std': 5.0},
            'efficiency': {'mean': 110.0, 'std': 8.0}
        }
    
    def normalize_team_name(self, name: str) -> str:
        """
        Normalize team names from various sources to standard format
        """
        if pd.isna(name) or not name:
            return None
        
        # Clean the name
        name = str(name).strip()
        name = re.sub(r'\s+', ' ', name)  # Remove extra spaces
        
        # Direct mapping
        if name in self.team_name_mapping:
            return self.team_name_mapping[name]
        
        # Try case-insensitive partial matching
        name_lower = name.lower()
        for key, standard in self.team_name_mapping.items():
            if key.lower() in name_lower or name_lower in key.lower():
                return standard
        
        # If no match found, return cleaned name (log warning)
        logger.warning(f"Unmapped team name: {name}")
        return name
    
    def normalize_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize scores to handle outliers and scale features
        """
        df = df.copy()
        
        # Clip extreme values (NBA scores typically 70-150)
        if 'home_score' in df.columns:
            df['home_score'] = df['home_score'].clip(70, 150)
        if 'away_score' in df.columns:
            df['away_score'] = df['away_score'].clip(70, 150)
        
        # total_score_normalized is intentionally NOT computed here.
        # It is a direct linear transformation of the target and was causing
        # near-zero MAE leakage when included as a training feature.
        # Feature engineering handles all normalization internally.
        
        return df
    
    def validate_game_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean game data
        """
        df = df.copy()
        
        # Remove duplicate games
        if 'date' in df.columns and 'home_team' in df.columns and 'away_team' in df.columns:
            df = df.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='last')
        
        # Remove games with missing critical data
        required_cols = ['home_team', 'away_team']
        for col in required_cols:
            if col in df.columns:
                df = df.dropna(subset=[col])
        
        # Normalize team names
        if 'home_team' in df.columns:
            df['home_team'] = df['home_team'].apply(self.normalize_team_name)
        if 'away_team' in df.columns:
            df['away_team'] = df['away_team'].apply(self.normalize_team_name)
        
        # Remove rows where team names couldn't be normalized
        df = df.dropna(subset=['home_team', 'away_team'])
        
        return df
    
    def scrape_historical_data(self, seasons: int = 5) -> bool:
        """
        Fetch historical NBA data using ESPN's JSON scoreboard API.
        NBA season runs October through June, so season_year is the year it ends.

        Season logic (fixed):
          - The current in-progress season is ALWAYS included as season N.
          - If we are in Oct-Dec, the season that started this October ends NEXT year.
          - If we are in Jan-Sep, the season in progress started last October and ends this year.
          - 'seasons' controls how many seasons back to fetch in total.
        """
        try:
            now = datetime.now()

            # Determine the season_year for the season currently in progress.
            if now.month >= 10:
                current_season_year = now.year + 1  # e.g. Oct 2025 -> season ends 2026
            else:
                current_season_year = now.year      # e.g. Feb 2026 -> season ends 2026

            print(f"\n{'='*60}")
            print(f"  NBA Historical Data Fetch")
            print(f"  Current season: {current_season_year-1}-{str(current_season_year)[2:]}")
            print(f"  Fetching {seasons} season(s) back from current")
            print(f"{'='*60}\n")

            all_games = []
            total_games_fetched = 0

            for i in range(seasons):
                season_year = current_season_year - i
                season_label = f"{season_year - 1}-{str(season_year)[2:]}"
                season_games = self._fetch_season(season_year, season_label)

                if season_games:
                    season_df = pd.DataFrame(season_games)
                    season_df = self.validate_game_data(season_df)
                    season_df = self.normalize_scores(season_df)
                    valid_count = len(season_df)
                    all_games.extend(season_df.to_dict('records'))
                    total_games_fetched += valid_count
                    print(f"\n  \u2713 Season {season_label} complete \u2014 {valid_count} valid games "
                          f"(running total: {total_games_fetched})\n")
                else:
                    print(f"\n  \u2717 Season {season_label} \u2014 no games found\n")

            if all_games:
                print(f"{'='*60}")
                print(f"  Saving {total_games_fetched} games to cache...")
                df = pd.DataFrame(all_games)
                df = self.validate_game_data(df)
                self.cache_manager.save_historical_data(df)
                all_teams = sorted(pd.concat([df['home_team'], df['away_team']]).unique())
                print(f"  \u2713 Saved {len(df)} games across {len(all_teams)} teams")
                print(f"{'='*60}\n")
                logger.info(f"Historical fetch complete: {len(df)} games saved")
                return True

            print("  \u2717 No games fetched across any season \u2014 check network connection")
            logger.warning("No games were fetched across all seasons")
            return False

        except Exception as e:
            print(f"\n  \u2717 Historical data fetch failed: {e}")
            logger.error(f"Historical data fetch failed: {str(e)}")
            return False

    def update_historical_data(self) -> bool:
        """
        Incrementally fetch only the dates missing from the cache.

        If the gap between the latest cached date and today spans an off-season
        (July-September), skip straight to October 1st of the next season rather
        than crawling through months with no NBA games.
        Falls back to a full single-season scrape if the cache is empty.
        """
        try:
            existing_df = self.cache_manager.load_historical_data()
            yesterday = (datetime.now() - timedelta(days=1)).date()

            if existing_df.empty:
                logger.info("No cached data found — running single-season scrape")
                print("  No cached data found. Fetching current season only...")
                return self.scrape_historical_data(seasons=1)

            existing_df['date'] = pd.to_datetime(existing_df['date'])
            latest_cached = existing_df['date'].max().date()

            if latest_cached >= yesterday:
                logger.info("Cache is already up to date")
                print(f"  \u2713 Cache is already up to date (latest: {latest_cached})")
                return True

            raw_start = latest_cached + timedelta(days=1)

            # Skip the off-season (July-September has no NBA games).
            # If raw_start lands in Jul/Aug/Sep, jump straight to Oct 1.
            if raw_start.month in (7, 8, 9):
                next_oct = raw_start.replace(month=10, day=1)
                if next_oct <= yesterday:
                    print(f"  \u2192 Off-season gap — jumping from {raw_start} to {next_oct} "
                          f"(skipping {(next_oct - raw_start).days} off-season days)")
                    logger.info(f"Skipping off-season: {raw_start} -> {next_oct}")
                    start_date = next_oct
                else:
                    print(f"  \u2713 Season ended and next season has not started yet. Nothing to fetch.")
                    return True
            else:
                start_date = raw_start

            total_days = (yesterday - start_date).days + 1

            print(f"\n{'='*60}")
            print(f"  NBA Incremental Update")
            print(f"  Fetching {total_days} day(s): {start_date} \u2192 {yesterday}")
            print(f"{'='*60}\n")

            new_games = []
            current = datetime(start_date.year, start_date.month, start_date.day)
            end_dt = datetime(yesterday.year, yesterday.month, yesterday.day)
            day_num = 0

            while current <= end_dt:
                date_str = current.strftime("%Y%m%d")
                day_games = self._fetch_date(date_str)
                if day_games:
                    new_games.extend(day_games)

                day_num += 1
                if day_num % 10 == 0 or current.date() == yesterday:
                    pct = int(day_num / total_days * 100)
                    bar = "\u2588" * int(pct / 5) + "\u2591" * (20 - int(pct / 5))
                    print(f"     [{bar}] {pct:3d}%  day {day_num}/{total_days}  "
                          f"{len(new_games)} new games found")

                current += timedelta(days=1)
                time.sleep(0.25)

            if new_games:
                new_df = pd.DataFrame(new_games)
                new_df = self.validate_game_data(new_df)
                new_df = self.normalize_scores(new_df)
                self.cache_manager.save_historical_data(new_df)
                print(f"\n  \u2713 Appended {len(new_df)} new games to cache\n")
                logger.info(f"Incremental update complete: {len(new_df)} new games added")
            else:
                print(f"\n  \u2713 No completed games found in date range\n")
                logger.info("Incremental update: no new games to add")

            return True

        except Exception as e:
            logger.error(f"Incremental update failed: {str(e)}")
            print(f"\n  \u2717 Incremental update failed: {e}\n")
            return False

    def _fetch_date(self, date_str: str) -> List[Dict]:
        """
        Fetch completed NBA games for a single date from ESPN's JSON API.
        date_str is YYYYMMDD. Returns a list of game dicts.
        Mirrors _fetch_espn_date() from the working NCAA scraper exactly,
        adapted for the NBA endpoint and NBA score validation ranges.
        """
        params = {"dates": date_str, "limit": 20}
        max_retries = 3
        data = None

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    self.espn_api_url,
                    params=params,
                    headers=self.headers,
                    timeout=20
                )
                response.raise_for_status()
                data = response.json()
                break
            except requests.exceptions.ConnectionError:
                msg = f"Network error on {date_str} (attempt {attempt+1}/{max_retries}) — no connection"
                print(f"    ⚠ {msg}")
                logger.warning(msg)
                if attempt == max_retries - 1:
                    print(f"    ✗ Giving up on {date_str} after {max_retries} attempts")
                    return []
                time.sleep(3 * (attempt + 1))
            except requests.exceptions.Timeout:
                msg = f"Timeout on {date_str} (attempt {attempt+1}/{max_retries})"
                print(f"    ⚠ {msg}")
                logger.warning(msg)
                if attempt == max_retries - 1:
                    return []
                time.sleep(3 * (attempt + 1))
            except Exception as e:
                msg = f"ESPN API error on {date_str} (attempt {attempt+1}/{max_retries}): {e}"
                print(f"    ⚠ {msg}")
                logger.warning(msg)
                if attempt == max_retries - 1:
                    return []
                time.sleep(3 * (attempt + 1))

        if data is None:
            return []

        NBA_TEAMS = {
            "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
            "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
            "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
            "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
            "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans",
            "New York Knicks", "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers",
            "Phoenix Suns", "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs",
            "Toronto Raptors", "Utah Jazz", "Washington Wizards"
        }
        records = []
        for event in data.get("events", []):
            # Only completed games
            status = event.get("status", {}).get("type", {})
            if not status.get("completed", False):
                continue

            comps = event.get("competitions", [{}])
            if not comps:
                continue
            comp = comps[0]

            # ── OT detection (fixed) ──────────────────────────────────────────
            # status.type.detail explicitly says "Final/OT" for OT games.
            # status.type is at the EVENT level, not comp level.
            # We check both detail string AND period > 4 as a belt-and-suspenders.
            status_type   = event.get("status", {}).get("type", {})
            detail        = status_type.get("detail", "")
            period        = event.get("status", {}).get("period", 4)
            if "OT" in detail or period > 4:
                logger.info(f"Skipping OT game on {date_str} (detail='{detail}', period={period})")
                continue

            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue

            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if home is None or away is None:
                continue

            try:
                home_score = float(home.get("score", 0) or 0)
                away_score = float(away.get("score", 0) or 0)
            except (ValueError, TypeError):
                continue

            # Skip placeholder/cancelled games
            if home_score == 0 and away_score == 0:
                continue

            # Validate NBA score ranges (record low: 49, realistic ceiling: 200)
            NBA_SCORE_MIN, NBA_SCORE_MAX = 50, 200
            suspicious = []
            if not (NBA_SCORE_MIN <= home_score <= NBA_SCORE_MAX):
                suspicious.append(f"home={home_score}")
            if not (NBA_SCORE_MIN <= away_score <= NBA_SCORE_MAX):
                suspicious.append(f"away={away_score}")
            if suspicious:
                home_name = home.get("team", {}).get("displayName", "?")
                away_name = away.get("team", {}).get("displayName", "?")
                logger.warning(
                    f"Suspicious score(s) [{', '.join(suspicious)}] for "
                    f"{away_name} vs {home_name} on {date_str} — skipped"
                )
                continue

            date_fmt  = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            game_date = datetime.strptime(date_fmt, "%Y-%m-%d")

            home_team = self.normalize_team_name(
                home.get("team", {}).get("displayName", ""))
            away_team = self.normalize_team_name(
                away.get("team", {}).get("displayName", ""))

            if home_team not in NBA_TEAMS or away_team not in NBA_TEAMS:
                print(f"    ⚠ Skipping non-NBA game: {away_team} vs {home_team} ({date_str})")
                continue

            # ── Helper: extract a stat value by name from statistics list ─────
            def _stat(stats_list, name, default=None):
                for s in stats_list:
                    if s.get("name") == name:
                        try:
                            return float(s.get("displayValue", default))
                        except (TypeError, ValueError):
                            return default
                return default

            # ── Team-level box score stats (same API response, no extra call) ─
            home_stats = home.get("statistics", [])
            away_stats = away.get("statistics", [])

            home_fg_pct  = _stat(home_stats, "fieldGoalPct")
            away_fg_pct  = _stat(away_stats, "fieldGoalPct")
            home_3p_pct  = _stat(home_stats, "threePointPct")
            away_3p_pct  = _stat(away_stats, "threePointPct")
            home_ft_pct  = _stat(home_stats, "freeThrowPct")
            away_ft_pct  = _stat(away_stats, "freeThrowPct")
            home_reb     = _stat(home_stats, "rebounds")
            away_reb     = _stat(away_stats, "rebounds")
            home_ast     = _stat(home_stats, "assists")
            away_ast     = _stat(away_stats, "assists")
            home_fga     = _stat(home_stats, "fieldGoalsAttempted")
            away_fga     = _stat(away_stats, "fieldGoalsAttempted")
            home_fta     = _stat(home_stats, "freeThrowsAttempted")
            away_fta     = _stat(away_stats, "freeThrowsAttempted")
            home_3pa     = _stat(home_stats, "threePointFieldGoalsAttempted")
            away_3pa     = _stat(away_stats, "threePointFieldGoalsAttempted")

            # ── Quarter scores (linescores) ───────────────────────────────────
            def _quarters(competitor):
                qs = {ls["period"]: ls["value"]
                      for ls in competitor.get("linescores", [])
                      if ls.get("period", 0) <= 4}
                return [qs.get(p) for p in range(1, 5)]

            home_q1, home_q2, home_q3, home_q4 = _quarters(home)
            away_q1, away_q2, away_q3, away_q4 = _quarters(away)

            # ── Game-level metadata ───────────────────────────────────────────
            attendance  = comp.get("attendance")
            neutral     = comp.get("neutralSite", False)
            # National TV: check broadcasts for ESPN/ABC/TNT/NBC
            nat_tv_nets = {"ESPN", "ABC", "TNT", "NBC", "ESPN2", "ESPNU"}
            broadcasts  = comp.get("broadcasts", [])
            national_tv = int(any(
                name in nat_tv_nets
                for b in broadcasts
                for name in b.get("names", [])
                if b.get("market") == "national"
            ))

            records.append({
                "date":         date_fmt,
                "home_team":    home_team,
                "away_team":    away_team,
                "home_score":   int(home_score),
                "away_score":   int(away_score),
                "total_score":  int(home_score + away_score),
                "season":       game_date.year if game_date.month >= 10 else game_date.year - 1,
                "espn_id":      event.get("id", ""),
                # Box score stats
                "home_fg_pct":  home_fg_pct,
                "away_fg_pct":  away_fg_pct,
                "home_3p_pct":  home_3p_pct,
                "away_3p_pct":  away_3p_pct,
                "home_ft_pct":  home_ft_pct,
                "away_ft_pct":  away_ft_pct,
                "home_reb":     home_reb,
                "away_reb":     away_reb,
                "home_ast":     home_ast,
                "away_ast":     away_ast,
                "home_fga":     home_fga,
                "away_fga":     away_fga,
                "home_fta":     home_fta,
                "away_fta":     away_fta,
                "home_3pa":     home_3pa,
                "away_3pa":     away_3pa,
                # Quarter scores
                "home_q1":      home_q1,
                "home_q2":      home_q2,
                "home_q3":      home_q3,
                "home_q4":      home_q4,
                "away_q1":      away_q1,
                "away_q2":      away_q2,
                "away_q3":      away_q3,
                "away_q4":      away_q4,
                # Game metadata
                "attendance":   attendance,
                "neutral_site": int(neutral),
                "national_tv":  national_tv,
            })

        return records

    def _fetch_season(self, season_year: int, season_label: str = None) -> List[Dict]:
        """
        Iterate every date in an NBA season and collect completed games.
        season_year is the year the season ends (Oct prev_year → Jun season_year).
        Logs progress every 10 days in the style of the working NCAA scraper.
        """
        if season_label is None:
            season_label = f"{season_year-1}-{str(season_year)[2:]}"

        start = datetime(season_year - 1, 10, 1)
        end = min(
            datetime(season_year, 6, 30),
            datetime.now() - timedelta(days=1)
        )

        all_games = []
        current = start
        total_days = (end - start).days + 1
        day_num = 0
        consecutive_errors = 0

        print(f"  ── Season {season_label} ──────────────────────────────────")
        print(f"     Date range: {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')} "
              f"({total_days} days)")

        while current <= end:
            date_str = current.strftime("%Y%m%d")
            day_games = self._fetch_date(date_str)

            if day_games is None:
                # None signals a hard network failure from _fetch_date
                consecutive_errors += 1
            else:
                consecutive_errors = 0
                all_games.extend(day_games)

            # Alert if we're seeing sustained network failures
            if consecutive_errors >= 5:
                print(f"    ✗ {consecutive_errors} consecutive network failures — "
                      f"check your internet connection")
                logger.error(f"Season {season_label}: {consecutive_errors} consecutive fetch failures")
                consecutive_errors = 0  # reset so we don't spam

            day_num += 1

            # Progress print every 10 days — matches the style of your NCAA scraper
            if day_num % 10 == 0 or current == end:
                pct = int(day_num / total_days * 100)
                bar_filled = int(pct / 5)
                bar = "█" * bar_filled + "░" * (20 - bar_filled)
                print(f"     [{bar}] {pct:3d}%  day {day_num}/{total_days}  "
                      f"{len(all_games)} games collected")

            current += timedelta(days=1)
            time.sleep(0.25)  # polite — same rate as working NCAA scraper

        print(f"     Done — {len(all_games)} raw games for {season_label}")
        return all_games

    def update_yesterday(self) -> pd.DataFrame:
        """
        Fetch yesterday's completed games and append to the historical cache.
        Call this at app startup to keep data current without a full rescrape.
        """
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        games = self._fetch_date(yesterday)

        if not games:
            logger.info("No completed NBA games from yesterday")
            return pd.DataFrame()

        df = pd.DataFrame(games)
        df = self.validate_game_data(df)
        df = self.normalize_scores(df)

        if not df.empty:
            self.cache_manager.save_historical_data(df)
            logger.info(f"Appended {len(df)} games from yesterday to cache")

        return df
    

    def fetch_injury_data(self) -> dict:
        """
        Fetch current NBA injury report from ESPN's injury endpoint.

        Returns a dict keyed by NORMALIZED team name so it can be joined
        directly against the home_team / away_team columns in games_df:

            {
              'Boston Celtics': {
                  'out':            2,   # confirmed Out
                  'doubtful':       1,
                  'questionable':   2,
                  'key_player_out': 1,   # starter (G/F/C) confirmed Out
                  'notes':          ['🔴 Jaylen Brown (Knee) - Out', ...]
              },
              ...
            }

        Falls back to {} on any network/parse error — predictions are
        never blocked by an injury-fetch failure.
        """
        injury_map = {}
        try:
            response = requests.get(
                self.espn_injury_url,
                headers=self.headers,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            # ESPN structures this as a top-level dict of team-slug -> team block.
            # Each block has a 'team' sub-object and an 'injuries' list.
            for _slug, team_block in data.items():
                if not isinstance(team_block, dict):
                    continue

                team_info = team_block.get('team', {})
                raw_name  = (team_info.get('displayName')
                             or team_info.get('shortDisplayName')
                             or team_info.get('name', ''))
                team_name = self.normalize_team_name(raw_name)
                if not team_name:
                    continue

                counts = {
                    'out': 0, 'doubtful': 0, 'questionable': 0,
                    'key_player_out': 0, 'notes': []
                }

                for entry in team_block.get('injuries', []):
                    athlete  = entry.get('athlete', {})
                    # ESPN uses 'status' at the top level of each injury entry
                    status   = (entry.get('status') or '').strip().lower()

                    # Injury detail — can be a dict or a plain string
                    detail   = entry.get('details', {})
                    if isinstance(detail, dict):
                        inj_type = detail.get('type', '')
                        location = detail.get('location', '')
                        side     = detail.get('side', '')
                    else:
                        inj_type = str(detail)
                        location = ''
                        side     = ''

                    player_name = (athlete.get('displayName')
                                   or athlete.get('shortName', 'Unknown'))

                    # Position tells us if this is a key/starter-calibre player
                    pos_info = athlete.get('position', {})
                    pos_abbr = (pos_info.get('abbreviation', '')
                                if isinstance(pos_info, dict) else '')
                    # All five main position codes treated as key players
                    is_key = pos_abbr in ('G', 'F', 'C', 'PG', 'SG', 'SF', 'PF')

                    inj_parts = [p for p in [side, location, inj_type] if p]
                    inj_label = ' '.join(inj_parts).strip()
                    note_body = f"{player_name}" + (f" ({inj_label})" if inj_label else "")

                    if status == 'out':
                        counts['out'] += 1
                        if is_key:
                            counts['key_player_out'] += 1
                        counts['notes'].append(f"🔴 {note_body} - Out")
                    elif status == 'doubtful':
                        counts['doubtful'] += 1
                        counts['notes'].append(f"🟠 {note_body} - Doubtful")
                    elif status in ('questionable', 'probable'):
                        counts['questionable'] += 1
                        counts['notes'].append(f"🟡 {note_body} - Questionable")

                injury_map[team_name] = counts

            logger.info(f"Injury data fetched for {len(injury_map)} teams")

        except Exception as e:
            logger.warning(f"Injury fetch failed (non-fatal): {e}")

        return injury_map

    def fetch_todays_games(self) -> pd.DataFrame:
        """
        Fetch today's scheduled NBA games from ESPN's scoreboard API.
        Returns both upcoming and in-progress games, with over/under line
        extracted from ESPN's odds field where available.
        Falls back to the Odds API if a key is configured and ESPN has no odds.
        No API key required for the schedule itself.
        """
        try:
            # Check cache first — valid for today only
            cached = self.cache_manager.load_todays_games()
            if cached is not None:
                logger.info("Loading today's games from cache")
                return self.validate_game_data(cached)

            today_str = datetime.now().strftime('%Y%m%d')
            today_fmt = datetime.now().strftime('%Y-%m-%d')

            params = {"dates": today_str, "limit": 20}
            response = requests.get(
                self.espn_api_url, params=params,
                headers=self.headers, timeout=20
            )
            response.raise_for_status()
            data = response.json()

            games = []
            for event in data.get("events", []):
                comps = event.get("competitions", [{}])
                if not comps:
                    continue
                comp = comps[0]

                competitors = comp.get("competitors", [])
                if len(competitors) < 2:
                    continue

                home = next((c for c in competitors if c.get("homeAway") == "home"), None)
                away = next((c for c in competitors if c.get("homeAway") == "away"), None)
                if home is None or away is None:
                    continue

                home_team = self.normalize_team_name(
                    home.get("team", {}).get("displayName", ""))
                away_team = self.normalize_team_name(
                    away.get("team", {}).get("displayName", ""))

                if not home_team or not away_team:
                    continue

                # Parse commence time
                commence_time = event.get("date", "")
                game_time = self._format_game_time(commence_time)

                game_data = {
                    'game_id':      event.get("id", ""),
                    'date':         today_fmt,
                    'away_team':    away_team,
                    'home_team':    home_team,
                    'commence_time': commence_time,
                    'game_time':    game_time,
                    'over_line':    None,
                    'under_line':   None,
                    'over_price':   None,
                    'under_price':  None,
                }

                # ESPN includes odds in competitions[0].odds for many games
                # Structure: {"details": "...", "overUnder": 224.5, ...}
                odds_list = comp.get("odds", [])
                if odds_list:
                    odds = odds_list[0]
                    over_under = odds.get("overUnder") or odds.get("total", {}).get("alternateDisplayValue")
                    if over_under:
                        try:
                            line = float(over_under)
                            game_data['over_line']  = line
                            game_data['under_line'] = line
                            # ESPN sometimes provides over/under prices
                            game_data['over_price']  = odds.get("overOdds", -110)
                            game_data['under_price'] = odds.get("underOdds", -110)
                        except (ValueError, TypeError):
                            pass

                games.append(game_data)

            df = pd.DataFrame(games) if games else pd.DataFrame()

            # If ESPN had no odds and we have a real Odds API key, enrich with it
            if not df.empty and df['over_line'].isna().all() and not self.demo_mode:
                logger.info("ESPN had no odds data — enriching from Odds API")
                df = self._enrich_with_odds_api(df, today_fmt)

            if not df.empty:
                df = self.validate_game_data(df)

                # ── Enrich each game row with current injury data ──────────────
                # fetch_injury_data() returns {} on failure so this is always safe.
                injury_map = self.fetch_injury_data()
                if injury_map:
                    for side in ('home', 'away'):
                        team_col = f'{side}_team'
                        df[f'{side}_inj_out']         = df[team_col].map(
                            lambda t: injury_map.get(t, {}).get('out', 0))
                        df[f'{side}_inj_doubtful']    = df[team_col].map(
                            lambda t: injury_map.get(t, {}).get('doubtful', 0))
                        df[f'{side}_inj_questionable']= df[team_col].map(
                            lambda t: injury_map.get(t, {}).get('questionable', 0))
                        df[f'{side}_key_player_out']  = df[team_col].map(
                            lambda t: injury_map.get(t, {}).get('key_player_out', 0))
                        # Store notes as a pipe-separated string (CSV-safe)
                        df[f'{side}_inj_notes'] = df[team_col].map(
                            lambda t: ' | '.join(
                                injury_map.get(t, {}).get('notes', [])
                            )
                        )
                    logger.info("Injury data merged into today\'s games")
                else:
                    # No injury data — initialise columns to clean defaults
                    for side in ('home', 'away'):
                        df[f'{side}_inj_out']          = 0
                        df[f'{side}_inj_doubtful']     = 0
                        df[f'{side}_inj_questionable'] = 0
                        df[f'{side}_key_player_out']   = 0
                        df[f'{side}_inj_notes']        = ''

                self.cache_manager.save_todays_games(df)
                logger.info(f"Fetched {len(df)} games for today from ESPN API")

            return df if not df.empty else pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to fetch today's games: {str(e)}")
            return pd.DataFrame()

    def _enrich_with_odds_api(self, games_df: pd.DataFrame, today: str) -> pd.DataFrame:
        """
        Enrich an existing games DataFrame with over/under lines from the Odds API.
        Only called when ESPN doesn't carry odds and a real API key is available.
        """
        try:
            url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
            params = {
                'apiKey': self.odds_api_key,
                'regions': 'us',
                'markets': 'totals',
                'oddsFormat': 'american',
                'date': today
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Build lookup: (away_team, home_team) -> odds
            odds_lookup = {}
            for game in data:
                away = self.normalize_team_name(game.get('away_team', ''))
                home = self.normalize_team_name(game.get('home_team', ''))
                for bookmaker in game.get('bookmakers', []):
                    if bookmaker['key'] == 'draftkings':
                        for market in bookmaker.get('markets', []):
                            if market['key'] == 'totals':
                                entry = {}
                                for outcome in market.get('outcomes', []):
                                    if outcome['name'] == 'Over':
                                        entry['over_line']  = float(outcome['point'])
                                        entry['over_price'] = outcome['price']
                                    else:
                                        entry['under_line']  = float(outcome['point'])
                                        entry['under_price'] = outcome['price']
                                if entry:
                                    odds_lookup[(away, home)] = entry

            for idx, row in games_df.iterrows():
                key = (row['away_team'], row['home_team'])
                if key in odds_lookup:
                    for col, val in odds_lookup[key].items():
                        games_df.at[idx, col] = val

            logger.info(f"Enriched {sum(games_df['over_line'].notna())} games with Odds API lines")
        except Exception as e:
            logger.warning(f"Odds API enrichment failed: {e}")

        return games_df
    
    def _format_game_time(self, commence_time: str) -> str:
        """Format game time for display"""
        try:
            dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
            return dt.strftime('%I:%M %p ET')
        except:
            return "Time TBD"
    
    def fetch_results_for_date(self, date_str: str) -> pd.DataFrame:
        """
        Fetch actual final scores for a completed date (YYYY-MM-DD format).
        Used by the backtesting engine to compare predictions to results.
        Returns a DataFrame with home_team, away_team, actual_home, actual_away,
        actual_total — or empty DataFrame if no completed games found.
        """
        try:
            espn_date = date_str.replace('-', '')
            games = self._fetch_date(espn_date)
            if not games:
                return pd.DataFrame()
            df = pd.DataFrame(games)
            df = df.rename(columns={
                'home_score': 'actual_home',
                'away_score': 'actual_away',
                'total_score': 'actual_total',
            })
            return df[['date', 'home_team', 'away_team',
                        'actual_home', 'actual_away', 'actual_total']]
        except Exception as e:
            logger.warning(f"Could not fetch results for {date_str}: {e}")
            return pd.DataFrame()

    def load_historical_data(self) -> pd.DataFrame:
        """Load and validate historical data from cache"""
        df = self.cache_manager.load_historical_data()
        if not df.empty:
            df = self.validate_game_data(df)
            df = self.normalize_scores(df)
        return df
    
    def get_team_list(self) -> List[str]:
        """Get list of all normalized team names"""
        historical = self.load_historical_data()
        if not historical.empty:
            teams = pd.concat([historical['home_team'], historical['away_team']]).unique()
            return sorted(teams)
        return sorted(list(set(self.team_name_mapping.values())))
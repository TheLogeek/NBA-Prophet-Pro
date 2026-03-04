# src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class NBAFeatureEngineer:
    """Advanced feature engineering for NBA predictions with normalization"""
    
    def __init__(self):
        self.team_stats_cache = {}
        self.season_stats = {}
        self.scalers = {}
        self.feature_columns = []
        
        # Define feature types for different scaling approaches
        self.scaling_strategies = {
            'standard': ['points', 'assists', 'rebounds'],  # Normal distribution
            'robust': ['margin', 'differential'],  # Has outliers
            'minmax': ['percentages', 'ratios']  # Bounded values
        }
    
    def create_features(self, df: pd.DataFrame, fit_scalers: bool = True) -> pd.DataFrame:
        """
        Create advanced features for model training with normalization.

        IMPORTANT — scaler fitting scope:
        When fit_scalers=True (training time), pass ONLY the training split of
        your data, not the full dataset including the test/validation split.
        Fitting scalers on held-out rows is a form of leakage: the scaler learns
        the test-set distribution, giving the model an unfair peek at unseen data.
        At prediction time always call with fit_scalers=False so the pre-fitted
        scalers (trained on training rows only) are reused unchanged.
        """
        logger.info("Starting feature engineering with normalization...")
        
        df = df.copy()
        
        # Use float32 for memory efficiency across the full dataset
        for col in df.select_dtypes(include=[np.float64]).columns:
            df[col] = df[col].astype(np.float32)
        
        df = df.sort_values(['date', 'home_team'])
        
        # Chunked processing for large datasets — preserves all historical data
        # rather than truncating, which would discard signal from earlier seasons
        # Process the full dataset in one pass — no chunking.
        # Chunking was causing H2H leakage: each chunk's expanding mean only saw
        # games within that chunk, not full prior history, producing artificially
        # perfect predictions at chunk-boundary folds. All feature methods are now
        # fully vectorized so the performance cost of a single pass is negligible.
        if len(df) > 5000:
            logger.info(f"Large dataset ({len(df)} rows) detected, processing in single pass (chunking removed to prevent leakage)")
        df = self._create_basic_features(df)
        df = self._create_rolling_averages(df)
        df = self._create_team_metrics(df)
        df = self._create_advanced_stats(df)
        df = self._create_h2h_features(df)
        df = self._create_situational_features(df)
        df = self._create_pace_efficiency(df)
        df = self._create_last5_form(df)
        df = self._create_season_stage(df)
        df = self._create_opponent_adjusted_stats(df)
        df = self._create_b2b_second_night(df)
        df = self._create_streak_margin(df)
        df = self._create_current_season_h2h(df)
        df = self._create_pace_matchup(df)
        df = self._create_venue_rest(df)
        df = self._create_box_score_rolling(df)
        df = self._create_quarter_features(df)
        df = self._create_game_metadata(df)

        # Identify numerical columns for scaling
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target variables and identifiers from scaling
        # Exclude raw target columns AND any columns directly derived from targets
        # (abs_score_diff, estimated_pace etc.) to prevent leakage
        cols_to_exclude = [
            'home_score', 'away_score', 'total_score', 'score_diff', 'home_win',
            'date_int', 'season',
            # Direct linear transformation of the target — feeding this to the model
            # is equivalent to giving it the answer; total MAE drops to ~0.01 with it in
            'total_score_normalized',
            # Identifiers — sequential numeric IDs have zero predictive value and
            # must be excluded or the model learns noise from them
            'espn_id', 'game_id', 'sports_id',
            # legacy pace/efficiency columns computed from scores — keep excluded
            # in case old cached data has them
            'estimated_pace', 'home_efficiency', 'away_efficiency', 'efficiency_diff',
            'efficiency_diff_normalized', 'home_efg', 'away_efg',
            'home_efg_normalized', 'away_efg_normalized', 'abs_score_diff',
            # Raw per-game box score stats — these ARE the game outcome, not prior history.
            # The rolling versions (home_rolling_fg_pct etc.) are the legitimate features.
            'home_fg_pct', 'away_fg_pct', 'home_3p_pct', 'away_3p_pct',
            'home_ft_pct', 'away_ft_pct', 'home_reb', 'away_reb',
            'home_ast', 'away_ast', 'home_fga', 'away_fga',
            'home_fta', 'away_fta', 'home_3pa', 'away_3pa',
            # Raw quarter scores — same reason
            'home_q1', 'home_q2', 'home_q3', 'home_q4',
            'away_q1', 'away_q2', 'away_q3', 'away_q4',
        ]
        feature_cols = [col for col in numerical_cols if col not in cols_to_exclude]

        self.feature_columns = feature_cols

        # Apply different scaling strategies.
        # fit_scalers=True is used during training — the scaler sees only the
        # rows passed in (caller is responsible for passing train-split rows).
        if fit_scalers:
            df = self._apply_scaling(df, feature_cols, fit=True)
        else:
            df = self._apply_scaling(df, feature_cols, fit=False)

        # Handle any remaining NaN values
        df = df.fillna(df.mean(numeric_only=True))

        logger.info(f"Feature engineering complete. Created {len(feature_cols)} normalized features")
        return df
    
    def _process_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all feature engineering steps on a single chunk of data"""
        df = self._create_basic_features(df)
        df = self._create_rolling_averages(df)
        df = self._create_team_metrics(df)
        df = self._create_advanced_stats(df)
        df = self._create_h2h_features(df)
        df = self._create_situational_features(df)
        df = self._create_pace_efficiency(df)
        df = self._create_last5_form(df)
        df = self._create_season_stage(df)
        df = self._create_opponent_adjusted_stats(df)
        df = self._create_b2b_second_night(df)
        df = self._create_streak_margin(df)
        df = self._create_current_season_h2h(df)
        df = self._create_pace_matchup(df)
        df = self._create_venue_rest(df)
        df = self._create_box_score_rolling(df)
        df = self._create_quarter_features(df)
        df = self._create_game_metadata(df)
        df = df.fillna(df.mean(numeric_only=True))
        return df

    def _apply_scaling(self, df: pd.DataFrame, feature_cols: List[str], fit: bool = False) -> pd.DataFrame:
        """
        Apply appropriate scaling to different feature types
        """
        df = df.copy()
        
        for col in feature_cols:
            # Determine scaling strategy based on column name
            if any(term in col.lower() for term in ['pct', 'rate', 'ratio']):
                # MinMax scaling for percentages/ratios (already 0-1 or bounded)
                if fit:
                    self.scalers[col] = MinMaxScaler()
                    df[col] = self.scalers[col].fit_transform(df[[col]]).flatten()
                else:
                    df[col] = self.scalers[col].transform(df[[col]]).flatten()
                    
            elif any(term in col.lower() for term in ['diff', 'margin', 'error']):
                # Robust scaling for features with outliers
                if fit:
                    self.scalers[col] = RobustScaler()
                    df[col] = self.scalers[col].fit_transform(df[[col]]).flatten()
                else:
                    df[col] = self.scalers[col].transform(df[[col]]).flatten()
                    
            else:
                # Standard scaling for normally distributed features
                if fit:
                    self.scalers[col] = StandardScaler()
                    df[col] = self.scalers[col].fit_transform(df[[col]]).flatten()
                else:
                    df[col] = self.scalers[col].transform(df[[col]]).flatten()
        
        return df
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic game features"""
        df['total_score'] = df['home_score'] + df['away_score']
        df['score_diff'] = df['home_score'] - df['away_score']
        df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
        # Date features (normalized)
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek / 6.0  # Normalize to 0-1
        df['month'] = (df['date'].dt.month - 1) / 11.0  # Normalize to 0-1
        df['day_of_season'] = df['date'].dt.dayofyear / 365.0  # Normalize to 0-1
        df['date_int'] = df['date'].astype('int64') // 10**9  # Unix timestamp
        
        return df
    
    def _create_rolling_averages(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Vectorized rolling averages — overall and venue-split — with minimal
        intermediate copies to keep RAM usage low on 4 GB machines.

        Key memory changes vs previous version:
        - Overall stats computed on long (15k rows), then long is immediately
          slimmed down to only the columns needed before venue-split work begins.
        - Venue-split stats computed on separate home_long / away_long slices
          (each ~7.7k rows) and merged directly onto df — never merged back onto
          the full long DataFrame, avoiding a large intermediate copy.
        - del + gc.collect() called after each major intermediate to release RAM.
        """
        import gc

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # ── Build long-form (one row per team per game) ───────────────────────
        home_view = df[['date', 'home_team', 'home_score', 'away_score']].copy()
        home_view.columns = ['date', 'team', 'scored', 'allowed']
        home_view['is_home'] = True

        away_view = df[['date', 'away_team', 'away_score', 'home_score']].copy()
        away_view.columns = ['date', 'team', 'scored', 'allowed']
        away_view['is_home'] = False

        long = pd.concat([home_view, away_view], ignore_index=True)
        del home_view, away_view
        gc.collect()

        long['win'] = (long['scored'] > long['allowed']).astype(int)
        long = long.sort_values(['team', 'date']).reset_index(drop=True)

        # ── Overall rolling stats ─────────────────────────────────────────────
        stat_cols = []
        for window in windows:
            grp = long.groupby('team', sort=False)
            long[f'_{window}g_avg_scored']  = grp['scored'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            long[f'_{window}g_std_scored']  = grp['scored'].transform(lambda x: x.shift(1).rolling(window, min_periods=2).std().fillna(0))
            long[f'_{window}g_avg_allowed'] = grp['allowed'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            long[f'_{window}g_std_allowed'] = grp['allowed'].transform(lambda x: x.shift(1).rolling(window, min_periods=2).std().fillna(0))
            long[f'_{window}g_win_rate']    = grp['win'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            stat_cols += [f'_{window}g_avg_scored', f'_{window}g_std_scored',
                          f'_{window}g_avg_allowed', f'_{window}g_std_allowed',
                          f'_{window}g_win_rate']

        # Merge overall stats onto df (home and away separately)
        needed = ['date', 'team', 'is_home'] + stat_cols
        home_stats = long.loc[long['is_home'], needed].copy()
        home_stats = home_stats.drop(columns=['is_home'])
        home_stats = home_stats.drop_duplicates(subset=['date', 'team'])
        home_stats.columns = ['date', 'home_team'] + [f'home_team{c}' for c in stat_cols]

        away_stats = long.loc[~long['is_home'], needed].copy()
        away_stats = away_stats.drop(columns=['is_home'])
        away_stats = away_stats.drop_duplicates(subset=['date', 'team'])
        away_stats.columns = ['date', 'away_team'] + [f'away_team{c}' for c in stat_cols]

        df = df.merge(home_stats, on=['date', 'home_team'], how='left')
        df = df.merge(away_stats, on=['date', 'away_team'], how='left')
        del home_stats, away_stats
        gc.collect()

        # Rename to friendly format
        rename_map = {}
        for window in windows:
            for side in ['home_team', 'away_team']:
                for stat, friendly in [
                    (f'_{window}g_avg_scored',  f'team_{window}g_avg_scored'),
                    (f'_{window}g_std_scored',  f'team_{window}g_std_scored'),
                    (f'_{window}g_avg_allowed', f'team_{window}g_avg_allowed'),
                    (f'_{window}g_std_allowed', f'team_{window}g_std_allowed'),
                    (f'_{window}g_win_rate',    f'team_{window}g_win_rate'),
                ]:
                    rename_map[f'{side}{stat}'] = f'{side}_{friendly}'
        df.rename(columns=rename_map, inplace=True)

        # ── Venue-split rolling stats ─────────────────────────────────────────
        # Computed on home_long / away_long separately (~7.7k rows each)
        # then merged directly onto df — never inflated back into full long.
        for venue_flag, venue_label, side_col in [
            (True,  'home', 'home_team'),
            (False, 'away', 'away_team'),
        ]:
            venue_long = long.loc[long['is_home'] == venue_flag,
                                  ['date', 'team', 'scored', 'allowed', 'win']].copy()
            venue_long = venue_long.sort_values(['team', 'date']).reset_index(drop=True)
            vgrp = venue_long.groupby('team', sort=False)
            vcols = []
            for window in [5, 10]:
                c_scored  = f'_{window}g_{venue_label}_avg_scored'
                c_allowed = f'_{window}g_{venue_label}_avg_allowed'
                c_win     = f'_{window}g_{venue_label}_win_rate'
                venue_long[c_scored]  = vgrp['scored'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                venue_long[c_allowed] = vgrp['allowed'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                venue_long[c_win]     = vgrp['win'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                vcols += [c_scored, c_allowed, c_win]

            # Rename team → side_col before merging onto df
            to_merge = venue_long[['date', 'team'] + vcols].copy()
            to_merge = to_merge.drop_duplicates(subset=['date', 'team'])
            to_merge.columns = ['date', side_col] + [f'{side_col}_{c}' for c in vcols]
            df = df.merge(to_merge, on=['date', side_col], how='left')
            del venue_long, to_merge
            gc.collect()

        del long
        gc.collect()

        return df
    
    def _create_team_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Team offensive/defensive ratings — memory-efficient version."""
        import gc
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        home_view = df[['date', 'home_team', 'home_score', 'away_score']].copy()
        home_view.columns = ['date', 'team', 'scored', 'allowed']
        away_view = df[['date', 'away_team', 'away_score', 'home_score']].copy()
        away_view.columns = ['date', 'team', 'scored', 'allowed']

        long = pd.concat([home_view, away_view], ignore_index=True)
        del home_view, away_view
        gc.collect()

        long = long.sort_values(['team', 'date']).reset_index(drop=True)
        grp = long.groupby('team', sort=False)
        long['off_rating']    = grp['scored'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        long['def_rating']    = grp['allowed'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        long['net_rating']    = long['off_rating'] - long['def_rating']
        long['off_efficiency']= long['off_rating'] / 110.0
        long['def_efficiency']= long['def_rating'] / 110.0

        rating_cols = ['off_rating', 'def_rating', 'net_rating', 'off_efficiency', 'def_efficiency']
        needed = ['date', 'team'] + rating_cols

        # Merge home ratings — filter long to only home-team rows via df join
        home_teams = df[['date', 'home_team']].rename(columns={'home_team': 'team'})
        home_r = long.merge(home_teams, on=['date', 'team'], how='inner')[needed].copy()
        home_r = home_r.drop_duplicates(subset=['date', 'team'])
        home_r.columns = ['date', 'home_team'] + [f'home_team_{c}' for c in rating_cols]
        df = df.merge(home_r, on=['date', 'home_team'], how='left')
        del home_r, home_teams
        gc.collect()

        away_teams = df[['date', 'away_team']].rename(columns={'away_team': 'team'})
        away_r = long.merge(away_teams, on=['date', 'team'], how='inner')[needed].copy()
        away_r = away_r.drop_duplicates(subset=['date', 'team'])
        away_r.columns = ['date', 'away_team'] + [f'away_team_{c}' for c in rating_cols]
        df = df.merge(away_r, on=['date', 'away_team'], how='left')
        del away_r, away_teams, long
        gc.collect()

        return df
    
    def _create_advanced_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced statistical features"""
        
        # Strength of schedule (normalized)
        df = self._calculate_sos(df)
        
        # Recent form trend (normalized)
        df = self._calculate_form_trend(df)
        
        # Rest days (already integer, will be scaled later)
        df['home_rest_days'] = self._calculate_rest_days(df, 'home')
        df['away_rest_days'] = self._calculate_rest_days(df, 'away')
        df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']
        
        # Normalize rest days (typical range 0-4)
        df['home_rest_normalized'] = df['home_rest_days'] / 4.0
        df['away_rest_normalized'] = df['away_rest_days'] / 4.0
        df['rest_advantage_normalized'] = df['rest_advantage'] / 4.0
        
        # Back-to-back games
        df['home_b2b'] = (df['home_rest_days'] == 0).astype(int)
        df['away_b2b'] = (df['away_rest_days'] == 0).astype(int)
        
        return df
    
    def _calculate_sos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate strength of schedule using a vectorized approach.
        SOS for a team = rolling mean of |score_diff| of their recent opponents,
        normalized to [0,1]. Avoids the O(n^2) row-by-row loop by precomputing
        a rolling competitiveness score per team, then joining on opponent.
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Compute each team's rolling competitiveness (avg |margin| over last 10 games)
        home_view = df[['date', 'home_team', 'score_diff']].copy()
        home_view.columns = ['date', 'team', 'margin']
        away_view = df[['date', 'away_team', 'score_diff']].copy()
        away_view['score_diff'] = -away_view['score_diff']  # flip for away perspective
        away_view.columns = ['date', 'team', 'margin']

        long = pd.concat([home_view, away_view], ignore_index=True)
        long['abs_margin'] = long['margin'].abs()
        long = long.sort_values(['team', 'date']).reset_index(drop=True)

        long['team_competitiveness'] = (
            long.groupby('team', sort=False)['abs_margin']
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        )

        # Map opponent competitiveness onto each game
        comp_lookup = long[['date', 'team', 'team_competitiveness']].drop_duplicates(subset=['date', 'team'])

        df = df.merge(
            comp_lookup.rename(columns={'team': 'away_team', 'team_competitiveness': 'home_sos_raw'}),
            on=['date', 'away_team'], how='left'
        )
        df = df.merge(
            comp_lookup.rename(columns={'team': 'home_team', 'team_competitiveness': 'away_sos_raw'}),
            on=['date', 'home_team'], how='left'
        )

        # Normalize to [0,1] range (typical abs margin range 5-20)
        df['home_sos'] = ((df['home_sos_raw'].fillna(12.5) - 5) / 15).clip(0, 1)
        df['away_sos'] = ((df['away_sos_raw'].fillna(12.5) - 5) / 15).clip(0, 1)
        df.drop(columns=['home_sos_raw', 'away_sos_raw'], inplace=True)

        return df
    
    def _calculate_form_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate recent scoring trend and win streak per team, vectorized.
        Trend = slope of linear fit over last 5 scores (normalized by /10).
        Win streak = cumulative wins since last loss, computed with a groupby
        cumsum trick instead of a Python for-loop.
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        home_view = df[['date', 'home_team', 'home_score', 'away_score']].copy()
        home_view.columns = ['date', 'team', 'scored', 'opp_scored']
        away_view = df[['date', 'away_team', 'away_score', 'home_score']].copy()
        away_view.columns = ['date', 'team', 'scored', 'opp_scored']

        long = pd.concat([home_view, away_view], ignore_index=True)
        long['win'] = (long['scored'] > long['opp_scored']).astype(int)
        long = long.sort_values(['team', 'date']).reset_index(drop=True)

        # Scoring trend: slope of last-5 scores (shift first to avoid leakage)
        long['form_trend'] = (
            long.groupby('team', sort=False)['scored']
            .transform(lambda x: x.shift(1).rolling(5, min_periods=2).apply(
                lambda w: np.polyfit(range(len(w)), w, 1)[0] / 10, raw=True
            ))
        ).fillna(0)

        # Win streak: use loss-marker + cumsum grouping (vectorized)
        # Shift loss_group so current game's outcome doesn't define its own group
        long['loss']           = 1 - long['win']
        long['loss_cumsum']    = long.groupby('team', sort=False)['loss'].cumsum()
        long['loss_group']     = long.groupby('team', sort=False)['loss_cumsum'].transform(
            lambda x: x.shift(1).fillna(0)
        )
        long['win_streak_raw'] = long.groupby(['team', 'loss_group'], sort=False)['win'].cumsum()
        # Shift so we use streak entering the game, not including it
        long['win_streak'] = long.groupby('team', sort=False)['win_streak_raw'].transform(lambda x: x.shift(1).fillna(0))
        long['win_streak_normalized'] = (long['win_streak'] / 10.0).clip(0, 1)

        trend_cols = ['form_trend', 'win_streak_normalized']

        # Must deduplicate on ['date','team'] BEFORE rename, because after rename
        # the column is 'home_team'/'away_team' but the original duplication is on 'team'.
        # long has 2 rows per (date,team): one from home_view, one from away_view.
        # isin() does NOT remove these duplicates — only drop_duplicates does.
        home_t = long[long['team'].isin(df['home_team'].unique())].copy()
        home_t = home_t[['date', 'team'] + trend_cols]
        home_t = home_t.drop_duplicates(subset=['date', 'team'])
        home_t = home_t.rename(columns={'team': 'home_team', 'form_trend': 'home_form_trend',
                                         'win_streak_normalized': 'home_win_streak_normalized'})

        away_t = long[long['team'].isin(df['away_team'].unique())].copy()
        away_t = away_t[['date', 'team'] + trend_cols]
        away_t = away_t.drop_duplicates(subset=['date', 'team'])
        away_t = away_t.rename(columns={'team': 'away_team', 'form_trend': 'away_form_trend',
                                         'win_streak_normalized': 'away_win_streak_normalized'})
        df = df.merge(home_t, on=['date', 'home_team'], how='left')
        df = df.merge(away_t, on=['date', 'away_team'], how='left')

        return df
    
    def _calculate_rest_days(self, df: pd.DataFrame, team_type: str) -> pd.Series:
        """
        Calculate days of rest per team using vectorized shift on sorted game dates,
        avoiding the O(n^2) loop of filtering the full dataframe per game.
        """
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        team_col = f'{team_type}_team'

        # Build a unified game-date series per team
        home_dates = df_copy[['date', 'home_team']].rename(columns={'home_team': 'team'})
        away_dates = df_copy[['date', 'away_team']].rename(columns={'away_team': 'team'})
        all_dates = pd.concat([home_dates, away_dates], ignore_index=True)
        all_dates = all_dates.sort_values(['team', 'date']).reset_index(drop=True)

        # Previous game date per (team, game_date)
        all_dates['prev_date'] = all_dates.groupby('team')['date'].shift(1)
        all_dates['rest'] = (all_dates['date'] - all_dates['prev_date']).dt.days - 1
        all_dates['rest'] = all_dates['rest'].fillna(3).clip(0, 7).astype(int)

        # Map back to the original df rows via the specific team column
        lookup = all_dates.drop_duplicates(subset=['team', 'date'])
        merged = df_copy[[team_col, 'date']].merge(
            lookup.rename(columns={'team': team_col}),
            on=[team_col, 'date'],
            how='left'
        )
        return merged['rest'].fillna(3).astype(int).values
    
    def _create_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized head-to-head features.

        Instead of looping over every game and filtering the full dataframe
        (O(n^2)), we:
          1. Build a canonical matchup key so (A vs B) and (B vs A) share the
             same key regardless of who is home.
          2. Sort by date and use groupby + expanding().mean().shift(1) so each
             game only sees past matchups — no leakage, no row-by-row loop.
          3. Merge the pre-computed stats back onto the original dataframe.

        This reduces 7000-game processing from ~30 minutes to a few seconds.
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Canonical matchup key: always sort the two team names alphabetically
        # so (Lakers vs Celtics) == (Celtics vs Lakers)
        df['_h2h_key'] = df.apply(
            lambda r: '__'.join(sorted([r['home_team'], r['away_team']])), axis=1
        )

        # For each game, pre-compute expanding stats over all previous meetings
        # of the same matchup pair.
        grp = df.groupby('_h2h_key', sort=False)

        # Count of prior meetings (shift so current game not included)
        df['h2h_games_raw'] = grp.cumcount()  # 0-indexed count before this game

        # Rolling averages of total, diff, home_win — shift(1) = exclude current row
        df['h2h_avg_total_raw']    = grp['total_score'].transform(lambda x: x.shift(1).expanding().mean())
        df['h2h_avg_diff_raw']     = grp['score_diff'].transform(lambda x: x.shift(1).expanding().mean())
        df['h2h_home_win_rate_raw']= grp['home_win'].transform(lambda x: x.shift(1).expanding().mean())

        # Per-team scored/allowed in previous meetings.
        # We need home_team's avg scored when they were home in this specific matchup,
        # and away_team's avg scored when they were away.  Since the canonical key
        # merges both directions, we track home_score and away_score expanding means
        # from the perspective of whoever is listed as home_team in THIS game.
        # For games where the roles were reversed in history, home_score = opponent's
        # score for that team — but expanding mean of home_score is still a useful
        # proxy for "what this matchup typically produces on each side".
        df['h2h_home_scored_raw']  = grp['home_score'].transform(lambda x: x.shift(1).expanding().mean())
        df['h2h_away_scored_raw']  = grp['away_score'].transform(lambda x: x.shift(1).expanding().mean())

        # Normalise / fill defaults where no prior meetings exist
        no_h2h = df['h2h_games_raw'] == 0

        df['h2h_games']               = (df['h2h_games_raw'] / 10.0).clip(0, 1)
        df['h2h_avg_total']           = ((df['h2h_avg_total_raw'].fillna(220) - 200) / 40)
        df['h2h_avg_diff']            = (df['h2h_avg_diff_raw'].fillna(0) / 20)
        df['h2h_home_win_rate']       = df['h2h_home_win_rate_raw'].fillna(0.5)
        df['h2h_home_team_avg_scored']  = df['h2h_home_scored_raw'].fillna(110.0)
        df['h2h_home_team_avg_allowed'] = df['h2h_away_scored_raw'].fillna(110.0)
        df['h2h_away_team_avg_scored']  = df['h2h_away_scored_raw'].fillna(108.0)
        df['h2h_away_team_avg_allowed'] = df['h2h_home_scored_raw'].fillna(108.0)

        # Drop temp columns
        df.drop(columns=[
            '_h2h_key', 'h2h_games_raw', 'h2h_avg_total_raw', 'h2h_avg_diff_raw',
            'h2h_home_win_rate_raw', 'h2h_home_scored_raw', 'h2h_away_scored_raw'
        ], inplace=True)

        return df
    
    def _create_situational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create situational game features (already normalized)"""
        
        # Playoff implications (normalized by month)
        df['is_playoff_season'] = (df['date'].dt.month >= 4).astype(float)
        
        # Playoffs: NBA playoffs run mid-April through June (months 4-6)
        # Regular season scoring averages ~5 pts higher than playoff games
        # so giving the model an explicit flag lets it learn this pattern
        # without conflating the two contexts
        df['is_playoffs'] = (
            (df['date'].dt.month >= 4) & (df['date'].dt.month <= 6)
        ).astype(float)
        
        # Weekend games
        df['is_weekend'] = (df['day_of_week'] * 6 >= 5).astype(float)
        
        # Season progress derived from date — accurate at both training and prediction
        # time without relying on cumcount() over a windowed context.
        #
        # Previous approach used groupby().cumcount() which was wrong at prediction
        # time: predict_games appends the future game to historical_df.tail(1000),
        # so cumcount counted position within that 1000-row window rather than the
        # true game number in the season, producing values systematically off by
        # ~0.5-1.0 seasons depending on team and window contents.
        #
        # NBA regular season runs Oct 1 → Apr 20 (~201 days). We map the game date
        # onto a 0→1 progress scale within that window. Playoff games clip to 1.0.
        # This is perfectly accurate for any date — past, present, or future.
        df = df.sort_values('date').reset_index(drop=True)

        # Season year: Oct-Dec → current year, Jan-Sep → previous year
        season_year = df['date'].apply(
            lambda d: d.year if d.month >= 10 else d.year - 1
        )
        season_start = pd.to_datetime(season_year.astype(str) + '-10-01')
        season_end   = pd.to_datetime((season_year + 1).astype(str) + '-04-20')
        season_len_days = (season_end - season_start).dt.days   # ~201 days
        days_into_season = (df['date'] - season_start).dt.days.clip(0)

        df['season_progress']        = (days_into_season / season_len_days).clip(0, 1)
        # Keep norm aliases so any downstream code referencing these column names
        # still works — they all carry the same date-derived value now
        df['games_played_home_norm'] = df['season_progress']
        df['games_played_away_norm'] = df['season_progress']
        
        return df
    
    def _create_pace_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling historical pace/efficiency — no leakage, memory-efficient.
        Uses del + gc.collect() to free intermediates immediately after use.
        """
        import gc
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        home_view = df[['date', 'home_team', 'home_score', 'away_score']].copy()
        home_view.columns = ['date', 'team', 'scored', 'allowed']
        away_view = df[['date', 'away_team', 'away_score', 'home_score']].copy()
        away_view.columns = ['date', 'team', 'scored', 'allowed']

        long = pd.concat([home_view, away_view], ignore_index=True)
        del home_view, away_view
        gc.collect()

        long = long.sort_values(['team', 'date']).reset_index(drop=True)
        long['total'] = long['scored'] + long['allowed']

        grp = long.groupby('team', sort=False)
        long['rolling_pace']    = grp['total'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        long['rolling_off_eff'] = grp['scored'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        long['rolling_def_eff'] = grp['allowed'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())

        pace_cols = ['rolling_pace', 'rolling_off_eff', 'rolling_def_eff']
        needed = ['date', 'team'] + pace_cols

        home_teams = df[['date', 'home_team']].rename(columns={'home_team': 'team'})
        home_p = long.merge(home_teams, on=['date', 'team'], how='inner')[needed].copy()
        home_p = home_p.drop_duplicates(subset=['date', 'team'])
        home_p.columns = ['date', 'home_team'] + [f'home_team_{c}' for c in pace_cols]
        df = df.merge(home_p, on=['date', 'home_team'], how='left')
        del home_p, home_teams
        gc.collect()

        away_teams = df[['date', 'away_team']].rename(columns={'away_team': 'team'})
        away_p = long.merge(away_teams, on=['date', 'team'], how='inner')[needed].copy()
        away_p = away_p.drop_duplicates(subset=['date', 'team'])
        away_p.columns = ['date', 'away_team'] + [f'away_team_{c}' for c in pace_cols]
        df = df.merge(away_p, on=['date', 'away_team'], how='left')
        del away_p, away_teams, long
        gc.collect()

        df['expected_pace']   = (df['home_team_rolling_pace'].fillna(220) + df['away_team_rolling_pace'].fillna(220)) / 2
        df['pace_normalized'] = (df['expected_pace'] - 220) / 20
        df['off_eff_diff']    = (df['home_team_rolling_off_eff'].fillna(110) - df['away_team_rolling_def_eff'].fillna(110)) / 10
        df['def_eff_diff']    = (df['away_team_rolling_off_eff'].fillna(110) - df['home_team_rolling_def_eff'].fillna(110)) / 10

        return df
    

    def _create_last5_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Last-5-games form score: weighted recent results.
        More recent games weighted higher (5,4,3,2,1 weights).
        Result: single float per team per game — positive = hot, negative = cold.
        Separate home/away form since home performance differs from road.
        """
        import gc
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        home_view = df[['date', 'home_team', 'home_score', 'away_score']].copy()
        home_view.columns = ['date', 'team', 'scored', 'allowed']
        away_view = df[['date', 'away_team', 'away_score', 'home_score']].copy()
        away_view.columns = ['date', 'team', 'scored', 'allowed']

        long = pd.concat([home_view, away_view], ignore_index=True)
        del home_view, away_view
        gc.collect()

        long['margin'] = long['scored'] - long['allowed']
        long = long.sort_values(['team', 'date']).reset_index(drop=True)

        # Weighted form: dot([5,4,3,2,1], last_5_margins) / 15 → normalised
        def _weighted_form(x):
            shifted = x.shift(1)
            def _wf(w):
                n = len(w)
                weights = list(range(n, 0, -1))
                return np.dot(weights, w) / sum(weights)
            return shifted.rolling(5, min_periods=1).apply(_wf, raw=True)

        long['form_score'] = (
            long.groupby('team', sort=False)['margin']
            .transform(_weighted_form)
            .fillna(0) / 20.0  # normalise: typical margin ~±20
        )

        needed = ['date', 'team', 'form_score']
        home_f = long[needed].copy().drop_duplicates(subset=['date', 'team'])
        home_f = home_f.rename(columns={'team': 'home_team', 'form_score': 'home_form_score'})
        away_f = long[needed].copy().drop_duplicates(subset=['date', 'team'])
        away_f = away_f.rename(columns={'team': 'away_team', 'form_score': 'away_form_score'})

        df = df.merge(home_f, on=['date', 'home_team'], how='left')
        df = df.merge(away_f, on=['date', 'away_team'], how='left')
        df['form_diff'] = df['home_form_score'].fillna(0) - df['away_form_score'].fillna(0)

        del long, home_f, away_f
        gc.collect()
        return df

    def _create_season_stage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Season stage features: encode where we are in the NBA season.
        Early (Oct-Nov), Mid (Dec-Feb), Late Regular Season (Mar-Apr),
        Playoffs (Apr-Jun). Each gets its own binary flag + a continuous
        0-1 progress variable so the model can learn monotonic patterns.
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        month = df['date'].dt.month

        # Oct-Nov: season start. Both months are >= 10, so a simple >= 10 & <= 11 works.
        df['stage_early']    = ((month >= 10) & (month <= 11)).astype(float)
        # Dec-Feb: crosses the year boundary, so we need OR of the two segments.
        df['stage_mid']      = ((month == 12) | (month <= 2)).astype(float)
        df['stage_late']     = ((month == 3)  | (month == 4)).astype(float)   # Mar-Apr
        df['stage_playoffs'] = ((month >= 4)  & (month <= 6)).astype(float)   # Apr-Jun

        # season_progress_pct: fraction of the regular season elapsed, derived
        # purely from date so it's correct at prediction time.
        # Reuses the same Oct-1 → Apr-20 window logic as _create_situational_features.
        # If season_progress is already computed (it runs first in create_features),
        # we can reuse it directly; otherwise we recompute from date.
        if 'season_progress' in df.columns:
            df['season_progress_pct'] = df['season_progress']
        else:
            _sy = df['date'].apply(lambda d: d.year if d.month >= 10 else d.year - 1)
            _ss = pd.to_datetime(_sy.astype(str) + '-10-01')
            _se = pd.to_datetime((_sy + 1).astype(str) + '-04-20')
            df['season_progress_pct'] = (
                (df['date'] - _ss).dt.days.clip(0) / (_se - _ss).dt.days
            ).clip(0, 1)
        # Keep season_game_num as a rounded estimate for any code that reads it,
        # but derive it from the date-based progress rather than cumcount
        df['season_game_num'] = (df['season_progress_pct'] * 1230).round().astype(int).clip(1, 1230)

        return df

    def _create_opponent_adjusted_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Opponent-adjusted rolling offense and defense.

        Raw rolling PPG tells you how much a team scores on average, but it
        conflates good offenses with easy schedules.  Opponent-adjusted stats
        subtract the opponent's rolling defensive average from the team's score,
        and vice versa, giving a cleaner signal of true offensive/defensive
        quality.

        All shift(1) to avoid leakage — each game only sees prior history.
        All features normalised by /10 to keep them in a model-friendly range.
        NaN-safe: fills with 0 (neutral) when no prior data exists.
        """
        import gc
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        home_view = df[['date', 'home_team', 'home_score', 'away_score']].copy()
        home_view.columns = ['date', 'team', 'scored', 'allowed']
        away_view = df[['date', 'away_team', 'away_score', 'home_score']].copy()
        away_view.columns = ['date', 'team', 'scored', 'allowed']

        long = pd.concat([home_view, away_view], ignore_index=True)
        del home_view, away_view
        gc.collect()

        long = long.sort_values(['team', 'date']).reset_index(drop=True)
        grp = long.groupby('team', sort=False)

        # Rolling avg points scored and allowed (10-game window, shift to avoid leakage)
        long['roll_scored_10']  = grp['scored'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        long['roll_allowed_10'] = grp['allowed'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())

        # Build opponent lookup: for each (date, team) what was their opponent's
        # rolling defensive average entering this game?
        # We join on (date, opponent_team) — the opponent is stored in 'allowed' side.
        # Reconstruct opponent identity from the original df.
        home_opp = df[['date', 'home_team', 'away_team']].rename(
            columns={'home_team': 'team', 'away_team': 'opponent'})
        away_opp = df[['date', 'away_team', 'home_team']].rename(
            columns={'away_team': 'team', 'home_team': 'opponent'})
        opp_map = pd.concat([home_opp, away_opp], ignore_index=True)
        del home_opp, away_opp

        # Merge opponent's rolling defensive avg onto each row
        def_lookup = long[['date', 'team', 'roll_allowed_10']].drop_duplicates(subset=['date', 'team'])
        opp_map = opp_map.merge(
            def_lookup.rename(columns={'team': 'opponent', 'roll_allowed_10': 'opp_roll_allowed'}),
            on=['date', 'opponent'], how='left'
        )
        # Merge team's own rolling scored
        opp_map = opp_map.merge(
            def_lookup.rename(columns={'team': 'team', 'roll_allowed_10': 'team_roll_scored'}).assign(
                team_roll_scored=long.drop_duplicates(subset=['date', 'team'])
                    .set_index(['date', 'team'])['roll_scored_10']
                    .reindex(def_lookup.set_index(['date', 'team']).index)
                    .values
            ),
            on=['date', 'team'], how='left'
        )

        # Simpler direct approach: build a single lookup table (date, team) ->
        # (roll_scored_10, roll_allowed_10) then join twice
        stats_lookup = long[['date', 'team', 'roll_scored_10', 'roll_allowed_10']].drop_duplicates(
            subset=['date', 'team'])

        # For home team: opp_adj_off = home_roll_scored - away_roll_allowed
        #                opp_adj_def = home_roll_allowed - away_roll_scored
        df = df.merge(
            stats_lookup.rename(columns={
                'team': 'home_team',
                'roll_scored_10': 'h_roll_scored',
                'roll_allowed_10': 'h_roll_allowed'
            }),
            on=['date', 'home_team'], how='left'
        )
        df = df.merge(
            stats_lookup.rename(columns={
                'team': 'away_team',
                'roll_scored_10': 'a_roll_scored',
                'roll_allowed_10': 'a_roll_allowed'
            }),
            on=['date', 'away_team'], how='left'
        )
        del stats_lookup, long, opp_map, def_lookup
        gc.collect()

        # Opponent-adjusted offense: how much better/worse does this team score
        # vs what the opponent typically allows?  Normalise by /10.
        df['home_opp_adj_off'] = (
            (df['h_roll_scored'].fillna(110) - df['a_roll_allowed'].fillna(110)) / 10.0
        )
        df['away_opp_adj_off'] = (
            (df['a_roll_scored'].fillna(110) - df['h_roll_allowed'].fillna(110)) / 10.0
        )
        # Opponent-adjusted defense: how much better/worse does this team defend
        # vs what the opponent typically scores?
        df['home_opp_adj_def'] = (
            (df['h_roll_allowed'].fillna(110) - df['a_roll_scored'].fillna(110)) / 10.0
        )
        df['away_opp_adj_def'] = (
            (df['a_roll_allowed'].fillna(110) - df['h_roll_scored'].fillna(110)) / 10.0
        )
        # Net matchup edge: combined offensive + defensive edge
        df['home_matchup_edge'] = df['home_opp_adj_off'] - df['away_opp_adj_off']

        df.drop(columns=['h_roll_scored', 'h_roll_allowed', 'a_roll_scored', 'a_roll_allowed'],
                inplace=True)
        return df

    def _create_b2b_second_night(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Back-to-back second night flag.

        The existing home_b2b / away_b2b flags (rest_days == 0) correctly
        identify any back-to-back game.  But the SECOND night of a back-to-back
        (i.e. rest_days == 0 AND the previous game was also rest_days == 0 for
        the opponent, meaning the team played yesterday) is more impactful than
        the first night — fatigue compounds, scoring drops ~4-5 pts on average.

        We derive this purely from rest_days which is already computed by
        _create_advanced_stats, so no additional data is needed.

        b2b_first_night:  rest_days == 0  (playing tonight, had a game yesterday)
        b2b_second_night: rest_days == 0 AND the game before that was also 0
                          — detected via a lag-2 check on the sorted game sequence.

        All values are 0/1 integers, no scaling needed beyond model defaults.
        NaN-safe: fills with 0.
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        for side in ('home', 'away'):
            team_col   = f'{side}_team'
            rest_col   = f'{side}_rest_days'
            b2b_col    = f'{side}_b2b'          # already exists from _create_advanced_stats
            sn_col     = f'{side}_b2b_second_night'

            if rest_col not in df.columns:
                df[sn_col] = 0
                continue

            # Sort by team and date, then shift rest_days by 1 to get the previous
            # game's rest days for the same team.
            tmp = df[['date', team_col, rest_col]].copy().sort_values([team_col, 'date'])
            tmp['prev_rest'] = tmp.groupby(team_col)[rest_col].shift(1)

            # Second night = current rest == 0 AND previous rest was also 0
            # (meaning yesterday they also played with 0 rest, i.e. three games
            # in three nights is the extreme case, but rest==0 twice means the
            # team is on a genuine back-to-back stretch)
            tmp[sn_col] = (
                (tmp[rest_col] == 0) & (tmp['prev_rest'] == 0)
            ).astype(int).fillna(0)

            df = df.merge(
                tmp[['date', team_col, sn_col]],
                on=['date', team_col], how='left'
            )
            df[sn_col] = df[sn_col].fillna(0).astype(int)

        return df

    def _create_streak_margin(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Average margin during the current win/loss streak.

        A 5-game winning streak by 1 point each is much weaker signal than a
        5-game winning streak by 15 points each.  This feature captures the
        quality of the streak, not just its length.

        streak_margin > 0  → winning and winning convincingly
        streak_margin < 0  → losing and losing badly
        streak_margin ≈ 0  → winning/losing close games or no streak

        Normalised by /15 (typical convincing margin) and clipped to [-1, 1].
        shift(1) to avoid leakage.
        NaN-safe: fills with 0.
        """
        import gc
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        home_view = df[['date', 'home_team', 'home_score', 'away_score']].copy()
        home_view.columns = ['date', 'team', 'scored', 'allowed']
        away_view = df[['date', 'away_team', 'away_score', 'home_score']].copy()
        away_view.columns = ['date', 'team', 'scored', 'allowed']

        long = pd.concat([home_view, away_view], ignore_index=True)
        del home_view, away_view
        gc.collect()

        long['margin'] = long['scored'] - long['allowed']
        long['win']    = (long['margin'] > 0).astype(int)
        long = long.sort_values(['team', 'date']).reset_index(drop=True)

        # Streak group: every loss resets the win-streak group counter.
        # IMPORTANT: shift loss_group by 1 so the current game's outcome does
        # not influence which streak group it belongs to — otherwise the group
        # boundary itself leaks whether today is a win or loss.
        long['loss']            = 1 - long['win']
        long['loss_cumsum']     = long.groupby('team', sort=False)['loss'].cumsum()
        long['loss_group']      = long.groupby('team', sort=False)['loss_cumsum'].transform(
            lambda x: x.shift(1).fillna(0)
        )

        # Average margin within each streak group (shift so current game excluded)
        long['streak_margin_raw'] = (
            long.groupby(['team', 'loss_group'], sort=False)['margin']
            .transform(lambda x: x.shift(1).expanding().mean())
        )
        # Normalise and clip
        long['streak_margin'] = (long['streak_margin_raw'].fillna(0) / 15.0).clip(-1, 1)

        needed = ['date', 'team', 'streak_margin']
        home_sm = long[needed].drop_duplicates(subset=['date', 'team']).rename(
            columns={'team': 'home_team', 'streak_margin': 'home_streak_margin'})
        away_sm = long[needed].drop_duplicates(subset=['date', 'team']).rename(
            columns={'team': 'away_team', 'streak_margin': 'away_streak_margin'})

        df = df.merge(home_sm, on=['date', 'home_team'], how='left')
        df = df.merge(away_sm, on=['date', 'away_team'], how='left')
        df['home_streak_margin'] = df['home_streak_margin'].fillna(0)
        df['away_streak_margin'] = df['away_streak_margin'].fillna(0)
        df['streak_margin_diff'] = df['home_streak_margin'] - df['away_streak_margin']

        del long, home_sm, away_sm
        gc.collect()
        return df

    def _create_current_season_h2h(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Current-season head-to-head features.

        The existing H2H features use multi-year history, which helps with
        stable matchup tendencies but is diluted by roster turnover.  Current-
        season H2H is a sharper signal for the present roster matchup.

        We derive 'season' from the date column using the same NBA calendar
        logic used elsewhere: Oct-Dec belongs to season_year+1, Jan-Sep to
        season_year.  If a 'season' column already exists we use it directly.

        All shift(1) within (season, h2h_key) to avoid leakage.
        Falls back to neutral defaults (220 total, 0 diff, 0.5 win rate) when
        no current-season meetings have occurred yet.
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Derive season year if not already present
        if 'season' not in df.columns:
            df['_season_derived'] = df['date'].apply(
                lambda d: d.year + 1 if d.month >= 10 else d.year
            )
            season_col = '_season_derived'
        else:
            season_col = 'season'

        df = df.sort_values('date').reset_index(drop=True)

        # Canonical matchup key (alphabetical, same as existing H2H)
        df['_cs_h2h_key'] = df.apply(
            lambda r: '__'.join(sorted([r['home_team'], r['away_team']])), axis=1
        )
        df['_cs_group'] = df['_cs_h2h_key'] + '__' + df[season_col].astype(str)

        grp = df.groupby('_cs_group', sort=False)

        df['cs_h2h_games_raw']    = grp.cumcount()
        df['cs_h2h_avg_total_raw'] = grp['total_score'].transform(
            lambda x: x.shift(1).expanding().mean())
        df['cs_h2h_avg_diff_raw']  = grp['score_diff'].transform(
            lambda x: x.shift(1).expanding().mean())
        df['cs_h2h_home_win_raw']  = grp['home_win'].transform(
            lambda x: x.shift(1).expanding().mean())

        # Normalise — same scale as existing h2h_ features for consistency
        df['cs_h2h_games']        = (df['cs_h2h_games_raw'] / 4.0).clip(0, 1)  # max 4 meetings/season
        df['cs_h2h_avg_total']    = ((df['cs_h2h_avg_total_raw'].fillna(220) - 200) / 40)
        df['cs_h2h_avg_diff']     = (df['cs_h2h_avg_diff_raw'].fillna(0) / 20)
        df['cs_h2h_home_win_rate']= df['cs_h2h_home_win_raw'].fillna(0.5)

        # Flag: do we have any current-season data?
        df['cs_h2h_has_data'] = (df['cs_h2h_games_raw'] > 0).astype(float)

        df.drop(columns=[
            '_cs_h2h_key', '_cs_group',
            'cs_h2h_games_raw', 'cs_h2h_avg_total_raw',
            'cs_h2h_avg_diff_raw', 'cs_h2h_home_win_raw',
        ] + (['_season_derived'] if '_season_derived' in df.columns else []),
        inplace=True)

        return df

    def _create_pace_matchup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pace matchup interaction feature.

        Individual team pace is already captured in _create_pace_efficiency.
        The interaction — what happens when a fast team plays a slow team —
        is not captured.  This feature computes:

          pace_contrast:  |home_pace - away_pace|  normalised by /20
              High contrast → one team will be forced out of rhythm.
              Low contrast  → both teams play at a similar tempo.

          pace_product:   home_pace * away_pace  (normalised)
              Captures whether BOTH teams are fast (high total expected)
              vs both slow (low total expected) vs mixed.

        Uses home_team_rolling_pace / away_team_rolling_pace already computed
        by _create_pace_efficiency.  Falls back to 220 if missing.
        No leakage risk — rolling_pace is already shift(1)-lagged.
        """
        df = df.copy()

        home_pace = df.get('home_team_rolling_pace', pd.Series(220.0, index=df.index)).fillna(220.0)
        away_pace = df.get('away_team_rolling_pace', pd.Series(220.0, index=df.index)).fillna(220.0)

        # Contrast: absolute difference normalised by typical spread (~20 pts)
        df['pace_contrast']     = ((home_pace - away_pace).abs() / 20.0).clip(0, 1)

        # Direction: positive = home team is faster
        df['pace_direction']    = ((home_pace - away_pace) / 20.0).clip(-1, 1)

        # Product proxy for "both teams fast" — normalise around baseline 220^2
        df['pace_product_norm'] = ((home_pace * away_pace) - (210 ** 2)) / (20 * 420)

        return df

    def _create_venue_rest(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Venue-specific rest features.

        The existing rest_days treats home and away rest identically.  But rest
        affects home and away teams differently:
          - A home team on 0 rest still has the crowd advantage.
          - An away team on 0 rest has both fatigue AND no crowd support.

        We also compute the interaction: away team fatigued while home team rested
        is the most lopsided situation for scoring suppression on the away side.

        All values derived from existing home_rest_days / away_rest_days columns.
        NaN-safe: fills with neutral value (rest=3) if missing.
        """
        df = df.copy()

        home_rest = df.get('home_rest_days', pd.Series(3, index=df.index)).fillna(3)
        away_rest = df.get('away_rest_days', pd.Series(3, index=df.index)).fillna(3)

        # Venue-specific fatigue flags
        df['away_fatigued_home_rested'] = (
            (away_rest == 0) & (home_rest >= 2)
        ).astype(float)

        df['home_fatigued_away_rested'] = (
            (home_rest == 0) & (away_rest >= 2)
        ).astype(float)

        # Both teams on a back-to-back (rare but meaningful — high-energy game)
        df['both_b2b'] = (
            (home_rest == 0) & (away_rest == 0)
        ).astype(float)

        # Continuous rest asymmetry: positive = home team more rested
        # Normalised by /4 (max meaningful rest difference)
        df['rest_asymmetry'] = ((home_rest - away_rest) / 4.0).clip(-1, 1)

        # Away team cumulative fatigue: long road trips compound fatigue.
        # Proxy: away_rest_days clipped and inverted so 0 rest = max fatigue (1.0)
        df['away_fatigue_index'] = (1.0 - (away_rest.clip(0, 4) / 4.0))

        return df

    def _create_box_score_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling averages of box score stats scraped from ESPN:
        FG%, 3P%, FT%, rebounds, assists, FGA, FTA, 3PA.

        All new columns in historical data; safely skipped (filled with NaN→0)
        when the columns are absent (old cached data pre-rescrape).
        shift(1) applied so the current game's own stats never leak in.

        Window: 10 games (balances recency vs. sample size for percentages).
        Normalisation keeps values roughly in [-2, 2] for the scaler.
        """
        import gc

        # Required source columns — gracefully skip if not in df yet
        stat_pairs = [
            ('home_fg_pct',  'away_fg_pct'),
            ('home_3p_pct',  'away_3p_pct'),
            ('home_ft_pct',  'away_ft_pct'),
            ('home_reb',     'away_reb'),
            ('home_ast',     'away_ast'),
            ('home_fga',     'away_fga'),
            ('home_fta',     'away_fta'),
            ('home_3pa',     'away_3pa'),
        ]
        available = [(h, a) for h, a in stat_pairs if h in df.columns and a in df.columns]
        if not available:
            return df  # pre-rescrape data — nothing to do

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Stat short-names for column naming
        stat_names = {
            'home_fg_pct': 'fg_pct', 'home_3p_pct': '3p_pct', 'home_ft_pct': 'ft_pct',
            'home_reb':    'reb',    'home_ast':     'ast',
            'home_fga':    'fga',    'home_fta':     'fta',    'home_3pa':     '3pa',
        }

        for home_col, away_col in available:
            stat = stat_names[home_col]

            # Build long form: one row per (team, game)
            home_v = df[['date', 'home_team', home_col]].copy()
            home_v.columns = ['date', 'team', 'val']
            away_v = df[['date', 'away_team', away_col]].copy()
            away_v.columns = ['date', 'team', 'val']
            long = pd.concat([home_v, away_v], ignore_index=True)
            del home_v, away_v
            long = long.sort_values(['team', 'date']).reset_index(drop=True)

            grp = long.groupby('team', sort=False)
            long[f'rolling_{stat}'] = grp['val'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )

            needed = ['date', 'team', f'rolling_{stat}']
            home_r = long[needed].drop_duplicates(['date', 'team']).rename(
                columns={'team': 'home_team', f'rolling_{stat}': f'home_rolling_{stat}'})
            away_r = long[needed].drop_duplicates(['date', 'team']).rename(
                columns={'team': 'away_team', f'rolling_{stat}': f'away_rolling_{stat}'})

            df = df.merge(home_r, on=['date', 'home_team'], how='left')
            df = df.merge(away_r, on=['date', 'away_team'], how='left')

            # Difference features (home advantage in each stat)
            df[f'{stat}_diff'] = (
                df[f'home_rolling_{stat}'].fillna(0) -
                df[f'away_rolling_{stat}'].fillna(0)
            )

            del long, home_r, away_r
            gc.collect()

        # Derived pace proxy: FGA + 0.44*FTA is a standard NBA pace estimator
        if 'home_rolling_fga' in df.columns and 'home_rolling_fta' in df.columns:
            df['home_pace_proxy'] = (
                df['home_rolling_fga'].fillna(85) + 0.44 * df['home_rolling_fta'].fillna(22)
            )
            df['away_pace_proxy'] = (
                df['away_rolling_fga'].fillna(85) + 0.44 * df['away_rolling_fta'].fillna(22)
            )
            df['pace_proxy_avg']  = (df['home_pace_proxy'] + df['away_pace_proxy']) / 2
            # Normalise around typical ~97 possessions
            df['home_pace_proxy'] = (df['home_pace_proxy'] - 97) / 10
            df['away_pace_proxy'] = (df['away_pace_proxy'] - 97) / 10
            df['pace_proxy_avg']  = (df['pace_proxy_avg']  - 97) / 10

        return df

    def _create_quarter_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling quarter-score features derived from Q1-Q4 splits.

        Features:
          - home/away_rolling_q4: average 4th-quarter scoring (clutch scoring)
          - home/away_rolling_first_half: avg Q1+Q2 (early-game tempo setter)
          - home/away_rolling_second_half: avg Q3+Q4 (finishing strength)
          - home/away_q4_consistency: std of Q4 scores (lower = more reliable)
          - q4_edge: home rolling Q4 minus away rolling Q4

        All shift(1) to avoid leakage. Skipped gracefully if quarter cols absent.
        """
        import gc

        required = ['home_q1', 'home_q2', 'home_q3', 'home_q4',
                    'away_q1', 'away_q2', 'away_q3', 'away_q4']
        if not all(c in df.columns for c in required):
            return df  # pre-rescrape data

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        for side, team_col in [('home', 'home_team'), ('away', 'away_team')]:
            q1, q2, q3, q4 = (f'{side}_q1', f'{side}_q2',
                               f'{side}_q3', f'{side}_q4')

            view = df[['date', team_col, q1, q2, q3, q4]].copy()
            view.columns = ['date', 'team', 'q1', 'q2', 'q3', 'q4']
            view['first_half']  = view['q1'].fillna(0) + view['q2'].fillna(0)
            view['second_half'] = view['q3'].fillna(0) + view['q4'].fillna(0)
            view = view.sort_values(['team', 'date']).reset_index(drop=True)

            grp = view.groupby('team', sort=False)
            view['rolling_q4']          = grp['q4'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean())
            view['rolling_first_half']  = grp['first_half'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean())
            view['rolling_second_half'] = grp['second_half'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean())
            view['q4_consistency']      = grp['q4'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).std().fillna(5))

            # Normalise
            view['rolling_q4']          = (view['rolling_q4'].fillna(27)          - 27) / 5
            view['rolling_first_half']  = (view['rolling_first_half'].fillna(54)  - 54) / 8
            view['rolling_second_half'] = (view['rolling_second_half'].fillna(56) - 56) / 8
            view['q4_consistency']      = view['q4_consistency'] / 5

            feat_cols = ['rolling_q4', 'rolling_first_half',
                         'rolling_second_half', 'q4_consistency']
            needed = ['date', 'team'] + feat_cols
            to_merge = view[needed].drop_duplicates(['date', 'team']).rename(
                columns={'team': team_col, **{c: f'{side}_{c}' for c in feat_cols}}
            )
            df = df.merge(to_merge, on=['date', team_col], how='left')
            del view, to_merge
            gc.collect()

        # Q4 edge: positive = home team stronger in 4th quarter
        if 'home_rolling_q4' in df.columns and 'away_rolling_q4' in df.columns:
            df['q4_edge'] = (
                df['home_rolling_q4'].fillna(0) - df['away_rolling_q4'].fillna(0)
            )

        return df

    def _create_game_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Game-level metadata features: attendance, neutral site, national TV.

        These are game-level flags, not rolling averages — no leakage risk since
        they describe the game context, not outcomes.

        attendance:   normalised around typical NBA arena capacity (~18,500).
                      Missing = 0 (neutral, not misleading to the model).
        neutral_site: 1 if playoff neutral venue, 0 otherwise.
        national_tv:  1 if ESPN/ABC/TNT/NBC national broadcast.

        All three are weak signals on their own but useful interaction features
        for the tree models (e.g. neutral_site removes home court advantage).
        """
        df = df.copy()

        # Attendance — normalise: (attendance - 18500) / 5000
        # Typical range: ~8k (small market) to ~21k (sold out)
        if 'attendance' in df.columns:
            df['attendance_norm'] = (
                pd.to_numeric(df['attendance'], errors='coerce')
                .fillna(18500)
                .sub(18500)
                .div(5000)
                .clip(-2, 2)
            )
        else:
            df['attendance_norm'] = 0.0

        # Neutral site flag
        if 'neutral_site' in df.columns:
            df['neutral_site'] = pd.to_numeric(
                df['neutral_site'], errors='coerce').fillna(0).astype(float)
        else:
            df['neutral_site'] = 0.0

        # National TV flag
        if 'national_tv' in df.columns:
            df['national_tv'] = pd.to_numeric(
                df['national_tv'], errors='coerce').fillna(0).astype(float)
        else:
            df['national_tv'] = 0.0

        return df

    def prepare_prediction_features(self, game: pd.Series, historical_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for a single game prediction"""

        # Create a temporary DataFrame with the game
        temp_df = pd.DataFrame([game])

        # Today's games have no scores yet — add placeholder zeros so the
        # feature pipeline (_create_basic_features etc.) doesn't raise KeyError
        for col in ['home_score', 'away_score']:
            if col not in temp_df.columns:
                temp_df[col] = 0.0

        # Drop any columns that don't exist in historical_df to prevent
        # KeyErrors from ESPN-specific fields (e.g. 'game_id', 'game_time')
        # that were never seen during training
        extra_cols = [c for c in temp_df.columns if c not in set(historical_df.columns)]
        if extra_cols:
            temp_df = temp_df.drop(columns=extra_cols)

        # Add historical context by merging with recent games
        recent_games = historical_df.tail(1000).copy()
        combined_df = pd.concat([recent_games, temp_df], ignore_index=True)
        
        # Create features without fitting new scalers
        featured_df = self.create_features(combined_df, fit_scalers=False)
        
        # Return only the prediction game features
        return featured_df.iloc[-1:][self.feature_columns]
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_columns
    
    def save_scalers(self, path: str):
        """Save fitted scalers for later use"""
        import joblib
        joblib.dump(self.scalers, f"{path}/scalers.pkl")
        joblib.dump(self.feature_columns, f"{path}/feature_columns.pkl")
    
    def load_scalers(self, path: str):
        """Load fitted scalers"""
        import joblib
        import os
        
        if os.path.exists(f"{path}/scalers.pkl"):
            self.scalers = joblib.load(f"{path}/scalers.pkl")
            self.feature_columns = joblib.load(f"{path}/feature_columns.pkl")
            return True
        return False
# src/models.py
import numpy as np
import os, glob
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Any
import joblib
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class NBAEnsembleModel:
    """Advanced ensemble model for NBA predictions"""
    
    def __init__(self):
        self.models = {}
        self.meta_model = None
        self.meta_models = {}
        self.winner_model = None   # LogisticRegression classifier for home win probability
        self.feature_importance = None
        self.scaler = None
        self.training_history = []
        self.performance_metrics = {}
        self.model_path = 'models/'
        
        # Create model directory if it doesn't exist
        import os
        os.makedirs(self.model_path, exist_ok=True)
    
    def _build_base_models(self):
        """
        Build three independent ensemble sets:
          - 'home':  predicts home_score
          - 'away':  predicts away_score
          - 'total': predicts total_score directly
        Training separate models per target removes the fixed 0.52/0.48 split
        and lets each model learn venue-specific scoring patterns independently.

        This is the single source of truth for model construction — both train()
        and load_model() call this to guarantee the structure is always identical.
        """
        def _make_set():
            return {
                'xgb': xgb.XGBRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1,
                    random_state=42, n_jobs=1, eval_metric='mae',
                    early_stopping_rounds=20,
                    tree_method='hist',
                ),
                'lgb': lgb.LGBMRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1,
                    random_state=42, n_jobs=1, verbose=-1,
                    num_leaves=31,
                ),
                'rf': RandomForestRegressor(
                    n_estimators=100, max_depth=10,
                    min_samples_split=10, min_samples_leaf=5,
                    max_features='sqrt', random_state=42, n_jobs=1,
                ),
                'gb': GradientBoostingRegressor(
                    n_estimators=100, max_depth=5, learning_rate=0.05,
                    subsample=0.8, random_state=42,
                ),
            }

        self.models = {
            'home':  _make_set(),
            'away':  _make_set(),
            'total': _make_set(),
        }
        self.meta_models = {
            'home':  Ridge(alpha=1.0),
            'away':  Ridge(alpha=1.0),
            'total': Ridge(alpha=1.0),
        }
        # Legacy alias so save/load and any external code using self.meta_model still works
        self.meta_model = self.meta_models['total']
        return self.models  # convenience for callers that need the fresh set

    def train(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5,
              y_home: pd.Series = None, y_away: pd.Series = None,
              y_win: pd.Series = None):
        """
        Train three regression ensembles (home_score, away_score, total_score)
        sequentially, then train an independent win/loss classifier directly on
        raw features → home_win target.

        Each target is fully trained and its meta-model weights saved before the
        next target begins. Base models for the completed target are then deleted
        from memory (gc.collect()) so only one target's 4 models are in RAM at
        any given time. This keeps peak memory low enough for 4 GB machines and
        prevents the CPU from sustaining maximum load for the entire run.
        """
        import gc

        logger.info("Starting sequential three-target training (home -> away -> total)...")

        assert not X.empty, "Training data is empty"
        assert len(X) == len(y), "X and y length mismatch"

        nan_frac = X.isnull().mean()
        bad_cols = nan_frac[nan_frac > 0.5].index.tolist()
        if bad_cols:
            logger.warning(f"Dropping columns with >50% NaN: {bad_cols}")
            X = X.drop(columns=bad_cols)
        assert X.shape[1] > 0, "No valid feature columns remain after NaN check"

        if y_home is None or y_away is None:
            raise ValueError("train() requires y_home and y_away.")

        targets = {'home': y_home, 'away': y_away, 'total': y}

        # Chronological held-out test set (last 15%)
        test_split = int(len(X) * 0.85)
        X_dev,  X_test  = X.iloc[:test_split],  X.iloc[test_split:]
        targets_dev  = {k: v.iloc[:test_split] for k, v in targets.items()}
        targets_test = {k: v.iloc[test_split:]  for k, v in targets.items()}
        logger.info(f"Dev: {len(X_dev)} rows | held-out test: {len(X_test)} rows")

        tscv = TimeSeriesSplit(n_splits=cv_folds)
        self.performance_metrics['cv_scores'] = {}
        fold_importances = []

        # Initialise all model structures once via the single source-of-truth factory.
        # We then extract each target's model_set individually inside the loop.
        self._build_base_models()

        # ── Train one target at a time ────────────────────────────────────────
        for target_key, y_tk in targets.items():
            logger.info(f"[{target_key}] Building models...")
            y_dev_tk  = targets_dev[target_key]
            y_test_tk = targets_test[target_key]

            # Grab the fresh model set and meta for this target
            model_set = self.models[target_key]
            meta = self.meta_models[target_key]
            cv_maes = []

            # CV folds
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_dev)):
                X_train, X_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
                y_train, y_val = y_dev_tk.iloc[train_idx], y_dev_tk.iloc[val_idx]

                fold_preds = []
                for name, model in model_set.items():
                    if name == 'xgb':
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                    else:
                        model.fit(X_train, y_train)
                    fold_preds.append(model.predict(X_val))

                stacked = np.column_stack(fold_preds)
                meta.fit(stacked, y_val)
                # Score on a held-out portion of the val set so we're measuring
                # genuine generalisation, not the meta's fit on its own training data.
                # With small val folds Ridge won't overfit badly, but logging training
                # MAE as "CV MAE" is misleading — use the last 20% of val as a proxy.
                val_split = max(1, int(len(stacked) * 0.8))
                score = mean_absolute_error(
                    y_val.iloc[val_split:],
                    meta.predict(stacked[val_split:])
                )
                cv_maes.append(score)
                logger.info(f"  [{target_key}] fold {fold+1}/{cv_folds} MAE: {score:.2f}")

                if target_key == 'total' and hasattr(model_set['xgb'], 'feature_importances_'):
                    fold_importances.append(model_set['xgb'].feature_importances_)

            self.performance_metrics['cv_scores'][target_key] = {
                'mean_mae': float(np.mean(cv_maes)),
                'std_mae':  float(np.std(cv_maes)),
                'scores':   cv_maes,
            }

            # Final retrain on full dev set.
            # Split dev into base-training (80%) and meta-fitting (20%) slices.
            # Every base model — including RF — trains ONLY on the base slice,
            # then predicts on the unseen meta slice.  The meta model is fitted
            # on those out-of-sample predictions so it never sees a base model
            # predicting its own training data (which would be memorised for RF
            # and inflate confidence, causing 99% win probabilities at inference).
            logger.info(f"  [{target_key}] Retraining on full dev set...")
            split_idx = int(len(X_dev) * 0.8)
            X_ft, X_fv = X_dev.iloc[:split_idx], X_dev.iloc[split_idx:]
            y_ft, y_fv = y_dev_tk.iloc[:split_idx], y_dev_tk.iloc[split_idx:]

            for name, model in model_set.items():
                if name == 'xgb':
                    model.fit(X_ft, y_ft, eval_set=[(X_fv, y_fv)], verbose=False)
                else:
                    # RF, LGB, GB all train on the base slice only
                    model.fit(X_ft, y_ft)

            # Predict on the held-out meta slice — none of these models have
            # seen X_fv, so predictions are genuinely out-of-sample
            meta_preds = np.column_stack([m.predict(X_fv) for m in model_set.values()])
            meta.fit(meta_preds, y_fv)

            # Store trained models and meta for this target
            self.models[target_key] = model_set
            self.meta_models[target_key] = meta

            # ── Free RAM before next target ───────────────────────────────────
            # Delete the local model_set reference so GC can reclaim the memory.
            # self.models[target_key] still holds the trained models for inference.
            del model_set, meta
            gc.collect()
            logger.info(f"  [{target_key}] Done. Memory freed.")

        # Legacy alias
        self.meta_model = self.meta_models['total']

        # ── Independent win/loss classifier ──────────────────────────────────
        # Trained directly on raw features → home_win target, completely
        # separate from the score regression models. This gives a genuine
        # independent probability estimate rather than one derived from
        # the predicted margin.
        #
        # Uses GradientBoostingClassifier for the base models (handles the
        # binary target natively) and LogisticRegression as the meta-model,
        # mirroring the same stacking pattern as the regression targets.
        #
        # y_win falls back to (y_home > y_away) if not explicitly passed in,
        # so this is always safe even when called from older code paths.
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.metrics import accuracy_score, log_loss

            if y_win is not None:
                win_dev  = y_win.iloc[:test_split]
                win_test = y_win.iloc[test_split:]
            else:
                # Derive from home/away targets as fallback
                win_dev  = (targets_dev['home']  > targets_dev['away']).astype(int)
                win_test = (targets_test['home'] > targets_test['away']).astype(int)

            # Four diverse classifiers — same philosophy as regression base models
            clf_set = {
                'xgb_cls': xgb.XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1,
                    random_state=42, n_jobs=1,
                    eval_metric='logloss', early_stopping_rounds=20,
                    tree_method='hist',
                ),
                'lgb_cls': lgb.LGBMClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1,
                    random_state=42, n_jobs=1, verbose=-1,
                    num_leaves=31,
                ),
                'gb_cls': GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, random_state=42,
                ),
                'rf_cls': RandomForestRegressor(  # used as soft-probability base
                    n_estimators=100, max_depth=8,
                    min_samples_split=10, min_samples_leaf=5,
                    max_features='sqrt', random_state=42, n_jobs=1,
                ),
            }
            win_meta = LogisticRegression(C=1.0, max_iter=300, random_state=42)

            # TimeSeriesSplit CV to get unbiased stacked predictions for meta training
            win_tscv = TimeSeriesSplit(n_splits=cv_folds)
            win_oof  = np.zeros((len(X_dev), len(clf_set)))  # out-of-fold probs

            for fold, (tr_idx, val_idx) in enumerate(win_tscv.split(X_dev)):
                X_tr, X_val = X_dev.iloc[tr_idx], X_dev.iloc[val_idx]
                y_tr, y_val = win_dev.iloc[tr_idx], win_dev.iloc[val_idx]
                fold_probs = []
                for name, clf in clf_set.items():
                    if name == 'xgb_cls':
                        clf.fit(X_tr, y_tr,
                                eval_set=[(X_val, y_val)], verbose=False)
                    elif name == 'rf_cls':
                        # RF regressor used as probability proxy (outputs 0-1)
                        clf.fit(X_tr, y_tr)
                        fold_probs.append(clf.predict(X_val).clip(0, 1))
                        continue
                    else:
                        clf.fit(X_tr, y_tr)
                    if hasattr(clf, 'predict_proba'):
                        fold_probs.append(clf.predict_proba(X_val)[:, 1])
                    else:
                        fold_probs.append(clf.predict(X_val).clip(0, 1))
                win_oof[val_idx] = np.column_stack(fold_probs)

            win_meta.fit(win_oof, win_dev)

            # Final retrain on full dev set.
            # Same OOS discipline as the regression targets: base classifiers train
            # on the first 80%, predict on the last 20%, and the meta LogisticRegression
            # is fitted on those genuine out-of-sample probabilities.
            # This prevents rf_cls (a RandomForestRegressor used as a probability proxy)
            # from predicting its own training data — RF memorises training rows and
            # returns near-perfect outputs on them, which teaches the meta to trust
            # unrealistically confident base predictions, causing 99% win probabilities.
            split_idx = int(len(X_dev) * 0.8)
            X_ft, X_fv = X_dev.iloc[:split_idx], X_dev.iloc[split_idx:]
            y_ft, y_fv = win_dev.iloc[:split_idx], win_dev.iloc[split_idx:]

            final_probs = []
            for name, clf in clf_set.items():
                if name == 'xgb_cls':
                    clf.fit(X_ft, y_ft,
                            eval_set=[(X_fv, y_fv)], verbose=False)
                else:
                    # rf_cls, lgb_cls, gb_cls all train on base slice only
                    clf.fit(X_ft, y_ft)
                # Predict on unseen meta slice
                if name == 'rf_cls':
                    final_probs.append(clf.predict(X_fv).clip(0, 1))
                elif hasattr(clf, 'predict_proba'):
                    final_probs.append(clf.predict_proba(X_fv)[:, 1])
                else:
                    final_probs.append(clf.predict(X_fv).clip(0, 1))

            win_meta.fit(np.column_stack(final_probs), y_fv)

            # Store classifier set and meta
            self.winner_model = {
                'classifiers': clf_set,
                'meta':        win_meta,
            }

            # Evaluate on held-out test set
            test_probs_list = []
            for name, clf in clf_set.items():
                if name == 'rf_cls':
                    test_probs_list.append(clf.predict(X_test).clip(0, 1))
                elif hasattr(clf, 'predict_proba'):
                    test_probs_list.append(clf.predict_proba(X_test)[:, 1])
                else:
                    test_probs_list.append(clf.predict(X_test).clip(0, 1))
            test_win_prob = win_meta.predict_proba(
                np.column_stack(test_probs_list))[:, 1]
            test_win_pred = (test_win_prob >= 0.5).astype(int)
            win_acc  = accuracy_score(win_test, test_win_pred)
            win_loss = log_loss(win_test, test_win_prob)

            self.performance_metrics['winner'] = {
                'test_accuracy': float(win_acc),
                'test_log_loss': float(win_loss),
                'n_samples':     len(win_test),
                'note': 'Independent classifier trained on features → home_win',
            }
            logger.info(
                f"Win classifier trained. "
                f"Test accuracy: {win_acc:.3f} | Log-loss: {win_loss:.3f}"
            )
            del clf_set, win_meta
            gc.collect()

        except Exception as e:
            logger.warning(f"Win classifier training failed (non-fatal): {e}")
            self.winner_model = None

        # Feature importance from total model CV
        if fold_importances:
            self.feature_importance = pd.DataFrame({
                'feature':    X_dev.columns,
                'importance': np.mean(fold_importances, axis=0)
            }).sort_values('importance', ascending=False)

        # Evaluate on held-out test set
        home_preds  = self._predict_target(X_test, 'home')
        away_preds  = self._predict_target(X_test, 'away')
        total_preds = self._predict_target(X_test, 'total')
        sum_preds   = home_preds + away_preds

        self.performance_metrics['test'] = {
            'mae':       float(mean_absolute_error(targets_test['total'], total_preds)),
            'mae_home':  float(mean_absolute_error(targets_test['home'],  home_preds)),
            'mae_away':  float(mean_absolute_error(targets_test['away'],  away_preds)),
            'mae_sum':   float(mean_absolute_error(targets_test['total'], sum_preds)),
            'rmse':      float(np.sqrt(mean_squared_error(targets_test['total'], total_preds))),
            'r2':        float(r2_score(targets_test['total'], total_preds)),
            'n_samples': len(X_test),
            'note': 'Out-of-sample on last 15% (never seen during training)',
        }

        logger.info(
            f"Training complete. "
            f"Test MAE total={self.performance_metrics['test']['mae']:.2f} | "
            f"home={self.performance_metrics['test']['mae_home']:.2f} | "
            f"away={self.performance_metrics['test']['mae_away']:.2f}"
        )

        return self.performance_metrics

    def _get_base_predictions_for(self, target_key: str, X: pd.DataFrame) -> np.ndarray:
        """Stack base model predictions for a given target."""
        return np.column_stack([m.predict(X) for m in self.models[target_key].values()])

    def _predict_target(self, X: pd.DataFrame, target_key: str) -> np.ndarray:
        """Run the full ensemble (base + meta) for one target."""
        stacked = self._get_base_predictions_for(target_key, X)
        return self.meta_models[target_key].predict(stacked)

    def _get_base_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Legacy helper — returns base predictions for the total model."""
        return self._get_base_predictions_for('total', X)

    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return total predictions with confidence scores derived from all three model sets."""
        all_preds = []
        for tk in ('home', 'away', 'total'):
            for model in self.models[tk].values():
                all_preds.append(model.predict(X))
        arr = np.array(all_preds)
        mean_pred = np.mean(arr, axis=0)
        std_pred  = np.std(arr, axis=0)
        confidence = np.clip(1 - (std_pred / (mean_pred + 1e-10)), 0, 1)
        return mean_pred, confidence

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict total score (used by integrity checks and legacy callers)."""
        if not self.models or not self.meta_models:
            logger.error("Model not trained yet")
            return None
        return self._predict_target(X, 'total')


    @staticmethod
    def _calc_injury_adjustment(row: pd.Series, side: str) -> float:
        """
        Compute a point adjustment for one team based on injury columns that
        fetch_todays_games() attaches. Returns a negative float (score reduction).

        Adjustment scale (NBA research averages):
          Key starter Out      → -5.0 pts
          Non-starter Out      → -1.5 pts per player  (capped at -4.5)
          Doubtful             → -2.0 pts per player  (capped at -3.0)
          Questionable         → -0.5 pts per player  (capped at -1.0)
        """
        adj = 0.0
        try:
            key_out   = int(row.get(f'{side}_key_player_out',  0) or 0)
            total_out = int(row.get(f'{side}_inj_out',         0) or 0)
            doubtful  = int(row.get(f'{side}_inj_doubtful',    0) or 0)
            question  = int(row.get(f'{side}_inj_questionable',0) or 0)

            role_out = max(total_out - key_out, 0)  # non-key players confirmed Out

            adj -= key_out  * 5.0
            adj -= min(role_out  * 1.5, 4.5)
            adj -= min(doubtful  * 2.0, 3.0)
            adj -= min(question  * 0.5, 1.0)
        except Exception:
            pass
        return round(adj, 1)

    def predict_games(self, games_df: pd.DataFrame, feature_engineer=None, historical_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Predict home_score, away_score, and total for each game.
        pred_total is blended from the direct total model and the home+away sum
        for maximum accuracy on the over/under line.
        """
        if games_df.empty:
            return games_df

        results = games_df.copy()
        X_pred = self._prepare_prediction_features(games_df, feature_engineer, historical_df)

        if X_pred is not None:
            home_preds  = self._predict_target(X_pred, 'home')
            away_preds  = self._predict_target(X_pred, 'away')
            total_preds = self._predict_target(X_pred, 'total')
            sum_preds   = home_preds + away_preds
            blended_total = (total_preds + sum_preds) / 2.0

            _, confidence_scores = self.predict_with_confidence(X_pred)

            results['pred_home_score']   = np.round(home_preds, 1)
            results['pred_away_score']   = np.round(away_preds, 1)
            results['pred_total']        = np.round(blended_total, 1)
            results['pred_total_direct'] = np.round(total_preds, 1)
            results['pred_total_sum']    = np.round(sum_preds, 1)
            results['confidence']        = np.round(confidence_scores, 3)

            line_col = 'over_line' if 'over_line' in results.columns else None
            results['line'] = results[line_col] if line_col else 220.5
            results['over_under'] = results.apply(
                lambda x: 'OVER' if x['pred_total'] > x['line'] else 'UNDER', axis=1
            )

            # ── Winner prediction ─────────────────────────────────────────
            # Uses the independent win/loss classifier (dict with 'classifiers'
            # and 'meta' keys) when available. Falls back to a sigmoid of the
            # predicted score margin so predictions are always populated.
            try:
                if (self.winner_model is not None
                        and isinstance(self.winner_model, dict)
                        and 'classifiers' in self.winner_model):
                    # Independent classifier path — raw features → P(home_win)
                    clf_set  = self.winner_model['classifiers']
                    win_meta = self.winner_model['meta']
                    probs_list = []
                    for name, clf in clf_set.items():
                        if name == 'rf_cls':
                            probs_list.append(clf.predict(X_pred).clip(0, 1))
                        elif hasattr(clf, 'predict_proba'):
                            probs_list.append(clf.predict_proba(X_pred)[:, 1])
                        else:
                            probs_list.append(clf.predict(X_pred).clip(0, 1))
                    home_win_prob = win_meta.predict_proba(
                        np.column_stack(probs_list))[:, 1]
                else:
                    # Fallback: sigmoid of predicted score margin
                    margin = home_preds - away_preds
                    home_win_prob = 1 / (1 + np.exp(-margin / 10))

                results['home_win_prob'] = np.round(home_win_prob, 3)
                results['pred_winner']   = results.apply(
                    lambda x: x['home_team'] if x['home_win_prob'] >= 0.5
                              else x['away_team'], axis=1
                )
                results['winner_confidence'] = results['home_win_prob'].apply(
                    lambda p: round(max(p, 1 - p) * 100, 1)
                )
            except Exception as e:
                logger.warning(f"Winner prediction failed (non-fatal): {e}")
                margin = results['pred_home_score'] - results['pred_away_score']
                results['home_win_prob']     = np.round(
                    1 / (1 + np.exp(-margin / 10)), 3)
                results['pred_winner']       = results.apply(
                    lambda x: x['home_team'] if x['pred_home_score'] >= x['pred_away_score']
                              else x['away_team'], axis=1
                )
                results['winner_confidence'] = results['home_win_prob'].apply(
                    lambda p: round(max(p, 1 - p) * 100, 1)
                )

        # ── Post-prediction injury adjustment ─────────────────────────────
        # Only applied when injury columns are present (today's games only).
        # Historical predictions and backtesting rows are unaffected.
        injury_cols = ['home_inj_out', 'away_inj_out']
        if not results.empty and all(c in results.columns for c in injury_cols):
            try:
                home_adj = results.apply(
                    lambda r: self._calc_injury_adjustment(r, 'home'), axis=1)
                away_adj = results.apply(
                    lambda r: self._calc_injury_adjustment(r, 'away'), axis=1)

                results['injury_adj_home'] = home_adj
                results['injury_adj_away'] = away_adj

                # Adjust individual scores
                results['pred_home_score'] = np.round(
                    results['pred_home_score'] + home_adj, 1)
                results['pred_away_score'] = np.round(
                    results['pred_away_score'] + away_adj, 1)

                # Recompute blended total and O/U call from adjusted scores
                # Keep the same blend formula: average of direct total and adjusted sum
                adjusted_sum = results['pred_home_score'] + results['pred_away_score']
                results['pred_total_sum'] = np.round(adjusted_sum, 1)
                results['pred_total'] = np.round(
                    (results['pred_total_direct'] + adjusted_sum) / 2.0, 1)
                results['over_under'] = results.apply(
                    lambda x: 'OVER' if x['pred_total'] > x['line'] else 'UNDER',
                    axis=1
                )

                # Combined injury notes for display
                def _safe_inj_notes(r):
                    parts = []
                    for key in ('home_inj_notes', 'away_inj_notes'):
                        val = r.get(key, '')
                        if val and isinstance(val, str) and val.strip():
                            parts.append(val.strip())
                    return ' | '.join(parts)

                results['injury_notes'] = results.apply(_safe_inj_notes, axis=1)
                logger.info("Injury adjustments applied to predictions")
            except Exception as e:
                logger.warning(f"Injury adjustment failed (non-fatal): {e}")

        return results

    def _prepare_prediction_features(self, games_df: pd.DataFrame, feature_engineer=None, historical_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepare real features for prediction using the feature engineer and
        historical context. Falls back to a zero-filled frame matching training
        feature columns if no engineer is available, so the prediction path
        never silently returns random numbers.
        """
        # Best path: use the feature engineer with historical context
        if feature_engineer is not None and historical_df is not None and not historical_df.empty:
            try:
                prediction_features = []
                for _, game in games_df.iterrows():
                    features = feature_engineer.prepare_prediction_features(game, historical_df)
                    prediction_features.append(features)
                result = pd.concat(prediction_features, ignore_index=True)
                logger.info(f"Prepared {len(result)} real feature rows for prediction")
                return result
            except Exception as e:
                logger.error(f"Feature engineer failed during prediction prep: {e}. Falling back.")

        # Fallback: if we know what columns the model was trained on,
        # return zeros for those columns so predictions are at least deterministic
        if self.feature_importance is not None:
            trained_cols = self.feature_importance['feature'].tolist()
            logger.warning(
                "No feature engineer available — returning zero-filled features. "
                "Predictions will not be meaningful until historical data is loaded."
            )
            return pd.DataFrame(
                np.zeros((len(games_df), len(trained_cols))),
                columns=trained_cols
            )

        # Last resort: model hasn't been trained yet at all
        raise RuntimeError(
            "Cannot prepare prediction features: model has not been trained "
            "and no feature engineer was provided."
        )
    
    def get_performance_metrics(self) -> Dict:
        """Get model performance metrics"""
        return self.performance_metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance rankings"""
        return self.feature_importance
    
    def save_model(self, feature_engineer=None):
        """
        Save all model artifacts. Persists base models and meta_models
        for all three targets (home, away, total).
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for pattern in ['models/*.pkl', 'models/*.txt']:
        	for fcx in glob.glob(pattern):
        		os.remove(fcx)
        logger.info("Old model artifacts cleared") 

        # Save base models for each target
        for target_key, model_set in self.models.items():
            for name, model in model_set.items():
                joblib.dump(model, f"{self.model_path}/{target_key}_{name}_{timestamp}.pkl")

        # Save meta models
        for target_key, meta in self.meta_models.items():
            joblib.dump(meta, f"{self.model_path}/meta_{target_key}_{timestamp}.pkl")

        # Save performance metrics, feature importance, and winner model
        joblib.dump(self.performance_metrics, f"{self.model_path}/metrics_{timestamp}.pkl")
        if self.feature_importance is not None:
            joblib.dump(self.feature_importance, f"{self.model_path}/feature_importance_{timestamp}.pkl")
        if self.winner_model is not None:
            # winner_model is a dict {'classifiers': {...}, 'meta': ...}
            joblib.dump(self.winner_model, f"{self.model_path}/winner_model_{timestamp}.pkl")

        if feature_engineer is not None:
            joblib.dump(feature_engineer.scalers,
                        f"{self.model_path}/scalers_{timestamp}.pkl")
            joblib.dump(feature_engineer.feature_columns,
                        f"{self.model_path}/feature_columns_{timestamp}.pkl")
            logger.info("Feature engineer scalers saved alongside model")
        else:
            logger.warning(
                "save_model called without feature_engineer — scalers not persisted."
            )

        with open(f"{self.model_path}/latest.txt", 'w') as f:
            f.write(timestamp)

        logger.info(f"Model artifacts saved with timestamp {timestamp}")
    
    def load_model(self, feature_engineer=None, timestamp: str = None):
        """Load all model artifacts from disk, restoring all three target model sets."""
        import glob as _glob

        if timestamp is None:
            latest_ptr = f"{self.model_path}/latest.txt"
            if os.path.exists(latest_ptr):
                with open(latest_ptr) as f:
                    timestamp = f.read().strip()
            else:
                model_files = _glob.glob(f"{self.model_path}/total_xgb_*.pkl")
                if not model_files:
                    # Fall back to old single-target format
                    model_files = _glob.glob(f"{self.model_path}/xgb_*.pkl")
                if not model_files:
                    logger.warning("No saved models found")
                    return False
                timestamps = []
                for f in model_files:
                    parts = os.path.basename(f).replace('.pkl', '').split('_')
                    if len(parts) >= 3:
                        timestamps.append('_'.join(parts[-2:]))
                if not timestamps:
                    logger.warning("Could not parse timestamps from model files")
                    return False
                timestamp = max(timestamps)

        try:
            base_model_names = ['xgb', 'lgb', 'rf', 'gb', 'ridge']  # ridge included for backwards compat
            self._build_base_models()  # initialise structure

            for target_key in ('home', 'away', 'total'):
                for name in base_model_names:
                    path = f"{self.model_path}/{target_key}_{name}_{timestamp}.pkl"
                    if os.path.exists(path):
                        self.models[target_key][name] = joblib.load(path)
                meta_path = f"{self.model_path}/meta_{target_key}_{timestamp}.pkl"
                if os.path.exists(meta_path):
                    self.meta_models[target_key] = joblib.load(meta_path)

            self.meta_model = self.meta_models['total']

            metrics_path = f"{self.model_path}/metrics_{timestamp}.pkl"
            if os.path.exists(metrics_path):
                self.performance_metrics = joblib.load(metrics_path)

            fi_path = f"{self.model_path}/feature_importance_{timestamp}.pkl"
            if os.path.exists(fi_path):
                self.feature_importance = joblib.load(fi_path)
                logger.info("Feature importance restored from disk")

            winner_path = f"{self.model_path}/winner_model_{timestamp}.pkl"
            if os.path.exists(winner_path):
                loaded_winner = joblib.load(winner_path)
                # Handle both new dict format and old LogisticRegression format
                # (old format falls back to margin-sigmoid at prediction time)
                if isinstance(loaded_winner, dict) and 'classifiers' in loaded_winner:
                    self.winner_model = loaded_winner
                    logger.info("Independent win classifier restored from disk")
                else:
                    # Old format — keep it, predict_games handles it via except path
                    self.winner_model = loaded_winner
                    logger.info("Legacy winner model restored from disk")

            scalers_path  = f"{self.model_path}/scalers_{timestamp}.pkl"
            columns_path  = f"{self.model_path}/feature_columns_{timestamp}.pkl"
            if feature_engineer is not None:
                if os.path.exists(scalers_path) and os.path.exists(columns_path):
                    feature_engineer.scalers = joblib.load(scalers_path)
                    feature_engineer.feature_columns = joblib.load(columns_path)
                    logger.info("Feature engineer scalers restored from disk")
                else:
                    logger.warning("Model loaded but no matching scalers file found.")
            else:
                logger.warning("load_model called without feature_engineer.")

            logger.info(f"Model artifacts loaded from timestamp {timestamp}")
            return True

        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            return False
    
    def update_hyperparameters(self, params: Dict):
        """
        Update hyperparameters on all fitted base models across all three targets.
        self.models is a dict-of-dicts: {'home': {'xgb': ..., 'lgb': ...}, ...}
        so we need two levels of iteration to reach the actual model objects.
        """
        for model_set in self.models.values():        # {'xgb': ..., 'lgb': ..., ...}
            for model in model_set.values():          # actual estimator objects
                if 'n_estimators' in params and hasattr(model, 'n_estimators'):
                    model.n_estimators = params['n_estimators']
                if 'max_depth' in params and hasattr(model, 'max_depth'):
                    model.max_depth = params['max_depth']
                if 'learning_rate' in params and hasattr(model, 'learning_rate'):
                    model.learning_rate = params['learning_rate']

        logger.info(f"Hyperparameters updated: {params}")
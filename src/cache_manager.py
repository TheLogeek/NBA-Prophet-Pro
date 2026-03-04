# src/cache_manager.py
import os
import pandas as pd
import pickle
from datetime import datetime, timedelta
import hashlib
import json
import logging
from typing import Optional, Any, Dict, List

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching for NBA data"""
    
    def __init__(self, cache_dir: str = 'cache'):
        self.cache_dir = cache_dir
        
        # Ensure directory exists with proper permissions
        try:
            os.makedirs(cache_dir, mode=0o755, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(cache_dir, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            logger.error(f"Cannot write to cache directory: {e}")
            # Fallback to temp directory
            import tempfile
            self.cache_dir = tempfile.mkdtemp()
            logger.info(f"Using temp directory: {self.cache_dir}")
        
        # Set file paths only after cache_dir is fully resolved
        self.history_file = os.path.join(self.cache_dir, 'historical_data.csv')
        self.today_file = os.path.join(self.cache_dir, 'todays_games.csv')
        self.predictions_file = os.path.join(self.cache_dir, 'predictions_history.csv')
        self.last_update_file = os.path.join(self.cache_dir, 'last_update.txt')
        self.model_status_file = os.path.join(self.cache_dir, 'model_status.txt')
        self.backtest_file = os.path.join(self.cache_dir, 'backtest_results.csv')
        
        # Initialize tracking files
        self._init_tracking_files()
    
    def _init_tracking_files(self):
        """Initialize tracking files if they don't exist"""
        if not os.path.exists(self.last_update_file):
            with open(self.last_update_file, 'w') as f:
                f.write('1970-01-01 00:00:00')
        
        if not os.path.exists(self.model_status_file):
            with open(self.model_status_file, 'w') as f:
                f.write('not_trained')
    
    def save_historical_data(self, df: pd.DataFrame):
        """Save historical data to cache"""
        try:
            # Append if file exists, otherwise create new
            if os.path.exists(self.history_file):
                existing = pd.read_csv(self.history_file)
                combined = pd.concat([existing, df]).drop_duplicates(
                    subset=['date', 'home_team', 'away_team'], keep='last'
                )
                combined.to_csv(self.history_file, index=False)
            else:
                df.to_csv(self.history_file, index=False)
            
            logger.info(f"Saved {len(df)} historical games to cache")
            
        except Exception as e:
            logger.error(f"Failed to save historical data: {str(e)}")
    
    def load_historical_data(self) -> pd.DataFrame:
        """Load historical data from cache"""
        try:
            if os.path.exists(self.history_file):
                df = pd.read_csv(self.history_file)
                logger.info(f"Loaded {len(df)} historical games from cache")
                return df
            else:
                logger.info("No historical data found in cache")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to load historical data: {str(e)}")
            return pd.DataFrame()
    
    def save_todays_games(self, df: pd.DataFrame):
        """Save today's games to cache"""
        try:
            df.to_csv(self.today_file, index=False)
            logger.info(f"Saved {len(df)} today's games to cache")
        except Exception as e:
            logger.error(f"Failed to save today's games: {str(e)}")
    
    def load_todays_games(self) -> Optional[pd.DataFrame]:
        """Load today's games from cache"""
        try:
            if os.path.exists(self.today_file):
                # Check if cache is from today
                mod_time = datetime.fromtimestamp(os.path.getmtime(self.today_file))
                if mod_time.date() == datetime.now().date():
                    df = pd.read_csv(self.today_file)
                    logger.info(f"Loaded {len(df)} today's games from cache")
                    return df
                else:
                    logger.info("Today's games cache is outdated")
                    return None
            return None
        except Exception as e:
            logger.error(f"Failed to load today's games: {str(e)}")
            return None
    
    def save_predictions(self, df: pd.DataFrame):
        """Save predictions to history"""
        try:
            # Add timestamp
            df['prediction_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Append to existing file
            if os.path.exists(self.predictions_file):
                existing = pd.read_csv(self.predictions_file)
                combined = pd.concat([existing, df]).drop_duplicates(
                    subset=['date', 'home_team', 'away_team', 'prediction_timestamp'], 
                    keep='last'
                )
                combined.to_csv(self.predictions_file, index=False)
            else:
                df.to_csv(self.predictions_file, index=False)
            
            logger.info(f"Saved {len(df)} predictions to history")
            
        except Exception as e:
            logger.error(f"Failed to save predictions: {str(e)}")
    
    def load_historical_predictions(self) -> pd.DataFrame:
        """Load historical predictions"""
        try:
            if os.path.exists(self.predictions_file):
                df = pd.read_csv(self.predictions_file)
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to load predictions: {str(e)}")
            return pd.DataFrame()
    
    def update_last_update(self):
        """Update last update timestamp"""
        try:
            with open(self.last_update_file, 'w') as f:
                f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        except Exception as e:
            logger.error(f"Failed to update last update timestamp: {str(e)}")
    
    def get_last_update(self) -> str:
        """Get last update timestamp"""
        try:
            with open(self.last_update_file, 'r') as f:
                return f.read().strip()
        except:
            return "Never"
    
    def set_model_status(self, status: str):
        """Set model training status"""
        try:
            with open(self.model_status_file, 'w') as f:
                f.write(status)
        except Exception as e:
            logger.error(f"Failed to set model status: {str(e)}")
    
    def get_model_status(self) -> str:
        """Get model training status"""
        try:
            with open(self.model_status_file, 'r') as f:
                return f.read().strip()
        except:
            return "not_trained"
    
    def clear_cache(self):
        """Clear all cache files"""
        try:
            for file in [self.history_file, self.today_file, 
                        self.predictions_file, self.last_update_file,
                        self.model_status_file]:
                if os.path.exists(file):
                    os.remove(file)
            
            # Reinitialize tracking files
            self._init_tracking_files()
            
            logger.info("Cache cleared successfully")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
    
    def list_cache_files(self) -> List[Dict]:
        """List all cache files with metadata"""
        files = []
        try:
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.isfile(filepath):
                    stats = os.stat(filepath)
                    files.append({
                        'filename': filename,
                        'size': f"{stats.st_size / 1024:.2f} KB",
                        'modified': datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'created': datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
                    })
        except Exception as e:
            logger.error(f"Failed to list cache files: {str(e)}")
        
        return files
    
    def get_cache_size(self) -> float:
        """Get total cache size in MB"""
        total_size = 0
        try:
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)
        except:
            pass
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def update_settings(self, settings: Dict):
        """Update cache settings"""
        settings_file = os.path.join(self.cache_dir, 'settings.json')
        try:
            with open(settings_file, 'w') as f:
                json.dump(settings, f)
            logger.info("Settings saved")
        except Exception as e:
            logger.error(f"Failed to save settings: {str(e)}")
    
    def load_settings(self) -> Dict:
        """Load cache settings"""
        settings_file = os.path.join(self.cache_dir, 'settings.json')
        try:
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load settings: {str(e)}")
        
        return {}
    
    def save_backtest_results(self, df: pd.DataFrame):
        """Append new backtested prediction rows to the backtest results file."""
        try:
            if os.path.exists(self.backtest_file):
                existing = pd.read_csv(self.backtest_file)
                combined = pd.concat([existing, df]).drop_duplicates(
                    subset=['date', 'home_team', 'away_team'], keep='last'
                )
                combined.to_csv(self.backtest_file, index=False)
            else:
                df.to_csv(self.backtest_file, index=False)
            logger.info(f"Saved {len(df)} backtest rows")
        except Exception as e:
            logger.error(f"Failed to save backtest results: {e}")

    def load_backtest_results(self) -> pd.DataFrame:
        """Load all backtest results from disk."""
        try:
            if os.path.exists(self.backtest_file):
                return pd.read_csv(self.backtest_file)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to load backtest results: {e}")
            return pd.DataFrame()

    def get_cache_info(self) -> Dict:
        """Get comprehensive cache information"""
        return {
            'cache_directory': self.cache_dir,
            'total_size_mb': self.get_cache_size(),
            'files': self.list_cache_files(),
            'last_update': self.get_last_update(),
            'model_status': self.get_model_status(),
            'historical_data_exists': os.path.exists(self.history_file),
            'todays_games_exists': os.path.exists(self.today_file),
            'predictions_history_exists': os.path.exists(self.predictions_file)
        }
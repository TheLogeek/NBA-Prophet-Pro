# main.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import os
import logging
from typing import Tuple, Dict, List
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nba_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="NBA Prophet Pro",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import other modules after page config
from src.data_collector import NBADataCollector
from src.feature_engineering import NBAFeatureEngineer
from src.models import NBAEnsembleModel
from src.visualization import create_visualizations
from src.cache_manager import CacheManager
from src.ui_components import UIManager

# Custom CSS
def load_css():
    st.markdown("""
    <style>
        /* Global Styles */
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Card Styles */
        .prediction-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 25px;
            margin: 15px 0;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
        }
        
        .prediction-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 50px rgba(0, 0, 0, 0.3);
        }
        
        /* Team Score Styles */
        .team-score {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 10px 0;
        }
        
        /* Metric Styles */
        .metric-container {
            background: white;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #2c3e50;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Button Styles */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1.1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
            border-color: white;
        }
        
        /* Header Styles */
        .app-header {
            text-align: center;
            padding: 40px 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 30px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .app-title {
            font-size: 4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #fff 0%, #e0e0e0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .app-subtitle {
            font-size: 1.2rem;
            color: rgba(255, 255, 255, 0.9);
            font-weight: 300;
        }
        
        /* Analytics Page Styles */
        .analytics-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            margin: 20px 0;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        }
        
        /* Progress Bar */
        .stProgress > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Sidebar */
        .css-1d391kg {
            background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 50px;
            backdrop-filter: blur(10px);
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 50px;
            padding: 10px 25px;
            color: white;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        /* DataFrames */
        .dataframe {
            background: white;
            border-radius: 15px;
            padding: 20px;
            border: none;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
        }
        
        /* Success/Warning Messages */
        .success-message {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            color: white;
            padding: 15px;
            border-radius: 15px;
            font-weight: 600;
            text-align: center;
        }
        
        .warning-message {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            color: white;
            padding: 15px;
            border-radius: 15px;
            font-weight: 600;
            text-align: center;
        }
        
        /* Animations */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .pulse-animation {
            animation: pulse 2s infinite;
        }
        
        /* Loading Spinner */
        .loading-spinner {
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top: 4px solid white;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables with conflict prevention"""
    required_keys = {
        'initialized': False,
        'model_trained': False,
        'data_collected': False,
        'predictions_made': False,
        'current_page': "Predictions",
        'today_predictions': None,
        'cache_manager': None,
        'data_collector': None,
        'feature_engineer': None,
        'model': None,
        'ui_manager': None
    }
    
    for key, default_value in required_keys.items():
        if key not in st.session_state:
            if key in ['cache_manager', 'data_collector', 'feature_engineer', 'model', 'ui_manager']:
                # Lazy initialization
                st.session_state[key] = None
            else:
                st.session_state[key] = default_value
    
    # Initialize objects only once
    if st.session_state.cache_manager is None:
        st.session_state.cache_manager = CacheManager()
    if st.session_state.data_collector is None:
        st.session_state.data_collector = NBADataCollector(st.session_state.cache_manager)
    if st.session_state.feature_engineer is None:
        st.session_state.feature_engineer = NBAFeatureEngineer()
    if st.session_state.model is None:
        st.session_state.model = NBAEnsembleModel()
    if st.session_state.ui_manager is None:
        st.session_state.ui_manager = UIManager()
    
    # On first load, attempt to restore a previously trained model and its scalers
    # so users don't have to retrain after every restart
    if not st.session_state.model_trained:
        loaded = st.session_state.model.load_model(
            feature_engineer=st.session_state.feature_engineer
        )
        if loaded:
            st.session_state.model_trained = True
            st.session_state.data_collected = True
            logger.info("Restored saved model and scalers from disk on startup")
            # Silently append yesterday's games to keep data current
            try:
                st.session_state.data_collector.update_historical_data()
            except Exception as e:
                logger.warning(f"Yesterday update failed on startup: {e}")
            # Auto-retrain if last training was more than 7 days ago
            try:
                last_update_str = st.session_state.cache_manager.get_last_update()
                last_update = datetime.strptime(last_update_str, '%Y-%m-%d %H:%M:%S')
                days_since_train = (datetime.now() - last_update).days
                if days_since_train >= 7:
                    logger.info(f"Model is {days_since_train} days old — scheduling background retrain")
                    st.session_state['needs_retrain'] = True
                else:
                    st.session_state['needs_retrain'] = False
            except Exception as e:
                logger.warning(f"Could not check model age: {e}")
                st.session_state['needs_retrain'] = False

def main():
    """Main application entry point"""
    # Load CSS
    load_css()
    
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">🏀 NBA Prophet Pro</h1>
        <p class="app-subtitle">Advanced AI-Powered NBA Game Predictions with Ensemble Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🎯 Navigation")
        pages = ["Predictions", "Backtesting", "Model Performance", "Data Management", "Settings"]
        
        for page in pages:
            if st.button(page, key=f"nav_{page}"):
                st.session_state.current_page = page
                st.rerun()
        
        st.markdown("---")
        
        # System status
        st.markdown("## 📊 System Status")
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            st.markdown("**Model Status**")
            if st.session_state.model_trained:
                st.success("✅ Trained")
            else:
                st.warning("⏳ Not Trained")
        
        with status_col2:
            st.markdown("**Data Status**")
            if st.session_state.data_collected:
                st.success("✅ Ready")
            else:
                st.warning("⏳ Pending")
        
        # Last update
        st.markdown("---")
        last_update = st.session_state.cache_manager.get_last_update()
        st.markdown(f"**Last Update:** {last_update}")
        
        # Cache info
        cache_size = st.session_state.cache_manager.get_cache_size()
        st.markdown(f"**Cache Size:** {cache_size:.2f} MB")
    
    # Main content area
    if st.session_state.current_page == "Predictions":
        show_predictions_page()
    #elif st.session_state.current_page == "Analytics":
        #show_analytics_page()
    elif st.session_state.current_page == "Backtesting":
        show_backtesting_page()
    elif st.session_state.current_page == "Model Performance":
        show_model_performance_page()
    elif st.session_state.current_page == "Data Management":
        show_data_management_page()
    elif st.session_state.current_page == "Settings":
        show_settings_page()

def show_predictions_page():
    """Display predictions page"""
    st.markdown("## 🎯 Today's Game Predictions")
    
    # Stale model warning — shown when auto-retrain is needed
    if st.session_state.get('needs_retrain', False):
        st.warning(
            "⚠️ Model was trained more than 7 days ago. "
            "Predictions may be stale. Consider retraining."
        )
        if st.button("🔄 Retrain Now", key="retrain_stale"):
            with st.spinner("Retraining model with latest data..."):
                initialize_system()
            st.session_state['needs_retrain'] = False
    
    # Check if model needs training
    if not st.session_state.model_trained:
        st.markdown("""
        <div class="warning-message">
            ⚠️ Model not trained yet. Please initialize the system first.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Initialize System", key="init_system"):
            with st.spinner("Initializing system..."):
                initialize_system()
    
    else:
        today_str = datetime.now().strftime('%Y-%m-%d')
        predictions_cache_file = os.path.join('cache', f'predictions_{today_str}.csv')

        # Load from session state first (navigation within same session)
        predictions = st.session_state.get('today_predictions')

        # If not in session state, try loading from today's cache file
        if predictions is None and os.path.exists(predictions_cache_file):
            try:
                predictions = pd.read_csv(predictions_cache_file)
                st.session_state.today_predictions = predictions
                logger.info(f"Loaded today's predictions from cache file")
            except Exception as e:
                logger.warning(f"Could not load predictions cache: {e}")
                predictions = None

        # If still no predictions, generate them now
        if predictions is None:
            games_df = st.session_state.data_collector.fetch_todays_games()
            if games_df.empty:
                st.markdown("""
                <div class="warning-message">
                    🏀 No games scheduled for today
                </div>
                """, unsafe_allow_html=True)
                predictions = pd.DataFrame()
            else:
                historical_df = st.session_state.data_collector.load_historical_data()
                predictions = st.session_state.model.predict_games(
                    games_df,
                    feature_engineer=st.session_state.feature_engineer,
                    historical_df=historical_df
                )
                # Save to session state and to today's cache file
                st.session_state.today_predictions = predictions
                try:
                    os.makedirs('cache', exist_ok=True)
                    predictions.to_csv(predictions_cache_file, index=False)
                    logger.info(f"Saved today's predictions to cache file")
                except Exception as e:
                    logger.warning(f"Could not save predictions cache: {e}")
                # Also save to predictions_history.csv so backtesting and analytics work
                try:
                    st.session_state.cache_manager.save_predictions(predictions.copy())
                except Exception as e:
                    logger.warning(f"Could not save predictions to history: {e}")

        if predictions is not None and not predictions.empty:
            col_refresh, _ = st.columns([1, 4])
            with col_refresh:
                if st.button("🔄 Refresh Predictions", key="refresh_preds"):
                    st.session_state.today_predictions = None
                    if os.path.exists(predictions_cache_file):
                        os.remove(predictions_cache_file)
                    st.rerun()
            
            # Display predictions in a beautiful grid
            for idx, game in predictions.iterrows():
                line_missing = pd.isna(game.get('line')) or game.get('line') == 0
                ou_color     = "#43e97b" if game["over_under"] == "OVER" else ("#fa709a" if game["over_under"] == "UNDER" else "#aaaaaa")
                confidence   = game.get("confidence", None)
                conf_pct     = f"{confidence * 100:.0f}%" if confidence is not None else "N/A"
                if confidence is not None and confidence >= 0.70:
                    conf_color = "#43e97b"
                elif confidence is not None and confidence >= 0.50:
                    conf_color = "#f9a825"
                else:
                    conf_color = "#fa709a"
                direct_total = game.get("pred_total_direct", None)
                sum_total    = game.get("pred_total_sum",    None)
                direct_str   = f"{direct_total:.1f}" if direct_total is not None else "N/A"
                sum_str      = f"{sum_total:.1f}"    if sum_total    is not None else "N/A"
                conf_val = confidence if confidence is not None else 0.0
                spread   = abs(direct_total - sum_total) if (direct_total is not None and sum_total is not None) else 999
                if   spread <= 3  and conf_val >= 0.70: stars = 5
                elif spread <= 5  and conf_val >= 0.60: stars = 4
                elif spread <= 7  and conf_val >= 0.50: stars = 3
                elif spread <= 10:                      stars = 2
                else:                                   stars = 1

                # ── Regressor vs classifier conflict penalty ──────────────
                # If the regression scores imply a different winner than the
                # independent classifier, the models disagree → reduce stars
                # by 1 and flag the conflict visually.
                reg_home_wins = float(game.get('pred_home_score', 0) or 0) >= float(game.get('pred_away_score', 0) or 0)
                _hwp = game.get('home_win_prob', 0.5)
                try:
                    _hwp = float(_hwp)
                except (ValueError, TypeError):
                    _hwp = 0.5
                clf_home_wins  = _hwp >= 0.5
                models_conflict = reg_home_wins != clf_home_wins
                if models_conflict:
                    stars = max(1, stars - 1)
                conflict_badge = (
                    '<div style="margin-top:6px;padding:4px 10px;background:#fff3cd;'
                    'border:1px solid #ffc107;border-radius:20px;display:inline-block;'
                    'font-size:0.72rem;color:#856404;font-weight:600;">'
                    '&#9888;&#xFE0F; Score &amp; classifier disagree</div>'
                ) if models_conflict else ''
                # ─────────────────────────────────────────────────────────

                star_html  = "".join(
                    '<span style="color:' + ("#f4c542" if i < stars else "#ddd") + '; font-size:1.4rem;">&#9733;</span>'
                    for i in range(5)
                )
                star_label = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}[stars]
                pred_winner     = game.get("pred_winner", game.get("home_team", "?"))
                winner_conf     = game.get("winner_confidence", None)
                home_win_prob   = game.get("home_win_prob", 0.5)
                try:
                    home_win_prob = float(home_win_prob)
                except (ValueError, TypeError):
                    home_win_prob = 0.5
                home_win_pct    = f"{home_win_prob * 100:.0f}%"
                away_win_pct    = f"{(1 - home_win_prob) * 100:.0f}%"
                home_bar_w      = f"{home_win_prob * 100:.0f}%"
                away_bar_w      = f"{(1 - home_win_prob) * 100:.0f}%"
                winner_is_home  = pred_winner == game.get("home_team", "")
                home_name_style = "font-weight:900; color:#667eea;" if winner_is_home else ""
                away_name_style = "font-weight:900; color:#667eea;" if not winner_is_home else ""
                trophy_home     = " 🏆" if winner_is_home else ""
                trophy_away     = " 🏆" if not winner_is_home else ""
                away_team   = game['away_team']
                home_team   = game['home_team']
                away_score  = game['pred_away_score']
                home_score  = game['pred_home_score']
                game_time   = game['game_time']
                pred_total  = game['pred_total']
                line        = game['line']
                over_under  = game['over_under']
                # NaN-safe display values for line and over/under
                line_display       = f"{line:.1f}" if not pd.isna(line) else "N/A"
                over_under_display = over_under if not pd.isna(line) and over_under in ('OVER', 'UNDER') else "N/A"
                # Injury data — only present for today's fresh games
                home_adj = float(game.get('injury_adj_home', 0) or 0)
                away_adj = float(game.get('injury_adj_away', 0) or 0)
                home_notes_raw = str(game.get('home_inj_notes', '') or '')
                away_notes_raw = str(game.get('away_inj_notes', '') or '')
                home_notes = [n.strip() for n in home_notes_raw.split('|') if n.strip()]
                away_notes = [n.strip() for n in away_notes_raw.split('|') if n.strip()]
                # Adjustment label shown next to score (empty string if no adjustment)
                home_adj_html = (
                    f' <span style="color:#fa709a;font-size:0.8rem;">(adj {home_adj:+.1f})</span>'
                    if home_adj != 0 else ""
                )
                away_adj_html = (
                    f' <span style="color:#fa709a;font-size:0.8rem;">(adj {away_adj:+.1f})</span>'
                    if away_adj != 0 else ""
                )
                # Build injury section HTML block
                inj_html = ""
                if home_notes or away_notes:
                    inj_rows_html = ""
                    if home_notes:
                        notes_joined = " &nbsp;·&nbsp; ".join(home_notes)
                        inj_rows_html += (
                            f'<div style="margin-bottom:4px;">'
                            f'<strong style="color:#667eea;">{home_team}:</strong> '
                            f'{notes_joined}</div>'
                        )
                    if away_notes:
                        notes_joined = " &nbsp;·&nbsp; ".join(away_notes)
                        inj_rows_html += (
                            f'<div>'
                            f'<strong style="color:#667eea;">{away_team}:</strong> '
                            f'{notes_joined}</div>'
                        )
                    inj_html = (
                        '<div style="margin-top:16px;padding:12px 16px;background:#fff8f0;'
                        'border-left:4px solid #fa709a;border-radius:8px;'
                        'font-size:0.82rem;line-height:1.6;">'
                        '<div style="font-weight:700;color:#c0392b;margin-bottom:6px;">'
                        '🏥 Injury Report</div>'
                        + inj_rows_html +
                        '</div>'
                    )
                with st.container():
                    st.markdown(
                        f'<div class="prediction-card">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<div style="text-align:center;flex:1;">'
                        f'<h2 style="{away_name_style}">{away_team}{trophy_away}</h2>'
                        f'<div class="team-score">{away_score:.1f}{away_adj_html}</div>'
                        f'<p style="color:#666;">Away &nbsp;·&nbsp; {away_win_pct}</p>'
                        f'</div>'
                        f'<div style="text-align:center;flex:0.5;">'
                        f'<h1 style="font-size:4rem;color:#667eea;">VS</h1>'
                        f'<div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:5px 15px;border-radius:50px;color:white;">{game_time}</div>'
                        f'<div style="margin-top:8px;">{star_html}</div>'
                        f'<div style="font-size:0.75rem;color:#888;margin-top:2px;">{star_label} Conviction</div>'
                        f'<div style="font-size:0.8rem;margin-top:4px;color:{conf_color};">🎯 {conf_pct} confidence</div>'
                        f'{conflict_badge}'
                        f'</div>'
                        f'<div style="text-align:center;flex:1;">'
                        f'<h2 style="{home_name_style}">{home_team}{trophy_home}</h2>'
                        f'<div class="team-score">{home_score:.1f}{home_adj_html}</div>'
                        f'<p style="color:#666;">Home &nbsp;·&nbsp; {home_win_pct}</p>'
                        f'</div>'
                        f'</div>'
                        f'<div style="margin:12px 0 20px;height:8px;border-radius:4px;background:#eee;overflow:hidden;display:flex;">'
                        f'<div style="width:{away_bar_w};background:#fa709a;"></div>'
                        f'<div style="width:{home_bar_w};background:#43e97b;"></div>'
                        f'</div>'
                        f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:15px;margin-top:10px;">'
                        f'<div class="metric-container"><div class="metric-value">{pred_total:.1f}</div><div class="metric-label">Blended Total</div></div>'
                        f'<div class="metric-container"><div class="metric-value" style="font-size:1.4rem;">{direct_str}</div><div class="metric-label">Model Total</div></div>'
                        f'<div class="metric-container"><div class="metric-value" style="font-size:1.4rem;">{sum_str}</div><div class="metric-label">H+A Sum</div></div>'
                        f'<div class="metric-container"><div class="metric-value">{line_display}</div><div class="metric-label">Vegas Line</div></div>'
                        f'<div class="metric-container"><div class="metric-value" style="color:{ou_color};">{over_under_display}</div><div class="metric-label">Prediction</div></div>'
                        f'</div>'
                        f'{inj_html}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            # Export predictions
            date_str = datetime.now().strftime('%Y%m%d')
            st.markdown("---")
            st.markdown("#### 📥 Download Predictions")
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                csv_data = predictions.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name=f"nba_predictions_{date_str}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with col2:
                import io
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    predictions.to_excel(writer, index=False, sheet_name='Predictions')
                excel_data = excel_buffer.getvalue()
                st.download_button(
                    label="⬇️ Download Excel",
                    data=excel_data,
                    file_name=f"nba_predictions_{date_str}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

def show_model_performance_page():
    """Display model performance page"""
    st.markdown("### 🤖 Model Performance Analysis")
    
    if not st.session_state.model_trained:
        st.markdown("""
        <div class="warning-message">
            Model not trained yet. Please initialize the system first.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Load model metrics
    metrics = st.session_state.model.get_performance_metrics()
    
    # Show held-out test metrics dashboard
    if 'test' in metrics or 'final' in metrics:
        from src.visualization import create_model_performance_dashboard
        st.markdown("### 📈 Held-Out Test Performance")
        fig = create_model_performance_dashboard(metrics)
        st.plotly_chart(fig, use_container_width=True)
        if 'test' in metrics:
            st.caption(metrics['test'].get('note', ''))
    
    # Display model information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="analytics-card">
            <h4>Model Architecture</h4>
            <ul style="list-style-type: none; padding: 0;">
                <li>🔹 <strong>Base Models:</strong> XGBoost, LightGBM, Random Forest</li>
                <li>🔹 <strong>Meta Model:</strong> Ridge Regression</li>
                <li>🔹 <strong>Features:</strong> 150+ engineered features</li>
                <li>🔹 <strong>Training Data:</strong> Last 5 seasons</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature importance
        st.markdown("#### 🔑 Top 10 Feature Importance")
        feature_importance = st.session_state.model.get_feature_importance()
        st.dataframe(feature_importance.head(10), use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="analytics-card">
            <h4>Performance Metrics</h4>
        """, unsafe_allow_html=True)
        
        # Cross-validation scores
        cv_scores = metrics.get('cv_scores', {})
        for target, scores in cv_scores.items():
            if isinstance(scores, dict):
                mean_mae = scores.get('mean_mae', 0)
                std_mae  = scores.get('std_mae', 0)
                st.metric(
                    f"{target.title()} CV MAE",
                    f"{mean_mae:.2f} pts",
                    delta=f"± {std_mae:.2f}",
                    delta_color="off"
                )
            else:
                st.metric(target.replace('_', ' ').title(), f"{scores:.3f}")

        # Winner classifier accuracy
        winner_metrics = metrics.get('winner', {})
        if winner_metrics:
            st.metric(
                "Winner Accuracy (test)",
                f"{winner_metrics.get('test_accuracy', 0) * 100:.1f}%",
                delta=f"log-loss {winner_metrics.get('test_log_loss', 0):.3f}",
                delta_color="off"
            )
            st.caption(winner_metrics.get('note', ''))
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Confusion matrix if classification
        if 'confusion_matrix' in metrics:
            st.markdown("#### 📊 Confusion Matrix")
            fig = st.session_state.ui_manager.plot_confusion_matrix(
                metrics['confusion_matrix']
            )
            st.plotly_chart(fig, use_container_width=True)

def show_data_management_page():
    """Display data management page"""
    st.markdown("### 💾 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="analytics-card">
            <h4>📥 Data Collection</h4>
        """, unsafe_allow_html=True)
        
        if st.button("⚡ Sync Missing Games", key="sync_data"):
            with st.spinner("Fetching games not yet in cache..."):
                success = st.session_state.data_collector.update_historical_data()
                if success:
                    st.success("✅ Cache synced with latest games!")
                    st.session_state.data_collected = True
                else:
                    st.error("❌ Sync failed — check logs")

        if st.button("🔄 Full Rescrape (all seasons)", key="scrape_data"):
            with st.spinner("Scraping all historical data from ESPN..."):
                success = st.session_state.data_collector.scrape_historical_data()
                if success:
                    st.success("✅ Historical data collected successfully!")
                    st.session_state.data_collected = True
                else:
                    st.error("❌ Failed to collect historical data")

        if st.button("🏀 Fetch Today's Games", key="fetch_games"):
            with st.spinner("Fetching today's games from Odds API..."):
                games = st.session_state.data_collector.fetch_todays_games()
                if not games.empty:
                    st.success(f"✅ Found {len(games)} games for today")
                else:
                    st.warning("No games found for today")
        
        if st.button("🔁 Sync Missing Games", key="sync_missing", help="Fetches only dates not yet in the cache — no full rescrape"):
            with st.spinner("Syncing missing games from ESPN..."):
                success = st.session_state.data_collector.update_historical_data()
                if success:
                    updated_df = st.session_state.data_collector.load_historical_data()
                    st.success(f"✅ Cache synced! Total games in cache: {len(updated_df)}")
                else:
                    st.error("❌ Sync failed — check logs")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="analytics-card">
            <h4>🗑️ Cache Management</h4>
        """, unsafe_allow_html=True)
        
        cache_files = st.session_state.cache_manager.list_cache_files()
        st.write(f"**Cache files:** {len(cache_files)}")
        
        if st.button("🧹 Clear All Cache", key="clear_cache"):
            st.session_state.cache_manager.clear_cache()
            st.success("✅ Cache cleared successfully!")
            st.rerun()
        
        if st.button("📦 Export Cache Info", key="export_cache"):
            cache_info = st.session_state.cache_manager.get_cache_info()
            st.json(cache_info)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display cache files
    st.markdown("### 📁 Cache Files")
    cache_files = st.session_state.cache_manager.list_cache_files()
    if cache_files:
        cache_df = pd.DataFrame(cache_files)
        st.dataframe(cache_df, use_container_width=True)

def show_settings_page():
    """Display settings page"""
    st.markdown("### ⚙️ Settings")

    # ── Manual Retrain ────────────────────────────────────────────────────
    st.markdown("""
    <div class="analytics-card">
        <h4>🤖 Model Retraining</h4>
    """, unsafe_allow_html=True)
    st.write("Manually retrain the model with the latest cached data. This clears today's predictions so fresh ones are generated afterwards.")
    if st.button("🔁 Retrain Model Now", key="manual_retrain"):
        # Clear today's cached predictions so they're regenerated after retrain
        today_str = datetime.now().strftime('%Y-%m-%d')
        predictions_cache_file = os.path.join('cache', f'predictions_{today_str}.csv')
        if os.path.exists(predictions_cache_file):
            os.remove(predictions_cache_file)
        st.session_state.today_predictions = None
        with st.spinner("Retraining model — this may take a few minutes..."):
            initialize_system()
        st.session_state['needs_retrain'] = False
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="analytics-card">
            <h4>🔧 Model Settings</h4>
        """, unsafe_allow_html=True)
        
        # Model hyperparameters
        n_estimators = st.slider("Number of Estimators", 50, 300, 100, 10)
        max_depth = st.slider("Max Depth", 3, 15, 7)
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
        
        if st.button("Save Model Settings"):
            st.session_state.model.update_hyperparameters({
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate
            })
            st.success("✅ Model settings updated!")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="analytics-card">
            <h4>🔄 Update Frequency</h4>
        """, unsafe_allow_html=True)
        
        update_freq = st.selectbox(
            "Data Update Frequency",
            ["Hourly", "Daily", "Weekly", "Manual"]
        )
        
        auto_retrain = st.checkbox("Auto-retrain model weekly", value=True)
        
        if st.button("Save Settings"):
            st.session_state.cache_manager.update_settings({
                'update_frequency': update_freq,
                'auto_retrain': auto_retrain
            })
            st.success("✅ Settings saved!")
        
        st.markdown("</div>", unsafe_allow_html=True)

def _assert_pipeline(condition: bool, message: str, st_error: bool = True):
    """Raise a clear error if a pipeline integrity check fails"""
    if not condition:
        logger.error(f"Pipeline integrity check failed: {message}")
        if st_error:
            st.error(f"❌ Pipeline error: {message}")
        raise AssertionError(message)

def initialize_system():
    """Initialize the system with data collection and model training"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Ensure historical data is available.
        # If the cache already has data, skip scraping and go straight to training.
        # Only scrape if the cache is empty.
        status_text.text("📊 Checking data cache...")
        historical_df = st.session_state.data_collector.load_historical_data()

        if historical_df.empty:
            logger.info("No cached data — starting data collection from ESPN...")
            status_text.text("📊 No cache found — collecting historical data...")
            success = st.session_state.data_collector.scrape_historical_data()
            _assert_pipeline(success, "Data collection returned False — scraper may have failed or found no games")
            historical_df = st.session_state.data_collector.load_historical_data()
        else:
            logger.info(f"Cache hit — {len(historical_df)} games already available, skipping scrape")
            status_text.text(f"📊 Using {len(historical_df)} cached games — skipping scrape...")
        progress_bar.progress(20)

        # Step 2: Validate data
        status_text.text("✅ Validating data...")
        _assert_pipeline(not historical_df.empty, "No historical data loaded from cache after scraping")
        _assert_pipeline(
            'home_score' in historical_df.columns and 'away_score' in historical_df.columns,
            f"Expected score columns missing. Got: {list(historical_df.columns)}"
        )
        _assert_pipeline(
            len(historical_df) >= 100,
            f"Too few games to train on: only {len(historical_df)} rows. Need at least 100."
        )
        logger.info(f"Loaded {len(historical_df)} games for feature engineering")
        progress_bar.progress(35)

        # Step 3: Feature engineering
        status_text.text("🔧 Engineering advanced features...")
        featured_df = st.session_state.feature_engineer.create_features(historical_df)
        _assert_pipeline(not featured_df.empty, "Feature engineering returned an empty DataFrame")
        feature_cols = st.session_state.feature_engineer.get_feature_names()
        _assert_pipeline(
            len(feature_cols) > 0,
            "Feature engineer produced no feature columns"
        )
        nan_rate = featured_df[feature_cols].isnull().mean().mean()
        _assert_pipeline(
            nan_rate < 0.5,
            f"Feature matrix is {nan_rate:.0%} NaN — feature engineering likely failed"
        )
        logger.info(f"Feature engineering complete: {len(feature_cols)} features, {nan_rate:.1%} NaN rate")
        progress_bar.progress(55)

        # Step 4: Prepare train targets
        status_text.text("🤖 Training ensemble model...")
        target_col = 'total_score'
        _assert_pipeline(
            target_col in featured_df.columns,
            f"Target column '{target_col}' missing from featured data"
        )
        _assert_pipeline(
            'home_score' in featured_df.columns and 'away_score' in featured_df.columns,
            "home_score / away_score columns missing from featured data"
        )
        _assert_pipeline(
            'home_win' in featured_df.columns,
            "home_win column missing from featured data"
        )
        X = featured_df[feature_cols].copy()
        y       = featured_df[target_col].copy()
        y_home  = featured_df['home_score'].copy()
        y_away  = featured_df['away_score'].copy()
        y_win   = featured_df['home_win'].copy()

        # Drop rows where any target is NaN
        valid_mask = y.notna() & y_home.notna() & y_away.notna() & y_win.notna()
        X, y, y_home, y_away, y_win = (
            X[valid_mask], y[valid_mask], y_home[valid_mask],
            y_away[valid_mask], y_win[valid_mask]
        )
        _assert_pipeline(len(X) >= 100, f"Only {len(X)} valid training rows after dropping NaN targets")

        # Step 5: Train model
        metrics = st.session_state.model.train(X, y, y_home=y_home, y_away=y_away, y_win=y_win)
        _assert_pipeline(
            'test' in metrics,
            "Training completed but no test metrics were recorded"
        )
        test_mae      = metrics['test']['mae']
        test_mae_home = metrics['test'].get('mae_home', None)
        test_mae_away = metrics['test'].get('mae_away', None)
        logger.info(f"Model trained. Held-out MAE: total={test_mae:.2f} | home={test_mae_home} | away={test_mae_away}")
        progress_bar.progress(85)

        # Step 6: Persist and update state
        status_text.text("💾 Saving model...")
        st.session_state.model.save_model(
            feature_engineer=st.session_state.feature_engineer
        )
        st.session_state.model_trained = True
        st.session_state.data_collected = True
        st.session_state.cache_manager.update_last_update()
        progress_bar.progress(100)

        status_text.text("✅ System initialized successfully!")
        home_str = f"{test_mae_home:.1f}" if isinstance(test_mae_home, float) else "—"
        away_str = f"{test_mae_away:.1f}" if isinstance(test_mae_away, float) else "—"
        st.success(
            f"System ready! Test MAE — **total: {test_mae:.1f} pts** | "
            f"home: {home_str} pts | away: {away_str} pts "
            f"({metrics['test']['n_samples']} held-out games)"
        )
        logger.info("System initialization completed successfully")
        time.sleep(2)
        st.rerun()

    except AssertionError:
        pass  # Already shown to user via st.error
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        st.error(f"Initialization failed: {str(e)}")


def show_backtesting_page():
    """
    Backtesting page — compares saved predictions to actual ESPN results.
    Automatically resolves any prediction dates that have actual results
    available, then displays O/U hit rate, winner accuracy, MAE and a
    running accuracy chart.
    """
    st.markdown("## 🔬 Backtesting — Predictions vs Reality")

    cache   = st.session_state.cache_manager
    dc      = st.session_state.data_collector
    bt_df   = cache.load_backtest_results()
    pred_df = cache.load_historical_predictions()

    # ── Auto-resolve unresolved prediction dates ──────────────────────────
    if not pred_df.empty:
        pred_df['date'] = pd.to_datetime(pred_df['date']).dt.strftime('%Y-%m-%d')
        today = datetime.now().strftime('%Y-%m-%d')

        # Dates that have predictions but no backtest result yet
        resolved_dates = set(bt_df['date'].unique()) if not bt_df.empty else set()
        all_pred_dates = set(pred_df['date'].unique())
        # Only resolve dates before today (can't resolve future games)
        pending = sorted([d for d in all_pred_dates if d < today and d not in resolved_dates])

        if pending:
            with st.spinner(f"Fetching actual results for {len(pending)} unresolved date(s)..."):
                # Load historical data once — used as primary source for actual results
                hist_df = dc.load_historical_data()
                if not hist_df.empty:
                    hist_df['date'] = pd.to_datetime(hist_df['date']).dt.strftime('%Y-%m-%d')

                new_rows = []
                for date_str in pending:
                    # Check historical cache first — avoids unnecessary ESPN calls
                    # and works even when ESPN no longer carries old game data
                    actual = pd.DataFrame()
                    if not hist_df.empty and date_str in hist_df['date'].values:
                        day_hist = hist_df[hist_df['date'] == date_str].copy()
                        if not day_hist.empty:
                            actual = day_hist.rename(columns={
                                'home_score': 'actual_home',
                                'away_score': 'actual_away',
                            })[['date', 'home_team', 'away_team', 'actual_home', 'actual_away']]
                            actual['actual_total'] = actual['actual_home'] + actual['actual_away']

                    # Fall back to ESPN only if not in historical cache
                    if actual.empty:
                        actual = dc.fetch_results_for_date(date_str)

                    if actual.empty:
                        continue
                    preds_on_day = pred_df[pred_df['date'] == date_str].copy()
                    merged = preds_on_day.merge(
                        actual, on=['date', 'home_team', 'away_team'], how='inner'
                    )
                    if merged.empty:
                        continue
                    # Compute result columns
                    # ou_correct is only meaningful when a Vegas line was available
                    merged['ou_correct'] = np.where(
                        merged['line'].isna() | (merged['over_under'] == 'N/A'),
                        np.nan,
                        ((merged['over_under'] == 'OVER')  & (merged['actual_total'] > merged['line'])) |
                        ((merged['over_under'] == 'UNDER') & (merged['actual_total'] < merged['line']))
                    )
                    merged['winner_correct'] = (
                        ((merged['pred_winner'] == merged['home_team']) & (merged['actual_home'] > merged['actual_away'])) |
                        ((merged['pred_winner'] == merged['away_team']) & (merged['actual_away'] > merged['actual_home']))
                    ).astype(int)
                    merged['total_error'] = abs(merged['pred_total'] - merged['actual_total'])
                    merged['home_error']  = abs(merged['pred_home_score'] - merged['actual_home'])
                    merged['away_error']  = abs(merged['pred_away_score'] - merged['actual_away'])
                    new_rows.append(merged)

                if new_rows:
                    new_bt = pd.concat(new_rows, ignore_index=True)
                    cache.save_backtest_results(new_bt)
                    bt_df = cache.load_backtest_results()
                    st.success(f"Resolved {len(new_bt)} games across {len(pending)} date(s).")

    if bt_df.empty:
        st.info(
            "No backtesting data yet. Predictions are automatically compared to "
            "actual results once the games are completed. Come back tomorrow after "
            "today's games finish — results will be fetched automatically."
        )
        return

    bt_df['date'] = pd.to_datetime(bt_df['date'])

    # ── Summary metrics ───────────────────────────────────────────────────
    total_games  = len(bt_df)
    ou_acc       = bt_df['ou_correct'].mean(skipna=True) * 100  if 'ou_correct'      in bt_df.columns else 0
    win_acc      = bt_df['winner_correct'].mean() * 100 if 'winner_correct' in bt_df.columns else 0
    mae_total    = bt_df['total_error'].mean()   if 'total_error' in bt_df.columns else 0
    mae_home     = bt_df['home_error'].mean()    if 'home_error'  in bt_df.columns else 0
    mae_away     = bt_df['away_error'].mean()    if 'away_error'  in bt_df.columns else 0

    st.markdown("### 📊 Overall Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Games Tracked", total_games)
    with c2:
        st.metric("O/U Hit Rate", f"{ou_acc:.1f}%")
    with c3:
        st.metric("Winner Accuracy", f"{win_acc:.1f}%")
    with c4:
        st.metric("Total MAE", f"{mae_total:.1f} pts")
    with c5:
        st.metric("Home/Away MAE", f"{(mae_home + mae_away) / 2:.1f} pts")

    # ── Rolling accuracy chart ─────────────────────────────────────────────
    st.markdown("### 📈 Rolling Accuracy (10-game window)")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        bt_sorted = bt_df.sort_values('date').reset_index(drop=True)
        window = min(10, len(bt_sorted))
        bt_sorted['ou_rolling']  = bt_sorted['ou_correct'].rolling(window, min_periods=1).mean() * 100
        bt_sorted['win_rolling'] = bt_sorted['winner_correct'].rolling(window, min_periods=1).mean() * 100
        bt_sorted['mae_rolling'] = bt_sorted['total_error'].rolling(window, min_periods=1).mean()
        bt_sorted['label']       = bt_sorted['away_team'] + ' @ ' + bt_sorted['home_team']

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("O/U & Winner Hit Rate (%)", "Rolling Total MAE (pts)"),
            vertical_spacing=0.12
        )
        fig.add_trace(go.Scatter(
            x=bt_sorted.index, y=bt_sorted['ou_rolling'],
            name='O/U %', line=dict(color='#667eea', width=2),
            hovertext=bt_sorted['label']
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=bt_sorted.index, y=bt_sorted['win_rolling'],
            name='Winner %', line=dict(color='#43e97b', width=2),
            hovertext=bt_sorted['label']
        ), row=1, col=1)
        fig.add_hline(y=50, line_dash='dash', line_color='gray',
                      annotation_text='50% baseline', row=1, col=1)
        fig.add_trace(go.Scatter(
            x=bt_sorted.index, y=bt_sorted['mae_rolling'],
            name='MAE', line=dict(color='#fa709a', width=2),
            hovertext=bt_sorted['label']
        ), row=2, col=1)
        fig.update_layout(
            height=500, showlegend=True,
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Chart unavailable: {e}")

    # ── By-star-rating breakdown ───────────────────────────────────────────
    if 'confidence' in bt_df.columns:
        st.markdown("### ⭐ Performance by Conviction Level")
        def _stars(row):
            conf = row.get('confidence', 0) or 0
            try:
                conf = float(conf)
            except:
                conf = 0
            dt = row.get('pred_total_direct')
            st_ = row.get('pred_total_sum')
            spread = abs(float(dt) - float(st_)) if (dt is not None and st_ is not None) else 999
            if   spread <= 3  and conf >= 0.70: s = 5
            elif spread <= 5  and conf >= 0.60: s = 4
            elif spread <= 7  and conf >= 0.50: s = 3
            elif spread <= 10:                  s = 2
            else:                               s = 1
            # Conflict penalty: regression winner vs classifier winner
            try:
                reg_home = float(row.get('pred_home_score', 0) or 0) >= float(row.get('pred_away_score', 0) or 0)
                hwp = float(row.get('home_win_prob', 0.5) or 0.5)
                if reg_home != (hwp >= 0.5):
                    s = max(1, s - 1)
            except Exception:
                pass
            return s
        bt_df['stars'] = bt_df.apply(_stars, axis=1)
        star_groups = bt_df.groupby('stars').agg(
            Games=('ou_correct', 'count'),
            OU_Pct=('ou_correct', lambda x: f"{x.mean()*100:.1f}%"),
            Winner_Pct=('winner_correct', lambda x: f"{x.mean()*100:.1f}%"),
            MAE=('total_error', lambda x: f"{x.mean():.1f}")
        ).sort_index(ascending=False).reset_index()
        star_groups['stars'] = star_groups['stars'].map(
            {5: '⭐⭐⭐⭐⭐', 4: '⭐⭐⭐⭐', 3: '⭐⭐⭐', 2: '⭐⭐', 1: '⭐'}
        )
        star_groups.columns = ['Conviction', 'Games', 'O/U Hit Rate', 'Winner Accuracy', 'Total MAE']
        st.dataframe(star_groups, use_container_width=True, hide_index=True)

    # ── Raw results table ──────────────────────────────────────────────────
    st.markdown("### 📋 Game-by-Game Results")
    display_cols = ['date', 'away_team', 'home_team',
                    'pred_total', 'actual_total', 'total_error',
                    'over_under', 'ou_correct',
                    'pred_winner', 'winner_correct']
    show_cols = [c for c in display_cols if c in bt_df.columns]
    show_df = bt_df[show_cols].sort_values('date', ascending=False).copy()
    show_df['date'] = show_df['date'].dt.strftime('%Y-%m-%d')
    if 'ou_correct' in show_df.columns:
        show_df['ou_correct'] = show_df['ou_correct'].map({1: '✅', 0: '❌'})
    if 'winner_correct' in show_df.columns:
        show_df['winner_correct'] = show_df['winner_correct'].map({1: '✅', 0: '❌'})
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    # ── Manual refresh button ──────────────────────────────────────────────
    if st.button("🔄 Re-fetch Latest Results", key="bt_refresh"):
        # Clear resolved set so pending logic re-runs
        import os
        if os.path.exists(cache.backtest_file):
            os.remove(cache.backtest_file)
        st.rerun()


if __name__ == "__main__":
    main()

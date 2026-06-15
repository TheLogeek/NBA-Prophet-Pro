# NBA Prophet Pro - Setup Instructions

> ⚡ **Deployment Architecture:** Optimized for localized execution to manage multi-season data scraping and high-compute ensemble training with zero server costs[span_7](start_span)[span_7](end_span).


## 🏀 Advanced NBA Game Prediction System

### System Requirements

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- Internet connection for data fetching
- ESPN access (for historical data)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/nba-prophet-pro.git
cd nba-prophet-pro
```

1. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. Install dependencies

```bash
pip install -r requirements.txt
```

1. Set up environment variables
   Create a .env file in the root directory:

```env
ODDS_API_KEY=your_odds_api_key_here
```

Project Structure

```
nba-predictor-pro/
├── main.py                 # Main Streamlit application
├── src/
│   ├── data_collector.py   # ESPN and Odds API data collection
│   ├── feature_engineering.py # Advanced feature creation
│   ├── models.py           # Ensemble ML models
│   ├── cache_manager.py    # Caching system
│   ├── ui_components.py    # UI components and styling
│   └── visualization.py    # Data visualization functions
├── cache/                  # Cache directory (auto-created)
├── models/                 # Saved models directory (auto-created)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

Features

1. Advanced Ensemble Learning

· Multiple base models (XGBoost, LightGBM, Random Forest, Gradient Boosting)
· Stacking ensemble with meta-learner
· Time series cross-validation
· Feature importance analysis

2. Comprehensive Data Collection

· ESPN historical game data (up to 5 seasons)
· Odds API integration for live lines
· Automatic cache management
· Incremental updates

3. Beautiful Streamlit UI

· Gradient-based modern design
· Interactive visualizations
· Real-time predictions
· Performance analytics

4. Robust Caching

· Prevents unnecessary API calls
· Text file-based date tracking
· Automatic cache invalidation
· Manual cache management options

Usage

1. Start the application

```bash
streamlit run main.py
```

1. Initial Setup
   · Click "Initialize System" on first run
   · System will scrape historical data (takes 5-10 minutes)
   · Model training begins automatically
2. Daily Predictions
   · App checks for new games automatically
   · Predictions generated for today's games
   · View over/under predictions with confidence scores
3. Analytics
   · Track prediction accuracy over time
   · View model performance metrics
   · Analyze team statistics

Caching Strategy

· Historical data cached in historical_data.csv
· Today's games cached with date validation
· Model saved after training
· Last update tracked in text files
· Cache auto-invalidates after 24 hours

Model Features

· 200+ engineered features including:
  · Rolling averages (5, 10, 20 games)
  · Team efficiency ratings
  · Pace of play metrics
  · Head-to-head history
  · Rest days advantage
  · Strength of schedule

Performance Optimization

· Lazy loading of heavy components
· Efficient pandas operations
· Optimized feature engineering
· Parallel model training
· Streamlit caching decorators

Troubleshooting

Issue: Historical data scraping fails
Solution: Check internet connection; ESPN may block excessive requests

Issue: Model training takes too long
Solution: Reduce n_estimators in settings; use fewer seasons of data

Issue: Cache not updating
Solution: Clear cache manually in Data Management page

API Keys

Get your Odds API key at: https://the-odds-api.com/

License

MIT License - feel free to use and modify

Disclaimer

Predictions are for entertainment purposes only. Always gamble responsibly.

---

🎯 Quick Start Guide

```bash
# One-liner to get started
git clone https://github.com/yourusername/nba-predictor-pro.git && 
cd nba-predictor-pro && 
python -m venv venv && 
source venv/bin/activate && 
pip install -r requirements.txt && 
streamlit run main.py
```

📊 Expected Performance

After training on 5 seasons of data:

· MAE: 8-10 points
· Accuracy: 55-60% on over/under
· R² Score: 0.65-0.75

🔄 Updates

The system will automatically:

· Check for new games daily
· Update predictions
· Retrain model weekly
· Clear old cache files

---

Note: First-time setup requires significant data download. Ensure stable internet connection.

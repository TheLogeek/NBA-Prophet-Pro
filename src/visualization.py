# src/visualization.py
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def create_visualizations(df: pd.DataFrame, predictions: Optional[pd.DataFrame] = None):
    """Create comprehensive visualizations"""
    
    figs = {}
    
    # 1. Prediction Distribution
    if predictions is not None and 'pred_total' in predictions.columns:
        figs['distribution'] = create_prediction_distribution(predictions)
    
    # 2. Team Performance Heatmap
    if not df.empty:
        figs['team_performance'] = create_team_performance_heatmap(df)
    
    # 3. Time Series Analysis
    if 'date' in df.columns:
        figs['time_series'] = create_time_series_analysis(df)
    
    # 4. Head-to-Head Analysis
    if not df.empty:
        figs['h2h'] = create_head_to_head_analysis(df)
    
    # 5. Feature Importance
    if 'feature_importance' in df.columns:
        figs['feature_importance'] = create_feature_importance_chart(df)
    
    return figs

def create_prediction_distribution(predictions: pd.DataFrame) -> go.Figure:
    """Create prediction distribution chart"""
    
    fig = go.Figure()
    
    # Histogram of predictions
    fig.add_trace(go.Histogram(
        x=predictions['pred_total'],
        nbinsx=30,
        name='Predictions',
        marker_color='#667eea',
        opacity=0.7
    ))
    
    # Add actual line if available
    if 'actual_total' in predictions.columns:
        fig.add_trace(go.Histogram(
            x=predictions['actual_total'],
            nbinsx=30,
            name='Actual',
            marker_color='#fa709a',
            opacity=0.7
        ))
    
    fig.update_layout(
        title='Prediction Distribution',
        xaxis_title='Total Points',
        yaxis_title='Frequency',
        barmode='overlay',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_team_performance_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create team performance heatmap"""
    
    # Calculate team statistics
    team_stats = []
    
    for team in pd.concat([df['home_team'], df['away_team']]).unique():
        team_games = df[(df['home_team'] == team) | (df['away_team'] == team)]
        
        if len(team_games) > 0:
            scored  = team_games.apply(lambda x: x['home_score'] if x['home_team'] == team else x['away_score'], axis=1)
            allowed = team_games.apply(lambda x: x['away_score'] if x['home_team'] == team else x['home_score'], axis=1)
            team_stats.append({
                'team':        team,
                'avg_scored':  scored.mean(),
                'avg_allowed': allowed.mean(),
                'win_rate':    (scored > allowed).mean(),  # correct: wins = games where team scored more
                'games':       len(team_games)
            })
    
    stats_df = pd.DataFrame(team_stats)
    stats_df = stats_df.sort_values('win_rate', ascending=False).head(15)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=stats_df[['avg_scored', 'avg_allowed', 'win_rate']].values,
        x=['Points Scored', 'Points Allowed', 'Win Rate'],
        y=stats_df['team'],
        colorscale='Viridis',
        text=stats_df[['avg_scored', 'avg_allowed', 'win_rate']].round(2).values,
        texttemplate='%{text}',
        textfont={'size': 10},
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title='Top 15 Teams Performance Heatmap',
        xaxis_title='Metrics',
        yaxis_title='Team',
        height=600,
        template='plotly_white'
    )
    
    return fig

def create_time_series_analysis(df: pd.DataFrame) -> go.Figure:
    """Create time series analysis chart"""
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate rolling averages
    df['total_score'] = df['home_score'] + df['away_score']
    df['rolling_avg'] = df['total_score'].rolling(50).mean()
    df['rolling_std'] = df['total_score'].rolling(50).std()
    
    fig = go.Figure()
    
    # Add scatter plot of games
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['total_score'],
        mode='markers',
        name='Games',
        marker=dict(
            size=4,
            color=df['home_score'] - df['away_score'],
            colorscale='RdBu',
            showscale=True,
            colorbar=dict(title="Score Diff")
        ),
        opacity=0.5
    ))
    
    # Add rolling average
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['rolling_avg'],
        mode='lines',
        name='50-Game Average',
        line=dict(color='#667eea', width=3)
    ))
    
    # Add confidence bands
    fig.add_trace(go.Scatter(
        x=pd.concat([df['date'], df['date'][::-1]]),
        y=pd.concat([df['rolling_avg'] + df['rolling_std'], 
                    (df['rolling_avg'] - df['rolling_std'])[::-1]]),
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='±1 Std Dev',
        showlegend=True
    ))
    
    fig.update_layout(
        title='NBA Scoring Trends Over Time',
        xaxis_title='Date',
        yaxis_title='Total Points',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_head_to_head_analysis(df: pd.DataFrame) -> go.Figure:
    """Create head-to-head analysis chart"""
    
    # Get most common matchups
    df['matchup'] = df.apply(
        lambda x: f"{x['away_team']} @ {x['home_team']}" if x['away_team'] < x['home_team'] 
        else f"{x['home_team']} @ {x['away_team']}", axis=1
    )
    
    top_matchups = df['matchup'].value_counts().head(10).index
    
    matchup_stats = []
    for matchup in top_matchups:
        games = df[df['matchup'] == matchup]
        matchup_stats.append({
            'matchup': matchup,
            'games': len(games),
            'avg_total': games['home_score'].mean() + games['away_score'].mean(),
            'avg_diff': abs(games['home_score'] - games['away_score']).mean()
        })
    
    stats_df = pd.DataFrame(matchup_stats)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=stats_df['matchup'],
        y=stats_df['avg_total'],
        name='Avg Total',
        marker_color='#667eea',
        text=stats_df['avg_total'].round(1),
        textposition='auto',
    ))
    
    fig.add_trace(go.Scatter(
        x=stats_df['matchup'],
        y=stats_df['avg_diff'],
        name='Avg Margin',
        mode='lines+markers',
        line=dict(color='#fa709a', width=3),
        marker=dict(size=10),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Top 10 Matchups Analysis',
        xaxis_title='Matchup',
        yaxis_title='Average Total Points',
        yaxis2=dict(
            title='Average Margin',
            overlaying='y',
            side='right',
            range=[0, stats_df['avg_diff'].max() + 5]
        ),
        template='plotly_white',
        height=500,
        xaxis_tickangle=-45,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_feature_importance_chart(feature_importance: pd.DataFrame) -> go.Figure:
    """Create feature importance chart"""
    
    top_features = feature_importance.head(15)
    
    fig = go.Figure(go.Bar(
        x=top_features['importance'],
        y=top_features['feature'],
        orientation='h',
        marker=dict(
            color=top_features['importance'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=top_features['importance'].round(3),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Top 15 Feature Importance',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        template='plotly_white',
        height=600,
        margin=dict(l=200)  # Make room for feature names
    )
    
    return fig

def create_correlation_matrix(df: pd.DataFrame, features: List[str]) -> go.Figure:
    """Create correlation matrix heatmap"""
    
    corr_matrix = df[features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=features,
        y=features,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2),
        texttemplate='%{text}',
        textfont={'size': 8},
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        xaxis_title='Features',
        yaxis_title='Features',
        height=800,
        width=800,
        template='plotly_white',
        xaxis_tickangle=-45
    )
    
    return fig

def create_model_performance_dashboard(model_metrics: Dict) -> go.Figure:
    """Create model performance dashboard"""
    
    fig = go.Figure()
    
    # Add gauge charts for different metrics
    metrics = ['MAE', 'RMSE', 'R²']
    # Prefer held-out test metrics (honest); fall back to 'final' for older saved models
    metric_source = model_metrics.get('test', model_metrics.get('final', {}))
    values = [
        metric_source.get('mae', 0),
        metric_source.get('rmse', 0),
        metric_source.get('r2', 0) * 100  # Convert to percentage
    ]
    
    # Normalize for gauge display
    max_values = [15, 20, 100]  # Approximate max values
    
    for i, (metric, value, max_val) in enumerate(zip(metrics, values, max_values)):
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': metric},
            domain={'row': 0, 'column': i},
            gauge={
                'axis': {'range': [None, max_val]},
                'bar': {'color': '#667eea'},
                'steps': [
                    {'range': [0, max_val * 0.5], 'color': '#43e97b'},
                    {'range': [max_val * 0.5, max_val * 0.75], 'color': '#f1c40f'},
                    {'range': [max_val * 0.75, max_val], 'color': '#fa709a'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))
    
    fig.update_layout(
        grid={'rows': 1, 'columns': 3, 'pattern': "independent"},
        title='Model Performance Metrics',
        height=300,
        template='plotly_white'
    )
    
    return fig
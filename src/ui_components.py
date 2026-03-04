# src/ui_components.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class UIManager:
    """Manages UI components and visualizations"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#43e97b',
            'warning': '#fa709a',
            'info': '#38f9d7',
            'dark': '#2c3e50',
            'light': '#f8f9fa'
        }
    
    def calculate_performance_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate performance metrics from historical predictions"""
        
        if df.empty:
            return {
                'accuracy': 0,
                'total_predictions': 0,
                'avg_error': 0,
                'roi': 0
            }
        
        # Calculate accuracy if 'correct' column exists
        if 'correct' in df.columns:
            accuracy = (df['correct'].sum() / len(df)) * 100
        else:
            accuracy = 0
        
        # Calculate average error
        if 'pred_total' in df.columns and 'actual_total' in df.columns:
            errors = abs(df['pred_total'] - df['actual_total'])
            avg_error = errors.mean()
        else:
            avg_error = 0
        
        # Calculate ROI (simplified)
        roi = (accuracy - 50) * 2  # Simplified ROI calculation
        
        return {
            'accuracy': accuracy,
            'total_predictions': len(df),
            'avg_error': avg_error,
            'roi': roi
        }
    
    def create_performance_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create performance trend chart"""
        
        if df.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        # Add accuracy over time
        if 'date' in df.columns and 'correct' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            daily_accuracy = df.groupby(df['date'].dt.date)['correct'].agg(['mean', 'count'])
            daily_accuracy.columns = ['accuracy', 'count']
            daily_accuracy['accuracy'] = daily_accuracy['accuracy'] * 100
            
            fig.add_trace(go.Scatter(
                x=daily_accuracy.index,
                y=daily_accuracy['accuracy'],
                mode='lines+markers',
                name='Daily Accuracy',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8)
            ))
            
            # Add 7-day moving average
            daily_accuracy['ma_7'] = daily_accuracy['accuracy'].rolling(7).mean()
            fig.add_trace(go.Scatter(
                x=daily_accuracy.index,
                y=daily_accuracy['ma_7'],
                mode='lines',
                name='7-Day Average',
                line=dict(color='#fa709a', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title='Prediction Accuracy Trend',
            xaxis_title='Date',
            yaxis_title='Accuracy (%)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        fig.update_yaxes(range=[40, 80])
        
        return fig
    
    def create_model_comparison_chart(self, metrics: Dict) -> go.Figure:
        """Create model comparison chart"""
        
        fig = go.Figure()
        
        models = list(metrics.get('cv_scores', {}).keys())
        mae_scores = [metrics['cv_scores'][m]['mean_mae'] for m in models]
        std_scores = [metrics['cv_scores'][m]['std_mae'] for m in models]
        
        color_pool = ['#667eea', '#764ba2', '#43e97b', '#fa709a', '#38f9d7',
                      '#f1c40f', '#e74c3c', '#2ecc71', '#3498db', '#9b59b6']
        colors = [color_pool[i % len(color_pool)] for i in range(len(models))]

        fig.add_trace(go.Bar(
            x=models,
            y=mae_scores,
            error_y=dict(type='data', array=std_scores, visible=True),
            marker_color=colors,
            text=[f'{score:.1f}' for score in mae_scores],
            textposition='auto',
        ))
        
        fig.update_layout(
            title='Model Comparison - Mean Absolute Error',
            xaxis_title='Model',
            yaxis_title='MAE',
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def plot_confusion_matrix(self, cm: np.ndarray) -> go.Figure:
        """Plot confusion matrix"""
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Over', 'Predicted Under'],
            y=['Actual Over', 'Actual Under'],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale='Viridis',
            showscale=False
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=400,
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def create_team_performance_radar(self, team_stats: Dict) -> go.Figure:
        """Create radar chart for team performance"""
        
        categories = ['Offense', 'Defense', 'Pace', '3PT%', 'FT%', 'Rebounds']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=[
                team_stats.get('off_rating', 0),
                team_stats.get('def_rating', 0),
                team_stats.get('pace', 0),
                team_stats.get('three_pt_pct', 0) * 100,
                team_stats.get('ft_pct', 0) * 100,
                team_stats.get('rebounds', 0)
            ],
            theta=categories,
            fill='toself',
            name=team_stats.get('team_name', 'Team'),
            marker_color='#667eea'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 120]
                )),
            showlegend=True,
            title=f"{team_stats.get('team_name', 'Team')} Performance Radar",
            height=500
        )
        
        return fig
    
    def create_prediction_distribution(self, predictions: pd.Series) -> go.Figure:
        """Create prediction distribution chart"""
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=predictions,
            nbinsx=30,
            marker_color='#667eea',
            opacity=0.7,
            name='Predictions'
        ))
        
        # Add mean line
        fig.add_vline(
            x=predictions.mean(),
            line_dash="dash",
            line_color="#fa709a",
            annotation_text=f"Mean: {predictions.mean():.1f}",
            annotation_position="top"
        )
        
        fig.update_layout(
            title='Distribution of Predicted Totals',
            xaxis_title='Predicted Total',
            yaxis_title='Frequency',
            template='plotly_white',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_game_timeline(self, games_df: pd.DataFrame) -> go.Figure:
        """Create timeline of games"""
        
        fig = go.Figure()
        
        for idx, game in games_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[game['game_time']],
                y=[f"{game['away_team']} @ {game['home_team']}"],
                mode='markers',
                marker=dict(
                    size=20,
                    color='#667eea' if game.get('pred_total', 0) > game.get('line', 220) else '#fa709a',
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                text=f"Pred: {game.get('pred_total', 0):.1f}<br>Line: {game.get('line', 220)}",
                hoverinfo='text',
                showlegend=False
            ))
        
        fig.update_layout(
            title='Today\'s Games Timeline',
            xaxis_title='Game Time',
            yaxis_title='Matchup',
            height=400,
            template='plotly_white',
            xaxis=dict(tickangle=45)
        )
        
        return fig
    
    def style_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply styling to dataframe"""
        
        styled_df = df.style.applymap(
            self._color_cells,
            subset=['over_under'] if 'over_under' in df.columns else []
        ).format({
            'pred_total': '{:.1f}',
            'line': '{:.1f}',
            'confidence': '{:.1%}'
        })
        
        return styled_df
    
    def _color_cells(self, val):
        """Color cells based on value"""
        if val == 'OVER':
            return 'color: #43e97b; font-weight: bold'
        elif val == 'UNDER':
            return 'color: #fa709a; font-weight: bold'
        return ''
    
    def create_betting_card(self, game: pd.Series) -> str:
        """Create HTML for betting card"""
        
        confidence_color = '#43e97b' if game.get('confidence', 0) > 0.7 else '#fa709a'
        
        html = f"""
        <div style="background: white; border-radius: 15px; padding: 20px; margin: 10px 0;
                    box-shadow: 0 5px 20px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="text-align: center; flex: 1;">
                    <h3>{game['away_team']}</h3>
                    <p style="font-size: 2rem; color: #667eea;">{game.get('pred_away_score', 0):.1f}</p>
                </div>
                <div style="text-align: center; flex: 0.5;">
                    <h2 style="color: #764ba2;">@</h2>
                </div>
                <div style="text-align: center; flex: 1;">
                    <h3>{game['home_team']}</h3>
                    <p style="font-size: 2rem; color: #667eea;">{game.get('pred_home_score', 0):.1f}</p>
                </div>
            </div>
            <div style="display: flex; justify-content: space-around; margin-top: 20px;">
                <div>
                    <p style="color: #666;">Line</p>
                    <p style="font-size: 1.5rem;">{game.get('line', 220)}</p>
                </div>
                <div>
                    <p style="color: #666;">Prediction</p>
                    <p style="font-size: 1.5rem; color: {confidence_color}; font-weight: bold;">
                        {game.get('over_under', 'N/A')}
                    </p>
                </div>
                <div>
                    <p style="color: #666;">Confidence</p>
                    <p style="font-size: 1.5rem;">{game.get('confidence', 0):.1%}</p>
                </div>
            </div>
        </div>
        """
        
        return html
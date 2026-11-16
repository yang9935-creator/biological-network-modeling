import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import json

class Dashboard:
    """Interactive dashboard for model visualization"""
    
    def __init__(self, title: str = "BioCompressor Dashboard"):
        self.title = title
        self.figures = []
    
    def create_prediction_scatter(self, y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 output_names: List[str] = None) -> go.Figure:
        """Create interactive scatter plot"""
        
        fig = go.Figure()
        
        if y_true.ndim == 1:
            # Single output
            fig.add_trace(go.Scatter(
                x=y_true, y=y_pred,
                mode='markers',
                marker=dict(size=8, opacity=0.6, color='blue'),
                name='Predictions'
            ))
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                line=dict(dash='dash', color='red', width=2),
                name='Perfect Prediction'
            ))
            
            fig.update_layout(
                title='Prediction Performance',
                xaxis_title='True Values',
                yaxis_title='Predicted Values',
                showlegend=True
            )
            
        else:
            # Multiple outputs - create subplots
            n_outputs = y_true.shape[1]
            
            for i in range(n_outputs):
                fig.add_trace(go.Scatter(
                    x=y_true[:, i], y=y_pred[:, i],
                    mode='markers',
                    marker=dict(size=6, opacity=0.5),
                    name=output_names[i] if output_names else f'Output {i+1}'
                ))
        
        return fig
    
    def create_dose_response_3d(self, X: np.ndarray, 
                               y: np.ndarray,
                               feature_names: List[str],
                               output_name: str = 'Response') -> go.Figure:
        """Create 3D dose-response surface"""
        
        fig = go.Figure(data=[
            go.Scatter3d(
                x=X[:, 0], y=X[:, 1], z=y.flatten(),
                mode='markers',
                marker=dict(
                    size=4,
                    color=y.flatten(),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=output_name)
                ),
                text=[f'{feature_names[0]}: {x:.3f}<br>{feature_names[1]}: {y:.3f}<br>{output_name}: {z:.3f}'
                     for x, y, z in zip(X[:, 0], X[:, 1], y.flatten())],
                hoverinfo='text'
            )
        ])
        
        fig.update_layout(
            title='3D Dose-Response Surface',
            scene=dict(
                xaxis_title=feature_names[0],
                yaxis_title=feature_names[1],
                zaxis_title=output_name
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_parallel_coordinates(self, df: pd.DataFrame,
                                   class_column: Optional[str] = None) -> go.Figure:
        """Create parallel coordinates plot"""
        
        dimensions = []
        
        for col in df.columns:
            if col != class_column:
                dimensions.append(dict(
                    label=col,
                    values=df[col],
                    range=[df[col].min(), df[col].max()]
                ))
        
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=df[class_column] if class_column else 'blue',
                colorscale='Viridis',
                showscale=True
            ),
            dimensions=dimensions
        ))
        
        fig.update_layout(
            title='Parallel Coordinates Plot',
            width=1000,
            height=600
        )
        
        return fig
    
    def create_animation(self, X_list: List[np.ndarray],
                        y_list: List[np.ndarray],
                        frame_names: List[str],
                        feature_names: List[str]) -> go.Figure:
        """Create animated plot over time/conditions"""
        
        fig = go.Figure()
        
        frames = []
        for i, (X, y) in enumerate(zip(X_list, y_list)):
            frame = go.Frame(
                data=[go.Scatter(
                    x=X[:, 0], y=y.flatten(),
                    mode='markers',
                    marker=dict(size=8, opacity=0.6, color=X[:, 1],
                              colorscale='Viridis', showscale=True)
                )],
                name=frame_names[i]
            )
            frames.append(frame)
        
        fig.frames = frames
        
        # Add initial data
        fig.add_trace(go.Scatter(
            x=X_list[0][:, 0], y=y_list[0].flatten(),
            mode='markers',
            marker=dict(size=8, opacity=0.6, color=X_list[0][:, 1],
                      colorscale='Viridis', showscale=True)
        ))
        
        # Animation controls
        fig.update_layout(
            title='Animation over Conditions',
            xaxis_title=feature_names[0],
            yaxis_title='Response',
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[dict(
                    label='Play',
                    method='animate',
                    args=[None, {'frame': {'duration': 500, 'redraw': True},
                                'fromcurrent': True}]
                )]
            )],
            sliders=[dict(
                steps=[dict(
                    method='animate',
                    args=[[f.name], {'mode': 'immediate', 'frame': {'duration': 300}}],
                    label=f.name
                ) for f in frames]
            )]
        )
        
        return fig
    
    def to_html(self, filename: str):
        """Save dashboard to HTML file"""
        from plotly.subplots import make_subplots
        
        # Combine all figures into one dashboard
        dashboard = make_subplots(
            rows=len(self.figures), cols=1,
            subplot_titles=[f'Figure {i+1}' for i in range(len(self.figures))]
        )
        
        for i, fig in enumerate(self.figures):
            for trace in fig.data:
                dashboard.add_trace(trace, row=i+1, col=1)
        
        dashboard.update_layout(height=400 * len(self.figures), title=self.title)
        dashboard.write_html(filename)
        
        return filename
"""
Visualization functions for movement analysis data
"""
import plotly.graph_objects as go
import numpy as np

def create_velocity_plot(timestamps, velocity_data):
    """
    Create velocity profile plot
    
    Args:
        timestamps: Time points in seconds
        velocity_data: Velocity magnitudes at each time point
        
    Returns:
        plotly.graph_objects.Figure: Velocity plot
    """
    fig = go.Figure()
    
    # Add velocity trace
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=velocity_data,
        mode='lines',
        name='Velocity',
        line=dict(color='#3498db', width=2),
        fill='tonexty',
        fillcolor='rgba(52, 152, 219, 0.2)'
    ))
    
    # Add moving average
    window_size = min(10, len(velocity_data) // 4)
    if window_size > 1:
        moving_avg = np.convolve(velocity_data, np.ones(window_size)/window_size, mode='valid')
        avg_timestamps = timestamps[window_size//2:len(timestamps)-window_size//2+1]
        
        fig.add_trace(go.Scatter(
            x=avg_timestamps,
            y=moving_avg,
            mode='lines',
            name='Moving Average',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Velocity Profile During Movement",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Time (seconds)",
        yaxis_title="Velocity (m/s)",
        height=350,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified',
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def create_energy_plot(timestamps, energy_cumulative):
    """
    Create cumulative energy expenditure plot
    
    Args:
        timestamps: Time points in seconds
        energy_cumulative: Cumulative energy at each time point
        
    Returns:
        plotly.graph_objects.Figure: Energy plot
    """
    fig = go.Figure()
    
    # Add energy trace
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=energy_cumulative,
        mode='lines',
        name='Cumulative Energy',
        line=dict(color='#27ae60', width=3),
        fill='tozeroy',
        fillcolor='rgba(39, 174, 96, 0.2)'
    ))
    
    # Add rate of energy expenditure (derivative)
    if len(energy_cumulative) > 1:
        energy_rate = np.gradient(energy_cumulative)
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=energy_rate,
            mode='lines',
            name='Energy Rate',
            line=dict(color='#f39c12', width=2, dash='dot'),
            yaxis='y2',
            opacity=0.7
        ))
    
    # Update layout with dual y-axes
    fig.update_layout(
        title={
            'text': "Energy Expenditure Over Time",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Time (seconds)",
        yaxis_title="Cumulative Energy (J)",
        yaxis2=dict(
            title="Energy Rate (J/s)",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        height=350,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified',
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def create_intensity_gauge(intensity_value):
    """
    Create a gauge chart for activity intensity (METs)
    
    Args:
        intensity_value: METs value
        
    Returns:
        plotly.graph_objects.Figure: Gauge chart
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=intensity_value,
        title={'text': "Activity Intensity (METs)"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 15], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 3], 'color': "lightgreen"},
                {'range': [3, 6], 'color': "yellow"},
                {'range': [6, 9], 'color': "orange"},
                {'range': [9, 15], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': intensity_value
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        font={'size': 14}
    )
    
    return fig

def create_3d_trajectory(positions):
    """
    Create 3D trajectory visualization of center of mass
    
    Args:
        positions: Array of 3D positions over time
        
    Returns:
        plotly.graph_objects.Figure: 3D trajectory plot
    """
    fig = go.Figure(data=[go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='lines+markers',
        marker=dict(
            size=3,
            color=np.arange(len(positions)),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Time")
        ),
        line=dict(
            color='darkblue',
            width=2
        )
    )])
    
    fig.update_layout(
        title="3D Movement Trajectory",
        scene=dict(
            xaxis_title="X Position",
            yaxis_title="Y Position",
            zaxis_title="Z Position",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=400
    )
    
    return fig

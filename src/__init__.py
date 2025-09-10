"""
Biomechanical Movement Tracker Package
"""

from .biomechanical_calculator import BiomechanicalCalculator
from .video_processor import VideoProcessor
from .visualization import (
    create_velocity_plot,
    create_energy_plot,
    create_intensity_gauge,
    create_3d_trajectory
)

__version__ = "1.0.0"
__author__ = "Biomechanics Research Team"

__all__ = [
    "BiomechanicalCalculator",
    "VideoProcessor",
    "create_velocity_plot",
    "create_energy_plot",
    "create_intensity_gauge",
    "create_3d_trajectory"
]

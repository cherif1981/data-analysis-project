# src/visualizer/__init__.py
"""
Visualization module
"""

from .visualizer import Visualizer
from .dashboard import DashboardGenerator  # ✅ Add this line

__all__ = ['Visualizer', 'DashboardGenerator']
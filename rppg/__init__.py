"""
rPPG Real-time Heart Rate Detection
POS Method Implementation

Modules:
- config: Configuration parameters
- face_detector: Face detection and ROI extraction
- signal_processor: Signal processing with POS method
- rppg_system: Main rPPG system
- visualizer: Visualization and UI
- main: Application entry point
"""

__version__ = "1.0.0"
__author__ = "Aloisius Edwin"
__description__ = "Real-time Heart Rate Detection using rPPG with POS Method"

from .config import Config, LandmarkIndices
from .face_detector import FaceDetector
from .signal_processor import SignalProcessor
from .rppg_system import RPPGSystem, HRTracker
from .visualizer import Visualizer, PerformanceMonitor

__all__ = [
    'Config',
    'LandmarkIndices',
    'FaceDetector',
    'SignalProcessor',
    'RPPGSystem',
    'HRTracker',
    'Visualizer',
    'PerformanceMonitor'
]

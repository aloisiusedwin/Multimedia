"""
rPPG System Module
Main system class yang mengintegrasikan semua komponen
"""

import time
import numpy as np
from collections import deque
from typing import Tuple, Optional, List

from config import Config
from face_detector import FaceDetector
from signal_processor import SignalProcessor


class RPPGSystem:
    """Real-time rPPG system dengan sliding window processing"""
    
    def __init__(self):
        """Initialize rPPG system"""
        print("Initializing rPPG System...")
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.signal_processor = SignalProcessor(fps=Config.FPS)
        
        # Signal buffers (sliding window)
        self.rgb_buffer = deque(maxlen=Config.WINDOW_SIZE)
        self.timestamp_buffer = deque(maxlen=Config.WINDOW_SIZE)
        
        # Results
        self.current_hr = 0.0
        self.hr_history = deque(maxlen=100)  # Store last 100 measurements
        self.filtered_signal = np.array([])
        self.frequencies = np.array([])
        self.power_spectrum = np.array([])
        
        # State
        self.is_ready = False
        self.last_roi_mask = None
        
        print("✓ rPPG System initialized")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Process single frame dan update signal buffer
        
        Args:
            frame: Input BGR frame dari webcam
            
        Returns:
            Tuple of (success, roi_mask)
            - success: True jika face detected dan data added to buffer
            - roi_mask: ROI mask untuk visualization
        """
        # Process frame dengan face detector
        success, roi_mask, roi_pixels = self.face_detector.process_frame(frame)
        
        if not success or roi_pixels is None:
            return False, roi_mask
        
        # Spatial averaging untuk get RGB mean
        rgb_mean = self.signal_processor.spatial_averaging(roi_pixels)
        
        # Add to buffer
        self.rgb_buffer.append(rgb_mean)
        self.timestamp_buffer.append(time.time())
        
        # Store ROI mask untuk visualization
        self.last_roi_mask = roi_mask
        
        # Update ready status
        if len(self.rgb_buffer) >= Config.WINDOW_SIZE:
            self.is_ready = True
        
        return True, roi_mask
    
    def update_heart_rate(self) -> bool:
        """
        Update heart rate estimation dari signal buffer
        
        Returns:
            True jika berhasil update, False jika buffer belum cukup
        """
        # Check jika buffer sudah cukup
        if len(self.rgb_buffer) < Config.WINDOW_SIZE:
            return False
        
        # Convert buffer to numpy array
        rgb_signals = np.array(self.rgb_buffer)
        
        # Process signal dengan signal processor
        hr, filtered, freqs, power = self.signal_processor.process_signal(rgb_signals)
        
        # Validate heart rate
        if self.signal_processor.validate_heart_rate(hr):
            self.current_hr = hr
            self.hr_history.append(hr)
        
        # Store signal dan spectrum untuk visualization
        self.filtered_signal = filtered
        self.frequencies = freqs
        self.power_spectrum = power
        
        return True
    
    def get_heart_rate(self) -> float:
        """
        Get current heart rate
        
        Returns:
            Current heart rate in BPM
        """
        return self.current_hr
    
    def get_smoothed_heart_rate(self, window: int = None) -> float:
        """
        Get smoothed heart rate menggunakan moving average
        
        Args:
            window: Moving average window size (default: Config.HR_SMOOTHING_WINDOW)
            
        Returns:
            Smoothed heart rate in BPM
        """
        if window is None:
            window = Config.HR_SMOOTHING_WINDOW
        
        if len(self.hr_history) < window:
            return self.current_hr
        
        # Get recent measurements
        recent_hrs = list(self.hr_history)[-window:]
        
        # Filter outliers (optional)
        valid_hrs = [hr for hr in recent_hrs 
                    if Config.HR_MIN <= hr <= Config.HR_MAX]
        
        if len(valid_hrs) == 0:
            return self.current_hr
        
        return np.mean(valid_hrs)
    
    def get_buffer_size(self) -> int:
        """Get current buffer size"""
        return len(self.rgb_buffer)
    
    def get_buffer_percentage(self) -> float:
        """Get buffer fill percentage"""
        return (len(self.rgb_buffer) / Config.WINDOW_SIZE) * 100
    
    def is_system_ready(self) -> bool:
        """Check jika system ready untuk heart rate estimation"""
        return self.is_ready
    
    def get_status(self) -> str:
        """Get system status string"""
        if not self.is_ready:
            return "Initializing..."
        elif self.current_hr == 0:
            return "Processing..."
        else:
            return "Ready"
    
    def get_signal_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get signal data untuk visualization
        
        Returns:
            Tuple of (filtered_signal, frequencies, power_spectrum)
        """
        return self.filtered_signal, self.frequencies, self.power_spectrum
    
    def get_statistics(self) -> dict:
        """
        Get statistics dari HR measurements
        
        Returns:
            Dictionary containing statistics
        """
        if len(self.hr_history) == 0:
            return {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        # Filter valid measurements
        valid_hrs = [hr for hr in self.hr_history 
                    if Config.HR_MIN <= hr <= Config.HR_MAX]
        
        if len(valid_hrs) == 0:
            return {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        return {
            'count': len(valid_hrs),
            'mean': np.mean(valid_hrs),
            'std': np.std(valid_hrs),
            'min': np.min(valid_hrs),
            'max': np.max(valid_hrs),
            'median': np.median(valid_hrs)
        }
    
    def reset(self):
        """Reset system state"""
        self.rgb_buffer.clear()
        self.timestamp_buffer.clear()
        self.hr_history.clear()
        self.current_hr = 0.0
        self.is_ready = False
        self.filtered_signal = np.array([])
        self.frequencies = np.array([])
        self.power_spectrum = np.array([])
        
        if Config.DEBUG_MODE:
            print("✓ System reset")
    
    def print_statistics(self):
        """Print session statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("Session Statistics")
        print("="*60)
        
        if stats['count'] > 0:
            print(f"Total measurements: {stats['count']}")
            print(f"Mean HR: {stats['mean']:.2f} BPM")
            print(f"Median HR: {stats['median']:.2f} BPM")
            print(f"Std Dev: {stats['std']:.2f} BPM")
            print(f"Min HR: {stats['min']:.2f} BPM")
            print(f"Max HR: {stats['max']:.2f} BPM")
            print(f"Range: {stats['max'] - stats['min']:.2f} BPM")
        else:
            print("No valid measurements recorded")
        
        print("="*60)
    
    def release(self):
        """Release all resources"""
        self.face_detector.release()
        
        if Config.DEBUG_MODE:
            print("✓ rPPG System released")


class HRTracker:
    """Helper class untuk tracking dan analyzing heart rate trends"""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize HR tracker
        
        Args:
            window_size: Maximum number of measurements to track
        """
        self.measurements = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
    
    def add_measurement(self, hr: float, timestamp: float = None):
        """
        Add new heart rate measurement
        
        Args:
            hr: Heart rate in BPM
            timestamp: Timestamp (default: current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.measurements.append(hr)
        self.timestamps.append(timestamp)
    
    def get_trend(self, window: int = 10) -> str:
        """
        Analyze HR trend (increasing, decreasing, stable)
        
        Args:
            window: Number of recent measurements to analyze
            
        Returns:
            Trend string: "increasing", "decreasing", "stable"
        """
        if len(self.measurements) < window:
            return "insufficient_data"
        
        recent = list(self.measurements)[-window:]
        
        # Linear regression untuk detect trend
        x = np.arange(len(recent))
        y = np.array(recent)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Threshold untuk considering trend
        threshold = 0.5  # BPM per measurement
        
        if slope > threshold:
            return "increasing"
        elif slope < -threshold:
            return "decreasing"
        else:
            return "stable"
    
    def get_variability(self) -> float:
        """
        Calculate heart rate variability (simplified)
        
        Returns:
            Standard deviation of HR measurements
        """
        if len(self.measurements) < 2:
            return 0.0
        
        return np.std(self.measurements)
    
    def detect_anomaly(self, threshold_std: float = 2.0) -> bool:
        """
        Detect jika latest measurement adalah anomaly
        
        Args:
            threshold_std: Number of standard deviations untuk threshold
            
        Returns:
            True jika anomaly detected
        """
        if len(self.measurements) < 3:
            return False
        
        mean = np.mean(self.measurements)
        std = np.std(self.measurements)
        latest = self.measurements[-1]
        
        return abs(latest - mean) > threshold_std * std


if __name__ == "__main__":
    """Test rPPG system"""
    import cv2
    
    print("Testing RPPGSystem...")
    print("Press 'q' to quit, 'r' to reset\n")
    
    # Initialize system
    system = RPPGSystem()
    
    # Open camera
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        exit(1)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            success, roi_mask = system.process_frame(frame)
            
            # Update HR periodically
            if frame_count % Config.OVERLAP == 0:
                system.update_heart_rate()
            
            # Display info
            hr = system.get_smoothed_heart_rate()
            status = system.get_status()
            
            cv2.putText(frame, f"HR: {hr:.1f} BPM", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"Buffer: {system.get_buffer_size()}/{Config.WINDOW_SIZE}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {status}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("rPPG System Test", frame)
            
            frame_count += 1
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                system.reset()
                print("System reset")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        system.print_statistics()
        system.release()
        
        print("\nTest complete!")

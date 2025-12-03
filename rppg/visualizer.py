"""
Visualization Module
Real-time plotting dan UI overlay
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Tuple

from config import Config


class Visualizer:
    """Real-time visualization untuk rPPG signals dan metrics"""
    
    def __init__(self, enable_plot: bool = Config.SHOW_VISUALIZATION):
        """
        Initialize visualizer
        
        Args:
            enable_plot: Enable matplotlib real-time plotting
        """
        self.enable_plot = enable_plot
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.line_signal = None
        self.line_spectrum = None
        self.hr_text = None
        
        if self.enable_plot:
            self._setup_plots()
        
        if Config.DEBUG_MODE:
            print("✓ Visualizer initialized")
    
    def _setup_plots(self):
        """Setup matplotlib figures dan axes"""
        plt.ion()  # Interactive mode
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('Real-time rPPG Monitoring (POS Method)', 
                         fontsize=14, fontweight='bold')
        
        # Signal plot
        self.line_signal, = self.ax1.plot([], [], 'b-', linewidth=1.5)
        self.ax1.set_xlabel('Time (seconds)')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.set_title('Filtered Pulse Signal')
        self.ax1.grid(True, alpha=0.3)
        
        # Spectrum plot
        self.line_spectrum, = self.ax2.plot([], [], 'r-', linewidth=1.5)
        self.ax2.set_xlabel('Frequency (Hz)')
        self.ax2.set_ylabel('Power')
        self.ax2.set_title('Power Spectrum')
        self.ax2.set_xlim(Config.FREQ_MIN, Config.FREQ_MAX)
        self.ax2.grid(True, alpha=0.3)
        
        # HR annotation
        self.hr_text = self.ax2.text(
            0.02, 0.95, '', 
            transform=self.ax2.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=12, 
            fontweight='bold'
        )
        
        plt.tight_layout()
        plt.show(block=False)
    
    def update_plots(self, 
                    filtered_signal: np.ndarray,
                    frequencies: np.ndarray,
                    power_spectrum: np.ndarray,
                    heart_rate: float):
        """
        Update real-time plots dengan data terbaru
        
        Args:
            filtered_signal: Filtered pulse signal
            frequencies: Frequency array dari FFT
            power_spectrum: Power spectrum dari FFT
            heart_rate: Current heart rate in BPM
        """
        if not self.enable_plot or self.fig is None:
            return
        
        try:
            # Update signal plot
            if len(filtered_signal) > 0:
                time_axis = np.arange(len(filtered_signal)) / Config.FPS
                self.line_signal.set_data(time_axis, filtered_signal)
                
                # Auto-scale axes
                self.ax1.set_xlim(0, len(filtered_signal) / Config.FPS)
                
                y_min, y_max = np.min(filtered_signal), np.max(filtered_signal)
                margin = (y_max - y_min) * 0.1 if y_max > y_min else 1
                self.ax1.set_ylim(y_min - margin, y_max + margin)
            
            # Update spectrum plot
            if len(frequencies) > 0 and len(power_spectrum) > 0:
                # Filter to display range
                valid_idx = (frequencies >= Config.FREQ_MIN) & (frequencies <= Config.FREQ_MAX)
                valid_freq = frequencies[valid_idx]
                valid_power = power_spectrum[valid_idx]
                
                if len(valid_freq) > 0:
                    self.line_spectrum.set_data(valid_freq, valid_power)
                    
                    # Auto-scale y-axis
                    max_power = np.max(valid_power)
                    if max_power > 0:
                        self.ax2.set_ylim(0, max_power * 1.1)
            
            # Update HR text
            self.hr_text.set_text(f'HR: {heart_rate:.1f} BPM')
            
            # Redraw
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        
        except Exception as e:
            if Config.DEBUG_MODE:
                print(f"Plot update error: {e}")
    
    def draw_info_overlay(self,
                         frame: np.ndarray,
                         heart_rate: float,
                         buffer_size: int,
                         fps: float,
                         status: str = "Ready") -> np.ndarray:
        """
        Draw information overlay pada frame
        
        Args:
            frame: Input frame
            heart_rate: Current heart rate
            buffer_size: Current buffer size
            fps: Current FPS
            status: System status
            
        Returns:
            Frame dengan info overlay
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Semi-transparent background box
        box_width = 320
        box_height = 160
        cv2.rectangle(overlay, (10, 10), (10 + box_width, 10 + box_height), 
                     Config.COLOR_INFO_BG, -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Text information
        y_offset = 45
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Heart rate (large, prominent)
        cv2.putText(frame, f"Heart Rate: {heart_rate:.1f} BPM", 
                   (20, y_offset), font, 0.8, Config.COLOR_TEXT_PRIMARY, 2)
        
        y_offset += 35
        # Buffer status
        buffer_text = f"Buffer: {buffer_size}/{Config.WINDOW_SIZE}"
        buffer_pct = (buffer_size / Config.WINDOW_SIZE) * 100
        cv2.putText(frame, buffer_text, (20, y_offset), font, 0.6, 
                   Config.COLOR_TEXT_SECONDARY, 1)
        
        # Progress bar untuk buffer
        bar_x = 20
        bar_y = y_offset + 10
        bar_width = 280
        bar_height = 10
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)
        
        # Progress bar
        progress_width = int(bar_width * buffer_pct / 100)
        bar_color = Config.COLOR_TEXT_PRIMARY if buffer_pct >= 100 else Config.COLOR_TEXT_WARNING
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + progress_width, bar_y + bar_height),
                     bar_color, -1)
        
        y_offset += 40
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset), font, 0.6,
                   Config.COLOR_TEXT_SECONDARY, 1)
        
        y_offset += 30
        # Status
        status_color = Config.COLOR_TEXT_PRIMARY if status == "Ready" else Config.COLOR_TEXT_WARNING
        cv2.putText(frame, f"Status: {status}", (20, y_offset), font, 0.6,
                   status_color, 1)
        
        # Instructions (bottom)
        instructions = [
            "Press 'q' to quit",
            "Press 's' to save frame"
        ]
        
        y_bottom = h - 20
        for i, instruction in enumerate(reversed(instructions)):
            cv2.putText(frame, instruction, (20, y_bottom - i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def draw_roi_overlay(self, 
                        frame: np.ndarray, 
                        mask: np.ndarray, 
                        alpha: float = 0.3) -> np.ndarray:
        """
        Draw ROI overlay pada frame
        
        Args:
            frame: Input frame
            mask: ROI mask
            alpha: Transparency
            
        Returns:
            Frame dengan ROI overlay
        """
        if mask is None:
            return frame
        
        overlay = frame.copy()
        overlay[mask > 0] = Config.COLOR_OVERLAY
        
        return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    
    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """
        Draw FPS counter (simple version)
        
        Args:
            frame: Input frame
            fps: Current FPS
            
        Returns:
            Frame dengan FPS counter
        """
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return frame
    
    def save_frame(self, frame: np.ndarray, filename: str = None) -> str:
        """
        Save current frame to file
        
        Args:
            frame: Frame to save
            filename: Output filename (optional, auto-generated if None)
            
        Returns:
            Saved filename
        """
        if filename is None:
            import time
            timestamp = int(time.time())
            filename = f"rppg_capture_{timestamp}.png"
        
        cv2.imwrite(filename, frame)
        return filename
    
    def close(self):
        """Close all visualization windows"""
        if self.enable_plot and self.fig is not None:
            plt.close(self.fig)
        
        cv2.destroyAllWindows()
        
        if Config.DEBUG_MODE:
            print("✓ Visualizer closed")


class PerformanceMonitor:
    """Monitor dan display performance metrics"""
    
    def __init__(self):
        """Initialize performance monitor"""
        self.frame_count = 0
        self.start_time = None
        self.fps = 0.0
        self.last_update = 0.0
    
    def start(self):
        """Start monitoring"""
        import time
        self.start_time = time.time()
        self.last_update = self.start_time
    
    def update(self) -> float:
        """
        Update FPS calculation
        
        Returns:
            Current FPS
        """
        import time
        
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_update
        
        # Update FPS every interval
        if elapsed >= Config.FPS_UPDATE_INTERVAL:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_update = current_time
        
        return self.fps
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.fps
    
    def reset(self):
        """Reset counters"""
        import time
        self.frame_count = 0
        self.start_time = time.time()
        self.last_update = self.start_time
        self.fps = 0.0


if __name__ == "__main__":
    """Test visualizer"""
    print("Testing Visualizer...")
    
    # Create test data
    visualizer = Visualizer(enable_plot=True)
    
    # Generate synthetic signal
    t = np.linspace(0, 10, 300)
    signal = np.sin(2 * np.pi * 1.2 * t)  # 72 BPM
    
    # Generate spectrum
    from scipy.fft import fft, fftfreq
    freqs = fftfreq(len(signal), 1/30)
    power = np.abs(fft(signal)) ** 2
    
    # Update plots
    visualizer.update_plots(signal, freqs, power, 72.0)
    
    print("Plot updated. Close window to continue...")
    input("Press Enter to close...")
    
    visualizer.close()
    print("Test complete!")

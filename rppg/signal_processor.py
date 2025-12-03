"""
Signal Processing Module
Implementasi POS method dan signal processing pipeline
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Tuple, Optional

from config import Config


class SignalProcessor:
    """Signal processing untuk rPPG menggunakan POS method"""
    
    def __init__(self, fps: float = Config.FPS):
        """
        Initialize signal processor
        
        Args:
            fps: Frame rate untuk filter design
        """
        self.fps = fps
        self.nyquist = fps / 2.0
        
        # Design bandpass filter
        self.sos = self._design_bandpass_filter()
        
        if Config.DEBUG_MODE:
            print(f"✓ SignalProcessor initialized (FPS: {fps})")
    
    def _design_bandpass_filter(self):
        """
        Design Butterworth bandpass filter
        
        Returns:
            Second-order sections representation dari filter
        """
        sos = signal.butter(
            Config.FILTER_ORDER,
            [Config.FREQ_MIN / self.nyquist, Config.FREQ_MAX / self.nyquist],
            btype='band',
            output='sos'
        )
        return sos
    
    def spatial_averaging(self, roi_pixels: np.ndarray) -> np.ndarray:
        """
        Compute spatial average dari ROI pixels
        
        Args:
            roi_pixels: Array of RGB pixels (N x 3)
            
        Returns:
            Mean RGB values (3,)
        """
        return np.mean(roi_pixels, axis=0)
    
    def normalize_signal(self, signal_window: np.ndarray) -> np.ndarray:
        """
        Normalize signal window
        
        Normalisasi dilakukan dengan membagi setiap channel dengan mean-nya
        untuk menghilangkan DC component dan scale differences.
        
        Args:
            signal_window: RGB signal array (T x 3)
            
        Returns:
            Normalized signal (T x 3)
        """
        mean = np.mean(signal_window, axis=0)
        
        # Avoid division by zero
        mean = np.where(mean == 0, 1, mean)
        
        normalized = signal_window / mean
        
        return normalized
    
    def pos_method(self, signal_window: np.ndarray) -> np.ndarray:
        """
        Apply POS (Plane-Orthogonal-to-Skin) method
        
        POS method projects RGB signals onto a plane orthogonal to skin color,
        reducing motion artifacts and illumination changes.
        
        Reference: 
        Wang, W., et al. (2017). Algorithmic Principles of Remote PPG.
        IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.
        
        Math:
            S1 = C1 - C2 (Red - Green)
            S2 = C1 + C2 - 2*C3 (Red + Green - 2*Blue)
            α = σ(S1) / σ(S2)
            Pulse = S1 - α * S2
        
        Args:
            signal_window: Normalized RGB signals (T x 3)
            
        Returns:
            Pulse signal (T,)
        """
        # Transpose untuk easier computation: (3 x T)
        C = signal_window.T
        
        # Projection matrices
        S1 = C[0] - C[1]  # Red - Green
        S2 = C[0] + C[1] - 2 * C[2]  # Red + Green - 2*Blue
        
        # Calculate standard deviations
        std_S1 = np.std(S1)
        std_S2 = np.std(S2)
        
        # Avoid division by zero
        if std_S1 == 0:
            std_S1 = 1e-10
        if std_S2 == 0:
            std_S2 = 1e-10
        
        # Alpha calculation (ratio of standard deviations)
        alpha = std_S1 / std_S2
        
        # Pulse signal: orthogonal to skin color plane
        pulse_signal = S1 - alpha * S2
        
        return pulse_signal
    
    def detrend_signal(self, pulse_signal: np.ndarray, lambda_param: int = None) -> np.ndarray:
        """
        Remove trend dari pulse signal menggunakan moving average
        
        Detrending menghilangkan slow variations (DC drift) yang bukan
        merupakan bagian dari pulse signal.
        
        Args:
            pulse_signal: Raw pulse signal
            lambda_param: Window size untuk moving average (default: Config.DETREND_LAMBDA)
            
        Returns:
            Detrended signal
        """
        if lambda_param is None:
            lambda_param = Config.DETREND_LAMBDA
        
        # Simple case: subtract mean jika signal terlalu pendek
        if len(pulse_signal) < lambda_param:
            return pulse_signal - np.mean(pulse_signal)
        
        # Moving average filter untuk estimate trend
        kernel = np.ones(lambda_param) / lambda_param
        trend = np.convolve(pulse_signal, kernel, mode='same')
        
        # Remove trend
        detrended = pulse_signal - trend
        
        return detrended
    
    def bandpass_filter(self, pulse_signal: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter (0.67 - 4.0 Hz untuk 40-240 BPM)
        
        Args:
            pulse_signal: Input signal
            
        Returns:
            Filtered signal
        """
        # Check minimum signal length untuk filtering
        if len(pulse_signal) < Config.FILTER_ORDER * 3:
            return pulse_signal
        
        try:
            # Apply zero-phase filtering (forward-backward filter)
            filtered = signal.sosfiltfilt(self.sos, pulse_signal)
            return filtered
        
        except Exception as e:
            if Config.DEBUG_MODE:
                print(f"Filter error: {e}")
            return pulse_signal
    
    def estimate_heart_rate_fft(self, pulse_signal: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Estimate heart rate menggunakan Fast Fourier Transform (FFT)
        
        FFT mengkonversi signal dari time domain ke frequency domain,
        kemudian mencari peak frequency yang corresponds to heart rate.
        
        Args:
            pulse_signal: Filtered pulse signal
            
        Returns:
            Tuple of (heart_rate_bpm, frequencies, power_spectrum)
        """
        N = len(pulse_signal)
        
        # Compute FFT
        fft_values = fft(pulse_signal)
        fft_freq = fftfreq(N, 1.0 / self.fps)
        
        # Get positive frequencies only (symmetry property)
        positive_freq_idx = fft_freq > 0
        fft_freq = fft_freq[positive_freq_idx]
        
        # Compute power spectrum
        fft_power = np.abs(fft_values[positive_freq_idx]) ** 2
        
        # Filter valid frequency range (0.67 - 4.0 Hz)
        valid_idx = (fft_freq >= Config.FREQ_MIN) & (fft_freq <= Config.FREQ_MAX)
        valid_freq = fft_freq[valid_idx]
        valid_power = fft_power[valid_idx]
        
        # Handle empty case
        if len(valid_power) == 0:
            return 0.0, fft_freq, fft_power
        
        # Find peak frequency (dominant frequency)
        peak_idx = np.argmax(valid_power)
        peak_freq = valid_freq[peak_idx]
        
        # Convert frequency to BPM
        heart_rate = peak_freq * 60.0
        
        return heart_rate, fft_freq, fft_power
    
    def estimate_heart_rate_peaks(self, pulse_signal: np.ndarray) -> float:
        """
        Estimate heart rate menggunakan peak detection (alternative method)
        
        Args:
            pulse_signal: Filtered pulse signal
            
        Returns:
            Heart rate in BPM
        """
        from scipy.signal import find_peaks
        
        # Find peaks dalam signal
        peaks, _ = find_peaks(pulse_signal, distance=self.fps * 0.5)  # Min 0.5s between peaks
        
        if len(peaks) < 2:
            return 0.0
        
        # Calculate average interval between peaks
        peak_intervals = np.diff(peaks)
        avg_interval = np.mean(peak_intervals)
        
        # Convert to BPM
        heart_rate = (self.fps / avg_interval) * 60.0
        
        # Validate range
        if Config.HR_MIN <= heart_rate <= Config.HR_MAX:
            return heart_rate
        else:
            return 0.0
    
    def process_signal(self, rgb_signals: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete signal processing pipeline
        
        Pipeline:
        1. Normalize RGB signals
        2. Apply POS method untuk extract pulse
        3. Detrend pulse signal
        4. Bandpass filter
        5. Estimate heart rate dengan FFT
        
        Args:
            rgb_signals: Raw RGB signals (T x 3) dari spatial averaging
            
        Returns:
            Tuple of (heart_rate, filtered_signal, frequencies, power_spectrum)
        """
        # Check minimum signal length
        if len(rgb_signals) < Config.POS_L:
            return 0.0, np.array([]), np.array([]), np.array([])
        
        # 1. Normalize signals
        normalized = self.normalize_signal(rgb_signals)
        
        # 2. Apply POS method
        pulse_signal = self.pos_method(normalized)
        
        # 3. Detrend
        detrended = self.detrend_signal(pulse_signal)
        
        # 4. Bandpass filter
        filtered = self.bandpass_filter(detrended)
        
        # 5. Estimate heart rate
        hr, freqs, power = self.estimate_heart_rate_fft(filtered)
        
        return hr, filtered, freqs, power
    
    def validate_heart_rate(self, hr: float) -> bool:
        """
        Validate jika heart rate dalam range yang valid
        
        Args:
            hr: Heart rate in BPM
            
        Returns:
            True jika valid, False otherwise
        """
        return Config.HR_MIN <= hr <= Config.HR_MAX


if __name__ == "__main__":
    """Test signal processor dengan synthetic signal"""
    print("Testing SignalProcessor...")
    
    # Create synthetic RGB signal dengan known frequency
    processor = SignalProcessor()
    
    # Generate synthetic pulse signal (72 BPM = 1.2 Hz)
    duration = 10  # seconds
    t = np.linspace(0, duration, int(Config.FPS * duration))
    true_hr = 72  # BPM
    true_freq = true_hr / 60.0  # Hz
    
    # Create RGB signals dengan pulse
    pulse = np.sin(2 * np.pi * true_freq * t)
    rgb_signals = np.column_stack([
        pulse + 0.5 * np.random.randn(len(t)),  # Red
        pulse + 0.5 * np.random.randn(len(t)),  # Green
        pulse + 0.5 * np.random.randn(len(t))   # Blue
    ])
    
    # Add DC offset
    rgb_signals += 128
    
    # Process signal
    estimated_hr, filtered, freqs, power = processor.process_signal(rgb_signals)
    
    # Display results
    print(f"\nTrue HR: {true_hr} BPM")
    print(f"Estimated HR: {estimated_hr:.1f} BPM")
    print(f"Error: {abs(estimated_hr - true_hr):.1f} BPM")
    print(f"Valid: {processor.validate_heart_rate(estimated_hr)}")
    
    # Plot results (optional, requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Signal plot
        ax1.plot(t, filtered)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Filtered Pulse Signal')
        ax1.grid(True, alpha=0.3)
        
        # Spectrum plot
        valid_idx = (freqs >= Config.FREQ_MIN) & (freqs <= Config.FREQ_MAX)
        ax2.plot(freqs[valid_idx] * 60, power[valid_idx])  # Convert to BPM
        ax2.axvline(estimated_hr, color='r', linestyle='--', label=f'Estimated: {estimated_hr:.1f} BPM')
        ax2.axvline(true_hr, color='g', linestyle='--', label=f'True: {true_hr} BPM')
        ax2.set_xlabel('Heart Rate (BPM)')
        ax2.set_ylabel('Power')
        ax2.set_title('Power Spectrum')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available for plotting")
    
    print("\nTest complete!")

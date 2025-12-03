"""
Configuration Module untuk rPPG System
Berisi semua constants dan parameters yang digunakan dalam sistem
"""


class Config:
    """Configuration parameters untuk rPPG system"""
    
    # ==================== Video Parameters ====================
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30  # Target frame rate
    
    # ==================== Signal Processing Parameters ====================
    WINDOW_SIZE = 300  # 10 seconds at 30 FPS
    OVERLAP = 30  # Frame overlap untuk smooth transition
    
    # ==================== Heart Rate Constraints (BPM) ====================
    HR_MIN = 40  # Minimum heart rate
    HR_MAX = 240  # Maximum heart rate
    FREQ_MIN = HR_MIN / 60.0  # 0.67 Hz
    FREQ_MAX = HR_MAX / 60.0  # 4.0 Hz
    
    # ==================== Bandpass Filter Parameters ====================
    FILTER_ORDER = 4
    
    # ==================== MediaPipe Parameters ====================
    DETECTION_CONFIDENCE = 0.7
    TRACKING_CONFIDENCE = 0.7
    
    # ==================== POS Parameters ====================
    POS_L = 32  # Window length untuk POS method
    
    # ==================== Skin Segmentation Parameters ====================
    # HSV color range untuk skin detection
    SKIN_LOWER_HSV = (0, 20, 70)
    SKIN_UPPER_HSV = (20, 255, 255)
    
    # Minimum pixel threshold untuk valid ROI
    MIN_ROI_PIXELS = 100
    
    # ==================== Visualization Parameters ====================
    SHOW_VISUALIZATION = True
    PLOT_UPDATE_INTERVAL = 30  # Update plot setiap N frames
    
    # UI colors (BGR format untuk OpenCV)
    COLOR_OVERLAY = (0, 255, 0)  # Green untuk ROI overlay
    COLOR_INFO_BG = (0, 0, 0)  # Black untuk info background
    COLOR_TEXT_PRIMARY = (0, 255, 0)  # Green untuk text utama
    COLOR_TEXT_SECONDARY = (255, 255, 255)  # White untuk text sekunder
    COLOR_TEXT_WARNING = (0, 255, 255)  # Yellow untuk warning
    
    # ==================== Performance Parameters ====================
    HR_SMOOTHING_WINDOW = 5  # Window untuk moving average HR
    FPS_UPDATE_INTERVAL = 1.0  # Update FPS setiap N seconds
    
    # ==================== Detrending Parameters ====================
    DETREND_LAMBDA = 300  # Window size untuk moving average detrending
    
    # ==================== Debug Parameters ====================
    DEBUG_MODE = False
    SAVE_FRAMES = False
    LOG_LEVEL = "INFO"  # INFO, DEBUG, WARNING, ERROR
    
    @classmethod
    def print_config(cls):
        """Print semua konfigurasi"""
        print("=" * 60)
        print("rPPG System Configuration")
        print("=" * 60)
        print(f"Camera: Index {cls.CAMERA_INDEX}, {cls.FRAME_WIDTH}x{cls.FRAME_HEIGHT} @ {cls.FPS} FPS")
        print(f"Signal Window: {cls.WINDOW_SIZE} frames ({cls.WINDOW_SIZE/cls.FPS:.1f} seconds)")
        print(f"Heart Rate Range: {cls.HR_MIN}-{cls.HR_MAX} BPM")
        print(f"Frequency Range: {cls.FREQ_MIN:.2f}-{cls.FREQ_MAX:.2f} Hz")
        print(f"Bandpass Filter Order: {cls.FILTER_ORDER}")
        print(f"Visualization: {'Enabled' if cls.SHOW_VISUALIZATION else 'Disabled'}")
        print("=" * 60)
    
    @classmethod
    def validate(cls):
        """Validate konfigurasi parameters"""
        assert cls.FPS > 0, "FPS must be positive"
        assert cls.WINDOW_SIZE > cls.POS_L, "WINDOW_SIZE must be larger than POS_L"
        assert cls.HR_MIN < cls.HR_MAX, "HR_MIN must be less than HR_MAX"
        assert 0 <= cls.DETECTION_CONFIDENCE <= 1, "DETECTION_CONFIDENCE must be in [0, 1]"
        assert 0 <= cls.TRACKING_CONFIDENCE <= 1, "TRACKING_CONFIDENCE must be in [0, 1]"
        print("✓ Configuration validation passed")


# Facial landmark indices untuk ROI extraction
class LandmarkIndices:
    """Indices untuk MediaPipe face landmarks"""
    
    # Left cheek landmarks
    LEFT_CHEEK = [
        116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
        126, 127, 128, 129, 130
    ]
    
    # Right cheek landmarks
    RIGHT_CHEEK = [
        345, 346, 347, 348, 349, 350, 351, 352, 353, 354,
        355, 356, 357, 358, 359
    ]
    
    # Forehead landmarks
    FOREHEAD = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127
    ]
    
    @classmethod
    def get_all_roi_indices(cls):
        """Get semua ROI landmark indices"""
        return cls.LEFT_CHEEK + cls.RIGHT_CHEEK + cls.FOREHEAD


if __name__ == "__main__":
    # Test configuration
    Config.validate()
    Config.print_config()
    
    print("\nROI Landmark Indices:")
    print(f"Left Cheek: {len(LandmarkIndices.LEFT_CHEEK)} points")
    print(f"Right Cheek: {len(LandmarkIndices.RIGHT_CHEEK)} points")
    print(f"Forehead: {len(LandmarkIndices.FOREHEAD)} points")
    print(f"Total ROI: {len(LandmarkIndices.get_all_roi_indices())} points")

"""
Face Detection Module
Menggunakan MediaPipe untuk deteksi wajah dan ekstraksi ROI
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple

from config import Config, LandmarkIndices


class FaceDetector:
    """Face detection dan ROI extraction menggunakan MediaPipe Face Mesh"""
    
    def __init__(self):
        """Initialize MediaPipe Face Mesh"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=Config.DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.TRACKING_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Get ROI indices
        self.roi_indices = LandmarkIndices.get_all_roi_indices()
        
        if Config.DEBUG_MODE:
            print("✓ FaceDetector initialized")
    
    def detect_face(self, frame: np.ndarray) -> Optional[object]:
        """
        Detect face dalam frame
        
        Args:
            frame: Input BGR frame dari OpenCV
            
        Returns:
            Face landmarks jika terdeteksi, None jika tidak ada wajah
        """
        # Convert BGR ke RGB untuk MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.face_mesh.process(rgb_frame)
        
        # Return first face landmarks jika ada
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        
        return None
    
    def extract_roi_mask(self, frame: np.ndarray, face_landmarks: object) -> Optional[np.ndarray]:
        """
        Extract Region of Interest (ROI) mask dari wajah
        
        Menggunakan kombinasi cheeks dan forehead untuk mendapatkan
        sinyal yang lebih stabil dan representatif.
        
        Args:
            frame: Input BGR frame
            face_landmarks: MediaPipe face landmarks
            
        Returns:
            Binary mask (H x W) untuk ROI, atau None jika gagal
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Extract landmark coordinates
        landmarks = face_landmarks.landmark
        
        # Create polygon points dari ROI indices
        points = []
        for idx in self.roi_indices:
            landmark = landmarks[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append([x, y])
        
        if len(points) == 0:
            return None
        
        # Create convex hull untuk smooth boundary
        points_array = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(points_array)
        
        # Fill convex hull dengan white (255)
        cv2.fillConvexPoly(mask, hull, 255)
        
        return mask
    
    def apply_skin_segmentation(self, frame: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """
        Apply skin color segmentation untuk filtering non-skin pixels
        
        Args:
            frame: Input BGR frame
            roi_mask: Binary ROI mask
            
        Returns:
            Refined mask dengan skin segmentation
        """
        # Convert ke HSV untuk skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create skin mask berdasarkan HSV range
        lower_skin = np.array(Config.SKIN_LOWER_HSV, dtype=np.uint8)
        upper_skin = np.array(Config.SKIN_UPPER_HSV, dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Combine ROI mask dengan skin mask
        combined_mask = cv2.bitwise_and(roi_mask, skin_mask)
        
        return combined_mask
    
    def get_roi_pixels(self, frame: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract pixel values dari ROI dengan skin segmentation
        
        Args:
            frame: Input BGR frame
            mask: Binary mask untuk ROI
            
        Returns:
            Array of RGB pixels (N x 3) atau None jika insufficient pixels
        """
        if mask is None:
            return None
        
        # Apply skin segmentation
        refined_mask = self.apply_skin_segmentation(frame, mask)
        
        # Convert BGR ke RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract pixels dari ROI
        roi_pixels = rgb_frame[refined_mask > 0]
        
        # Check minimum pixel threshold
        if len(roi_pixels) < Config.MIN_ROI_PIXELS:
            return None
        
        return roi_pixels
    
    def draw_roi_overlay(self, frame: np.ndarray, mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        Visualize ROI pada frame dengan semi-transparent overlay
        
        Args:
            frame: Input frame
            mask: ROI mask
            alpha: Transparency level (0.0 - 1.0)
            
        Returns:
            Frame dengan ROI overlay
        """
        if mask is None:
            return frame
        
        # Create overlay
        overlay = frame.copy()
        overlay[mask > 0] = Config.COLOR_OVERLAY
        
        # Blend dengan original frame
        output = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        return output
    
    def draw_landmarks(self, frame: np.ndarray, face_landmarks: object) -> np.ndarray:
        """
        Draw face landmarks pada frame (untuk debugging)
        
        Args:
            frame: Input frame
            face_landmarks: MediaPipe face landmarks
            
        Returns:
            Frame dengan landmarks drawn
        """
        if face_landmarks is None:
            return frame
        
        # Draw landmarks
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=1, circle_radius=1
            )
        )
        
        return frame
    
    def process_frame(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Complete processing pipeline untuk single frame
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (success, roi_mask, roi_pixels)
            - success: Boolean indicating jika face detected dan ROI extracted
            - roi_mask: Binary mask untuk ROI
            - roi_pixels: RGB pixel values (N x 3)
        """
        # Detect face
        face_landmarks = self.detect_face(frame)
        
        if face_landmarks is None:
            return False, None, None
        
        # Extract ROI mask
        roi_mask = self.extract_roi_mask(frame, face_landmarks)
        
        if roi_mask is None:
            return False, None, None
        
        # Get ROI pixels
        roi_pixels = self.get_roi_pixels(frame, roi_mask)
        
        if roi_pixels is None:
            return False, roi_mask, None
        
        return True, roi_mask, roi_pixels
    
    def release(self):
        """Release MediaPipe resources"""
        self.face_mesh.close()
        
        if Config.DEBUG_MODE:
            print("✓ FaceDetector released")


if __name__ == "__main__":
    """Test face detector dengan webcam"""
    print("Testing FaceDetector...")
    print("Press 'q' to quit\n")
    
    # Initialize
    detector = FaceDetector()
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        exit(1)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            success, roi_mask, roi_pixels = detector.process_frame(frame)
            
            # Visualize
            if success and roi_mask is not None:
                frame = detector.draw_roi_overlay(frame, roi_mask)
                
                # Show info
                cv2.putText(frame, f"ROI Pixels: {len(roi_pixels) if roi_pixels is not None else 0}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Face Detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Face Detector Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        print("\nTest complete!")

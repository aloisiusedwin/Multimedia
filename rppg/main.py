"""
Main Application
Entry point untuk real-time rPPG system
"""

import cv2
import argparse
import time
from typing import Optional

from config import Config
from rppg_system import RPPGSystem
from visualizer import Visualizer, PerformanceMonitor


class RPPGApplication:
    """Main application class untuk rPPG system"""
    
    def __init__(self, camera_index: int = None, enable_visualization: bool = None):
        """
        Initialize application
        
        Args:
            camera_index: Camera device index (default: Config.CAMERA_INDEX)
            enable_visualization: Enable real-time plotting (default: Config.SHOW_VISUALIZATION)
        """
        # Override config jika provided
        if camera_index is not None:
            Config.CAMERA_INDEX = camera_index
        
        if enable_visualization is not None:
            Config.SHOW_VISUALIZATION = enable_visualization
        
        # Initialize components
        self.rppg_system = RPPGSystem()
        self.visualizer = Visualizer(enable_plot=Config.SHOW_VISUALIZATION)
        self.performance_monitor = PerformanceMonitor()
        
        # Camera
        self.camera = None
        
        # State
        self.running = False
        self.frame_count = 0
    
    def initialize_camera(self) -> bool:
        """
        Initialize camera
        
        Returns:
            True jika berhasil, False jika gagal
        """
        print(f"Opening camera {Config.CAMERA_INDEX}...")
        
        self.camera = cv2.VideoCapture(Config.CAMERA_INDEX)
        
        if not self.camera.isOpened():
            print("✗ Error: Cannot open camera")
            return False
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, Config.FPS)
        
        # Verify settings
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.camera.get(cv2.CAP_PROP_FPS))
        
        print(f"✓ Camera opened: {actual_width}x{actual_height} @ {actual_fps} FPS")
        
        return True
    
    def print_instructions(self):
        """Print usage instructions"""
        print("\n" + "="*70)
        print(" " * 15 + "Real-time rPPG Heart Rate Detection")
        print(" " * 20 + "POS Method Implementation")
        print("="*70)
        print("\n📋 Instructions:")
        print("  • Position your face in front of the camera")
        print("  • Keep still and ensure good lighting")
        print("  • Wait ~10 seconds for initialization")
        print("  • Green overlay shows detected face region (ROI)")
        print("\n⌨️  Keyboard Controls:")
        print("  • Press 'q' - Quit application")
        print("  • Press 's' - Save current frame")
        print("  • Press 'r' - Reset system")
        print("  • Press 'p' - Toggle visualization")
        print("="*70 + "\n")
    
    def process_keyboard_input(self, key: int) -> bool:
        """
        Process keyboard input
        
        Args:
            key: Key code dari cv2.waitKey()
            
        Returns:
            True untuk continue, False untuk quit
        """
        if key == ord('q'):
            print("\nQuitting...")
            return False
        
        elif key == ord('s'):
            # Save frame
            ret, frame = self.camera.read()
            if ret:
                filename = self.visualizer.save_frame(frame)
                print(f"✓ Frame saved: {filename}")
        
        elif key == ord('r'):
            # Reset system
            self.rppg_system.reset()
            print("✓ System reset")
        
        elif key == ord('p'):
            # Toggle visualization
            Config.SHOW_VISUALIZATION = not Config.SHOW_VISUALIZATION
            status = "enabled" if Config.SHOW_VISUALIZATION else "disabled"
            print(f"✓ Visualization {status}")
        
        return True
    
    def run(self):
        """Main application loop"""
        # Print instructions
        self.print_instructions()
        
        # Initialize camera
        if not self.initialize_camera():
            return
        
        # Start performance monitoring
        self.performance_monitor.start()
        
        # Set running state
        self.running = True
        
        print("🚀 System ready! Starting capture...\n")
        
        try:
            while self.running:
                # Read frame
                ret, frame = self.camera.read()
                
                if not ret:
                    print("✗ Error: Cannot read frame")
                    break
                
                # Process frame
                success, roi_mask = self.rppg_system.process_frame(frame)
                
                # Update heart rate periodically
                if self.frame_count % Config.OVERLAP == 0:
                    self.rppg_system.update_heart_rate()
                    
                    # Update visualization
                    if Config.SHOW_VISUALIZATION and self.frame_count % Config.PLOT_UPDATE_INTERVAL == 0:
                        filtered, freqs, power = self.rppg_system.get_signal_data()
                        hr = self.rppg_system.get_smoothed_heart_rate()
                        self.visualizer.update_plots(filtered, freqs, power, hr)
                
                # Draw ROI overlay
                if success and roi_mask is not None:
                    frame = self.visualizer.draw_roi_overlay(frame, roi_mask)
                
                # Update FPS
                fps = self.performance_monitor.update()
                
                # Draw info overlay
                hr = self.rppg_system.get_smoothed_heart_rate()
                buffer_size = self.rppg_system.get_buffer_size()
                status = self.rppg_system.get_status()
                
                frame = self.visualizer.draw_info_overlay(
                    frame, hr, buffer_size, fps, status
                )
                
                # Display frame
                cv2.imshow('rPPG - POS Method', frame)
                
                # Increment frame counter
                self.frame_count += 1
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key pressed
                    if not self.process_keyboard_input(key):
                        break
        
        except KeyboardInterrupt:
            print("\n✗ Interrupted by user")
        
        except Exception as e:
            print(f"\n✗ Error: {e}")
            if Config.DEBUG_MODE:
                import traceback
                traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up...")
        
        # Release camera
        if self.camera is not None:
            self.camera.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Release system
        self.rppg_system.release()
        
        # Close visualizer
        self.visualizer.close()
        
        # Print statistics
        self.rppg_system.print_statistics()
        
        print("\n✓ Done!")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Real-time Heart Rate Detection using rPPG with POS Method",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run dengan default settings
  python main.py --camera 1         # Gunakan camera index 1
  python main.py --no-viz           # Disable visualization
  python main.py --debug            # Enable debug mode
  python main.py --config           # Show configuration
        """
    )
    
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=None,
        help=f'Camera device index (default: {Config.CAMERA_INDEX})'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable real-time visualization plots'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--config',
        action='store_true',
        help='Show configuration and exit'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Set debug mode
    if args.debug:
        Config.DEBUG_MODE = True
    
    # Show config jika diminta
    if args.config:
        Config.validate()
        Config.print_config()
        return
    
    # Validate configuration
    try:
        Config.validate()
    except AssertionError as e:
        print(f"✗ Configuration error: {e}")
        return
    
    # Create dan run application
    app = RPPGApplication(
        camera_index=args.camera,
        enable_visualization=not args.no_viz
    )
    
    app.run()


if __name__ == "__main__":
    main()

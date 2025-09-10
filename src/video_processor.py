"""
Simplified video processing module using optical flow instead of MediaPipe
Alternative for Streamlit Cloud deployment issues
"""
import cv2
import numpy as np

class VideoProcessor:
    """
    Process video files using optical flow for movement detection
    Alternative to MediaPipe for better compatibility
    """
    
    def __init__(self):
        """Initialize video processor with optical flow parameters"""
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        self.timestamps = []
        self.motion_magnitudes = []
        
    def process_video(self, video_path, progress_bar=None):
        """
        Process video using dense optical flow (Farneback method)
        
        Args:
            video_path: Path to video file
            progress_bar: Streamlit progress bar object
            
        Returns:
            tuple: (velocity_magnitude, acceleration, intensity)
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0:
            fps = 30  # Default fallback
            
        # Read first frame
        ret, frame1 = cap.read()
        if not ret:
            return None, None, None
            
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        frame_count = 0
        flow_magnitudes = []
        
        while True:
            ret, frame2 = cap.read()
            if not ret:
                break
                
            next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prvs, next_gray, None,
                **self.flow_params
            )
            
            # Calculate magnitude of flow
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Average magnitude for this frame
            avg_magnitude = np.mean(magnitude)
            flow_magnitudes.append(avg_magnitude)
            self.timestamps.append(frame_count / fps)
            
            prvs = next_gray
            frame_count += 1
            
            # Update progress bar
            if progress_bar is not None:
                progress_bar.progress(min(frame_count / total_frames, 1.0))
        
        cap.release()
        
        # Convert to motion metrics
        return self.analyze_motion(flow_magnitudes, fps)
    
    def analyze_motion(self, flow_magnitudes, fps):
        """
        Convert optical flow to movement metrics
        
        Args:
            flow_magnitudes: List of average flow magnitudes per frame
            fps: Video frame rate
            
        Returns:
            tuple: (velocity_magnitude, acceleration, intensity)
        """
        if len(flow_magnitudes) < 2:
            return None, None, None
        
        flow_array = np.array(flow_magnitudes)
        
        # Normalize and scale to approximate velocity
        # Calibration factor (pixels to meters approximation)
        pixel_to_meter = 0.01  # Approximate conversion
        velocity_magnitude = flow_array * pixel_to_meter * fps
        
        # Smooth the signal
        window_size = min(5, len(velocity_magnitude) // 4)
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            velocity_magnitude = np.convolve(velocity_magnitude, kernel, mode='same')
        
        # Calculate acceleration
        dt = 1 / fps
        if len(velocity_magnitude) > 1:
            acceleration = np.gradient(velocity_magnitude) / dt
            # Convert to 2D array for compatibility
            acceleration = np.column_stack([acceleration * 0.7, acceleration * 0.3])
        else:
            acceleration = np.zeros((len(velocity_magnitude), 2))
        
        # Estimate intensity
        intensity = self.estimate_intensity_from_flow(velocity_magnitude)
        
        return velocity_magnitude, acceleration, intensity
    
    def estimate_intensity_from_flow(self, velocity_magnitude):
        """
        Estimate METs from optical flow-based velocity
        
        Args:
            velocity_magnitude: Array of velocity estimates
            
        Returns:
            float: Estimated METs value
        """
        mean_velocity = np.mean(velocity_magnitude)
        std_velocity = np.std(velocity_magnitude)
        max_velocity = np.max(velocity_magnitude)
        
        # Combined metric
        intensity_score = (mean_velocity * 0.5 + 
                          max_velocity * 0.3 + 
                          std_velocity * 0.2)
        
        # Map to METs scale (adjusted for optical flow)
        if intensity_score < 0.2:
            return 1.5  # Sedentary
        elif intensity_score < 0.5:
            return 2.5  # Light
        elif intensity_score < 0.8:
            return 4.0  # Light-moderate
        elif intensity_score < 1.2:
            return 5.5  # Moderate
        elif intensity_score < 1.6:
            return 7.0  # Moderate-vigorous
        elif intensity_score < 2.0:
            return 8.5  # Vigorous
        elif intensity_score < 2.5:
            return 10.0  # Very vigorous
        else:
            return 12.0  # Near maximal

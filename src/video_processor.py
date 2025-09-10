"""
Video processing module using optical flow for movement detection
Compatible with Streamlit Cloud - No MediaPipe dependencies
"""
import cv2
import numpy as np

class VideoProcessor:
    """
    Process video files using optical flow for movement analysis
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
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps == 0 or fps > 120:  # Sanity check
                fps = 30  # Default fallback
                
            # Read first frame
            ret, frame1 = cap.read()
            if not ret:
                return None, None, None
                
            # Resize frame for faster processing if too large
            height, width = frame1.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame1 = cv2.resize(frame1, (new_width, new_height))
                
            prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            
            frame_count = 0
            flow_magnitudes = []
            
            while True:
                ret, frame2 = cap.read()
                if not ret:
                    break
                    
                # Resize if needed
                if width > 640:
                    frame2 = cv2.resize(frame2, (new_width, new_height))
                    
                next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prvs, next_gray, None,
                    **self.flow_params
                )
                
                # Calculate magnitude of flow
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Average magnitude for this frame (filtering outliers)
                mag_flat = magnitude.flatten()
                # Remove top 5% outliers
                threshold = np.percentile(mag_flat, 95)
                mag_filtered = mag_flat[mag_flat < threshold]
                avg_magnitude = np.mean(mag_filtered) if len(mag_filtered) > 0 else 0
                
                flow_magnitudes.append(avg_magnitude)
                self.timestamps.append(frame_count / fps)
                
                prvs = next_gray
                frame_count += 1
                
                # Update progress bar
                if progress_bar is not None:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                    
            cap.release()
            
            # Convert to motion metrics
            if len(flow_magnitudes) > 0:
                return self.analyze_motion(flow_magnitudes, fps)
            else:
                return None, None, None
                
        except Exception as e:
            print(f"Error in video processing: {e}")
            return None, None, None
    
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
        
        # Remove zeros and outliers
        flow_array = flow_array[flow_array > 0]
        if len(flow_array) < 2:
            return None, None, None
            
        # Normalize and scale to approximate velocity
        # Calibration factor (pixels to meters approximation)
        # Assuming average human height ~1.7m occupies ~300 pixels
        pixel_to_meter = 1.7 / 300  # Approximate conversion
        velocity_magnitude = flow_array * pixel_to_meter * fps
        
        # Smooth the signal using moving average
        window_size = min(5, len(velocity_magnitude) // 4)
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            velocity_magnitude = np.convolve(velocity_magnitude, kernel, mode='same')
        
        # Cap unrealistic velocities (human sprint ~10 m/s)
        velocity_magnitude = np.clip(velocity_magnitude, 0, 10)
        
        # Calculate acceleration
        dt = 1 / fps
        if len(velocity_magnitude) > 1:
            acceleration = np.gradient(velocity_magnitude) / dt
            # Convert to 2D array for compatibility with calculator
            # Assume 70% horizontal, 30% vertical movement
            acceleration = np.column_stack([
                acceleration * 0.7,  # horizontal component
                np.abs(acceleration) * 0.3  # vertical component (always positive)
            ])
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
        if len(velocity_magnitude) == 0:
            return 1.0
            
        mean_velocity = np.mean(velocity_magnitude)
        std_velocity = np.std(velocity_magnitude)
        max_velocity = np.percentile(velocity_magnitude, 90)  # Use 90th percentile instead of max
        
        # Combined metric with adjusted weights
        intensity_score = (mean_velocity * 0.6 + 
                          max_velocity * 0.3 + 
                          std_velocity * 0.1)
        
        # Map to METs scale based on velocity ranges
        # Walking: 1.0-1.5 m/s (~3-4 METs)
        # Jogging: 2.0-3.0 m/s (~7-8 METs)  
        # Running: 3.0-5.0 m/s (~10-12 METs)
        
        if intensity_score < 0.3:
            return 1.5  # Sedentary/very light
        elif intensity_score < 0.8:
            return 2.5  # Light activity
        elif intensity_score < 1.2:
            return 4.0  # Moderate walking
        elif intensity_score < 1.8:
            return 6.0  # Brisk walking
        elif intensity_score < 2.5:
            return 8.0  # Jogging
        elif intensity_score < 3.5:
            return 10.0  # Running
        elif intensity_score < 4.5:
            return 12.0  # Fast running
        else:
            return 14.0  # Sprint

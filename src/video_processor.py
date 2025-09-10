"""
Video processing module for body pose detection and movement analysis
"""
import cv2
import mediapipe as mp
import numpy as np

class VideoProcessor:
    """
    Process video files to extract body movement data using MediaPipe
    """
    
    def __init__(self):
        """Initialize MediaPipe pose detection"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True
        )
        self.landmark_positions = []
        self.timestamps = []
        
    def process_video(self, video_path, progress_bar=None):
        """
        Process video file and extract pose landmarks
        
        Args:
            video_path: Path to video file
            progress_bar: Streamlit progress bar object (optional)
            
        Returns:
            tuple: (velocity_magnitude, acceleration, intensity)
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0:
            fps = 30  # Default fallback
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            
            # Process frame
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Extract 33 landmarks (x, y, z coordinates)
                landmarks = np.array([
                    [lm.x, lm.y, lm.z] 
                    for lm in results.pose_landmarks.landmark
                ])
                self.landmark_positions.append(landmarks)
                self.timestamps.append(frame_count / fps)
            
            frame_count += 1
            
            # Update progress bar if provided
            if progress_bar is not None:
                progress_bar.progress(min(frame_count / total_frames, 1.0))
        
        cap.release()
        
        # Analyze the collected movement data
        return self.analyze_movement(fps)
    
    def analyze_movement(self, fps):
        """
        Analyze collected pose data to calculate movement metrics
        
        Args:
            fps: Frames per second of the video
            
        Returns:
            tuple: (velocity_magnitude, acceleration, intensity)
        """
        if len(self.landmark_positions) < 2:
            return None, None, None
        
        positions = np.array(self.landmark_positions)
        
        # Calculate center of mass (using torso landmarks)
        # Using landmarks: shoulders (11,12), hips (23,24)
        torso_indices = [11, 12, 23, 24]
        com = np.mean(positions[:, torso_indices, :], axis=1)
        
        # Time step
        dt = 1 / fps
        
        # Calculate velocity (first derivative of position)
        velocity = np.diff(com, axis=0) / dt
        velocity_magnitude = np.linalg.norm(velocity, axis=1)
        
        # Calculate acceleration (second derivative of position)
        if len(velocity) > 1:
            acceleration = np.diff(velocity, axis=0) / dt
        else:
            acceleration = np.zeros_like(velocity)
        
        # Estimate activity intensity (METs)
        intensity = self.estimate_intensity(velocity_magnitude)
        
        return velocity_magnitude, acceleration, intensity
    
    def estimate_intensity(self, velocity_magnitude):
        """
        Estimate exercise intensity in METs from velocity data
        
        Args:
            velocity_magnitude: Array of velocity magnitudes
            
        Returns:
            float: Estimated METs value
            
        Reference:
            Ainsworth et al. (2011). Compendium of Physical Activities
        """
        mean_velocity = np.mean(velocity_magnitude)
        max_velocity = np.max(velocity_magnitude)
        velocity_std = np.std(velocity_magnitude)
        
        # Combined metric considering mean, max, and variability
        intensity_score = mean_velocity * 0.5 + max_velocity * 0.3 + velocity_std * 0.2
        
        # Map to METs scale
        if intensity_score < 0.3:
            return 1.5  # Sedentary
        elif intensity_score < 0.6:
            return 2.5  # Light
        elif intensity_score < 1.0:
            return 4.0  # Light-moderate
        elif intensity_score < 1.5:
            return 5.5  # Moderate
        elif intensity_score < 2.0:
            return 7.0  # Moderate-vigorous
        elif intensity_score < 2.5:
            return 8.5  # Vigorous
        elif intensity_score < 3.0:
            return 10.0  # Very vigorous
        else:
            return 12.0  # Near maximal
    
    def get_landmark_names(self):
        """
        Get MediaPipe pose landmark names for reference
        
        Returns:
            list: Names of all 33 pose landmarks
        """
        return [
            "nose", "left_eye_inner", "left_eye", "left_eye_outer",
            "right_eye_inner", "right_eye", "right_eye_outer",
            "left_ear", "right_ear", "mouth_left", "mouth_right",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_pinky", "right_pinky",
            "left_index", "right_index", "left_thumb", "right_thumb",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle", "left_heel", "right_heel",
            "left_foot_index", "right_foot_index"
        ]

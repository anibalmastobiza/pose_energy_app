"""
Main Streamlit application for biomechanical movement tracking
Compatible with Streamlit Cloud - Using Optical Flow instead of MediaPipe
"""
import streamlit as st
import tempfile
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.video_processor_optical import VideoProcessor
from src.biomechanical_calculator import BiomechanicalCalculator
from src.visualization import create_velocity_plot, create_energy_plot
import numpy as np

def main():
    st.set_page_config(
        page_title="Biomechanical Movement Tracker", 
        layout="wide",
        page_icon="🏃"
    )
    
    st.title("🏃 Biomechanical Movement Analysis")
    st.markdown("""
    Upload a video to track body movement and calculate energy expenditure (Joules) 
    and protein requirements based on biomechanically optimal formulas.
    
    **Note:** Using optical flow analysis for movement detection (compatible with all environments).
    """)
    
    # Sidebar for user parameters
    with st.sidebar:
        st.header("User Parameters")
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=1.0)
        age = st.number_input("Age", min_value=10, max_value=100, value=30)
        sex = st.selectbox("Biological Sex", ["male", "female"])
        
        st.markdown("---")
        st.caption("**Scientific References:**")
        st.caption("• Cunningham (1991) - RMR equation")
        st.caption("• Cavagna & Kaneko (1977) - Metabolic efficiency")
        st.caption("• ISSN Position Stand (2017) - Protein requirements")
        st.caption("• Moore et al. (2015) - MPS optimization")
        
        st.markdown("---")
        st.info("💡 **Tip:** Ensure the camera is stable and the subject's full body is visible for best results.")
    
    # Main content
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=['mp4', 'avi', 'mov'],
        help="Upload a video with clear view of full body movement. Max size: 200MB"
    )
    
    if uploaded_file is not None:
        # Check file size
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
        if file_size > 200:
            st.error("File size exceeds 200MB limit. Please upload a smaller video.")
            return
            
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            temp_path = tfile.name
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Processing Video")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize processors
            processor = VideoProcessor()
            calculator = BiomechanicalCalculator(weight, height, age, sex)
            
            # Process video
            status_text.text("Analyzing movement patterns...")
            try:
                velocity, acceleration, intensity = processor.process_video(
                    temp_path, progress_bar
                )
                
                if velocity is not None and len(velocity) > 0:
                    duration = len(processor.timestamps) if hasattr(processor, 'timestamps') else len(velocity) / 30
                    
                    # Calculate energy and protein
                    energy_j = calculator.calculate_energy_expenditure(
                        velocity, acceleration, duration
                    )
                    protein_g = calculator.estimate_protein_needs(energy_j, intensity)
                    
                    # Display results
                    status_text.empty()
                    st.success("✅ Analysis Complete!")
                    
                    st.markdown("### 📈 Results")
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric(
                            "Energy Expenditure", 
                            f"{energy_j:.0f} J", 
                            f"≈ {energy_j/4184:.1f} kcal",
                            help="Total metabolic energy based on mechanical work and efficiency"
                        )
                    with metrics_col2:
                        st.metric(
                            "Protein Needs", 
                            f"{protein_g:.1f} g",
                            f"≈ {protein_g/weight:.2f} g/kg",
                            help="Optimal protein for recovery based on activity intensity"
                        )
                    with metrics_col3:
                        st.metric(
                            "Activity Intensity", 
                            f"{intensity:.1f} METs",
                            help="Metabolic Equivalent of Task"
                        )
                    
                    # Additional metrics
                    st.markdown("### 📊 Detailed Metrics")
                    detail_col1, detail_col2 = st.columns(2)
                    with detail_col1:
                        st.info(f"**Average Velocity:** {np.mean(velocity):.2f} m/s")
                        st.info(f"**Peak Velocity:** {np.max(velocity):.2f} m/s")
                    with detail_col2:
                        st.info(f"**Duration:** {duration:.1f} seconds")
                        st.info(f"**Lean Body Mass:** {calculator.lean_mass:.1f} kg")
                else:
                    st.error("❌ Could not detect sufficient movement in video. Please ensure the subject is moving and the video quality is good.")
            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
                st.info("Please try with a different video or check the format.")
        
        with col2:
            if 'velocity' in locals() and velocity is not None and len(velocity) > 0:
                st.subheader("📉 Movement Analysis")
                
                # Create timestamps if not available
                if not hasattr(processor, 'timestamps'):
                    timestamps = np.linspace(0, duration, len(velocity))
                else:
                    timestamps = processor.timestamps[:len(velocity)]
                
                # Velocity plot
                fig_velocity = create_velocity_plot(timestamps, velocity)
                st.plotly_chart(fig_velocity, use_container_width=True)
                
                # Energy accumulation plot
                energy_cumulative = np.cumsum(np.abs(velocity)) * weight * 9.81 / 0.23
                fig_energy = create_energy_plot(timestamps, energy_cumulative)
                st.plotly_chart(fig_energy, use_container_width=True)
        
        # Clean up
        try:
            os.unlink(temp_path)
        except:
            pass
    
    # Technical documentation
    with st.expander("📚 Technical Documentation & Formulas"):
        st.markdown("""
        ### Movement Detection Method
        
        This application uses **Optical Flow Analysis** (Farneback method) to detect movement:
        - Tracks pixel displacement between consecutive frames
        - Estimates velocity from motion vectors
        - Robust to different lighting conditions
        
        ### Energy Calculation Method
        
        **Mechanical Work Components:**
        - Kinetic Energy: KE = ½mv²
        - Potential Energy: PE = mgh (vertical displacement)
        - Internal Work: ~10% of body weight × acceleration (Willems et al., 1995)
        
        **Total Energy Formula:**
        ```
        E_metabolic = (W_kinetic + W_potential + W_internal) / η + BMR_component
        where η = 0.23 (23% efficiency)
        ```
        
        ### Protein Synthesis Optimization
        
        **Activity-Based Requirements:**
        - Light (<3 METs): 0.8 g/kg body weight
        - Moderate (3-6 METs): 1.2 g/kg body weight  
        - Vigorous (6-9 METs): 1.6 g/kg body weight
        - Very Vigorous (>9 METs): 2.0 g/kg body weight
        
        **Minimum Effective Dose:** 20g (leucine threshold ~2.5g)
        
        ### Limitations & Assumptions
        - Motion detection via optical flow (not pose-specific)
        - Approximates center of mass movement
        - Individual metabolic variations not captured
        - Mechanical efficiency range: 20-25% (using mean 23%)
        
        ### References
        1. Cavagna GA, Kaneko M. (1977). J Physiol. 268(2):467-81
        2. Cunningham JJ. (1991). Am J Clin Nutr. 54(6):963-9
        3. Moore DR, et al. (2015). J Appl Physiol. 119(3):290-301
        4. Jäger R, et al. (2017). J Int Soc Sports Nutr. 14:20
        5. Willems PA, et al. (1995). J Exp Biol. 198:379-393
        """)

if __name__ == "__main__":
    main()

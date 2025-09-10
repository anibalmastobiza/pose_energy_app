
import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import tempfile
import math
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pose → Energy (kJ) + Protein (g)", layout="wide")

st.title("Video Pose Tracker → Energy (kJ) & Protein (g) Estimator")
st.caption("Upload a video, we track your body, estimate METs from movement, convert to energy (kJ), and estimate protein oxidized (g).")

with st.expander("Method (concise) & assumptions"):
    st.markdown("""
**Tracking:** MediaPipe Pose (33 landmarks) on sampled frames.

**Movement index (MI):** For each frame *t*, we compute landmark displacements from *t-1* (2D pixels normalized by torso size), then sum squared displacements across landmarks.

**Heuristic MI→MET mapping:** We robust‑normalize MI by median & MAD to get a z‑score \\(z\\). We set \\(\\text{MET} = \\mathrm{clip}(1 + 2z, 1, 12)\\). This is a practical proxy (requires no wearable) and should be **re‑calibrated** if you have ground‑truth (e.g., a known activity).

**Energy:** kcal/min = MET × 3.5 × (mass_kg) / 200 (ACSM). kJ = kcal × 4.184.  
**Protein:** We assume protein contributes **2–5%** of exercise energy (low–moderate intensity), up to ~10% in prolonged/high‑intensity or glycogen‑depleted states. We convert energy from protein to grams using **17 kJ/g** (FAO/WHO). You can override this fraction.
""")

st.sidebar.header("Inputs")
mass_kg = st.sidebar.number_input("Body mass (kg)", min_value=20.0, max_value=300.0, value=75.0, step=0.5)
sample_fps = st.sidebar.slider("Analysis FPS (processing rate)", 2, 30, 10)
user_protein_frac = st.sidebar.slider("Protein energy fraction (%)", 0, 15, 0)
st.sidebar.caption("Leave at 0% to use automatic rule: ≤3 MET: 2%, 3–6 MET: 3%, >6 MET: 5%.")

uploaded = st.file_uploader("Upload a video (mp4, mov, avi, mkv)", type=["mp4","mov","avi","mkv"])

if uploaded is not None:
    # Save to a temp file for OpenCV
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
    tfile.write(uploaded.read())
    tfile.flush()
    video_path = tfile.name

    st.video(uploaded)

    # Init MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / max(native_fps, 1.0)

    st.write(f"Native FPS: {native_fps:.2f} | Frames: {total_frames} | Duration: {duration_s:.1f}s")

    # Sampling stride
    stride = max(int(round(native_fps / sample_fps)), 1)

    prev_landmarks = None
    mi_series = []  # movement index per processed frame
    met_series = []
    time_series = []

    # Helper to compute torso size (pixel scale) to normalize motion
    def torso_size(landmarks):
        # Use distance between left shoulder and right hip as a proxy scale (or shoulder-hip avg)
        def dist(a,b):
            return math.hypot(landmarks[a][0]-landmarks[b][0], landmarks[a][1]-landmarks[b][1])
        # Fallback: shoulder distance
        l_sh, r_sh = 11, 12
        l_hip, r_hip = 23, 24
        d1 = dist(l_sh, r_hip)
        d2 = dist(r_sh, l_hip)
        d3 = dist(l_sh, r_sh)
        return np.median([d for d in [d1,d2,d3] if d>0]) or 1.0

    frame_idx = 0
    processed = 0

    # For optional annotated preview
    preview_frames = []
    preview_every = max(int(1 * sample_fps), 1)  # one preview frame per ~1s

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        lm = None
        if res.pose_landmarks:
            h, w, _ = frame.shape
            lm = [(int(p.x*w), int(p.y*h)) for p in res.pose_landmarks.landmark]

        if lm is not None and prev_landmarks is not None:
            # normalize by torso size
            scale = torso_size(lm)
            if scale <= 0:
                scale = 1.0
            # sum squared displacement across all available landmarks
            disp2 = 0.0
            count = 0
            for i in range(min(len(lm), len(prev_landmarks))):
                dx = (lm[i][0] - prev_landmarks[i][0]) / scale
                dy = (lm[i][1] - prev_landmarks[i][1]) / scale
                disp2 += dx*dx + dy*dy
                count += 1
            mi = disp2 / max(count,1)
            mi_series.append(mi)
            time_series.append(frame_idx / max(native_fps, 1.0))
        elif lm is not None and prev_landmarks is None:
            # first valid landmark set
            mi_series.append(0.0)
            time_series.append(frame_idx / max(native_fps, 1.0))

        prev_landmarks = lm
        processed += 1

        # Annotated preview (sparse)
        if res.pose_landmarks is not None and (processed % preview_every == 0):
            annotated = frame.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                annotated, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            preview_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        frame_idx += 1

    cap.release()
    pose.close()

    if len(mi_series) == 0:
        st.error("No pose detected. Try a clearer video (single person, good light, full body).")
        st.stop()

    mi = np.array(mi_series)
    # Robust z-score using median & MAD
    med = np.median(mi)
    mad = np.median(np.abs(mi - med)) or 1e-6
    z = (mi - med) / (1.4826 * mad)

    # Heuristic mapping to METs
    met = 1.0 + 2.0 * z
    met = np.clip(met, 1.0, 12.0)

    # Auto protein fraction if user not set
    mean_met = float(np.mean(met))
    if user_protein_frac > 0:
        protein_frac = user_protein_frac / 100.0
    else:
        if mean_met <= 3.0:
            protein_frac = 0.02
        elif mean_met <= 6.0:
            protein_frac = 0.03
        else:
            protein_frac = 0.05

    # Energy: kcal/min = MET * 3.5 * mass / 200 ; kJ = kcal * 4.184
    # We compute per-frame rate using the per-frame MET and frame duration.
    frame_dt = stride / max(native_fps, 1.0)
    kcal_per_min = met * 3.5 * mass_kg / 200.0
    kcal_per_sec = kcal_per_min / 60.0
    kcal_total = float(np.sum(kcal_per_sec * frame_dt))
    kJ_total = kcal_total * 4.184

    # Protein grams from fraction
    kJ_protein = kJ_total * protein_frac
    protein_g = kJ_protein / 17.0  # FAO/WHO

    # Export CSV
    df = pd.DataFrame({
        "time_s": time_series,
        "movement_index": mi,
        "z_score": z,
        "MET": met,
        "kcal_per_min": kcal_per_min
    })
    csv_path = "per_frame_metrics.csv"
    df.to_csv(csv_path, index=False)

    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Duration (s)", f"{duration_s:.1f}")
    c2.metric("Mean MET", f"{mean_met:.2f}")
    c3.metric("Protein fraction", f"{protein_frac*100:.1f}%")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total energy (kJ)", f"{kJ_total:.1f}")
    c2.metric("Total energy (kcal)", f"{kcal_total:.1f}")
    c3.metric("Estimated protein oxidized (g)", f"{protein_g:.2f}")

    st.download_button("Download per‑frame CSV", data=df.to_csv(index=False), file_name="per_frame_metrics.csv", mime="text/csv")

    # Plot MET over time
    st.subheader("METs over time")
    fig = plt.figure()
    plt.plot(time_series, met)
    plt.xlabel("Time (s)")
    plt.ylabel("Estimated MET")
    plt.title("Estimated METs from pose‑based movement")
    st.pyplot(fig)

    # Preview annotated frames (carousel)
    if len(preview_frames) > 0:
        st.subheader("Annotated pose preview")
        st.image(preview_frames, caption=None, use_column_width=True)

else:
    st.info("Upload a video to begin. For best results: single person, whole body visible, good lighting, stable camera.")

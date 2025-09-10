
import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import tempfile, os, math
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pose → Energy (kJ) + Protein (g)", layout="wide")
st.title("Video Pose Tracker → Energy (kJ) & Protein (g) (MediaPipe)")
st.caption("Requiere Python 3.10 en Streamlit Cloud.")

with st.expander("Método & supuestos (breve)"):
    st.markdown("""
**Pose:** MediaPipe Pose (33 landmarks).  
**Índice de movimiento (MI):** desplazamiento medio cuadrático de landmarks normalizado por torso.  
**MI→MET:** z-score robusto (mediana/MAD): `MET = clip(1 + 2·z, 1, 12)`.  
**Energía:** kcal/min = MET × 3.5 × masa_kg / 200 (ACSM); kJ = kcal × 4.184.  
**Proteína:** 2–5% (hasta ~10%) de la energía; conversión 17 kJ/g.
""")

st.sidebar.header("Parámetros")
mass_kg = st.sidebar.number_input("Masa corporal (kg)", 20.0, 300.0, 75.0, 0.5)
sample_fps = st.sidebar.slider("FPS de análisis", 2, 30, 10)
user_protein_frac = st.sidebar.slider("Fracción proteína (%)", 0, 15, 0)
st.sidebar.caption("0% = automático: ≤3 MET: 2%, 3–6: 3%, >6: 5%.")

uploaded = st.file_uploader("Sube un vídeo (mp4, mov, avi, mkv)", type=["mp4","mov","avi","mkv"])

mp_pose = mp.solutions.pose

def torso_size(lm_xy):
    # lm_xy: list[(x,y)] en píxeles
    def dist(a,b): return math.hypot(lm_xy[a][0]-lm_xy[b][0], lm_xy[a][1]-lm_xy[b][1])
    L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24
    vals = []
    for a,b in [(L_SH,R_SH),(L_SH,R_HIP),(R_SH,L_HIP)]:
        try:
            d = dist(a,b)
            if d>0: vals.append(d)
        except Exception:
            pass
    return np.median(vals) if vals else 1.0

if uploaded:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
    tfile.write(uploaded.read()); tfile.flush()
    video_path = tfile.name
    st.video(uploaded)

    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / max(native_fps, 1.0)
    st.write(f"FPS nativo: {native_fps:.2f} | Frames: {total_frames} | Duración: {duration_s:.1f}s")

    stride = max(int(round(native_fps / sample_fps)), 1)

    prev_xy = None
    mi_series, time_series = [], []
    preview_frames = []
    preview_every = max(int(sample_fps), 1)
    frame_idx, processed = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % stride != 0:
            frame_idx += 1; continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            h,w,_ = frame.shape
            xy = [(p.x*w, p.y*h) for p in res.pose_landmarks.landmark]

            if prev_xy is not None:
                scale = max(torso_size(xy), 1.0)
                disp2, cnt = 0.0, 0
                for i in range(min(len(xy), len(prev_xy))):
                    dx = (xy[i][0]-prev_xy[i][0])/scale
                    dy = (xy[i][1]-prev_xy[i][1])/scale
                    disp2 += dx*dx + dy*dy; cnt += 1
                mi_series.append(disp2/max(cnt,1))
            else:
                mi_series.append(0.0)

            time_series.append(frame_idx/max(native_fps,1.0))
            prev_xy = xy

            if processed % preview_every == 0:
                annotated = frame.copy()
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated, res.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
                preview_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            processed += 1

        frame_idx += 1

    cap.release(); pose.close()

    if not mi_series:
        st.error("No se detectó la pose. Prueba con un vídeo claro y cuerpo completo en cuadro."); st.stop()

    mi = np.array(mi_series)
    med = float(np.median(mi)); mad = float(np.median(np.abs(mi-med))) or 1e-6
    z = (mi-med)/(1.4826*mad)
    met = np.clip(1.0 + 2.0*z, 1.0, 12.0)
    mean_met = float(np.mean(met))

    protein_frac = user_protein_frac/100.0 if user_protein_frac>0 else (0.02 if mean_met<=3 else (0.03 if mean_met<=6 else 0.05))

    frame_dt = stride/max(native_fps,1.0)
    kcal_per_min = met*3.5*mass_kg/200.0
    kcal_total = float(np.sum((kcal_per_min/60.0)*frame_dt))
    kJ_total = kcal_total*4.184
    protein_g = (kJ_total*protein_frac)/17.0

    st.subheader("Resultados")
    c1,c2,c3 = st.columns(3)
    c1.metric("Energía total (kJ)", f"{kJ_total:.1f}")
    c2.metric("Energía total (kcal)", f"{kcal_total:.1f}")
    c3.metric("Proteína oxidada (g)", f"{protein_g:.2f}")

    st.subheader("MET estimado en el tiempo")
    fig = plt.figure()
    plt.plot(time_series, met)
    plt.xlabel("Tiempo (s)"); plt.ylabel("MET estimado"); plt.title("MET vs tiempo (MediaPipe)")
    st.pyplot(fig)

    df = pd.DataFrame({"time_s":time_series, "movement_index":mi, "z_score":z, "MET":met, "kcal_per_min":kcal_per_min})
    st.download_button("Descargar CSV", df.to_csv(index=False), file_name="per_frame_metrics.csv")

    if preview_frames:
        st.subheader("Previsualización anotada")
        st.image(preview_frames, use_column_width=True)

else:
    st.info("Sube un vídeo para comenzar.")

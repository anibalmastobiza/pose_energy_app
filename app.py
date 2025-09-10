
import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile, os
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pose → Energy (kJ) + Protein (g) — YOLOv8", layout="wide")
st.title("Video Pose Tracker → Energy (kJ) & Protein (g) (YOLOv8 Pose)")
st.caption("Compatible con Python 3.13 (sin MediaPipe). Puede tardar más en instalar dependencias.")

with st.expander("Método & supuestos"):
    st.markdown("""
**Pose:** YOLOv8n‑pose (17 keypoints).  
**Movimiento:** desplazamiento medio cuadrático de keypoints normalizado por torso.  
**MI→MET:** `MET = clip(1 + 2·z, 1, 12)`.  
**Energía:** kcal/min = MET × 3.5 × masa_kg / 200 (ACSM); kJ = kcal × 4.184.  
**Proteína:** 2–5% (hasta 10%); 17 kJ/g.
""")

st.sidebar.header("Parámetros")
mass_kg = st.sidebar.number_input("Masa (kg)", 20.0, 300.0, 75.0, 0.5)
sample_fps = st.sidebar.slider("FPS de análisis", 2, 30, 10)
user_protein_frac = st.sidebar.slider("Fracción proteína (%)", 0, 15, 0)

uploaded = st.file_uploader("Sube un vídeo", type=["mp4","mov","avi","mkv"])

def torso_scale_from_kps_xy(xy):
    idxs = [5,6,11,12]
    pts = [xy[i] for i in idxs if i < len(xy)]
    if len(pts) < 2: return 1.0
    return max(np.linalg.norm(pts[0]-pts[-1]), 1.0)

if uploaded:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
    tfile.write(uploaded.read()); tfile.flush()
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total/max(fps,1.0)
    st.write(f"FPS: {fps:.1f} | Frames: {total} | Duración: {dur:.1f}s")

    stride = max(int(round(fps / sample_fps)), 1)
    model = YOLO("yolov8n-pose.pt")

    prev = None; mi=[]; t=[]; idx=0
    preview=[]; preview_every = max(int(sample_fps),1)

    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % stride != 0:
            idx += 1; continue
        h,w=frame.shape[:2]
        res = model(frame, verbose=False)[0]
        xy = None
        if res.keypoints is not None and len(res.keypoints)>0:
            xyn = res.keypoints.xyn[0].cpu().numpy()  # (17,2) [0,1]
            xy = xyn * np.array([w,h], dtype=np.float32)
        if xy is not None:
            if prev is not None:
                scale = torso_scale_from_kps_xy(xy) or 1.0
                disp = np.mean(((xy - prev)/scale)**2)
                mi.append(disp); t.append(idx/max(fps,1.0))
            else:
                mi.append(0.0); t.append(idx/max(fps,1.0))
            prev = xy
            if len(preview) < 12 and idx % (preview_every*stride) == 0:
                annotated = res.plot()
                import cv2 as _cv2
                preview.append(_cv2.cvtColor(annotated, _cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()

    if len(mi)==0:
        st.error("No se detectó la pose. Prueba con mejor iluminación y cuerpo completo en el encuadre.")
        st.stop()

    mi=np.array(mi); med=np.median(mi); mad=np.median(np.abs(mi-med)) or 1e-6
    z=(mi-med)/(1.4826*mad)
    met=np.clip(1+2*z,1,12); mean_met=float(np.mean(met))

    prot_frac = user_protein_frac/100 if user_protein_frac>0 else (0.02 if mean_met<=3 else 0.03 if mean_met<=6 else 0.05)

    kcal_min = met*3.5*mass_kg/200.0
    frame_dt = stride/max(fps,1.0)
    kcal_total = float(np.sum(kcal_min/60.0 * frame_dt))
    kJ_total = kcal_total*4.184
    protein_g = (kJ_total*prot_frac)/17.0

    st.subheader("Resultados")
    c1,c2,c3 = st.columns(3)
    c1.metric("Energía total (kJ)", f"{kJ_total:.1f}")
    c2.metric("Energía total (kcal)", f"{kcal_total:.1f}")
    c3.metric("Proteína oxidada (g)", f"{protein_g:.2f}")

    st.subheader("MET estimado")
    st.line_chart({"MET": met}, x=t)

    df = pd.DataFrame({"time_s":t,"movement_index":mi,"MET":met,"kcal_per_min":kcal_min})
    st.download_button("Descargar CSV", df.to_csv(index=False), "per_frame_metrics.csv")

    if preview:
        st.subheader("Previsualización anotada")
        st.image(preview, use_column_width=True)

else:
    st.info("Sube un vídeo para comenzar.")

import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile, os, math
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pose → Energy (kJ) + Protein (g) — YOLOv8", layout="wide")
st.title("Video Pose Tracker → Energy (kJ) & Protein (g) (YOLOv8 Pose)")
st.caption("Compatible con Python 3.13 en Streamlit Cloud (no usa MediaPipe).")

with st.expander("Method & assumptions"):
    st.markdown("""
**Pose:** YOLOv8n-pose (17 keypoints).  
**Movement index:** desplazamiento medio cuadrático de keypoints normalizado por el torso.  
**Heurístico MET:** `MET = clip(1 + 2z, 1, 12)` con z-score robusto.  
**Energía:** kcal/min = MET × 3.5 × (peso kg) / 200 (ACSM).  
**Proteína:** 2–5% (hasta 10%) de la energía, convertido con 17 kJ/g (FAO/WHO).
""")

st.sidebar.header("Inputs")
mass_kg = st.sidebar.number_input("Body mass (kg)", 20.0, 300.0, 75.0, 0.5)
sample_fps = st.sidebar.slider("Analysis FPS", 2, 30, 10)
user_protein_frac = st.sidebar.slider("Protein energy fraction (%)", 0, 15, 0)

uploaded = st.file_uploader("Upload a video", type=["mp4","mov","avi","mkv"])

def torso_scale(xy):
    idxs = [5,6,11,12]  # shoulders & hips
    pts = [xy[i] for i in idxs if i < len(xy)]
    if len(pts) < 2: return 1.0
    return max(np.linalg.norm(pts[0]-pts[-1]), 1.0)

if uploaded:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
    tfile.write(uploaded.read()); tfile.flush()
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps
    st.write(f"FPS: {fps:.1f} | Frames: {total} | Duration: {dur:.1f}s")

    stride = max(int(round(fps / sample_fps)), 1)
    model = YOLO("yolov8n-pose.pt")

    prev = None; mi=[]; t=[]
    idx=0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % stride==0:
            h,w=frame.shape[:2]
            res = model(frame, verbose=False)[0]
            if res.keypoints is not None and len(res.keypoints)>0:
                xy = res.keypoints.xyn[0].cpu().numpy()*[w,h]
                if prev is not None:
                    scale=torso_scale(xy)
                    disp=np.mean(((xy-prev)/scale)**2)
                    mi.append(disp); t.append(idx/fps)
                else:
                    mi.append(0.0); t.append(idx/fps)
                prev=xy
        idx+=1
    cap.release()

    mi=np.array(mi); med=np.median(mi); mad=np.median(np.abs(mi-med)) or 1e-6
    z=(mi-med)/(1.4826*mad); met=np.clip(1+2*z,1,12); mean_met=np.mean(met)
    prot_frac = user_protein_frac/100 if user_protein_frac>0 else (0.02 if mean_met<=3 else 0.03 if mean_met<=6 else 0.05)

    kcal_min=met*3.5*mass_kg/200; kcal_total=np.sum(kcal_min/60*stride/fps)
    kJ_total=kcal_total*4.184; prot_g=kJ_total*prot_frac/17

    st.metric("Total energy (kJ)", f"{kJ_total:.1f}")
    st.metric("Protein oxidized (g)", f"{prot_g:.2f}")

    st.line_chart({"MET": met}, x=t)
    df=pd.DataFrame({"time_s":t,"movement_index":mi,"MET":met})
    st.download_button("Download CSV", df.to_csv(index=False), "metrics.csv")
else:
    st.info("Upload a video to begin.")


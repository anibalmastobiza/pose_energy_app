# Pose → Energy (kJ) & Protein (g) Estimator

A minimal Streamlit app that:
- Tracks human pose in an uploaded video (MediaPipe Pose).
- Estimates movement intensity → METs via a robust, calibration‑friendly heuristic.
- Computes energy expenditure from METs and body mass.
- Estimates grams of protein oxidized from the exercise energy (configurable).

## How it works (formulas & sources)

- **Pose tracking:** MediaPipe Pose (33 landmarks). See Google AI Edge Pose Landmarker docs.
- **Movement index:** Per‑frame, we sum squared landmark displacements normalized by torso size.
- **MI → MET mapping:** Robust z‑score (median/MAD) on MI, then `MET = clip(1 + 2*z, 1, 12)`. This is a practical proxy. Re‑calibrate if you have ground‑truth.
- **Energy from METs:** `kcal/min = MET × 3.5 × body_mass_kg / 200` (ACSM convention). Convert to kJ with `kJ = kcal × 4.184`.
- **Protein oxidation:** Literature suggests ~2–5% of exercise energy (up to ~10% in prolonged/high‑intensity or glycogen‑depleted states) can come from protein. Convert energy‑from‑protein to grams using `17 kJ/g` (FAO/WHO Atwater general factors).

### References
- Ainsworth et al., **Compendium of Physical Activities**; definition of MET = 3.5 ml O₂·kg⁻¹·min⁻¹.
- ACSM metabolic calculation convention: `kcal/min = MET × 3.5 × kg / 200`.
- FAO/WHO energy conversion factors: Protein = 4 kcal/g ≈ 17 kJ/g.
- Reviews on substrate contribution during endurance exercise (protein typically minor, ~2–5%, up to ~10%).

## Quickstart (local)

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run
streamlit run app.py
```

Open the local URL printed by Streamlit, upload a video (single person, whole body visible), set your body mass, and read results.

## Notes & Limitations

- The MI→MET mapping is **heuristic** and video‑dependent (camera motion, occlusion). For research use, calibrate against known activities or wearable‑derived METs.
- Protein oxidation varies with duration, intensity, training status, and glycogen availability. Treat the estimate as an **order‑of‑magnitude** guide, not a diagnostic metric.
- No personal data leave your machine; everything runs locally.
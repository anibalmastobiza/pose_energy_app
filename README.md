# Biomechanical Movement Tracker 🏃

A Streamlit application for analyzing human movement from video, calculating energy expenditure in Joules and optimal protein requirements using scientifically validated biomechanical formulas.

## Features

- **Pose Detection**: Real-time body tracking using MediaPipe (33 landmarks)
- **Energy Calculation**: Biomechanically accurate energy expenditure in Joules
- **Protein Optimization**: Evidence-based protein recommendations for recovery
- **Visual Analytics**: Interactive plots for velocity and energy profiles
- **METs Classification**: Automatic activity intensity assessment

## Scientific Foundation

### Energy Expenditure Formula
```
E_metabolic = (W_kinetic + W_potential + W_internal) / η + BMR
```
- η = 0.23 (23% metabolic efficiency - Cavagna & Kaneko, 1977)
- BMR via Cunningham equation (1991)

### Protein Requirements
- Based on ISSN Position Stand (2017)
- Scaled by activity intensity (0.8-2.0 g/kg)
- Minimum effective dose: 20g (Moore et al., 2015)

## Installation

### Local Setup
```bash
# Clone repository
git clone https://github.com/yourusername/biomechanical-tracker.git
cd biomechanical-tracker

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy with:
   - Repository: `yourusername/biomechanical-tracker`
   - Branch: `main`
   - Main file path: `app.py`

## Project Structure
```
biomechanical-tracker/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation
├── .gitignore                     # Git ignore file
├── src/
│   ├── __init__.py               # Package initializer
│   ├── biomechanical_calculator.py # Energy & protein calculations
│   ├── video_processor.py         # MediaPipe pose detection
│   └── visualization.py           # Plotly visualizations
└── config/
    └── settings.yaml              # Configuration parameters
```

## Usage

1. **Upload Video**: Support for MP4, AVI, MOV formats
2. **Set Parameters**: Enter weight, height, age, and biological sex
3. **Analysis**: Automatic pose detection and biomechanical calculations
4. **Results**: 
   - Energy expenditure (Joules & kcal)
   - Protein requirements (grams)
   - Activity intensity (METs)
   - Velocity and energy plots

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- Streamlit
- NumPy
- Plotly

## Scientific References

1. **Cavagna GA, Kaneko M** (1977). Mechanical work and efficiency in level walking and running. *J Physiol*. 268(2):467-81.

2. **Cunningham JJ** (1991). Body composition as a determinant of energy expenditure. *Am J Clin Nutr*. 54(6):963-9.

3. **Jäger R, et al.** (2017). International Society of Sports Nutrition Position Stand: protein and exercise. *J Int Soc Sports Nutr*. 14:20.

4. **Moore DR, et al.** (2015). Protein ingestion to stimulate myofibrillar protein synthesis requires greater relative protein intakes in healthy older versus younger men. *J Appl Physiol*. 119(3):290-301.

5. **Willems PA, et al.** (1995). External, internal and total work in human locomotion. *J Exp Biol*. 198:379-393.

## Limitations

- 2D pose estimation (depth approximated)
- Assumes uniform body segment density
- Individual metabolic variations not captured
- Requires clear full-body visibility in video

## Contributing

Pull requests welcome. For major changes, please open an issue first.

## License

MIT License - See LICENSE file for details

## Author

Created for biomechanical research applications in sports science and rehabilitation.

## Acknowledgments

- MediaPipe team for pose detection models
- Streamlit for web framework
- Scientific community for foundational research

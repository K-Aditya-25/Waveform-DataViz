# Power Quality Disturbances Visualization

This Streamlit application visualizes various Power Quality Disturbances (PQDs) using interactive Plotly charts. It implements mathematical models for 12 different types of power quality disturbances as presented in the reference table.

## Features

- Visualization of 12 different types of Power Quality Disturbances (PQDs)
- Interactive plots using Plotly
- Three viewing modes:
  - Individual PQD visualization with multiple analysis methods:
    - Raw Waveform
    - Frequency Spectrum (FFT)
    - Wavelet Transform
    - Spectrogram
  - Compare two PQDs side-by-side
  - Overview of all PQDs in a grid layout
- Mathematical models displayed in LaTeX format for each disturbance type
- Advanced signal processing capabilities (FFT, Wavelet Transform, Spectrogram)
- Explanatory content about each type of disturbance

## Power Quality Disturbances Included

1. **S1: Normal** - Ideal sinusoidal waveform with no disturbances
2. **S2: Sag/Dip** - Temporary reduction in voltage magnitude
3. **S3: Swell** - Temporary increase in voltage magnitude
4. **S4: Interruption** - Nearly complete loss of voltage for a short period
5. **S5: Harmonics** - Presence of signals at frequencies that are multiples of the fundamental frequency
6. **S6: Flicker** - Rapid, repetitive voltage variations
7. **S7: Oscillatory Transient** - Temporary oscillations in voltage or current
8. **S8: Spike** - Brief high-energy impulse superimposed on the waveform
9. **S9: Harmonics + Sag** - Harmonic distortion combined with voltage sag
10. **S10: Harmonics + Swell** - Harmonic distortion combined with voltage swell
11. **S11: Harmonics + Interruption** - Harmonic distortion combined with interruption
12. **S12: Harmonics + Flicker** - Harmonic distortion combined with flicker

## Installation

This project uses a Python virtual environment managed by `uv` as specified in the project requirements.

1. Clone the repository
2. Create a virtual environment:
```bash
uv venv .venv
```

3. Activate the virtual environment:
```bash
source .venv/bin/activate
```

4. Install the required packages:
```bash
uv pip install -r requirements.txt
```

## Advanced Signal Processing

This application includes advanced signal processing capabilities for analyzing Power Quality Disturbances:

1. **FFT (Fast Fourier Transform)**: Visualize the frequency components of each disturbance
2. **Wavelet Transform**: Time-frequency representation to identify when specific frequency components occur
3. **Spectrogram**: Visual representation of how frequency content changes over time

These capabilities help in better understanding the characteristics of different power quality disturbances and their impacts.

## Running the Application

After installing the dependencies, run the Streamlit app:

```bash
streamlit run app.py
```

The application will be accessible at http://localhost:8501 in your web browser.

## Project Structure

- `app.py`: Main Streamlit application
- `pqd_models.py`: Implementation of Power Quality Disturbance mathematical models
- `requirements.txt`: Required Python packages
- `README.md`: This file

## Dependencies

- streamlit==1.32.2
- plotly==5.18.0
- numpy==1.26.4
- pandas==2.2.1

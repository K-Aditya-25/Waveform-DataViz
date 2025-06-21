import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from pqd_models import PQDModels
import scipy.signal as signal
from scipy.fft import fft, fftfreq
import pywt

# Set page configuration
st.set_page_config(
    page_title="Power Quality Disturbances Visualization",
    page_icon="⚡",
    layout="wide",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-title {
        font-size: 1.5rem;
        font-weight: 500;
        text-align: center;
        margin-bottom: 2rem;
    }
    .description-text {
        font-size: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title and subtitle
st.markdown('<div class="main-title">Power Quality Disturbances (PQDs) Visualization</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Interactive visualization of various power quality disturbances using Plotly</div>', unsafe_allow_html=True)

# Mathematical models in LaTeX format for each PQD type
math_models = {
    'S1': r"f(t) = A\sin(2\pi f_0 t), A = 1, f_0 = 50Hz",
    'S2': r"f(t) = A[1 - h(u(t - t_1) - u(t - t_2))]\sin(2\pi f_0 t), 0.1 < h < 0.9, T < t_2 - t_1 < 9T",
    'S3': r"f(t) = A[1 + h(u(t - t_1) - u(t - t_2))]\sin(2\pi f_0 t), 0.1 < h < 0.8, T < t_2 - t_1 < 9T",
    'S4': r"f(t) = A[1 - h(u(t - t_1) - u(t - t_2))]\sin(2\pi f_0 t), 0.9 < h < 1, T < t_2 - t_1 < 9T",
    'S5': r"f(t) = A\sin(2\pi f_0 t) + \sum_{k=2}^{K} A_k\sin(2\pi kf_0 t + \phi_k), 0.05 < A_k < 0.15",
    'S6': r"f(t) = A[1 + \alpha\sin(2\pi f_r t)]\sin(2\pi f_0 t), 0.1 < \alpha < 0.2, 5 < f_r < 25",
    'S7': r"f(t) = A\sin(2\pi f_0 t) + \alpha[u(t - t_1) - u(t - t_2)]\sin(2\pi f_c(t-t_1))e^{-\frac{t-t_1}{\tau}}, 0.1 < \alpha < 0.8, t_1 < t < 0.05T, 300 < f_c < 900, 0.5T < \tau < 3T",
    'S8': r"f(t) = A\sin(2\pi f_0 t) + \alpha\sin(2\pi f_0 t)\sum_{i=0}^{n-1} [h(t-(t_1+0.02i)) - h(t-(t_2+0.02i))], 0.1 < \alpha < 0.4, 0.01T < t_2 - t_1 < 0.05T",
    'S9': r"f(t) = A[1 - h(u(t - t_1) - u(t - t_2))][A\sin(2\pi f_0 t) + \sum_{k=2}^{K} A_k\sin(2\pi kf_0 t + \phi_k)], 0.1 < h < 0.9, t_1 < t_2 < 9T, 0.05 < A_k < 0.15",
    'S10': r"f(t) = A[1 + h(u(t - t_1) - u(t - t_2))][A\sin(2\pi f_0 t) + \sum_{k=2}^{K} A_k\sin(2\pi kf_0 t + \phi_k)], 0.1 < h < 0.8, t_1 < t_2 < 9T, 0.05 < A_k < 0.15",
    'S11': r"f(t) = A[1 - h(u(t - t_1) - u(t - t_2))][A\sin(2\pi f_0 t) + \sum_{k=2}^{K} A_k\sin(2\pi kf_0 t + \phi_k)], 0.9 < h < 1, t_1 < t_2 < 9T, 0.05 < A_k < 0.15",
    'S12': r"f(t) = A[1 + \alpha\sin(2\pi f_r t)][A\sin(2\pi f_0 t) + \sum_{k=2}^{K} A_k\sin(2\pi kf_0 t + \phi_k)], 0.1 < \alpha < 0.2, 5 < f_r < 25, 0.05 < A_k < 0.15"
}

# Create PQD models instance
pqd = PQDModels()
signals = pqd.get_all_signals()

# App description
st.markdown("""
### About this application
This application visualizes different types of Power Quality Disturbances (PQDs) that can occur in electrical power systems.
Each visualization is based on the mathematical models described in the reference table.
""")

# Visualization modes
view_mode = st.radio("Select View Mode", ["Individual PQD", "Compare PQDs", "View All PQDs"], horizontal=True)

if view_mode == "Individual PQD":
    # Individual PQD visualization
    st.header("Individual PQD Visualization")
    
    # Signal selection
    signal_key = st.selectbox(
        "Select Power Quality Disturbance Type:",
        list(signals.keys()),
        format_func=lambda x: f"{x}: {signals[x]['title']}"
    )
    
    # Get selected signal
    selected_signal = signals[signal_key]
    
    # Show description
    st.markdown(f"**{selected_signal['title']}**: {selected_signal['description']}")
    
    # Visualization type selection
    viz_type = st.radio(
        "Select Visualization Method:",
        ["Raw Waveform", "Frequency Spectrum (FFT)", "Wavelet Transform", "Spectrogram"],
        horizontal=True
    )
    
    # Get signal data
    time = selected_signal['time']
    signal_data = selected_signal['signal']
    fs = len(time) / (time[-1] - time[0])  # Sampling frequency
    
    if viz_type == "Raw Waveform":
        # Create interactive plot for raw waveform
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=time,
                y=signal_data,
                mode='lines',
                name=selected_signal['title'],
                line=dict(color='royalblue', width=2)
            )
        )
        
        fig.update_layout(
            title=f"{signal_key}: {selected_signal['title']} - Raw Waveform",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude (pu)",
            height=500,
            hovermode='closest',
            template="plotly_white",
        )
        
    elif viz_type == "Frequency Spectrum (FFT)":
        # Compute FFT
        n = len(signal_data)
        fft_vals = np.abs(fft(signal_data)[:n//2]) / n * 2
        freqs = fftfreq(n, 1/fs)[:n//2]
        
        # Create interactive plot for FFT
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=fft_vals,
                mode='lines',
                name="FFT",
                line=dict(color='green', width=2)
            )
        )
        
        fig.update_layout(
            title=f"{signal_key}: {selected_signal['title']} - Frequency Spectrum",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Amplitude",
            height=500,
            hovermode='closest',
            template="plotly_white",
            xaxis=dict(range=[0, 1000])  # Limit frequency display range
        )
        
    elif viz_type == "Wavelet Transform":
        # Compute Continuous Wavelet Transform
        scales = np.arange(1, 128)
        coeffs, frequencies = pywt.cwt(signal_data, scales, 'morl', 1/fs)
        power = np.abs(coeffs)**2
        
        # Create a heatmap for the wavelet transform
        fig = go.Figure(data=go.Heatmap(
            z=power,
            x=time,
            y=frequencies,
            colorscale='Viridis',
            colorbar=dict(title='Power')
        ))
        
        fig.update_layout(
            title=f"{signal_key}: {selected_signal['title']} - Wavelet Transform",
            xaxis_title="Time (seconds)",
            yaxis_title="Frequency (Hz)",
            height=500,
            template="plotly_white",
            yaxis=dict(
                type='log',
                autorange=True
            )
        )
        
    else:  # Spectrogram
        # Compute Spectrogram
        f, t, Sxx = signal.spectrogram(signal_data, fs)
        
        # Create a heatmap for the spectrogram
        fig = go.Figure(data=go.Heatmap(
            z=10 * np.log10(Sxx + 1e-10),  # Convert to dB scale
            x=t,
            y=f,
            colorscale='Jet',
            colorbar=dict(title='Power/frequency (dB/Hz)')
        ))
        
        fig.update_layout(
            title=f"{signal_key}: {selected_signal['title']} - Spectrogram",
            xaxis_title="Time (seconds)",
            yaxis_title="Frequency (Hz)",
            height=500,
            template="plotly_white",
        )
    
    # Display the selected visualization
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation for each visualization type
    if viz_type == "Raw Waveform":
        st.info("**Raw Waveform**: Shows the signal amplitude over time, representing the basic time-domain view of the power quality disturbance.")
    elif viz_type == "Frequency Spectrum (FFT)":
        st.info("**Frequency Spectrum (FFT)**: Shows the frequency components present in the signal. Helps identify harmonic content and dominant frequencies.")
    elif viz_type == "Wavelet Transform":
        st.info("**Wavelet Transform**: Provides time-frequency representation that can identify both when and at what frequencies disturbances occur. Especially useful for transient events.")
    else:  # Spectrogram
        st.info("**Spectrogram**: Shows how the frequency content changes over time, with color representing power. Useful for identifying time-varying frequency components.")
    
    # Show mathematical model with LaTeX formula
    with st.expander("Show Mathematical Model"):
        st.markdown(f"### Mathematical Model for {selected_signal['title']}")
        st.markdown(f"The mathematical representation for **{selected_signal['title']}** is:")
        st.latex(math_models[signal_key])
        st.markdown("""
        Where:
        - A: Signal amplitude (typically 1.0 pu)
        - f₀: Fundamental frequency (50 Hz)
        - t: Time in seconds
        - h: Magnitude parameter (for sag, swell, interruption)
        - u(t): Unit step function
        - t₁, t₂: Event start and end times
        - A_k: Amplitude of harmonic components
        - φ_k: Phase angles of harmonic components
        - α: Modulation index (for flicker)
        - f_r: Modulation frequency (for flicker)
        - T: Period of fundamental component (1/f₀)
        """)

elif view_mode == "Compare PQDs":
    # Compare PQDs visualization
    st.header("Compare PQDs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        signal_key1 = st.selectbox(
            "Select First PQD:",
            list(signals.keys()),
            format_func=lambda x: f"{x}: {signals[x]['title']}"
        )
    
    with col2:
        signal_key2 = st.selectbox(
            "Select Second PQD:",
            list(signals.keys()),
            format_func=lambda x: f"{x}: {signals[x]['title']}",
            index=1
        )
    
    # Get selected signals
    selected_signal1 = signals[signal_key1]
    selected_signal2 = signals[signal_key2]
    
    # Create subplot with two rows
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=[f"{signal_key1}: {selected_signal1['title']}", 
                                      f"{signal_key2}: {selected_signal2['title']}"],
                       shared_xaxes=True,
                       vertical_spacing=0.1)
    
    # Add first signal
    fig.add_trace(
        go.Scatter(
            x=selected_signal1['time'],
            y=selected_signal1['signal'],
            mode='lines',
            name=selected_signal1['title'],
            line=dict(color='royalblue', width=2)
        ),
        row=1, col=1
    )
    
    # Add second signal
    fig.add_trace(
        go.Scatter(
            x=selected_signal2['time'],
            y=selected_signal2['signal'],
            mode='lines',
            name=selected_signal2['title'],
            line=dict(color='firebrick', width=2)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title="PQD Comparison",
        height=700,
        hovermode='closest',
        template="plotly_white",
    )
    
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude (pu)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (pu)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show descriptions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{selected_signal1['title']}**: {selected_signal1['description']}")
    
    with col2:
        st.markdown(f"**{selected_signal2['title']}**: {selected_signal2['description']}")

elif view_mode == "FFT":
    # FFT visualization
    st.header("FFT of Selected PQD")
    
    # Signal selection
    signal_key = st.selectbox(
        "Select Power Quality Disturbance Type:",
        list(signals.keys()),
        format_func=lambda x: f"{x}: {signals[x]['title']}"
    )
    
    # Get selected signal
    selected_signal = signals[signal_key]
    
    # Compute FFT
    N = len(selected_signal['signal'])
    T = selected_signal['time'][1] - selected_signal['time'][0]
    yf = fft(selected_signal['signal'])
    xf = fftfreq(N, T)[:N//2]
    
    # Plot FFT
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xf, y=2.0/N * np.abs(yf[0:N//2]), mode='lines', name='FFT', line=dict(color='royalblue')))
    fig.update_layout(title=f"FFT of {selected_signal['title']}", xaxis_title="Frequency (Hz)", yaxis_title="Amplitude", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

elif view_mode == "Wavelet Transform":
    # Wavelet Transform visualization
    st.header("Wavelet Transform of Selected PQD")
    
    # Signal selection
    signal_key = st.selectbox(
        "Select Power Quality Disturbance Type:",
        list(signals.keys()),
        format_func=lambda x: f"{x}: {signals[x]['title']}"
    )
    
    # Get selected signal
    selected_signal = signals[signal_key]
    
    # Compute Continuous Wavelet Transform
    coeffs, freqs = pywt.cwt(selected_signal['signal'], 'cmor', scales=np.arange(1, 128))
    
    # Plot Wavelet Transform
    fig = go.Figure(data=go.Heatmap(z=np.abs(coeffs), x=selected_signal['time'], y=freqs, colorscale='Viridis'))
    fig.update_layout(title=f"Wavelet Transform of {selected_signal['title']}", xaxis_title="Time (seconds)", yaxis_title="Frequency (Hz)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

elif view_mode == "Spectrogram":
    # Spectrogram visualization
    st.header("Spectrogram of Selected PQD")
    
    # Signal selection
    signal_key = st.selectbox(
        "Select Power Quality Disturbance Type:",
        list(signals.keys()),
        format_func=lambda x: f"{x}: {signals[x]['title']}"
    )
    
    # Get selected signal
    selected_signal = signals[signal_key]
    
    # Compute Spectrogram
    f, t, Sxx = signal.spectrogram(selected_signal['signal'], fs=1/(selected_signal['time'][1] - selected_signal['time'][0]))
    
    # Plot Spectrogram
    fig = go.Figure(data=go.Heatmap(z=10 * np.log10(Sxx), x=t, y=f, colorscale='Viridis'))
    fig.update_layout(title=f"Spectrogram of {selected_signal['title']}", xaxis_title="Time (seconds)", yaxis_title="Frequency (Hz)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

else:
    # View All PQDs in a grid
    st.header("All Power Quality Disturbances")
    
    # Determine grid layout
    n_signals = len(signals)
    n_cols = 3
    n_rows = (n_signals + n_cols - 1) // n_cols
    
    # Create subplot grid
    fig = make_subplots(rows=n_rows, cols=n_cols, 
                       subplot_titles=[f"{k}: {v['title']}" for k, v in signals.items()],
                       shared_xaxes=True,
                       vertical_spacing=0.12,
                       horizontal_spacing=0.05)
    
    # Add each signal to the grid
    i = 0
    for signal_key, signal_data in signals.items():
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        fig.add_trace(
            go.Scatter(
                x=signal_data['time'],
                y=signal_data['signal'],
                mode='lines',
                name=signal_key,
                line=dict(width=1.5),
                showlegend=False
            ),
            row=row, col=col
        )
        
        i += 1
    
    # Update layout
    fig.update_layout(
        title="Overview of All Power Quality Disturbances",
        height=250 * n_rows,
        template="plotly_white",
    )
    
    # Add x-axis titles only to bottom row
    for i in range(1, n_cols + 1):
        if (n_rows - 1) * n_cols + i <= n_signals:  # Check if the subplot exists
            fig.update_xaxes(title_text="Time (s)", row=n_rows, col=i)
    
    # Add y-axis titles only to left column
    for i in range(1, n_rows + 1):
        fig.update_yaxes(title_text="Amplitude", row=i, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

# Explanation section
st.markdown("---")
st.header("Understanding Power Quality Disturbances")

with st.expander("What are Power Quality Disturbances?"):
    st.markdown("""
    Power Quality Disturbances (PQDs) are deviations from the ideal voltage or current waveform in electrical power systems.
    These disturbances can affect the operation of electrical equipment, causing issues like equipment malfunction, reduced efficiency, or complete failure.
    
    The ideal waveform for AC power is a perfect sine wave with constant amplitude and frequency. Any deviation from this ideal form is considered a power quality disturbance.
    """)

with st.expander("Types of Power Quality Disturbances"):
    st.markdown("""
    ### Basic Disturbances:
    - **Normal (S1)**: Ideal sinusoidal waveform with no disturbances
    - **Sag/Dip (S2)**: Temporary reduction in voltage magnitude
    - **Swell (S3)**: Temporary increase in voltage magnitude
    - **Interruption (S4)**: Nearly complete loss of voltage for a short period
    
    ### Complex Disturbances:
    - **Harmonics (S5)**: Presence of signals at frequencies that are multiples of the fundamental frequency
    - **Flicker (S6)**: Rapid, repetitive voltage variations
    - **Oscillatory Transient (S7)**: Temporary oscillations in voltage or current
    - **Spike (S8)**: Brief high-energy impulse superimposed on the waveform
    
    ### Combined Disturbances:
    - **Harmonics + Sag (S9)**: Harmonic distortion combined with voltage sag
    - **Harmonics + Swell (S10)**: Harmonic distortion combined with voltage swell
    - **Harmonics + Interruption (S11)**: Harmonic distortion combined with interruption
    - **Harmonics + Flicker (S12)**: Harmonic distortion combined with flicker
    """)

# Footer
st.markdown("---")
st.markdown("Created with Streamlit and Plotly")


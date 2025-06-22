import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Power Quality Disturbances Visualization",
    page_icon="⚡",
    layout="wide",
)

st.title("Power Quality Disturbances Visualization")
st.subheader("Simplified Version")

# Generate simple sine wave as placeholder
t = np.linspace(0, 0.2, 1000)
y = np.sin(2 * np.pi * 50 * t)  # 50Hz sine wave

# Create a dataframe
df = pd.DataFrame({
    'Time': t,
    'Amplitude': y
})

# Plot using plotly
fig = px.line(df, x='Time', y='Amplitude', title='Normal Signal (50Hz)')
st.plotly_chart(fig, use_container_width=True)

st.info("This is a simplified version of the app. If you're seeing this, there might be an issue with the main app.py file.")

# Show versions
st.markdown("### Environment Information")
st.code(f"""
Python version: {np.version.version}
NumPy version: {np.__version__}
Pandas version: {pd.__version__}
Streamlit version: {st.__version__}
""")

st.markdown("Please check the logs for more information.")

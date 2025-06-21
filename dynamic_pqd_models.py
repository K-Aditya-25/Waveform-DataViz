import numpy as np

class DynamicPQDModels:
    """
    Enhanced class to generate different types of Power Quality Disturbances (PQD)
    with dynamically adjustable parameters.
    """
    
    def __init__(self, amplitude=1.0, frequency=50, phase=0, duration=0.2, sample_rate=10000):
        """
        Initialize PQD model with configurable parameters
        
        Args:
            amplitude: Base amplitude for the signal (default: 1.0 pu)
            frequency: Fundamental frequency in Hz (default: 50 Hz)
            phase: Initial phase in radians (default: 0)
            duration: Signal duration in seconds (default: 0.2 s)
            sample_rate: Sampling frequency (default: 10000 Hz)
        """
        # Common parameters
        self.amplitude = amplitude
        self.f = frequency
        self.phase = phase 
        self.fs = sample_rate
        self.t_max = duration
        self.t = np.linspace(0, self.t_max, int(self.fs * self.t_max))  # Time array
        self.omega = 2 * np.pi * self.f  # Angular frequency
        
    def normal(self):
        """S1: Normal signal"""
        signal = self.amplitude * np.sin(self.omega * self.t + self.phase)
        
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Normal Signal',
                'description': 'Pure sinusoidal waveform with no disturbances'}
    
    def sag(self, depth=0.5, start_time=0.05, end_time=0.15):
        """
        S2: Voltage sag signal
        
        Args:
            depth: Sag depth (0 to 1)
            start_time: Start time of sag in seconds
            end_time: End time of sag in seconds
        """
        # Generate signal with sag
        signal = np.zeros_like(self.t)
        for i, time in enumerate(self.t):
            if start_time <= time <= end_time:
                signal[i] = self.amplitude * (1 - depth) * np.sin(self.omega * time + self.phase)
            else:
                signal[i] = self.amplitude * np.sin(self.omega * time + self.phase)
                
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Voltage Sag',
                'description': 'Temporary reduction in voltage magnitude'}
    
    def swell(self, magnitude=0.5, start_time=0.05, end_time=0.15):
        """
        S3: Voltage swell signal
        
        Args:
            magnitude: Swell magnitude (0 to 1)
            start_time: Start time of swell in seconds
            end_time: End time of swell in seconds
        """
        # Generate signal with swell
        signal = np.zeros_like(self.t)
        for i, time in enumerate(self.t):
            if start_time <= time <= end_time:
                signal[i] = self.amplitude * (1 + magnitude) * np.sin(self.omega * time + self.phase)
            else:
                signal[i] = self.amplitude * np.sin(self.omega * time + self.phase)
                
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Voltage Swell',
                'description': 'Temporary increase in voltage magnitude'}
    
    def interruption(self, depth=0.9, start_time=0.05, end_time=0.15):
        """
        S4: Interruption signal
        
        Args:
            depth: Interruption depth (0 to 1)
            start_time: Start time of interruption in seconds
            end_time: End time of interruption in seconds
        """
        # Generate signal with interruption
        signal = np.zeros_like(self.t)
        for i, time in enumerate(self.t):
            if start_time <= time <= end_time:
                signal[i] = self.amplitude * (1 - depth) * np.sin(self.omega * time + self.phase)
            else:
                signal[i] = self.amplitude * np.sin(self.omega * time + self.phase)
                
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Interruption',
                'description': 'Temporary near-zero voltage condition'}
    
    def harmonics(self, h3_amp=0.2, h5_amp=0.1, h7_amp=0.05):
        """
        S5: Harmonics signal
        
        Args:
            h3_amp: 3rd harmonic amplitude (0 to 1)
            h5_amp: 5th harmonic amplitude (0 to 1)
            h7_amp: 7th harmonic amplitude (0 to 1)
        """
        # Generate signal with harmonics
        signal = self.amplitude * np.sin(self.omega * self.t + self.phase)  # Fundamental component
        signal += self.amplitude * h3_amp * np.sin(3 * self.omega * self.t + self.phase)  # 3rd harmonic
        signal += self.amplitude * h5_amp * np.sin(5 * self.omega * self.t + self.phase)  # 5th harmonic
        signal += self.amplitude * h7_amp * np.sin(7 * self.omega * self.t + self.phase)  # 7th harmonic
        
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Harmonics',
                'description': 'Waveform distortion due to multiple harmonic components'}
    
    def flicker(self, mod_index=0.1, mod_freq=10):
        """
        S6: Flicker signal
        
        Args:
            mod_index: Amplitude modulation index (0 to 0.5)
            mod_freq: Modulation frequency in Hz (1 to 25)
        """
        # Generate signal with flicker
        envelope = 1 + mod_index * np.sin(2 * np.pi * mod_freq * self.t)
        signal = self.amplitude * envelope * np.sin(self.omega * self.t + self.phase)
        
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Flicker',
                'description': 'Voltage fluctuations causing visible light variations'}
    
    def oscillatory_transient(self, amp=0.8, decay=8, start_time=0.05, freq=500):
        """
        S7: Oscillatory transient signal
        
        Args:
            amp: Transient amplitude (0 to 1)
            decay: Decay factor (1 to 20)
            start_time: Transient start time in seconds
            freq: Transient frequency in Hz (100 to 1000)
        """
        # Generate signal with transient
        signal = self.amplitude * np.sin(self.omega * self.t + self.phase)  # Base signal
        transient = np.zeros_like(self.t)
        
        # Add transient
        for i, time in enumerate(self.t):
            if time >= start_time:
                transient[i] = amp * np.exp(-decay * (time - start_time)) * np.sin(2 * np.pi * freq * (time - start_time))
                
        signal += transient
        
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Oscillatory Transient',
                'description': 'Brief, high-frequency oscillation in voltage or current'}
    
    def spike(self, amp=0.5, start_time=0.04, end_time=0.07):
        """
        S8: Spike signal
        
        Args:
            amp: Spike amplitude (0 to 1)
            start_time: Spike start time in seconds
            end_time: Spike end time in seconds
        """
        # Generate base signal
        signal = self.amplitude * np.sin(self.omega * self.t + self.phase)
        
        # Add spikes
        mask = (self.t >= start_time) & (self.t <= end_time)
        spike_shape = amp * np.sin(20 * self.omega * self.t[mask])
        signal[mask] += spike_shape
        
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Spike',
                'description': 'Short-duration high-amplitude impulse superimposed on waveform'}
    
    def harmonics_sag(self, sag_depth=0.3, start_time=0.05, end_time=0.15, h3_amp=0.2, h5_amp=0.1):
        """
        S9: Harmonics with sag signal
        
        Args:
            sag_depth: Sag depth (0 to 1)
            start_time: Start time of sag in seconds
            end_time: End time of sag in seconds
            h3_amp: 3rd harmonic amplitude (0 to 1)
            h5_amp: 5th harmonic amplitude (0 to 1)
        """
        # Generate base signal with harmonics
        signal = self.amplitude * np.sin(self.omega * self.t + self.phase)
        signal += self.amplitude * h3_amp * np.sin(3 * self.omega * self.t + self.phase)
        signal += self.amplitude * h5_amp * np.sin(5 * self.omega * self.t + self.phase)
        
        # Apply sag
        for i, time in enumerate(self.t):
            if start_time <= time <= end_time:
                signal[i] *= (1 - sag_depth)
                
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Harmonics with Sag',
                'description': 'Combined distortion from harmonics and voltage sag'}
    
    def harmonics_swell(self, swell_mag=0.3, start_time=0.05, end_time=0.15, h3_amp=0.2, h5_amp=0.1):
        """
        S10: Harmonics with swell signal
        
        Args:
            swell_mag: Swell magnitude (0 to 1)
            start_time: Start time of swell in seconds
            end_time: End time of swell in seconds
            h3_amp: 3rd harmonic amplitude (0 to 1)
            h5_amp: 5th harmonic amplitude (0 to 1)
        """
        # Generate base signal with harmonics
        signal = self.amplitude * np.sin(self.omega * self.t + self.phase)
        signal += self.amplitude * h3_amp * np.sin(3 * self.omega * self.t + self.phase)
        signal += self.amplitude * h5_amp * np.sin(5 * self.omega * self.t + self.phase)
        
        # Apply swell
        for i, time in enumerate(self.t):
            if start_time <= time <= end_time:
                signal[i] *= (1 + swell_mag)
                
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Harmonics with Swell',
                'description': 'Combined distortion from harmonics and voltage swell'}
    
    def harmonics_interruption(self, int_depth=0.9, start_time=0.05, end_time=0.15, h3_amp=0.2, h5_amp=0.1):
        """
        S11: Harmonics with interruption signal
        
        Args:
            int_depth: Interruption depth (0 to 1)
            start_time: Start time of interruption in seconds
            end_time: End time of interruption in seconds
            h3_amp: 3rd harmonic amplitude (0 to 1)
            h5_amp: 5th harmonic amplitude (0 to 1)
        """
        # Generate base signal with harmonics
        signal = self.amplitude * np.sin(self.omega * self.t + self.phase)
        signal += self.amplitude * h3_amp * np.sin(3 * self.omega * self.t + self.phase)
        signal += self.amplitude * h5_amp * np.sin(5 * self.omega * self.t + self.phase)
        
        # Apply interruption
        for i, time in enumerate(self.t):
            if start_time <= time <= end_time:
                signal[i] *= (1 - int_depth)
                
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Harmonics with Interruption',
                'description': 'Combined distortion from harmonics with voltage interruption'}
    
    def harmonics_flicker(self, mod_index=0.1, mod_freq=10, h3_amp=0.2, h5_amp=0.1):
        """
        S12: Harmonics with flicker signal
        
        Args:
            mod_index: Amplitude modulation index (0 to 0.5)
            mod_freq: Modulation frequency in Hz (1 to 25)
            h3_amp: 3rd harmonic amplitude (0 to 1)
            h5_amp: 5th harmonic amplitude (0 to 1)
        """
        # Generate base signal with harmonics
        signal = self.amplitude * np.sin(self.omega * self.t + self.phase)
        signal += self.amplitude * h3_amp * np.sin(3 * self.omega * self.t + self.phase)
        signal += self.amplitude * h5_amp * np.sin(5 * self.omega * self.t + self.phase)
        
        # Apply flicker modulation
        envelope = 1 + mod_index * np.sin(2 * np.pi * mod_freq * self.t)
        signal *= envelope
                
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Harmonics with Flicker',
                'description': 'Combined distortion from harmonics with voltage fluctuations'}

import numpy as np

class PQDModels:
    """
    Class to generate different types of Power Quality Disturbances (PQD)
    based on the mathematical models provided in the reference image.
    """
    
    def __init__(self):
        """Initialize PQD model parameters"""
        # Common parameters
        self.f = 50  # Fundamental frequency (50 Hz)
        self.fs = 10000  # Sampling frequency
        self.t_max = 0.2  # Maximum time (seconds)
        self.t = np.linspace(0, self.t_max, int(self.fs * self.t_max))  # Time array
        self.omega = 2 * np.pi * self.f  # Angular frequency
        
    def normal(self):
        """S1: Normal signal"""
        return {'signal': np.sin(self.omega * self.t), 
                'time': self.t, 
                'title': 'Normal Signal',
                'description': 'Pure sinusoidal waveform with no disturbances'}
    
    def sag(self):
        """S2: Voltage sag signal"""
        # Parameters for sag
        alpha = 0.5  # Sag depth
        t1 = 0.05  # Start time
        t2 = 0.15  # End time
        
        # Generate signal with sag
        signal = np.zeros_like(self.t)
        for i, time in enumerate(self.t):
            if t1 <= time <= t2:
                signal[i] = (1 - alpha) * np.sin(self.omega * time)
            else:
                signal[i] = np.sin(self.omega * time)
                
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Voltage Sag',
                'description': 'Temporary reduction in voltage magnitude'}
    
    def swell(self):
        """S3: Voltage swell signal"""
        # Parameters for swell
        alpha = 0.5  # Swell magnitude
        t1 = 0.05  # Start time
        t2 = 0.15  # End time
        
        # Generate signal with swell
        signal = np.zeros_like(self.t)
        for i, time in enumerate(self.t):
            if t1 <= time <= t2:
                signal[i] = (1 + alpha) * np.sin(self.omega * time)
            else:
                signal[i] = np.sin(self.omega * time)
                
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Voltage Swell',
                'description': 'Temporary increase in voltage magnitude'}
    
    def interruption(self):
        """S4: Interruption signal"""
        # Parameters for interruption
        alpha = 0.9  # Interruption depth
        t1 = 0.05  # Start time
        t2 = 0.15  # End time
        
        # Generate signal with interruption
        signal = np.zeros_like(self.t)
        for i, time in enumerate(self.t):
            if t1 <= time <= t2:
                signal[i] = (1 - alpha) * np.sin(self.omega * time)
            else:
                signal[i] = np.sin(self.omega * time)
                
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Interruption',
                'description': 'Temporary near-zero voltage condition'}
    
    def harmonics(self):
        """S5: Harmonics signal"""
        # Parameters for harmonics
        alpha_3 = 0.2  # 3rd harmonic amplitude
        alpha_5 = 0.1  # 5th harmonic amplitude
        
        # Generate signal with harmonics
        signal = np.sin(self.omega * self.t)  # Fundamental component
        signal += alpha_3 * np.sin(3 * self.omega * self.t)  # 3rd harmonic
        signal += alpha_5 * np.sin(5 * self.omega * self.t)  # 5th harmonic
        
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Harmonics',
                'description': 'Waveform distortion due to multiple harmonic components'}
    
    def flicker(self):
        """S6: Flicker signal"""
        # Parameters for flicker
        alpha = 0.1  # Flicker amplitude
        f_flicker = 10  # Flicker frequency (Hz)
        
        # Generate signal with flicker
        envelope = 1 + alpha * np.sin(2 * np.pi * f_flicker * self.t)
        signal = envelope * np.sin(self.omega * self.t)
        
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Flicker',
                'description': 'Voltage fluctuations causing visible light variations'}
    
    def oscillatory_transient(self):
        """S7: Oscillatory transient signal"""
        # Parameters for oscillatory transient
        alpha = 0.8  # Transient amplitude
        beta = 8  # Decay factor
        t1 = 0.05  # Transient start time
        f_trans = 500  # Transient frequency (Hz)
        
        # Generate signal with transient
        signal = np.sin(self.omega * self.t)  # Base signal
        transient = np.zeros_like(self.t)
        
        # Add transient
        for i, time in enumerate(self.t):
            if time >= t1:
                transient[i] = alpha * np.exp(-beta * (time - t1)) * np.sin(2 * np.pi * f_trans * (time - t1))
                
        signal += transient
        
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Oscillatory Transient',
                'description': 'Brief, high-frequency oscillation in voltage or current'}
    
    def spike(self):
        """S8: Spike signal"""
        # Parameters for spike
        alpha = 0.5  # Spike amplitude
        t1 = 0.04  # Spike start time
        t2 = 0.07  # Spike end time
        
        # Generate base signal
        signal = np.sin(self.omega * self.t)
        
        # Add spikes
        mask = (self.t >= t1) & (self.t <= t2)
        spike_shape = alpha * np.sin(20 * self.omega * self.t[mask])
        signal[mask] += spike_shape
        
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Spike',
                'description': 'Short-duration high-amplitude impulse superimposed on waveform'}
    
    def harmonics_sag(self):
        """S9: Harmonics with sag signal"""
        # Parameters for harmonics with sag
        alpha_sag = 0.3  # Sag depth
        t1 = 0.05  # Start time
        t2 = 0.15  # End time
        alpha_3 = 0.2  # 3rd harmonic amplitude
        alpha_5 = 0.1  # 5th harmonic amplitude
        
        # Generate base signal with harmonics
        signal = np.sin(self.omega * self.t)
        signal += alpha_3 * np.sin(3 * self.omega * self.t)
        signal += alpha_5 * np.sin(5 * self.omega * self.t)
        
        # Apply sag
        for i, time in enumerate(self.t):
            if t1 <= time <= t2:
                signal[i] *= (1 - alpha_sag)
                
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Harmonics with Sag',
                'description': 'Combined distortion from harmonics and voltage sag'}
    
    def harmonics_swell(self):
        """S10: Harmonics with swell signal"""
        # Parameters for harmonics with swell
        alpha_swell = 0.3  # Swell magnitude
        t1 = 0.05  # Start time
        t2 = 0.15  # End time
        alpha_3 = 0.2  # 3rd harmonic amplitude
        alpha_5 = 0.1  # 5th harmonic amplitude
        
        # Generate base signal with harmonics
        signal = np.sin(self.omega * self.t)
        signal += alpha_3 * np.sin(3 * self.omega * self.t)
        signal += alpha_5 * np.sin(5 * self.omega * self.t)
        
        # Apply swell
        for i, time in enumerate(self.t):
            if t1 <= time <= t2:
                signal[i] *= (1 + alpha_swell)
                
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Harmonics with Swell',
                'description': 'Combined distortion from harmonics and voltage swell'}
    
    def harmonics_interruption(self):
        """S11: Harmonics with interruption signal"""
        # Parameters for harmonics with interruption
        alpha_int = 0.9  # Interruption depth
        t1 = 0.05  # Start time
        t2 = 0.15  # End time
        alpha_3 = 0.2  # 3rd harmonic amplitude
        alpha_5 = 0.1  # 5th harmonic amplitude
        
        # Generate base signal with harmonics
        signal = np.sin(self.omega * self.t)
        signal += alpha_3 * np.sin(3 * self.omega * self.t)
        signal += alpha_5 * np.sin(5 * self.omega * self.t)
        
        # Apply interruption
        for i, time in enumerate(self.t):
            if t1 <= time <= t2:
                signal[i] *= (1 - alpha_int)
                
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Harmonics with Interruption',
                'description': 'Combined distortion from harmonics with voltage interruption'}
    
    def harmonics_flicker(self):
        """S12: Harmonics with flicker signal"""
        # Parameters for harmonics with flicker
        alpha_flicker = 0.1  # Flicker amplitude
        f_flicker = 10  # Flicker frequency (Hz)
        alpha_3 = 0.2  # 3rd harmonic amplitude
        alpha_5 = 0.1  # 5th harmonic amplitude
        
        # Generate base signal with harmonics
        signal = np.sin(self.omega * self.t)
        signal += alpha_3 * np.sin(3 * self.omega * self.t)
        signal += alpha_5 * np.sin(5 * self.omega * self.t)
        
        # Apply flicker modulation
        envelope = 1 + alpha_flicker * np.sin(2 * np.pi * f_flicker * self.t)
        signal *= envelope
                
        return {'signal': signal, 
                'time': self.t, 
                'title': 'Harmonics with Flicker',
                'description': 'Combined distortion from harmonics with voltage fluctuations'}
    
    def get_all_signals(self):
        """Generate all PQD signals"""
        return {
            'S1': self.normal(),
            'S2': self.sag(),
            'S3': self.swell(),
            'S4': self.interruption(),
            'S5': self.harmonics(),
            'S6': self.flicker(),
            'S7': self.oscillatory_transient(),
            'S8': self.spike(),
            'S9': self.harmonics_sag(),
            'S10': self.harmonics_swell(),
            'S11': self.harmonics_interruption(),
            'S12': self.harmonics_flicker()
        }

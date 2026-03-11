import numpy as np
from scipy.integrate import solve_ivp
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math

# ====================
# 1. KONFIGURASI & SETUP
# ====================

@dataclass
class CookingConfig:
    """Konfigurasi parameter memasak"""
    # Parameter bahan
    rice_mass: float = 10.0          # kg
    water_mass: float = 15.0         # kg
    pan_mass: float = 5.0           # kg
    
    # Parameter termal
    initial_temp: float = 25.0      # °C
    target_temp: float = 100.0      # °C
    ambient_temp: float = 25.0      # °C
    boiling_temp: float = 100.0     # °C
    
    # Parameter proses
    gelatinization_start: float = 60.0   # °C
    gelatinization_end: float = 80.0     # °C
    gelatinization_time: float = 15.0    # menit
    
    # Parameter peralatan
    burner_power: float = 3000.0    # Watt
    pan_surface_area: float = 0.5   # m²
    
    # Koefisien fisik
    Cp_water: float = 4186.0        # J/kg°C
    Cp_rice: float = 2000.0         # J/kg°C
    Cp_pan: float = 500.0           # J/kg°C
    latent_heat_vaporization: float = 2260000.0  # J/kg
    
    # Koefisien transfer panas
    heat_transfer_coeff: float = 50.0    # W/m²°C
    ambient_loss_coeff: float = 10.0     # W/m²°C
    heating_efficiency: float = 0.7      # efisiensi
    
    # Parameter simulasi
    simulation_time: float = 60.0    # menit
    time_step: float = 1.0          # detik
    
    # Atribut yang dihitung (bukan parameter input)
    total_mass: float = field(init=False, default=None)
    
    def __post_init__(self):
        """Validasi konfigurasi dan hitung atribut turunan"""
        self.total_mass = self.rice_mass + self.water_mass
        if self.water_mass / self.rice_mass < 1.0:
            st.warning("Peringatan: Rasio air:beras terlalu rendah (< 1:1)")
    
    def copy(self):
        """Buat salinan konfigurasi"""
        # Hanya copy parameter input, bukan atribut yang dihitung
        params = {k: v for k, v in self.__dict__.items() 
                 if k not in ['total_mass']}
        # Buat instance baru
        new_config = CookingConfig(**params)
        return new_config
    
    def update_parameter(self, parameter_name: str, value: float):
        """Update satu parameter dan hitung ulang atribut turunan"""
        if parameter_name in self.__annotations__:
            setattr(self, parameter_name, value)
            # Hitung ulang atribut turunan
            self.__post_init__()
        else:
            raise ValueError(f"Parameter {parameter_name} tidak valid")

# ====================
# 2. MODEL FISIKA
# ====================

class PhysicsModel:
    """Model fisika untuk proses memasak"""
    
    def __init__(self, config: CookingConfig):
        self.config = config
    
    def calculate_effective_heat_capacity(self) -> float:
        """Hitung kapasitas panas efektif sistem"""
        # Mass fractions
        water_frac = self.config.water_mass / self.config.total_mass
        rice_frac = self.config.rice_mass / self.config.total_mass
        
        # Heat capacity of mixture
        Cp_mix = (water_frac * self.config.Cp_water + 
                 rice_frac * self.config.Cp_rice)
        
        # Total heat capacity including pan
        total_capacity = (self.config.total_mass * Cp_mix + 
                         self.config.pan_mass * self.config.Cp_pan)
        
        return total_capacity
    
    def heat_input(self, temperature: float, water_content: float) -> float:
        """Hitung input panas dari burner"""
        if (temperature < self.config.target_temp and 
            water_content > 0.1 * self.config.water_mass):
            return self.config.burner_power * self.config.heating_efficiency
        return 0.0
    
    def heat_loss(self, temperature: float) -> float:
        """Hitung kehilangan panas ke lingkungan"""
        delta_T = temperature - self.config.ambient_temp
        return (self.config.ambient_loss_coeff * 
                self.config.pan_surface_area * delta_T)
    
    def evaporation_rate(self, temperature: float) -> float:
        """Hitung laju penguapan"""
        if temperature >= self.config.boiling_temp:
            # Evaporation increases with temperature above boiling
            rate_base = 0.01  # kg/menit
            temp_factor = (temperature - self.config.boiling_temp) / 10.0 + 1.0
            return rate_base * temp_factor / 60.0  # konversi ke kg/detik
        return 0.0
    
    def gelatinization_rate(self, temperature: float) -> float:
        """Hitung laju gelatinisasi"""
        if (temperature >= self.config.gelatinization_start and 
            temperature <= self.config.gelatinization_end):
            # Linear rate based on temperature
            temp_range = self.config.gelatinization_end - self.config.gelatinization_start
            return 0.1 * (temperature - self.config.gelatinization_start) / temp_range
        return 0.0

# ====================
# 3. SISTEM PERSAMAAN DIFERENSIAL
# ====================

class DifferentialEquations:
    """Sistem persamaan diferensial untuk simulasi kontinu"""
    
    def __init__(self, physics_model: PhysicsModel):
        self.physics = physics_model
        self.config = physics_model.config
    
    def system_equations(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Sistem persamaan diferensial:
        y = [temperature, water_mass, gelatinization]
        
        Returns:
        dy/dt = [dT/dt, dm/dt, dG/dt]
        """
        T, m_water, G = y
        
        # 1. Effective heat capacity
        effective_cp = self.physics.calculate_effective_heat_capacity()
        
        # 2. Heat terms
        Q_in = self.physics.heat_input(T, m_water)
        Q_loss = self.physics.heat_loss(T)
        
        # 3. Evaporation
        evap_rate = self.physics.evaporation_rate(T)
        dm_dt = -evap_rate if m_water > 0 else 0.0
        Q_evap = evap_rate * self.config.latent_heat_vaporization if m_water > 0 else 0.0
        
        # 4. Gelatinization
        gel_rate = self.physics.gelatinization_rate(T)
        dG_dt = gel_rate if G < 1.0 else 0.0
        Q_gel = 100.0 * dG_dt  # Heat for chemical process
        
        # 5. Temperature change
        net_heat = Q_in - Q_loss - Q_evap - Q_gel
        dT_dt = net_heat / effective_cp if effective_cp > 0 else 0.0
        
        # Limit temperature change rate
        dT_dt = np.clip(dT_dt, -0.5, 2.0)
        
        return np.array([dT_dt, dm_dt, dG_dt])
    
    def get_initial_conditions(self) -> np.ndarray:
        """Kondisi awal sistem"""
        return np.array([
            self.config.initial_temp,      # temperature
            self.config.water_mass,        # water mass
            0.0                            # gelatinization progress
        ])

# ====================
# 4. SIMULATOR UTAMA
# ====================

class RiceCookingSimulator:
    """Simulator utama proses memasak nasi"""
    
    def __init__(self, config: CookingConfig):
        self.config = config
        self.physics = PhysicsModel(config)
        self.equations = DifferentialEquations(self.physics)
        
        # Results storage
        self.time_history = None
        self.temperature_history = None
        self.water_history = None
        self.gelatinization_history = None
        self.results = None
    
    def run_simulation(self) -> Dict:
        """Jalankan simulasi"""
        
        # Setup time
        t_span = (0, self.config.simulation_time * 60)  # Convert to seconds
        t_eval = np.arange(0, self.config.simulation_time * 60, 
                          self.config.time_step)
        
        # Initial conditions
        y0 = self.equations.get_initial_conditions()
        
        # Solve ODE system
        solution = solve_ivp(
            fun=self.equations.system_equations,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-9,
            dense_output=True
        )
        
        # Store results
        self.time_history = solution.t / 60.0  # Convert to minutes
        self.temperature_history = solution.y[0]
        self.water_history = solution.y[1] / self.config.water_mass * 100.0
        self.gelatinization_history = solution.y[2] * 100.0
        
        # Calculate metrics
        self.results = self._calculate_metrics()
        
        return self.results
    
    def _calculate_metrics(self) -> Dict:
        """Hitung metrik kualitas memasak"""
        if self.time_history is None:
            raise ValueError("Jalankan simulasi terlebih dahulu")
        
        metrics = {
            # Time metrics
            'time_to_boiling': self._get_time_to_temperature(self.config.boiling_temp),
            'time_to_target': self._get_time_to_temperature(self.config.target_temp),
            'time_full_gelatinization': self._get_time_to_gelatinization(95.0),
            
            # Temperature metrics
            'max_temperature': np.max(self.temperature_history),
            'avg_temperature': np.mean(self.temperature_history),
            'final_temperature': self.temperature_history[-1],
            
            # Quality metrics
            'final_gelatinization': self.gelatinization_history[-1],
            'final_water_content': self.water_history[-1],
            'max_water_loss': 100.0 - np.min(self.water_history),
            
            # Energy metrics
            'energy_consumption': self._calculate_energy_consumption(),
            'cooking_efficiency': self._calculate_efficiency()
        }
        
        return metrics
    
    def _get_time_to_temperature(self, target_temp: float) -> Optional[float]:
        """Waktu untuk mencapai suhu tertentu"""
        indices = np.where(self.temperature_history >= target_temp)[0]
        if len(indices) > 0:
            return self.time_history[indices[0]]
        return None
    
    def _get_time_to_gelatinization(self, target_gel: float) -> Optional[float]:
        """Waktu untuk mencapai tingkat gelatinisasi tertentu"""
        indices = np.where(self.gelatinization_history >= target_gel)[0]
        if len(indices) > 0:
            return self.time_history[indices[0]]
        return None
    
    def _calculate_energy_consumption(self) -> float:
        """Hitung konsumsi energi total"""
        cooking_time = self.time_history[-1] * 60.0  # seconds
        energy = self.config.burner_power * cooking_time / 3600000.0  # kWh
        return energy
    
    def _calculate_efficiency(self) -> float:
        """Hitung efisiensi memasak"""
        # Heat required to cook rice (theoretical minimum)
        heat_required = (self.config.total_mass * self.physics.calculate_effective_heat_capacity() *
                        (self.config.target_temp - self.config.initial_temp))
        
        # Actual energy used
        energy_used = self._calculate_energy_consumption() * 3600000.0  # Joules
        
        if energy_used > 0:
            return min(heat_required / energy_used * 100.0, 100.0)
        return 0.0

# ====================
# 5. VISUALISASI dengan PLOTLY
# ====================

class PlotlyVisualization:
    """Kelas untuk visualisasi hasil simulasi dengan Plotly"""
    
    @staticmethod
    def plot_temperature_profile(simulator: RiceCookingSimulator):
        """Plot profil suhu"""
        fig = go.Figure()
        
        time = simulator.time_history
        temp = simulator.temperature_history
        config = simulator.config
        
        # Tambah garis suhu
        fig.add_trace(go.Scatter(
            x=time, 
            y=temp,
            mode='lines',
            name='Suhu Nasi',
            line=dict(color='blue', width=3),
            hovertemplate='Waktu: %{x:.1f} menit<br>Suhu: %{y:.1f}°C<extra></extra>'
        ))
        
        # Tambah garis referensi
        fig.add_hline(y=config.target_temp, line_dash="dash", 
                     line_color="red", opacity=0.7,
                     annotation_text=f"Target ({config.target_temp}°C)")
        
        fig.add_hline(y=config.boiling_temp, line_dash="dash", 
                     line_color="green", opacity=0.7,
                     annotation_text=f"Titik Didih ({config.boiling_temp}°C)")
        
        fig.add_hline(y=config.gelatinization_start, line_dash="dot", 
                     line_color="orange", opacity=0.6,
                     annotation_text="Mulai Gelatinisasi")
        
        fig.add_hline(y=config.gelatinization_end, line_dash="dot", 
                     line_color="orange", opacity=0.6,
                     annotation_text="Akhir Gelatinisasi")
        
        # Tambah zona gelatinisasi
        fig.add_hrect(y0=config.gelatinization_start, y1=config.gelatinization_end,
                     fillcolor="orange", opacity=0.1, line_width=0,
                     annotation_text="Zona Gelatinisasi Optimal")
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Profil Suhu Selama Memasak',
                font=dict(size=20, family="Arial, sans-serif", color="darkblue")
            ),
            xaxis_title="Waktu (menit)",
            yaxis_title="Suhu (°C)",
            hovermode="x unified",
            showlegend=True,
            height=500,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def plot_quality_metrics(simulator: RiceCookingSimulator):
        """Plot metrik kualitas dalam subplot"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Proses Gelatinisasi Pati', 
                          'Perubahan Kandungan Air',
                          'Diagram Fase Memasak', 
                          'Akumulasi Konsumsi Energi'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        time = simulator.time_history
        
        # Plot 1: Gelatinization progress
        fig.add_trace(
            go.Scatter(
                x=time, 
                y=simulator.gelatinization_history,
                mode='lines',
                name='Gelatinisasi',
                line=dict(color='green', width=2.5),
                hovertemplate='Waktu: %{x:.1f} menit<br>Gelatinisasi: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        fig.add_hline(y=95, line_dash="dash", line_color="red", opacity=0.7,
                     annotation_text="95% (Matang)", row=1, col=1)
        
        # Plot 2: Water content
        fig.add_trace(
            go.Scatter(
                x=time, 
                y=simulator.water_history,
                mode='lines',
                name='Kandungan Air',
                line=dict(color='purple', width=2.5),
                hovertemplate='Waktu: %{x:.1f} menit<br>Air: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=2
        )
        fig.add_hline(y=50, line_dash="dash", line_color="red", opacity=0.7,
                     annotation_text="Batas Minimal 50%", row=1, col=2)
        
        # Plot 3: Phase diagram
        PlotlyVisualization._plot_phase_diagram(simulator, fig, row=2, col=1)
        
        # Plot 4: Energy accumulation
        PlotlyVisualization._plot_energy_accumulation(simulator, fig, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            hovermode="x unified",
            template="plotly_white"
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Waktu (menit)", row=2, col=1)
        fig.update_xaxes(title_text="Waktu (menit)", row=2, col=2)
        fig.update_yaxes(title_text="Gelatinisasi (%)", row=1, col=1)
        fig.update_yaxes(title_text="Kandungan Air (%)", row=1, col=2)
        fig.update_yaxes(title_text="Intensitas Proses", row=2, col=1)
        fig.update_yaxes(title_text="Energi (kWh)", row=2, col=2)
        
        return fig
    
    @staticmethod
    def _plot_phase_diagram(simulator: RiceCookingSimulator, fig, row, col):
        """Plot diagram fase proses"""
        time = simulator.time_history
        temp = simulator.temperature_history
        config = simulator.config
        
        # Define phases
        phases = []
        for T in temp:
            if T < config.gelatinization_start:
                phases.append(0)  # Pemanasan awal
            elif T < config.gelatinization_end:
                phases.append(1)  # Gelatinisasi
            elif T < config.boiling_temp:
                phases.append(2)  # Pemanasan lanjut
            elif T >= config.boiling_temp:
                phases.append(3)  # Pendidihan
            else:
                phases.append(4)  # Pendinginan
        
        # Color mapping
        phase_colors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'red']
        phase_names = ['Pemanasan Awal', 'Gelatinisasi', 'Pemanasan Lanjut', 
                      'Pendidihan', 'Pematangan']
        
        # Create a trace for each phase
        current_phase = phases[0]
        start_idx = 0
        
        for i, phase in enumerate(phases[1:], 1):
            if phase != current_phase or i == len(phases) - 1:
                # Add filled area for this phase
                fig.add_trace(
                    go.Scatter(
                        x=[time[start_idx], time[i-1], time[i-1], time[start_idx]],
                        y=[0, 0, 100, 100],
                        fill='toself',
                        fillcolor=phase_colors[current_phase],
                        opacity=0.4,
                        line=dict(width=0),
                        name=phase_names[current_phase],
                        showlegend=False,
                        hoverinfo='text',
                        text=f'{phase_names[current_phase]}<br>{time[start_idx]:.1f} - {time[i-1]:.1f} menit'
                    ),
                    row=row, col=col
                )
                
                # Add phase label
                mid_time = (time[start_idx] + time[i-1]) / 2
                fig.add_annotation(
                    x=mid_time,
                    y=50,
                    text=phase_names[current_phase],
                    showarrow=False,
                    textangle=90,
                    font=dict(size=10, color='black'),
                    row=row, col=col
                )
                
                current_phase = phase
                start_idx = i
    
    @staticmethod
    def _plot_energy_accumulation(simulator: RiceCookingSimulator, fig, row, col):
        """Plot akumulasi energi"""
        time = simulator.time_history
        temp = simulator.temperature_history
        config = simulator.config
        
        # Calculate cumulative energy
        energy = np.zeros_like(time)
        for i in range(1, len(time)):
            dt = (time[i] - time[i-1]) * 60  # seconds
            if temp[i] < config.target_temp:
                energy[i] = energy[i-1] + config.burner_power * dt / 3600000  # kWh
            else:
                energy[i] = energy[i-1]
        
        fig.add_trace(
            go.Scatter(
                x=time, 
                y=energy,
                mode='lines',
                name='Energi Kumulatif',
                line=dict(color='blue', width=2),
                hovertemplate='Waktu: %{x:.1f} menit<br>Energi: %{y:.3f} kWh<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add efficiency annotation
        efficiency = simulator.results['cooking_efficiency']
        fig.add_annotation(
            x=0.05,
            y=0.95,
            xref='paper',
            yref='paper',
            text=f'Efisiensi: {efficiency:.1f}%',
            showarrow=False,
            bgcolor='wheat',
            opacity=0.7,
            row=row, col=col
        )
    
    @staticmethod
    def plot_comparison_chart(simulators: List[RiceCookingSimulator], 
                             labels: List[str]):
        """Plot perbandingan beberapa simulasi"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Perbandingan Profil Suhu', 
                          'Perbandingan Proses Gelatinisasi',
                          'Perbandingan Kandungan Air', 
                          'Perbandingan Metrik Kinerja'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Plot 1: Temperature comparison
        for i, sim in enumerate(simulators):
            fig.add_trace(
                go.Scatter(
                    x=sim.time_history,
                    y=sim.temperature_history,
                    mode='lines',
                    name=labels[i],
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate='Waktu: %{x:.1f} menit<br>Suhu: %{y:.1f}°C<extra>'+labels[i]+'</extra>'
                ),
                row=1, col=1
            )
        
        # Plot 2: Gelatinization comparison
        for i, sim in enumerate(simulators):
            fig.add_trace(
                go.Scatter(
                    x=sim.time_history,
                    y=sim.gelatinization_history,
                    mode='lines',
                    name=labels[i],
                    line=dict(color=colors[i % len(colors)], width=2),
                    showlegend=False,
                    hovertemplate='Waktu: %{x:.1f} menit<br>Gelatinisasi: %{y:.1f}%<extra>'+labels[i]+'</extra>'
                ),
                row=1, col=2
            )
        
        # Plot 3: Water content comparison
        for i, sim in enumerate(simulators):
            fig.add_trace(
                go.Scatter(
                    x=sim.time_history,
                    y=sim.water_history,
                    mode='lines',
                    name=labels[i],
                    line=dict(color=colors[i % len(colors)], width=2),
                    showlegend=False,
                    hovertemplate='Waktu: %{x:.1f} menit<br>Air: %{y:.1f}%<extra>'+labels[i]+'</extra>'
                ),
                row=2, col=1
            )
        
        # Plot 4: Metrics comparison (bar chart)
        metrics = ['time_to_target', 'final_gelatinization', 
                  'final_water_content', 'energy_consumption']
        metric_labels = ['Waktu ke 100°C (mnt)', 'Gelatinisasi Akhir (%)',
                        'Air Akhir (%)', 'Energi (kWh)']
        
        for i, sim in enumerate(simulators):
            values = [sim.results[metric] for metric in metrics]
            # Handle None values
            values = [0 if v is None else v for v in values]
            
            fig.add_trace(
                go.Bar(
                    name=labels[i],
                    x=metric_labels,
                    y=values,
                    marker_color=colors[i % len(colors)],
                    opacity=0.7,
                    showlegend=False,
                    hovertemplate='Metrik: %{x}<br>Nilai: %{y:.2f}<extra>'+labels[i]+'</extra>'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            barmode='group',
            hovermode="closest",
            template="plotly_white"
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Waktu (menit)", row=1, col=1)
        fig.update_xaxes(title_text="Waktu (menit)", row=1, col=2)
        fig.update_xaxes(title_text="Waktu (menit)", row=2, col=1)
        fig.update_yaxes(title_text="Suhu (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Gelatinisasi (%)", row=1, col=2)
        fig.update_yaxes(title_text="Kandungan Air (%)", row=2, col=1)
        fig.update_yaxes(title_text="Nilai", row=2, col=2)
        
        return fig

# ====================
# 6. ANALISIS SENSITIVITAS
# ====================

class SensitivityAnalysis:
    """Analisis sensitivitas parameter"""
    
    @staticmethod
    def analyze_parameter_sensitivity(base_config: CookingConfig,
                                     parameter_name: str,
                                     values: List[float]) -> Dict:
        """
        Analisis sensitivitas untuk satu parameter
        
        Returns:
            Dictionary dengan hasil untuk setiap nilai parameter
        """
        results = []
        
        for value in values:
            # Create new config with modified parameter menggunakan copy()
            config = base_config.copy()
            config.update_parameter(parameter_name, value)
            
            # Run simulation
            simulator = RiceCookingSimulator(config)
            metrics = simulator.run_simulation()
            
            results.append({
                'value': value,
                'simulator': simulator,
                'metrics': metrics
            })
        
        return {
            'parameter': parameter_name,
            'results': results
        }
    
    @staticmethod
    def multi_parameter_analysis(base_config: CookingConfig,
                               parameter_ranges: Dict[str, List[float]]) -> Dict:
        """
        Analisis sensitivitas untuk multiple parameters
        """
        all_results = {}
        
        for param_name, values in parameter_ranges.items():
            st.write(f"Analisis sensitivitas untuk parameter: {param_name}")
            results = SensitivityAnalysis.analyze_parameter_sensitivity(
                base_config, param_name, values
            )
            all_results[param_name] = results
        
        return all_results

# ====================
# 7. APLIKASI STREAMLIT
# ====================

def create_sidebar():
    """Buat sidebar untuk input parameter"""
    st.sidebar.title("⚙️ Parameter")
    
    st.sidebar.subheader("Parameter Bahan")
    rice_mass = st.sidebar.slider("Massa Beras (kg)", 1.0, 50.0, 10.0, 0.5)
    water_mass = st.sidebar.slider("Massa Air (kg)", 5.0, 50.0, 15.0, 0.5)
    pan_mass = st.sidebar.slider("Massa Panci (kg)", 1.0, 20.0, 5.0, 0.5)
    
    st.sidebar.subheader("Parameter Termal")
    initial_temp = st.sidebar.slider("Suhu Awal (°C)", 0.0, 50.0, 25.0, 1.0)
    target_temp = st.sidebar.slider("Suhu Target (°C)", 80.0, 120.0, 100.0, 1.0)
    ambient_temp = st.sidebar.slider("Suhu Lingkungan (°C)", 10.0, 40.0, 25.0, 1.0)
    
    st.sidebar.subheader("Parameter Peralatan")
    burner_power = st.sidebar.slider("Daya Burner (Watt)", 1000.0, 10000.0, 3000.0, 100.0)
    pan_surface_area = st.sidebar.slider("Luas Permukaan Panci (m²)", 0.1, 2.0, 0.5, 0.1)
    
    st.sidebar.subheader("Parameter Proses")
    gelatinization_start = st.sidebar.slider("Mulai Gelatinisasi (°C)", 50.0, 70.0, 60.0, 1.0)
    gelatinization_end = st.sidebar.slider("Akhir Gelatinisasi (°C)", 70.0, 90.0, 80.0, 1.0)
    
    st.sidebar.subheader("Parameter Simulasi")
    simulation_time = st.sidebar.slider("Waktu Simulasi (menit)", 10, 120, 60, 5)
    
    # Advanced parameters (expandable)
    with st.sidebar.expander("Parameter Lanjutan"):
        heating_efficiency = st.slider("Efisiensi Pemanasan", 0.1, 1.0, 0.7, 0.05)
        heat_transfer_coeff = st.slider("Koef. Transfer Panas (W/m²°C)", 10.0, 200.0, 50.0, 5.0)
        ambient_loss_coeff = st.slider("Koef. Kehilangan Panas (W/m²°C)", 1.0, 50.0, 10.0, 1.0)
    
    # Buat konfigurasi
    config = CookingConfig(
        rice_mass=rice_mass,
        water_mass=water_mass,
        pan_mass=pan_mass,
        initial_temp=initial_temp,
        target_temp=target_temp,
        ambient_temp=ambient_temp,
        boiling_temp=100.0,
        gelatinization_start=gelatinization_start,
        gelatinization_end=gelatinization_end,
        burner_power=burner_power,
        pan_surface_area=pan_surface_area,
        heating_efficiency=heating_efficiency,
        heat_transfer_coeff=heat_transfer_coeff,
        ambient_loss_coeff=ambient_loss_coeff,
        simulation_time=float(simulation_time)
    )
    
    return config

def display_results(simulator, results):
    """Tampilkan hasil simulasi"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Waktu ke 100°C", 
                 f"{results['time_to_target']:.1f} menit" if results['time_to_target'] else "Tidak tercapai",
                 delta=None)
        st.metric("Suhu Maksimum", f"{results['max_temperature']:.1f}°C")
    
    with col2:
        st.metric("Gelatinisasi Akhir", f"{results['final_gelatinization']:.1f}%")
        st.metric("Kandungan Air Akhir", f"{results['final_water_content']:.1f}%")
    
    with col3:
        st.metric("Konsumsi Energi", f"{results['energy_consumption']:.2f} kWh")
        st.metric("Kehilangan Air Maks", f"{results['max_water_loss']:.1f}%")
    
    with col4:
        st.metric("Efisiensi Memasak", f"{results['cooking_efficiency']:.1f}%")
        st.metric("Suhu Rata-rata", f"{results['avg_temperature']:.1f}°C")

def main():
    """Aplikasi utama Streamlit"""
    st.set_page_config(
        page_title="Simulasi Memasak Nasi",
        page_icon="🍚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🍚 Simulasi Kontinu Proses Memasak Nasi")
    st.markdown("""
    Aplikasi ini mensimulasikan proses memasak nasi secara kontinu menggunakan model fisika termodinamika.
    Sesuaikan parameter di sidebar dan lihat hasil simulasi secara real-time.
    """)
    
    # Sidebar untuk input parameter
    config = create_sidebar()
    
    # Jalankan simulasi
    with st.spinner("Menjalankan simulasi..."):
        simulator = RiceCookingSimulator(config)
        results = simulator.run_simulation()
    
    # Tampilkan hasil
    st.success("✅ Simulasi selesai!")
    display_results(simulator, results)
    
    # Tab untuk visualisasi
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Profil Suhu", "📊 Metrik Kualitas", "🔍 Analisis Sensitivitas", "📋 Data"])
    
    with tab1:
        st.subheader("Profil Suhu selama Memasak")
        fig_temp = PlotlyVisualization.plot_temperature_profile(simulator)
        st.plotly_chart(fig_temp, use_container_width=True)
        
        # Tampilkan fase memasak
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Fase Memasak")
            phases_info = {
                "Pemanasan Awal": f"Suhu < {config.gelatinization_start}°C",
                "Gelatinisasi": f"{config.gelatinization_start}°C - {config.gelatinization_end}°C",
                "Pendidihan": f"Suhu ≥ {config.boiling_temp}°C",
                "Pematangan": "Setelah gelatinisasi"
            }
            for phase, desc in phases_info.items():
                st.info(f"**{phase}**: {desc}")
    
    with tab2:
        st.subheader("Metrik Kualitas Memasak")
        fig_quality = PlotlyVisualization.plot_quality_metrics(simulator)
        st.plotly_chart(fig_quality, use_container_width=True)
        
        # Evaluasi kualitas
        st.subheader("Evaluasi Kualitas")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Normalisasi nilai gelatinisasi ke rentang 0-1
            gel_score = min(results['final_gelatinization'] / 100.0, 1.0)
            st.progress(gel_score, text=f"Kematangan: {results['final_gelatinization']:.1f}%")
            if results['final_gelatinization'] >= 95:
                st.success("✅ Nasi matang sempurna")
            elif results['final_gelatinization'] >= 80:
                st.warning("⚠️ Nasi cukup matang")
            else:
                st.error("❌ Nasi kurang matang")
        
        with col2:
            # Normalisasi kandungan air
            water_score = min(results['final_water_content'] / 100.0, 1.0)
            water_score = max(water_score, 0.0)  # Pastikan tidak negatif
            st.progress(water_score, text=f"Kandungan Air: {results['final_water_content']:.1f}%")
            if results['final_water_content'] >= 50:
                st.success("✅ Kandungan air optimal")
            else:
                st.warning("⚠️ Nasi terlalu kering")
        
        with col3:
            # Normalisasi efisiensi
            efficiency_score = min(results['cooking_efficiency'] / 100.0, 1.0)
            efficiency_score = max(efficiency_score, 0.0)  # Pastikan tidak negatif
            st.progress(efficiency_score, text=f"Efisiensi: {results['cooking_efficiency']:.1f}%")
            if results['cooking_efficiency'] >= 70:
                st.success("✅ Efisiensi tinggi")
            elif results['cooking_efficiency'] >= 50:
                st.warning("⚠️ Efisiensi sedang")
            else:
                st.error("❌ Efisiensi rendah")
    
    with tab3:
        st.subheader("Analisis Sensitivitas Parameter")
        
        # Pilih parameter untuk analisis sensitivitas
        param_options = {
            "Daya Burner": "burner_power",
            "Massa Air": "water_mass",
            "Massa Beras": "rice_mass",
            "Suhu Awal": "initial_temp"
        }
        
        selected_param = st.selectbox(
            "Pilih parameter untuk analisis sensitivitas:",
            list(param_options.keys())
        )
        
        # Buat range nilai
        param_name = param_options[selected_param]
        
        if param_name == "burner_power":
            values = [2000, 2500, 3000, 3500, 4000]
        elif param_name == "water_mass":
            values = [12, 13, 14, 15, 16, 17, 18]
        elif param_name == "rice_mass":
            values = [8, 9, 10, 11, 12]
        else:  # initial_temp
            values = [15, 20, 25, 30, 35]
        
        # Jalankan analisis
        if st.button("Jalankan Analisis Sensitivitas", type="primary"):
            with st.spinner(f"Menjalankan analisis untuk {selected_param}..."):
                analysis = SensitivityAnalysis.analyze_parameter_sensitivity(
                    config, param_name, values
                )
                
                # Buat dataframe hasil
                analysis_data = []
                for result in analysis['results']:
                    analysis_data.append({
                        'Nilai': result['value'],
                        'Waktu ke 100°C (menit)': result['metrics']['time_to_target'] or 0,
                        'Gelatinisasi Akhir (%)': result['metrics']['final_gelatinization'],
                        'Konsumsi Energi (kWh)': result['metrics']['energy_consumption'],
                        'Efisiensi (%)': result['metrics']['cooking_efficiency']
                    })
                
                df_analysis = pd.DataFrame(analysis_data)
                
                # Tampilkan tabel
                st.dataframe(df_analysis.style.format({
                    'Waktu ke 100°C (menit)': '{:.1f}',
                    'Gelatinisasi Akhir (%)': '{:.1f}',
                    'Konsumsi Energi (kWh)': '{:.3f}',
                    'Efisiensi (%)': '{:.1f}'
                }), use_container_width=True)
                
                # Buat grafik sensitivitas
                fig_sens = go.Figure()
                
                for metric, color in [('Waktu ke 100°C (menit)', 'red'), 
                                     ('Gelatinisasi Akhir (%)', 'green'),
                                     ('Efisiensi (%)', 'blue')]:
                    fig_sens.add_trace(go.Scatter(
                        x=df_analysis['Nilai'],
                        y=df_analysis[metric],
                        mode='lines+markers',
                        name=metric,
                        line=dict(color=color, width=2),
                        hovertemplate=f'{selected_param}: %{{x}}<br>{metric}: %{{y:.1f}}<extra></extra>'
                    ))
                
                fig_sens.update_layout(
                    title=f"Sensitivitas {selected_param}",
                    xaxis_title=selected_param,
                    yaxis_title="Nilai Metrik",
                    hovermode="x unified",
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig_sens, use_container_width=True)
                
                # Simulasi perbandingan
                st.subheader("Simulasi Perbandingan Skenario")
                
                # Buat skenario perbandingan
                scenarios = []
                labels = []
                
                for i, result in enumerate(analysis['results']):
                    scenarios.append(result['simulator'])
                    labels.append(f"{selected_param} = {result['value']}")
                
                if len(scenarios) > 1:
                    fig_comp = PlotlyVisualization.plot_comparison_chart(scenarios, labels)
                    st.plotly_chart(fig_comp, use_container_width=True)
    
    with tab4:
        st.subheader("Data Simulasi")
        
        # Buat dataframe dari hasil
        data = {
            'Waktu (menit)': simulator.time_history,
            'Suhu (°C)': simulator.temperature_history,
            'Gelatinisasi (%)': simulator.gelatinization_history,
            'Kandungan Air (%)': simulator.water_history
        }
        
        df = pd.DataFrame(data)
        
        # Tampilkan tabel
        st.dataframe(df.style.format({
            'Waktu (menit)': '{:.1f}',
            'Suhu (°C)': '{:.1f}',
            'Gelatinisasi (%)': '{:.1f}',
            'Kandungan Air (%)': '{:.1f}'
        }), use_container_width=True, height=400)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data sebagai CSV",
            data=csv,
            file_name="hasil_simulasi_memasak.csv",
            mime="text/csv"
        )
        
        # Tampilkan parameter simulasi
        with st.expander("📋 Parameter Simulasi Lengkap"):
            param_dict = {
                "Massa Beras": f"{config.rice_mass} kg",
                "Massa Air": f"{config.water_mass} kg",
                "Massa Panci": f"{config.pan_mass} kg",
                "Suhu Awal": f"{config.initial_temp}°C",
                "Suhu Target": f"{config.target_temp}°C",
                "Daya Burner": f"{config.burner_power} W",
                "Luas Permukaan Panci": f"{config.pan_surface_area} m²",
                "Efisiensi Pemanasan": f"{config.heating_efficiency*100:.1f}%",
                "Waktu Simulasi": f"{config.simulation_time} menit"
            }
            
            for param, value in param_dict.items():
                st.write(f"**{param}:** {value}")
    
    # Footer
    st.markdown("---")

if __name__ == "__main__":
    main()
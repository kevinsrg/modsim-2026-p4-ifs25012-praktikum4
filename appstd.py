import numpy as np
from scipy.integrate import solve_ivp
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math

# ====================
# 1. KONFIGURASI & SETUP
# ====================

@dataclass
class TankConfig:
    """Konfigurasi parameter tangki air"""
    # Dimensi tangki (silinder)
    tank_radius: float = 1.0          # m
    tank_height_max: float = 2.0      # m

    # Parameter pipa inlet
    inlet_flow_rate: float = 0.005    # m³/s (laju aliran masuk)
    inlet_active: bool = True          # katup inlet terbuka/tertutup

    # Parameter pipa outlet
    outlet_diameter: float = 0.05     # m (diameter pipa outlet)
    outlet_discharge_coeff: float = 0.6  # koefisien debit (Cd)
    outlet_active: bool = True         # katup outlet terbuka/tertutup

    # Kondisi awal
    initial_height: float = 0.0      # m (ketinggian awal)

    # Parameter simulasi
    simulation_time: float = 120.0    # menit
    time_step: float = 1.0            # detik

    # Atribut turunan
    tank_area: float = field(init=False, default=None)
    outlet_area: float = field(init=False, default=None)
    tank_volume: float = field(init=False, default=None)
    g: float = field(init=False, default=9.81)  # gravitasi m/s²

    def __post_init__(self):
        """Hitung atribut turunan"""
        self.tank_area = math.pi * self.tank_radius ** 2       # m²
        self.outlet_area = math.pi * (self.outlet_diameter / 2) ** 2  # m²
        self.tank_volume = self.tank_area * self.tank_height_max       # m³

    def copy(self):
        params = {k: v for k, v in self.__dict__.items()
                  if k not in ['tank_area', 'outlet_area', 'tank_volume', 'g']}
        return TankConfig(**params)


# ====================
# 2. MODEL FISIKA
# ====================

class TankPhysicsModel:
    """Model fisika sistem tangki air"""

    def __init__(self, config: TankConfig):
        self.config = config

    def inlet_flow(self) -> float:
        """Laju aliran masuk dari pipa inlet (m³/s)"""
        if self.config.inlet_active:
            return self.config.inlet_flow_rate
        return 0.0

    def outlet_flow(self, height: float) -> float:
        """
        Laju aliran keluar berdasarkan Hukum Torricelli.
        Q_out = Cd * A_out * sqrt(2 * g * h)
        """
        if self.config.outlet_active and height > 0:
            v_out = math.sqrt(2 * self.config.g * max(height, 0.0))
            return self.config.outlet_discharge_coeff * self.config.outlet_area * v_out
        return 0.0

    def net_flow(self, height: float) -> float:
        """Laju aliran bersih (m³/s)"""
        return self.inlet_flow() - self.outlet_flow(height)


# ====================
# 3. SISTEM PERSAMAAN DIFERENSIAL
# ====================

class TankDifferentialEquations:
    """Sistem persamaan diferensial untuk simulasi tangki air"""

    def __init__(self, physics: TankPhysicsModel):
        self.physics = physics
        self.config = physics.config

    def system_equations(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Persamaan diferensial:
        y = [h]  →  ketinggian air (m)

        dh/dt = (Q_in - Q_out) / A_tank

        Returns: [dh/dt]
        """
        h = y[0]

        # Batasi ketinggian [0, h_max]
        h = np.clip(h, 0.0, self.config.tank_height_max)

        Q_in  = self.physics.inlet_flow()
        Q_out = self.physics.outlet_flow(h)

        # Jika tangki sudah penuh, hentikan pengisian
        if h >= self.config.tank_height_max and Q_in > Q_out:
            Q_in = Q_out

        # Jika tangki sudah kosong, hentikan pengosongan
        if h <= 0.0 and Q_out > Q_in:
            Q_out = Q_in

        dh_dt = (Q_in - Q_out) / self.config.tank_area

        return np.array([dh_dt])

    def get_initial_conditions(self) -> np.ndarray:
        """Kondisi awal sistem"""
        return np.array([self.config.initial_height])


# ====================
# 4. SIMULATOR UTAMA
# ====================

class WaterTankSimulator:
    """Simulator utama sistem tangki air"""

    def __init__(self, config: TankConfig):
        self.config = config
        self.physics = TankPhysicsModel(config)
        self.equations = TankDifferentialEquations(self.physics)

        self.time_history = None
        self.height_history = None
        self.volume_history = None
        self.flow_in_history = None
        self.flow_out_history = None
        self.results = None

    def run_simulation(self) -> Dict:
        t_span = (0, self.config.simulation_time * 60)
        t_eval = np.arange(0, self.config.simulation_time * 60,
                           self.config.time_step)

        y0 = self.equations.get_initial_conditions()

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

        self.time_history    = solution.t / 60.0   # menit
        self.height_history  = np.clip(solution.y[0], 0.0, self.config.tank_height_max)
        self.volume_history  = self.height_history * self.config.tank_area

        # Hitung aliran di setiap titik waktu
        self.flow_in_history  = np.full_like(self.time_history,
                                             self.physics.inlet_flow() * 1000)  # L/s
        self.flow_out_history = np.array([
            self.physics.outlet_flow(h) * 1000 for h in self.height_history
        ])  # L/s

        self.results = self._calculate_metrics()
        return self.results

    def _calculate_metrics(self) -> Dict:
        h_max   = self.config.tank_height_max
        h_init  = self.config.initial_height
        A_tank  = self.config.tank_area
        V_max   = self.config.tank_volume

        # Waktu pengisian penuh
        time_full = None
        if self.config.inlet_active:
            idx = np.where(self.height_history >= h_max * 0.995)[0]
            if len(idx) > 0:
                time_full = self.time_history[idx[0]]

        # Waktu pengosongan
        time_empty = None
        if self.config.outlet_active:
            idx = np.where(self.height_history <= h_max * 0.005)[0]
            if len(idx) > 0:
                time_empty = self.time_history[idx[0]]

        final_height = self.height_history[-1]
        final_volume = self.volume_history[-1]
        fill_percent = (final_height / h_max) * 100

        # Volume air yang masuk / keluar
        dt = self.config.time_step
        vol_in  = np.sum(self.flow_in_history  / 1000) * dt   # m³
        vol_out = np.sum(self.flow_out_history / 1000) * dt   # m³

        return {
            'time_full':       time_full,
            'time_empty':      time_empty,
            'final_height':    final_height,
            'final_volume':    final_volume,
            'fill_percent':    fill_percent,
            'volume_in':       vol_in,
            'volume_out':      vol_out,
            'max_outlet_flow': float(np.max(self.flow_out_history)),
            'avg_outlet_flow': float(np.mean(self.flow_out_history)),
            'tank_volume':     V_max,
            'tank_area':       A_tank,
        }


# ====================
# 5. VISUALISASI PLOTLY
# ====================

class TankVisualization:

    @staticmethod
    def plot_height_profile(simulator: WaterTankSimulator) -> go.Figure:
        """Plot profil ketinggian air terhadap waktu"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=simulator.time_history,
            y=simulator.height_history,
            mode='lines',
            name='Ketinggian Air',
            line=dict(color='#1f77b4', width=3),
            fill='tozeroy',
            fillcolor='rgba(31,119,180,0.15)'
        ))

        # Garis referensi kapasitas penuh
        fig.add_hline(
            y=simulator.config.tank_height_max,
            line_dash='dash', line_color='red',
            annotation_text=f'Kapasitas Penuh ({simulator.config.tank_height_max} m)',
            annotation_position='top right'
        )

        # Garis ketinggian awal
        if simulator.config.initial_height > 0:
            fig.add_hline(
                y=simulator.config.initial_height,
                line_dash='dot', line_color='orange',
                annotation_text='Ketinggian Awal'
            )

        fig.update_layout(
            title='Profil Ketinggian Air dalam Tangki terhadap Waktu',
            xaxis_title='Waktu (menit)',
            yaxis_title='Ketinggian Air (m)',
            yaxis=dict(range=[0, simulator.config.tank_height_max * 1.1]),
            template='plotly_white',
            height=420,
            legend=dict(x=0.01, y=0.99)
        )
        return fig

    @staticmethod
    def plot_flow_rates(simulator: WaterTankSimulator) -> go.Figure:
        """Plot laju aliran masuk dan keluar"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=simulator.time_history,
            y=simulator.flow_in_history,
            mode='lines',
            name='Aliran Masuk (Q_in)',
            line=dict(color='green', width=2.5)
        ))

        fig.add_trace(go.Scatter(
            x=simulator.time_history,
            y=simulator.flow_out_history,
            mode='lines',
            name='Aliran Keluar (Q_out)',
            line=dict(color='red', width=2.5)
        ))

        fig.update_layout(
            title='Laju Aliran Masuk vs Keluar',
            xaxis_title='Waktu (menit)',
            yaxis_title='Laju Aliran (L/s)',
            template='plotly_white',
            height=380,
            legend=dict(x=0.01, y=0.99)
        )
        return fig

    @staticmethod
    def plot_volume_profile(simulator: WaterTankSimulator) -> go.Figure:
        """Plot profil volume air"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=simulator.time_history,
            y=simulator.volume_history,
            mode='lines',
            name='Volume Air',
            line=dict(color='#9467bd', width=3),
            fill='tozeroy',
            fillcolor='rgba(148,103,189,0.15)'
        ))

        fig.add_hline(
            y=simulator.config.tank_volume,
            line_dash='dash', line_color='red',
            annotation_text=f'Volume Maksimum ({simulator.config.tank_volume:.2f} m³)'
        )

        fig.update_layout(
            title='Profil Volume Air dalam Tangki',
            xaxis_title='Waktu (menit)',
            yaxis_title='Volume Air (m³)',
            yaxis=dict(range=[0, simulator.config.tank_volume * 1.1]),
            template='plotly_white',
            height=380,
        )
        return fig

    @staticmethod
    def plot_phase_analysis(configs_results: list) -> go.Figure:
        """
        Plot analisis ukuran tangki optimal:
        configs_results = list of (label, metrics_dict)
        """
        labels        = [r[0] for r in configs_results]
        times_full    = [r[1]['time_full']  if r[1]['time_full']  else 0 for r in configs_results]
        times_empty   = [r[1]['time_empty'] if r[1]['time_empty'] else 0 for r in configs_results]
        volumes       = [r[1]['tank_volume'] for r in configs_results]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Waktu Pengisian vs Volume Tangki',
                            'Waktu Pengosongan vs Volume Tangki']
        )

        fig.add_trace(go.Bar(
            x=labels, y=times_full,
            name='Waktu Pengisian (menit)',
            marker_color='steelblue'
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=labels, y=times_empty,
            name='Waktu Pengosongan (menit)',
            marker_color='coral'
        ), row=1, col=2)

        fig.update_layout(
            title='Analisis Ukuran Tangki Optimal',
            template='plotly_white',
            height=420,
            showlegend=False
        )
        return fig


# ====================
# 6. STREAMLIT UI
# ====================

def main():
    st.set_page_config(
        page_title='Simulasi Tangki Air – MODSIM 2026 P4',
        page_icon='💧',
        layout='wide'
    )

    # ── Header ──────────────────────────────────────────────
    st.title('💧 Simulasi Sistem Distribusi Air Asrama')
    st.caption('[11S1221] Pemodelan dan Simulasi – Modul Praktikum 4: Continuous Simulation | Studi Kasus')

    st.markdown("""
    **Permasalahan:** Dalam sistem distribusi air di asrama, digunakan tangki air (pam) silindris
    yang diisi melalui pipa inlet dan dikosongkan melalui pipa outlet. Simulasi ini menjawab:
    - Berapa lama waktu yang dibutuhkan untuk **mengisi** tangki hingga penuh?
    - Berapa lama waktu untuk **mengosongkan** tangki?
    - Bagaimana **profil ketinggian air** berubah terhadap waktu?
    - Bagaimana jika **pengisian dan pengosongan terjadi bersamaan**?
    - Bagaimana menentukan **ukuran tangki optimal**?
    """)

    st.divider()

    # ── Sidebar – Parameter ─────────────────────────────────
    st.sidebar.header('⚙️ Parameter Simulasi')

    st.sidebar.subheader('Dimensi Tangki')
    tank_radius   = st.sidebar.slider('Jari-jari Tangki (m)',      0.5, 3.0, 1.0, 0.1)
    tank_height   = st.sidebar.slider('Tinggi Maksimum Tangki (m)', 1.0, 5.0, 2.0, 0.1)
    initial_h     = st.sidebar.slider('Ketinggian Awal Air (m)',    0.0, tank_height, 0.0, 0.1)

    st.sidebar.subheader('Pipa Inlet (Air Masuk)')
    inlet_active  = st.sidebar.checkbox('Aktifkan Inlet', value=True)
    inlet_flow    = st.sidebar.slider('Laju Aliran Inlet (L/s)', 0.5, 20.0, 5.0, 0.5)

    st.sidebar.subheader('Pipa Outlet (Air Keluar)')
    outlet_active = st.sidebar.checkbox('Aktifkan Outlet', value=True)
    outlet_diam   = st.sidebar.slider('Diameter Pipa Outlet (cm)', 1.0, 15.0, 5.0, 0.5)
    outlet_cd     = st.sidebar.slider('Koefisien Debit (Cd)',      0.3, 0.9, 0.6, 0.05)

    st.sidebar.subheader('Parameter Simulasi')
    sim_time      = st.sidebar.slider('Durasi Simulasi (menit)', 10, 300, 120, 10)

    # ── Build Config ─────────────────────────────────────────
    config = TankConfig(
        tank_radius         = tank_radius,
        tank_height_max     = tank_height,
        inlet_flow_rate     = inlet_flow / 1000.0,    # L/s → m³/s
        inlet_active        = inlet_active,
        outlet_diameter     = outlet_diam / 100.0,    # cm → m
        outlet_discharge_coeff = outlet_cd,
        outlet_active       = outlet_active,
        initial_height      = min(initial_h, tank_height),
        simulation_time     = sim_time,
        time_step           = 1.0,
    )

    # ── Info Tangki ──────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric('Volume Maksimum Tangki',
              f'{config.tank_volume:.2f} m³',
              f'{config.tank_volume * 1000:.0f} Liter')
    c2.metric('Luas Penampang Tangki',
              f'{config.tank_area:.3f} m²',
              f'r = {tank_radius} m')
    c3.metric('Ketinggian Awal',
              f'{initial_h:.1f} m',
              f'{(initial_h/tank_height)*100:.0f}% kapasitas')

    # ── Jalankan Simulasi ───────────────────────────────────
    if st.button('▶ Jalankan Simulasi', type='primary', use_container_width=True):
        st.session_state['sim_run'] = True
        with st.spinner('Menjalankan simulasi...'):
            simulator = WaterTankSimulator(config)
            results   = simulator.run_simulation()
        st.session_state['simulator'] = simulator
        st.session_state['results']   = results
    else:
        # Auto-run on first load
        if 'simulator' not in st.session_state:
            simulator = WaterTankSimulator(config)
            results   = simulator.run_simulation()
            st.session_state['simulator'] = simulator
            st.session_state['results']   = results

    simulator = st.session_state.get('simulator')
    results   = st.session_state.get('results')

    if simulator is None:
        st.info('Klik "Jalankan Simulasi" untuk memulai.')
        return

    # ── Metrics ──────────────────────────────────────────────
    st.subheader('📊 Hasil Simulasi')

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        if results['time_full'] is not None:
            st.metric('⏱ Waktu Pengisian Penuh',
                      f"{results['time_full']:.1f} menit")
        else:
            st.metric('⏱ Waktu Pengisian Penuh', 'Belum penuh')
    with m2:
        if results['time_empty'] is not None:
            st.metric('⏱ Waktu Pengosongan',
                      f"{results['time_empty']:.1f} menit")
        else:
            st.metric('⏱ Waktu Pengosongan', 'Belum kosong')
    with m3:
        st.metric('📏 Ketinggian Akhir',
                  f"{results['final_height']:.3f} m",
                  f"{results['fill_percent']:.1f}%")
    with m4:
        st.metric('💧 Volume Akhir',
                  f"{results['final_volume']:.3f} m³",
                  f"{results['final_volume']*1000:.0f} L")

    m5, m6, m7, m8 = st.columns(4)
    with m5:
        st.metric('↓ Volume Air Masuk',
                  f"{results['volume_in']:.3f} m³",
                  f"{results['volume_in']*1000:.0f} L")
    with m6:
        st.metric('↑ Volume Air Keluar',
                  f"{results['volume_out']:.3f} m³",
                  f"{results['volume_out']*1000:.0f} L")
    with m7:
        st.metric('🔴 Q_out Maksimum',
                  f"{results['max_outlet_flow']:.3f} L/s")
    with m8:
        st.metric('🔵 Q_out Rata-rata',
                  f"{results['avg_outlet_flow']:.3f} L/s")

    # ── Plot Utama ────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(['📈 Profil Ketinggian',
                                 '🔄 Laju Aliran',
                                 '📦 Profil Volume'])

    with tab1:
        st.plotly_chart(
            TankVisualization.plot_height_profile(simulator),
            use_container_width=True
        )
        st.info(
            '**Interpretasi:** Grafik ini menunjukkan bagaimana ketinggian air berubah '
            'terhadap waktu. Kenaikan berarti laju inlet > laju outlet; '
            'penurunan berarti sebaliknya. Garis merah putus-putus adalah batas penuh tangki.'
        )

    with tab2:
        st.plotly_chart(
            TankVisualization.plot_flow_rates(simulator),
            use_container_width=True
        )
        st.info(
            '**Interpretasi:** Q_out mengikuti Hukum Torricelli – '
            'semakin tinggi air, semakin besar tekanan hidrostatik, '
            'sehingga laju aliran keluar semakin besar. '
            'Q_in konstan selama inlet aktif.'
        )

    with tab3:
        st.plotly_chart(
            TankVisualization.plot_volume_profile(simulator),
            use_container_width=True
        )

    # ── Analisis Multi-Skenario ───────────────────────────────
    st.divider()
    st.subheader('🔬 Analisis Skenario & Ukuran Tangki Optimal')

    scenario_tab1, scenario_tab2 = st.tabs(
        ['Perbandingan Skenario Operasi', 'Analisis Ukuran Tangki Optimal']
    )

    with scenario_tab1:
        st.markdown("**Simulasi 3 Skenario: Pengisian Saja | Pengosongan Saja | Simultan**")

        # Scenario 1: Filling only
        cfg_fill = config.copy()
        cfg_fill.inlet_active   = True
        cfg_fill.outlet_active  = False
        cfg_fill.initial_height = 0.0
        sim_fill = WaterTankSimulator(cfg_fill)
        sim_fill.run_simulation()

        # Scenario 2: Emptying only
        cfg_empty = config.copy()
        cfg_empty.inlet_active   = False
        cfg_empty.outlet_active  = True
        cfg_empty.initial_height = tank_height
        sim_empty = WaterTankSimulator(cfg_empty)
        sim_empty.run_simulation()

        # Scenario 3: Simultaneous
        cfg_both = config.copy()
        cfg_both.inlet_active   = True
        cfg_both.outlet_active  = True
        cfg_both.initial_height = tank_height / 2
        sim_both = WaterTankSimulator(cfg_both)
        sim_both.run_simulation()

        # Plot comparison
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(
            x=sim_fill.time_history,  y=sim_fill.height_history,
            name='Skenario 1: Pengisian Saja',
            line=dict(color='green', width=2.5)
        ))
        fig_comp.add_trace(go.Scatter(
            x=sim_empty.time_history, y=sim_empty.height_history,
            name='Skenario 2: Pengosongan Saja',
            line=dict(color='red', width=2.5)
        ))
        fig_comp.add_trace(go.Scatter(
            x=sim_both.time_history,  y=sim_both.height_history,
            name='Skenario 3: Pengisian + Pengosongan',
            line=dict(color='blue', width=2.5)
        ))
        fig_comp.add_hline(y=tank_height, line_dash='dash',
                           line_color='black', opacity=0.4,
                           annotation_text='Kapasitas Penuh')
        fig_comp.update_layout(
            title='Perbandingan Profil Ketinggian Air – 3 Skenario',
            xaxis_title='Waktu (menit)',
            yaxis_title='Ketinggian Air (m)',
            template='plotly_white',
            height=450,
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Summary table
        r_fill  = sim_fill.results
        r_empty = sim_empty.results
        r_both  = sim_both.results

        df_summary = pd.DataFrame({
            'Skenario': [
                'Pengisian Saja',
                'Pengosongan Saja',
                'Pengisian + Pengosongan'
            ],
            'Kondisi Awal (m)': [0.0, tank_height, tank_height / 2],
            'Ketinggian Akhir (m)': [
                f"{r_fill['final_height']:.3f}",
                f"{r_empty['final_height']:.3f}",
                f"{r_both['final_height']:.3f}",
            ],
            'Waktu Penuh (menit)': [
                f"{r_fill['time_full']:.1f}"  if r_fill['time_full']  else '-',
                '-',
                f"{r_both['time_full']:.1f}"  if r_both['time_full']  else '-',
            ],
            'Waktu Kosong (menit)': [
                '-',
                f"{r_empty['time_empty']:.1f}" if r_empty['time_empty'] else '-',
                f"{r_both['time_empty']:.1f}"  if r_both['time_empty']  else '-',
            ],
        })
        st.dataframe(df_summary, hide_index=True, use_container_width=True)

    with scenario_tab2:
        st.markdown(
            "**Perbandingan berbagai ukuran tangki** untuk menentukan dimensi optimal "
            "berdasarkan waktu pengisian dan pengosongan."
        )
        radii      = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        opt_results = []
        for r in radii:
            cfg_opt = config.copy()
            cfg_opt.tank_radius      = r
            cfg_opt.inlet_active     = True
            cfg_opt.outlet_active    = True
            cfg_opt.initial_height   = 0.0
            cfg_opt.simulation_time  = 300.0
            sim_opt = WaterTankSimulator(cfg_opt)
            m_opt = sim_opt.run_simulation()
            opt_results.append((f'r={r}m', m_opt))

        fig_opt = TankVisualization.plot_phase_analysis(opt_results)
        st.plotly_chart(fig_opt, use_container_width=True)

        df_opt = pd.DataFrame([{
            'Jari-jari (m)':         r[0],
            'Volume Maks (m³)':      f"{r[1]['tank_volume']:.2f}",
            'Waktu Pengisian (mnt)': f"{r[1]['time_full']:.1f}"  if r[1]['time_full']  else '>300',
            'Vol Air Masuk (m³)':    f"{r[1]['volume_in']:.3f}",
            'Vol Air Keluar (m³)':   f"{r[1]['volume_out']:.3f}",
        } for r in opt_results])
        st.dataframe(df_opt, hide_index=True, use_container_width=True)
        st.info(
            '💡 **Kesimpulan Optimasi:** Tangki yang lebih besar menyimpan lebih banyak air '
            'namun membutuhkan waktu pengisian yang lebih lama. '
            'Pilih ukuran tangki di mana waktu pengisian ≤ jam non-puncak '
            'dan volume cukup untuk memenuhi kebutuhan harian.'
        )

    # ── Model Matematika ──────────────────────────────────────
    st.divider()
    st.subheader('📐 Model Matematika')

    col_eq1, col_eq2 = st.columns(2)
    with col_eq1:
        st.markdown(r"""
**Persamaan Diferensial Utama:**

$$\frac{dh}{dt} = \frac{Q_{in} - Q_{out}}{A_{tangki}}$$

**Laju Aliran Keluar (Hukum Torricelli):**

$$Q_{out} = C_d \cdot A_{outlet} \cdot \sqrt{2gh}$$

**Laju Aliran Masuk:**

$$Q_{in} = \text{konstan (pompa aktif)}$$
        """)
    with col_eq2:
        st.markdown(r"""
**Keterangan Simbol:**

| Simbol | Keterangan | Satuan |
|---|---|---|
| $h$ | Ketinggian air | m |
| $A_{tangki}$ | Luas penampang tangki | m² |
| $Q_{in}$ | Laju aliran masuk | m³/s |
| $Q_{out}$ | Laju aliran keluar | m³/s |
| $C_d$ | Koefisien debit | - |
| $g$ | Percepatan gravitasi | m/s² |
| $A_{outlet}$ | Luas penampang outlet | m² |
        """)

    # ── Data Tabel ────────────────────────────────────────────
    st.divider()
    st.subheader('📋 Data Simulasi')

    df_data = pd.DataFrame({
        'Waktu (menit)':      np.round(simulator.time_history[::60], 2),
        'Ketinggian (m)':     np.round(simulator.height_history[::60], 4),
        'Volume (m³)':        np.round(simulator.volume_history[::60], 4),
        'Q_in (L/s)':         np.round(simulator.flow_in_history[::60], 4),
        'Q_out (L/s)':        np.round(simulator.flow_out_history[::60], 4),
    })
    st.dataframe(df_data, use_container_width=True, height=300)

    csv = df_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        '⬇ Unduh Data CSV',
        data=csv,
        file_name='simulasi_tangki_air.csv',
        mime='text/csv'
    )

    # ── Footer ────────────────────────────────────────────────
    st.divider()
    st.caption(
        'Dibuat untuk **Modul Praktikum 4 – Studi Kasus** | '
        '[11S1221] Pemodelan dan Simulasi (MODSIM) 2026 | '
        'Continuous Simulation – Water Tank Distribution System'
    )


if __name__ == '__main__':
    main()
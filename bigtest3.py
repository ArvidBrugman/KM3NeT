import numpy as np
import plotly.graph_objects as go

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from scipy.interpolate import RegularGridInterpolator

# =====================
# PARAMETERS
# =====================
distance_between_points = 1000
max_radius = 3300
max_height = 3000
z_step = 1000

r_particle = 1500
theta_particle = np.pi / 3
z_particle = 1000

c_sound = 1500
v_track = 3000

# =====================
# BUILD DETECTOR
# =====================
def build_detector_positions():
    radii = np.arange(0, max_radius, distance_between_points)
    z_layers = np.arange(0, max_height + 1, z_step)

    positions = []
    lines = []

    for r in radii:
        if r == 0:
            theta = [0]
        else:
            points = int(2 * np.pi * r / distance_between_points)
            theta = np.linspace(0, 2*np.pi, points, endpoint=False)

        for t in theta:
            x = r * np.cos(t)
            y = r * np.sin(t)

            line = []
            for z in z_layers:
                positions.append([x, y, z])
                line.append([x, y, z])

            lines.append(np.array(line))

    return np.array(positions), lines, len(lines), len(z_layers)

# =====================
# LOAD DATA
# =====================
def load_pulse_interpolator():
    data = np.load("pulses.npz")

    return RegularGridInterpolator(
        (data["R"], data["Z"]),
        data["signal"],
        bounds_error=False,
        fill_value=0
    ), data["t"], data["R"], data["Z"]

# =====================
# COMPUTE
# =====================
def compute_all(detector_positions, source_pos, direction, interpolator, t_vals, R_vals, Z_vals):

    detector_data = []
    amplitudes = []
    arrival_times = []

    for i, pos in enumerate(detector_positions):

        r_vec = pos - source_pos

        Z = np.dot(r_vec, direction)
        r_perp = r_vec - Z * direction
        R = np.linalg.norm(r_perp)

        t_parallel = Z / v_track
        t_perp = R / c_sound
        arrival = t_parallel + t_perp

        if (R < R_vals.min() or R > R_vals.max() or
            Z < Z_vals.min() or Z > Z_vals.max()):
            signal = np.zeros_like(t_vals)
        else:
            signal = interpolator([[R, Z]])[0]

        t_shifted = t_vals + arrival
        amp = np.max(signal) - np.min(signal)

        detector_data.append({
            "id": i,
            "pos": pos,
            "R": R,
            "Z": Z,
            "time": t_shifted,
            "signal": signal,
            "arrival": arrival
        })

        amplitudes.append(amp)
        arrival_times.append(arrival)

    return detector_data, np.array(amplitudes), np.array(arrival_times)

# =====================
# INIT
# =====================
detector_positions, detector_lines, line_count, detectors_per_line = build_detector_positions()

detector_count = len(detector_positions)
total_cable_km = (line_count * max_height) / 1000

x0 = r_particle * np.cos(theta_particle)
y0 = r_particle * np.sin(theta_particle)
z0 = z_particle
source_pos = np.array([x0, y0, z0])

interp, t_vals, R_vals, Z_vals = load_pulse_interpolator()

dummy_dir = np.array([0,0,1])
_, _, arrival_init = compute_all(
    detector_positions, source_pos, dummy_dir,
    interp, t_vals, R_vals, Z_vals
)

t_max_ms = np.max(arrival_init) * 1e3

# =====================
# DASH APP
# =====================
app = dash.Dash(__name__)

def pi_marks(max_val):
    return {i/4: f"{i}/4π" if i != 0 else "0" for i in range(int(max_val*4)+1)}

app.layout = html.Div([

    html.Div("Theta_dir (×π)"),
    dcc.Slider(id='theta-dir', min=0, max=1, step=0.05, value=0, marks=pi_marks(1)),

    html.Div("Phi_dir (×π)"),
    dcc.Slider(id='phi-dir', min=0, max=2, step=0.05, value=0, marks=pi_marks(2)),

    html.Div("Time [ms]"),
    dcc.Slider(id='time-slider', min=0, max=t_max_ms, step=t_max_ms/100, value=0),

    html.Div([
        html.Div(id="info-panel", style={'flex': '1', 'padding': '20px'}),
        html.Div([dcc.Graph(id="3d-plot", style={"height": "600px"})], style={'flex': '3'})
    ], style={'display': 'flex'}),

    dcc.Graph(id="waveform")

])

# =====================
# CALLBACK
# =====================
@app.callback(
    [Output("3d-plot", "figure"),
     Output("waveform", "figure"),
     Output("info-panel", "children")],
    [Input("time-slider", "value"),
     Input("theta-dir", "value"),
     Input("phi-dir", "value"),
     Input("3d-plot", "clickData")]
)
def update_all(t_current_ms, theta_slider, phi_slider, clickData):

    theta_dir = theta_slider * np.pi
    phi_dir = phi_slider * np.pi

    direction = np.array([
        np.sin(theta_dir) * np.cos(phi_dir),
        np.sin(theta_dir) * np.sin(phi_dir),
        np.cos(theta_dir)
    ])

    detector_data, amplitudes, arrival_times = compute_all(
        detector_positions,
        source_pos,
        direction,
        interp,
        t_vals,
        R_vals,
        Z_vals
    )

    amps = amplitudes * 1000
    t_current = t_current_ms / 1000
    active = arrival_times < t_current

    fig3d = go.Figure()

    for line in detector_lines:
        fig3d.add_trace(go.Scatter3d(
            x=line[:,0], y=line[:,1], z=line[:,2],
            mode='lines',
            line=dict(color='black'),
            showlegend=False
        ))

    fig3d.add_trace(go.Scatter3d(
        x=detector_positions[:,0],
        y=detector_positions[:,1],
        z=detector_positions[:,2],
        mode='markers',
        marker=dict(
            size=6,
            color=np.where(active, amps, 0),
            colorscale='Plasma',
            cmin=0,
            cmax=amps.max(),
            colorbar=dict(title="Signal [mPa]")
        ),
        name="Detectors",
        customdata=np.arange(len(detector_positions))
    ))

    selected_idx = None
    try:
        selected_idx = clickData["points"][0]["customdata"]
    except (TypeError, KeyError):
        pass

    if selected_idx is not None:
        pos = detector_positions[selected_idx]
        fig3d.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers',
            marker=dict(size=10, color='lime'),
            name="Selected detector"
        ))

    fig3d.add_trace(go.Scatter3d(
        x=[x0], y=[y0], z=[z0],
        mode='markers',
        marker=dict(size=10, color='red'),
        name="Neutrino"
    ))

    scale = 2000
    fig3d.add_trace(go.Scatter3d(
        x=[x0, x0 + direction[0]*scale],
        y=[y0, y0 + direction[1]*scale],
        z=[z0, z0 + direction[2]*scale],
        mode='lines',
        line=dict(color='red', width=5),
        name="Direction"
    ))

    # 🔥 FIX legenda positie
    fig3d.update_layout(
        scene=dict(
            xaxis=dict(range=[-3500, 3500]),
            yaxis=dict(range=[-3500, 3500]),
            zaxis=dict(range=[0, 3500]),
            aspectmode='cube'
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255,255,255,0.6)'
        )
    )

    base_info = [
        html.H3("Detector Info"),
        html.P(f"Total detectors: {detector_count}"),
        html.P(f"Lines: {line_count}"),
        html.P(f"Detectors/line: {detectors_per_line}"),
        html.P(f"Cable length: {total_cable_km:.1f} km"),
    ]

    if selected_idx is None:
        return fig3d, go.Figure(), base_info

    d = detector_data[selected_idx]

    arrival = d["arrival"] * 1e3
    t = (d["time"] * 1e3) - arrival
    s = d["signal"] * 1e3

    peak_idx = np.argmax(np.abs(s))
    peak_val = s[peak_idx]
    peak_time = t[peak_idx]

    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(x=t, y=s, name="Signal"))

    fig_wave.add_trace(go.Scatter(
        x=[0,0],
        y=[-1.2*np.max(np.abs(s)), 1.2*np.max(np.abs(s))],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name="Arrival"
    ))

    fig_wave.add_trace(go.Scatter(
        x=[peak_time, peak_time],
        y=[-1.2*np.max(np.abs(s)), 1.2*np.max(np.abs(s))],
        mode='lines',
        line=dict(color='green', dash='dot'),
        name="Peak"
    ))

    fig_wave.add_trace(go.Scatter(
        x=[peak_time], y=[peak_val],
        mode='markers',
        marker=dict(color='green', size=8),
        name="Peak value"
    ))

    # 🔥 FIX waveform zoom
    mask = np.abs(s) > 0.05 * np.max(np.abs(s))

    if np.any(mask):
        t_min = t[mask].min()
        t_max = t[mask].max()
    else:
        t_min, t_max = -0.05, 0.05

    window = max(abs(t_min), abs(t_max))
    window = min(window, 0.5)

    ymax = np.max(np.abs(s))

    fig_wave.update_layout(
        title=f"Detector {selected_idx}",
        xaxis=dict(range=[-window-0.05, window+0.05], title="Time relative to arrival [ms]"),
        yaxis=dict(range=[-1.3*ymax, 1.3*ymax], title="Amplitude [mPa]"),
        legend=dict(x=0.01, y=0.99)
    )

    extra_info = base_info + [
        html.Hr(),
        html.H4(f"Selected detector {selected_idx}"),
        html.P(f"R = {d['R']:.1f} m"),
        html.P(f"Z = {d['Z']:.1f} m"),
        html.P(f"Arrival = {arrival:.2f} ms"),
        html.P(f"Peak amplitude = {peak_val:.2f} mPa"),
        html.P(f"Peak time = {peak_time:.2f} ms"),
    ]

    return fig3d, fig_wave, extra_info

# =====================
# RUN
# =====================
if __name__ == "__main__":
    app.run(debug=True)
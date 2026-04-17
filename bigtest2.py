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
def compute_all(detector_positions, source_pos, interpolator, t_vals, R_vals, Z_vals):

    detector_data = []
    amplitudes = []
    arrival_times = []

    for i, pos in enumerate(detector_positions):

        r_vec = pos - source_pos

        arrival = np.linalg.norm(r_vec) / c_sound

        R = np.linalg.norm(r_vec[:2])
        Z = r_vec[2]

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
# PRECOMPUTE
# =====================
detector_positions, detector_lines, line_count, detectors_per_line = build_detector_positions()

detector_count = len(detector_positions)
total_cable_km = (line_count * max_height) / 1000

x0 = r_particle * np.cos(theta_particle)
y0 = r_particle * np.sin(theta_particle)
z0 = z_particle
source_pos = np.array([x0, y0, z0])

interp, t_vals, R_vals, Z_vals = load_pulse_interpolator()

detector_data, amplitudes, arrival_times = compute_all(
    detector_positions,
    source_pos,
    interp,
    t_vals,
    R_vals,
    Z_vals
)

amps = amplitudes * 1000

# =====================
# SLIDER
# =====================
t_max_ms = np.ceil(arrival_times.max() * 1e3)
marks = {int(i): f"{int(i)}" for i in range(int(t_max_ms)+1)}

# =====================
# DASH APP
# =====================
app = dash.Dash(__name__)

app.layout = html.Div([

    html.Div(style={'height': '30px'}),

    dcc.Slider(
        id='time-slider',
        min=0,
        max=t_max_ms,
        value=0,
        step=0.1,
        marks=marks,
        tooltip={"always_visible": True}
    ),

    html.Div("Time [ms]", style={'textAlign': 'center'}),

    html.Div([

        html.Div(id="info-panel", style={
            'flex': '1',
            'padding': '20px',
            'backgroundColor': '#f8f8f8',
            'height': '600px',
            'overflowY': 'auto'
        }),

        html.Div([
            dcc.Graph(id="3d-plot", style={"height": "600px"})
        ], style={'flex': '3'})

    ], style={'display': 'flex', 'gap': '20px'}),

    html.Div([
        dcc.Graph(id="waveform")
    ], style={'marginTop': '20px'})

])

# =====================
# 3D UPDATE
# =====================
@app.callback(
    Output("3d-plot", "figure"),
    [Input("time-slider", "value"),
     Input("3d-plot", "clickData")]
)
def update_3d(t_current_ms, clickData):

    t_current = t_current_ms / 1000
    active = arrival_times < t_current

    selected_idx = None
    if clickData is not None:
        selected_idx = clickData["points"][0]["customdata"]

    fig = go.Figure()

    # lijnen
    for line in detector_lines:
        fig.add_trace(go.Scatter3d(
            x=line[:,0],
            y=line[:,1],
            z=line[:,2],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ))

    # detectors
    fig.add_trace(go.Scatter3d(
        x=detector_positions[:,0],
        y=detector_positions[:,1],
        z=detector_positions[:,2],
        mode='markers',
        marker=dict(
            size=6,
            color=np.where(active, amps, 0),
            colorscale='Plasma',
            colorbar=dict(title="Signal [mPa]", x=0.85),
            cmin=0,
            cmax=amps.max()
        ),
        customdata=np.arange(len(detector_data)),
        name="Detectors"
    ))

    # 🔥 SELECTED DETECTOR HIGHLIGHT
    if selected_idx is not None:
        pos = detector_positions[selected_idx]

        fig.add_trace(go.Scatter3d(
            x=[pos[0]],
            y=[pos[1]],
            z=[pos[2]],
            mode='markers',
            marker=dict(size=10, color='lime'),
            name="Selected detector"
        ))

    # neutrino
    fig.add_trace(go.Scatter3d(
        x=[x0], y=[y0], z=[z0],
        mode='markers',
        marker=dict(size=10, color='red'),
        name="Neutrino"
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis=dict(range=[-3500, 3500]),
            yaxis=dict(range=[-3500, 3500]),
            zaxis=dict(range=[0, 3500]),
            aspectmode='cube'
        )
    )

    return fig

# =====================
# INFO + WAVEFORM
# =====================
@app.callback(
    [Output("waveform", "figure"),
     Output("info-panel", "children")],
    Input("3d-plot", "clickData")
)
def update_info(clickData):

    base_info = [
        html.H3("Detector Info"),
        html.P(f"Total detectors: {detector_count}"),
        html.P(f"Lines: {line_count}"),
        html.P(f"Detectors/line: {detectors_per_line}"),
        html.P(f"Cable length: {total_cable_km:.1f} km"),
    ]

    if clickData is None:
        return go.Figure(), base_info

    idx = clickData["points"][0]["customdata"]
    d = detector_data[idx]

    t = d["time"] * 1e3
    s = d["signal"] * 1e3

    peak_idx = np.argmax(np.abs(s))
    peak_val = s[peak_idx]
    peak_time = t[peak_idx]

    arrival = d["arrival"] * 1e3

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=s, name="Signal"))

    fig.add_trace(go.Scatter(
        x=[arrival, arrival], y=[-10, 10],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Arrival time'
    ))

    fig.add_trace(go.Scatter(
        x=[peak_time, peak_time], y=[-10, 10],
        mode='lines',
        line=dict(color='green', dash='dot'),
        name='Signal peak'
    ))

    fig.add_trace(go.Scatter(
        x=[peak_time],
        y=[peak_val],
        mode='markers',
        marker=dict(size=8, color='green'),
        name='Peak value'
    ))

    mask = np.abs(s) > 0.05 * np.max(np.abs(s))

    if np.any(mask):
        t_min = t[mask].min()
        t_max = t[mask].max()
    else:
        t_min, t_max = -1, 1

    ymax = np.max(np.abs(s))

    fig.update_layout(
        title=f"Detector {idx}",
        xaxis=dict(range=[t_min - 0.2, t_max + 0.2], title="Time [ms]"),
        yaxis=dict(range=[-1.3*ymax, 1.3*ymax], title="Amplitude [mPa]"),
        legend=dict(x=0.01, y=0.99)
    )

    extra_info = base_info + [
        html.Hr(),
        html.H4(f"Selected detector {idx}"),
        html.P(f"R = {d['R']:.1f} m"),
        html.P(f"Z = {d['Z']:.1f} m"),
        html.P(f"Arrival = {arrival:.2f} ms"),
        html.P(f"Peak amplitude = {peak_val:.2f} mPa"),
        html.P(f"Peak time = {peak_time:.2f} ms"),
    ]

    return fig, extra_info

# =====================
# RUN
# =====================
if __name__ == "__main__":
    app.run(debug=True)
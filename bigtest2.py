import numpy as np
import plotly.graph_objects as go

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from scipy.interpolate import RegularGridInterpolator

# =====================
# PARAMETERS
# =====================
# geometry of the detector
distance_between_points = 1000
max_radius = 3300
max_height = 3000
z_step = 1000

# position of the source (neutrino)
r_particle = 1500
theta_particle = np.pi / 3
z_particle = 1000

c_sound = 1500

# =====================
# BUILD DETECTOR
# =====================
def build_detector_positions():
    # makes rings of detectors at different radii  
    radii = np.arange(0, max_radius, distance_between_points)
    # makes layers of detectors at different heights
    z_layers = np.arange(0, max_height + 1, z_step)

    positions = []
    lines = []

    for r in radii:
        # special case for r=0 to avoid division by zero and create a single detector at the center
        if r == 0:
            theta = [0]
        else:
            # decides how many detectors on every ring/circle
            # = circumference / distance between points
            points = int(2 * np.pi * r / distance_between_points)
            # distributes the detectors evenly on the circle
            theta = np.linspace(0, 2*np.pi, points, endpoint=False)

        # coordinate transformation from cylindrical to cartesian
        for t in theta:
            x = r * np.cos(t)
            y = r * np.sin(t)

            line = []
            # makes vertical lines of detectors at each (x,y) position
            for z in z_layers:
                positions.append([x, y, z])
                line.append([x, y, z])

            lines.append(np.array(line))

    return np.array(positions), lines, len(lines), len(z_layers)

# =====================
# LOAD DATA
# =====================
def load_pulse_interpolator():
    # loades the .npz file
    data = np.load("pulses.npz")

    # creates an interpolator object that has input (R,Z) to get outputted the signal at that position
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

    # loop of all detectors to compute the signal at each position and time of arrival
    for i, pos in enumerate(detector_positions):
        # vector from source to detector
        r_vec = pos - source_pos

        # time of arrival = distance / speed of sound (in water)
        arrival = np.linalg.norm(r_vec) / c_sound

        # convert to cylindrical coordinates for interpolation
        R = np.linalg.norm(r_vec[:2])
        Z = r_vec[2]

        # if the detector is outside the precomputed grid, we set the signal to zero
        if (R < R_vals.min() or R > R_vals.max() or
            Z < Z_vals.min() or Z > Z_vals.max()):
            signal = np.zeros_like(t_vals)
        else:
            signal = interpolator([[R, Z]])[0]

        # shift the time axis so that t=0 corresponds to the arrival time of the signal at the detector
        t_shifted = t_vals + arrival

        # compute the amplitude as the peak-to-peak value of the signal
        amp = np.max(signal) - np.min(signal)

        # store all relevant data for this detector in a dictionary and append to the list
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
        
    # for every detector we have:
    # - distance to the source
    # - arrival time of the signal
    # - interpolated signal as a function of time
    return detector_data, np.array(amplitudes), np.array(arrival_times)

# =====================
# PRECOMPUTE
# =====================
# we precompute the signals on a grid of (R,Z) values and save it to a file
detector_positions, detector_lines, line_count, detectors_per_line = build_detector_positions()

# some summary info about the detector for the info panel in the app 
detector_count = len(detector_positions)
total_cable_km = (line_count * max_height) / 1000

# position of the source in cylindrical coordinates
x0 = r_particle * np.cos(theta_particle)
y0 = r_particle * np.sin(theta_particle)
z0 = z_particle

# position of the source (neutrino) in cartesian coordinates
source_pos = np.array([x0, y0, z0])

# we load the precomputed pulse data and create an interpolator function to get the signal at any (R,Z) position
interp, t_vals, R_vals, Z_vals = load_pulse_interpolator()

# we compute the signal at each detector position and the arrival times of the signals
detector_data, amplitudes, arrival_times = compute_all(
    detector_positions,
    source_pos,
    interp,
    t_vals,
    R_vals,
    Z_vals
)

# we convert the amplitudes to mPa for better visualization in the app
amps = amplitudes * 1000

# =====================
# SLIDER
# =====================
# slider settings to get nice tick marks at every ms
t_max_ms = np.ceil(arrival_times.max() * 1e3)
marks = {int(i): f"{int(i)}" for i in range(int(t_max_ms)+1)}

# =====================
# DASH APP
# =====================
# we create a Dash app to visualize the detector and the signals
app = dash.Dash(__name__)

# the layout consists of a slider to select the time, an info panel to show details about the selected detector,
# a 3D plot of the detector and a waveform plot for the selected detector
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
# this callback updates the 3D plot based on the selected time from the slider and highlights
# the selected detector when clicked
@app.callback(
    Output("3d-plot", "figure"),
    [Input("time-slider", "value"),
     Input("3d-plot", "clickData")]
)
def update_3d(t_current_ms, clickData):

    t_current = t_current_ms / 1000
    
    # a detector is active if the current time is greater than its arrival time (the signal has reached the detector)
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

    # highlight selected detector
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
# this callback updates the info panel and the waveform plot based on the selected detector in the 3D plot
@app.callback(
    [Output("waveform", "figure"),
     Output("info-panel", "children")],
    Input("3d-plot", "clickData")
)

# when a detector is clicked, we show the waveform of the signal at that detector and some extra info about it
def update_info(clickData):

    base_info = [
        html.H3("Detector Info"),
        html.P(f"Total detectors: {detector_count}"),
        html.P(f"Lines: {line_count}"),
        html.P(f"Detectors/line: {detectors_per_line}"),
        html.P(f"Cable length: {total_cable_km:.1f} km"),
    ]

    # if no detector is selected, we just show the general info and an empty plot
    if clickData is None:
        return go.Figure(), base_info

    # we get the index of the selected detector from the clickData and retrieve its data from the detector_data list
    idx = clickData["points"][0]["customdata"]
    d = detector_data[idx]

    # we convert the time to ms and the signal to mPa for better visualization
    t = d["time"] * 1e3
    s = d["signal"] * 1e3

    # we find the peak of the signal to show it in the plot and info panel
    peak_idx = np.argmax(np.abs(s))
    peak_val = s[peak_idx]
    peak_time = t[peak_idx]

    # we also convert the arrival time to ms for the info panel and to show it in the plot
    arrival = d["arrival"] * 1e3

    # we create the waveform plot with the signal, arrival time and peak highlighted
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=s, name="Signal"))

    # we add vertical lines for the arrival time and the peak time, and a marker for the peak value
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

    # we set the x-axis limits to show a window around the arrival time, and the y-axis limits to show the signal clearly
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
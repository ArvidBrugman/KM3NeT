import numpy as np
import plotly.graph_objects as go

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm 

# =====================
# PARAMETERS
# =====================
# detector geometry
distance_between_points = 1000
max_radius = 3300
max_height = 3000
z_step = 1000

# source geometry (cilindrical coordinates)
r_particle = 1500
theta_particle = np.pi / 3
z_particle = 1000

# speed of sound in water (m/s)
c_sound = 1500

# stacking selection (pancake), only detectors close to the source in z direction are used
z_threshold = 200

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
        if r == 0:
            theta = [0]
        else:
            # decides how many detectors on every ring/circle
            # = circumference / distance between points
            points = int(2 * np.pi * r / distance_between_points)
            # angles are evenly spaced around the circle
            theta = np.linspace(0, 2*np.pi, points, endpoint=False)

        # cilinderic to cartesian conversion
        for t in theta:
            x = r * np.cos(t)
            y = r * np.sin(t)

            # builds the vertical line of detectors at this (x,y) position
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
    # load precumputed pulses on a grid of (R,Z) values, and create an interpolator function that we can call later for each detector position
    data = np.load("pulses.npz")

    return RegularGridInterpolator(
        (data["R"], data["Z"]),
        data["signal"],
        bounds_error=False,
        fill_value=0
    ), data["t"], data["R"], data["Z"]

# we do the same for noise, so we can add realistic noise to the signal for each detector
def load_noise_interpolator():
    data = np.load("noise.npz")

    return RegularGridInterpolator(
        (data["R"], data["Z"]),
        data["noise"],
        bounds_error=False,
        fill_value=0
    )

# =====================
# COMPUTE
# =====================
def compute_all(detector_positions, source_pos, interpolator, noise_interp, t_vals, R_vals, Z_vals):

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

        # if the detector is outside the precomputed grid (grid made in the pulses.npz file), we set the signal to zero
        if (R < R_vals.min() or R > R_vals.max() or
            Z < Z_vals.min() or Z > Z_vals.max()):
            signal = np.zeros_like(t_vals)
            noise = np.zeros_like(t_vals)
        # otherwise we interpolate the signal and noise for this (R,Z) position
        else:
            signal = interpolator([[R, Z]])[0]
            noise = noise_interp([[R, Z]])[0]

        # shift the time axis so that t=0 corresponds to the arrival time of the signal at the detector
        t_shifted = t_vals + arrival

        # compute the ampltude as the peak-to-peak value of the signal (without noise)
        amp = np.max(signal) - np.min(signal)

        detector_data.append({
            "id": i,
            "pos": pos,
            "R": R,
            "Z": Z,
            "time": t_shifted,
            "signal": signal,
            "noise": noise,
            "arrival": arrival
        })

        amplitudes.append(amp)
        arrival_times.append(arrival)

    return detector_data, np.array(amplitudes), np.array(arrival_times)

# =====================
# PRECOMPUTE
# =====================
# we precompute the signal and noise on a grid of (R,Z) values, so we can interpolate later for each detector position
detector_positions, detector_lines, line_count, detectors_per_line = build_detector_positions()

# some summary info about the detector for the info panel in the app
detector_count = len(detector_positions)
total_cable_km = (line_count * max_height) / 1000

# convert source position from cylindrical to cartesian coordinates
x0 = r_particle * np.cos(theta_particle)
y0 = r_particle * np.sin(theta_particle)
z0 = z_particle

source_pos = np.array([x0, y0, z0])

# load the interpolators for signal and noise, and compute the signal, noise, arrival times and amplitudes for all detectors
interp, t_vals, R_vals, Z_vals = load_pulse_interpolator()
noise_interp = load_noise_interpolator()

# we compute the signal, noise, arrival times and amplitudes for all detectors
detector_data, amplitudes, arrival_times = compute_all(
    detector_positions,
    source_pos,
    interp,
    noise_interp,
    t_vals,
    R_vals,
    Z_vals
)

# we convert the amplitudes to mPa for better visualization in the app
amps = amplitudes * 1000
# size of the time step (we assume uniform)
dt = t_vals[1] - t_vals[0]

# =====================
# SLIDER
# =====================
# states the maximum time for the slider based on the maximum arrival time of the signal at the detectors, rounded up to the nearest ms for nice slider marks
t_max_ms = np.ceil(arrival_times.max() * 1e3)
# we create marks for every ms, but only show the label for every ...(100) ms to avoid clutter
marks = {int(i): f"{int(i)}" for i in range(int(t_max_ms)+1)}

# =====================
# DASH APP
# =====================
# starts the dash app
app = dash.Dash(__name__)

# define the layout of the app
app.layout = html.Div([

    # little spacing on top
    html.Div(style={'height': '20px'}),

    # slider to select the current time in the animation, which will update the 3D plot and the info panel
    dcc.Slider(
        id='time-slider',
        min=0,
        max=t_max_ms,
        value=0,
        step=0.1,
        marks=marks,
        tooltip={"always_visible": True}
    ),

    # label for the slider
    html.Div("Time [ms]", style={'textAlign': 'center'}),

    # main content: 3D plot on the right, info panel on the left, and below the 3 plots for the selected detector (waveform, noise histogram and stack plot)
    html.Div([

        # info panel with summary info about the detector and details about the selected detector, will be updated when clicking on a detector in the 3D plot
        html.Div(id="info-panel", style={
            'flex': '1',
            'padding': '20px',
            'backgroundColor': '#f8f8f8',
            'height': '600px',
            'overflowY': 'auto'
        }),

        # 3D plot of the detector array, where the color of the detectors indicates the signal amplitude at the current time (from the slider), and clicking on a detector will show more details in the info panel and the plots below
        html.Div([
            dcc.Graph(id="3d-plot", style={"height": "600px"})
        ], style={'flex': '3'})

    ], style={'display': 'flex', 'gap': '20px'}),

    # spacing between 3D plot and the plots below
    html.Div([
        dcc.Graph(id="waveform"),
        dcc.Graph(id="noise-hist"),
        dcc.Graph(id="stack-plot")
    ], style={'marginTop': '20px'})

])

# =====================
# 3D UPDATE
# =====================
# this callback updates the 3D plot based on the current time from the slider, and also highlights the selected detector when clicking on it
@app.callback(
    Output("3d-plot", "figure"),
    [Input("time-slider", "value"),
     Input("3d-plot", "clickData")]
)

# this function updates the 3D plot based on the current time from the slider, and also highlights the selected detector when clicking on it
def update_3d(t_current_ms, clickData):

    # coverts time from ms to seconds for comparison with arrival times
    t_current = t_current_ms / 1000
    # active = detectors that have received the signal by the current time
    active = arrival_times < t_current

    # if a detector is clicked, we get its index from the clickData to highlight it in the plot and show its details in the info panel
    selected_idx = None
    if clickData is not None:
        selected_idx = clickData["points"][0]["customdata"]

    # create figure
    fig = go.Figure()

    # we add the lines of detectors to the plot for better visualization of the geometry
    for line in detector_lines:
        fig.add_trace(go.Scatter3d(
            x=line[:,0], y=line[:,1], z=line[:,2],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ))

    # we add the detectors as markers, where the color indicates the signal amplitude at the current time (active detectors are colored, inactive are the standard given color)
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

    # if a detector is selected, we highlight it with a different color and size
    if selected_idx is not None:
        pos = detector_positions[selected_idx]
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers',
            marker=dict(size=10, color='lime'),
            name="Selected detector"
        ))

    # we also add the position of the neutrino interaction as a red marker
    fig.add_trace(go.Scatter3d(
        x=[x0], y=[y0], z=[z0],
        mode='markers',
        marker=dict(size=10, color='red'),
        name="Neutrino"
    ))

    # we set the layout of the plot with a fixed aspect ratio and limits to better visualize the geometry of the detector array
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
# INFO + PLOTS
# =====================
# this callback updates the info panel and the 3 plots (waveform, noise histogram and stack plot) based on the selected detector in the 3D plot and the current time from the slider
@app.callback(
    [Output("waveform", "figure"),
     Output("noise-hist", "figure"),
     Output("stack-plot", "figure"),
     Output("info-panel", "children")],
    [Input("3d-plot", "clickData"),
     Input("time-slider", "value")]
)

# this function updates the info panel and the 3 plots (waveform, noise histogram and stack plot) based on the selected detector in the 3D plot and the current time from the slider
def update_info(clickData, t_current_ms):

    # base info for statistical info about the detector
    base_info = [
        html.H3("Detector Info"),
        html.P(f"Total detectors: {detector_count}"),
        html.P(f"Lines: {line_count}"),
        html.P(f"Detectors/line: {detectors_per_line}"),
        html.P(f"Cable length: {total_cable_km:.1f} km"),
    ]

    # if no detector is selected, we return empty plots and just the base info
    if clickData is None:
        return go.Figure(), go.Figure(), go.Figure(), base_info

    # if a detector is selected, we get its index from the clickData and retrieve its data to create the plots and show the details in the info panel
    idx = clickData["points"][0]["customdata"]
    d = detector_data[idx]

    # we convert the time to ms and the signal and noise to mPa for better visualization in the plots
    t = d["time"] * 1e3
    s = d["signal"] * 1e3
    n = d["noise"] * 1e3
    total = s + n

    # define the peak of the signal
    peak_idx = np.argmax(np.abs(total))
    peak_val = total[peak_idx]
    peak_time = t[peak_idx]

    # we also get the arrival time of the signal at this detector for visualization in the plots
    arrival = d["arrival"] * 1e3

    # waveform plot
    fig1 = go.Figure()

    # focus around the arrival time of the signal
    window = np.abs(t - arrival) < 2.0

    # hard code against outliers, so not to big and not to small either
    if np.any(window):
        combined = s[window] + n[window]
        ymax = np.percentile(np.abs(combined), 95)
    else:
        ymax = np.percentile(np.abs(s + n), 95)

    # not to big, then the details will not be visible anymore
    ymax = min(ymax, 20)

    # not to small aswell
    ymax = max(ymax, 5)

    # noise (transparent, then signal is better visible)
    fig1.add_trace(go.Scatter(
        x=t, y=n,
        name="Noise",
        opacity=0.3,
        line=dict(color='blue')
    ))

    # signal (red, more visible)
    fig1.add_trace(go.Scatter(
        x=t, y=s,
        name="Signal",
        line=dict(color='red', width=2)
    ))
    # arrival line, dashed red
    fig1.add_trace(go.Scatter(
        x=[arrival, arrival], y=[-1.3*ymax, 1.3*ymax],
        mode='lines', line=dict(color='red', dash='dash'),
        name='Arrival'
    ))
    # peak line, dashed green
    fig1.add_trace(go.Scatter(
        x=[peak_time, peak_time], y=[-1.3*ymax, 1.3*ymax],
        mode='lines', line=dict(color='green', dash='dot'),
        name='Peak'
    ))

    # smart zoom focussing on signal already
    # where the signal is significantly larger then the rest
    mask = np.abs(s) > 0.3 * np.max(np.abs(s))

    # this is were
    if np.any(mask):
        t_min = t[mask].min()
        t_max = t[mask].max()
    else:
        t_min = arrival - 0.5
        t_max = arrival + 0.5

    # 
    fig1.update_layout(
        title=f"Detector {idx}",
        xaxis=dict(
            range=[t_min - 0.2, t_max + 0.2],
            title="Time [ms]"
        ),
        yaxis=dict(
            range=[-1.3*ymax, 1.3*ymax],
            title="Amplitude [mPa]"
        ),
        legend=dict(x=0.01, y=0.99)
    )

    # ===== Gaussian fit =====
    mu, sigma = norm.fit(n)

    # histogram data (voor mooie fit lijn)
    hist_vals, bin_edges = np.histogram(n, bins=80, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # gaussian curve
    gauss = norm.pdf(bin_centers, mu, sigma)

    fig2 = go.Figure()

    # histogram
    fig2.add_trace(go.Histogram(
        x=n,
        histnorm='probability density',
        nbinsx=80,
        name="Noise distribution"
    ))

    # gaussian fit lijn
    fig2.add_trace(go.Scatter(
        x=bin_centers,
        y=gauss,
        mode='lines',
        name='Gaussian fit',
        line=dict(color='red', width=3)
    ))

    # layout met labels + sigma
    fig2.update_layout(
        title=f"Noise distribution (μ = {mu:.2f}, σ = {sigma:.2f} mPa)",
        xaxis_title="Amplitude [mPa]",
        yaxis_title="Probability density",
        legend=dict(x=0.7, y=0.95)
    )

    # ===== STACKING MET CORRECTE TIME ALIGNMENT =====
    t_current = t_current_ms / 1000

    # we select the detectors that have received the signal by the current time, and are within the z_threshold of the particle (pancake selection)
    active_indices = [i for i,d in enumerate(detector_data)
        if d["arrival"] < t_current
        and abs(d["pos"][2] - z0) < z_threshold]

    # focus alleen op relevante tijd rond pulse
    t_ref = np.linspace(-0.002, 0.002, len(t_vals))  # ±2 ms window
    stack = np.zeros_like(t_ref)
    count = 0

    for i in active_indices:
        d2 = detector_data[i]

        total = d2["signal"] * 5 + d2["noise"]  # versterk signaal voor zichtbaarheid

        # verschuif tijd zodat arrival = 0
        t_shifted = d2["time"] - d2["arrival"]

        # interpoleer naar referentie grid
        aligned = np.interp(t_ref, t_shifted, total, left=0, right=0)

        stack += aligned
        count += 1

    if count > 0:
        stack /= np.sqrt(count)   # fysisch correct (ruis ~ sqrt(N))

    # ===== PLOT =====
    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(
        x=t_ref * 1e3,
        y=stack,
        name="Stacked signal",
        line=dict(color='blue', width=3)
    ))

    fig3.update_layout(
        title=f"Stacked signal ({count} detectors)",
        xaxis=dict(range=[-2, 2], title="Time [ms]"),
        yaxis_title="Amplitude [a.u.]"
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

    return fig1, fig2, fig3, extra_info

# =====================
# RUN
# =====================
if __name__ == "__main__":
    app.run(debug=True)
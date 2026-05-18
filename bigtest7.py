import numpy as np
import plotly.graph_objects as go

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from scipy import signal
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

# load the noise data
def load_noise_data():
    data = np.load("noise.npz")
    return data["noise"]

# =====================
# COMPUTE
# =====================
def compute_chi2_simple(signal, noise, t_vals, arrival):

    y = signal + noise
    sigma = np.std(noise)

    if sigma == 0:
        return np.inf, np.inf, 0, None

    # vast window van ±0.2 ms rond arrival (= t=0)
    window_half_width = 0.0002   # seconden

    # mask rond arrival
    mask = np.abs(t_vals) < window_half_width

    # selecteer alleen data in het window
    y_sel = y[mask]
    s_sel = signal[mask]

    # chi2 berekeningen
    chi2_h0 = np.sum((y_sel)**2) / sigma**2
    chi2_h1 = np.sum((y_sel - s_sel)**2) / sigma**2

    delta_chi2 = chi2_h0 - chi2_h1

    return chi2_h0, chi2_h1, delta_chi2, mask

def compute_all(detector_positions, source_pos, interpolator, noise_data, t_vals, R_vals, Z_vals):

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

        # always noise generated (independent of the signal location)
        i_rand = np.random.randint(0, noise_data.shape[0])
        j_rand = np.random.randint(0, noise_data.shape[1])
        noise = noise_data[i_rand, j_rand, :]
        noise = noise * (0.01 / np.std(noise))

        if (R < R_vals.min() or R > R_vals.max() or
            Z < Z_vals.min() or Z > Z_vals.max()):

            raw_signal = np.zeros_like(t_vals)

        else:
            raw_signal = interpolator([[R, Z]])[0]

        # shift signal physically in time
        signal = raw_signal

        # compute the ampltude as the peak-to-peak value of the signal (without noise)
        amp = np.max(signal) - np.min(signal)

        chi2_h0, chi2_h1, delta_chi2, mask = compute_chi2_simple(
        signal,
        noise,
        t_vals,
        arrival
    )

        detector_data.append({
            "id": i,
            "pos": pos,
            "R": R,
            "Z": Z,
            "time": t_vals + arrival,
            "signal": raw_signal,
            "noise": noise,
            "measured": signal + noise,
            "arrival": arrival,
            "chi2_h0": chi2_h0,
            "chi2_h1": chi2_h1,
            "delta_chi2": delta_chi2,
            "mask": mask
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
noise_data = load_noise_data()

# we compute the signal, noise, arrival times and amplitudes for all detectors
detector_data, amplitudes, arrival_times = compute_all(
    detector_positions,
    source_pos,
    interp,
    noise_data,
    t_vals,
    R_vals,
    Z_vals
)


# =====================
# GLOBAL CHI2
# =====================

# only detectors close to the pancake
chi2_detectors = [
    d for d in detector_data
    if abs(d["Z"]) < z_threshold
    and np.max(np.abs(d["signal"])) > np.std(d["noise"])
]

# global chi2 sums
global_chi2_h0 = np.sum([d["chi2_h0"] for d in chi2_detectors])

global_chi2_h1 = np.sum([d["chi2_h1"] for d in chi2_detectors])

global_delta_chi2 = np.sum([d["delta_chi2"] for d in chi2_detectors])

# =====================
# REDUCED CHI2
# =====================

# number of samples inside the fitting window
n_samples = np.sum(detector_data[0]["mask"])

# total number of detectors
n_detectors = len(chi2_detectors)

# total degrees of freedom
ndof = n_samples * n_detectors

# reduced chi2
reduced_chi2_h0 = global_chi2_h0 / ndof

reduced_chi2_h1 = global_chi2_h1 / ndof


# we convert the amplitudes to mPa for better visualization in the app
amps = amplitudes * 1000
# size of the time step (we assume uniform)
dt = t_vals[1] - t_vals[0]


# =====================
# VERTEX RECONSTRUCTION (2D)
# =====================

# local scan
x_scan = np.linspace(x0-1, x0+1, 20)
y_scan = np.linspace(y0-1, y0+1, 20)

# storage matrix for all global chi2 values
# rows = y positions, columns = x positions
chi2_grid = np.zeros((len(y_scan), len(x_scan)))

# loop over all hypothetical y positions
for iy, y_hyp in enumerate(y_scan):

    # loop over all hypothetical x positions
    for ix, x_hyp in enumerate(x_scan):

        # hypothetical vertex position
        # z is kept fixed for now
        hyp_pos = np.array([x_hyp, y_hyp, z0])

        # start global chi2 at zero
        global_chi2 = 0

        # loop over all detectors to compute what every detector
        # should measure if the neutrino was at this hypothetical position
        for d in chi2_detectors:

            # detector position
            det_pos = d["pos"]

            # vector from hypothesis position to detector
            r_vec = det_pos - hyp_pos

            # arrival time for this hypothesis
            arrival_hyp = np.linalg.norm(r_vec) / c_sound

            # cylindrical coordinates
            R = np.linalg.norm(r_vec[:2])
            Z = r_vec[2]

            # interpolate waveform
            if (R < R_vals.min() or R > R_vals.max() or
                Z < Z_vals.min() or Z > Z_vals.max()):

                expected_signal = np.zeros_like(t_vals)

            else:
                expected_signal = interp([[R, Z]])[0]

            # detector time axis
            measured_time = d["time"]

            # measured waveform
            measured = d["measured"]

            # shifted hypothesis waveform
            expected_shifted = np.interp(
                measured_time,
                t_vals + arrival_hyp,
                expected_signal,
                left=0,
                right=0
            )

            # noise sigma
            sigma = np.std(d["noise"])

            # timing-sensitive chi2
            chi2 = np.sum(
                (measured - expected_shifted)**2
                / sigma**2
            )

            global_chi2 += chi2

        # store global chi2 value in the 2D chi2 grid
        chi2_grid[iy, ix] = global_chi2

# search for the minimum chi2 position in the 2D grid
best_idx = np.unravel_index(
    np.argmin(chi2_grid),
    chi2_grid.shape
)

delta_chi2_grid = chi2_grid - np.min(chi2_grid)

# best-fit reconstructed coordinates
best_y = y_scan[best_idx[0]]
best_x = x_scan[best_idx[1]]



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
        dcc.Graph(id="combined-waveform"),
        dcc.Graph(id="noise-hist"),
        dcc.Graph(id="stack-plot"),
        dcc.Graph(id="delta-chi2-hist"),
        dcc.Graph(id="vertex-reco"),
        dcc.Graph(id="shift-test")        
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
     Output("combined-waveform", "figure"),
     Output("noise-hist", "figure"),
     Output("stack-plot", "figure"),
     Output("delta-chi2-hist", "figure"),
     Output("vertex-reco", "figure"),
     Output("shift-test", "figure"),
     Output("info-panel", "children")],

    [Input("3d-plot", "clickData"),
     Input("delta-chi2-hist", "clickData"),
     Input("time-slider", "value")]
)

# this function updates the info panel and the 3 plots (waveform, noise histogram and stack plot) based on the selected detector in the 3D plot and the current time from the slider
def update_info(clickData_3d, clickData_dchi2, t_current_ms):

    # base info for statistical info about the detector
    base_info = [
        html.H3("Detector Info"),

        html.P(f"Total detectors: {detector_count}"),
        html.P(f"Lines: {line_count}"),
        html.P(f"Detectors/line: {detectors_per_line}"),
        html.P(f"Cable length: {total_cable_km:.1f} km"),

        html.Hr(),

        html.H4("Global fit"),

        html.P(f"Global χ² (H0) = {global_chi2_h0:.1f}"),

        html.P(f"Global χ² (H1) = {global_chi2_h1:.1f}"),

        html.P(f"Global Δχ² = {global_delta_chi2:.1f}"),

        html.Hr(),

        html.H4("Reduced χ²"),

        html.P(f"Reduced χ² (H0) = {reduced_chi2_h0:.3f}"),

        html.P(f"Reduced χ² (H1) = {reduced_chi2_h1:.3f}"),

        html.P(f"Degrees of freedom = {ndof}")]
    
    # if no detector is selected, we return empty plots and just the base info
    selected_idx = None

    # klik op Δχ² plot
    if clickData_dchi2 is not None:
        selected_idx = int(clickData_dchi2["points"][0]["x"])

    # klik op 3D plot
    elif clickData_3d is not None:
        selected_idx = clickData_3d["points"][0]["customdata"]

    # niets geselecteerd
    if selected_idx is None:
        return (
    go.Figure(),
    go.Figure(),
    go.Figure(),
    go.Figure(),
    go.Figure(),
    go.Figure(),
    go.Figure(),
    base_info
)

    idx = selected_idx

    d = detector_data[idx]
    chi2_h0 = d["chi2_h0"]
    chi2_h1 = d["chi2_h1"]
    dchi2 = d["delta_chi2"]
    mask = d["mask"]

    # =====================
    # BEST-FIT TEMPLATE
    # =====================

    # reconstructed best-fit source position
    best_fit_pos = np.array([best_x, best_y, z0])

    # vector from reconstructed vertex to detector
    r_vec_fit = d["pos"] - best_fit_pos

    # cylindrical coordinates
    R_fit = np.linalg.norm(r_vec_fit[:2])
    Z_fit = r_vec_fit[2]

    # compute expected waveform from best-fit position
    if (R_fit < R_vals.min() or R_fit > R_vals.max() or
        Z_fit < Z_vals.min() or Z_fit > Z_vals.max()):

        best_fit_signal = np.zeros_like(t_vals)

    else:
        best_fit_signal = interp([[R_fit, Z_fit]])[0]

    # convert to mPa
    best_fit_signal *= 1e3

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
    window = mask

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



    t_start = arrival - 0.2
    t_end   = arrival + 0.2

    fig1.add_trace(go.Scatter(
        x=[t_start, t_start],
        y=[-1.3*ymax, 1.3*ymax],
        mode='lines',
        line=dict(color='purple', dash='dot'),
        name='Window start'
    ))

    fig1.add_trace(go.Scatter(
        x=[t_end, t_end],
        y=[-1.3*ymax, 1.3*ymax],
        mode='lines',
        line=dict(color='purple', dash='dot'),
        name='Window end'
    ))

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

    # best fit
    fig1.add_trace(go.Scatter(
        x=t,
        y=best_fit_signal,
        mode='lines',
        name='Best-fit template',
        line=dict(
            color='orange',
            width=3,
            dash='dash'
        )
    ))

    # peak line, dashed green
    fig1.add_trace(go.Scatter(
        x=[peak_time, peak_time], y=[-1.3*ymax, 1.3*ymax],
        mode='lines', line=dict(color='green', dash='dot'),
        name='Peak'
    ))

    # smart zoom focussing on signal already
    # where the signal is significantly larger then the rest
    signal_mask = np.abs(s) > 0.3 * np.max(np.abs(s))
    

    # takes only time were the signal is
    if np.any(signal_mask):
        t_min = t[signal_mask].min()
        t_max = t[signal_mask].max()
    # otherwise when no clear signal, taken around arrival time
    else:
        t_min = arrival - 0.5
        t_max = arrival + 0.5

    # plot layout settings
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

        # =====================
    # COMBINED SIGNAL + NOISE
    # =====================

    fig_combined = go.Figure()

    fig_combined.add_trace(go.Scatter(
        x=t,
        y=total,
        mode='lines',
        name='Signal + Noise',
        line=dict(color='darkblue', width=2)
    ))

    # arrival line
    fig_combined.add_trace(go.Scatter(
        x=[arrival, arrival],
        y=[-1.3*ymax, 1.3*ymax],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Arrival'
    ))

    # peak line
    fig_combined.add_trace(go.Scatter(
        x=[peak_time, peak_time],
        y=[-1.3*ymax, 1.3*ymax],
        mode='lines',
        line=dict(color='green', dash='dot'),
        name='Peak'
    ))


    fig_combined.add_trace(go.Scatter(
    x=[t_start, t_start],
    y=[-1.3*ymax, 1.3*ymax],
    mode='lines',
    line=dict(color='purple', dash='dot'),
    name='Window start'
))

    fig_combined.add_trace(go.Scatter(
        x=[t_end, t_end],
        y=[-1.3*ymax, 1.3*ymax],
        mode='lines',
        line=dict(color='purple', dash='dot'),
        name='Window end'
    ))

    fig_combined.update_layout(
        title=f"Detector {idx}: Signal + Noise",
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

    # Gaussian fit, mhu = average, sigma = spread
    mu, sigma = norm.fit(n)

    # Histogram distribution of the noise
    hist_vals, bin_edges = np.histogram(n, bins=80, density=True)
    # takes the middle of every bin
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # theoretical gaussian curve, to compare if the noise is gaussian distributed
    gauss = norm.pdf(bin_centers, mu, sigma)

    fig2 = go.Figure()

    # histogram
    fig2.add_trace(go.Histogram(
        x=n,
        histnorm='probability density',
        nbinsx=80,
        name="Noise distribution"
    ))

    # gaussian fit line
    fig2.add_trace(go.Scatter(
        x=bin_centers,
        y=gauss,
        mode='lines',
        name='Gaussian fit',
        line=dict(color='red', width=3)
    ))

    # layout with labels/title
    fig2.update_layout(
        title=f"Noise distribution (μ = {mu:.2f}, σ = {sigma:.2f} mPa)",
        xaxis_title="Amplitude [mPa]",
        yaxis_title="Probability density",
        legend=dict(x=0.7, y=0.95)
    )

    # stacking uses seconds instead of ms
    t_current = t_current_ms / 1000

    # we select the detectors that have received the signal by the current time, and are within the z_threshold of the particle (pancake selection)
    active_indices = [i for i,d in enumerate(detector_data)
        if d["arrival"] < t_current
        and abs(d["pos"][2] - z0) < z_threshold]

    # focus only on the relevant time around the pulse
    t_ref = np.linspace(-0.002, 0.002, len(t_vals))  
    # initialize stack
    stack = np.zeros_like(t_ref)
    count = 0

    # loop over detectecors
    for i in active_indices:
        d2 = detector_data[i]

        # signal and noise for this detector
        total = d2["signal"] + d2["noise"]  

        # signal peak aligments such that arrival = 0, so peaks are stacked
        t_shifted = d2["time"] - d2["arrival"]

        # interpolate to reference grid
        aligned = np.interp(t_ref, t_shifted, total, left=0, right=0)

        # summation 
        stack += aligned
        count += 1

    # normalization fysiscal correct takes noise ~ sqrt(N)
    if count > 0:
        stack /= np.sqrt(count)   

    # stack plot
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

    # info panel
    extra_info = base_info + [
        html.Hr(),
        html.H4(f"Selected detector {idx}"),
        html.P(f"R = {d['R']:.1f} m"),
        html.P(f"Z = {d['Z']:.1f} m"),
        html.P(f"Arrival = {arrival:.2f} ms"),
        html.P(f"Peak amplitude = {peak_val:.2f} mPa"),
        html.P(f"Peak time = {peak_time:.2f} ms"),
        html.P(f"Chi² (H0) = {chi2_h0:.1f}"),
        html.P(f"Chi² (H1) = {chi2_h1:.1f}"),
        html.P(f"ΔChi² = {dchi2:.1f}"), 

        html.Hr(),
        html.H4("Vertex reconstruction"),
        html.P(f"True x = {x0:.1f} m"),
        html.P(f"Best-fit x = {best_x:.1f} m"),
        html.P(f"True y = {y0:.1f} m"),
        html.P(f"Best-fit y = {best_y:.1f} m"),

        html.P(
            f"Reconstruction error = "
            f"{100*np.sqrt((best_x-x0)**2 + (best_y-y0)**2):.4f} cm"
        ),
    ]

    # verzamel alle delta chi2 waarden
    all_dchi2 = [d["delta_chi2"] for d in detector_data]

    indices = list(range(len(detector_data)))
    values = [d["delta_chi2"] for d in detector_data]

    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(
        x=indices,
        y=values,
        mode='markers',
        marker=dict(size=6),
        name="Detectors"
    ))

    # highlight selected detector
    fig4.add_trace(go.Scatter(
        x=[idx],
        y=[dchi2],
        mode='markers',
        marker=dict(size=12, color='red'),
        name="Selected"
    ))

    fig4.update_layout(
        title="ΔChi² per detector",
        xaxis_title="Detector index",
        yaxis_title="ΔChi²"
    )


    fig5 = go.Figure()

    fig5.add_trace(go.Heatmap(
        x=x_scan,
        y=y_scan,
        z=np.log10(delta_chi2_grid + 1e-6),

        colorscale='Viridis',

        zmin=-6,
        zmax=2,

        colorbar=dict(
            title="log10(Δχ²)",
            y=0.45,
            len=0.8
        )
    ))


    fig5.add_trace(go.Contour(
        x=x_scan,
        y=y_scan,
        z=delta_chi2_grid,
        contours=dict(
        start=0,
        end=2,
        size=0.1
    ),
        line=dict(width=1, color='white'),
        showscale=False
    ))


    # chi2 difference between true position and best fit
    true_delta_chi2 = delta_chi2_grid[
        np.argmin(np.abs(y_scan - y0)),
        np.argmin(np.abs(x_scan - x0))
    ]

    # true neutrino position
    fig5.add_trace(go.Scatter(
        x=[x0],
        y=[y0],
        mode='markers+text',
        marker=dict(size=12, color='red'),
        text=[f"Δχ² = {true_delta_chi2:.1f}"],
        textposition="top center",
        name='True position'
    ))

    # reconstructed best-fit position
    fig5.add_trace(go.Scatter(
        x=[best_x],
        y=[best_y],
        mode='markers',
        marker=dict(
            size=12,
            color='cyan',
            line=dict(color='black', width=1)
        ),
        name='Best fit'
    ))

    fig5.update_layout(
        title="2D Vertex reconstruction",

        xaxis=dict(
            title="Hypothesis x-position [m]",
            range=[x_scan.min(), x_scan.max()]
        ),

        yaxis=dict(
            title="Hypothesis y-position [m]",
            range=[y_scan.min(), y_scan.max()],
            scaleanchor="x",
            scaleratio=1
        ),

        legend=dict(
            x=1.02,
            y=1.0
        ),

        width=900,
        height=900
    )


    # =====================
    # 1.5 m SHIFT TEST
    # =====================

    # use currently selected detector
    test_detector = d["pos"]

    # original source
    source1 = np.array([x0, y0, z0])

    # shifted source (+1.5 m in x)
    source2 = np.array([x0 + 1.5, y0, z0])

    # helper function
    def get_waveform(det_pos, source_pos):

        r_vec = det_pos - source_pos

        arrival = np.linalg.norm(r_vec) / c_sound

        R = np.linalg.norm(r_vec[:2])
        Z = r_vec[2]

        if (R < R_vals.min() or R > R_vals.max() or
            Z < Z_vals.min() or Z > Z_vals.max()):

            signal = np.zeros_like(t_vals)

        else:
            signal = interp([[R, Z]])[0]

        # absolute detector time axis
        t_abs = t_vals + arrival

        return t_abs, signal





    # compute both waveforms
    t1, s1 = get_waveform(test_detector, source1)
    t2, s2 = get_waveform(test_detector, source2)

    # convert units
    t1_ms = t1 * 1e3
    t2_ms = t2 * 1e3

    s1_mpa = s1 * 1e3
    s2_mpa = s2 * 1e3

    # amplitude schaal
    combined_shift = np.concatenate([s1_mpa, s2_mpa])

    ymax_shift = 1.1 * np.max(np.abs(combined_shift))

    ymax_shift = min(ymax_shift, 20)
    ymax_shift = max(ymax_shift, 5)

    # zoom rond de puls
    signal_mask_shift = np.abs(s1_mpa) > 0.3 * np.max(np.abs(s1_mpa))

    if np.any(signal_mask_shift):
        tmin_shift = t1_ms[signal_mask_shift].min()
        tmax_shift = t1_ms[signal_mask_shift].max()
    else:
        tmin_shift = t1_ms.min()
        tmax_shift = t1_ms.max()

    # peak timing difference

    peak1_idx = np.argmax(np.abs(s1_mpa))
    peak2_idx = np.argmax(np.abs(s2_mpa))

    peak1_time = t1_ms[peak1_idx]
    peak2_time = t2_ms[peak2_idx]

    delta_t = peak2_time - peak1_time
    
    fig_shift = go.Figure()
    
    # original waveform
    fig_shift.add_trace(go.Scatter(
        x=t1_ms,
        y=s1_mpa,
        mode='lines',
        name='Original'
    ))

    # shifted waveform
    fig_shift.add_trace(go.Scatter(
        x=t2_ms,
        y=s2_mpa,
        mode='lines',
        name='Shifted (+1.5 m)'
    ))

    # original peak
    fig_shift.add_trace(go.Scatter(
        x=[peak1_time, peak1_time],
        y=[-ymax_shift, ymax_shift],
        mode='lines',
        line=dict(color='blue', dash='dash'),
        name='Original peak'
    ))

    # shifted peak
    fig_shift.add_trace(go.Scatter(
        x=[peak2_time, peak2_time],
        y=[-ymax_shift, ymax_shift],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Shifted peak'
    ))

    fig_shift.add_annotation(
    x=0.5*(peak1_time + peak2_time),
    y=0.8*ymax_shift,
    text=f"Δt = {delta_t:.4f} ms",
    showarrow=False,
    font=dict(size=16)
)
    
    fig_shift.update_layout(
    title="Effect of 1.5 m source shift",

    xaxis=dict(
        range=[tmin_shift - 0.2, tmax_shift + 0.2],
        title="Time [ms]"
    ),

    yaxis=dict(
        range=[-1.3*ymax_shift, 1.3*ymax_shift],
        title="Amplitude [mPa]"
    ),

    legend=dict(x=0.01, y=0.99)
)

    return fig1, fig_combined, fig2, fig3, fig4, fig5, fig_shift, extra_info

# =====================
# RUN
# =====================
if __name__ == "__main__":
    app.run(debug=True)
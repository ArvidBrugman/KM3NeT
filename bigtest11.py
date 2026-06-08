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

# neutrino direction
theta_nu = 0
phi_nu = 0      # azimuth

# speed of sound in water (m/s)
c_sound = 1500

# stacking selection (pancake), only detectors close to the source in z direction are used
z_threshold = 200

def get_direction_vector(theta, phi):

    vx = np.sin(theta) * np.cos(phi)
    vy = np.sin(theta) * np.sin(phi)
    vz = -np.cos(theta)

    v = np.array([vx, vy, vz])

    return v / np.linalg.norm(v)

nu_dir = get_direction_vector(
    theta_nu,
    phi_nu
)

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
        Z = np.dot(r_vec, nu_dir)

        perp_vec = r_vec - Z * nu_dir

        R = np.linalg.norm(perp_vec)

        # always noise generated (independent of the signal location)
        i_rand = np.random.randint(0, noise_data.shape[0])
        j_rand = np.random.randint(0, noise_data.shape[1])
        noise = noise_data[i_rand, j_rand, :]

        noise_std = np.std(noise)

        if not np.isfinite(noise_std) or noise_std < 1e-12:
            continue

        noise = noise * (0.01 / noise_std)

        if (R < R_vals.min() or R > R_vals.max() or
            Z < Z_vals.min() or Z > Z_vals.max()):

            raw_signal = np.zeros_like(t_vals)

        else:
            raw_signal = interpolator([[R, Z]])[0]

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


def compute_global_delta_chi2(
    measured_event,
    hyp_pos,
    detector_positions,
    interp,
    t_vals,
    R_vals,
    Z_vals
):

    global_chi2_h0 = 0
    global_chi2_h1 = 0

    n_used = 0

    for i, d in enumerate(chi2_detectors):

        det_pos = d["pos"]

        # detector -> hypothesis vector
        r_vec = det_pos - hyp_pos

        # arrival time hypothesis
        arrival_hyp = np.linalg.norm(r_vec) / c_sound

        # cylindrical coordinates
        Z = np.dot(r_vec, nu_dir)

        perp_vec = r_vec - Z * nu_dir

        R = np.linalg.norm(perp_vec)

        # build template
        if (
            R < R_vals.min() or
            R > R_vals.max() or
            Z < Z_vals.min() or
            Z > Z_vals.max()
        ):

            expected_signal = np.zeros_like(t_vals)

        else:

            expected_signal = interp([[R, Z]])[0]

        # skip empty templates
        if np.max(np.abs(expected_signal)) < 1e-12:
            continue

        # detector measured waveform
        if isinstance(measured_event[i], dict):
            measured = measured_event[i]["waveform"]
        else:
            measured = measured_event[i]

        sigma = np.std(d["noise"])

        if not np.isfinite(sigma) or sigma < 1e-12:
            continue

        # shift template in time
        measured_time = d["time"]

        expected_shifted = np.interp(
            measured_time,
            t_vals + arrival_hyp,
            expected_signal,
            left=0,
            right=0
        )

        # mask around signal region
        mask = np.abs(t_vals) < 0.0002

        m = measured[mask]
        s = expected_shifted[mask]

        chi2_h0 = np.sum((m)**2) / sigma**2

        chi2_h1 = np.sum((m - s)**2) / sigma**2

        delta = chi2_h0 - chi2_h1

        n_used += 1

        global_chi2_h0 += chi2_h0
        global_chi2_h1 += chi2_h1
    if n_used == 0:
        return np.nan

    return global_chi2_h0 - global_chi2_h1

def reconstruct_x(measured_event):

    chi2_x = []

    x_scan = np.linspace(
        x0 - 0.05,
        x0 + 0.05,
        200
    )

    for x_hyp in x_scan:

        hyp_pos = np.array([
            x_hyp,
            y0,
            z0
        ])

        delta = compute_global_delta_chi2(
            measured_event,
            hyp_pos,
            detector_positions,
            interp,
            t_vals,
            R_vals,
            Z_vals
        )

        chi2_x.append(-delta)

    chi2_x = np.array(chi2_x)

    delta_chi2_x = chi2_x - np.min(chi2_x)

    best_x = x_scan[np.argmin(delta_chi2_x)]

    return best_x


def reconstruct_y(measured_event):

    chi2_y = []

    y_scan = np.linspace(
        y0 - 0.05,
        y0 + 0.05,
        200
    )

    for y_hyp in y_scan:

        hyp_pos = np.array([
            x0,
            y_hyp,
            z0
        ])

        delta = compute_global_delta_chi2(
            measured_event,
            hyp_pos,
            detector_positions,
            interp,
            t_vals,
            R_vals,
            Z_vals
        )

        chi2_y.append(-delta)

    chi2_y = np.array(chi2_y)

    delta_chi2_y = (
        chi2_y -
        np.min(chi2_y)
    )

    best_y = y_scan[
        np.argmin(delta_chi2_y)
    ]

    return best_y



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

n_events = 1

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


chi2_x = []
chi2_y = []
chi2_z = []

delta_chi2_x = []
delta_chi2_y = []
delta_chi2_z = []

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

measured_event = []

for d in chi2_detectors:

    measured_event.append({
        "id": d["id"],
        "waveform": d["measured"]
    })

x_scan_1d = np.linspace(
    x0 - 2,
    x0 + 2,
    80 
)

for x_hyp in x_scan_1d:

    hyp_pos = np.array([
        x_hyp,
        y0,
        z0
    ])

    delta = compute_global_delta_chi2(
        measured_event,
        hyp_pos,
        detector_positions,
        interp,
        t_vals,
        R_vals,
        Z_vals
    )

    chi2_x.append(-delta)

chi2_x = np.array(chi2_x)

delta_chi2_x = chi2_x - np.min(chi2_x)

best_x_1d = x_scan_1d[np.argmin(delta_chi2_x)]

# =====================
# FINE X SCAN
# =====================

chi2_x_fine = []

x_scan_fine = np.linspace(
    best_x_1d - 0.05,
    best_x_1d + 0.05,
    200
)

for x_hyp in x_scan_fine:

    hyp_pos = np.array([
        x_hyp,
        y0,
        z0
    ])

    delta = compute_global_delta_chi2(
        measured_event,
        hyp_pos,
        detector_positions,
        interp,
        t_vals,
        R_vals,
        Z_vals
    )

    chi2_x_fine.append(-delta)

chi2_x_fine = np.array(
    chi2_x_fine
)

delta_chi2_x_fine = (
    chi2_x_fine -
    np.min(chi2_x_fine)
)

best_x_fine = x_scan_fine[
    np.argmin(delta_chi2_x_fine)
]

x_error_cm = abs(best_x_fine - x0) * 100

indices_1sigma_x = np.where(
    delta_chi2_x_fine <= 1
)[0]


if len(indices_1sigma_x) > 0:

    x_low_1sigma = x_scan_fine[
        indices_1sigma_x[0]
    ]

    x_high_1sigma = x_scan_fine[
        indices_1sigma_x[-1]
    ]

    sigma_x = (
        x_high_1sigma -
        x_low_1sigma
    ) / 2

    x_low_2sigma  = best_x_fine - 2*sigma_x
    x_high_2sigma = best_x_fine + 2*sigma_x

    x_low_3sigma  = best_x_fine - 3*sigma_x
    x_high_3sigma = best_x_fine + 3*sigma_x

else:

    sigma_x = np.nan

    x_low_1sigma = np.nan
    x_high_1sigma = np.nan

    x_low_2sigma = np.nan
    x_high_2sigma = np.nan

    x_low_3sigma = np.nan
    x_high_3sigma = np.nan





y_scan_1d = np.linspace(y0 - 2, y0 + 2, 80)

for y_hyp in y_scan_1d:

    hyp_pos = np.array([
        x0,
        y_hyp,
        z0
    ])

    delta = compute_global_delta_chi2(
        measured_event,
        hyp_pos,
        detector_positions,
        interp,
        t_vals,
        R_vals,
        Z_vals
    )

    chi2_y.append(-delta)

chi2_y = np.array(chi2_y)

delta_chi2_y = chi2_y - np.min(chi2_y)

best_y_1d = y_scan_1d[np.argmin(delta_chi2_y)]

y_error_cm = abs(best_y_1d - y0) * 100


# =====================
# FINE Y SCAN
# =====================

chi2_y_fine = []

y_scan_fine = np.linspace(
    best_y_1d - 0.05,
    best_y_1d + 0.05,
    200
)

for y_hyp in y_scan_fine:

    hyp_pos = np.array([
        x0,
        y_hyp,
        z0
    ])

    delta = compute_global_delta_chi2(
        measured_event,
        hyp_pos,
        detector_positions,
        interp,
        t_vals,
        R_vals,
        Z_vals
    )

    chi2_y_fine.append(-delta)

chi2_y_fine = np.array(
    chi2_y_fine
)

delta_chi2_y_fine = (
    chi2_y_fine -
    np.min(chi2_y_fine)
)

best_y_fine = y_scan_fine[
    np.argmin(delta_chi2_y_fine)
]

y_error_cm = abs(best_y_fine - y0) * 100

indices_1sigma_y = np.where(
    delta_chi2_y_fine <= 1
)[0]

if len(indices_1sigma_y) > 0:

    y_low_1sigma = y_scan_fine[
        indices_1sigma_y[0]
    ]

    y_high_1sigma = y_scan_fine[
        indices_1sigma_y[-1]
    ]

    sigma_y = (
        y_high_1sigma -
        y_low_1sigma
    ) / 2

    y_low_2sigma  = best_y_fine - 2*sigma_y
    y_high_2sigma = best_y_fine + 2*sigma_y

    y_low_3sigma  = best_y_fine - 3*sigma_y
    y_high_3sigma = best_y_fine + 3*sigma_y

else:

    sigma_y = np.nan
    y_low_1sigma = np.nan
    y_high_1sigma = np.nan

    y_low_2sigma = np.nan
    y_high_2sigma = np.nan

    y_low_3sigma = np.nan
    y_high_3sigma = np.nan




z_scan_1d = np.linspace(z0 - 2, z0 + 2, 80)

for z_hyp in z_scan_1d:

    hyp_pos = np.array([
        x0,
        y0,
        z_hyp
    ])

    delta = compute_global_delta_chi2(
        measured_event,
        hyp_pos,
        detector_positions,
        interp,
        t_vals,
        R_vals,
        Z_vals
    )

    chi2_z.append(-delta)

chi2_z = np.array(chi2_z)

delta_chi2_z = chi2_z - np.min(chi2_z)

best_z_1d = z_scan_1d[np.argmin(delta_chi2_z)]

z_error_cm = abs(best_z_1d - z0) * 100

# we convert the amplitudes to mPa for better visualization in the app
amps = amplitudes * 1000
# size of the time step (we assume uniform)
dt = t_vals[1] - t_vals[0]


# =====================
# X MONTE CARLO
# =====================

n_reco_events = 20

x_reco_values = []

for evt in range(n_reco_events):

    measured_event = []

    for d in chi2_detectors:

        signal_template = d["signal"]

        i_rand = np.random.randint(
            0,
            noise_data.shape[0]
        )

        j_rand = np.random.randint(
            0,
            noise_data.shape[1]
        )

        noise = noise_data[i_rand, j_rand, :]

        noise_std = np.std(noise)

        if not np.isfinite(noise_std) or noise_std < 1e-12:
            continue

        noise = noise * (0.01 / noise_std)

        measured = signal_template + noise

        measured_event.append({
            "id": d["id"],
            "waveform": measured
        })

    best_x_evt = reconstruct_x(
        measured_event
    )

    x_reco_values.append(
        best_x_evt
    )

x_reco_values = np.array(
    x_reco_values
)

dx_cm = (
    x_reco_values - x0
) * 100

sigma_mc_x = np.std(dx_cm)

mean_mc_x = np.mean(dx_cm)


# =====================
# MONTE CARLO Y
# =====================

y_reco_values = []

for evt in range(n_reco_events):

    measured_event = []

    for d in chi2_detectors:

        signal_template = d["signal"]

        i_rand = np.random.randint(
            0,
            noise_data.shape[0]
        )

        j_rand = np.random.randint(
            0,
            noise_data.shape[1]
        )

        noise = noise_data[i_rand, j_rand, :]

        noise_std = np.std(noise)

        if not np.isfinite(noise_std) or noise_std < 1e-12:
            continue

        noise = noise * (0.01 / noise_std)

        measured = signal_template + noise

        measured_event.append({
            "id": d["id"],
            "waveform": measured
        })

    best_y_evt = reconstruct_y(
        measured_event
    )

    y_reco_values.append(
        best_y_evt
    )

y_reco_values = np.array(
    y_reco_values
)

dy_cm = (
    y_reco_values - y0
) * 100

sigma_mc_y = np.std(
    dy_cm
)

mean_mc_y = np.mean(
    dy_cm
)

# =====================
# NOISE-ONLY DELTA CHI2 DISTRIBUTION
# =====================

n_pseudo_experiments = 100

noise_delta_chi2_distribution = []

for evt in range(n_pseudo_experiments):

    # build one pure noise event
    measured_event = []
    

    for d in chi2_detectors:

        # random noise realization
        i_rand = np.random.randint(0, noise_data.shape[0])
        j_rand = np.random.randint(0, noise_data.shape[1])

        noise = noise_data[i_rand, j_rand, :]

        noise_std = np.std(noise)

        if not np.isfinite(noise_std) or noise_std < 1e-12:
            continue

        noise = noise * (0.01 / noise_std)

        measured_event.append(noise)

    # compare THIS noise event
    # with THIS hypothesis template
    delta = compute_global_delta_chi2(
        measured_event,
        hyp_pos,
        detector_positions,
        interp,
        t_vals,
        R_vals,
        Z_vals
    )

    noise_delta_chi2_distribution.append(delta)

noise_delta_chi2_distribution = np.array(
    noise_delta_chi2_distribution
)

# =====================
# SIGNAL+NOISE DELTA CHI2 DISTRIBUTION
# =====================

signal_delta_chi2_distribution = []

for evt in range(n_pseudo_experiments):

    measured_event = []

    for d in chi2_detectors:

        # TRUE signal
        signal_template = d["signal"]

        # random noise
        i_rand = np.random.randint(0, noise_data.shape[0])
        j_rand = np.random.randint(0, noise_data.shape[1])

        noise = noise_data[i_rand, j_rand, :]

        noise_std = np.std(noise)

        if not np.isfinite(noise_std) or noise_std < 1e-12:
            continue

        noise = noise * (0.01 / noise_std)

        # signal + noise
        measured = signal_template + noise

        measured_event.append(measured)

    # compare with SAME hypothesis template
    delta = compute_global_delta_chi2(
        measured_event,
        hyp_pos,
        detector_positions,
        interp,
        t_vals,
        R_vals,
        Z_vals
    )

    signal_delta_chi2_distribution.append(delta)

signal_delta_chi2_distribution = np.array(
    signal_delta_chi2_distribution
)

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
        dcc.Graph(id="delta-x-plot"),
        dcc.Graph(id="delta-x-fine-plot"),

        dcc.Graph(id="delta-y-plot"),
        dcc.Graph(id="delta-y-fine-plot"),

        dcc.Graph(id="delta-z-plot"),
        dcc.Graph(id="noise-dchi2-distribution"),
        dcc.Graph(id="signal-dchi2-distribution"),
        dcc.Graph(id="combined-dchi2-distribution"),
        dcc.Graph(id="waveform"),
        dcc.Graph(id="combined-waveform"),
        dcc.Graph(id="noise-hist"),
        dcc.Graph(id="stack-plot"),
        dcc.Graph(id="delta-chi2-hist"),      
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

    L = 2500

    start = source_pos - L * nu_dir
    end = source_pos

    arrow_size = 400
    tmp = np.array([0,0,1])

    if abs(np.dot(tmp, nu_dir)) > 0.9:
        tmp = np.array([1,0,0])

    v1 = np.cross(nu_dir, tmp)
    v1 /= np.linalg.norm(v1)

    left = (
    end
    - arrow_size * nu_dir
    + 0.4 * arrow_size * v1
)

    right = (
        end
        - arrow_size * nu_dir
        - 0.4 * arrow_size * v1
    )

    fig.add_trace(go.Scatter3d(
    x=[left[0], end[0], right[0]],
    y=[left[1], end[1], right[1]],
    z=[left[2], end[2], right[2]],
    mode='lines',
    line=dict(
        color='cyan',
        width=14
    ),
    showlegend=False
))


    n_points = 60

    t = np.linspace(0, 1, n_points)

    x_dash = start[0] + t * (end[0] - start[0])
    y_dash = start[1] + t * (end[1] - start[1])
    z_dash = start[2] + t * (end[2] - start[2])

    fig.add_trace(go.Scatter3d(
    x=x_dash[::2],
    y=y_dash[::2],
    z=z_dash[::2],
    mode='markers',
    marker=dict(
        size=4,
        color='cyan'
    ),
    name='Neutrino direction'
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
@app.callback([
    Output("delta-x-plot", "figure"),
    Output("delta-x-fine-plot", "figure"),

    Output("delta-y-plot", "figure"),
    Output("delta-y-fine-plot", "figure"),

    Output("delta-z-plot", "figure"),
    Output("noise-dchi2-distribution", "figure"),
    Output("signal-dchi2-distribution", "figure"),
    Output("combined-dchi2-distribution", "figure"),
    Output("waveform", "figure"),
     Output("combined-waveform", "figure"),
     Output("noise-hist", "figure"),
     Output("stack-plot", "figure"),
     Output("delta-chi2-hist", "figure"),
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

        html.P(f"Degrees of freedom = {ndof}"),

        html.Hr(),

        html.H4("1D X reconstruction"),

        html.P(f"True x = {x0:.3f} m"),

        html.P(f"Best-fit x = {best_x_fine:.3f} m"),

        html.P(f"Bias = {x_error_cm:.2f} cm"),

        html.P(f"σx = {sigma_x*100:.2f} cm"),

        html.P(f"MC bias = {mean_mc_x:.2f} cm"),

        html.P(f"MC σx = {sigma_mc_x:.2f} cm"),


        html.P(
            f"1σ interval = "
            f"[{x_low_1sigma:.3f}, "
            f"{x_high_1sigma:.3f}] m"
        ),

        html.Hr(),

        html.H4("1D Y reconstruction"),

        html.P(f"True y = {y0:.3f} m"),

        html.P(f"Best-fit y = {best_y_fine:.3f} m"),

        html.P(f"Bias = {y_error_cm:.2f} cm"),

        html.P(f"σy = {sigma_y*100:.2f} cm"),

        html.P(f"MC bias = {mean_mc_y:.2f} cm"),

        html.P(f"MC σy = {sigma_mc_y:.2f} cm"),

        html.P(
            f"1σ interval = "
            f"[{y_low_1sigma:.3f}, "
            f"{y_high_1sigma:.3f}] m"
        ),

        ]
    
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
        html.P(f"ΔChi² = {dchi2:.1f}")]



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
   
    fig_delta_x = go.Figure()

    fig_delta_x.add_trace(
        go.Scatter(
            x=x_scan_1d,
            y=delta_chi2_x,
            mode="lines",
            name="Δχ²(x)"
        )
    )
    fig_delta_x.add_hline(
    y=1,
    line_dash="dot",
    line_color="green"
)

    fig_delta_x.add_vline(
        x=x0,
        line_dash="dash"
    )

    fig_delta_x.add_trace(
    go.Scatter(
        x=[x0],
        y=[0],
        mode="markers",
        marker=dict(
            size=12,
            color="red"
        ),
        name="True x"
    )
)

    fig_delta_x.add_trace(
        go.Scatter(
            x=[best_x_1d],
            y=[0],
            mode="markers",
            marker=dict(
                size=12,
                color="cyan"
            ),
            name="Best fit"
        )
    )

    fig_delta_x.update_layout(
        title="Δχ² scan in x",
        xaxis_title="x [m]",
        yaxis_title="Δχ²"
    )

    fig_delta_x_fine = go.Figure()

    # 3σ band
    fig_delta_x_fine.add_vrect(
        x0=x_low_3sigma,
        x1=x_high_3sigma,
        fillcolor="orange",
        opacity=0.2,
        layer="below",
        line_width=0
    )

    # 2σ band
    fig_delta_x_fine.add_vrect(
        x0=x_low_2sigma,
        x1=x_high_2sigma,
        fillcolor="gold",
        opacity=0.20,
        layer="below",
        line_width=0
    )

    # 1σ band
    fig_delta_x_fine.add_vrect(
        x0=x_low_1sigma,
        x1=x_high_1sigma,
        fillcolor="limegreen",
        opacity=0.3,
        layer="below",
        line_width=0
    )

    fig_delta_x_fine.add_trace(
    go.Scatter(
        x=x_scan_fine,
        y=delta_chi2_x_fine,
        mode="lines",
        name="Δχ² fine"
    )
)

    fig_delta_x_fine.update_yaxes(
    range=[0,10]
)

    fig_delta_x_fine.add_trace(
    go.Scatter(
        x=[x_scan_fine[0], x_scan_fine[-1]],
        y=[1, 1],
        mode="lines",
        name="Δχ² = 1 (1σ)",
        line=dict(
            color="green",
            dash="dot"
        )
    )
)
    fig_delta_x_fine.add_trace(
    go.Scatter(
        x=[x_scan_fine[0], x_scan_fine[-1]],
        y=[4,4],
        mode="lines",
        name="Δχ² = 4 (2σ)",
        line=dict(
            color="gold",
            dash="dot"
        )
    )
)

    fig_delta_x_fine.add_trace(
        go.Scatter(
            x=[x_scan_fine[0], x_scan_fine[-1]],
            y=[9,9],
            mode="lines",
            name="Δχ² = 9 (3σ)",
            line=dict(
                color="orange",
                dash="dot"
            )
        )
    )

    fig_delta_x_fine.add_trace(
    go.Scatter(
        x=[best_x_fine, best_x_fine],
        y=[0, 10],
        mode="lines",
        name="Best fit",
        line=dict(
            color="black",
            dash="dash"
        )
    )
)

    fig_delta_x_fine.add_trace(
    go.Scatter(
        x=[x0, x0],
        y=[0,10],
        mode="lines",
        name="True position",
        line=dict(
            color="magenta",
            dash="dashdot"
        )
    )
)

    fig_delta_x_fine.update_layout(
        title="Zoom around minimum",
        xaxis_title="x [m]",
        yaxis_title="Δχ²"
    )

    fig_delta_x_fine.update_layout(
    title=f"Zoom around minimum (σx = {sigma_x*100:.2f} cm)"
)
    fig_delta_x_fine.update_xaxes(
    range=[
        best_x_fine - 0.025,
        best_x_fine + 0.025
    ]
)





    fig_delta_y = go.Figure()

    fig_delta_y.add_trace(
        go.Scatter(
            x=y_scan_1d,
            y=delta_chi2_y,
            mode="lines",
            name="Δχ²(y)"
        )
    )

    fig_delta_y.add_hline(
    y=1,
    line_dash="dot",
    line_color="green"
)

    fig_delta_y.add_vline(
        x=y0,
        line_dash="dash"
    )

    fig_delta_y.add_trace(
    go.Scatter(
        x=[y0],
        y=[0],
        mode="markers",
        marker=dict(
            size=12,
            color="red"
        ),
        name="True y"
    )
)

    fig_delta_y.add_trace(
        go.Scatter(
            x=[best_y_1d],
            y=[0],
            mode="markers",
            marker=dict(
                size=12,
                color="cyan"
            ),
            name="Best fit"
        )
    )

    fig_delta_y.update_layout(
        title="Δχ² scan in y",
        xaxis_title="y [m]",
        yaxis_title="Δχ²"
    )

    fig_delta_y_fine = go.Figure()

    # 3σ band
    fig_delta_y_fine.add_vrect(
        x0=y_low_3sigma,
        x1=y_high_3sigma,
        fillcolor="orange",
        opacity=0.2,
        layer="below",
        line_width=0
    )

    # 2σ band
    fig_delta_y_fine.add_vrect(
        x0=y_low_2sigma,
        x1=y_high_2sigma,
        fillcolor="gold",
        opacity=0.20,
        layer="below",
        line_width=0
    )

    # 1σ band
    fig_delta_y_fine.add_vrect(
        x0=y_low_1sigma,
        x1=y_high_1sigma,
        fillcolor="limegreen",
        opacity=0.3,
        layer="below",
        line_width=0
    )

    fig_delta_y_fine.add_trace(
        go.Scatter(
            x=y_scan_fine,
            y=delta_chi2_y_fine,
            mode="lines",
            name="Δχ² fine"
        )
    )

    fig_delta_y_fine.update_yaxes(
        range=[0,10]
    )

    fig_delta_y_fine.add_trace(
        go.Scatter(
            x=[y_scan_fine[0], y_scan_fine[-1]],
            y=[1,1],
            mode="lines",
            name="Δχ² = 1 (1σ)",
            line=dict(
                color="green",
                dash="dot"
            )
        )
    )

    fig_delta_y_fine.add_trace(
        go.Scatter(
            x=[y_scan_fine[0], y_scan_fine[-1]],
            y=[4,4],
            mode="lines",
            name="Δχ² = 4 (2σ)",
            line=dict(
                color="gold",
                dash="dot"
            )
        )
    )

    fig_delta_y_fine.add_trace(
        go.Scatter(
            x=[y_scan_fine[0], y_scan_fine[-1]],
            y=[9,9],
            mode="lines",
            name="Δχ² = 9 (3σ)",
            line=dict(
                color="orange",
                dash="dot"
            )
        )
    )

    fig_delta_y_fine.add_trace(
        go.Scatter(
            x=[best_y_fine, best_y_fine],
            y=[0,10],
            mode="lines",
            name="Best fit",
            line=dict(
                color="black",
                dash="dash"
            )
        )
    )

    fig_delta_y_fine.add_trace(
        go.Scatter(
            x=[y0, y0],
            y=[0,10],
            mode="lines",
            name="True position",
            line=dict(
                color="magenta",
                dash="dashdot"
            )
        )
    )

    fig_delta_y_fine.update_layout(
        title=f"Zoom around minimum (σy = {sigma_y*100:.2f} cm)",
        xaxis_title="y [m]",
        yaxis_title="Δχ²"
    )

    fig_delta_y_fine.update_xaxes(
        range=[
            best_y_fine - 0.025,
            best_y_fine + 0.025
        ]
    )





    fig_delta_z = go.Figure()

    fig_delta_z.add_trace(
        go.Scatter(
            x=z_scan_1d,
            y=delta_chi2_z,
            mode="lines",
            name="Δχ²(z)"
        )
    )

    fig_delta_z.add_vline(
        x=z0,
        line_dash="dash"
    )

    fig_delta_z.add_trace(
    go.Scatter(
        x=[z0],
        y=[0],
        mode="markers",
        marker=dict(
            size=12,
            color="red"
        ),
        name="True z"
    )
)

    fig_delta_z.add_trace(
        go.Scatter(
            x=[best_z_1d],
            y=[0],
            mode="markers",
            marker=dict(
                size=12,
                color="cyan"
            ),
            name="Best fit"
        )
    )

    fig_delta_z.update_layout(
        title="Δχ² scan in z",
        xaxis_title="z [m]",
        yaxis_title="Δχ²"
    )


    # =====================
    # NOISE Δχ² DISTRIBUTION
    # =====================

    fig_noise_dchi2 = go.Figure()

    # histogram only for density estimation
    hist_noise, bins_noise = np.histogram(
        noise_delta_chi2_distribution,
        bins=40,
        density=True
    )

    # bin centers
    x_noise = 0.5 * (bins_noise[1:] + bins_noise[:-1])

    # scatter points
    fig_noise_dchi2.add_trace(go.Scatter(
        x=x_noise,
        y=hist_noise,
        mode='markers',
        marker=dict(size=7),
        name='Noise-only'
    ))

    # Gaussian fit
    mu_noise, sigma_noise = norm.fit(
        noise_delta_chi2_distribution
    )

    xfit_noise = np.linspace(
        noise_delta_chi2_distribution.min(),
        noise_delta_chi2_distribution.max(),
        500
    )

    yfit_noise = norm.pdf(
        xfit_noise,
        mu_noise,
        sigma_noise
    )

    fig_noise_dchi2.add_trace(go.Scatter(
        x=xfit_noise,
        y=yfit_noise,
        mode='lines',
        line=dict(width=3),
        name='Gaussian fit'
    ))

    fig_noise_dchi2.update_layout(
        title=f"Noise-only Δχ² distribution (μ={mu_noise:.1f}, σ={sigma_noise:.1f})",
        xaxis_title="Δχ²",
        yaxis_title="Probability density"
    )



   # =====================
    # SIGNAL+NOISE Δχ² DISTRIBUTION
    # =====================

    fig_signal_dchi2 = go.Figure()

    hist_signal, bins_signal = np.histogram(
        signal_delta_chi2_distribution,
        bins=40,
        density=True
    )

    x_signal = 0.5 * (bins_signal[1:] + bins_signal[:-1])

    # scatter points
    fig_signal_dchi2.add_trace(go.Scatter(
        x=x_signal,
        y=hist_signal,
        mode='markers',
        marker=dict(size=7),
        name='Signal+noise'
    ))

    # Gaussian fit
    mu_signal, sigma_signal = norm.fit(
        signal_delta_chi2_distribution
    )

    xfit_signal = np.linspace(
        signal_delta_chi2_distribution.min(),
        signal_delta_chi2_distribution.max(),
        500
    )

    yfit_signal = norm.pdf(
        xfit_signal,
        mu_signal,
        sigma_signal
    )

    fig_signal_dchi2.add_trace(go.Scatter(
        x=xfit_signal,
        y=yfit_signal,
        mode='lines',
        line=dict(width=3),
        name='Gaussian fit'
    ))

    fig_signal_dchi2.update_layout(
        title=f"Signal+noise Δχ² distribution (μ={mu_signal:.1f}, σ={sigma_signal:.1f})",
        xaxis_title="Δχ²",
        yaxis_title="Probability density"
    )

    # =====================
    # COMBINED Δχ² DISTRIBUTIONS
    # =====================

    fig_combined_dchi2 = go.Figure()

    # -----------------
    # NOISE
    # -----------------

    fig_combined_dchi2.add_trace(go.Scatter(
        x=x_noise,
        y=hist_noise,
        mode='markers',
        marker=dict(size=7),
        name='Noise-only'
    ))

    fig_combined_dchi2.add_trace(go.Scatter(
        x=xfit_noise,
        y=yfit_noise,
        mode='lines',
        line=dict(width=3),
        name='Noise Gaussian fit'
    ))

    # -----------------
    # SIGNAL
    # -----------------

    fig_combined_dchi2.add_trace(go.Scatter(
        x=x_signal,
        y=hist_signal,
        mode='markers',
        marker=dict(size=7),
        name='Signal+noise'
    ))

    fig_combined_dchi2.add_trace(go.Scatter(
        x=xfit_signal,
        y=yfit_signal,
        mode='lines',
        line=dict(width=3),
        name='Signal Gaussian fit'
    ))

    fig_combined_dchi2.update_layout(
        title="Combined Δχ² distributions",
        xaxis_title="Δχ²",
        yaxis_title="Probability density"
    )


    return (
    fig_delta_x,
    fig_delta_x_fine,

    fig_delta_y,
    fig_delta_y_fine,

    fig_delta_z,

    fig_noise_dchi2,
    fig_signal_dchi2,
    fig_combined_dchi2,
    fig1,
    fig_combined,
    fig2,
    fig3,
    fig4,
    extra_info
)

# =====================
# RUN
# =====================
if __name__ == "__main__":
    app.run(debug=True)
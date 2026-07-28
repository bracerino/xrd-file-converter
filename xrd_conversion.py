# xrd_axis_converter.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from io import StringIO, BytesIO
import zipfile


BG_METHODS = [
    "None",
    "Polynomial Fit",
    "SNIP Algorithm",
    "Rolling Ball Algorithm",
    "airPLS (Adaptive Baseline)",
]


def compute_background(x, y, method, params):
    """Estimate the background/baseline of an XRD pattern.

    Returns an array the same length as ``y`` holding the estimated
    background. Supported methods match ``BG_METHODS``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if method == "Polynomial Fit":
        poly_degree = int(params.get("poly_degree", 6))
        n_iters = int(params.get("n_iters", 40))
        sort_idx = np.argsort(x)
        xs, ys_sorted = x[sort_idx], y[sort_idx]
        ywork = ys_sorted.copy()
        for _ in range(n_iters):
            try:
                coeffs = np.polyfit(xs, ywork, poly_degree)
                fitted = np.polyval(coeffs, xs)
            except Exception:
                fitted = ywork
            ywork = np.minimum(ywork, fitted)
        coeffs = np.polyfit(xs, ywork, poly_degree)
        bg_sorted = np.polyval(coeffs, xs)
        unsort = np.argsort(sort_idx)
        return bg_sorted[unsort]

    elif method == "SNIP Algorithm":
        snip_iters = int(params.get("snip_iters", 20))
        v = np.log(np.log(np.sqrt(np.maximum(y, 0.0) + 1.0) + 1.0) + 1.0)
        for p in range(1, snip_iters + 1):
            avg = (np.roll(v, p) + np.roll(v, -p)) / 2.0
            v_new = np.minimum(v, avg)
            v_new[:p] = v[:p]
            v_new[-p:] = v[-p:]
            v = v_new
        return np.maximum((np.exp(np.exp(v) - 1.0) - 1.0) ** 2 - 1.0, 0.0)

    elif method == "Rolling Ball Algorithm":
        from scipy.signal import savgol_filter as _savgol
        ball_radius = int(params.get("ball_radius", 30))
        ball_smoothing = int(params.get("ball_smoothing", 3))
        ys = y.copy()
        if ball_smoothing > 0:
            wlen = min(len(ys) // 10, 20)
            if wlen % 2 == 0:
                wlen += 1
            if wlen >= 3:
                for _ in range(ball_smoothing):
                    ys = _savgol(ys, wlen, 2)
        return np.array([
            np.min(ys[max(0, i - ball_radius):
                      min(len(ys), i + ball_radius + 1)])
            for i in range(len(ys))
        ])

    else:  # airPLS (Adaptive Baseline) — asymmetric least squares
        from scipy.sparse import diags as _sp_diags
        from scipy.sparse.linalg import spsolve as _spsolve
        lam = float(params.get("lam", 1e6))
        p = float(params.get("p", 0.01))
        n_iter = int(params.get("n_iter", 15))
        yy = y.astype(float)
        n = len(yy)
        if n < 3:
            return np.zeros_like(yy)
        D = _sp_diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
        H = lam * D.T.dot(D)
        w = np.ones(n)
        z = yy.copy()
        for _ in range(n_iter):
            W = _sp_diags(w, 0, shape=(n, n))
            z = _spsolve((W + H).tocsc(), w * yy)
            w = p * (yy > z) + (1 - p) * (yy <= z)
        return z


def remove_background(x, y, method, params):
    """Return ``y`` with the estimated background subtracted (clipped at 0)."""
    if not method or method == "None":
        return np.asarray(y, dtype=float)
    try:
        bg = compute_background(x, y, method, params)
        return np.maximum(0.0, np.asarray(y, dtype=float) - bg)
    except Exception:
        return np.asarray(y, dtype=float)


def parse_xy_simple(file_content):
    try:
        lines = file_content.splitlines()
        if not lines:
            return None

        first_line = lines[0]
        has_header = any(char.isalpha() for char in first_line)

        data_io = StringIO(file_content)
        skiprows = 1 if has_header else 0

        df = pd.read_csv(
            data_io,
            sep=r'[\s,;\t]+',
            engine='python',
            header=None,
            skiprows=skiprows,
            names=['X', 'Y'],
            comment='#'
        )
        return df.dropna().astype(float)
    except Exception as e:
        st.error(f"Failed to parse file. Error: {e}")
        return None


def convert_xaxis_data(x_values, input_format, output_format,
                       input_wavelength=None, output_wavelength=None):
    wavelength_map = {
        '2theta_Cu': 1.54056,
        '2theta_Co': 1.78897,
        '2theta_Mo': 0.70932,
        '2theta_Cr': 2.28970,
        '2theta_Fe': 1.93604,
        '2theta_Ag': 0.55941,
        'd-spacing': None,
        'q-vector': None,
        '2theta_custom': input_wavelength
    }

    lambda_in = wavelength_map.get(input_format)
    if input_format == '2theta_custom':
        lambda_in = input_wavelength

    lambda_out = wavelength_map.get(output_format)
    if output_format == '2theta_custom':
        lambda_out = output_wavelength

    try:
        if input_format == 'q-vector' and output_format == 'd-spacing':
            valid = x_values > 0
            result = np.zeros_like(x_values)
            result[valid] = 2 * np.pi / x_values[valid]
            result[~valid] = np.nan
            return result

        elif input_format == 'd-spacing' and output_format == 'q-vector':
            valid = x_values > 0
            result = np.zeros_like(x_values)
            result[valid] = 2 * np.pi / x_values[valid]
            result[~valid] = np.nan
            return result

        elif input_format == 'q-vector' and '2theta' in output_format:
            if lambda_out is None:
                st.error("Output wavelength required for q-vector to 2theta conversion")
                return x_values

            valid = x_values >= 0
            sin_arg = (x_values[valid] * lambda_out) / (4 * np.pi)
            mask = (sin_arg >= -1) & (sin_arg <= 1)

            theta = np.arcsin(sin_arg[mask])
            twotheta = 2 * np.degrees(theta)

            result = np.zeros_like(x_values)
            result_indices = np.where(valid)[0][mask]
            result[result_indices] = twotheta
            result[~valid] = np.nan
            return result

        elif '2theta' in input_format and output_format == 'q-vector':
            if lambda_in is None:
                st.error("Input wavelength required for 2theta to q-vector conversion")
                return x_values

            theta_rad = np.radians(x_values) / 2
            q_values = (4 * np.pi * np.sin(theta_rad)) / lambda_in
            return q_values

        elif '2theta' in input_format and output_format == 'd-spacing':
            if lambda_in is None:
                st.error("Input wavelength required for 2theta to d-spacing conversion")
                return x_values

            theta_rad = np.radians(x_values / 2)
            valid = np.abs(np.sin(theta_rad)) > 1e-6

            result = np.zeros_like(x_values)
            result[valid] = lambda_in / (2 * np.sin(theta_rad[valid]))
            result[~valid] = np.nan
            return result

        elif input_format == 'd-spacing' and '2theta' in output_format:
            if lambda_out is None:
                st.error("Output wavelength required for d-spacing to 2theta conversion")
                return x_values

            valid = x_values > 0
            sin_arg = lambda_out / (2 * x_values[valid])
            sin_arg = np.clip(sin_arg, 0, 1)

            theta = np.degrees(np.arcsin(sin_arg))
            result = np.zeros_like(x_values)
            result[valid] = 2 * theta
            result[~valid] = np.nan
            return result

        elif '2theta' in input_format and '2theta' in output_format:
            if lambda_in is None or lambda_out is None:
                st.error("Both wavelengths required for 2theta to 2theta conversion")
                return x_values

            if abs(lambda_in - lambda_out) < 1e-6:
                return x_values

            theta_rad = np.radians(x_values / 2)
            valid = np.abs(np.sin(theta_rad)) > 1e-6
            d = np.zeros_like(x_values)
            d[valid] = lambda_in / (2 * np.sin(theta_rad[valid]))

            sin_arg = lambda_out / (2 * d[valid])
            sin_arg = np.clip(sin_arg, 0, 1)
            theta_new = np.degrees(np.arcsin(sin_arg))

            result = np.zeros_like(x_values)
            result[valid] = 2 * theta_new
            result[~valid] = np.nan
            return result

        else:
            st.warning(f"No conversion logic for {input_format} to {output_format}")
            return x_values

    except Exception as e:
        st.error(f"Error during conversion: {e}")
        return x_values


def apply_slit_conversion(x_data_2theta, y_data, slit_type, fixed_slit_size, irradiated_length):
    if slit_type == "No conversion":
        return y_data

    theta_rad = np.radians(x_data_2theta / 2)
    valid_mask = np.abs(np.sin(theta_rad)) > 1e-6

    y_converted = np.copy(y_data)

    if slit_type == "Auto slit to fixed slit":
        adjustment_factor = np.ones_like(y_data)
        adjustment_factor[valid_mask] = fixed_slit_size / (
                irradiated_length * np.sin(theta_rad[valid_mask])
        )
        y_converted = y_data * adjustment_factor

    elif slit_type == "Fixed slit to auto slit":
        adjustment_factor = np.ones_like(y_data)
        adjustment_factor[valid_mask] = (
                irradiated_length * np.sin(theta_rad[valid_mask]) / fixed_slit_size
        )
        y_converted = y_data * adjustment_factor

    return y_converted


def apply_y_transformations(y_data, normalize, y_scale):
    y_transformed = np.copy(y_data)

    if normalize:
        max_val = np.max(y_transformed)
        if max_val > 0:
            y_transformed = (y_transformed / max_val) * 100.0

    if y_scale == "log":
        y_transformed = np.where(y_transformed > 0, np.log10(y_transformed), np.nan)
    elif y_scale == "sqrt":
        y_transformed = np.where(y_transformed >= 0, np.sqrt(y_transformed), np.nan)

    return y_transformed


def get_axis_label(format_type, wavelength=None):
    labels = {
        '2theta_Cu': '2θ (°) [Cu Kα, λ=1.54056Å]',
        '2theta_Co': '2θ (°) [Co Kα, λ=1.78897Å]',
        '2theta_Mo': '2θ (°) [Mo Kα, λ=0.70932Å]',
        '2theta_Cr': '2θ (°) [Cr Kα, λ=2.28970Å]',
        '2theta_Fe': '2θ (°) [Fe Kα, λ=1.93604Å]',
        '2theta_Ag': '2θ (°) [Ag Kα, λ=0.55941Å]',
        '2theta_custom': f'2θ (°) [λ={wavelength:.5f}Å]' if wavelength else '2θ (°)',
        'd-spacing': 'd-spacing (Å)',
        'q-vector': 'q (Å⁻¹)'
    }
    return labels.get(format_type, 'X-axis')


def get_y_axis_label(normalize, y_scale):
    if normalize and y_scale == "linear":
        return "Normalized Intensity (%)"
    elif normalize and y_scale == "log":
        return "log₁₀(Normalized Intensity)"
    elif normalize and y_scale == "sqrt":
        return "√(Normalized Intensity)"
    elif y_scale == "log":
        return "log₁₀(Intensity)"
    elif y_scale == "sqrt":
        return "√(Intensity)"
    else:
        return "Intensity (counts)"


def run_axis_converter():
    st.markdown("### 🔄 XRD Data X/Y-Axis Converter")
    with st.expander(f"How to **Cite**", icon="📚", expanded=False):
        st.markdown("""
        If you like the app, please cite the following source:
        - **XRDlicious, 2025** – [Lebeda, Miroslav, et al. XRDlicious: an interactive web-based platform for online calculation of diffraction patterns and radial distribution functions from crystal structures. Applied Crystallography, 2025, 58.5.](https://doi.org/10.1107/S1600576725005370).
        """)
    st.info(
        "📊 Convert your XRD data between different x-axis formats: "
        "**2θ** (different wavelengths) ↔️ **d-spacing** ↔️ **q-vector**, "
        "apply **divergence slit** corrections (fixed ↔️ auto), and "
        "**remove the background** from the intensity. "
        "Upload single or multiple files - batch mode activates automatically."
    )

    # The uploader key holds a counter. Incrementing it (in the clear button
    # callback below) forces Streamlit to mount a fresh, empty file_uploader,
    # which is the reliable way to programmatically remove all uploaded files.
    if "axis_uploader_key" not in st.session_state:
        st.session_state.axis_uploader_key = 0

    def _clear_axis_uploaded_files():
        st.session_state.axis_uploader_key += 1

    uploaded_files_raw = st.file_uploader(
        "Upload XRD Data File(s) (.xy, .txt, .dat, .csv)",
        type=["xy", "txt", "dat", "csv", "data"],
        accept_multiple_files=True,
        key=f"axis_file_uploader_{st.session_state.axis_uploader_key}"
    )

    if not uploaded_files_raw:
        st.info("👆 Upload your data file(s) to begin")
        return

    uploaded_files = uploaded_files_raw if isinstance(uploaded_files_raw, list) else [uploaded_files_raw]

    is_batch = len(uploaded_files) > 1

    # Which file is shown in the preview column. In batch mode a slider below
    # the plot lets the user switch files; its value is stored in session_state
    # so it can be read here (before the widget itself is rendered).
    preview_idx = 0
    if is_batch:
        names = [f.name for f in uploaded_files]
        stored_name = st.session_state.get("preview_file_name")
        if stored_name in names:
            preview_idx = names.index(stored_name)
        elif stored_name is not None:
            # Uploaded files changed; drop the stale selection so the
            # select_slider below doesn't error on an invalid option.
            del st.session_state["preview_file_name"]

    first_file = uploaded_files[preview_idx]

    file_content = first_file.getvalue().decode("utf-8", errors='replace')
    data_df = parse_xy_simple(file_content)

    if data_df is None:
        st.error("Failed to parse the uploaded file.")
        return

    col1, col2 = st.columns([1, 1.5])

    with col1:
        hdr_col, btn_col = st.columns([1.5, 1])
        with hdr_col:
            st.markdown("#### ⚙️ Conversion Settings")
        with btn_col:
            st.markdown(
                """
                <style>
                /* Friendly blue for the primary action buttons
                   (apply / prepare / download). */
                button[data-testid^="stBaseButton-primary"] {
                    background-color: #3b82f6;
                    border-color: #3b82f6;
                    color: #ffffff;
                }
                button[data-testid^="stBaseButton-primary"]:hover {
                    background-color: #2563eb;
                    border-color: #2563eb;
                    color: #ffffff;
                }
                /* A deeper blue for the actual file-download buttons, to set
                   them apart from the "prepare / apply" buttons. */
                [data-testid="stDownloadButton"] button {
                    background-color: #0e4d92;
                    border-color: #0e4d92;
                    color: #ffffff;
                }
                [data-testid="stDownloadButton"] button:hover {
                    background-color: #0a3a6e;
                    border-color: #0a3a6e;
                    color: #ffffff;
                }
                /* Light gray for the "Remove all files" button (more
                   specific, so it wins over the blue rule above). */
                .st-key-remove_all_files_axis button[data-testid^="stBaseButton-primary"] {
                    background-color: #9ca3af;
                    border-color: #9ca3af;
                    color: #ffffff;
                }
                .st-key-remove_all_files_axis button[data-testid^="stBaseButton-primary"]:hover {
                    background-color: #868e96;
                    border-color: #868e96;
                    color: #ffffff;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.button("🗑️ Remove all files",
                      key="remove_all_files_axis",
                      on_click=_clear_axis_uploaded_files,
                      type="primary",
                      use_container_width=True)

        if is_batch:
            st.success(f"✅ **Batch mode active:** {len(uploaded_files)} files uploaded")
            st.caption("Settings will apply to all files")

        tab1, tab2, tab3 = st.tabs(
            ["📐 X-Axis", "📊 Y-Axis", "🧹 Background"])

        with tab1:
            st.markdown("##### Format Conversion")

            input_format = st.selectbox(
                "Input data format:",
                options=[
                    'No conversion',
                    '2theta_Cu',
                    '2theta_Co',
                    '2theta_Mo',
                    '2theta_Cr',
                    '2theta_Fe',
                    '2theta_Ag',
                    '2theta_custom',
                    'd-spacing',
                    'q-vector'
                ],
                format_func=lambda x: {
                    'No conversion': 'No conversion',
                    '2theta_Cu': '2θ (Copper Kα, λ=1.54056Å)',
                    '2theta_Co': '2θ (Cobalt Kα, λ=1.78897Å)',
                    '2theta_Mo': '2θ (Molybdenum Kα, λ=0.70932Å)',
                    '2theta_Cr': '2θ (Chromium Kα, λ=2.28970Å)',
                    '2theta_Fe': '2θ (Iron Kα, λ=1.93604Å)',
                    '2theta_Ag': '2θ (Silver Kα, λ=0.55941Å)',
                    '2theta_custom': '2θ (Custom wavelength)',
                    'd-spacing': 'd-spacing (Å)',
                    'q-vector': 'q-vector (Å⁻¹)'
                }[x]
            )

            input_wavelength = None
            if input_format == '2theta_custom':
                input_wavelength = st.number_input(
                    "Input wavelength (Å):",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.54056,
                    step=0.00001,
                    format="%.5f"
                )

            if input_format != 'No conversion':
                output_options = [
                    '2theta_Cu',
                    '2theta_Co',
                    '2theta_Mo',
                    '2theta_Cr',
                    '2theta_Fe',
                    '2theta_Ag',
                    '2theta_custom',
                    'd-spacing',
                    'q-vector'
                ]

                if input_format in output_options and input_format != '2theta_custom':
                    output_options.remove(input_format)

                output_format = st.selectbox(
                    "Convert to:",
                    options=output_options,
                    format_func=lambda x: {
                        '2theta_Cu': '2θ (Copper Kα, λ=1.54056Å)',
                        '2theta_Co': '2θ (Cobalt Kα, λ=1.78897Å)',
                        '2theta_Mo': '2θ (Molybdenum Kα, λ=0.70932Å)',
                        '2theta_Cr': '2θ (Chromium Kα, λ=2.28970Å)',
                        '2theta_Fe': '2θ (Iron Kα, λ=1.93604Å)',
                        '2theta_Ag': '2θ (Silver Kα, λ=0.55941Å)',
                        '2theta_custom': '2θ (Custom wavelength)',
                        'd-spacing': 'd-spacing (Å)',
                        'q-vector': 'q-vector (Å⁻¹)'
                    }[x]
                )

                output_wavelength = None
                if output_format == '2theta_custom':
                    output_wavelength = st.number_input(
                        "Output wavelength (Å):",
                        min_value=0.1,
                        max_value=10.0,
                        value=1.78897,
                        step=0.00001,
                        format="%.5f"
                    )
            else:
                output_format = None

            st.markdown("---")
            st.markdown("##### Divergence Slit Conversion")

            with st.expander("❓ How does Divergence Slit Conversion work?"):
                st.markdown("""
                ### Divergence Slit Conversion Explained

                #### Auto Slit
                - The slit **automatically adjusts** with angle (2θ) to **keep the irradiated area constant**.
                - Produces intensity that remains relatively **consistent** across angles.

                #### Fixed Slit
                - The slit has a **fixed opening angle**.
                - As 2θ increases, the **irradiated area is smaller**.
                - Results in **reduced intensity at higher angles**.

                #### Conversion Types

                - **Fixed Slit → Auto Slit**  
                  Adjusts for loss of intensity at higher angles by simulating constant irradiated area:

                  $$
                  \\text{Intensity}_{\\text{auto}} = \\text{Intensity}_{\\text{fixed}} \\times \\frac{\\text{Irradiated Length} \\times \\sin(\\theta)}{\\text{Fixed Slit Size}}
                  $$

                - **Auto Slit → Fixed Slit**  
                  Simulates reduced illuminated area at higher angles:

                  $$
                  \\text{Intensity}_{\\text{fixed}} = \\text{Intensity}_{\\text{auto}} \\times \\frac{\\text{Fixed Slit Size}}{\\text{Irradiated Length} \\times \\sin(\\theta)}
                  $$

                #### Parameters

                - **Fixed slit size (degrees)**: The opening angle of the slit in degrees.
                - **Irradiated sample length (mm)**: Physical length of sample that is illuminated.
                  - Reflection geometry: *10–20 mm*  
                  - Transmission geometry: *1–2 mm*

                **Note:** Slit conversion only works when data is in 2θ format.
                """)

            slit_conversion_type = st.selectbox(
                "Slit conversion type:",
                options=[
                    "No conversion",
                    "Auto slit to fixed slit",
                    "Fixed slit to auto slit"
                ]
            )

            fixed_slit_size = None
            irradiated_length = None

            if slit_conversion_type != "No conversion":
                col_slit1, col_slit2 = st.columns(2)
                with col_slit1:
                    fixed_slit_size = st.number_input(
                        "Fixed slit size (degrees)",
                        min_value=0.1,
                        max_value=2.0,
                        value=1.0,
                        step=0.1,
                        format="%.2f"
                    )
                with col_slit2:
                    irradiated_length = st.number_input(
                        "Irradiated sample length (mm)",
                        min_value=1.0,
                        max_value=50.0,
                        value=10.0,
                        step=1.0
                    )
                st.caption("Typical: 10-20 mm (reflection), 1-2 mm (transmission)")

        with tab2:
            st.markdown("##### Intensity Transformations")

            normalize_y = st.checkbox(
                "Normalize to maximum",
                value=False,
                help="Scale intensity so maximum = 100%"
            )

            y_scale = st.selectbox(
                "Intensity scale:",
                options=["linear", "log", "sqrt"],
                format_func=lambda x: {
                    "linear": "Linear",
                    "log": "Logarithmic (log₁₀)",
                    "sqrt": "Square root (√)"
                }[x],
                help="Log and sqrt scales can help visualize weak peaks"
            )

            st.markdown("---")
            st.markdown("##### Scale Information")

            if y_scale == "log":
                st.info(
                    "**Logarithmic scale** compresses high intensities and expands low intensities, making weak peaks more visible.")
            elif y_scale == "sqrt":
                st.info(
                    "**Square root scale** provides a middle ground between linear and log, often used in XRD to enhance weak peaks while maintaining relative intensities.")
            else:
                st.info("**Linear scale** shows raw intensity values without transformation.")

            if normalize_y:
                st.success("✅ Normalization will scale the strongest peak to 100%")

        with tab3:
            st.markdown("##### Background Removal")
            st.caption(
                "Estimate and subtract the background from the intensity. "
                "The result is previewed on the right; use the toggle below "
                "the graph to apply it to the converted/downloaded data."
            )

            bg_method = st.radio(
                "Background Estimation Method",
                BG_METHODS,
                index=0,
                key="bg_method",
                horizontal=True,
            )

            bg_params = {}
            if bg_method == "Polynomial Fit":
                bg_params["poly_degree"] = st.slider(
                    "Polynomial Degree", 1, 15, 6, 1, key="bg_poly_degree",
                    help="Higher degree follows the baseline more closely.")
                bg_params["n_iters"] = st.slider(
                    "Iterations", 1, 200, 40, 1, key="bg_poly_iters",
                    help="More iterations suppress peaks more aggressively "
                         "before fitting.")
            elif bg_method == "SNIP Algorithm":
                bg_params["snip_iters"] = st.slider(
                    "SNIP Iterations (M)", 1, 100, 20, 1, key="bg_snip_iters",
                    help="Controls the maximum half-width of spectral features "
                         "that are removed. Larger M removes broader peaks / "
                         "follows a more gradually varying baseline.")
            elif bg_method == "Rolling Ball Algorithm":
                bg_params["ball_radius"] = st.slider(
                    "Ball Radius", 1, 100, 30, 1, key="bg_ball_radius")
                bg_params["ball_smoothing"] = st.slider(
                    "Smoothing Passes", 0, 10, 3, 1, key="bg_ball_smoothing")
            elif bg_method == "airPLS (Adaptive Baseline)":
                bg_params["lam"] = st.select_slider(
                    "Smoothness (λ)",
                    options=[1e3, 1e4, 1e5, 5e5, 1e6, 5e6, 1e7, 1e8],
                    value=1e6, format_func=lambda v: f"{v:.0e}",
                    key="bg_airpls_lam",
                    help="Higher = smoother/flatter baseline. 1e5–1e7 suits "
                         "most diffraction data.")
                bg_params["p"] = st.slider(
                    "Asymmetry (p)", 0.001, 0.05, 0.01, 0.001, format="%.3f",
                    key="bg_airpls_p",
                    help="Fraction of points considered background. "
                         "Lower = baseline hugs the valley more tightly.")
                bg_params["n_iter"] = st.slider(
                    "Iterations", 5, 50, 15, 1, key="bg_airpls_iters")

            if bg_method != "None":
                st.success(
                    "✅ Background removal is applied to the downloaded data."
                    + (" (same method/parameters for every file in batch mode)"
                       if is_batch else ""))

        # Background subtraction is applied to the output whenever a method
        # other than "None" is selected.
        bg_active = bg_method != "None"

        st.markdown("---")

        st.markdown("#### 💾 Download Options")

        include_header = st.checkbox("Include header in output file", value=True)

        any_conversion = (input_format != 'No conversion' and output_format) or \
                         slit_conversion_type != "No conversion" or \
                         normalize_y or \
                         y_scale != "linear" or \
                         bg_active

        if any_conversion:
            if is_batch:
                if st.button("📦 Prepare All Converted Files for Download (.zip)",
                             type="primary",
                             width='stretch'):

                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for uploaded_file in uploaded_files:
                            content = uploaded_file.getvalue().decode("utf-8", errors='replace')
                            df = parse_xy_simple(content)

                            if df is not None:
                                x_data = df['X'].values
                                y_data = df['Y'].values

                                if bg_active:
                                    y_data = remove_background(
                                        x_data, y_data, bg_method, bg_params)

                                if input_format != 'No conversion' and output_format:
                                    x_converted = convert_xaxis_data(
                                        x_data,
                                        input_format,
                                        output_format,
                                        input_wavelength,
                                        output_wavelength
                                    )

                                    valid_mask = ~np.isnan(x_converted)
                                    x_data = x_converted[valid_mask]
                                    y_data = y_data[valid_mask]

                                is_2theta = False
                                if input_format == 'No conversion':
                                    is_2theta = True
                                elif output_format and '2theta' in output_format:
                                    is_2theta = True

                                if slit_conversion_type != "No conversion" and is_2theta:
                                    y_data = apply_slit_conversion(
                                        x_data,
                                        y_data,
                                        slit_conversion_type,
                                        fixed_slit_size,
                                        irradiated_length
                                    )

                                y_data = apply_y_transformations(y_data, normalize_y, y_scale)

                                valid_mask = ~np.isnan(y_data)
                                x_data = x_data[valid_mask]
                                y_data = y_data[valid_mask]

                                output_df = pd.DataFrame({
                                    'X': x_data,
                                    'Y': y_data
                                })

                                output = StringIO()
                                if include_header:
                                    if output_format:
                                        x_label = get_axis_label(output_format, output_wavelength)
                                    else:
                                        x_label = "2θ (°)"
                                    y_label = get_y_axis_label(normalize_y, y_scale)
                                    output.write(f"# {x_label}\t{y_label}\n")

                                output_df.to_csv(output, sep='\t', header=False,
                                                 index=False, float_format='%.6f')

                                new_filename = uploaded_file.name.rsplit('.', 1)[0] + '_converted.xy'
                                zf.writestr(new_filename, output.getvalue())

                    st.download_button(
                        label="⬇️ Download ZIP",
                        data=zip_buffer.getvalue(),
                        file_name="converted_xrd_files.zip",
                        mime="application/zip",
                        type = "primary",
                        width='stretch'
                    )
            else:
                default_name = first_file.name.rsplit('.', 1)[0] + '_converted.xy'
                download_filename = st.text_input("Output filename:", default_name)

                x_data = data_df['X'].values
                y_data = data_df['Y'].values

                if bg_active:
                    y_data = remove_background(
                        x_data, y_data, bg_method, bg_params)

                if input_format != 'No conversion' and output_format:
                    x_converted = convert_xaxis_data(
                        x_data,
                        input_format,
                        output_format,
                        input_wavelength,
                        output_wavelength
                    )

                    valid_mask = ~np.isnan(x_converted)
                    x_data = x_converted[valid_mask]
                    y_data = y_data[valid_mask]

                is_2theta = False
                if input_format == 'No conversion':
                    is_2theta = True
                elif output_format and '2theta' in output_format:
                    is_2theta = True

                if slit_conversion_type != "No conversion" and is_2theta:
                    y_data = apply_slit_conversion(
                        x_data,
                        y_data,
                        slit_conversion_type,
                        fixed_slit_size,
                        irradiated_length
                    )

                y_data = apply_y_transformations(y_data, normalize_y, y_scale)

                valid_mask = ~np.isnan(y_data)
                x_data = x_data[valid_mask]
                y_data = y_data[valid_mask]

                output = StringIO()
                if include_header:
                    if output_format:
                        x_label = get_axis_label(output_format, output_wavelength)
                    else:
                        x_label = "2θ (°)"
                    y_label = get_y_axis_label(normalize_y, y_scale)
                    output.write(f"# {x_label}\t{y_label}\n")

                output_df = pd.DataFrame({'X': x_data, 'Y': y_data})
                output_df.to_csv(output, sep='\t', header=False,
                                 index=False, float_format='%.6f')

                st.download_button(
                    label="⬇️ Download Converted File",
                    data=output.getvalue(),
                    file_name=download_filename,
                    mime="text/plain",
                    type="primary",
                    width='stretch'
                )

    with col2:
        st.markdown("#### 📊 Data Preview & Comparison")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data_df['X'],
            y=data_df['Y'],
            mode='lines',
            name='Original',
            line=dict(color='#0984e3', width=2)
        ))

        raw_x = data_df['X'].values
        raw_y = data_df['Y'].values

        # Estimate the background on the raw pattern (before any x-conversion).
        bg_curve = None
        if bg_method != "None":
            try:
                bg_curve = compute_background(raw_x, raw_y, bg_method, bg_params)
            except Exception as e:
                st.warning(f"⚠️ Background estimation failed: {e}")
                bg_curve = None

        if bg_curve is not None:
            fig.add_trace(go.Scatter(
                x=raw_x, y=bg_curve, mode='lines', name='Background',
                line=dict(color='#636e72', width=2, dash='dot')
            ))
            fig.add_trace(go.Scatter(
                x=raw_x, y=np.maximum(0.0, raw_y - bg_curve), mode='lines',
                name='Background subtracted',
                line=dict(color='#00b894', width=2)
            ))

        # Intensity that feeds the conversion pipeline: background-subtracted
        # only when the user opted to apply it (toggle below the graph).
        if bg_active and bg_curve is not None:
            base_y = np.maximum(0.0, raw_y - bg_curve)
        else:
            base_y = raw_y

        x_display = raw_x
        y_display = base_y
        x_axis_title = "X-axis"

        if input_format != 'No conversion' and output_format:
            x_converted = convert_xaxis_data(
                raw_x,
                input_format,
                output_format,
                input_wavelength,
                output_wavelength
            )

            valid_mask = ~np.isnan(x_converted)
            x_display = x_converted[valid_mask]
            y_display = base_y[valid_mask]

            x_axis_title = get_axis_label(output_format, output_wavelength)

            if len(x_display) < len(data_df):
                st.warning(
                    f"⚠️ {len(data_df) - len(x_display)} points were removed "
                    "due to invalid conversion values"
                )

        is_2theta = False
        if input_format == 'No conversion':
            is_2theta = True
            x_axis_title = "2θ (°)"
        elif output_format and '2theta' in output_format:
            is_2theta = True

        if slit_conversion_type != "No conversion" and is_2theta:
            y_display = apply_slit_conversion(
                x_display,
                y_display,
                slit_conversion_type,
                fixed_slit_size,
                irradiated_length
            )
        elif slit_conversion_type != "No conversion" and not is_2theta:
            st.warning("⚠️ Slit conversion only works when data is in 2θ format")

        y_display = apply_y_transformations(y_display, normalize_y, y_scale)

        valid_mask = ~np.isnan(y_display)
        x_display = x_display[valid_mask]
        y_display = y_display[valid_mask]

        y_axis_title = get_y_axis_label(normalize_y, y_scale)

        any_conversion = (input_format != 'No conversion' and output_format) or \
                         (slit_conversion_type != "No conversion" and is_2theta) or \
                         normalize_y or \
                         y_scale != "linear"

        if any_conversion:
            fig.add_trace(go.Scatter(
                x=x_display,
                y=y_display,
                mode='lines',
                name='Converted',
                line=dict(color='#d63031', width=2, dash='dash')
            ))

        fig.update_layout(
            title=f"Preview: {first_file.name}" +
                  (f" (showing 1 of {len(uploaded_files)} files)" if is_batch else ""),
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            xaxis=dict(title_font=dict(size=22, color="black"), tickfont=dict(size=20, color="black")),
            yaxis=dict(title_font=dict(size=22, color="black"), tickfont=dict(size=20, color="black")),
            height=600,
            hovermode='x unified',
            font=dict(size=20, color="black"),
            title_font=dict(size=22, color="black"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font = dict(size=18, color="black"),
            )
        )

        st.plotly_chart(fig)

        if is_batch:
            st.select_slider(
                "Preview file (with applied conversion)",
                options=[f.name for f in uploaded_files],
                key="preview_file_name",
            )
            st.caption(
                f"Previewing file {preview_idx + 1} of {len(uploaded_files)}: "
                f"**{first_file.name}**")

        if any_conversion:
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Original Range (X)",
                          f"{data_df['X'].min():.3f} - {data_df['X'].max():.3f}")
                st.metric("Original Max (Y)",
                          f"{data_df['Y'].max():.1f}")
            with col_stat2:
                st.metric("Converted Range (X)",
                          f"{x_display.min():.3f} - {x_display.max():.3f}")
                st.metric("Converted Max (Y)",
                          f"{y_display.max():.2f}")

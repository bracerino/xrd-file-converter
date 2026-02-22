"""
chi_scan_section.py
--------------------
Analyser for 2D chi-scan diffraction data (e.g. PANalytical Empyrean CSV).

Data format expected:
  - File contains a metadata header followed by [Scan points]
  - Under [Scan points]: columns  "2Theta position, Chi position, Intensity"

Provides:
  • 3-D interactive surface plot
  • Interactive heatmap (2θ × χ, colour = intensity)
  • Per-chi-slice 1-D plot with background subtraction overlay
  • Batch .xy download (one file per chi, chi value in filename)
"""

import io
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from scipy.signal import savgol_filter as _savgol
from scipy.sparse import diags as _sp_diags
from scipy.sparse.linalg import spsolve as _spsolve


FONT_SIZE  = 18
HOVER_FONT = 17


def _parse_chi_csv(uploaded_file) -> tuple[dict, pd.DataFrame]:
    raw = uploaded_file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="replace")
    uploaded_file.seek(0)

    lines = text.splitlines()

    metadata = {}
    scan_line = None
    for i, line in enumerate(lines):
        if "[Scan points]" in line:
            scan_line = i
            break
        if "," in line:
            parts = line.split(",", 1)
            key = parts[0].strip()
            val = parts[1].strip().strip('"') if len(parts) > 1 else ""
            if key:
                metadata[key] = val

    if scan_line is None:
        st.error("Could not locate '[Scan points]' section in the file.")
        return metadata, pd.DataFrame()

    data_start = scan_line + 2  # skip [Scan points] + header row
    data_text = "\n".join(lines[data_start:])
    try:
        df = pd.read_csv(
            io.StringIO(data_text),
            header=None,
            names=["two_theta", "chi", "intensity"],
            skipinitialspace=True,
        )
        df = df.dropna().astype(float)
    except Exception as exc:
        st.error(f"Failed to parse data section: {exc}")
        return metadata, pd.DataFrame()

    return metadata, df


def _bg_poly(y: np.ndarray, degree: int, n_iters: int) -> np.ndarray:
    x = np.arange(len(y), dtype=float)
    ywork = y.copy()
    for _ in range(n_iters):
        try:
            coeffs = np.polyfit(x, ywork, degree)
            fitted = np.polyval(coeffs, x)
        except Exception:
            fitted = ywork
        ywork = np.minimum(ywork, fitted)
    coeffs = np.polyfit(x, ywork, degree)
    return np.polyval(coeffs, x)


def _bg_snip(y: np.ndarray, n_iters: int, window: int) -> np.ndarray:
    half = window // 2
    bg = y.copy()
    for _ in range(n_iters):
        for i in range(half, len(y) - half):
            bg[i] = min(bg[i], 0.5 * (bg[i - half] + bg[i + half]))
    return bg


def _bg_rolling_ball(y: np.ndarray, radius: int) -> np.ndarray:
    return np.array([
        np.min(y[max(0, i - radius): min(len(y), i + radius + 1)])
        for i in range(len(y))
    ])


def _bg_airpls(y: np.ndarray, lam: float, p: float, n_iter: int) -> np.ndarray:
    n = len(y)
    D = _sp_diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
    H = lam * D.T.dot(D)
    w = np.ones(n)
    z = y.copy()
    for _ in range(n_iter):
        W = _sp_diags(w, 0, shape=(n, n))
        z = _spsolve(W + H, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def _bg_sonneveld_visser(y: np.ndarray, n_iters: int, bending: float,
                          sample_fraction: float) -> np.ndarray:

    n = len(y)
    step = max(1, round(1.0 / sample_fraction))
    idx_sparse = np.arange(0, n, step)
    if idx_sparse[-1] != n - 1:
        idx_sparse = np.append(idx_sparse, n - 1)
    bg_sparse = y[idx_sparse].copy()

    bg = np.interp(np.arange(n), idx_sparse, bg_sparse)

    for _ in range(n_iters):
        bg_new = bg.copy()
        for i in range(1, n - 1):
            m_i = 0.5 * (bg[i - 1] + bg[i + 1])
            if bg[i] > m_i + bending:
                bg_new[i] = m_i
        bg = bg_new

    return bg


def _apply_background(y: np.ndarray, method: str, params: dict) -> np.ndarray:
    if method == "Polynomial Fit":
        bg = _bg_poly(y, params["poly_degree"], params["poly_iters"])
    elif method == "SNIP Algorithm":
        bg = _bg_snip(y, params["snip_iters"], params["snip_window"])
    elif method == "Rolling Ball":
        bg = _bg_rolling_ball(y, params["ball_radius"])
    elif method == "airPLS":
        bg = _bg_airpls(y, params["airpls_lam"], params["airpls_p"],
                        params["airpls_iters"])
    elif method == "Sonneveld-Visser":
        bg = _bg_sonneveld_visser(y, params["sv_iters"], params["sv_bending"],
                                   params["sv_fraction"])
    else:
        bg = np.zeros_like(y)
    return bg


def _subtract_all_chi(df: pd.DataFrame, method: str, params: dict) -> pd.DataFrame:
    result = df.copy()
    for chi_val in sorted(df["chi"].unique()):
        mask = df["chi"] == chi_val
        y = df.loc[mask, "intensity"].values.astype(float)
        bg = _apply_background(y, method, params)
        result.loc[mask, "intensity"] = np.maximum(0.0, y - bg)
    return result



def _make_heatmap(df: pd.DataFrame, title: str,
                   zmin: float | None = None,
                   zmax: float | None = None) -> go.Figure:
    chi_vals = sorted(df["chi"].unique())

    grid = pd.pivot_table(
        df, values="intensity", index="chi", columns="two_theta", aggfunc="mean"
    ).reindex(index=chi_vals)

    fig = go.Figure(go.Heatmap(
        z=grid.values,
        x=grid.columns.tolist(),
        y=chi_vals,
        colorscale="Viridis",
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title="Intensity"),
        hovertemplate="2θ: %{x:.3f}°<br>χ: %{y:.1f}°<br>Intensity: %{z:.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        xaxis_title="2θ (°)",
        yaxis_title="χ (°)",
        xaxis=dict(tickfont=dict(size=FONT_SIZE), title_font=dict(size=FONT_SIZE)),
        yaxis=dict(tickfont=dict(size=FONT_SIZE), title_font=dict(size=FONT_SIZE)),
        font=dict(size=FONT_SIZE),
        hoverlabel=dict(font=dict(size=HOVER_FONT)),
        height=500,
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig


def _make_surface(df: pd.DataFrame, title: str,
                   zmin: float | None = None,
                   zmax: float | None = None) -> go.Figure:
    chi_vals = sorted(df["chi"].unique())

    grid = pd.pivot_table(
        df, values="intensity", index="chi", columns="two_theta", aggfunc="mean"
    ).reindex(index=chi_vals)

    tt_arr  = np.array(grid.columns.tolist())
    chi_arr = np.array(chi_vals)
    TT, CHI = np.meshgrid(tt_arr, chi_arr)

    z_data = grid.values.copy().astype(float)
    if zmin is not None:
        z_data = np.where(z_data < zmin, np.nan, z_data)
    if zmax is not None:
        z_data = np.where(z_data > zmax, np.nan, z_data)

    axis_font  = dict(size=FONT_SIZE)
    tick_font  = dict(size=FONT_SIZE - 1)

    fig = go.Figure(go.Surface(
        x=TT,
        y=CHI,
        z=z_data,
        colorscale="Viridis",
        cmin=zmin,
        cmax=zmax,
        colorbar=dict(
            title=dict(text="Intensity", font=dict(size=FONT_SIZE)),
            tickfont=dict(size=FONT_SIZE - 1),
        ),
        hovertemplate=(
            "<span style=\'font-size:14px\'>"
            "2θ: %{x:.3f}°<br>χ: %{y:.1f}°<br>Intensity: %{z:.0f}"
            "</span><extra></extra>"
        ),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        scene=dict(
            xaxis=dict(
                title=dict(text="2θ (°)", font=axis_font),
                tickfont=tick_font,
                showticklabels=True,
                showaxeslabels=True,
                visible=True,
            ),
            yaxis=dict(
                title=dict(text="χ (°)", font=axis_font),
                tickfont=tick_font,
                showticklabels=True,
                showaxeslabels=True,
                visible=True,
            ),
            zaxis=dict(
                title=dict(text="Intensity", font=axis_font),
                tickfont=tick_font,
                showticklabels=True,
                showaxeslabels=True,
                visible=True,
            ),
            bgcolor="rgba(30,30,50,0.0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=FONT_SIZE),
        hoverlabel=dict(font=dict(size=HOVER_FONT)),
        height=620,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def _chi_mask(df: pd.DataFrame, chi_val: float) -> pd.Series:
    return np.isclose(df["chi"].values, chi_val, atol=1e-6)


def _make_chi_slice(df_raw: pd.DataFrame, df_bg_sub: pd.DataFrame | None,
                    chi_val: float, method: str) -> go.Figure:
    mask = _chi_mask(df_raw, chi_val)
    sub  = df_raw.loc[mask].sort_values("two_theta")
    tt   = sub["two_theta"].values
    y    = sub["intensity"].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tt, y=y, mode="lines", name="Raw",
        line=dict(color="#2563eb", width=1.8),
    ))

    if df_bg_sub is not None and method != "None":
        mask2 = _chi_mask(df_bg_sub, chi_val)
        sub2  = df_bg_sub.loc[mask2].sort_values("two_theta")
        y_sub = sub2["intensity"].values

        bg = np.maximum(0.0, y - y_sub)
        fig.add_trace(go.Scatter(
            x=tt, y=bg, mode="lines", name="Background",
            line=dict(color="#dc2626", width=1.2, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=tt, y=y_sub, mode="lines", name="Background subtracted",
            line=dict(color="#16a34a", width=1.8),
        ))

    fig.update_layout(
        title=dict(text=f"χ = {chi_val:.1f}°", font=dict(size=FONT_SIZE + 2)),
        xaxis=dict(title=dict(text="2θ (°)", font=dict(size=FONT_SIZE)),
                   tickfont=dict(size=FONT_SIZE)),
        yaxis=dict(title=dict(text="Intensity (a.u.)", font=dict(size=FONT_SIZE)),
                   tickfont=dict(size=FONT_SIZE)),
        font=dict(size=FONT_SIZE),
        hoverlabel=dict(font=dict(size=HOVER_FONT)),
        legend=dict(font=dict(size=FONT_SIZE),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=470,
        margin=dict(l=60, r=30, t=80, b=50),
    )
    return fig


def _make_zip(df: pd.DataFrame, base_name: str, subtract: bool,
              df_sub: pd.DataFrame | None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for chi_val in sorted(df["chi"].unique()):
            mask  = df["chi"] == chi_val
            tt    = df.loc[mask, "two_theta"].values
            y_raw = df.loc[mask, "intensity"].values

            if subtract and df_sub is not None:
                mask2 = df_sub["chi"] == chi_val
                y_out = df_sub.loc[mask2, "intensity"].values
            else:
                y_out = y_raw

            chi_str   = f"{chi_val:.2f}".replace(".", "p")
            filename  = f"{base_name}_chi{chi_str}.xy"
            lines     = [f"# 2Theta  Intensity  (chi = {chi_val:.2f} deg)\n"]
            lines    += [f"{t:.6f}  {i:.4f}\n" for t, i in zip(tt, y_out)]
            zf.writestr(filename, "".join(lines))
    return buf.getvalue()


def _bg_settings_ui() -> tuple[str, dict]:
    method = st.selectbox(
        "Background method",
        ["None", "Polynomial Fit", "SNIP Algorithm",
         "Rolling Ball", "airPLS", "Sonneveld-Visser"],
        index=5,
        key="chi_bg_method",
    )

    params: dict = {}

    if method == "Polynomial Fit":
        params["poly_degree"] = st.slider("Degree", 1, 15, 6, key="chi_poly_deg")
        params["poly_iters"]  = st.slider("Iterations", 1, 200, 40, key="chi_poly_it")

    elif method == "SNIP Algorithm":
        params["snip_iters"]  = st.slider("SNIP iterations", 1, 100, 20, key="chi_snip_it")
        w = st.slider("Window size", 3, 101, 21, 2, key="chi_snip_w")
        params["snip_window"] = w if w % 2 == 1 else w + 1

    elif method == "Rolling Ball":
        params["ball_radius"] = st.slider("Ball radius", 1, 200, 40, key="chi_ball_r")

    elif method == "airPLS":
        params["airpls_lam"]   = st.select_slider(
            "Smoothness (λ)",
            options=[1e3, 1e4, 1e5, 5e5, 1e6, 5e6, 1e7, 1e8],
            value=1e6,
            format_func=lambda v: f"{v:.0e}",
            key="chi_airpls_lam",
        )
        params["airpls_p"]     = st.slider("Asymmetry (p)", 0.001, 0.05, 0.01,
                                            0.001, format="%.3f", key="chi_airpls_p")
        params["airpls_iters"] = st.slider("Iterations", 5, 50, 15, key="chi_airpls_it")

    elif method == "Sonneveld-Visser":
        params["sv_iters"]    = st.slider(
            "Iterations", 5, 100, 30, key="chi_sv_iters",
            help="Number of neighbour-mean suppression passes. "
                 "More iterations → flatter background.")
        params["sv_bending"]  = st.number_input(
            "Bending factor c", value=0.0, min_value=0.0, max_value=500.0,
            step=1.0, format="%.1f", key="chi_sv_bending",
            help="Allows background to curve slightly. c=0 is strictest. "
                 "Increase if peaks are over-suppressed near bent regions.")
        params["sv_fraction"] = st.select_slider(
            "Sampling fraction",
            options=[0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20],
            value=0.02,
            format_func=lambda v: f"{v*100:.0f}%",
            key="chi_sv_fraction",
            help="Fraction of points used as initial background estimate "
                 "(Sonneveld & Visser used ~5%).")

    return method, params


def run_chi_scan_section():
    st.markdown("### 🌐 2D Chi-Scan Viewer & Analyser")
    st.info(
        "Upload a **PANalytical Empyrean** (or compatible) CSV file containing a "
        "chi-scan: columns *2Theta position*, *Chi position*, *Intensity*. "
        "The tool plots the full 2D dataset as a heatmap and 3D surface, lets you "
        "subtract background, inspect individual χ slices, and export all scans as "
        "**.xy** files in a ZIP archive."
    )

    uploaded = st.file_uploader(
        "Upload chi-scan CSV file",
        type=["csv", "txt", "dat"],
        key="chi_scan_upload",
    )

    if not uploaded:
        return

    metadata, df = _parse_chi_csv(uploaded)

    if df.empty:
        st.error("No data could be parsed from the file.")
        return

    chi_vals = sorted(df["chi"].unique())
    n_chi    = len(chi_vals)
    n_pts    = df["two_theta"].nunique()

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("χ positions", n_chi)
    col_info2.metric("2θ points per scan", n_pts)
    col_info3.metric("2θ range",
                     f"{df['two_theta'].min():.2f}° – {df['two_theta'].max():.2f}°")

    with st.expander("📋 File metadata", expanded=False):
        if metadata:
            meta_df = pd.DataFrame(list(metadata.items()), columns=["Parameter", "Value"])
            st.dataframe(meta_df, width='stretch', hide_index=True)
        else:
            st.info("No metadata found.")

    with st.sidebar:
        st.markdown("---")
        st.markdown("#### 🌐 Chi-Scan Settings")

        st.markdown("**Background subtraction**")
        bg_method, bg_params = _bg_settings_ui()

        st.markdown("**Download**")
        dl_subtract = st.checkbox(
            "Apply background subtraction to download",
            value=(bg_method != "None"),
            key="chi_dl_subtract",
        )

    cache_key = (uploaded.name, bg_method, repr(sorted(bg_params.items())))
    if (st.session_state.get("chi_bg_cache_key") != cache_key
            or "chi_df_subtracted" not in st.session_state):
        if bg_method != "None":
            with st.spinner("Applying background subtraction to all χ scans…"):
                st.session_state.chi_df_subtracted = _subtract_all_chi(
                    df, bg_method, bg_params)
            st.session_state.chi_bg_cache_key = cache_key
        else:
            st.session_state.chi_df_subtracted = None
            st.session_state.chi_bg_cache_key  = cache_key

    df_sub = st.session_state.get("chi_df_subtracted")

    if "chi_selected" not in st.session_state:
        st.session_state["chi_selected"] = chi_vals[len(chi_vals) // 2]
    elif st.session_state["chi_selected"] not in chi_vals:
        st.session_state["chi_selected"] = chi_vals[len(chi_vals) // 2]

    tab_hm, tab_3d, tab_slice, tab_dl = st.tabs([
        "🗺️ Heatmap",
        "🧊 3D Surface",
        "📈 χ slice",
        "💾 Download",
    ])

    base_name = uploaded.name.rsplit(".", 1)[0]

    with tab_hm:
        int_all  = df["intensity"].values
        int_min_global = float(np.percentile(int_all, 2))
        int_max_global = float(np.percentile(int_all, 98))

        hm_ctrl1, hm_ctrl2, hm_ctrl3, hm_ctrl4 = st.columns(4)
        zmin_raw = hm_ctrl1.number_input(
            "Color min (raw)", value=int_min_global, format="%.0f", key="hm_zmin_raw")
        zmax_raw = hm_ctrl2.number_input(
            "Color max (raw)", value=int_max_global, format="%.0f", key="hm_zmax_raw")
        if df_sub is not None and bg_method != "None":
            int_sub_all = df_sub["intensity"].values
            zmin_sub = hm_ctrl3.number_input(
                "Color min (subtracted)", value=float(np.percentile(int_sub_all, 2)),
                format="%.0f", key="hm_zmin_sub")
            zmax_sub = hm_ctrl4.number_input(
                "Color max (subtracted)", value=float(np.percentile(int_sub_all, 98)),
                format="%.0f", key="hm_zmax_sub")

        hm_col1, hm_col2 = st.columns([1, 1])
        with hm_col1:
            st.plotly_chart(
                _make_heatmap(df, "Raw intensity", zmin=zmin_raw, zmax=zmax_raw),
                width='stretch',
            )
        with hm_col2:
            if df_sub is not None and bg_method != "None":
                st.plotly_chart(
                    _make_heatmap(df_sub, f"Background-subtracted ({bg_method})",
                                  zmin=zmin_sub, zmax=zmax_sub),
                    width='stretch',
                )
            else:
                st.info("Select a background method in the sidebar to see the "
                        "subtracted heatmap here.")

    with tab_3d:
        int_all_3d = df["intensity"].values
        surf_ctrl1, surf_ctrl2, surf_ctrl3, surf_ctrl4 = st.columns(4)
        zmin_surf_raw = surf_ctrl1.number_input(
            "Z min (raw)", value=float(np.percentile(int_all_3d, 2)),
            format="%.0f", key="surf_zmin_raw")
        zmax_surf_raw = surf_ctrl2.number_input(
            "Z max (raw)", value=float(np.percentile(int_all_3d, 98)),
            format="%.0f", key="surf_zmax_raw")
        if df_sub is not None and bg_method != "None":
            int_sub_3d = df_sub["intensity"].values
            zmin_surf_sub = surf_ctrl3.number_input(
                "Z min (subtracted)", value=float(np.percentile(int_sub_3d, 2)),
                format="%.0f", key="surf_zmin_sub")
            zmax_surf_sub = surf_ctrl4.number_input(
                "Z max (subtracted)", value=float(np.percentile(int_sub_3d, 98)),
                format="%.0f", key="surf_zmax_sub")

        surf_col1, surf_col2 = st.columns([1, 1])
        with surf_col1:
            st.plotly_chart(
                _make_surface(df, "Raw intensity — 3D surface",
                              zmin=zmin_surf_raw, zmax=zmax_surf_raw),
                width='stretch',
            )
        with surf_col2:
            if df_sub is not None and bg_method != "None":
                st.plotly_chart(
                    _make_surface(df_sub,
                                  f"Background-subtracted — 3D surface ({bg_method})",
                                  zmin=zmin_surf_sub, zmax=zmax_surf_sub),
                    width='stretch',
                )
            else:
                st.info("Select a background method in the sidebar to see the "
                        "subtracted surface here.")

    with tab_slice:
        ctrl_col, plot_col = st.columns([1, 5])

        with ctrl_col:
            st.markdown("**χ (°)**")
            chi_selected = st.select_slider(
                "Select χ position (°)",
                options=chi_vals,
                value=st.session_state["chi_selected"],
                key="chi_selected",
                label_visibility="collapsed",
            )

        with plot_col:
            st.plotly_chart(
                _make_chi_slice(df, df_sub, chi_selected, bg_method),
                width='stretch',
            )

        slice_mask = _chi_mask(df, chi_selected)
        slice_sub  = df.loc[slice_mask].sort_values("two_theta")
        tt_s = slice_sub["two_theta"].values
        y_s  = slice_sub["intensity"].values
        chi_str_dl = f"{chi_selected:.2f}".replace(".", "p")

        dl_col1, dl_col2 = st.columns(2)
        raw_xy = "# 2Theta  Intensity\n" + "\n".join(
            f"{t:.6f}  {i:.4f}" for t, i in zip(tt_s, y_s))
        dl_col1.download_button(
            "💾 Download raw slice (.xy)",
            data=raw_xy,
            file_name=f"{base_name}_chi{chi_str_dl}.xy",
            mime="text/plain",
        )

        if df_sub is not None and bg_method != "None":
            slice_mask2 = _chi_mask(df_sub, chi_selected)
            slice_sub2  = df_sub.loc[slice_mask2].sort_values("two_theta")
            y_sub_s     = slice_sub2["intensity"].values
            sub_xy = "# 2Theta  Intensity_bg_subtracted\n" + "\n".join(
                f"{t:.6f}  {i:.4f}" for t, i in zip(tt_s, y_sub_s))
            dl_col2.download_button(
                "💾 Download bg-subtracted slice (.xy)",
                data=sub_xy,
                file_name=f"{base_name}_chi{chi_str_dl}_bg_sub.xy",
                mime="text/plain",
            )

    with tab_dl:
        st.markdown("#### Batch export — one **.xy** file per χ position")

        dl_sub_flag = dl_subtract and (df_sub is not None) and (bg_method != "None")

        if dl_sub_flag:
            st.success(
                f"Background subtraction (**{bg_method}**) will be applied to all "
                f"{n_chi} files.")
        else:
            if dl_subtract and bg_method == "None":
                st.warning("No background method selected — exporting raw data.")
            else:
                st.info(f"Exporting raw intensities for all {n_chi} χ positions.")

        if st.button("📦 Generate ZIP archive", type="primary"):
            with st.spinner(f"Packaging {n_chi} .xy files…"):
                zip_bytes = _make_zip(
                    df, base_name,
                    subtract=dl_sub_flag,
                    df_sub=df_sub if dl_sub_flag else None,
                )
            st.download_button(
                label=f"⬇️ Download ZIP  ({n_chi} files)",
                data=zip_bytes,
                file_name=f"{base_name}_chi_scans.zip",
                mime="application/zip",
                type="primary",
            )

        example_chi = f"{chi_vals[0]:.2f}".replace(".", "p")
        st.markdown("**File naming convention:**")
        st.code(f"{base_name}_chi<value>.xy   e.g.  {base_name}_chi{example_chi}.xy")
        st.caption("χ decimal point is replaced by 'p' for safe filenames (e.g. chi2p00).")

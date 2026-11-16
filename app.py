import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import xml.etree.ElementTree as ET
from io import StringIO, BytesIO
from datetime import datetime, timedelta
import re
import zipfile
import struct

from more_funct.xrd_conversion import run_axis_converter, convert_xaxis_data, get_axis_label, timestamp_suffix
from more_funct.chi_scan_section import run_chi_scan_section
from more_funct.chi_merge_section import run_chi_merge_section
from more_funct.plotting_section import run_plotting_section
from more_funct.data_reduction_section import run_data_reduction_section
from more_funct.image_digitizer_section import run_image_digitizer_section
from more_funct.download_log import log_download
from more_funct.ui_style import apply_button_style
from more_funct.xrd_parsers import (extract_key_ras_metadata, metadata_dataframe,
                         parse_xrdml, parse_ras, parse_rasx, parse_raw,
                         parse_xy, check_regular_grid, resample_to_regular_grid,
                         detect_x_axis)


# Characteristic wavelengths (Kα1, Kα2, Kβ) in Å of the usual laboratory
# anodes, so that a .xy file — which carries no instrument information at all —
# can be converted with a realistic header without typing four numbers.
ANODE_PRESETS = {
    "Cu": (1.540598, 1.544426, 1.392250),
    "Co": (1.788996, 1.792835, 1.620790),
    "Mo": (0.709319, 0.713609, 0.632305),
    "Cr": (2.289760, 2.293651, 2.084870),
    "Fe": (1.936042, 1.939980, 1.756610),
    "Ag": (0.559422, 0.563813, 0.497069),
}

# Which rows of the metadata table hold that information, per output format.
ANODE_ROWS = {
    "RAS": {"anode": "HW_XG_TARGET_NAME",
            "ka1": "HW_XG_WAVE_LENGTH_ALPHA1",
            "ka2": "HW_XG_WAVE_LENGTH_ALPHA2",
            "kb": "HW_XG_WAVE_LENGTH_BETA"},
    "RAW": {"anode": "X-ray Target",
            "ka1": "K-Alpha1 (Å)",
            "ka2": "K-Alpha2 (Å)",
            "kb": "K-Beta (Å)"},
    "XRDML": {"anode": "Anode Material",
              "ka1": "K-Alpha1 Wavelength (Å)",
              "ka2": "K-Alpha2 Wavelength (Å)",
              "kb": "K-Beta Wavelength (Å)"},
}


def apply_anode_preset(meta_df, output_format, anode):
    """Metadata table with the anode / wavelength rows set to the preset.

    Only the rows that the chosen output format actually knows are touched;
    everything the user has edited otherwise is kept. A row that is missing
    (the table is editable, so it can be deleted) is appended again.
    """
    ka1, ka2, kb = ANODE_PRESETS[anode]
    rows = ANODE_ROWS.get(output_format, {})
    values = {
        rows.get("anode"): anode,
        rows.get("ka1"): f"{ka1:.5f}",
        rows.get("ka2"): f"{ka2:.5f}",
        rows.get("kb"): f"{kb:.5f}",
    }
    values.pop(None, None)

    return set_metadata_rows(meta_df, values)


def set_metadata_rows(meta_df, values):
    """Metadata table with the given {parameter: value} rows set (appended if missing)."""
    df = meta_df.copy()
    for parameter, value in values.items():
        mask = df["Parameter"] == parameter
        if mask.any():
            df.loc[mask, "Value"] = value
        else:
            df.loc[len(df)] = [parameter, value]
    return df.reset_index(drop=True)


def _number(value):
    """Leading number of a metadata value ('45 kV' -> '45'), or None."""
    match = re.match(r'\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', str(value))
    return match.group(1) if match else None


def _iso_time(value):
    """ISO timestamp from the formats the parsers return, or None."""
    text = str(value).strip()
    for parse in (datetime.fromisoformat,
                  lambda t: datetime.strptime(t, '%m/%d/%Y %H:%M:%S')):
        try:
            return parse(text).isoformat()
        except ValueError:
            pass
    return None


def instrument_settings(key_metadata, full_metadata, step=None):
    """Instrument settings of an uploaded file under format-neutral names.

    Each parser names its fields differently (.xrdml 'X-ray Tube Tension',
    .ras 'MEAS_COND_XG_VOLTAGE', ...); this collects the ones that every
    output format can store. Missing or 'N/A' values are left out, so the
    output table keeps its defaults for them. ``step`` (the 2θ step of the
    data) converts between a scan speed and a counting time per step.
    """
    md = {**(full_metadata or {}), **(key_metadata or {})}

    def first(*keys):
        for key in keys:
            value = str(md.get(key, '')).strip()
            if value and value.upper() != 'N/A':
                return value
        return None

    settings = {
        'anode': first('Anode Material', 'X-ray Target', 'HW_XG_TARGET_NAME'),
        'ka1': _number(first('K-Alpha1 Wavelength (Å)', 'K-Alpha1 (Å)', 'HW_XG_WAVE_LENGTH_ALPHA1') or ''),
        'ka2': _number(first('K-Alpha2 Wavelength (Å)', 'K-Alpha2 (Å)', 'HW_XG_WAVE_LENGTH_ALPHA2') or ''),
        'kb': _number(first('K-Beta Wavelength (Å)', 'K-Beta (Å)', 'HW_XG_WAVE_LENGTH_BETA') or ''),
        'kv': _number(first('X-ray Tube Tension', 'Voltage (kV)', 'MEAS_COND_XG_VOLTAGE') or ''),
        'ma': _number(first('X-ray Tube Current', 'Current (mA)', 'MEAS_COND_XG_CURRENT') or ''),
        'operator': first('Author', 'FILE_OPERATOR'),
        'start': _iso_time(first('Start Time', 'MEAS_SCAN_START_TIME') or ''),
        'end': _iso_time(first('End Time', 'MEAS_SCAN_END_TIME') or ''),
        'time_per_step': _number(first('Time per Step (s)', 'Common Counting Time (s)') or ''),
        'ds': first('Divergence Slit (DS)'),
        'ss': first('Scattering Slit (SS)'),
        'rs': first('Receiving Slit (RS)'),
    }
    speed = _number(first('MEAS_SCAN_SPEED') or '')
    if speed and first('MEAS_SCAN_SPEED_UNIT') not in (None, 'deg/min'):
        speed = None  # only deg/min can be converted to a time per step
    if step and step > 0:
        if speed and not settings['time_per_step'] and float(speed) > 0:
            settings['time_per_step'] = f"{step / float(speed) * 60:.4g}"
        elif settings['time_per_step'] and not speed and float(settings['time_per_step']) > 0:
            speed = f"{step / float(settings['time_per_step']) * 60:.4f}"
    settings['speed'] = speed
    return {k: v for k, v in settings.items() if v}


# Which table row each setting from instrument_settings() goes to, per output
# format. Settings a format cannot store are simply not listed.
SETTINGS_ROWS = {
    "XRDML": {**ANODE_ROWS["XRDML"],
              "kv": "X-ray Tube Tension (kV)", "ma": "X-ray Tube Current (mA)",
              "operator": "Author", "start": "Start Time", "end": "End Time",
              "time_per_step": "Common Counting Time (s)"},
    "RAS": {**ANODE_ROWS["RAS"],
            "kv": "MEAS_COND_XG_VOLTAGE", "ma": "MEAS_COND_XG_CURRENT",
            "operator": "FILE_OPERATOR", "start": "MEAS_SCAN_START_TIME",
            "end": "MEAS_SCAN_END_TIME", "speed": "MEAS_SCAN_SPEED",
            "ds": "Divergence Slit (DS)", "ss": "Scattering Slit (SS)",
            "rs": "Receiving Slit (RS)"},
    "RAW": {**ANODE_ROWS["RAW"], "time_per_step": "Time per Step (s)"},
}


def apply_instrument_settings(meta_df, output_format, settings):
    """Metadata table with the rows filled from an uploaded file's settings."""
    anode = str(settings.get("anode", "")).capitalize()
    if anode in ANODE_PRESETS:
        # A file that names only Kα1 must not keep another anode's Kα2 / Kβ.
        meta_df = apply_anode_preset(meta_df, output_format, anode)
    rows = SETTINGS_ROWS.get(output_format, {})
    values = {rows[k]: v for k, v in settings.items() if k in rows}
    if output_format == "RAS" and "speed" in settings:
        values["MEAS_SCAN_SPEED_UNIT"] = "deg/min"
    return set_metadata_rows(meta_df, values)


def run_data_converter():

    def convert_to_xy(data_df, include_header=False):
        try:
            output = StringIO()
            output_df = data_df[['2Theta', 'Intensity']]

            header = ['2Theta', 'Intensity'] if include_header else False
            output_df.to_csv(output, sep='\t', header=header, index=False, float_format='%.6f')
            return output.getvalue()
        except KeyError:
            st.error("The DataFrame is missing the required '2Theta' or 'Intensity' columns.")
            return ""

    def generate_xrdml(metadata_df, data_df, filename=""):
        meta_dict = pd.Series(metadata_df.Value.values, index=metadata_df.Parameter).to_dict()
        start_2theta = data_df['2Theta'].min()
        end_2theta = data_df['2Theta'].max()
        intensities_str = ' '.join(map(lambda x: f"{x:.3f}", data_df['Intensity'].values))

        sample_name = filename.split('.')[0] if filename else meta_dict.get('Sample Name', 'Converted Sample')

        return f"""<?xml version="1.0" encoding="utf-8" standalone="no"?>
<xrdMeasurements xmlns="http://www.xrdml.com/XRDMeasurement/1.3" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.xrdml.com/XRDMeasurement/1.3 http://www.xrdml.com/XRDMeasurement/1.3/XRDMeasurement.xsd" status="{meta_dict.get('Status', 'Completed')}">
  <sample type="To be analyzed">
    <id>{meta_dict.get('Sample ID', 'N/A')}</id>
    <name>{sample_name}</name>
  </sample>
  <xrdMeasurement measurementType="{meta_dict.get('Measurement Type', 'Scan')}" status="{meta_dict.get('Status', 'Completed')}">
    <usedWavelength intended="K-Alpha">
      <kAlpha1 unit="Angstrom">{meta_dict.get('K-Alpha1 Wavelength (Å)', '1.54056')}</kAlpha1>
      <kAlpha2 unit="Angstrom">{meta_dict.get('K-Alpha2 Wavelength (Å)', '1.54439')}</kAlpha2>
      <kBeta unit="Angstrom">{meta_dict.get('K-Beta Wavelength (Å)', '1.39225')}</kBeta>
      <ratioKAlpha2KAlpha1>{meta_dict.get('Ratio K-Alpha2/K-Alpha1', '0.5')}</ratioKAlpha2KAlpha1>
    </usedWavelength>
    <incidentBeamPath>
      <xRayTube>
        <tension unit="kV">{meta_dict.get('X-ray Tube Tension (kV)', '45')}</tension>
        <current unit="mA">{meta_dict.get('X-ray Tube Current (mA)', '40')}</current>
        <anodeMaterial>{meta_dict.get('Anode Material', 'Cu')}</anodeMaterial>
      </xRayTube>
    </incidentBeamPath>
    <scan appendNumber="0" mode="{meta_dict.get('Scan Mode', 'Continuous')}" scanAxis="{meta_dict.get('Scan Axis', 'Gonio')}" status="Completed">
      <header>
        <startTimeStamp>{meta_dict.get('Start Time', datetime.now().isoformat())}</startTimeStamp>
        <endTimeStamp>{meta_dict.get('End Time', (datetime.now() + timedelta(minutes=30)).isoformat())}</endTimeStamp>
        <author>
          <name>{meta_dict.get('Author', 'XRDlicious User')}</name>
        </author>
        <source>
          <applicationSoftware version="1.0">{meta_dict.get('Application Software', 'XRDlicious')}</applicationSoftware>
        </source>
      </header>
      <dataPoints>
        <positions axis="2Theta" unit="deg">
          <startPosition>{start_2theta:.6f}</startPosition>
          <endPosition>{end_2theta:.6f}</endPosition>
        </positions>
        <commonCountingTime unit="seconds">{meta_dict.get('Common Counting Time (s)', '1.0')}</commonCountingTime>
        <intensities unit="counts">{intensities_str}</intensities>
      </dataPoints>
    </scan>
  </xrdMeasurement>
</xrdMeasurements>
"""

    def generate_ras(metadata_df, data_df):
        meta_dict = pd.Series(metadata_df.Value.values, index=metadata_df.Parameter).to_dict()
        start_angle = f"{data_df['2Theta'].min():.4f}"
        stop_angle = f"{data_df['2Theta'].max():.4f}"
        data_count = str(len(data_df))
        step_size = f"{abs(data_df['2Theta'].iloc[1] - data_df['2Theta'].iloc[0]):.6f}" if len(
            data_df) > 1 else "0.0100"

        try:
            start_iso = meta_dict.get('MEAS_SCAN_START_TIME', datetime.now().isoformat())
            end_iso = meta_dict.get('MEAS_SCAN_END_TIME', datetime.now().isoformat())
            start_dt = datetime.fromisoformat(start_iso)
            end_dt = datetime.fromisoformat(end_iso)
            formatted_start_time = start_dt.strftime('%m/%d/%Y %H:%M:%S')
            formatted_end_time = end_dt.strftime('%m/%d/%Y %H:%M:%S')
        except ValueError:
            now = datetime.now()
            formatted_start_time = now.strftime('%m/%d/%Y %H:%M:%S')
            formatted_end_time = (now + timedelta(minutes=20)).strftime('%m/%d/%Y %H:%M:%S')

        header_template = f"""*RAS_DATA_START
*RAS_HEADER_START
*DISP_LINE_COLOR "4294901760"
*FILE_COMMENT ""
*FILE_MD5 ""
*FILE_MEMO ""
*FILE_OPERATOR "{meta_dict.get('FILE_OPERATOR', 'Administrator')}"
*FILE_PACKAGE_NAME "Package_BB"
*FILE_PART_ID "GeneralMeasurement(BB)"
*FILE_SAMPLE ""
*FILE_SYSTEM_NAME "SmartLabXE"
*FILE_TYPE "RAS_RAW"
*FILE_USERGROUP "Administrators"
*FILE_VERSION "1"
*HW_ATTACHMENT_ID "ATT0025"
*HW_ATTACHMENT_NAME "Standard"
*HW_ATTACHMENT_NAME_INTERNAL "Standard"
*HW_COUNTER_ID-0 "CUT0051"
*HW_COUNTER_ID-2 "CMC0020"
*HW_COUNTER_MONOCHRO_ID "CMC0020"
*HW_COUNTER_MONOCHRO_NAME "None"
*HW_COUNTER_MONOCHRO_NAME_INTERNAL "None"
*HW_COUNTER_NAME_INTERNAL "HyPix3000(H)"
*HW_COUNTER_NAME-0 "HyPix3000(H)"
*HW_COUNTER_NAME-1 "None"
*HW_COUNTER_NAME-2 "None"
*HW_COUNTER_PIXEL_SIZE "0.1"
*HW_COUNTER_SELECT_NAME "HyPix3000(H)"
*HW_EXTERNAL_CONTROLLER_NAME "None"
*HW_EXTERNAL_CONTROLLER_NAME_INTERNAL "None"
*HW_GONIOMETER_ID "GON0022"
*HW_GONIOMETER_NAME "StandardInplane"
*HW_GONIOMETER_NAME_INTERNAL "StandardInplaneEnc"
*HW_GONIOMETER_RADIUS-0 "90.0"
*HW_GONIOMETER_RADIUS-1 "114.0"
*HW_GONIOMETER_RADIUS-2 "173.5"
*HW_GONIOMETER_RADIUS-3 "300.0"
*HW_GONIOMETER_RADIUS-4 "187.0"
*HW_GONIOMETER_RADIUS-5 "300.0"
*HW_GONIOMETER_RADIUS-6 "113.0"
*HW_GONIOMETER_RADIUS-7 "331.0"
*HW_I_CBO_ID "CBO0021"
*HW_I_CBO_NAME "CBO"
*HW_I_CBO_NAME_INTERNAL "CBO"
*HW_I_MONOCHRO_ID "ISO0021"
*HW_I_MONOCHRO_NAME "IPS_adaptor"
*HW_I_MONOCHRO_NAME_INTERNAL "IPS_adaptor"
*HW_I_OPT_ID-1 "CBO0021"
*HW_I_PRIMARY_NAME_INTERNAL "Standard"
*HW_I_SLIT_NAME_INTERNAL "AutoIntegrated"
*HW_R_ATTENUATER_AUTOMODE "0"
*HW_R_ATTENUATER_ID "ATN0020"
*HW_R_ATTENUATER_NAME "No_unit"
*HW_R_ATTENUATOR_NAME_INTERNAL "No_unit"
*HW_R_OPT_ID-0 "RSS0022"
*HW_R_OPT_ID-1 "RCR0022"
*HW_R_OPT_ID-2 "RSO0021"
*HW_R_OPT_ID-3 "RRS0022"
*HW_R_OPT_ID-4 "ATN0020"
*HW_R_ROD_ID "RCR0022"
*HW_R_ROD_NAME "ROD_adaptor"
*HW_R_ROD_NAME_INTERNAL "ROD_adaptor"
*HW_R_RPS_ID "RSO0021"
*HW_R_RPS_NAME "RPS_adaptor"
*HW_R_RPS_NAME_INTERNAL "RPS_adaptor"
*HW_R_RS_ID "RRS0022"
*HW_R_RS_NAME "Virtual"
*HW_R_RS_NAME_INTERNAL "Virtual"
*HW_R_SS_ID "RSS0022"
*HW_R_SS_NAME "Auto_Zr"
*HW_R_SS_NAME_INTERNAL "Auto_Zr"
*HW_ROBOT_NAME "No_unit"
*HW_ROBOT_NAME_INTERNAL "No_unit"
*HW_SAMPLE_CAMERA_NAME "SampleCamera_Inp"
*HW_SAMPLE_CAMERA_NAME_INTERNAL "SampleCamera_Inp"
*HW_SAMPLE_HOLDER_ID "SMP0021"
*HW_SAMPLE_HOLDER_NAME "Z_ChiPhi"
*HW_SAMPLE_HOLDER_NAME_INTERNAL "Z_ChiPhi"
*HW_SAMPLE_NAME "PowderSample"
*HW_SAMPLE_NAME_INTERNAL "PowderSample"
*HW_SAMPLE_PLATE_NAME "ForWafer"
*HW_SAMPLE_PLATE_NAME_INTERNAL "ForWafer"
*HW_SAMPLE_SPACER_NAME "3-6mm"
*HW_SAMPLE_SPACER_NAME_INTERNAL "3-6mm"
*HW_USERINFO_CATALOG_NO ""
*HW_USERINFO_INSTRUMENT_ID ""
*HW_USERINFO_INSTRUMENT_NAME ""
*HW_USERINFO_INSTRUMENT_NO ""
*HW_USERINFO_MODEL ""
*HW_USERINFO_ORDER_NO ""
*HW_USERINFO_SERIAL_NO ""
*HW_USERINFO_VERIFY_INSTRUMENT_ID "00-07-fe-01-26-9c"
*HW_VER_RCD_COUNTER_UNIT "N/A"
*HW_VER_RCD_CPU "3.3.7"
*HW_VER_RCD_FPGA "1.2.3"
*HW_VER_RCD_GONIO_UNIT "6.4.0"
*HW_VER_RCD_INCIDENT_UNIT "6.4.0"
*HW_VER_RCD_RECEIVING_UNIT "8.0.1"
*HW_VER_RCD_TYPE "RINC"
*HW_VER_XGC_CPU "3.3.7"
*HW_VER_XGC_CW "3.0.8"
*HW_VER_XGC_HV "3.0.8"
*HW_VER_XGC_RS1 "3.0.8"
*HW_VER_XGC_RS2 "N/A"
*HW_VER_XGC_RT "N/A"
*HW_VER_XGC_TYPE "RINC"
*HW_XG_CURRENT_UNIT "mA"
*HW_XG_FOCUS "0.4mm x 8mm"
*HW_XG_FOCUS_TYPE "Fine"
*HW_XG_TARGET_ATOMIC_NUMBER "29"
*HW_XG_TARGET_NAME "{meta_dict.get('HW_XG_TARGET_NAME', 'Cu')}"
*HW_XG_TYPE "Hermetic"
*HW_XG_VOLTAGE_UNIT "kV"
*HW_XG_WAVE_LENGTH_ALPHA1 "{meta_dict.get('HW_XG_WAVE_LENGTH_ALPHA1', '1.540593')}"
*HW_XG_WAVE_LENGTH_ALPHA2 "{meta_dict.get('HW_XG_WAVE_LENGTH_ALPHA2', '1.544414')}"
*HW_XG_WAVE_LENGTH_BETA "{meta_dict.get('HW_XG_WAVE_LENGTH_BETA', '1.392246')}"
*HW_XG_WAVE_LENGTH_UNIT "Angstrom"
*MEAS_COND_AXIS_NAME_INTERNAL-0 "ThetaS"
*MEAS_COND_AXIS_NAME_INTERNAL-1 "ThetaD"
*MEAS_COND_AXIS_NAME_INTERNAL-10 "PrimaryGeometry"
*MEAS_COND_AXIS_NAME_INTERNAL-100 "IncidentMonochromator"
*MEAS_COND_AXIS_NAME_INTERNAL-101 "ReceivingOptics"
*MEAS_COND_AXIS_NAME_INTERNAL-102 "CounterMonochromatorKind"
*MEAS_COND_AXIS_NAME_INTERNAL-11 "PrimaryMirrorType"
*MEAS_COND_AXIS_NAME_INTERNAL-12 "CBOType"
*MEAS_COND_AXIS_NAME_INTERNAL-13 "CBO-M"
*MEAS_COND_AXIS_NAME_INTERNAL-14 "CBO"
*MEAS_COND_AXIS_NAME_INTERNAL-15 "IncidentSollerSlit"
*MEAS_COND_AXIS_NAME_INTERNAL-16 "IncidentSlitBox"
*MEAS_COND_AXIS_NAME_INTERNAL-17 "IncidentSlitBox"
*MEAS_COND_AXIS_NAME_INTERNAL-18 "IncidentSlitBox"
*MEAS_COND_AXIS_NAME_INTERNAL-19 "Zs"
*MEAS_COND_AXIS_NAME_INTERNAL-2 "TwoTheta"
*MEAS_COND_AXIS_NAME_INTERNAL-20 "IncidentAxdSlit"
*MEAS_COND_AXIS_NAME_INTERNAL-21 "IncidentAxdSlit"
*MEAS_COND_AXIS_NAME_INTERNAL-22 "IncidentAxdSlit"
*MEAS_COND_AXIS_NAME_INTERNAL-23 "InFilter"
*MEAS_COND_AXIS_NAME_INTERNAL-24 "Chi"
*MEAS_COND_AXIS_NAME_INTERNAL-25 "Phi"
*MEAS_COND_AXIS_NAME_INTERNAL-26 "Z"
*MEAS_COND_AXIS_NAME_INTERNAL-27 "Alpha"
*MEAS_COND_AXIS_NAME_INTERNAL-28 "Beta"
*MEAS_COND_AXIS_NAME_INTERNAL-29 "TwoThetaChiPhi"
*MEAS_COND_AXIS_NAME_INTERNAL-3 "Omega"
*MEAS_COND_AXIS_NAME_INTERNAL-30 "AlphaI"
*MEAS_COND_AXIS_NAME_INTERNAL-31 "BetaI"
*MEAS_COND_AXIS_NAME_INTERNAL-32 "TwoThetaB"
*MEAS_COND_AXIS_NAME_INTERNAL-33 "ReceivingSlitBox1"
*MEAS_COND_AXIS_NAME_INTERNAL-34 "ReceivingSlitBox1"
*MEAS_COND_AXIS_NAME_INTERNAL-35 "ReceivingSlitBox1"
*MEAS_COND_AXIS_NAME_INTERNAL-36 "Zr"
*MEAS_COND_AXIS_NAME_INTERNAL-37 "Filter"
*MEAS_COND_AXIS_NAME_INTERNAL-38 "PSA"
*MEAS_COND_AXIS_NAME_INTERNAL-39 "ReceivingSollerSlit"
*MEAS_COND_AXIS_NAME_INTERNAL-4 "TwoThetaTheta"
*MEAS_COND_AXIS_NAME_INTERNAL-40 "ReceivingSlitBox2"
*MEAS_COND_AXIS_NAME_INTERNAL-41 "ReceivingSlitBox2"
*MEAS_COND_AXIS_NAME_INTERNAL-42 "ReceivingSlitBox2"
*MEAS_COND_AXIS_NAME_INTERNAL-43 "ModuleSensorType"
*MEAS_COND_AXIS_NAME_INTERNAL-44 "GainMode"
*MEAS_COND_AXIS_NAME_INTERNAL-45 "PHA"
*MEAS_COND_AXIS_NAME_INTERNAL-46 "DetectorBeamStop"
*MEAS_COND_AXIS_NAME_INTERNAL-47 "DetectorFilter"
*MEAS_COND_AXIS_NAME_INTERNAL-48 "DedicatedHolder"
*MEAS_COND_AXIS_NAME_INTERNAL-49 "Target_TargetTime"
*MEAS_COND_AXIS_NAME_INTERNAL-5 "TwoThetaOmega"
*MEAS_COND_AXIS_NAME_INTERNAL-50 "Target_XrayOnTime"
*MEAS_COND_AXIS_NAME_INTERNAL-51 "Target_TMPTime"
*MEAS_COND_AXIS_NAME_INTERNAL-52 "Target_RPTime"
*MEAS_COND_AXIS_NAME_INTERNAL-53 "Target_FilamentTime"
*MEAS_COND_AXIS_NAME_INTERNAL-54 "Target_IG"
*MEAS_COND_AXIS_NAME_INTERNAL-55 "Target_GP"
*MEAS_COND_AXIS_NAME_INTERNAL-56 "Target_FC"
*MEAS_COND_AXIS_NAME_INTERNAL-57 "HVPS_Bias"
*MEAS_COND_AXIS_NAME_INTERNAL-58 "HVPS_HVPSTime"
*MEAS_COND_AXIS_NAME_INTERNAL-59 "HVPS_Type1Time"
*MEAS_COND_AXIS_NAME_INTERNAL-6 "OmegaTwoTheta"
*MEAS_COND_AXIS_NAME_INTERNAL-60 "HVPS_Type2Time"
*MEAS_COND_AXIS_NAME_INTERNAL-61 "HVPS_Type3Time"
*MEAS_COND_AXIS_NAME_INTERNAL-62 "CW_IERTime"
*MEAS_COND_AXIS_NAME_INTERNAL-63 "CW_ECTime"
*MEAS_COND_AXIS_NAME_INTERNAL-64 "CW_Flow1"
*MEAS_COND_AXIS_NAME_INTERNAL-65 "CW_Temperature1"
*MEAS_COND_AXIS_NAME_INTERNAL-66 "CW_Pressure1"
*MEAS_COND_AXIS_NAME_INTERNAL-67 "CW_PressureIn"
*MEAS_COND_AXIS_NAME_INTERNAL-68 "CW_PressureOut"
*MEAS_COND_AXIS_NAME_INTERNAL-69 "CW_Flow2"
*MEAS_COND_AXIS_NAME_INTERNAL-7 "TwoThetaChi"
*MEAS_COND_AXIS_NAME_INTERNAL-70 "CW_Temperature2"
*MEAS_COND_AXIS_NAME_INTERNAL-71 "CW_Pressure2"
*MEAS_COND_AXIS_NAME_INTERNAL-72 "RE_EnclosureTemp"
*MEAS_COND_AXIS_NAME_INTERNAL-73 "RE_EnclosureHummidity"
*MEAS_COND_AXIS_NAME_INTERNAL-74 "RE_CabinetTemp"
*MEAS_COND_AXIS_NAME_INTERNAL-75 "RE_RPTemp"
*MEAS_COND_AXIS_NAME_INTERNAL-76 "RE_ShutterAXrayOnTime"
*MEAS_COND_AXIS_NAME_INTERNAL-77 "RE_ShutterATimes"
*MEAS_COND_AXIS_NAME_INTERNAL-78 "RE_ShutterAOpenCloseTime"
*MEAS_COND_AXIS_NAME_INTERNAL-79 "RE_ShutterACloseOpenTime"
*MEAS_COND_AXIS_NAME_INTERNAL-8 "GonioDirectBeamStop"
*MEAS_COND_AXIS_NAME_INTERNAL-80 "RE_ShutterBXrayOnTime"
*MEAS_COND_AXIS_NAME_INTERNAL-81 "RE_ShutterBTimes"
*MEAS_COND_AXIS_NAME_INTERNAL-82 "RE_ShutterBOpenCloseTime"
*MEAS_COND_AXIS_NAME_INTERNAL-83 "RE_ShutterBCloseOpenTime"
*MEAS_COND_AXIS_NAME_INTERNAL-84 "RE_XrayWarningRamp"
*MEAS_COND_AXIS_NAME_INTERNAL-85 "RE_ShutterAInstallation"
*MEAS_COND_AXIS_NAME_INTERNAL-86 "RE_ShutterARamp"
*MEAS_COND_AXIS_NAME_INTERNAL-87 "RE_ShutterBInstallation"
*MEAS_COND_AXIS_NAME_INTERNAL-88 "RE_ShutterBRamp"
*MEAS_COND_AXIS_NAME_INTERNAL-89 "RE_ExtShutterCLS"
*MEAS_COND_AXIS_NAME_INTERNAL-9 "Ts"
*MEAS_COND_AXIS_NAME_INTERNAL-90 "RE_ExtXrayOFF"
*MEAS_COND_AXIS_NAME_INTERNAL-91 "RE_ExtSafetyCircuit"
*MEAS_COND_AXIS_NAME_INTERNAL-92 "Version_CPU"
*MEAS_COND_AXIS_NAME_INTERNAL-93 "Version_RS1"
*MEAS_COND_AXIS_NAME_INTERNAL-94 "Version_RS2"
*MEAS_COND_AXIS_NAME_INTERNAL-95 "Version_HV"
*MEAS_COND_AXIS_NAME_INTERNAL-96 "Version_CW"
*MEAS_COND_AXIS_NAME_INTERNAL-97 "Version_RT"
*MEAS_COND_AXIS_NAME_INTERNAL-98 "Lens"
*MEAS_COND_AXIS_NAME_INTERNAL-99 "IncidentPrimary"
*MEAS_COND_AXIS_NAME-0 "ThetaS"
*MEAS_COND_AXIS_NAME-1 "ThetaD"
*MEAS_COND_AXIS_NAME-10 "PrimaryGeometry"
*MEAS_COND_AXIS_NAME-100 "IncidentMonochromator"
*MEAS_COND_AXIS_NAME-101 "ReceivingOptics"
*MEAS_COND_AXIS_NAME-102 "CounterMonochromatorKind"
*MEAS_COND_AXIS_NAME-11 "PrimaryMirrorType"
*MEAS_COND_AXIS_NAME-12 "CBOType"
*MEAS_COND_AXIS_NAME-13 "CBO-M"
*MEAS_COND_AXIS_NAME-14 "CBO"
*MEAS_COND_AXIS_NAME-15 "IncidentSollerSlit"
*MEAS_COND_AXIS_NAME-16 "IncidentSlitBox"
*MEAS_COND_AXIS_NAME-17 "IncidentSlitBox-_Axis"
*MEAS_COND_AXIS_NAME-18 "SlitDS"
*MEAS_COND_AXIS_NAME-19 "Zs"
*MEAS_COND_AXIS_NAME-2 "TwoTheta"
*MEAS_COND_AXIS_NAME-20 "IncidentAxdSlit"
*MEAS_COND_AXIS_NAME-21 "LLS"
*MEAS_COND_AXIS_NAME-22 "DHLSlit"
*MEAS_COND_AXIS_NAME-23 "InFilter"
*MEAS_COND_AXIS_NAME-24 "Chi"
*MEAS_COND_AXIS_NAME-25 "Phi"
*MEAS_COND_AXIS_NAME-26 "Z"
*MEAS_COND_AXIS_NAME-27 "Alpha"
*MEAS_COND_AXIS_NAME-28 "Beta"
*MEAS_COND_AXIS_NAME-29 "TwoThetaChiPhi"
*MEAS_COND_AXIS_NAME-3 "Omega"
*MEAS_COND_AXIS_NAME-30 "AlphaI"
*MEAS_COND_AXIS_NAME-31 "BetaI"
*MEAS_COND_AXIS_NAME-32 "TwoThetaB"
*MEAS_COND_AXIS_NAME-33 "ReceivingSlitBox1"
*MEAS_COND_AXIS_NAME-34 "ReceivingSlitBox1-_Axis"
*MEAS_COND_AXIS_NAME-35 "SlitSS"
*MEAS_COND_AXIS_NAME-36 "Zr"
*MEAS_COND_AXIS_NAME-37 "Filter"
*MEAS_COND_AXIS_NAME-38 "PSA"
*MEAS_COND_AXIS_NAME-39 "ReceivingSollerSlit"
*MEAS_COND_AXIS_NAME-4 "TwoThetaTheta"
*MEAS_COND_AXIS_NAME-40 "ReceivingSlitBox2"
*MEAS_COND_AXIS_NAME-41 "ReceivingSlitBox2-_Axis"
*MEAS_COND_AXIS_NAME-42 "SlitRS"
*MEAS_COND_AXIS_NAME-43 "ModuleSensorType"
*MEAS_COND_AXIS_NAME-44 "GainMode"
*MEAS_COND_AXIS_NAME-45 "PHA"
*MEAS_COND_AXIS_NAME-46 "DetectorBeamStop"
*MEAS_COND_AXIS_NAME-47 "DetectorFilter"
*MEAS_COND_AXIS_NAME-48 "DedicatedHolder"
*MEAS_COND_AXIS_NAME-49 "Target_TargetTime"
*MEAS_COND_AXIS_NAME-5 "TwoThetaOmega"
*MEAS_COND_AXIS_NAME-50 "Target_XrayOnTime"
*MEAS_COND_AXIS_NAME-51 "Target_TMPTime"
*MEAS_COND_AXIS_NAME-52 "Target_RPTime"
*MEAS_COND_AXIS_NAME-53 "Target_FilamentTime"
*MEAS_COND_AXIS_NAME-54 "Target_IG"
*MEAS_COND_AXIS_NAME-55 "Target_GP"
*MEAS_COND_AXIS_NAME-56 "Target_FC"
*MEAS_COND_AXIS_NAME-57 "HVPS_Bias"
*MEAS_COND_AXIS_NAME-58 "HVPS_HVPSTime"
*MEAS_COND_AXIS_NAME-59 "HVPS_Type1Time"
*MEAS_COND_AXIS_NAME-6 "OmegaTwoTheta"
*MEAS_COND_AXIS_NAME-60 "HVPS_Type2Time"
*MEAS_COND_AXIS_NAME-61 "HVPS_Type3Time"
*MEAS_COND_AXIS_NAME-62 "CW_IERTime"
*MEAS_COND_AXIS_NAME-63 "CW_ECTime"
*MEAS_COND_AXIS_NAME-64 "CW_Flow1"
*MEAS_COND_AXIS_NAME-65 "CW_Temperature1"
*MEAS_COND_AXIS_NAME-66 "CW_Pressure1"
*MEAS_COND_AXIS_NAME-67 "CW_PressureIn"
*MEAS_COND_AXIS_NAME-68 "CW_PressureOut"
*MEAS_COND_AXIS_NAME-69 "CW_Flow2"
*MEAS_COND_AXIS_NAME-7 "TwoThetaChi"
*MEAS_COND_AXIS_NAME-70 "CW_Temperature2"
*MEAS_COND_AXIS_NAME-71 "CW_Pressure2"
*MEAS_COND_AXIS_NAME-72 "RE_EnclosureTemp"
*MEAS_COND_AXIS_NAME-73 "RE_EnclosureHummidity"
*MEAS_COND_AXIS_NAME-74 "RE_CabinetTemp"
*MEAS_COND_AXIS_NAME-75 "RE_RPTemp"
*MEAS_COND_AXIS_NAME-76 "RE_ShutterAXrayOnTime"
*MEAS_COND_AXIS_NAME-77 "RE_ShutterATimes"
*MEAS_COND_AXIS_NAME-78 "RE_ShutterAOpenCloseTime"
*MEAS_COND_AXIS_NAME-79 "RE_ShutterACloseOpenTime"
*MEAS_COND_AXIS_NAME-8 "GonioDirectBeamStop"
*MEAS_COND_AXIS_NAME-80 "RE_ShutterBXrayOnTime"
*MEAS_COND_AXIS_NAME-81 "RE_ShutterBTimes"
*MEAS_COND_AXIS_NAME-82 "RE_ShutterBOpenCloseTime"
*MEAS_COND_AXIS_NAME-83 "RE_ShutterBCloseOpenTime"
*MEAS_COND_AXIS_NAME-84 "RE_XrayWarningRamp"
*MEAS_COND_AXIS_NAME-85 "RE_ShutterAInstallation"
*MEAS_COND_AXIS_NAME-86 "RE_ShutterARamp"
*MEAS_COND_AXIS_NAME-87 "RE_ShutterBInstallation"
*MEAS_COND_AXIS_NAME-88 "RE_ShutterBRamp"
*MEAS_COND_AXIS_NAME-89 "RE_ExtShutterCLS"
*MEAS_COND_AXIS_NAME-9 "Ts"
*MEAS_COND_AXIS_NAME-90 "RE_ExtXrayOFF"
*MEAS_COND_AXIS_NAME-91 "RE_ExtSafetyCircuit"
*MEAS_COND_AXIS_NAME-92 "Version_CPU"
*MEAS_COND_AXIS_NAME-93 "Version_RS1"
*MEAS_COND_AXIS_NAME-94 "Version_RS2"
*MEAS_COND_AXIS_NAME-95 "Version_HV"
*MEAS_COND_AXIS_NAME-96 "Version_CW"
*MEAS_COND_AXIS_NAME-97 "Version_RT"
*MEAS_COND_AXIS_NAME-98 "Lens"
*MEAS_COND_AXIS_NAME-99 "IncidentPrimary"
*MEAS_COND_AXIS_OFFSET-0 "1.1167"
*MEAS_COND_AXIS_OFFSET-1 "-0.0708"
*MEAS_COND_AXIS_OFFSET-10 "0"
*MEAS_COND_AXIS_OFFSET-100 "0"
*MEAS_COND_AXIS_OFFSET-101 "0"
*MEAS_COND_AXIS_OFFSET-102 "0"
*MEAS_COND_AXIS_OFFSET-11 "0"
*MEAS_COND_AXIS_OFFSET-12 "0"
*MEAS_COND_AXIS_OFFSET-13 "0"
*MEAS_COND_AXIS_OFFSET-14 "0"
*MEAS_COND_AXIS_OFFSET-15 "0"
*MEAS_COND_AXIS_OFFSET-16 "0"
*MEAS_COND_AXIS_OFFSET-17 "0"
*MEAS_COND_AXIS_OFFSET-18 "0"
*MEAS_COND_AXIS_OFFSET-19 "0"
*MEAS_COND_AXIS_OFFSET-2 "1.0459"
*MEAS_COND_AXIS_OFFSET-20 "0"
*MEAS_COND_AXIS_OFFSET-21 "0"
*MEAS_COND_AXIS_OFFSET-22 "0"
*MEAS_COND_AXIS_OFFSET-23 "0"
*MEAS_COND_AXIS_OFFSET-24 "0"
*MEAS_COND_AXIS_OFFSET-25 "0"
*MEAS_COND_AXIS_OFFSET-26 "0"
*MEAS_COND_AXIS_OFFSET-27 "0"
*MEAS_COND_AXIS_OFFSET-28 "0"
*MEAS_COND_AXIS_OFFSET-29 "0"
*MEAS_COND_AXIS_OFFSET-3 "1.1167"
*MEAS_COND_AXIS_OFFSET-30 "0"
*MEAS_COND_AXIS_OFFSET-31 "0"
*MEAS_COND_AXIS_OFFSET-32 "0"
*MEAS_COND_AXIS_OFFSET-33 "0"
*MEAS_COND_AXIS_OFFSET-34 "0"
*MEAS_COND_AXIS_OFFSET-35 "0"
*MEAS_COND_AXIS_OFFSET-36 "0"
*MEAS_COND_AXIS_OFFSET-37 "0"
*MEAS_COND_AXIS_OFFSET-38 "0"
*MEAS_COND_AXIS_OFFSET-39 "0"
*MEAS_COND_AXIS_OFFSET-4 "0"
*MEAS_COND_AXIS_OFFSET-40 "0"
*MEAS_COND_AXIS_OFFSET-41 "0"
*MEAS_COND_AXIS_OFFSET-42 "0"
*MEAS_COND_AXIS_OFFSET-43 "0"
*MEAS_COND_AXIS_OFFSET-44 "0"
*MEAS_COND_AXIS_OFFSET-45 "0"
*MEAS_COND_AXIS_OFFSET-46 "0"
*MEAS_COND_AXIS_OFFSET-47 "0"
*MEAS_COND_AXIS_OFFSET-48 "0"
*MEAS_COND_AXIS_OFFSET-49 "0"
*MEAS_COND_AXIS_OFFSET-5 "0"
*MEAS_COND_AXIS_OFFSET-50 "0"
*MEAS_COND_AXIS_OFFSET-51 "0"
*MEAS_COND_AXIS_OFFSET-52 "0"
*MEAS_COND_AXIS_OFFSET-53 "0"
*MEAS_COND_AXIS_OFFSET-54 "0"
*MEAS_COND_AXIS_OFFSET-55 "0"
*MEAS_COND_AXIS_OFFSET-56 "0"
*MEAS_COND_AXIS_OFFSET-57 "0"
*MEAS_COND_AXIS_OFFSET-58 "0"
*MEAS_COND_AXIS_OFFSET-59 "0"
*MEAS_COND_AXIS_OFFSET-6 "0"
*MEAS_COND_AXIS_OFFSET-60 "0"
*MEAS_COND_AXIS_OFFSET-61 "0"
*MEAS_COND_AXIS_OFFSET-62 "0"
*MEAS_COND_AXIS_OFFSET-63 "0"
*MEAS_COND_AXIS_OFFSET-64 "0"
*MEAS_COND_AXIS_OFFSET-65 "0"
*MEAS_COND_AXIS_OFFSET-66 "0"
*MEAS_COND_AXIS_OFFSET-67 "0"
*MEAS_COND_AXIS_OFFSET-68 "0"
*MEAS_COND_AXIS_OFFSET-69 "0"
*MEAS_COND_AXIS_OFFSET-7 "0"
*MEAS_COND_AXIS_OFFSET-70 "0"
*MEAS_COND_AXIS_OFFSET-71 "0"
*MEAS_COND_AXIS_OFFSET-72 "0"
*MEAS_COND_AXIS_OFFSET-73 "0"
*MEAS_COND_AXIS_OFFSET-74 "0"
*MEAS_COND_AXIS_OFFSET-75 "0"
*MEAS_COND_AXIS_OFFSET-76 "0"
*MEAS_COND_AXIS_OFFSET-77 "0"
*MEAS_COND_AXIS_OFFSET-78 "0"
*MEAS_COND_AXIS_OFFSET-79 "0"
*MEAS_COND_AXIS_OFFSET-8 "0"
*MEAS_COND_AXIS_OFFSET-80 "0"
*MEAS_COND_AXIS_OFFSET-81 "0"
*MEAS_COND_AXIS_OFFSET-82 "0"
*MEAS_COND_AXIS_OFFSET-83 "0"
*MEAS_COND_AXIS_OFFSET-84 "0"
*MEAS_COND_AXIS_OFFSET-85 "0"
*MEAS_COND_AXIS_OFFSET-86 "0"
*MEAS_COND_AXIS_OFFSET-87 "0"
*MEAS_COND_AXIS_OFFSET-88 "0"
*MEAS_COND_AXIS_OFFSET-89 "0"
*MEAS_COND_AXIS_OFFSET-9 "0"
*MEAS_COND_AXIS_OFFSET-90 "0"
*MEAS_COND_AXIS_OFFSET-91 "0"
*MEAS_COND_AXIS_OFFSET-92 "0"
*MEAS_COND_AXIS_OFFSET-93 "0"
*MEAS_COND_AXIS_OFFSET-94 "0"
*MEAS_COND_AXIS_OFFSET-95 "0"
*MEAS_COND_AXIS_OFFSET-96 "0"
*MEAS_COND_AXIS_OFFSET-97 "0"
*MEAS_COND_AXIS_OFFSET-98 "0"
*MEAS_COND_AXIS_OFFSET-99 "0"
*MEAS_COND_AXIS_POSITION-0 "0.0000"
*MEAS_COND_AXIS_POSITION-1 "0.0000"
*MEAS_COND_AXIS_POSITION-10 "Right"
*MEAS_COND_AXIS_POSITION-100 "IPS_adaptor"
*MEAS_COND_AXIS_POSITION-101 "PSA_open"
*MEAS_COND_AXIS_POSITION-102 "None"
*MEAS_COND_AXIS_POSITION-11 "None"
*MEAS_COND_AXIS_POSITION-12 "Type2"
*MEAS_COND_AXIS_POSITION-13 "0"
*MEAS_COND_AXIS_POSITION-14 "BB"
*MEAS_COND_AXIS_POSITION-15 "Soller_slit_2.5deg"
*MEAS_COND_AXIS_POSITION-16 "1/2deg"
*MEAS_COND_AXIS_POSITION-17 "1.514375"
*MEAS_COND_AXIS_POSITION-18 "{meta_dict.get('Divergence Slit (DS)', '1/2deg')}"
*MEAS_COND_AXIS_POSITION-19 "-0.0393750"
*MEAS_COND_AXIS_POSITION-2 "0.0000"
*MEAS_COND_AXIS_POSITION-20 "2mm"
*MEAS_COND_AXIS_POSITION-21 "2mm"
*MEAS_COND_AXIS_POSITION-22 "2mm"
*MEAS_COND_AXIS_POSITION-23 "None"
*MEAS_COND_AXIS_POSITION-24 "0.000"
*MEAS_COND_AXIS_POSITION-25 "0.000"
*MEAS_COND_AXIS_POSITION-26 "0.5094"
*MEAS_COND_AXIS_POSITION-27 "90.000"
*MEAS_COND_AXIS_POSITION-28 "0.000"
*MEAS_COND_AXIS_POSITION-29 "0.000"
*MEAS_COND_AXIS_POSITION-3 "0.0000"
*MEAS_COND_AXIS_POSITION-30 "90.00"
*MEAS_COND_AXIS_POSITION-31 "0.000"
*MEAS_COND_AXIS_POSITION-32 "0.0000"
*MEAS_COND_AXIS_POSITION-33 "Open"
*MEAS_COND_AXIS_POSITION-34 "20.000000"
*MEAS_COND_AXIS_POSITION-35 "{meta_dict.get('Scattering Slit (SS)', 'Open')}"
*MEAS_COND_AXIS_POSITION-36 "-0.2990625"
*MEAS_COND_AXIS_POSITION-37 "Cu_K-beta_1D"
*MEAS_COND_AXIS_POSITION-38 "PSA_open"
*MEAS_COND_AXIS_POSITION-39 "Soller_slit_2.5deg"
*MEAS_COND_AXIS_POSITION-4 "0.0000"
*MEAS_COND_AXIS_POSITION-40 "20.100mm"
*MEAS_COND_AXIS_POSITION-41 "20.1"
*MEAS_COND_AXIS_POSITION-42 "{meta_dict.get('Receiving Slit (RS)', '20.100mm')}"
*MEAS_COND_AXIS_POSITION-43 "AC"
*MEAS_COND_AXIS_POSITION-44 "Mo-HiC"
*MEAS_COND_AXIS_POSITION-46 "None"
*MEAS_COND_AXIS_POSITION-47 "None"
*MEAS_COND_AXIS_POSITION-48 "None"
*MEAS_COND_AXIS_POSITION-5 "0.0000"
*MEAS_COND_AXIS_POSITION-6 "0.0000"
*MEAS_COND_AXIS_POSITION-7 "0.0000"
*MEAS_COND_AXIS_POSITION-8 "Middle"
*MEAS_COND_AXIS_POSITION-9 "-4.0000000"
*MEAS_COND_AXIS_POSITION-98 "1"
*MEAS_COND_AXIS_POSITION-99 "Standard"
*MEAS_COND_AXIS_RESOLUTION-0 "0.0001"
*MEAS_COND_AXIS_RESOLUTION-1 "0.0001"
*MEAS_COND_AXIS_RESOLUTION-10 ""
*MEAS_COND_AXIS_RESOLUTION-11 ""
*MEAS_COND_AXIS_RESOLUTION-12 ""
*MEAS_COND_AXIS_RESOLUTION-13 "1"
*MEAS_COND_AXIS_RESOLUTION-14 ""
*MEAS_COND_AXIS_RESOLUTION-15 ""
*MEAS_COND_AXIS_RESOLUTION-16 "1/1600"
*MEAS_COND_AXIS_RESOLUTION-17 "1/1600"
*MEAS_COND_AXIS_RESOLUTION-18 "1/1600"
*MEAS_COND_AXIS_RESOLUTION-19 "0.0003125"
*MEAS_COND_AXIS_RESOLUTION-2 "0.0001"
*MEAS_COND_AXIS_RESOLUTION-20 ""
*MEAS_COND_AXIS_RESOLUTION-21 ""
*MEAS_COND_AXIS_RESOLUTION-22 ""
*MEAS_COND_AXIS_RESOLUTION-23 ""
*MEAS_COND_AXIS_RESOLUTION-24 "0.001"
*MEAS_COND_AXIS_RESOLUTION-25 "0.002"
*MEAS_COND_AXIS_RESOLUTION-26 "0.0001"
*MEAS_COND_AXIS_RESOLUTION-27 "0.001"
*MEAS_COND_AXIS_RESOLUTION-28 "0.002"
*MEAS_COND_AXIS_RESOLUTION-29 "0.004"
*MEAS_COND_AXIS_RESOLUTION-3 "0.0001"
*MEAS_COND_AXIS_RESOLUTION-30 "0.01"
*MEAS_COND_AXIS_RESOLUTION-31 "0.002"
*MEAS_COND_AXIS_RESOLUTION-32 "0.0001"
*MEAS_COND_AXIS_RESOLUTION-33 "1/1600"
*MEAS_COND_AXIS_RESOLUTION-34 "1/1600"
*MEAS_COND_AXIS_RESOLUTION-35 "1/1600"
*MEAS_COND_AXIS_RESOLUTION-36 "0.0003125"
*MEAS_COND_AXIS_RESOLUTION-37 ""
*MEAS_COND_AXIS_RESOLUTION-38 ""
*MEAS_COND_AXIS_RESOLUTION-39 ""
*MEAS_COND_AXIS_RESOLUTION-4 "0.0002"
*MEAS_COND_AXIS_RESOLUTION-40 "0.1"
*MEAS_COND_AXIS_RESOLUTION-41 "0.1"
*MEAS_COND_AXIS_RESOLUTION-42 "0.1"
*MEAS_COND_AXIS_RESOLUTION-43 ""
*MEAS_COND_AXIS_RESOLUTION-44 ""
*MEAS_COND_AXIS_RESOLUTION-45 "1"
*MEAS_COND_AXIS_RESOLUTION-46 ""
*MEAS_COND_AXIS_RESOLUTION-47 ""
*MEAS_COND_AXIS_RESOLUTION-48 ""
*MEAS_COND_AXIS_RESOLUTION-49 "0.01"
*MEAS_COND_AXIS_RESOLUTION-5 "0.0002"
*MEAS_COND_AXIS_RESOLUTION-50 "0.010000"
*MEAS_COND_AXIS_RESOLUTION-51 "0.01"
*MEAS_COND_AXIS_RESOLUTION-52 "0.01"
*MEAS_COND_AXIS_RESOLUTION-53 "0.01"
*MEAS_COND_AXIS_RESOLUTION-54 "1"
*MEAS_COND_AXIS_RESOLUTION-55 "1"
*MEAS_COND_AXIS_RESOLUTION-56 "1"
*MEAS_COND_AXIS_RESOLUTION-57 "1"
*MEAS_COND_AXIS_RESOLUTION-58 "0.01"
*MEAS_COND_AXIS_RESOLUTION-59 "0.01"
*MEAS_COND_AXIS_RESOLUTION-6 "0.0001"
*MEAS_COND_AXIS_RESOLUTION-60 "0.01"
*MEAS_COND_AXIS_RESOLUTION-61 "0.01"
*MEAS_COND_AXIS_RESOLUTION-62 "0.010000"
*MEAS_COND_AXIS_RESOLUTION-63 "1"
*MEAS_COND_AXIS_RESOLUTION-64 "0.100000"
*MEAS_COND_AXIS_RESOLUTION-65 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-66 "0.010000"
*MEAS_COND_AXIS_RESOLUTION-67 "0.01"
*MEAS_COND_AXIS_RESOLUTION-68 "0.01"
*MEAS_COND_AXIS_RESOLUTION-69 "0.1"
*MEAS_COND_AXIS_RESOLUTION-7 "0.0005"
*MEAS_COND_AXIS_RESOLUTION-70 "1"
*MEAS_COND_AXIS_RESOLUTION-71 "0.01"
*MEAS_COND_AXIS_RESOLUTION-72 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-73 "1"
*MEAS_COND_AXIS_RESOLUTION-74 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-75 "1"
*MEAS_COND_AXIS_RESOLUTION-76 "0.01"
*MEAS_COND_AXIS_RESOLUTION-77 "1"
*MEAS_COND_AXIS_RESOLUTION-78 "1"
*MEAS_COND_AXIS_RESOLUTION-79 "1"
*MEAS_COND_AXIS_RESOLUTION-8 ""
*MEAS_COND_AXIS_RESOLUTION-80 "0.01"
*MEAS_COND_AXIS_RESOLUTION-81 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-82 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-83 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-84 "-"
*MEAS_COND_AXIS_RESOLUTION-85 "-"
*MEAS_COND_AXIS_RESOLUTION-86 "-"
*MEAS_COND_AXIS_RESOLUTION-87 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-88 "-"
*MEAS_COND_AXIS_RESOLUTION-89 "-"
*MEAS_COND_AXIS_RESOLUTION-9 "0.0000625"
*MEAS_COND_AXIS_RESOLUTION-90 "-"
*MEAS_COND_AXIS_RESOLUTION-91 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-92 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-93 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-94 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-95 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-96 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-97 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-98 ""
*MEAS_COND_AXIS_STATE-0 "Fixed"
*MEAS_COND_AXIS_STATE-1 "Fixed"
*MEAS_COND_AXIS_STATE-10 "Fixed"
*MEAS_COND_AXIS_STATE-11 "Fixed"
*MEAS_COND_AXIS_STATE-12 "Fixed"
*MEAS_COND_AXIS_STATE-13 "Fixed"
*MEAS_COND_AXIS_STATE-14 "Fixed"
*MEAS_COND_AXIS_STATE-15 "Fixed"
*MEAS_COND_AXIS_STATE-16 "Fixed"
*MEAS_COND_AXIS_STATE-17 "Fixed"
*MEAS_COND_AXIS_STATE-18 "Fixed"
*MEAS_COND_AXIS_STATE-19 "Fixed"
*MEAS_COND_AXIS_STATE-2 "Fixed"
*MEAS_COND_AXIS_STATE-20 "Fixed"
*MEAS_COND_AXIS_STATE-21 "Fixed"
*MEAS_COND_AXIS_STATE-22 "Fixed"
*MEAS_COND_AXIS_STATE-23 "Fixed"
*MEAS_COND_AXIS_STATE-24 "Fixed"
*MEAS_COND_AXIS_STATE-25 "Fixed"
*MEAS_COND_AXIS_STATE-26 "Fixed"
*MEAS_COND_AXIS_STATE-27 "Fixed"
*MEAS_COND_AXIS_STATE-28 "Fixed"
*MEAS_COND_AXIS_STATE-29 "Fixed"
*MEAS_COND_AXIS_STATE-3 "Fixed"
*MEAS_COND_AXIS_STATE-30 "Fixed"
*MEAS_COND_AXIS_STATE-31 "Fixed"
*MEAS_COND_AXIS_STATE-32 "Fixed"
*MEAS_COND_AXIS_STATE-33 "Fixed"
*MEAS_COND_AXIS_STATE-34 "Fixed"
*MEAS_COND_AXIS_STATE-35 "Fixed"
*MEAS_COND_AXIS_STATE-36 "Fixed"
*MEAS_COND_AXIS_STATE-37 "Fixed"
*MEAS_COND_AXIS_STATE-38 "Fixed"
*MEAS_COND_AXIS_STATE-39 "Fixed"
*MEAS_COND_AXIS_STATE-4 "Scan"
*MEAS_COND_AXIS_STATE-40 "Fixed"
*MEAS_COND_AXIS_STATE-41 "Fixed"
*MEAS_COND_AXIS_STATE-42 "Fixed"
*MEAS_COND_AXIS_STATE-43 "Fixed"
*MEAS_COND_AXIS_STATE-44 "Fixed"
*MEAS_COND_AXIS_STATE-45 "Fixed"
*MEAS_COND_AXIS_STATE-46 "Fixed"
*MEAS_COND_AXIS_STATE-47 "Fixed"
*MEAS_COND_AXIS_STATE-48 "Fixed"
*MEAS_COND_AXIS_STATE-49 "Fixed"
*MEAS_COND_AXIS_STATE-5 "Fixed"
*MEAS_COND_AXIS_STATE-50 "Fixed"
*MEAS_COND_AXIS_STATE-51 "Fixed"
*MEAS_COND_AXIS_STATE-52 "Fixed"
*MEAS_COND_AXIS_STATE-53 "Fixed"
*MEAS_COND_AXIS_STATE-54 "Fixed"
*MEAS_COND_AXIS_STATE-55 "Fixed"
*MEAS_COND_AXIS_STATE-56 "Fixed"
*MEAS_COND_AXIS_STATE-57 "Fixed"
*MEAS_COND_AXIS_STATE-58 "Fixed"
*MEAS_COND_AXIS_STATE-59 "Fixed"
*MEAS_COND_AXIS_STATE-6 "Fixed"
*MEAS_COND_AXIS_STATE-60 "Fixed"
*MEAS_COND_AXIS_STATE-61 "Fixed"
*MEAS_COND_AXIS_STATE-62 "Fixed"
*MEAS_COND_AXIS_STATE-63 "Fixed"
*MEAS_COND_AXIS_STATE-64 "Fixed"
*MEAS_COND_AXIS_STATE-65 "Fixed"
*MEAS_COND_AXIS_STATE-66 "Fixed"
*MEAS_COND_AXIS_STATE-67 "Fixed"
*MEAS_COND_AXIS_STATE-68 "Fixed"
*MEAS_COND_AXIS_STATE-69 "Fixed"
*MEAS_COND_AXIS_STATE-7 "Fixed"
*MEAS_COND_AXIS_STATE-70 "Fixed"
*MEAS_COND_AXIS_STATE-71 "Fixed"
*MEAS_COND_AXIS_STATE-72 "Fixed"
*MEAS_COND_AXIS_STATE-73 "Fixed"
*MEAS_COND_AXIS_STATE-74 "Fixed"
*MEAS_COND_AXIS_STATE-75 "Fixed"
*MEAS_COND_AXIS_STATE-76 "Fixed"
*MEAS_COND_AXIS_STATE-77 "Fixed"
*MEAS_COND_AXIS_STATE-78 "Fixed"
*MEAS_COND_AXIS_STATE-79 "Fixed"
*MEAS_COND_AXIS_STATE-8 "Fixed"
*MEAS_COND_AXIS_STATE-80 "Fixed"
*MEAS_COND_AXIS_STATE-81 "Fixed"
*MEAS_COND_AXIS_STATE-82 "Fixed"
*MEAS_COND_AXIS_STATE-83 "Fixed"
*MEAS_COND_AXIS_STATE-84 "Fixed"
*MEAS_COND_AXIS_STATE-85 "Fixed"
*MEAS_COND_AXIS_STATE-86 "Fixed"
*MEAS_COND_AXIS_STATE-87 "Fixed"
*MEAS_COND_AXIS_STATE-88 "Fixed"
*MEAS_COND_AXIS_STATE-89 "Fixed"
*MEAS_COND_AXIS_STATE-9 "Fixed"
*MEAS_COND_AXIS_STATE-90 "Fixed"
*MEAS_COND_AXIS_STATE-91 "Fixed"
*MEAS_COND_AXIS_STATE-92 "Fixed"
*MEAS_COND_AXIS_STATE-93 "Fixed"
*MEAS_COND_AXIS_STATE-94 "Fixed"
*MEAS_COND_AXIS_STATE-95 "Fixed"
*MEAS_COND_AXIS_STATE-96 "Fixed"
*MEAS_COND_AXIS_STATE-97 "Fixed"
*MEAS_COND_AXIS_STATE-98 "Fixed"
*MEAS_COND_AXIS_UNIT-0 "deg"
*MEAS_COND_AXIS_UNIT-1 "deg"
*MEAS_COND_AXIS_UNIT-10 ""
*MEAS_COND_AXIS_UNIT-100 ""
*MEAS_COND_AXIS_UNIT-101 ""
*MEAS_COND_AXIS_UNIT-102 ""
*MEAS_COND_AXIS_UNIT-11 ""
*MEAS_COND_AXIS_UNIT-12 ""
*MEAS_COND_AXIS_UNIT-13 "pulse"
*MEAS_COND_AXIS_UNIT-14 ""
*MEAS_COND_AXIS_UNIT-15 ""
*MEAS_COND_AXIS_UNIT-16 ""
*MEAS_COND_AXIS_UNIT-17 "mm"
*MEAS_COND_AXIS_UNIT-18 ""
*MEAS_COND_AXIS_UNIT-19 "mm"
*MEAS_COND_AXIS_UNIT-2 "deg"
*MEAS_COND_AXIS_UNIT-20 ""
*MEAS_COND_AXIS_UNIT-21 ""
*MEAS_COND_AXIS_UNIT-22 ""
*MEAS_COND_AXIS_UNIT-23 ""
*MEAS_COND_AXIS_UNIT-24 "deg"
*MEAS_COND_AXIS_UNIT-25 "deg"
*MEAS_COND_AXIS_UNIT-26 "mm"
*MEAS_COND_AXIS_UNIT-27 "deg"
*MEAS_COND_AXIS_UNIT-28 "deg"
*MEAS_COND_AXIS_UNIT-29 "deg"
*MEAS_COND_AXIS_UNIT-3 "deg"
*MEAS_COND_AXIS_UNIT-30 "deg"
*MEAS_COND_AXIS_UNIT-31 "deg"
*MEAS_COND_AXIS_UNIT-32 "deg"
*MEAS_COND_AXIS_UNIT-33 ""
*MEAS_COND_AXIS_UNIT-34 "mm"
*MEAS_COND_AXIS_UNIT-35 ""
*MEAS_COND_AXIS_UNIT-36 "mm"
*MEAS_COND_AXIS_UNIT-37 ""
*MEAS_COND_AXIS_UNIT-38 ""
*MEAS_COND_AXIS_UNIT-39 ""
*MEAS_COND_AXIS_UNIT-4 "deg"
*MEAS_COND_AXIS_UNIT-40 ""
*MEAS_COND_AXIS_UNIT-41 "mm"
*MEAS_COND_AXIS_UNIT-42 ""
*MEAS_COND_AXIS_UNIT-43 ""
*MEAS_COND_AXIS_UNIT-44 ""
*MEAS_COND_AXIS_UNIT-45 "keV"
*MEAS_COND_AXIS_UNIT-46 ""
*MEAS_COND_AXIS_UNIT-47 ""
*MEAS_COND_AXIS_UNIT-48 ""
*MEAS_COND_AXIS_UNIT-49 "H"
*MEAS_COND_AXIS_UNIT-5 "deg"
*MEAS_COND_AXIS_UNIT-50 "H"
*MEAS_COND_AXIS_UNIT-51 "H"
*MEAS_COND_AXIS_UNIT-52 "H"
*MEAS_COND_AXIS_UNIT-53 "H"
*MEAS_COND_AXIS_UNIT-54 "mV"
*MEAS_COND_AXIS_UNIT-55 "V"
*MEAS_COND_AXIS_UNIT-56 "V"
*MEAS_COND_AXIS_UNIT-57 "V"
*MEAS_COND_AXIS_UNIT-58 "H"
*MEAS_COND_AXIS_UNIT-59 "H"
*MEAS_COND_AXIS_UNIT-6 "deg"
*MEAS_COND_AXIS_UNIT-60 "H"
*MEAS_COND_AXIS_UNIT-61 "H"
*MEAS_COND_AXIS_UNIT-62 "H"
*MEAS_COND_AXIS_UNIT-63 "uS/m"
*MEAS_COND_AXIS_UNIT-64 "L/min"
*MEAS_COND_AXIS_UNIT-65 "degree"
*MEAS_COND_AXIS_UNIT-66 "MPa"
*MEAS_COND_AXIS_UNIT-67 "MPa"
*MEAS_COND_AXIS_UNIT-68 "MPa"
*MEAS_COND_AXIS_UNIT-69 "L/min"
*MEAS_COND_AXIS_UNIT-7 "deg"
*MEAS_COND_AXIS_UNIT-70 "C"
*MEAS_COND_AXIS_UNIT-71 "MPa"
*MEAS_COND_AXIS_UNIT-72 "degree"
*MEAS_COND_AXIS_UNIT-73 "percent"
*MEAS_COND_AXIS_UNIT-74 "degree"
*MEAS_COND_AXIS_UNIT-75 "C"
*MEAS_COND_AXIS_UNIT-76 "H"
*MEAS_COND_AXIS_UNIT-77 "time"
*MEAS_COND_AXIS_UNIT-78 "msec"
*MEAS_COND_AXIS_UNIT-79 "msec"
*MEAS_COND_AXIS_UNIT-8 ""
*MEAS_COND_AXIS_UNIT-80 "H"
*MEAS_COND_AXIS_UNIT-81 "times"
*MEAS_COND_AXIS_UNIT-82 "msec"
*MEAS_COND_AXIS_UNIT-83 "msec"
*MEAS_COND_AXIS_UNIT-84 "-"
*MEAS_COND_AXIS_UNIT-85 "-"
*MEAS_COND_AXIS_UNIT-86 "-"
*MEAS_COND_AXIS_UNIT-87 ""
*MEAS_COND_AXIS_UNIT-88 "-"
*MEAS_COND_AXIS_UNIT-89 "-"
*MEAS_COND_AXIS_UNIT-9 "mm"
*MEAS_COND_AXIS_UNIT-90 "-"
*MEAS_COND_AXIS_UNIT-91 ""
*MEAS_COND_AXIS_UNIT-92 ""
*MEAS_COND_AXIS_UNIT-93 ""
*MEAS_COND_AXIS_UNIT-94 ""
*MEAS_COND_AXIS_UNIT-95 ""
*MEAS_COND_AXIS_UNIT-96 ""
*MEAS_COND_AXIS_UNIT-97 ""
*MEAS_COND_AXIS_UNIT-98 ""
*MEAS_COND_AXIS_UNIT-99 ""
*MEAS_COND_COUNTER_CENTER_X "382.5"
*MEAS_COND_COUNTER_CENTER_Y "197.5"
*MEAS_COND_COUNTER_COUNTMODE "Differential"
*MEAS_COND_COUNTER_DEADTIMECORRECTION "Enabled"
*MEAS_COND_COUNTER_DISTANCE "300"
*MEAS_COND_COUNTER_ENERGYMODE "Standard"
*MEAS_COND_COUNTER_INTEGRALMODE "Line"
*MEAS_COND_COUNTER_PHA_UNIT "keV"
*MEAS_COND_COUNTER_PHABASE "6.0"
*MEAS_COND_COUNTER_PHAWINDOW "4.0"
*MEAS_COND_COUNTER_PITCH_X "0.1"
*MEAS_COND_COUNTER_PITCH_Y "0.1"
*MEAS_COND_COUNTER_PITCHUNIT "mm"
*MEAS_COND_COUNTER_VALIDWIDTH_X "200"
*MEAS_COND_COUNTER_VALIDWIDTH_Y "201"
*MEAS_COND_OPT_ATTR "BB"
*MEAS_COND_OPT_NAME "User defined settings"
*MEAS_COND_XG_CURRENT "{meta_dict.get('MEAS_COND_XG_CURRENT', '30')}"
*MEAS_COND_XG_VOLTAGE "{meta_dict.get('MEAS_COND_XG_VOLTAGE', '40')}"
*MEAS_COND_XG_WAVE_TYPE "Ka"
*MEAS_DATA_COUNT "{data_count}"
*MEAS_SCAN_AXIS_X "{meta_dict.get('MEAS_SCAN_AXIS_X', 'TwoThetaTheta')}"
*MEAS_SCAN_AXIS_X_INTERNAL "TwoThetaTheta"
*MEAS_SCAN_END_TIME "{formatted_end_time}"
*MEAS_SCAN_MODE "{meta_dict.get('MEAS_SCAN_MODE', 'CONTINUOUS')}"
*MEAS_SCAN_MODE_INTERNAL "TDI_1D"
*MEAS_SCAN_RESOLUTION_X "0.0002"
*MEAS_SCAN_SPEED "{meta_dict.get('MEAS_SCAN_SPEED', '5.0000')}"
*MEAS_SCAN_SPEED_UNIT "{meta_dict.get('MEAS_SCAN_SPEED_UNIT', 'deg/min')}"
*MEAS_SCAN_START "{start_angle}"
*MEAS_SCAN_START_TIME "{formatted_start_time}"
*MEAS_SCAN_STEP "{step_size}"
*MEAS_SCAN_STOP "{stop_angle}"
*MEAS_SCAN_UNEQUALY_SPACED "False"
*MEAS_SCAN_UNIT_X "deg"
*MEAS_SCAN_UNIT_Y "counts"
*RAS_HEADER_END"""

        data_lines = []
        for _, row in data_df.iterrows():
            data_lines.append(f"{row['2Theta']:.6f} {row['Intensity']:.4f} 1.0000")

        return "\n".join([
            header_template,
            "*RAS_INT_START",
            *data_lines,
            "*RAS_INT_END",
            "*RAS_DATA_END",
            "*DSC_DATA_END"
        ])

    def generate_raw(metadata_df, data_df):
        meta_dict = pd.Series(metadata_df.Value.values, index=metadata_df.Parameter).to_dict()
        header = bytearray(2600)

        try:
            start_angle = float(meta_dict.get('Start Angle (°)', data_df['2Theta'].min()))
            num_points = len(data_df)
            step_size = float(meta_dict.get('Step Size (°)', (data_df['2Theta'].iloc[1] - data_df['2Theta'].iloc[0])))
            time_per_step = float(meta_dict.get('Time per Step (s)', 1.0))
            k_alpha1 = float(meta_dict.get('K-Alpha1 (Å)', 1.54060))
            k_alpha2 = float(meta_dict.get('K-Alpha2 (Å)', 1.54439))
            k_beta = float(meta_dict.get('K-Beta (Å)', 1.39225))
            target_name = meta_dict.get('X-ray Target', 'Cu').ljust(12).encode('utf-8')

            struct.pack_into('<f', header, 136, start_angle)
            struct.pack_into('<f', header, 140, step_size)
            struct.pack_into('<i', header, 148, num_points)
            struct.pack_into('<f', header, 152, time_per_step)
            struct.pack_into('12s', header, 244, target_name)
            struct.pack_into('<f', header, 308, k_alpha1)
            struct.pack_into('<f', header, 312, k_alpha2)
            struct.pack_into('<f', header, 316, k_beta)
            date_str = datetime.now().strftime("%d-%b-%Y").ljust(9).encode('utf-8')
            struct.pack_into('9s', header, 38, date_str)
        except Exception as e:
            st.error(f"Error preparing RAW file header: {e}")
            return None

        intensities = data_df['Intensity'].values.astype(np.float32)
        data_bytes = intensities.tobytes()
        return bytes(header) + data_bytes

    def get_default_metadata(format_type='XRDML'):
        now = datetime.now()
        if format_type == 'RAS':
            metadata = {
                'FILE_OPERATOR': 'XRDlicious User', 'HW_XG_TARGET_NAME': 'Cu',
                'HW_XG_WAVE_LENGTH_ALPHA1': '1.540593', 'HW_XG_WAVE_LENGTH_ALPHA2': '1.544414',
                'HW_XG_WAVE_LENGTH_BETA': '1.392246',
                'MEAS_COND_XG_VOLTAGE': '40', 'MEAS_COND_XG_CURRENT': '30',
                'MEAS_SCAN_AXIS_X': 'TwoThetaTheta', 'MEAS_SCAN_MODE': 'CONTINUOUS',
                'MEAS_SCAN_SPEED': '5.0000', 'MEAS_SCAN_SPEED_UNIT': 'deg/min',
                'Divergence Slit (DS)': '1/2deg', 'Scattering Slit (SS)': 'Open',
                'Receiving Slit (RS)': '20.100mm',
                'MEAS_SCAN_START_TIME': now.isoformat(),
                'MEAS_SCAN_END_TIME': (now + timedelta(minutes=20)).isoformat(),
            }
        elif format_type == 'RAW':
            metadata = {
                'X-ray Target': 'Cu', 'Time per Step (s)': '1.0',
                'K-Alpha1 (Å)': '1.54060', 'K-Alpha2 (Å)': '1.54439',
                'K-Beta (Å)': '1.39225',
            }
        else:  # XRDML
            metadata = {
                'Sample ID': '000000-0000 - Converted with XRDlicious', 'Sample Name': 'Converted Sample Name',
                'Status': 'Completed', 'Measurement Type': 'Scan', 'Scan Mode': 'Continuous',
                'Scan Axis': 'Gonio', 'Start Time': now.isoformat(),
                'End Time': (now + timedelta(minutes=30)).isoformat(),
                'Author': 'XRDlicious User', 'Application Software': 'XRDlicious Converter',
                'K-Alpha1 Wavelength (Å)': '1.54056', 'K-Alpha2 Wavelength (Å)': '1.54439',
                'K-Beta Wavelength (Å)': '1.39225', 'Ratio K-Alpha2/K-Alpha1': '0.5',
                'Common Counting Time (s)': '1.0', 'X-ray Tube Tension (kV)': '45',
                'X-ray Tube Current (mA)': '40', 'Anode Material': 'Cu',
            }
        return metadata_dataframe(metadata)

    st.markdown("### 📜 XRD File Format Converter (.xrdml, .ras, .rasx, .raw, .xy)")
    with st.expander(f"How to **Cite**", icon="📚", expanded=False):
        st.markdown("""
        If you like the app, please cite the following source:
        - **XRDlicious, 2025** – [Lebeda, Miroslav, et al. XRDlicious: an interactive web-based platform for online calculation of diffraction patterns and radial distribution functions from crystal structures. Journal of Applied Crystallography, 2025, 58.5.](https://journals.iucr.org/j/issues/2025/05/00/hat5006/index.html).
        """)
    #st.markdown(
    #    """
    #    <div style="background-color:#f8d7da; padding:6px 10px; border-radius:4px; border:1px solid #f5c2c7; width: fit-content;">
    #        <span style="color:#842029; font-size:14px;">🔧 Testing mode</span>
    #    </div>
    #    """,
    #    unsafe_allow_html=True
    #)
    st.info(
        "📄🔁📄 Upload one or more powder diffraction files and convert them to **.xy, .xrdml, .ras or .raw**. "
        "Accepted input: **.xrdml, .ras, .rasx, .raw** and two-column data (**.xy, .dat, .txt, .csv**) — "
        "formats may be **mixed** in one upload. \n\n ℹ️ For **.raw** (Bruker) files, please check that the converted "
        "axis values are reasonable. \n\n **Batch mode** is automatically activated when multiple files are uploaded.")

    # The uploader key holds a counter. Incrementing it (in the clear button
    # callback below) forces Streamlit to mount a fresh, empty file_uploader,
    # which is the reliable way to programmatically remove all uploaded files.
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    def _clear_uploaded_files():
        st.session_state.uploader_key += 1

    uploaded_files_raw = st.file_uploader("Upload Data File(s)",
                                          type=["xrdml", "xml", "ras", "rasx", "xy", "dat", "txt", "raw", "csv"],
                                          accept_multiple_files=True,
                                          key=f"file_uploader_{st.session_state.uploader_key}")

    def load_pattern(uploaded_file):
        """Parse any supported upload into (data_df, key_metadata, full_metadata, x_axis).

        x_axis is only detected for two-column text files (electron-diffraction
        headers); the instrument formats are always 2θ.
        """
        ext = uploaded_file.name.lower().rsplit('.', 1)[-1]
        if ext == 'raw':
            key_metadata, df = parse_raw(uploaded_file.getvalue())
            return df, key_metadata, key_metadata, None
        if ext == 'rasx':
            full_metadata, key_metadata, df = parse_rasx(uploaded_file)
            return df, key_metadata, full_metadata, None
        text = uploaded_file.getvalue().decode("utf-8", errors='replace')
        if ext == 'ras':
            full_metadata, key_metadata, df = parse_ras(text)
            return df, key_metadata, full_metadata, None
        if ext in ('xrdml', 'xml'):
            key_metadata, df = parse_xrdml(text)
            return df, key_metadata, key_metadata, None
        return parse_xy(text), {}, {}, detect_x_axis(text)

    def output_names(files, new_ext):
        """Output filename per upload; with mixed formats, sample.csv and
        sample.xy would both become sample.<new_ext>, so colliding names keep
        the source extension (sample_csv.<new_ext>, sample_xy.<new_ext>)."""
        stems = [f.name.rsplit('.', 1)[0] for f in files]
        names = []
        for f, stem in zip(files, stems):
            if stems.count(stem) > 1:
                stem = f"{stem}_{f.name.rsplit('.', 1)[-1].lower()}"
            names.append(f"{stem}.{new_ext}")
        return names

    if uploaded_files_raw:
        if isinstance(uploaded_files_raw, list):
            uploaded_files = uploaded_files_raw
        else:
            uploaded_files = [uploaded_files_raw]

        input_exts = sorted({f.name.lower().rsplit('.', 1)[-1] for f in uploaded_files})
        formats_label = ", ".join(f"**.{e}**" for e in input_exts)
        all_instrument = all(e in ('xrdml', 'xml', 'ras', 'rasx', 'raw') for e in input_exts)

        msg_col, clear_col = st.columns([4, 1])
        with msg_col:
            if len(uploaded_files) > 1:
                st.success(
                    f"✅ Successfully uploaded **{len(uploaded_files)}** files ({formats_label} format) - Batch mode activated")
            else:
                st.success(f"✅ Successfully uploaded **1** file ({formats_label} format)")
        with clear_col:
            apply_button_style(["remove_all_files_format"])
            st.button("🗑️ Remove all files",
                      key="remove_all_files_format",
                      on_click=_clear_uploaded_files,
                      type="primary",
                      width="stretch")

        # In batch mode the user can pick which file is shown in the preview.
        # The controlling slider is rendered below the preview plot; read its
        # stored value here (before the widget itself exists).
        preview_idx = 0
        if len(uploaded_files) > 1:
            names = [f.name for f in uploaded_files]
            stored_name = st.session_state.get("ffc_preview_file_name")
            if stored_name in names:
                preview_idx = names.index(stored_name)
            elif stored_name is not None:
                del st.session_state["ffc_preview_file_name"]

        def _render_preview_file_slider():
            if len(uploaded_files) > 1:
                st.select_slider(
                    "Preview file",
                    options=[f.name for f in uploaded_files],
                    key="ffc_preview_file_name",
                )

        first_file = uploaded_files[preview_idx]
        file_ext = first_file.name.lower().rsplit('.', 1)[-1]
        is_batch = len(uploaded_files) > 1
        data_df, key_metadata, full_metadata, x_axis = load_pattern(first_file)
        if data_df is None:
            st.error(f"Could not read any data from `{first_file.name}`.")
            return

        # Electron-diffraction files (e.g. ProcessDiffraction for TEM) declare
        # a k = 2π/d, s = 1/d or d x-axis in their header. The values are
        # written to the output unchanged; the header only labels the axis.
        is_2theta = x_axis is None or x_axis[0] == '2theta'
        x_label = "2θ (°)" if is_2theta else x_axis[2]
        u = "°" if is_2theta else ""  # unit appended to step values

        # .xrdml stores only the start/end angle (and .ras/.raw a single
        # step), so the points must be evenly spaced — otherwise every
        # intensity ends up at the wrong angle. An irregular x column is
        # interpolated onto an even grid before export.
        grid_info = check_regular_grid(data_df)
        interp_step = None

        def _export_df(df):
            info = check_regular_grid(df)
            if info['regular']:
                return df
            return resample_to_regular_grid(df, interp_step or info['median_step'])

        col1, col2 = st.columns([1, 1.5])
        with col1:
            if key_metadata:
                st.markdown(f"#### 📝 Key Parameters (from `{first_file.name}`)")
                st.table(metadata_dataframe(key_metadata))

                if full_metadata and file_ext in ['ras', 'rasx']:
                    with st.expander("Show Full Raw Header"):
                        st.dataframe(metadata_dataframe(full_metadata),
                                     height=300)

            st.markdown("#### 📝 Edit Details for Output File(s)")
            if is_batch:
                st.info(f"These settings will be applied to all **{len(uploaded_files)} files**.")

            # Instrument files are mostly converted to .xy, two-column data to
            # an instrument format, so that is what is preselected.
            output_format = st.selectbox("Select Output Format", ['XY', 'XRDML', 'RAS', 'RAW'],
                                         index=0 if all_instrument else 1)

            if output_format == 'XY':
                include_header = st.checkbox("Include header in .xy file", value=False)

                if is_batch:
                    st.write(f"**Batch conversion for {len(uploaded_files)} files.**")
                    zip_filename = st.text_input(
                        "Enter filename for the .zip download:",
                        f"converted_xy_files_{timestamp_suffix()}.zip",
                        key="zip_name_to_xy")
                    if not zip_filename.lower().endswith('.zip'):
                        zip_filename += '.zip'
                    if st.button("📦 Prepare All as .xy for Download (.zip)", type="primary", width='stretch'):
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                            for uploaded_file, new_filename in zip(uploaded_files,
                                                                   output_names(uploaded_files, 'xy')):
                                df_to_convert = load_pattern(uploaded_file)[0]
                                if df_to_convert is not None:
                                    zf.writestr(new_filename, convert_to_xy(df_to_convert, include_header))

                        st.download_button(
                            label="📦 Download ZIP",
                            data=zip_buffer.getvalue(),
                            file_name=zip_filename,
                            mime="application/zip",
                            width='stretch',
                            on_click=log_download,
                            args=("File Format Converter",),
                        )
                else:
                    default_name = first_file.name.rsplit('.', 1)[0] + '.xy'
                    download_filename = st.text_input("Enter filename for download:", default_name)
                    xy_data = convert_to_xy(data_df, include_header)
                    st.download_button("⬇️ Download as .xy File", xy_data, download_filename, "text/plain",
                                       type="primary", width='stretch',
                                       on_click=log_download,
                                       args=("File Format Converter",))
            else:
                if not is_2theta:
                    st.info(
                        f"ℹ️ The header of `{first_file.name}` declares the x-axis as "
                        f"**{x_axis[2]}** (e.g. TEM electron diffraction). The x values are "
                        f"kept unchanged in the output file.")
                if not grid_info['regular']:
                    interp_step = st.number_input(
                        f"Step for interpolation{' (°)' if is_2theta else ''}",
                        min_value=1e-6,
                        value=float(f"{grid_info['median_step']:.6g}"),
                        step=float(f"{grid_info['median_step'] / 10:.2g}"),
                        format="%.6f",
                        key=f"xy_interp_step_{st.session_state.uploader_key}",
                        help="Spacing of the regular grid written to the output file. "
                             "Defaults to the median step of the data. In batch mode "
                             "it is used for every file whose grid is irregular.")

                # The table is prefilled with the instrument settings (anode,
                # wavelengths, tube, operator, times, ...) of the first uploaded
                # file that has any; two-column files carry none.
                source_file, source_settings = None, {}
                for f in uploaded_files:
                    if f.name.lower().rsplit('.', 1)[-1] not in ('xrdml', 'xml', 'ras', 'rasx', 'raw'):
                        continue
                    df_, key_, full_, _ = (load_pattern(f) if f is not first_file
                                           else (data_df, key_metadata, full_metadata, None))
                    if df_ is not None and key_:
                        source_file = f
                        source_settings = instrument_settings(
                            key_, full_, check_regular_grid(df_)['median_step'])
                        break

                # The anode is offered as a one-click setting instead of four
                # values typed into the table below. It starts from the anode
                # of the uploaded files, and falls back to Cu for .xy files.
                # The key follows the uploads, so new files reset it. The choice
                # is also kept outside the widget: with XY output the widget is
                # not shown, and Streamlit then forgets its value.
                upload_names = tuple(f.name for f in uploaded_files)
                upload_sig = f"{st.session_state.uploader_key}_{abs(hash(upload_names))}"
                source_anode = source_settings.get('anode', '').capitalize()
                anode_options = list(ANODE_PRESETS)
                anode_pref_key = f"anode_pref_{upload_sig}"
                default_anode = st.session_state.get(anode_pref_key, source_anode)
                anode_choice = st.selectbox(
                    "Quick setting: X-ray source",
                    anode_options,
                    index=anode_options.index(default_anode) if default_anode in anode_options else 0,
                    key=f"anode_quick_choice_{upload_sig}",
                    help="Fills the anode and wavelength rows of the "
                         "table below with the characteristic values of "
                         "the chosen tube — only the rows the selected "
                         "output format knows.",
                )
                st.session_state[anode_pref_key] = anode_choice

                # Metadata settings apply to all uploaded files, so key the
                # table state by output format only (not the previewed file)
                # — switching the preview file then keeps the edited values.
                df_state_key = f"meta_df_{output_format}"
                # Tracks whether the user has clicked "Apply Changes" for the
                # current output format; the download section only appears once
                # this is True.
                applied_key = f"meta_applied_{output_format}"
                anode_state_key = f"meta_anode_{output_format}"
                editor_ver_key = f"meta_editor_ver_{output_format}"

                def _bump_editor():
                    # The editor keeps its own copy of the edits, so it is
                    # rebuilt under a new key to show the filled-in values.
                    st.session_state[editor_ver_key] = int(
                        st.session_state.get(editor_ver_key, 0)) + 1

                # Rebuilt for a new output format or new uploads: defaults, then
                # the settings of the uploaded file, then the quick setting if
                # the user picked another anode than the file has (so it is not
                # lost on a format switch).
                table_sig = (output_format, upload_sig)
                if st.session_state.get('meta_table_source') != table_sig:
                    table = apply_instrument_settings(
                        get_default_metadata(output_format), output_format, source_settings)
                    if anode_choice in ANODE_PRESETS and anode_choice != source_anode:
                        table = apply_anode_preset(table, output_format, anode_choice)
                    st.session_state[df_state_key] = table
                    st.session_state[anode_state_key] = anode_choice
                    st.session_state[applied_key] = False
                    st.session_state['meta_table_source'] = table_sig
                    _bump_editor()

                # A later change of the quick setting overwrites the anode and
                # wavelength rows (also those taken over from the file).
                if (anode_choice in ANODE_PRESETS
                        and st.session_state.get(anode_state_key) != anode_choice):
                    st.session_state[df_state_key] = apply_anode_preset(
                        st.session_state[df_state_key], output_format, anode_choice)
                    st.session_state[anode_state_key] = anode_choice
                    st.session_state[applied_key] = False
                    _bump_editor()

                if source_file is not None:
                    st.caption(f"📋 Prefilled with the instrument settings of `{source_file.name}`.")
                keep_own_settings = False
                if is_batch and source_file is not None:
                    keep_own_settings = st.checkbox(
                        "Keep each file's own instrument settings", value=True,
                        key=f"keep_own_settings_{upload_sig}",
                        help="Files that carry instrument settings (.xrdml, .ras, .rasx, "
                             ".raw) keep their own anode, wavelengths, tube voltage and "
                             "current, operator, times, counting time / scan speed and "
                             "slits in the output. The table then supplies only what a "
                             "file does not have, and everything for two-column files.")

                st.caption("✏️ All values in the table below can be edited.")

                edited_df = st.data_editor(
                    st.session_state[df_state_key], num_rows="dynamic", height=425,
                    key=f"editor_{output_format}_v"
                        f"{st.session_state.get(editor_ver_key, 0)}")

                if st.button("Apply Changes & Prepare Download", type="primary", width='stretch'):
                    st.session_state[df_state_key] = edited_df
                    st.session_state[applied_key] = True
                    st.success(f"Settings applied. {output_format} file is ready for download below.")

                file_extensions = {'XRDML': 'xrdml', 'RAS': 'ras', 'RAW': 'raw'}
                file_extension = file_extensions.get(output_format, 'txt')

                if st.session_state.get(applied_key):
                    applied_metadata_df = st.session_state[df_state_key]
                    if is_batch:
                        zip_filename = st.text_input(
                            "Enter filename for the .zip download:",
                            f"converted_to_{output_format}_{timestamp_suffix()}.zip",
                            key=f"zip_name_from_xy_{output_format}")
                        if not zip_filename.lower().endswith('.zip'):
                            zip_filename += '.zip'
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                            for uploaded_file, new_filename in zip(uploaded_files,
                                                                   output_names(uploaded_files, file_extension)):
                                df_to_convert, key_, full_, _ = load_pattern(uploaded_file)
                                if df_to_convert is not None:
                                    file_metadata_df = applied_metadata_df
                                    if keep_own_settings and key_:
                                        file_metadata_df = apply_instrument_settings(
                                            applied_metadata_df, output_format, instrument_settings(
                                                key_, full_, check_regular_grid(df_to_convert)['median_step']))
                                    df_to_convert = _export_df(df_to_convert)
                                    file_content_to_download = None
                                    if output_format == 'RAS':
                                        file_content_to_download = generate_ras(file_metadata_df, df_to_convert)
                                    elif output_format == 'XRDML':
                                        file_content_to_download = generate_xrdml(file_metadata_df,
                                                                                  df_to_convert,
                                                                                  filename=uploaded_file.name)
                                    elif output_format == 'RAW':
                                        file_content_to_download = generate_raw(file_metadata_df, df_to_convert)

                                    if file_content_to_download:
                                        zf.writestr(new_filename, file_content_to_download)

                        st.download_button(
                            label=f"📦 Download All as {output_format} (.zip)",
                            data=zip_buffer.getvalue(),
                            file_name=zip_filename,
                            mime="application/zip",
                            type="primary",
                            width='stretch',
                            on_click=log_download,
                            args=("File Format Converter",),
                        )
                    else:
                        default_name = first_file.name.rsplit('.', 1)[0] + f'.{file_extension}'
                        download_filename = st.text_input("Enter filename for download:", default_name)

                        file_content_to_download = None
                        mime_type = 'application/octet-stream'
                        export_df = _export_df(data_df)
                        if output_format == 'RAS':
                            file_content_to_download = generate_ras(applied_metadata_df, export_df)
                            mime_type = 'text/plain'
                        elif output_format == 'XRDML':
                            file_content_to_download = generate_xrdml(applied_metadata_df, export_df,
                                                                      filename=first_file.name)
                            mime_type = 'application/xml'
                        elif output_format == 'RAW':
                            file_content_to_download = generate_raw(applied_metadata_df, export_df)

                        if file_content_to_download:
                            st.download_button(
                                label=f"⬇️ Download as .{file_extension}",
                                data=file_content_to_download,
                                file_name=download_filename,
                                mime=mime_type,
                                type="primary",
                                width='stretch',
                                on_click=log_download,
                                args=("File Format Converter",),
                            )

        with col2:
            st.markdown("#### 📈 Diffraction Pattern")
            # The interpolation only happens for the instrument formats; a .xy
            # output keeps the points as they are.
            show_interp = output_format != 'XY' and not grid_info['regular']
            if not show_interp:
                fig = go.Figure(
                    go.Scatter(x=data_df['2Theta'], y=data_df['Intensity'], mode='lines', name='Intensity'))
            else:
                preview_df = _export_df(data_df)
                out_info = check_regular_grid(preview_df)
                fig = go.Figure([
                    go.Scatter(x=data_df['2Theta'], y=data_df['Intensity'], mode='markers',
                               name='Original data', marker=dict(size=6, color='rgba(120,120,120,0.6)')),
                    go.Scatter(x=preview_df['2Theta'], y=preview_df['Intensity'], mode='lines',
                               name=f'Interpolated (exported, step {out_info["implied_step"]:.6g}{u})'),
                ])
                st.caption(
                    f"Exported grid: {preview_df['2Theta'].iloc[0]:.6g}{u} → "
                    f"{preview_df['2Theta'].iloc[-1]:.6g}{u}, step {out_info['implied_step']:.6g}{u}, "
                    f"{len(preview_df)} points (original: {len(data_df)} points).")
            fig.update_layout(
                title=dict(text=f"Data from {first_file.name}", font=dict(size=24)),
                xaxis_title=x_label, yaxis_title="Intensity",
                height=550, margin=dict(l=40, r=40, t=60, b=40),
                font=dict(size=18),
                xaxis=dict(title_font=dict(size=22), tickfont=dict(size=16)),
                yaxis=dict(title_font=dict(size=22), tickfont=dict(size=16)),
                legend=dict(font=dict(size=18)))
            st.plotly_chart(fig, width='stretch')
            if show_interp:
                st.warning(
                    f"⚠️ The x column of `{first_file.name}` is **not evenly spaced** "
                    f"(steps from {grid_info['min_step']:.5g}{u} to {grid_info['max_step']:.5g}{u}, "
                    f"median {grid_info['median_step']:.5g}{u}). The output formats assume a "
                    f"constant step, so the data will be **linearly interpolated** onto a "
                    f"regular grid before export (shown in the chart).")
            _render_preview_file_slider()


import streamlit.components.v1 as components


import base64
import os


@st.cache_data(show_spinner=False)
def _logo_data_uri(relative_path):
    """Return a base64 data URI for an image shipped with the app (or None)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)
    try:
        with open(path, 'rb') as fh:
            mime = ("image/svg+xml" if path.endswith(".svg")
                    else "image/webp" if path.endswith(".webp") else "image/png")
            return f"data:{mime};base64," + base64.b64encode(fh.read()).decode('ascii')
    except OSError:
        return None


def display_conversion_visual():
    col_visual, col_ad = st.columns([1, 1.6])

    with col_visual:
        st.markdown("""
        <div style="margin: 25px 0; text-align: center; padding: 28px 16px; background: linear-gradient(135deg, #f8f9ff 0%, #e8f4fd 100%); border-radius: 14px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <div style="display: flex; align-items: center; justify-content: center; gap: 14px; flex-wrap: nowrap;">
                <div style="display: flex; flex-direction: column; align-items: center;">
                    <div style="background: linear-gradient(135deg, #74b9ff, #0984e3); color: white; padding: 14px 14px; border-radius: 9px; box-shadow: 0 3px 9px rgba(116, 185, 255, 0.4); margin: 4px; min-width: 104px; text-align: center;">
                        <div style="font-size: 1.15em; font-weight: bold; margin-bottom: 4px;">.xy</div>
                        <div style="font-size: 0.8em; opacity: 0.9;">.dat, .txt, .csv<br>(two-column)</div>
                    </div>
                </div>
                <div style="display: flex; flex-direction: column; align-items: center;">
                    <div style="font-size: 1.8em; color: #6c5ce7; line-height: 1.15;">→</div>
                    <div style="font-size: 1.8em; color: #6c5ce7; line-height: 1.15;">←</div>
                </div>
                <div style="display: flex; flex-direction: column; align-items: center;">
                    <div style="background: linear-gradient(135deg, #74b9ff, #0984e3); color: white; padding: 11px 14px; border-radius: 8px; box-shadow: 0 3px 8px rgba(116, 185, 255, 0.4); margin: 6px; min-width: 168px; text-align: center;">
                        <div style="font-size: 1.1em; font-weight: bold;">.xrdml</div>
                        <div style="font-size: 0.8em; opacity: 0.9;">PANalytical XML</div>
                    </div>
                    <div style="background: linear-gradient(135deg, #74b9ff, #0984e3); color: white; padding: 11px 14px; border-radius: 8px; box-shadow: 0 3px 8px rgba(116, 185, 255, 0.4); margin: 6px; min-width: 168px; text-align: center;">
                        <div style="font-size: 1.1em; font-weight: bold;">.ras / .rasx</div>
                        <div style="font-size: 0.8em; opacity: 0.9;">Rigaku ASCII / SmartLab</div>
                    </div>
                    <div style="background: linear-gradient(135deg, #74b9ff, #0984e3); color: white; padding: 11px 14px; border-radius: 8px; box-shadow: 0 3px 8px rgba(116, 185, 255, 0.4); margin: 6px; min-width: 168px; text-align: center;">
                        <div style="font-size: 1.1em; font-weight: bold;">.raw</div>
                        <div style="font-size: 0.8em; opacity: 0.9;">Bruker Binary</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_ad:
        display_ecm36_ad()


# GitHub mark (Octicons "mark-github", MIT license), same as in XRDlicious.
# Inherits the text color and size of the surrounding line.
GITHUB_ICON = (
    '<svg viewBox="0 0 16 16" width="1em" height="1em" fill="currentColor" '
    'aria-hidden="true" style="vertical-align:-0.125em;margin-right:0.25em;">'
    '<path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17'
    '-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65'
    '-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64'
    '-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1'
    '-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2'
    '-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27'
    '.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21'
    '-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"/></svg>'
)


# (name, website, logo file, logo height in px)
SUPPORTERS = [
    ("FZU – Institute of Physics of the Czech Academy of Sciences",
     "https://www.fzu.cz/en", "images/fzu_logo.svg", 34),
    ("IMPLANT group", "https://implant.fs.cvut.cz/", "images/implant_logo.png", 44),
]


# The XRDlicious article in J. Appl. Cryst.; shown below the supporter logos
# as the IUCr mark plus the journal name, linking to the article.
ARTICLE_URL = "https://doi.org/10.1107/S1600576725005370"
ARTICLE_TITLE = ("Lebeda et al. (2025). XRDlicious: an interactive web-based platform "
                 "for online calculation of diffraction patterns and radial distribution "
                 "functions from crystal structures. J. Appl. Cryst. 58")


def _journal_html():
    mark = _logo_data_uri("images/iucr_journals_logo_spaces.webp")
    mark_html = (f'<img src="{mark}" alt="IUCr" style="height:30px;width:30px;flex:none;">'
                 if mark else "")
    # without the mark, a coloured bar on the left takes its place
    bar = "" if mark else "border-left:4px solid #1f4e8c;padding-left:8px;"
    badge = (
        '<span style="display:inline-flex;align-items:center;gap:8px;">'
        + mark_html +
        f'<span style="display:inline-block;{bar}padding-top:2px;padding-bottom:2px;'
        'line-height:1.05;font-family:Georgia,\'Times New Roman\',serif;color:#1f4e8c;">'
        '<span style="font-size:0.78em;font-style:italic;">Journal of</span><br>'
        '<span style="font-size:0.95em;font-weight:700;white-space:nowrap;">'
        'Applied Crystallography</span></span></span>')
    return (
        '<div style="font-size:0.85em;opacity:0.8;margin:14px 0 6px 0;">Published in</div>'
        f'<a href="{ARTICLE_URL}" target="_blank" title="{ARTICLE_TITLE}" '
        'style="text-decoration:none;">' + badge + '</a>')


def display_supporters():
    """Supporter logos at the bottom of the sidebar, same block as in XRDlicious."""
    logos = []
    for name, url, relative_path, height in SUPPORTERS:
        logo = _logo_data_uri(relative_path)
        if not logo:
            continue
        logos.append(
            f'<a href="{url}" target="_blank" title="{name}">'
            f'<img src="{logo}" alt="{name}" style="height:{height}px;width:auto;"></a>')
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div style="font-size:0.85em;opacity:0.8;margin:6px 0;">'
        '<b>FZU</b> and the <b>IMPLANT group</b></div>'
        '<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">'
        + "".join(logos) + '</div>' + _journal_html(),
        unsafe_allow_html=True,
    )


# GoatCounter endpoint for the outbound clicks on the ECM-36 banner. Not a
# secret — it is visible in the page source of every site that uses GoatCounter.
GOATCOUNTER_ENDPOINT = "https://brace.goatcounter.com/count"


def display_ecm36_ad():
    """Announcement banner for ECM-36, the 36th European Crystallographic Meeting.

    Rendered in a component iframe rather than with ``st.markdown`` because
    Streamlit strips ``<script>`` from markdown, and the GoatCounter snippet
    that counts the clicks on the two outbound links has to run on the page.
    """
    logo = _logo_data_uri("images/ecm36_logo.png")
    logo_html = (
        f'<img src="{logo}" alt="ECM-36 Prague 2027 logo" '
        f'style="width: 100%; max-width: 190px; height: auto;">'
        if logo else
        '<div style="font-size: 2.2em; font-weight: 800; color: #d63031;">ECM<br>XXXVI</div>'
    )

    important_dates = [
        ("Nov 2026", "Call for abstracts"),
        ("31 Mar 2027", "Bursary applications"),
        ("15 Apr 2027", "Abstracts &mdash; oral contribution"),
        ("31 May 2027", "Early-bird registration"),
        ("15 Jun 2027", "Abstracts &mdash; posters"),
        ("30 Jun 2027", "Standard registration"),
    ]
    dates_html = "".join(
        f'<div style="background: #fff5f5; border: 1px solid #ffd0d0; border-radius: 9px; '
        f'padding: 7px 12px; font-size: 0.86em; color: #2d3436;">'
        f'<b>{when}</b><br><span style="opacity: 0.75;">{label}</span></div>'
        for when, label in important_dates
    )

    # The iframe carries none of the app's styling, so the font and the reset
    # are repeated here; the body padding keeps the card's shadow from being
    # clipped at the frame edge.
    card_html = f"""
    <style>
        html, body {{ margin: 0; padding: 0; background: transparent;
                      font-family: "Source Sans Pro", "Segoe UI", system-ui, -apple-system, sans-serif; }}
        body {{ padding: 8px; }}
        a {{ text-decoration: none; }}
    </style>
    <div id="ecm36-card"
         style="padding: 22px 26px; background: linear-gradient(135deg, #ffffff 0%, #eef3ff 100%);
                border: 1px solid #d7e0f5; border-radius: 16px;
                box-shadow: 0 6px 20px rgba(0,0,0,0.10);">
        <div style="display: flex; align-items: center; gap: 24px; flex-wrap: wrap;">
            <div style="flex: 0 0 190px; text-align: center; min-width: 140px;">
                {logo_html}
            </div>
            <div style="flex: 1; min-width: 260px;">
                <div style="font-size: 1.45em; font-weight: 800; color: #1e3a8a; line-height: 1.25; margin: 0 0 2px 0;">
                    ECM-36 &mdash; 36<sup>th</sup> European Crystallographic Meeting
                </div>
                <div style="font-size: 1.05em; color: #2d3436; margin-bottom: 10px;">
                    📅 <b>23&ndash;28 August 2027</b> &nbsp;•&nbsp; 📍 <b>Prague Congress Centre, Prague, Czech Republic</b>
                </div>
                <div style="font-size: 0.95em; color: #34495e; line-height: 1.5;">
                    Organised by the <b>European Crystallographic Association</b> together with the Czech Crystallographic
                    Society, Charles University, the Czech Technical University and the Czech Academy of Sciences.
                    Structural science from biology, pharmacy and chemistry to physics and materials science &mdash;
                    X-ray, neutron and electron methods.
                </div>
            </div>
        </div>
        <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px;">
            {dates_html}
        </div>
        <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px; align-items: center;">
            <a href="http://www.xray.cz/ecm36/" target="_blank" rel="noopener"
               data-goatcounter-click="ecm36-official-website"
               data-goatcounter-title="ECM-36 official website"
               style="background: linear-gradient(135deg, #74b9ff, #0984e3); color: white; text-decoration: none;
                      padding: 9px 18px; border-radius: 9px; font-weight: 700; font-size: 0.95em;
                      box-shadow: 0 4px 12px rgba(116, 185, 255, 0.4);">
                🌐 Official website
            </a>
            <a href="https://ecanews.org/meetings/" target="_blank" rel="noopener"
               data-goatcounter-click="ecm36-all-eca-meetings"
               data-goatcounter-title="All ECA meetings"
               style="color: #1e3a8a; text-decoration: none; padding: 9px 12px; border-radius: 9px;
                      font-weight: 600; font-size: 0.92em; border: 1px solid #c7d4f0; background: #ffffff;">
                📋 All ECA meetings
            </a>
        </div>
        <div style="margin-top: 10px; font-size: 0.78em; color: #7f8c8d;">
            Dates and deadlines as announced by the organisers &mdash; please check the official website for updates.
        </div>
    </div>

    <script>
        // allow_frame: count.js refuses to send anything from inside a frame
        // otherwise, and every Streamlit component lives in one. no_onload
        // skips the automatic pageview for the frame itself (which would be
        // logged under the meaningless "about:srcdoc" path); the click
        // handlers are bound by hand once the script has loaded instead.
        window.goatcounter = {{allow_frame: true, no_onload: true}};
    </script>
    <script data-goatcounter="{GOATCOUNTER_ENDPOINT}"
            src="https://gc.zgo.at/count.js"
            onload="window.goatcounter.bind_events()"></script>

    <script>
        // Streamlit gives the iframe the fixed height passed below, which
        // cannot know how the card wraps at the visitor's window width, so the
        // frame measures its own content and resizes itself.
        (function () {{
            var fit = function () {{
                try {{
                    var frame = window.frameElement;
                    if (!frame) return;
                    var height = document.getElementById('ecm36-card').offsetHeight + 16;
                    frame.style.height = height + 'px';
                    frame.height = height;
                }} catch (e) {{ /* keep the fallback height */ }}
            }};
            fit();
            window.addEventListener('load', fit);
            window.addEventListener('resize', fit);
            if (window.ResizeObserver)
                new ResizeObserver(fit).observe(document.getElementById('ecm36-card'));
        }})();
    </script>
    """
    # st.iframe replaces components.html (deprecated, slated for removal) from
    # newer Streamlit on; the fallback keeps older versions working.
    if hasattr(st, "iframe"):
        st.iframe(card_html, alt="ECM-36 conference announcement")
    else:
        components.html(card_html, height=520, scrolling=False)


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="XRD File Converter",
                       initial_sidebar_state="expanded")
    
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    [data-testid="stToolbarActions"] {display: none;}
    .viewerBadge_link__qRIco {display: none;}
    [data-testid="stStatusWidget"] {display: none;}
    [data-testid="stSidebarCollapsedControl"] {visibility: visible !important; display: block !important;}
    [data-testid="collapsedControl"] {visibility: visible !important; display: block !important;}

    /* The default open/close control is a thin chevron that is easy to miss
       once the sidebar is hidden, so both buttons become a solid grey button
       with a white icon. stExpandSidebarButton is the one shown over the page
       while the sidebar is collapsed, stSidebarCollapseButton the one inside
       the sidebar header. */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapseButton"] button {
        background-color: #6b7280 !important;
        color: #ffffff !important;
        border: 1px solid #ffffff !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35) !important;
        opacity: 1 !important;
        transition: background-color 0.15s ease, transform 0.15s ease;
    }
    [data-testid="stExpandSidebarButton"]:hover,
    [data-testid="stSidebarCollapseButton"] button:hover {
        background-color: #4b5563 !important;
        transform: scale(1.06);
    }
    /* Material icon fonts and inline SVGs both follow the button colour. */
    [data-testid="stExpandSidebarButton"] *,
    [data-testid="stSidebarCollapseButton"] button * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    [data-testid="stExpandSidebarButton"] {
        min-height: 34px !important;
        min-width: 34px !important;
        padding: 0 6px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    css = '''
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.15rem !important;
        color: #1e3a8a !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 20px !important;
    }

    .stTabs [data-baseweb="tab-list"] button {
        background-color: #f0f4ff !important;
        border-radius: 12px !important;
        padding: 8px 16px !important;
        transition: all 0.3s ease !important;
        border: none !important;
        color: #1e3a8a !important;
    }

    .stTabs [data-baseweb="tab-list"] button:hover {
        background-color: #dbe5ff !important;
        cursor: pointer;
    }

    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #e0e7ff !important;
        color: #1e3a8a !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 6px rgba(30, 58, 138, 0.3) !important;
    }

    .stTabs [data-baseweb="tab-list"] button:focus {
        outline: none !important;
    }
    </style>
    '''

    st.markdown(css, unsafe_allow_html=True)

    st.sidebar.title("XRD Converter Tools")
    st.sidebar.caption("**v0.9.0** — 2026-11-15")
    # Same look as st.info, written in HTML so the GitHub icon can be inlined.
    st.sidebar.markdown(
        '<div style="background-color:rgba(28,131,225,0.1);color:rgb(0,66,128);'
        'border-radius:8px;padding:16px;line-height:1.6;margin-bottom:1rem;">'
        'Visit also main app here: <a href="https://xrdlicious.com" target="_blank"><b>XRDlicious</b></a>. '
        '🌀 Developed by <a href="https://implant.fs.cvut.cz/" target="_blank"><b>IMPLANT team</b></a>. '
        '<a href="https://youtu.be/KwxVKadPZ6s?si=S1_67xF5J3sI7n69" target="_blank"><b>Tutorial here</b></a>. '
        'Spot a bug or have a feature idea? Let us know at: <b>lebedmi2@cvut.cz</b>. '
        'To compile the app locally, visit our '
        '<a href="https://github.com/bracerino/xrd-file-converter" target="_blank" style="white-space:nowrap;">'
        f'{GITHUB_ICON}<b>GitHub page</b></a>. '
        'If you like the app, please cite 📖 '
        '<a href="https://journals.iucr.org/j/issues/2025/05/00/hat5006/index.html" target="_blank">'
        '<b>article in IUCr</b></a>.</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.title("Select between format and X/Y-Axis Conversion")
    tool_choice = st.sidebar.radio(
        "**Select Tool:**",
        ["📄 File Format Converter", "🔄 X/Y-Axis Converter", "✂️ Reduction / Averaging", "📈 Plotting",
         "🌐 Chi-Scan Viewer", "🧬 Multiple .xy merge", "🖼️ Image/Photo → .xy"],
        index=0
    )

    if tool_choice == "📄 File Format Converter":
        run_data_converter()
        display_conversion_visual()
    elif tool_choice == "✂️ Reduction / Averaging":
        run_data_reduction_section()
    elif tool_choice == "📈 Plotting":
        run_plotting_section()
    elif tool_choice == "🌐 Chi-Scan Viewer":

        run_chi_scan_section()
    elif tool_choice == "🧬 Multiple .xy merge":
        run_chi_merge_section()
    elif tool_choice == "🖼️ Image/Photo → .xy":
        run_image_digitizer_section()
    else:
        run_axis_converter()

    def record_and_get_pageviews():
        """Count one page view per user session and return daily view statistics.

        Views are stored in a small JSON file keyed by date. Each browser session is
        counted only once (tracked via st.session_state). Note: on Streamlit Community
        Cloud the filesystem is ephemeral, so counts reset when the app reboots.
        """
        import os
        import json
        from datetime import date

        counts_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pageviews.json")
        today = date.today().isoformat()

        try:
            with open(counts_file, "r") as f:
                counts = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            counts = {}

        if not st.session_state.get("_pageview_counted", False):
            counts[today] = counts.get(today, 0) + 1
            try:
                with open(counts_file, "w") as f:
                    json.dump(counts, f)
                st.session_state["_pageview_counted"] = True
            except OSError:
                pass

        today_views = counts.get(today, 0)
        # Show up to the seven most recent finished days (dates before today)
        # that have recorded views. If there are none yet, nothing extra is
        # shown.
        finished = sorted(d for d in counts if d < today)[-7:]
        previous_days = [
            (f"{date.fromisoformat(d).day}.{date.fromisoformat(d).month}", counts[d])
            for d in finished
        ]
        return today_views, previous_days

    try:
        today_views, previous_days = record_and_get_pageviews()
        st.sidebar.markdown("---")
        caption = f"📈 Page views today: **{today_views}**"
        if previous_days:
            caption += " (" + ", ".join(
                f"{day} - **{views}**" for day, views in previous_days) + ")"
        st.sidebar.caption(caption + ".")
    except Exception:
        pass

    display_supporters()

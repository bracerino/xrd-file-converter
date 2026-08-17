"""Readers for the XRD file formats supported by the app.

These parsers were originally nested inside ``run_data_converter`` in
``app.py``; they live here so that both the converter and the plotting tool
can share exactly the same file-reading behaviour.
"""

import re
import struct
import zipfile
import xml.etree.ElementTree as ET
from io import StringIO, BytesIO

import numpy as np
import pandas as pd
import streamlit as st


def metadata_dataframe(metadata, columns=("Parameter", "Value")):
    """Two-column table of a metadata dict, with every value rendered as text.

    Metadata dicts mix strings with numbers ('Number of Points' is an int, for
    instance), so the value column ends up as an object column that Arrow
    cannot convert. Streamlit then dumps a full pyarrow traceback into the
    console for every such table before falling back to its own conversion.
    Text everywhere keeps the console clean and looks the same on screen.
    """
    rows = [(str(key), _metadata_text(value))
            for key, value in metadata.items()]
    return pd.DataFrame(rows, columns=list(columns))


def _metadata_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value)


def extract_key_ras_metadata(metadata_dict):
    key_metadata = {
        'X-ray Target': metadata_dict.get('HW_XG_TARGET_NAME', 'N/A'),
        'Voltage (kV)': metadata_dict.get('MEAS_COND_XG_VOLTAGE', 'N/A'),
        'Current (mA)': metadata_dict.get('MEAS_COND_XG_CURRENT', 'N/A'),
        'K-Alpha1 (Å)': metadata_dict.get('HW_XG_WAVE_LENGTH_ALPHA1', 'N/A'),
        'Scan Axis': metadata_dict.get('MEAS_SCAN_AXIS_X', 'N/A'),
        'Start Angle (°)': metadata_dict.get('MEAS_SCAN_START', 'N/A'),
        'Stop Angle (°)': metadata_dict.get('MEAS_SCAN_STOP', 'N/A'),
        'Step Size (°)': metadata_dict.get('MEAS_SCAN_STEP', 'N/A'),
        'Scan Speed': f"{metadata_dict.get('MEAS_SCAN_SPEED', 'N/A')} {metadata_dict.get('MEAS_SCAN_SPEED_UNIT', '')}".strip(),
        'Divergence Slit (DS)': metadata_dict.get('MEAS_COND_AXIS_POSITION-18', 'N/A'),
        'Scattering Slit (SS)': metadata_dict.get('MEAS_COND_AXIS_POSITION-35', 'N/A'),
        'Receiving Slit (RS)': metadata_dict.get('MEAS_COND_AXIS_POSITION-42', 'N/A')
    }
    return key_metadata


def parse_xrdml(file_content):
    try:
        root = ET.fromstring(file_content)
        namespace = ''
        if '}' in root.tag:
            namespace = root.tag.split('}')[0][1:]
        ns = {'xrd': namespace}

        def find_text(path, default='N/A'):
            element = root.find(path, ns)
            return element.text if element is not None else default

        def find_attrib(path, attribute, default='N/A'):
            element = root.find(path, ns)
            return element.attrib.get(attribute, default) if element is not None else default

        metadata = {
            'Status': find_attrib('xrd:xrdMeasurement', 'status'),
            'Measurement Type': find_attrib('xrd:xrdMeasurement', 'measurementType'),
            'Start Time': find_text('xrd:xrdMeasurement/xrd:scan/xrd:header/xrd:startTimeStamp'),
            'End Time': find_text('xrd:xrdMeasurement/xrd:scan/xrd:header/xrd:endTimeStamp'),
            'Author': find_text('xrd:xrdMeasurement/xrd:scan/xrd:header/xrd:author/xrd:name'),
            'Anode Material': find_text('xrd:xrdMeasurement/xrd:incidentBeamPath/xrd:xRayTube/xrd:anodeMaterial'),
            'X-ray Tube Tension': f"{find_text('xrd:xrdMeasurement/xrd:incidentBeamPath/xrd:xRayTube/xrd:tension')} {find_attrib('xrd:xrdMeasurement/xrd:incidentBeamPath/xrd:xRayTube/xrd:tension', 'unit')}",
            'X-ray Tube Current': f"{find_text('xrd:xrdMeasurement/xrd:incidentBeamPath/xrd:xRayTube/xrd:current')} {find_attrib('xrd:xrdMeasurement/xrd:incidentBeamPath/xrd:xRayTube/xrd:current', 'unit')}",
            'K-Alpha1 Wavelength (Å)': find_text('xrd:xrdMeasurement/xrd:usedWavelength/xrd:kAlpha1'),
            'Detector': find_attrib('xrd:xrdMeasurement/xrd:diffractedBeamPath/xrd:detector', 'name'),
            'Scan Axis': find_attrib('xrd:xrdMeasurement/xrd:scan', 'scanAxis'),
        }
        data_points_path = 'xrd:xrdMeasurement/xrd:scan/xrd:dataPoints'
        start_pos_2theta = float(find_text(f'{data_points_path}/xrd:positions[@axis="2Theta"]/xrd:startPosition'))
        end_pos_2theta = float(find_text(f'{data_points_path}/xrd:positions[@axis="2Theta"]/xrd:endPosition'))
        intensities_str = find_text(f'{data_points_path}/xrd:intensities')
        intensities = np.array(intensities_str.split(), dtype=float)
        two_theta_array = np.linspace(start_pos_2theta, end_pos_2theta, len(intensities))
        data_df = pd.DataFrame({'2Theta': two_theta_array, 'Intensity': intensities})
        return metadata, data_df
    except Exception as e:
        st.error(f"Failed to parse XRDML file. Error: {e}")
        return None, None


def parse_ras(file_content):
    try:
        full_metadata = {}
        data_lines = []
        in_header_section = False
        in_data_section = False

        for line in file_content.splitlines():
            line = line.strip()
            if not line:
                continue
            if line == '*RAS_HEADER_START':
                in_header_section = True
                continue
            if line == '*RAS_HEADER_END':
                in_header_section = False
                continue
            if line == '*RAS_INT_START':
                in_data_section = True
                continue
            if line == '*RAS_INT_END':
                in_data_section = False
                break

            if in_header_section and line.startswith('*'):
                parts = line[1:].split(None, 1)
                if len(parts) == 2:
                    key, value = parts
                    full_metadata[key] = value.strip('"')

            if in_data_section:
                data_parts = line.split()
                if len(data_parts) >= 2:
                    try:
                        angle = float(data_parts[0])
                        intensity = float(data_parts[1])
                        data_lines.append([angle, intensity])
                    except ValueError:
                        continue

        if not data_lines:
            st.error("No data points found in the RAS file.")
            return None, None, None

        data_df = pd.DataFrame(data_lines, columns=['2Theta', 'Intensity'])
        key_metadata = extract_key_ras_metadata(full_metadata)
        return full_metadata, key_metadata, data_df

    except Exception as e:
        st.error(f"Failed to parse RAS file. Error: {e}")
        return None, None, None


def _decode_rasx_text(raw_bytes):
    """RASX parts are UTF-8 with a BOM; older exports can be Shift-JIS."""
    for encoding in ('utf-8-sig', 'shift_jis'):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode('utf-8', errors='replace')


def _rasx_metadata_from_conditions(root):
    """Flatten a MesurementConditions XML tree into flat *.ras-style keys.

    The XML already carries a <RASHeader> block holding the very same
    `*KEY "value"` pairs that the flat .ras format uses, but Rigaku leaves
    the most interesting ones out of it and stores them structurally
    instead (generator, scan range, goniometer axes). So we take the
    RASHeader pairs first and then fill the gaps from the structured
    sections, which gives us a header that the .ras code paths understand.
    """
    metadata = {}

    ras_header = root.find('RASHeader')
    if ras_header is not None:
        for pair in ras_header.findall('Pair'):
            strings = pair.findall('string')
            if len(strings) >= 2:
                key = (strings[0].text or '').lstrip('*').strip()
                if key:
                    metadata[key] = (strings[1].text or '').strip()

    def section_text(section_name, tag, default=''):
        section = root.find(section_name)
        if section is None:
            return default
        element = section.find(tag)
        return (element.text or default).strip() if element is not None else default

    def put(key, value):
        # RASHeader stays authoritative wherever it actually has the key.
        if value not in ('', None) and not metadata.get(key):
            metadata[key] = value

    for tag, key in (('Operator', 'FILE_OPERATOR'), ('UserGroup', 'FILE_USERGROUP'),
                     ('Comment', 'FILE_COMMENT'), ('SampleName', 'FILE_SAMPLE'),
                     ('Memo', 'FILE_MEMO'), ('Type', 'FILE_TYPE'),
                     ('Version', 'FILE_VERSION'), ('SystemName', 'FILE_SYSTEM_NAME'),
                     ('PackageName', 'FILE_PACKAGE_NAME'), ('PartName', 'FILE_PART_ID')):
        put(key, section_text('GeneralInformation', tag))

    generator = root.find('HWConfigurations/XrayGenerator')
    if generator is not None:
        def gen(tag, default=''):
            element = generator.find(tag)
            return (element.text or default).strip() if element is not None else default

        put('HW_XG_TYPE', gen('Type'))
        put('HW_XG_TARGET_NAME', gen('TargetName'))
        put('HW_XG_TARGET_ATOMIC_NUMBER', gen('TargetAtomicNumber'))
        put('HW_XG_FOCUS_SIZE', gen('FocusSize'))
        put('HW_XG_FOCUS_TYPE', gen('FocusType'))
        put('HW_XG_WAVE_TYPE', gen('WaveType'))
        put('HW_XG_WAVE_LENGTH_ALPHA1', gen('WavelengthKalpha1'))
        put('HW_XG_WAVE_LENGTH_ALPHA2', gen('WavelengthKalpha2'))
        put('HW_XG_WAVE_LENGTH_BETA', gen('WavelengthKbeta'))
        put('MEAS_COND_XG_VOLTAGE', gen('Voltage'))
        put('MEAS_COND_XG_VOLTAGE_UNIT', gen('VoltageUnit'))
        put('MEAS_COND_XG_CURRENT', gen('Current'))
        put('MEAS_COND_XG_CURRENT_UNIT', gen('CurrentUnit'))

    put('HW_COUNTER_PIXEL_SIZE', section_text('HWConfigurations/Detector', 'PixelSize'))

    for tag, key in (('AxisName', 'MEAS_SCAN_AXIS_X'), ('Mode', 'MEAS_SCAN_MODE'),
                     ('Start', 'MEAS_SCAN_START'), ('Stop', 'MEAS_SCAN_STOP'),
                     ('Step', 'MEAS_SCAN_STEP'), ('Speed', 'MEAS_SCAN_SPEED'),
                     ('SpeedUnit', 'MEAS_SCAN_SPEED_UNIT'),
                     ('Resolution', 'MEAS_SCAN_RESOLUTION_X'),
                     ('PositionUnit', 'MEAS_SCAN_UNIT_X'),
                     ('IntensityUnit', 'MEAS_SCAN_UNIT_Y'),
                     ('DataCount', 'MEAS_DATA_COUNT')):
        put(key, section_text('ScanInformation', tag))

    for tag, key in (('StartTime', 'MEAS_SCAN_START_TIME'), ('EndTime', 'MEAS_SCAN_END_TIME')):
        # '2022-06-07T13:12:13Z' -> a value datetime.fromisoformat() accepts
        put(key, section_text('ScanInformation', tag).rstrip('Z'))

    # The Nth <Axis> element lines up with the *MEAS_COND_AXIS_*-N index
    # used by .ras headers, which is how the key parameters table finds the
    # DS / SS / RS slits.
    axes = root.find('Axes')
    if axes is not None:
        for index, axis in enumerate(axes.findall('Axis')):
            for attribute, prefix in (('Position', 'MEAS_COND_AXIS_POSITION'),
                                      ('Offset', 'MEAS_COND_AXIS_OFFSET'),
                                      ('Unit', 'MEAS_COND_AXIS_UNIT'),
                                      ('State', 'MEAS_COND_AXIS_STATE'),
                                      ('Resolution', 'MEAS_COND_AXIS_RESOLUTION')):
                put(f'{prefix}-{index}', axis.attrib.get(attribute, ''))
            if not metadata.get(f'MEAS_COND_AXIS_NAME-{index}'):
                put(f'MEAS_COND_AXIS_NAME-{index}', axis.attrib.get('Name', ''))

    return metadata


def _parse_rasx_profile(profile_text):
    """Profile*.txt holds tab-separated angle / intensity / attenuator rows."""
    rows = []
    for line in profile_text.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('*'):
            continue
        parts = line.replace(',', ' ').split()
        if len(parts) < 2:
            continue
        try:
            angle = float(parts[0])
            intensity = float(parts[1])
        except ValueError:
            continue
        # Third column is the attenuator factor the intensity was measured
        # through; the true count rate is the product of the two.
        if len(parts) >= 3:
            try:
                intensity *= float(parts[2])
            except ValueError:
                pass
        rows.append([angle, intensity])

    if not rows:
        return None
    return pd.DataFrame(rows, columns=['2Theta', 'Intensity'])


def parse_rasx(uploaded_file_object):
    """Parse a Rigaku SmartLab .rasx file.

    A .rasx is a ZIP archive: root.xml lists one Data<N> group per scan and
    each group holds a Profile<N>.txt with the measured points plus a
    MesurementConditions<N>.xml with the instrument settings.
    """
    try:
        uploaded_file_object.seek(0)
        with zipfile.ZipFile(uploaded_file_object, 'r') as zf:
            names = zf.namelist()

            def pick(predicate):
                return next(iter(sorted(f for f in names if predicate(f.lower()))), None)

            conditions_name = (
                pick(lambda n: 'conditions' in n and n.endswith('.xml'))
                or pick(lambda n: n.endswith('.xml') and not n.endswith('root.xml'))
            )
            full_metadata = {}
            if conditions_name:
                root = ET.fromstring(_decode_rasx_text(zf.read(conditions_name)))
                full_metadata = _rasx_metadata_from_conditions(root)

            profile_name = (
                pick(lambda n: 'profile' in n and n.endswith('.txt'))
                or pick(lambda n: n.endswith('.asc'))
                or pick(lambda n: n.endswith('.txt'))
            )
            if not profile_name:
                st.error("No profile data (Profile*.txt) found in the RASX archive.")
                return None, None, None

            data_df = _parse_rasx_profile(_decode_rasx_text(zf.read(profile_name)))

        if data_df is None or data_df.empty:
            st.error("No data points found in the RASX file.")
            return None, None, None

        key_metadata = extract_key_ras_metadata(full_metadata)
        return full_metadata, key_metadata, data_df
    except Exception as e:
        st.error(f"Failed to parse RASX file. Error: {e}")
        return None, None, None


def parse_raw_v1(file_content_bytes):
    """Parse the Bruker RAW1.01 (RAW version 3 / DIFFRACplus) binary format.

    Layout (little-endian):
      * File header (712 bytes):
          @8   file status (int32)
          @12  number of ranges (int32)
          @608 anode element symbol (2-char string)
          @624 K-Alpha1 wavelength (double)
          @632 K-Alpha2 wavelength (double)
      * Each range begins with a range header:
          cur+0   header length (int32, normally 304)
          cur+4   number of steps (int32)
          cur+8   start theta / omega (double)
          cur+16  start 2-theta (double)
          cur+176 step size / increment (double)
        The intensity data (float32 per step) follows the range header.

    The previous implementation guessed the angles by scanning for plausible
    floats and assumed an omega rocking-curve, which produced wrong 2-theta
    values for ordinary powder patterns. We now read the documented fields.
    """
    metadata = {}
    data = file_content_bytes
    file_size = len(data)
    file_header_size = 712

    try:
        range_cnt = struct.unpack_from('<i', data, 12)[0]
        if range_cnt < 1 or range_cnt > 1000:
            range_cnt = 1

        # X-ray anode / target.
        try:
            anode = struct.unpack_from('2s', data, 608)[0].decode(
                'ascii', errors='ignore').strip('\x00').strip()
            if anode:
                metadata['X-ray Target'] = anode
        except (struct.error, IndexError):
            pass

        # K-Alpha1 wavelength.
        try:
            wl = struct.unpack_from('<d', data, 624)[0]
            if 0.3 < wl < 3.0:
                metadata['K-Alpha1 (Å)'] = f"{wl:.5f}"
        except struct.error:
            pass

        cur = file_header_size
        header_len = struct.unpack_from('<i', data, cur)[0]
        if header_len < 304 or cur + header_len > file_size:
            header_len = 304

        num_points = struct.unpack_from('<i', data, cur + 4)[0]
        start_theta = struct.unpack_from('<d', data, cur + 8)[0]
        start_2theta = struct.unpack_from('<d', data, cur + 16)[0]
        step_size = struct.unpack_from('<d', data, cur + 176)[0]

        data_offset = cur + header_len
        # Validate / recover the point count and data offset.
        available = (file_size - data_offset) // 4
        if num_points <= 0 or num_points > available:
            num_points = available
        if num_points <= 0:
            st.error("Invalid number of data points for RAW 1.01 format.")
            return None, None
        # If a single range doesn't reach the end of file, the intensity
        # block is the trailing num_points float32 values.
        if range_cnt == 1 and (file_size - data_offset) != num_points * 4:
            data_offset = file_size - num_points * 4

        if not (np.isfinite(step_size) and 0 < abs(step_size) < 50):
            st.warning("Could not read a valid step size; deriving it is not possible.")
            step_size = 0.0
        if not np.isfinite(start_2theta):
            start_2theta = 0.0

        intensities = np.frombuffer(
            data, dtype=np.float32, count=num_points, offset=data_offset).astype(float)
        angles = np.arange(num_points) * step_size + start_2theta
        end_2theta = angles[-1] if num_points > 0 else start_2theta

        metadata['Number of Points'] = num_points
        metadata['Start Angle (2θ, °)'] = f"{start_2theta:.4f}"
        metadata['End Angle (2θ, °)'] = f"{end_2theta:.4f}"
        metadata['Step Size (°)'] = f"{step_size:.5f}"
        if np.isfinite(start_theta):
            metadata['Start Angle (θ, °)'] = f"{start_theta:.4f}"
        if range_cnt > 1:
            metadata['Note'] = f"File contains {range_cnt} ranges; only the first is loaded."

        data_df = pd.DataFrame({'2Theta': angles, 'Intensity': intensities})
        return metadata, data_df

    except Exception as e:
        st.error(f"Failed to parse RAW 1.01 file. Error: {e}")
        return None, None


def parse_raw_v4(file_content_bytes):
    metadata = {}
    file_size = len(file_content_bytes)
    data_offset = 2600

    try:
        try:
            metadata['Start Angle (°)'] = f"{struct.unpack_from('<f', file_content_bytes, offset=136)[0]:.4f}"
        except (struct.error, IndexError):
            metadata['Start Angle (°)'] = 'N/A'
        try:
            metadata['Step Size (°)'] = f"{struct.unpack_from('<f', file_content_bytes, offset=140)[0]:.4f}"
        except (struct.error, IndexError):
            metadata['Step Size (°)'] = 'N/A'
        try:
            metadata['Time per Step (s)'] = f"{struct.unpack_from('<f', file_content_bytes, offset=152)[0]:.2f}"
        except (struct.error, IndexError):
            metadata['Time per Step (s)'] = 'N/A'
        try:
            metadata['K-Alpha1 (Å)'] = f"{struct.unpack_from('<f', file_content_bytes, offset=308)[0]:.5f}"
        except (struct.error, IndexError):
            metadata['K-Alpha1 (Å)'] = 'N/A'
        try:
            target_name_bytes = struct.unpack_from('12s', file_content_bytes, offset=244)[0];
            metadata[
                'X-ray Target'] = target_name_bytes.decode('utf-8', errors='ignore').strip('\x00').strip()
        except (struct.error, IndexError):
            metadata['X-ray Target'] = 'N/A'

        num_points = 0
        try:
            num_points = struct.unpack_from('<i', file_content_bytes, offset=148)[0]
        except (struct.error, IndexError):
            pass

        if not num_points or num_points <= 0:
            st.warning("Could not read points from v4 header. Calculating from file size.")
            data_size = file_size - data_offset
            if data_size < 0:
                st.error("File is smaller than the expected v4 header size.")
                return None, None
            num_points = data_size // 4

        metadata['Number of Points'] = num_points
        if num_points <= 0: return metadata, None

        if metadata['Start Angle (°)'] == 'N/A' or metadata['Step Size (°)'] == 'N/A': return metadata, None

        intensities = np.frombuffer(file_content_bytes, dtype=np.float32, count=num_points, offset=data_offset)
        angles = np.arange(num_points) * float(metadata['Step Size (°)']) + float(metadata['Start Angle (°)'])
        data_df = pd.DataFrame({'2Theta': angles, 'Intensity': intensities})

        return metadata, data_df
    except Exception as e:
        st.error(f"A critical error occurred while parsing the RAW v4 file. Error: {e}")
        return None, None


def _raw4_find_anode(file_content_bytes, scan_off, kalpha):
    """Best-effort detection of the X-ray anode for a RAW4 file."""
    anodes = {'Cu': 1.54060, 'Co': 1.78900, 'Cr': 2.28970, 'Fe': 1.93600,
              'Mo': 0.70930, 'Ag': 0.55940, 'Ni': 1.65910, 'Mn': 2.10310,
              'W': 1.47640}
    header = file_content_bytes[:scan_off]
    # The anode element symbol is stored as a null-delimited token.
    for sym in anodes:
        if (b'\x00' + sym.encode() + b'\x00') in header:
            return sym
    # Fall back to inferring the anode from the measured wavelength.
    if kalpha:
        try:
            wl = float(kalpha)
            best = min(anodes, key=lambda s: abs(anodes[s] - wl))
            if abs(anodes[best] - wl) < 0.02:
                return best
        except ValueError:
            pass
    return None


def parse_raw4(file_content_bytes):
    """Parse the Bruker RAW4.00 binary format.

    Unlike the older RAW formats, RAW4 begins with variable-length metadata
    records (USER, SAMPLEID, COMMENT, ...), so the scan parameters do not sit
    at a fixed offset. Instead we locate the scan-range block by its
    signature: a double start angle (2theta), a double step size and an int32
    number of points, immediately followed by an intensity block of that many
    float32 values that ends at the end of the file.
    """
    metadata = {}
    data = file_content_bytes
    fs = len(data)
    try:
        scan_off = None
        s_ang = st_size = npts = None
        for o in range(8, fs - 20):
            try:
                start = struct.unpack_from('<d', data, o)[0]
                step = struct.unpack_from('<d', data, o + 8)[0]
                n = struct.unpack_from('<i', data, o + 16)[0]
            except struct.error:
                continue
            if not (-180.0 <= start <= 180.0):
                continue
            if not (1e-6 < step < 5.0):
                continue
            if not (2 <= n <= 50_000_000):
                continue
            if n * 4 > fs - 20:
                continue
            # The intensity block (float32) runs to the end of the file.
            arr = np.frombuffer(data, dtype=np.float32, count=n, offset=fs - n * 4)
            if not np.all(np.isfinite(arr)):
                continue
            if arr.min() < 0 or arr.max() <= 0:
                continue
            scan_off, s_ang, st_size, npts = o, start, step, n
            break

        if scan_off is None:
            st.error("Could not locate the scan data block in the RAW4 file.")
            return None, None

        intensities = np.frombuffer(
            data, dtype=np.float32, count=npts, offset=fs - npts * 4).astype(float)
        angles = np.arange(npts) * st_size + s_ang

        metadata['Start Angle (°)'] = f"{s_ang:.4f}"
        metadata['Step Size (°)'] = f"{st_size:.5f}"
        metadata['Number of Points'] = npts

        # K-Alpha1 wavelength is stored as a double within the scan block.
        try:
            wl = struct.unpack_from('<d', data, scan_off + 40)[0]
            if 0.3 < wl < 3.0:
                metadata['K-Alpha1 (Å)'] = f"{wl:.5f}"
        except struct.error:
            pass

        target = _raw4_find_anode(data, scan_off, metadata.get('K-Alpha1 (Å)'))
        if target:
            metadata['X-ray Target'] = target

        data_df = pd.DataFrame({'2Theta': angles, 'Intensity': intensities})
        return metadata, data_df
    except Exception as e:
        st.error(f"A critical error occurred while parsing the RAW4 file. Error: {e}")
        return None, None


def parse_raw(file_content_bytes):

    if file_content_bytes.startswith(b'RAW1.01'):
        st.success("Assuming Bruker RAW 1.01 file format.")
        return parse_raw_v1(file_content_bytes)
    elif file_content_bytes.startswith(b'RAW4'):
        st.success("Assuming Bruker RAW4 file format.")
        return parse_raw4(file_content_bytes)
    else:
        st.success("Assuming older Bruker RAW (v2/v3) file format.")
        return parse_raw_v4(file_content_bytes)


def parse_xy(file_content):
    try:
        lines = file_content.splitlines()
        if not lines:
            st.error("The XY file is empty.")
            return None
        first_line = lines[0]
        has_header = any(char.isalpha() for char in first_line)
        data_io = StringIO(file_content)
        skiprows = 1 if has_header else 0
        df = pd.read_csv(data_io, sep=r'[\s,;]+', engine='python', header=None, skiprows=skiprows,
                         names=['2Theta', 'Intensity'], comment='#')
        return df.dropna().astype(float)
    except Exception as e:
        st.error(f"Failed to parse XY file. Error: {e}")
        return None

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:36:05 2025

@author: alexlupu
"""

import os                           # Standard library for OS-level operations
import sys                          # Standard library for system-specific operations
import json                         # Standard library for parsing/writing JSON
import numpy as np                  # NumPy for numerical arrays & operations
import matplotlib.pyplot as plt     # Matplotlib for data visualization
from pathlib import Path            # Object-oriented filesystem paths
import time                         # For time-related utilities (like timestamps)
from collections import Counter     # For counting elements/frequencies
from scipy.signal import find_peaks # Scipy function for peak detection

###############################################################################
# SETTINGS & HARD-CODED PATH
###############################################################################
NTT = 645       # Number of samples (time ticks) per channel
NC = 128        # Total number of channels
NCC = 48        # "Collection" plane channels (subset of total)
AMP_THRESHOLD = 18  # Amplitude threshold for "tall" peaks
WIDTH_THRESHOLD = 40  # Min spacing (samples) between peaks in a single channel
IGNORED_FOLDER_PREFIX = "."  # Files/folders starting with '.' will be skipped
output_align = "[ANALYSIS]"  # Prefix for log messages

DATA_FOLDER = "/Users/alexlupu/github/dune-apex-umn/DATA/20230719/jsonData/"  # Folder with .json data

plt.rcParams.update({'font.size': 20})  # Update default font size for plots

###############################################################################
# MINIMAL UTILS
###############################################################################
def parse_args(argv, defaults):
    """
    Parse command-line arguments (sys.argv) and override defaults if flags are found.
    """
    d = dict(defaults)
    for arg in argv[1:]:
        if arg == "--no-display":
            d["input_evtdisplay_bool"] = False
            d["input_chndisplay_bool"] = False
        elif arg == "--no-filter":
            d["input_filter_bool"] = False
        elif arg == "--equalize":
            d["input_equalize_bool"] = True
    return (
        d["raw_data_folder_name"],      # 1) Name of the data folder
        d["input_evtdisplay_bool"],     # 2) Whether to display events
        d["input_chndisplay_bool"],     # 3) Whether to display channels
        d["input_equalize_bool"],       # 4) Whether to do equalization
        d["input_filter_bool"],         # 5) Whether to do cosmic filtering
        True                            # 6) Terminal bool (unused, but for structure)
    )

def find_time_now():
    """
    Returns a formatted time string for logging.
    """
    return time.strftime("%Y-%m-%d %H:%M:%S")  # Format: "YYYY-MM-DD HH:MM:SS"

def most_frequent(arr):
    """
    Return the most common value in a list or array.
    """
    c = Counter(arr)
    return c.most_common(1)[0][0]

###############################################################################
# COSMIC DETECTION PARAMETERS
###############################################################################
def get_cosmic_rejection_parameters(year_str):
    """
    Return (peak_width, peak_height, min_strips). Hard-coded (5, 25, 20).
    """
    return (5, 25, 20)

def ch_eq_year(year_str):
    """
    Return an array of scale factors for channel equalization (all 1.0 here).
    """
    return np.ones(NC, dtype=float)

###############################################################################
# FIND_PEAKS_50L
###############################################################################
def find_peak_range(data_1d, pk_indices):
    """
    Given 1D data and a list of peak indices, find left/right bounds and integral.
    """
    pk_start, pk_top, pk_end, pk_int = [], [], [], []
    n_samp = len(data_1d)

    for pk in pk_indices:
        temp_int = data_1d[pk]
        left = pk
        while left > 0 and data_1d[left] > 0:
            left -= 1
            temp_int += data_1d[left]
        right = pk
        while right < n_samp - 1 and data_1d[right] >= 0:
            right += 1
            temp_int += data_1d[right]
        pk_start.append(left)
        pk_top.append(pk)
        pk_end.append(right)
        pk_int.append(round(temp_int / 2.0, 1))

    return (pk_start, pk_top, pk_end, pk_int)

def find_peaks_50l(data_1d, chn, peak_height, peak_width):
    """
    1) Detect all peaks using scipy.signal.find_peaks with given height/width.
    2) Keep only peaks that are at least 50 samples away from previously kept peaks.
    3) Return the filtered peaks, their heights, and their ranges.
    """
    pks, props = find_peaks(data_1d, height=peak_height, width=peak_width)

    # Filter out peaks that lie within 50 samples of a taller peak
    filtered_pks = []
    filtered_props = {"peak_heights": []}
    if pks.size > 0:
        sorted_indices = np.argsort(props["peak_heights"])[::-1]  # Descending by peak height
        for idx in sorted_indices:
            current_peak = pks[idx]
            if all(abs(current_peak - fp) > 50 for fp in filtered_pks):
                filtered_pks.append(current_peak)
                filtered_props["peak_heights"].append(props["peak_heights"][idx])

    filtered_pks = np.array(filtered_pks)
    filtered_props["peak_heights"] = np.array(filtered_props["peak_heights"])

    pk_ranges = ([], [], [], [])
    if filtered_pks.size > 0:
        pk_ranges = find_peak_range(data_1d, filtered_pks)

    return filtered_pks, filtered_props, pk_ranges

###############################################################################
# COSMIC DETECTION (STRICT LINE DETECTION)
###############################################################################
def detect_cosmic_line(adc, find_peaks_func, peak_height, peak_width, debug=False):
    """
    A "strict" cosmic detection:
    1) check_peaks_per_channel => ensures no channel has close or excessive peaks
    2) detect_main_line       => attempts to see if there's a time-coherent line across channels
    """
    # 1) Per-channel check
    if not check_peaks_per_channel(adc, find_peaks_func, peak_height, peak_width, debug=debug):
        return False

    # 2) Main line check
    if not detect_main_line(adc, find_peaks_func, peak_height, peak_width, debug=debug):
        return False

    return True

###############################################################################
# CHECK_PEAKS_PER_CHANNEL
###############################################################################
def check_peaks_per_channel(adc, find_peaks_func, peak_height, peak_width, debug=False):
    """
    For each channel in [0..NCC-1], do a raw find_peaks. If two peaks are
    within WIDTH_THRESHOLD => cosmic. Also if >2 tall peaks => cosmic.
    """
    for chn in range(NCC):
        data = adc[:, chn]
        pks, props = find_peaks(data, height=peak_height, width=peak_width)

        # If any two peaks < 40 apart => cosmic
        if len(pks) > 1:
            pks_sorted = np.sort(pks)
            if np.any(np.diff(pks_sorted) < WIDTH_THRESHOLD):
                if debug:
                    print(f"{output_align}[DEBUG] ch{chn}: two peaks < 50 samples => exclude event.")
                return False

        # If you also want to exclude events with >2 tall peaks, uncomment below:
        if len(pks) > 0 and "peak_heights" in props:
            n_peaks_above_amp_threshold = sum(amp > AMP_THRESHOLD for amp in props["peak_heights"])
            if n_peaks_above_amp_threshold > 2:
                if debug:
                    print(f"{output_align}[DEBUG] ch{chn}: more than 2 tall peaks => exclude event.")
                return False

    return True

def detect_main_line(adc, find_peaks_func, peak_height, peak_width, debug=False):
    """
    Attempt to see if there's a "main line" of peaks across the first NCC channels.
    Requires >=20 channels with a tall peak, and no more than 5 channels are out-of-sync
    by more than time_tolerance = 50.
    """
    time_tolerance = 50
    not_in_range = 0
    total_peaks = 0

    channel_peaks = []
    for chn in range(NCC):
        data = adc[:, chn]
        pks, props, _ = find_peaks_func(data, chn, peak_height, peak_width)
        best_time = None
        if len(pks) > 0 and "peak_heights" in props:
            idx_sort = np.argsort(props["peak_heights"])[::-1]
            for idx in idx_sort:
                if props["peak_heights"][idx] >= AMP_THRESHOLD:
                    best_time = pks[idx]
                    break
        if best_time is not None:
            channel_peaks.append(np.array([best_time]))
            total_peaks += 1
        else:
            channel_peaks.append(np.array([]))

    # E.g., require at least 47 out of 48 channels to have tall peaks
    if total_peaks < NCC - 1:
        return False
    # If the very first channel has no peak => fail
    if len(channel_peaks[0]) == 0:
        return False

    # Check adjacency
    for chn in range(1, NCC):
        prev_pk = channel_peaks[chn - 1]
        curr_pk = channel_peaks[chn]
        if len(curr_pk) > 0:
            t = curr_pk[0]
            if len(prev_pk) == 0 or abs(t - prev_pk[0]) > time_tolerance:
                not_in_range += 1
            if not_in_range > 5:
                return False
    return True

###############################################################################
# SHOW_EVENT
###############################################################################
def show_event(adc, run_time, event_id,
               binary_file_id, binary_event_id,
               converted_file_id, converted_event_id,
               saveFileName, isbatchbool):
    """
    Produces a 2D pcolor(...) plot for the entire ADC array of the event,
    saved to 'saveFileName'. Additionally, appends a line describing the
    plot to 'plot_names.txt' in the specified folder.
    """

    # -------------------------------------------
    # 1) Build a descriptive string for the plot
    # -------------------------------------------
    title_str = (f"{run_time} - Evt ID: {event_id} "
                 f"({binary_file_id},{binary_event_id}) "
                 f"({converted_file_id},{converted_event_id})")

    # -------------------------------------------
    # 2) Prepare to write a line in plot_names.txt
    # -------------------------------------------
    from pathlib import Path
    txt_file_dir = "/Users/alexlupu/github/dune-apex-umn/get-started-2025/valid-cosmic-rays"
    Path(txt_file_dir).mkdir(parents=True, exist_ok=True)  # Create folder if needed

    import os
    text_file_path = os.path.join(txt_file_dir, "plot_names.txt")

    # Append the metadata line
    with open(text_file_path, "a") as txtf:
        txtf.write(f"{saveFileName} => {title_str}\n")

    # -------------------------------------------
    # 3) Create and save the actual figure
    # -------------------------------------------
    plt.figure(figsize=(10, 6), dpi=100)
    plt.pcolor(adc, vmin=-100, vmax=100, cmap='YlGnBu_r')
    plt.colorbar(label="ADC")

    # Keep the title in the plot
    plt.title(title_str)

    plt.xlabel("Strips")
    plt.ylabel("Time ticks [0.5 µs/tick]")
    plt.xticks(np.arange(0, adc.shape[1] + 1, 10))

    plt.savefig(saveFileName, bbox_inches='tight')
    if not isbatchbool:
        plt.show()
    plt.clf()
    plt.close()

###############################################################################
# PREPARE_DATA
###############################################################################
def prepare_data(equalize_bool,
                 filter_bool,
                 data_event_dict,
                 evt_title,
                 j_file_nr,
                 j_evt_nr,
                 base_output_name,
                 first_time=True,
                 input_evtdisplay_bool=True):
    """
    1) Build an ADC array from data_event_dict (each 'chnX' key).
    2) Optionally do cosmic filtering.
    3) Possibly produce a 2D event-level plot if cosmic is detected.
    4) Return (adc, is_event_ok).
    """
    is_event_ok = True
    is_data_dimension_incorrect = False

    adc = np.empty((NTT, NC))
    for chn in range(NC):
        ch_key = f"chn{chn}"
        wave = data_event_dict[ch_key]
        if len(wave) != NTT:
            if first_time:
                print(f"{output_align}!! Event {j_evt_nr}: strip {chn} length {len(wave)} != {NTT}")
            is_data_dimension_incorrect = True
            break

        freq_val = most_frequent(wave)
        wave_bs = np.array(wave) - freq_val

        if equalize_bool:
            eq_factors = ch_eq_year(evt_title[:4])
            wave_bs = np.round(wave_bs * eq_factors[chn], decimals=1)
        adc[:, chn] = wave_bs

    # If any strip had the wrong waveform length, mark event invalid
    if is_data_dimension_incorrect:
        return adc, False

    # If user does not want cosmic filtering, keep event
    if not filter_bool:
        return adc, True

    # Perform cosmic detection
    pk_width, pk_height, min_strips = get_cosmic_rejection_parameters(evt_title[:4])
    is_cosmic = detect_cosmic_line(adc, find_peaks_50l, pk_height, pk_width, debug=first_time)

    if is_cosmic:
        if first_time:
            print(f"{output_align}[DEBUG] cosmic => adjacency checks => Evt({j_file_nr},{j_evt_nr}).")
        if input_evtdisplay_bool:
            event_id = data_event_dict.get("eventId", "???")
            binary_file_id = data_event_dict.get("binaryFileID", "???")
            binary_event_id = data_event_dict.get("binaryEventID", "???")
            converted_file_id = data_event_dict.get("convertedFileID", "???")
            converted_event_id = data_event_dict.get("convertedEventID", "???")

            save_file_name = f"{base_output_name}_file{j_file_nr}_evt{j_evt_nr}.pdf"
            show_event(adc, evt_title,
                       event_id,
                       binary_file_id,
                       binary_event_id,
                       converted_file_id,
                       converted_event_id,
                       save_file_name,
                       isbatchbool=False)
        is_event_ok = False

    return adc, is_event_ok

###############################################################################
# READ_DATA
###############################################################################
def remove_coherent_noise(data_event_dict):
    """
    Placeholder for any noise-removal logic. Currently does nothing.
    """
    return data_event_dict

def read_data(input_evtdisplay_bool,
              input_chndisplay_bool,
              input_equalize_bool,
              input_filter_bool,
              raw_data_folder_name):
    """
    Read all .json files in DATA_FOLDER, process each event with prepare_data(...),
    optionally produce 2D plots if cosmic, and return total events processed.
    """
    path_json = DATA_FOLDER
    Path(path_json).mkdir(parents=True, exist_ok=True)

    print(f"\n[{find_time_now()}] > Read data starts")
    print(f"{output_align} reading from => {path_json}")

    allfiles = sorted(os.listdir(path_json))
    evt_count = 0
    file_count = 0

    base_output_dir = "/Users/alexlupu/github/dune-apex-umn/Plots"
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)

    for fname in allfiles:
        if not fname.endswith(".json"):
            continue
        if fname.startswith(IGNORED_FOLDER_PREFIX):
            continue

        file_count += 1
        fullpath = os.path.join(path_json, fname)
        print(f"{output_align}> Reading file {fname}")

        with open(fullpath, "r") as f:
            data_all = json.load(f)
        events = data_all["all"]

        for i, event_dict in enumerate(events):
            evt_count += 1

            event_nc = remove_coherent_noise(event_dict)

            # Base name for output plots, e.g. "myFile"
            base_name  = fname[:-5]  # e.g. remove ".json"
            # Complete path to "Plots/myFile"
            base_outname = os.path.join(base_output_dir, base_name)

            first_time = (i == 0)
            adc, is_ok = prepare_data(
                input_equalize_bool,
                input_filter_bool,
                event_nc,
                raw_data_folder_name,
                file_count,
                i,
                base_outname,
                first_time=first_time,
                input_evtdisplay_bool=input_evtdisplay_bool
            )

            if is_ok:
                print(f"{output_align}Event {i} => Not Cosmic.")
            else:
                print(f"{output_align}Event {i} => Cosmic.")

    return evt_count

###############################################################################
# MAIN
###############################################################################
DEFAULTS = {
    "raw_data_folder_name": "20230719",
    "input_evtdisplay_bool": True,
    "input_chndisplay_bool": True,
    "input_equalize_bool": False,
    "input_filter_bool": True
}

def main(raw_data_folder_name,
         input_evtdisplay_bool,
         input_chndisplay_bool,
         input_equalize_bool,
         input_filter_bool,
         terminal_bool):
    """
    The main function (entry point after parse_args) that reads data,
    processes events, and logs final counts.
    """
    if input_equalize_bool:
        print(f"{output_align}: Charge equalization is active => scale factors applied.")

    total_evts = read_data(
        input_evtdisplay_bool,
        input_chndisplay_bool,
        input_equalize_bool,
        input_filter_bool,
        raw_data_folder_name
    )
    print(f"\n[{find_time_now()}] - Read data ends => {total_evts} events processed.\n")

###############################################################################
# SCRIPT ENTRY
###############################################################################
if __name__ == "__main__":
    raw_data_folder_, evt_disp_, chn_disp_, eq_bool_, filt_bool_, term_bool_ = parse_args(sys.argv, DEFAULTS)
    main(raw_data_folder_, evt_disp_, chn_disp_, eq_bool_, filt_bool_, term_bool_)

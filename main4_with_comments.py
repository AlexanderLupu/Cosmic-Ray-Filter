#!/usr/bin/env python3  # Shebang: tells the environment how to run this file
# -*- coding: utf-8 -*-  # Encoding declaration for Python source code
"""
Created on Mon Jan 13 15:39:35 2025  # Date/time of creation (example)

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

DATA_FOLDER = "/Users/alexlupu/github/dune-apex-umn/DATA/20230717/jsonData/"  # Folder with .json data

plt.rcParams.update({'font.size': 20})  # Update default font size for plots

###############################################################################
# MINIMAL UTILS
###############################################################################
def parse_args(argv, defaults):
    """
    Parse command-line arguments (sys.argv) and override defaults if flags are found.
    """
    d = dict(defaults)            # Copy defaults into a dictionary
    for arg in argv[1:]:          # Iterate over CLI arguments (excluding script name)
        if arg == "--no-display": # If user passed '--no-display'
            d["input_evtdisplay_bool"] = False   # Turn off event display
            d["input_chndisplay_bool"] = False   # Turn off channel display
        elif arg == "--no-filter": # If user passed '--no-filter'
            d["input_filter_bool"] = False       # Turn off cosmic filtering
        elif arg == "--equalize":  # If user passed '--equalize'
            d["input_equalize_bool"] = True      # Turn on channel equalization
    return (
        d["raw_data_folder_name"],       # 1) Name of the data folder
        d["input_evtdisplay_bool"],      # 2) Whether to display events
        d["input_chndisplay_bool"],      # 3) Whether to display channels
        d["input_equalize_bool"],        # 4) Whether to do equalization
        d["input_filter_bool"],          # 5) Whether to do cosmic filtering
        True                             # 6) Terminal bool (unused, but for structure)
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
    c = Counter(arr)                 # Build a Counter object from 'arr'
    return c.most_common(1)[0][0]    # Extract the most frequent element

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
    pk_start, pk_top, pk_end, pk_int = [], [], [], []  # Lists for results
    n_samp = len(data_1d)                              # Number of samples
    for pk in pk_indices:                              # Loop over each peak index
        temp_int = data_1d[pk]                         # Start integral with peak amplitude
        left = pk                                      # We'll move left from the peak
        while left > 0 and data_1d[left] > 0:          # As long as data is > 0
            left -= 1                                  # Move one sample left
            temp_int += data_1d[left]                  # Accumulate integral
        right = pk                                     # We'll move right from the peak
        while right < n_samp - 1 and data_1d[right] >= 0:  # As long as data is >= 0
            right += 1
            temp_int += data_1d[right]
        pk_start.append(left)                          # Record the left boundary
        pk_top.append(pk)                              # Record the peak index
        pk_end.append(right)                           # Record the right boundary
        pk_int.append(round(temp_int / 2.0, 1))        # Some integrated measure

    return (pk_start, pk_top, pk_end, pk_int)  # Return all 4 lists as a tuple

def find_peaks_50l(data_1d, chn, peak_height, peak_width):
    """
    1) Detect all peaks using scipy.signal.find_peaks with given height/width.
    2) Keep only peaks that are at least 50 samples away from previously kept peaks.
    3) Return the filtered peaks, their heights, and their ranges.
    """
    pks, props = find_peaks(data_1d, height=peak_height, width=peak_width)  # Raw peak find

    # Filter out peaks that lie within 50 samples of a taller peak
    filtered_pks = []                               # List of kept peaks
    filtered_props = {"peak_heights": []}           # Corresponding heights
    if pks.size > 0:
        sorted_indices = np.argsort(props["peak_heights"])[::-1]  # Sort descending by peak height
        for idx in sorted_indices:                                 # Iterate in order of tallest to smallest
            current_peak = pks[idx]
            if all(abs(current_peak - fp) > 50 for fp in filtered_pks):  # If it's 50+ samples from every kept peak
                filtered_pks.append(current_peak)                        # Keep the peak
                filtered_props["peak_heights"].append(props["peak_heights"][idx])  # Keep its height

    filtered_pks = np.array(filtered_pks)                  # Convert to NumPy array
    filtered_props["peak_heights"] = np.array(filtered_props["peak_heights"])  # Same for heights

    pk_ranges = ([], [], [], [])        # Default if no peaks
    if filtered_pks.size > 0:           # If we have filtered peaks
        pk_ranges = find_peak_range(data_1d, filtered_pks)  # Determine start, top, end, integral

    return filtered_pks, filtered_props, pk_ranges  # Return the final peaks, heights, and ranges

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
        return False  # If fails => not cosmic

    # 2) Main line check
    if not detect_main_line(adc, find_peaks_func, peak_height, peak_width, debug=debug):
        return False  # If fails => not cosmic

    return True  # If both succeed => cosmic

###############################################################################
# NEW CHECK_PEAKS_PER_CHANNEL
# (We now directly call raw scipy.signal.find_peaks and verify no two peaks < 50 apart)
###############################################################################
def check_peaks_per_channel(adc, find_peaks_func, peak_height, peak_width, debug=False):
    """
    For each channel in [0..NCC-1], do a raw find_peaks. If two peaks are
    within WIDTH_THRESHOLD => cosmic. Also if >2 tall peaks => cosmic.
    """
    for chn in range(NCC):
        data = adc[:, chn]                                   # Waveform for this channel
        pks, props = find_peaks(data, height=peak_height, width=peak_width)  # All peaks

        # Check if any two peaks < 50 samples => cosmic
        if len(pks) > 1:
            pks_sorted = np.sort(pks)
            if np.any(np.diff(pks_sorted) < WIDTH_THRESHOLD):  # If difference < 40
                if debug:
                    print(f"{output_align}[DEBUG] ch{chn}: two peaks < 50 samples => exclude event.")
                return False

        # Check if >2 tall peaks => cosmic
        if len(pks) > 0 and "peak_heights" in props:
            n_peaks_above_amp_threshold = sum(amp > AMP_THRESHOLD for amp in props["peak_heights"])
            if n_peaks_above_amp_threshold > 2:
                if debug:
                    print(f"{output_align}[DEBUG] ch{chn}: more than 2 tall peaks => exclude event.")
                return False

    return True  # If we never returned False, it means no channel triggered cosmic

def detect_main_line(adc, find_peaks_func, peak_height, peak_width, debug=False):
    """
    Attempt to see if there's a "main line" of peaks across the first NCC channels.
    Requires >=20 channels with a tall peak, and no more than 5 channels are out-of-sync
    by more than time_tolerance = 50.
    """
    time_tolerance = 50                    # Max allowed difference in peak times for adjacent channels
    not_in_range = 0                       # Count how many channels are out-of-sync
    total_peaks = 0                        # Count how many channels actually have a tall peak

    channel_peaks = []                     # List of best-time peaks for each channel
    for chn in range(NCC):
        data = adc[:, chn]                                       # Waveform
        pks, props, _ = find_peaks_func(data, chn, peak_height, peak_width)  # Filtered peak-finding
        best_time = None                                         # We'll store the tallest peak >= AMP_THRESHOLD
        if len(pks) > 0 and "peak_heights" in props:
            idx_sort = np.argsort(props["peak_heights"])[::-1]   # Sort descending by height
            for idx in idx_sort:
                if props["peak_heights"][idx] >= AMP_THRESHOLD:
                    best_time = pks[idx]
                    break
        if best_time is not None:
            channel_peaks.append(np.array([best_time]))           # Store as an array
            total_peaks += 1
        else:
            channel_peaks.append(np.array([]))                    # No peak found

    if total_peaks < 20:                    # Must have >=20 channels with a tall peak
        return False
    if len(channel_peaks[0]) == 0:          # If the very first channel has no peak
        return False

    for chn in range(1, NCC):               # Check adjacency from channel to channel
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
    saved to 'saveFileName'. If isbatchbool=False, we also show the plot.
    """
    plt.figure(figsize=(10, 6), dpi=100)             # Create figure of size 10x6
    plt.pcolor(adc, vmin=-100, vmax=100, cmap='YlGnBu_r')  # Color plot with specified range and colormap
    plt.colorbar(label="ADC")                               # Add colorbar with label

    title_str = (f"{run_time} - Evt ID: {event_id} "
                 f"({binary_file_id},{binary_event_id}) "
                 f"({converted_file_id},{converted_event_id})")  # Build a descriptive title
    plt.title(title_str)                             # Set the title
    plt.xlabel("Strips")                             # X-axis label
    plt.ylabel("Time ticks [0.5 µs/tick]")           # Y-axis label
    plt.xticks(np.arange(0, adc.shape[1] + 1, 10))   # X-ticks every 10 channels

    plt.savefig(saveFileName, bbox_inches='tight')   # Save figure to file
    if not isbatchbool:                              # If not batch mode
        plt.show()                                   # Show the plot interactively
    plt.clf()                                        # Clear the figure
    plt.close()                                      # Close the figure

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
    is_event_ok = True                                      # Will mark False if cosmic
    is_data_dimension_incorrect = False                     # Flag if waveform lengths are wrong

    adc = np.empty((NTT, NC))                               # Allocate an array for ADC data
    for chn in range(NC):                                   # For each channel
        ch_key = f"chn{chn}"                                # "chn0", "chn1", ...
        wave = data_event_dict[ch_key]                      # Retrieve that channel's waveform
        if len(wave) != NTT:                                # Check if length matches NTT
            if first_time:
                print(f"{output_align}!! Event {j_evt_nr}: strip {chn} length {len(wave)} != {NTT}")
            is_data_dimension_incorrect = True
            break

        freq_val = most_frequent(wave)                      # Baseline = most frequent value
        wave_bs = np.array(wave) - freq_val                 # Subtract baseline

        if equalize_bool:                                   # If user wants equalization
            eq_factors = ch_eq_year(evt_title[:4])          # Get scale factors
            wave_bs = np.round(wave_bs * eq_factors[chn], decimals=1)  # Apply factor
        adc[:, chn] = wave_bs                               # Store in ADC array

    if is_data_dimension_incorrect:                         # If any channel length was off
        return adc, False                                   # Return (adc, false)

    if not filter_bool:                                     # If no filter => keep it
        return adc, True

    # If we do filter => cosmic detection
    pk_width, pk_height, min_strips = get_cosmic_rejection_parameters(evt_title[:4])  # e.g. (5, 25, 20)

    # detect_cosmic_line => calls check_peaks_per_channel & detect_main_line
    is_cosmic = detect_cosmic_line(
        adc,                # ADC data
        find_peaks_50l,     # Our custom function that merges close peaks
        pk_height,          # e.g., 25
        pk_width,           # e.g., 5
        debug=first_time    # Print debug if first event
    )

    if is_cosmic:
        if first_time:
            print(f"{output_align}[DEBUG] cosmic => adjacency checks => Evt({j_file_nr},{j_evt_nr}).")
        if input_evtdisplay_bool:                          # If user wants event display
            event_id          = data_event_dict.get("eventId", "???")            # Retrieve event ID
            binary_file_id    = data_event_dict.get("binaryFileID", "???")       # Retrieve binary file ID
            binary_event_id   = data_event_dict.get("binaryEventID", "???")      # Retrieve binary event ID
            converted_file_id = data_event_dict.get("convertedFileID", "???")    # Retrieve converted file ID
            converted_event_id= data_event_dict.get("convertedEventID", "???")   # Retrieve converted event ID

            save_file_name = f"{base_output_name}_file{j_file_nr}_evt{j_evt_nr}.pdf"  # Output filename
            show_event(adc, evt_title,            # Make a 2D color plot
                       event_id,
                       binary_file_id,
                       binary_event_id,
                       converted_file_id,
                       converted_event_id,
                       save_file_name,
                       isbatchbool=False)
        is_event_ok = False  # Mark event as cosmic => exclude

    return adc, is_event_ok  # Return the ADC array and whether it's kept or not

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
    path_json = DATA_FOLDER                                          # Where to look for .json
    Path(path_json).mkdir(parents=True, exist_ok=True)               # Ensure dir exists

    print(f"\n[{find_time_now()}] > Read data starts")               # Log start time
    print(f"{output_align} reading from => {path_json}")             # Announce the directory

    allfiles = sorted(os.listdir(path_json))                         # List all items in path_json
    evt_count = 0                                                    # Count total events
    file_count = 0                                                   # Count how many JSON files

    base_output_dir = "/Users/alexlupu/github/dune-apex-umn/Plots"   # Where we store plots
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)         # Make sure plot dir exists

    for fname in allfiles:                                           # Iterate over items
        if not fname.endswith(".json"):                              # Skip non-JSON
            continue
        if fname.startswith(IGNORED_FOLDER_PREFIX):                  # Skip hidden files/folders
            continue

        file_count += 1                                              # Increment file counter
        fullpath = os.path.join(path_json, fname)                    # Build full path
        print(f"{output_align}> Reading file {fname}")               # Log which file we read

        with open(fullpath, "r") as f:                               # Open the JSON file
            data_all = json.load(f)                                  # Parse as Python dict
        events = data_all["all"]                                     # The "all" key has the events list

        for i, event_dict in enumerate(events):                      # Iterate over events
            evt_count += 1                                           # Increment event counter

            event_nc = remove_coherent_noise(event_dict)             # Possibly remove noise

            base_name  = fname[:-5]                                  # e.g. remove ".json"
            base_outname = os.path.join(base_output_dir, base_name)  # e.g. "Plots/myFile"

            first_time = (i == 0)                                    # True if first event in file
            adc, is_ok = prepare_data(
                input_equalize_bool,    # If user wants eq
                input_filter_bool,      # If user wants cosmic filter
                event_nc,              # The event data
                raw_data_folder_name,  # A title or year string
                file_count,            # File index
                i,                     # Event index
                base_outname,          # Where to save plots
                first_time=first_time, # For debug prints
                input_evtdisplay_bool=input_evtdisplay_bool  # Display 2D if cosmic
            )

            if is_ok:
                print(f"{output_align}Event {i} => KEPT.")           # Non-cosmic
            else:
                print(f"{output_align}Event {i} => EXCLUDED (cosmic).")  # Cosmic

    return evt_count  # Return total events processed

###############################################################################
# MAIN
###############################################################################
DEFAULTS = {
    "raw_data_folder_name": "20230717",  # Default "year_str" or data folder label
    "input_evtdisplay_bool": True,       # Whether to produce cosmic 2D plots
    "input_chndisplay_bool": True,       # (Unused in this script, but kept for structure)
    "input_equalize_bool": False,        # By default, no channel eq
    "input_filter_bool": True            # By default, do cosmic filter
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
if __name__ == "__main__":  # If we run this file directly (not import)
    raw_data_folder_, evt_disp_, chn_disp_, eq_bool_, filt_bool_, term_bool_ = parse_args(sys.argv, DEFAULTS)  # Parse CLI
    main(raw_data_folder_, evt_disp_, chn_disp_, eq_bool_, filt_bool_, term_bool_)  # Call the main function

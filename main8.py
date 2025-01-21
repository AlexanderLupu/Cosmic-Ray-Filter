#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:47:55 2025

@author: alexlupu
"""

"""
Full code: 
 - Phase A: scans all JSON, detects cosmic, saves cosmic events to cosmic_events.csv
 - Phase B: re-opens cosmic events, does single_event(...) channel plotting
 - find_charge_cluster(...) no longer returns 0 if overshadowed,
   it just skips adding neighbor's partial charge.
 - Now includes event_id fix: if two close peaks are found, we also print EvtID in the debug message.
"""

import os                           # Standard library for OS-level path operations
import sys                          # Standard library for system-related functions/args
import csv                          # Standard library for reading/writing CSV files
import json                         # Standard library for parsing/writing JSON
import numpy as np                  # NumPy for numerical operations and arrays
import matplotlib.pyplot as plt     # Matplotlib for plotting
from pathlib import Path            # For object-oriented filesystem paths
import time                         # For time/date-related functions
from collections import Counter     # Helps find the most frequent value
from scipy.signal import find_peaks # Used for peak detection in 1D signals

###############################################################################
# SETTINGS & HARD-CODED PATH
###############################################################################
NTT = 645                            # Number of samples (time ticks) per channel
NC = 128                             # Total number of channels
NCC = 48                             # Number of "collection" plane channels (subset)
AMP_THRESHOLD = 18                   # Minimum amplitude required to consider a peak "tall"
WIDTH_SPACE_THRESHOLD = 40           # Min distance between two peaks for them to be considered separate
WIDTH_PEAK_THRESHOLD = 5            # If peak width is greater than this, we exclude
IGNORED_FOLDER_PREFIX = "."          # We ignore files/folders starting with "."
output_align = "[ANALYSIS]"          # String prefix used in log messages

DATA_FOLDER       = "/Users/alexlupu/github/dune-apex-umn/DATA/20230722/jsonData/"  # Folder containing .json files
PLOTS_OUTPUT_DIR  = "/Users/alexlupu/github/dune-apex-umn/Plots"                    # Where output plots are stored
COSMIC_CSV_PATH   = "/Users/alexlupu/github/dune-apex-umn/cosmic_events.csv"        # CSV path where cosmic events will be saved

plt.rcParams.update({'font.size': 20})  # Sets default plot font size to 20

###############################################################################
# DEFAULTS FOR MAIN
###############################################################################
DEFAULTS = {
    "raw_data_folder_name": "20230722",  # The default folder name or "run" identifier
    "input_evtdisplay_bool": True,       # Whether to display 2D event-level plots for cosmic
    "input_chndisplay_bool": True,       # Whether we want single-channel displays
    "input_equalize_bool": False,        # Whether we apply any channel equalization
    "input_filter_bool": True            # Whether to do cosmic filtering
}

###############################################################################
# UTILS
###############################################################################
def parse_args(argv, defaults):
    """
    Reads command-line arguments and modifies the default dictionary if flags are found.
    """
    d = dict(defaults)  # Make a copy of the defaults
    for arg in argv[1:]:
        if arg == "--no-display":
            d["input_evtdisplay_bool"] = False
            d["input_chndisplay_bool"] = False
        elif arg == "--no-filter":
            d["input_filter_bool"] = False
        elif arg == "--equalize":
            d["input_equalize_bool"] = True
    return (
        d["raw_data_folder_name"],      # 1) raw_data_folder_name
        d["input_evtdisplay_bool"],     # 2) input_evtdisplay_bool
        d["input_chndisplay_bool"],     # 3) input_chndisplay_bool
        d["input_equalize_bool"],       # 4) input_equalize_bool
        d["input_filter_bool"],         # 5) input_filter_bool
        True                            # 6) Terminal bool (unused but kept for structure)
    )

def find_time_now():
    """
    Returns a formatted time string (YYYY-MM-DD HH:MM:SS).
    """
    return time.strftime("%Y-%m-%d %H:%M:%S")

def most_frequent(arr):
    """
    Returns the most frequent (mode) value in an array using Counter.
    """
    c = Counter(arr)
    return c.most_common(1)[0][0]

###############################################################################
# COSMIC DETECTION PARAMETERS
###############################################################################
def get_cosmic_rejection_parameters(year_str):
    """
    Returns a tuple (peak_width, peak_height, min_strips).
    Here we set (5, AMP_THRESHOLD, NCC-1).
    """
    return (5, AMP_THRESHOLD, NCC - 1)

def ch_eq_year(year_str):
    """
    Returns an array of channel equalization factors (currently all 1.0).
    """
    return np.ones(NC, dtype=float)

###############################################################################
# PEAK-FINDING
###############################################################################
def find_peak_range(data_1d, pk_indices):
    """
    Given 1D data and indices of peaks, compute the left/right extents
    and integral under each peak.
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
    1) Use find_peaks to locate peaks with certain height/width
    2) Filter out peaks that lie <50 samples from a taller peak
    3) Return the remaining peak indices, their heights, and their ranges
    """
    pks, props = find_peaks(data_1d, height=peak_height, width=peak_width)
    filtered_pks = []
    filtered_props = {"peak_heights": []}
    if pks.size > 0:
        sort_idx_desc = np.argsort(props["peak_heights"])[::-1]
        for idx in sort_idx_desc:
            current_peak = pks[idx]
            if all(abs(current_peak - fp) > AMP_THRESHOLD for fp in filtered_pks):
                filtered_pks.append(current_peak)
                filtered_props["peak_heights"].append(props["peak_heights"][idx])
    filtered_pks = np.array(filtered_pks)
    filtered_props["peak_heights"] = np.array(filtered_props["peak_heights"])
    pk_ranges = ([], [], [], [])
    if filtered_pks.size > 0:
        pk_ranges = find_peak_range(data_1d, filtered_pks)
    return filtered_pks, filtered_props, pk_ranges

###############################################################################
# COSMIC DETECTION
###############################################################################
def detect_cosmic_line(adc, find_peaks_func, peak_height, peak_width, debug=False, event_id=None):
    """
    A two-step process:
     1) check_peaks_per_channel => ensures no channel has close/excessive peaks
     2) detect_main_line       => ensures a coherent time line across channels
    """
    if not check_peaks_per_channel(adc, find_peaks_func, peak_height, peak_width,
                                   debug=debug, event_id=event_id):
        return False
    if not detect_main_line(adc, find_peaks_func, peak_height, peak_width, debug=debug):
        return False
    return True

def check_peaks_per_channel(adc,
                            find_peaks_func,
                            peak_height,
                            peak_width,
                            debug=False,
                            event_id=None):
    """
    For the first NCC channels:
     - Find peaks
     - If any two peaks < WIDTH_SPACE_THRESHOLD => cosmic
     - If the peak width > WIDTH_PEAK_THRESHOLD => cosmic
    """
    from scipy.signal import find_peaks
    for chn in range(NCC):
        data = adc[:, chn]
        pks, props = find_peaks(data, height=peak_height, width=peak_width)

        if len(pks) > 1:
            pks_sorted = np.sort(pks)
            if np.any(np.diff(pks_sorted) < WIDTH_SPACE_THRESHOLD):
                if debug:
                    # Now we have event_id available
                    print(f"{output_align}[DEBUG] EvtID {event_id}, ch{chn}: "
                          f"2 peaks are less than {WIDTH_SPACE_THRESHOLD} apart")
                return False

        if peak_width > WIDTH_PEAK_THRESHOLD:
            if debug:
                print(f"{output_align}[DEBUG] ch{chn}: peak_width > {WIDTH_PEAK_THRESHOLD}")
            return False
    return True

def detect_main_line(adc, find_peaks_func, peak_height, peak_width, debug=False):
    """
    Ensures we have a coherent "line" across multiple strips:
     - at least NCC-1 channels have a tall peak
     - no more than 5 channels out of time sync by >50
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

    if total_peaks < NCC - 1:
        return False

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
# SHOW_EVENT (2D event-level)
###############################################################################
def show_event(adc, run_time, event_id,
               binary_file_id, binary_event_id,
               converted_file_id, converted_event_id,
               saveFileName, isbatchbool):
    """
    Creates a 2D color plot of the entire ADC array for this event,
    also writes a line in plot_names_20230725.csv with event metadata.
    """
    import csv
    title_str = (f"{run_time} - Evt ID: {event_id} "
                 f"({binary_file_id},{binary_event_id}) "
                 f"({converted_file_id},{converted_event_id})")

    csv_file_dir = "/Users/alexlupu/github/dune-apex-umn/get-started-2025/valid-cosmic-rays"
    Path(csv_file_dir).mkdir(parents=True, exist_ok=True)
    csv_file_path = os.path.join(csv_file_dir, "plot_names_20230725.csv")

    file_exists = os.path.exists(csv_file_path)
    with open(csv_file_path, "a", newline='') as csvf:
        writer = csv.writer(csvf)
        if not file_exists:
            writer.writerow(["#Folder_name", "EvtID", "Raw_file_nr", "Raw_Evt_nr", "File_nr", "Evt_nr"])
        writer.writerow([
            run_time,
            event_id,
            binary_file_id,
            binary_event_id,
            converted_file_id,
            converted_event_id
        ])

    plt.figure(figsize=(10, 6), dpi=100)
    plt.pcolor(adc, vmin=-100, vmax=100, cmap='YlGnBu_r')
    plt.colorbar(label="ADC")
    plt.title(title_str)
    plt.xlabel("Strips")
    plt.ylabel("Time ticks [0.5 µs/tick]")

    # Rotate x-ticks 45 degrees to avoid overlap
    plt.xticks(
        np.arange(0, adc.shape[1] + 1, 10),
        rotation=45
    )
    plt.tight_layout()

    plt.savefig(saveFileName, bbox_inches='tight')
    if not isbatchbool:
        plt.show()
    plt.clf()
    plt.close()

###############################################################################
# CLUSTERING LOGIC (NO ZERO RETURN)
###############################################################################
def find_charge_cluster(adc, chn_c, peak_charge, r, s, candidate_event_C):
    """
    Combines charge from adjacent strips if they overlap in time,
    but if the neighbor's peak is bigger, we skip adding neighbor's partial charge
    rather than zeroing out the entire cluster.
    """
    peak_charge_cluster = peak_charge
    overshadowed = False

    if chn_c >= NCC:
        return round(peak_charge_cluster, 1)

    if chn_c < (NCC - 1):
        c_strip_right = chn_c + 1
        charge_right = 0

        if len(r[c_strip_right]) == 0:
            pass
        else:
            for c_strip_right_event in range(len(s[c_strip_right])):
                if ( r[c_strip_right][0][1][c_strip_right_event] > r[chn_c][0][0][candidate_event_C]
                  and r[c_strip_right][0][1][c_strip_right_event] < r[chn_c][0][2][candidate_event_C] ):
                    if r[c_strip_right][0][3][c_strip_right_event] > r[chn_c][0][3][candidate_event_C]:
                        overshadowed = True
                        break
                else:
                    for tick in range(r[chn_c][0][0][candidate_event_C],
                                      r[chn_c][0][2][candidate_event_C]):
                        if tick < adc.shape[0] and c_strip_right < adc.shape[1]:
                            charge_right += adc[tick, c_strip_right]

            if (not overshadowed) and (charge_right > 0):
                charge_right /= 2.0
                peak_charge_cluster += charge_right

        if not overshadowed and chn_c > 0:
            c_strip_left = chn_c - 1
            charge_left = 0
            for tick in range(r[chn_c][0][0][candidate_event_C],
                              r[chn_c][0][2][candidate_event_C]):
                if tick < adc.shape[0] and c_strip_left < adc.shape[1]:
                    charge_left += adc[tick, c_strip_left]
            if charge_left > 0:
                charge_left /= 2.0
                peak_charge_cluster += charge_left

    return round(peak_charge_cluster, 1)

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
    1) Build a 2D ADC array from data_event_dict
    2) Optionally do cosmic detection
    3) If cosmic => create a 2D plot
    4) Return (adc, is_event_ok)
    """
    is_event_ok = True
    is_data_dimension_incorrect = False

    adc = np.empty((NTT, NC))
    for chn in range(NC):
        wave = data_event_dict.get(f"chn{chn}", [])
        if len(wave) != NTT:
            if first_time:
                print(f"{output_align}!! Event {j_evt_nr}: strip {chn} length {len(wave)} != {NTT}")
            is_data_dimension_incorrect = True
            break

        freq_val = most_frequent(wave)
        wave_bs  = np.array(wave) - freq_val

        if equalize_bool:
            eq_factors = ch_eq_year(evt_title[:4])
            wave_bs = np.round(wave_bs * eq_factors[chn], decimals=1)

        adc[:, chn] = wave_bs

    if is_data_dimension_incorrect:
        return adc, False

    if not filter_bool:
        return adc, True

    # We retrieve the event ID from data_event_dict, if present
    current_event_id = data_event_dict.get("eventId", "???")

    pk_width, pk_height, _ = get_cosmic_rejection_parameters(evt_title[:4])
    # Pass event_id to detect_cosmic_line so it can be used in check_peaks_per_channel
    is_cosmic = detect_cosmic_line(
        adc,
        find_peaks_50l,
        pk_height,
        pk_width,
        debug=first_time,
        event_id=current_event_id
    )
    if is_cosmic:
        if first_time:
            print(f"{output_align}[DEBUG] cosmic => adjacency => Evt({j_file_nr},{j_evt_nr}).")
        if input_evtdisplay_bool:
            event_id          = current_event_id
            binary_file_id    = data_event_dict.get("binaryFileID", "???")
            binary_event_id   = data_event_dict.get("binaryEventID", "???")
            converted_file_id = data_event_dict.get("convertedFileID", "???")
            converted_event_id= data_event_dict.get("convertedEventID", "???")

            save_file_name = f"{base_output_name}_file{j_file_nr}_evt{j_evt_nr}.pdf"
            show_event(
                adc,
                evt_title,
                event_id,
                binary_file_id,
                binary_event_id,
                converted_file_id,
                converted_event_id,
                save_file_name,
                isbatchbool=False
            )
        is_event_ok = False

    return adc, is_event_ok

###############################################################################
# PHASE A: SCAN & SAVE COSMIC EVENTS TO CSV
###############################################################################
def scan_and_save_cosmic_events(raw_data_folder_name,
                                input_evtdisplay_bool,
                                input_chndisplay_bool,
                                input_equalize_bool,
                                input_filter_bool):
    """
    1) Looks for .json files in DATA_FOLDER
    2) For each event, calls prepare_data(...); if cosmic => writes row to CSV
    3) No single-channel plots yet
    """
    path_json = DATA_FOLDER                               # Store the global DATA_FOLDER path in a local variable
    Path(path_json).mkdir(parents=True, exist_ok=True)    # Ensure the folder (and any parents) exist; create if needed

    csv_exists = os.path.exists(COSMIC_CSV_PATH)          # Check if the cosmic_events.csv file already exists
    with open(COSMIC_CSV_PATH, "a", newline='') as csvf:  # Open cosmic_events.csv in append mode
        writer = csv.writer(csvf)                         # Create a CSV writer object for adding lines
        if not csv_exists:                                # If the file did not exist before
            writer.writerow(["filename","event_index","file_id","event_id"])  # Write the header row

        allfiles = sorted(os.listdir(path_json))          # List and sort all files in the DATA_FOLDER
        file_count = 0                                    # Count how many JSON files we process
        evt_count = 0                                     # Count how many events we see overall

        print(f"\n[{find_time_now()}] > Start scanning => {COSMIC_CSV_PATH}")  # Print a timestamped start message

        for fname in allfiles:                            # Loop through each file in the folder
            if not fname.lower().endswith(".json"):       # Skip files that don't end with ".json"
                continue
            if fname.startswith(IGNORED_FOLDER_PREFIX):    # Also skip files that start with the ignored prefix (e.g. ".DS_Store")
                continue

            file_count += 1                               # Increment JSON file count
            fullpath = os.path.join(path_json, fname)     # Build the full path to this JSON file

            with open(fullpath, "r") as f:                # Open the JSON file in read mode
                data_all = json.load(f)                   # Load its contents into a Python dictionary
            events = data_all["all"]                      # 'events' is the list of event dictionaries under the 'all' key

            base_out_dir = PLOTS_OUTPUT_DIR               # The root directory where output plots (if any) would be saved
            Path(base_out_dir).mkdir(parents=True, exist_ok=True)  # Ensure that plots directory also exists

            base_name   = fname[:-5]                      # Strip off the ".json" extension from the filename
            base_outname= os.path.join(base_out_dir, base_name)  # Create a base output path for this file's potential plots

            for i, event_dict in enumerate(events):       # Loop over each event in the JSON file
                evt_count += 1                            # Increment total event count

                # Call prepare_data(...) which returns:
                #   adc => the 2D array of waveforms
                #   is_ok => True if event is NOT cosmic, False if cosmic
                adc, is_ok = prepare_data(
                    input_equalize_bool,                  # Whether we apply equalization
                    input_filter_bool,                    # Whether we do cosmic filtering
                    event_dict,                           # The event dictionary with waveforms and metadata
                    raw_data_folder_name,                 # The raw data folder name or "run" identifier
                    file_count,                           # Which file number we are on
                    i,                                    # The event index in the current file
                    base_outname,                         # Base name for saving potential plots
                    first_time=(i==0),                    # Flag if it's the first event in this file
                    input_evtdisplay_bool=input_evtdisplay_bool  # Whether to display cosmic events with 2D plots
                )

                if not is_ok:                             # If the event is cosmic (is_ok == False)
                    file_id = event_dict.get("binaryFileID", "???")  # Extract 'binaryFileID' if present, else "???"
                    evt_id  = event_dict.get("eventId", "???")       # Extract 'eventId' if present, else "???"
                    # Write a row in cosmic_events.csv indicating this cosmic event
                    writer.writerow([fname, i, file_id, evt_id])

        # After we've processed all JSON files and events, print a completion message
        print(f"{output_align} Done scanning => {COSMIC_CSV_PATH}")


###############################################################################
# PER-EVENT CHANNEL PLOTTING
###############################################################################
def nicer_single_evt_display(chn, y, peaks, save_file_name, evt_title, yrange, peak_ranges, charge, isbatchbool):
    """
    Draws a single channel's waveform, marks the main peak, and labels the cluster charge.
    """
    fig, ax = plt.subplots(figsize=(32, 9))               # Create a new figure (32" x 9") and axes
    str_chn = str(chn).zfill(3)                           # Convert channel number to a zero-padded string, e.g. "007"
    label = f"Chn_{str_chn}; Charge cluster: {charge} ADC*µs"  # Build a label showing channel and charge info

    peaks = peaks.astype(int)                             # Ensure 'peaks' is an integer array for indexing
    ax.plot(y, label=label)                               # Plot the entire waveform 'y', including a legend label

    zeros = y[peaks] if chn < NCC else np.zeros(len(peaks))  # If 'chn' is in the collection-plane range, get y[peaks], else zeros
    ax.plot(peaks, zeros, "x")                            # Mark the peaks with an 'x' on the plot

    # If 'peak_ranges' is properly structured with 4 arrays, each of matching length, fill the region under each peak
    if len(peak_ranges) == 4 and len(peak_ranges[0]) == len(peak_ranges[2]):
        pk_start = np.array(peak_ranges[0], dtype=int)    # Convert start indices to int
        pk_end   = np.array(peak_ranges[2], dtype=int)    # Convert end indices to int
        for s_tick, e_tick in zip(pk_start, pk_end):       # Loop over each start/end pair
            region_x = np.arange(s_tick, e_tick)          # Range of x indices for the peak
            region_y = y[region_x]                        # Corresponding y values
            if chn < NCC:                                 # Only fill if in collection-plane range
                ax.fill_between(region_x, 0, region_y, color='gold', alpha=0.5)

    ax.set_ylabel('ADC (baseline-subtracted)', labelpad=10, fontsize=24) # Label the Y-axis
    ax.set_xlabel('time ticks [0.5 µs/tick]', fontsize=24)              # Label the X-axis
    ax.legend(fontsize=24)                                             # Draw legend with bigger font
    ax.grid(True, which='major', axis='both', linewidth=1, color='gray')  # Major grid lines
    ax.grid(True, which='minor', axis='x', linewidth=0.5, linestyle='dashed', color='gray')  # Minor grid lines on X-axis

    ax.set_xticks(np.arange(0, NTT + 1, 50))        # Set major ticks every 50
    ax.set_xticks(np.arange(0, NTT + 1, 10), minor=True)  # Minor ticks every 10

    if yrange:                                      # If user wants a fixed vertical range
        if chn < NCC:                               # If it's a collection-plane channel
            ax.set_ylim(-20, 300)                   # Range from -20 to 300
        else:
            ax.set_ylim(-250, 250)                  # Else range from -250 to 250

    ax.set_xlim(0, len(y))                          # X-limits from 0 to length of waveform
    ax.yaxis.set_label_coords(-.04, 0.5)            # Shift Y-label slightly left
    ax.spines['right'].set_visible(False)           # Hide the right boundary spine

    plt.title(f"{evt_title} - Strip {chn + 1}", fontsize=30)       # Main title
    plt.subplots_adjust(left=0.06, right=0.94, top=0.96, bottom=0.08) # Adjust subplot padding
    plt.tight_layout()                                              # Auto-fix layout to prevent overlap

    str_strip = str(chn + 1).zfill(3)                               # Zero-pad the strip index
    out_fname = f"{save_file_name}_Strip{str_strip}.pdf"            # Build output PDF filename
    plt.savefig(out_fname, dpi=100)                                 # Save the figure
    if not isbatchbool:                                             # If we're not in batch mode
        plt.show()                                                  # Show the figure on screen
    plt.clf()                                                       # Clear the figure
    plt.close()                                                     # Close the figure completely

def single_event(adc, evt_title, out_dir):
    """
    Builds a data structure of found peaks => calls find_charge_cluster(...) =>
    plots each channel with nicer_single_evt_display.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)       # Ensure the output directory exists
    pk_width, pk_height, _ = get_cosmic_rejection_parameters(evt_title[:4]) # Get cosmic detection parameters

    r = [[] for _ in range(NC)]                            # 'r' will hold arrays of [start,top,end,integral] for each peak
    s = [[] for _ in range(NC)]                            # 's' will hold the number of peaks for each channel

    for chn in range(NC):                                  # Loop over all channels in the ADC
        data_1d = adc[:, chn]                              # Extract the waveform for this channel
        pks, props, pk_ranges = find_peaks_50l(data_1d, chn, pk_height, pk_width) # Find peaks in that waveform
        (pk_start, pk_top, pk_end, pk_int) = pk_ranges     # Unpack start, top, end, integral arrays
        r[chn].append([pk_start, pk_top, pk_end, pk_int])  # Store this info in r[chn][0]
        s[chn] = range(len(pk_start))                      # Record how many peaks we found for adjacency logic

    for chn in range(NC):                                  # Now loop again to plot each channel
        data_1d = adc[:, chn]                              # The waveform for channel 'chn'
        if len(r[chn]) == 0 or len(r[chn][0][0]) == 0:      # If no peaks found
            out_prefix = os.path.join(out_dir, f"Chn{chn}")# Build a path for saving the plot
            nicer_single_evt_display(
                chn=chn,
                y=data_1d,
                peaks=np.array([], dtype=int),             # No peak indices
                save_file_name=out_prefix,
                evt_title=evt_title,
                yrange=True,
                peak_ranges=([], [], [], []),              # Empty peak range
                charge=0,
                isbatchbool=True
            )
            continue                                       # Skip to the next channel

        pk_start = r[chn][0][0]                            # Start array of all peaks
        pk_top   = r[chn][0][1]                            # Top array (peak indices)
        pk_end   = r[chn][0][2]                            # End array
        pk_int   = r[chn][0][3]                            # Integral array

        idx_max = np.argmax(pk_int)                        # Find which peak has the largest integral
        peak_charge = pk_int[idx_max]                      # The integral of that biggest peak

        chg_cluster = find_charge_cluster(                 # Possibly merge with neighbors
            adc,
            chn,
            peak_charge,
            r,
            s,
            idx_max
        )

        out_prefix = os.path.join(out_dir, f"Chn{chn}")    # Build a path prefix for saving the PDF
        single_peak_array = np.array([pk_top[idx_max]], dtype=int)  # The peak index of the main peak
        single_pk_range = (                                # Build a sub-range (start, top, end, integral) for just that peak
            [pk_start[idx_max]],
            [pk_top[idx_max]],
            [pk_end[idx_max]],
            [pk_int[idx_max]]
        )

        # Finally plot the channel with the merged cluster charge
        nicer_single_evt_display(
            chn=chn,
            y=data_1d,
            peaks=single_peak_array,
            save_file_name=out_prefix,
            evt_title=evt_title,
            yrange=True,
            peak_ranges=single_pk_range,
            charge=chg_cluster,
            isbatchbool=True
        )

###############################################################################
# PHASE B: READ CSV => PLOT COSMIC EVENTS
###############################################################################
def plot_cosmic_events_from_csv():
    """
    Re-opens cosmic_events.csv, loops each row => re-opens the .json => single_event(...) 
    for per-channel plots of cosmic events only.
    """
    if not os.path.exists(COSMIC_CSV_PATH):                   # If cosmic_events.csv doesn't exist
        print(f"{output_align} No CSV found at {COSMIC_CSV_PATH}. Nothing to plot.")
        return

    with open(COSMIC_CSV_PATH, "r") as csvf:                  # Open the CSV in read mode
        reader = csv.DictReader(csvf)                         # Use DictReader to get each row as a dictionary
        rows = list(reader)                                   # Convert to a list so we can iterate more than once if needed

    cosmic_count = len(rows)                                  # How many cosmic events are recorded
    print(f"{output_align} Found {cosmic_count} cosmic events in {COSMIC_CSV_PATH}. Plotting...")

    for row in rows:                                          # Loop over each cosmic event row  #643-677
        fname     = row["filename"]                           # The JSON file name
        event_idx = int(row["event_index"])                   # Which event within that file
        file_id   = row["file_id"]                            # Additional metadata
        evt_id    = row["event_id"]                           # Additional metadata

        fullpath = os.path.join(DATA_FOLDER, fname)           # Build full path to that JSON
        if not os.path.exists(fullpath):                      # If it doesn't exist, skip
            print(f"{output_align} Missing file {fullpath}; skip.")
            continue

        with open(fullpath, "r") as f:                        # Open and parse the JSON
            data_all = json.load(f)
        events = data_all["all"]                              # The list of events under the "all" key

        if event_idx < 0 or event_idx >= len(events):         # If the event index is invalid, skip
            print(f"{output_align} Invalid event_idx {event_idx} in {fname}; skip.")
            continue

        event_dict = events[event_idx]                        # Grab the specific event dict
        adc = np.empty((NTT, NC))                             # Allocate a 2D array for waveforms
        is_bad_dim = False                                    # Flag to track dimension mismatches

        for chn in range(NC):                                 # Populate 'adc' channel by channel
            wave = event_dict.get(f"chn{chn}", [])
            if len(wave) != NTT:                              # If length is not the expected 645
                is_bad_dim = True
                break
            freq_val = most_frequent(wave)                    # Baseline value (most frequent sample)
            wave_bs  = np.array(wave) - freq_val             # Subtract baseline
            adc[:, chn] = wave_bs                             # Store in the ADC array

        if is_bad_dim:                                        # If any channel had a bad dimension, skip this event
            print(f"{output_align} Bad wave size in {fname} Evt {event_idx}; skip.")
            continue

        cosmic_label = f"FileID {file_id}, EvtID {evt_id} (COSMIC)"  # Construct a label for logging/plot titles
        out_subdir   = os.path.join(PLOTS_OUTPUT_DIR, "cosmic_from_csv", f"{fname}_Evt{event_idx}")  
        # Build a subdirectory path to store per-channel plots for this cosmic event

        single_event(adc, cosmic_label, out_subdir)           # Finally, plot each channel for this cosmic event

    print(f"{output_align} Done plotting cosmic events from CSV.")


###############################################################################
# MAIN
###############################################################################
def main():
    """
    1) Phase A: scan_and_save_cosmic_events(...) => cosmic_events.csv
    2) Phase B: plot_cosmic_events_from_csv(...) => single-channel plots for cosmic events
    """
    raw_data_folder_name  = DEFAULTS["raw_data_folder_name"]
    input_evtdisplay_bool = DEFAULTS["input_evtdisplay_bool"]
    input_chndisplay_bool = DEFAULTS["input_chndisplay_bool"]
    input_equalize_bool   = DEFAULTS["input_equalize_bool"]
    input_filter_bool     = DEFAULTS["input_filter_bool"]

    # Phase A => find cosmic events & store them
    scan_and_save_cosmic_events(
        raw_data_folder_name,
        input_evtdisplay_bool,
        input_chndisplay_bool,
        input_equalize_bool,
        input_filter_bool
    )

    # Phase B => read cosmic_events.csv & produce channel-level plots
    plot_cosmic_events_from_csv()

if __name__ == "__main__":
    main()

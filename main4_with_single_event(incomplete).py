#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os               # Standard library for operating system interactions
import sys              # Standard library for system-specific parameters and functions
import json             # Standard library for JSON parsing
import numpy as np      # NumPy for numerical arrays and operations
import matplotlib.pyplot as plt  # Matplotlib for plotting
from pathlib import Path         # For object-oriented filesystem paths
import time                      # Standard library for time-based operations
from collections import Counter  # For counting elements efficiently
from scipy.signal import find_peaks  # For peak-finding in signals

###############################################################################
# SETTINGS & HARD-CODED PATH
###############################################################################
NTT = 645       # Number of time ticks/samples per channel
NC = 128        # Total number of channels
NCC = 48        # Number of "collection" channels (subset)
AMP_THRESHOLD   = 18  # Amplitude threshold for detecting tall peaks
WIDTH_THRESHOLD = 40  # Min separation (in samples) for "close" peaks
IGNORED_FOLDER_PREFIX = "."  # Files/folders to ignore if they start with '.'
output_align = "[ANALYSIS]"   # Prefix for log messages

DATA_FOLDER = "/Users/alexlupu/github/dune-apex-umn/DATA/20230717/jsonData/"  # Path to JSON data
PLOTS_OUTPUT_DIR = "/Users/alexlupu/github/dune-apex-umn/Plots"              # Where to save plots

plt.rcParams.update({'font.size': 20})  # Update default font size for plots

###############################################################################
# MINIMAL UTILS
###############################################################################
def parse_args(argv, defaults):
    """
    Parse command-line arguments and override default settings if flags are found.
    """
    d = dict(defaults)  # Make a copy of the defaults
    for arg in argv[1:]:  # Iterate over command-line args (skipping the script name)
        if arg == "--no-display":        # If user wants no display
            d["input_evtdisplay_bool"] = False
            d["input_chndisplay_bool"] = False
        elif arg == "--no-filter":       # If user wants no filtering
            d["input_filter_bool"] = False
        elif arg == "--equalize":        # If user wants equalization
            d["input_equalize_bool"] = True
    return (
        d["raw_data_folder_name"],       # Return parsed "raw_data_folder_name"
        d["input_evtdisplay_bool"],      # Return whether event display is on
        d["input_chndisplay_bool"],      # Return whether channel display is on
        d["input_equalize_bool"],        # Return whether equalization is on
        d["input_filter_bool"],          # Return whether filtering is on
        True                             # Terminal bool (unused, but kept for consistency)
    )

def find_time_now():
    """
    Returns current time as a formatted string.
    """
    return time.strftime("%Y-%m-%d %H:%M:%S")  # Format "YYYY-MM-DD HH:MM:SS"

def most_frequent(arr):
    """
    Return the most frequent element in a list/array.
    """
    c = Counter(arr)  # Create a Counter from the array
    return c.most_common(1)[0][0]  # Extract the most common element

###############################################################################
# COSMIC DETECTION PARAMETERS
###############################################################################
def get_cosmic_rejection_parameters(year_str):
    """
    Returns (peak_width, peak_height, min_strips).
    Hard-coded for now: (5, 25, 20).
    """
    return (5, 25, 20)

def ch_eq_year(year_str):
    """
    Returns an array of scale factors (all 1.0) for each channel,
    used if equalization is desired.
    """
    return np.ones(NC, dtype=float)

###############################################################################
# SINGLE CHANNEL PLOTTING (OPTIONAL)
###############################################################################

def nicer_single_evt_display(chn, y, peaks, save_file_name, evt_title, yrange, peak_ranges, charge, isbatchbool):
    """
    A more elaborate single-channel display, including fill_between recognized peaks,
    custom axis ticks, etc.
    """
    fig, ax = plt.subplots(figsize=(32, 9))            # Create a figure and axis
    str_chn = str(chn).zfill(3)                       # Zero-pad channel number
    label = 'Chn_' + str_chn                           # Start with channel label

    # Ensure 'peaks' are integers (in case they became float)
    peaks = peaks.astype(int)

    if chn < NCC:                                      # If channel is in the "collection" set
        ps = [p * 0.5 for p in peaks]                  # Convert sample index to µs
        p_list = [y[p] for p in peaks]                 # Get amplitude at each peak
        label += f";\nPeaks: {ps} µs, {p_list} ADC\nCharge: {charge} ADC*µs"
    else:                                              # If channel is beyond the collection set
        ps = [p * 0.5 for p in peaks]
        label += f";\nCharged particle(s) detected at: {ps} µs"

    ax.plot(y, label=label)                            # Plot the waveform

    zeros = y[peaks] if chn < NCC else np.zeros(len(peaks))  # Y-values at peaks, or zeros if chn >= NCC
    ax.plot(peaks, zeros, "x")                                # Mark peak positions with 'x'

    # Fill area under recognized peaks if the arrays are matching in length
    if len(peak_ranges) > 0 and len(peak_ranges[0]) == len(peak_ranges[2]):
        pk_start = np.array(peak_ranges[0], dtype=int)  # Convert start indices to int
        pk_end   = np.array(peak_ranges[2], dtype=int)  # Convert end indices to int
        for s, e in zip(pk_start, pk_end):
            peak_region_x = np.arange(s, e)             # Range from start to end
            peak_region_y = y[peak_region_x]            # Y-values in that region
            if chn < NCC:
                ax.fill_between(peak_region_x, 0, peak_region_y, color='gold', alpha=0.5)

    ax.set_ylabel('ADC (baseline subtracted)', labelpad=10, fontsize=24)  # Set Y-axis label
    ax.set_xlabel('time ticks [0.5 µs/tick]', fontsize=24)                # Set X-axis label
    ax.legend(fontsize=24)                                               # Show legend with bigger font
    ax.grid(True, which='major', axis='both', linewidth=1, color='gray') # Major grid
    ax.grid(True, which='minor', axis='x', linewidth=0.5, linestyle='dashed', color='gray')  # Minor grid on x-axis
    ax.set_xticks(np.arange(0, NTT + 1, 50))   # X-ticks every 50
    ax.set_xticks(np.arange(0, NTT + 1, 10), minor=True)  # Minor ticks every 10

    if yrange:  # If user wants a certain y-range
        if chn < NCC:
            ax.set_ylim(-20, 300)
        else:
            ax.set_ylim(-250, 250)

    ax.set_xlim(0, len(y))           # X-limits from 0 to length of waveform
    ax.yaxis.set_label_coords(-.04, 0.5)   # Move Y-label slightly to the left
    ax.spines['right'].set_visible(False)  # Hide the right spine

    plt.title(f"{evt_title} - Strip {chn + 1}", fontsize=30)  # Title with event info
    plt.subplots_adjust(left=0.06, right=0.94, top=0.96, bottom=0.08)  # Adjust spacing
    plt.tight_layout()                                                # Auto-adjust layout

    str_strip = str(chn + 1).zfill(3)                # Zero-pad for the strip number
    plt.savefig(f"{save_file_name}_Strip{str_strip}.pdf", dpi=100)  # Save figure as PDF
    if not isbatchbool:  # If not in batch mode
        plt.show()       # Actually display the plot
    plt.clf()            # Clear the figure
    plt.close()          # Close the figure

###############################################################################
# FIND_PEAKS_50L
###############################################################################
def find_peak_range(data_1d, pk_indices):
    """
    Given 1D data and peak indices, find the left and right extents, top index,
    and integrated charge-like sum.
    """
    pk_start, pk_top, pk_end, pk_int = [], [], [], []   # Lists for start, top, end, integral
    n_samp = len(data_1d)                               # Number of samples
    for pk in pk_indices:
        temp_int = data_1d[pk]  # Start with the peak amplitude
        left = pk               # We'll move left
        while left > 0 and data_1d[left] > 0:
            left -= 1
            temp_int += data_1d[left]  # Accumulate for "integral"
        right = pk              # We'll move right
        while right < n_samp - 1 and data_1d[right] >= 0:
            right += 1
            temp_int += data_1d[right]
        pk_start.append(left)   # Record left bound
        pk_top.append(pk)       # Record peak (top) index
        pk_end.append(right)    # Record right bound
        pk_int.append(round(temp_int / 2.0, 1))  # Some integrated measure

    # Convert to NumPy arrays, ensuring int for indexing
    pk_start = np.array(pk_start, dtype=int)  # Start positions as int
    pk_top   = np.array(pk_top,   dtype=int)  # Peak positions as int
    pk_end   = np.array(pk_end,   dtype=int)  # End positions as int
    pk_int   = np.array(pk_int,   dtype=float)# Integral as float

    return (pk_start, pk_top, pk_end, pk_int)

def find_peaks_50l(data_1d, chn, peak_height, peak_width):
    """
    Finds peaks in data_1d, then filters out peaks that lie < 50 samples
    from a taller peak, returning only the major separated peaks.
    """
    pks, props = find_peaks(data_1d, height=peak_height, width=peak_width)  # Raw peaks from scipy

    filtered_pks = []                            # We'll store the good peaks
    filtered_props = {"peak_heights": []}        # And store their heights
    if pks.size > 0:
        sort_idx_desc = np.argsort(props["peak_heights"])[::-1]  # Descending amplitude
        for idx in sort_idx_desc:
            current_peak = pks[idx]
            if all(abs(current_peak - fp) > 50 for fp in filtered_pks):
                filtered_pks.append(current_peak)
                filtered_props["peak_heights"].append(props["peak_heights"][idx])

    filtered_pks = np.array(filtered_pks, dtype=int)              # Force integer indices
    filtered_props["peak_heights"] = np.array(filtered_props["peak_heights"], dtype=float)  # Heights as float

    pk_ranges = ([], [], [], [])
    if filtered_pks.size > 0:
        pk_ranges = find_peak_range(data_1d, filtered_pks)        # Get left, top, right, etc.

    return filtered_pks, filtered_props, pk_ranges

###############################################################################
# COSMIC DETECTION
###############################################################################
def detect_cosmic_line(adc, find_peaks_func, peak_height, peak_width, debug=False):
    """
    Calls check_peaks_per_channel(...) and detect_main_line(...) in sequence:
    1) check_peaks_per_channel => ensures no channel has close peaks or >2 tall peaks
    2) detect_main_line       => ensures there's a time-coherent line across the collection plane
    Returns True if cosmic is detected (i.e., these checks succeed).
    """
    if not check_peaks_per_channel(adc, find_peaks_func, peak_height, peak_width, debug=debug):
        return False  # If fails, not cosmic
    if not detect_main_line(adc, find_peaks_func, peak_height, peak_width, debug=debug):
        return False  # If fails, not cosmic
    return True        # If both pass => cosmic

def check_peaks_per_channel(adc, find_peaks_func, peak_height, peak_width, debug=False):
    """
    For each channel in the first NCC channels:
      - We do a raw find_peaks. If two peaks < WIDTH_THRESHOLD => cosmic
      - If >2 tall peaks above AMP_THRESHOLD => cosmic
    """
    from scipy.signal import find_peaks   # For raw peak detection
    for chn in range(NCC):                # Only check first NCC channels
        data_1d = adc[:, chn]
        pks, props = find_peaks(data_1d, height=peak_height, width=peak_width)

        if len(pks) > 1:
            pks_sorted = np.sort(pks)
            if np.any(np.diff(pks_sorted) < WIDTH_THRESHOLD):
                if debug:
                    print(f"{output_align}[DEBUG] ch{chn}: 2 close peaks => cosmic.")
                return False

        if len(pks) > 0 and "peak_heights" in props:
            n_big = sum(amp > AMP_THRESHOLD for amp in props["peak_heights"])
            if n_big > 2:
                if debug:
                    print(f"{output_align}[DEBUG] ch{chn}: >2 tall peaks => cosmic.")
                return False
    return True

def detect_main_line(adc, find_peaks_func, peak_height, peak_width, debug=False):
    """
    Ensures we have at least 20 channels with peaks >= AMP_THRESHOLD,
    and that no more than 5 channels are out-of-sync beyond time_tolerance=50.
    """
    time_tolerance = 50
    not_in_range = 0
    total_peaks = 0

    channel_peaks = []
    for chn in range(NCC):
        data_1d = adc[:, chn]
        pks, props, _ = find_peaks_func(data_1d, chn, peak_height, peak_width)

        best_time = None
        if len(pks) > 0 and "peak_heights" in props:
            idx_sort = np.argsort(props["peak_heights"])[::-1]
            for idx in idx_sort:
                if props["peak_heights"][idx] >= AMP_THRESHOLD:
                    best_time = pks[idx]
                    break
        if best_time is not None:
            channel_peaks.append(np.array([best_time], dtype=int))
            total_peaks += 1
        else:
            channel_peaks.append(np.array([], dtype=int))

    if total_peaks < 20:      # If fewer than 20 channels have a tall peak
        return False
    if channel_peaks[0].size == 0:  # If the first channel has no peak
        return False

    for chn in range(1, NCC):
        prev_pk = channel_peaks[chn - 1]
        curr_pk = channel_peaks[chn]
        if curr_pk.size > 0:
            t = curr_pk[0]
            if prev_pk.size == 0 or abs(t - prev_pk[0]) > time_tolerance:
                not_in_range += 1
            if not_in_range > 5:
                return False
    return True

###############################################################################
# SHOW_EVENT (2D)
###############################################################################
def show_event(adc, run_time, event_id,
               binary_file_id, binary_event_id,
               converted_file_id, converted_event_id,
               saveFileName, isbatchbool):
    """
    Produces a 2D pcolor(...) plot of the entire ADC array for this event.
    If isbatchbool is True, we skip plt.show().
    """
    plt.figure(figsize=(10, 6), dpi=100)
    plt.pcolor(adc, vmin=-100, vmax=100, cmap='YlGnBu_r')
    plt.colorbar(label="ADC")

    title_str = (f"{run_time} - Evt ID: {event_id} "
                 f"({binary_file_id},{binary_event_id}) "
                 f"({converted_file_id},{converted_event_id})")
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
# PREPARE_DATA (UPDATED)
###############################################################################
def prepare_data(equalize_bool,
                 filter_bool,
                 data_event_dict,
                 evt_title,
                 j_file_nr,
                 j_evt_nr,
                 base_output_name,
                 first_time=True,
                 input_evtdisplay_bool=True,
                 input_chndisplay_bool=True):
    """
    1) Build ADC array
    2) If 'filter_bool' => cosmic detection
    3) If event is NOT cosmic => produce per-channel plots
    """

    is_event_ok = True                             # By default, assume event is good
    is_data_dimension_incorrect = False            # For checking waveform lengths

    # Build the ADC array of shape (NTT, NC)
    adc = np.empty((NTT, NC))                      # Allocate an empty array
    for chn in range(NC):                          # Loop over each channel
        ch_key = f"chn{chn}"                       # Key in the dict: "chn0", "chn1", etc.
        wave = data_event_dict[ch_key]             # Retrieve the waveform
        if len(wave) != NTT:                       # If length is incorrect
            if first_time:
                print(f"{output_align}!! Event {j_evt_nr}: strip {chn} length {len(wave)} != {NTT}")
            is_data_dimension_incorrect = True
            break

        freq_val = most_frequent(wave)             # Baseline from most frequent value
        wave_bs = np.array(wave) - freq_val        # Subtract baseline
        if equalize_bool:
            eq_factors = ch_eq_year(evt_title[:4]) # Get equalization factors
            wave_bs = np.round(wave_bs * eq_factors[chn], decimals=1)

        adc[:, chn] = wave_bs                      # Fill into the ADC array

    if is_data_dimension_incorrect:
        return adc, False                          # Return (ADC, is_ok=False)

    # If user wants to filter => do cosmic detection
    if filter_bool:
        pk_width, pk_height, _ = get_cosmic_rejection_parameters(evt_title[:4])  # (5, 25, _)
        is_cosmic = detect_cosmic_line(
            adc,                    # The waveform data
            find_peaks_50l,        # The custom find_peaks function
            pk_height,             # e.g. 25
            pk_width,              # e.g. 5
            debug=first_time
        )
        if is_cosmic:
            if first_time:
                print(f"{output_align}[DEBUG] cosmic => adjacency checks => Evt({j_file_nr},{j_evt_nr}).")

            if input_evtdisplay_bool:
                # Show 2D plot for cosmic event if desired
                event_id          = data_event_dict.get("eventId", "???")
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
                    isbatchbool=(not input_evtdisplay_bool)
                )
            is_event_ok = False                    # Mark it as cosmic => exclude

    # Only if event is kept => produce single-channel plots
    if is_event_ok and input_chndisplay_bool:
        evt_subdir = f"{base_output_name}_evt{j_evt_nr}"      # Make a subfolder for this event
        Path(evt_subdir).mkdir(parents=True, exist_ok=True)   # Create the directory

        pk_width, pk_height, _ = get_cosmic_rejection_parameters(evt_title[:4])  # Reuse cosmic param for display
        for chn in range(NC):                                   # For each channel
            data_1d = adc[:, chn]
            pks, props, pk_ranges = find_peaks_50l(data_1d, chn, pk_height, pk_width)

            charge = 0   # placeholder for any integrated charge calc
            single_save_file_name = os.path.join(evt_subdir, f"Chn{chn}")
            nicer_single_evt_display(
                chn,
                data_1d,
                pks,
                save_file_name=single_save_file_name,
                evt_title=f"{evt_title}-File{j_file_nr}-Evt{j_evt_nr}",
                yrange=True,
                peak_ranges=pk_ranges,
                charge=charge,
                isbatchbool=(not input_evtdisplay_bool)
            )

    return adc, is_event_ok  # Return the ADC and whether the event is kept

###############################################################################
# READ_DATA
###############################################################################
def remove_coherent_noise(data_event_dict):
    """
    Placeholder function in case you'd like to remove or correct for coherent noise.
    Currently does nothing except return the input dict.
    """
    return data_event_dict

def read_data(input_evtdisplay_bool,
              input_chndisplay_bool,
              input_equalize_bool,
              input_filter_bool,
              raw_data_folder_name):
    """
    Reads all JSON files in DATA_FOLDER, processes each event,
    and prints out cosmic detection results plus optional plots.
    """
    path_json = DATA_FOLDER                     # Directory of JSON data
    Path(path_json).mkdir(parents=True, exist_ok=True)  # Ensure it exists (or create)

    print(f"\n[{find_time_now()}] > Read data starts")   # Log start
    print(f"{output_align} reading from => {path_json}") # Show the directory path

    allfiles = sorted(os.listdir(path_json))  # List all files, sorted
    evt_count = 0                             # Count total events processed
    file_count = 0                            # Count how many JSON files processed

    Path(PLOTS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)  # Ensure plot output directory exists

    for fname in allfiles:                                       # Iterate over files
        if not fname.lower().endswith(".json"):                  # Skip non-JSON
            continue
        if fname.startswith(IGNORED_FOLDER_PREFIX):              # Skip hidden files
            continue

        file_count += 1                                          # Increment file count
        fullpath = os.path.join(path_json, fname)                # Full path to the JSON
        print(f"{output_align}> Reading file {fname}")           # Log which file is read

        with open(fullpath, "r") as f:                           # Open the JSON
            data_all = json.load(f)                              # Load as Python dict
        events = data_all["all"]                                 # "all" key holds the events list

        for i, event_dict in enumerate(events):                  # Iterate over events in that file
            evt_count += 1                                       # Increment total event count
            event_nc = remove_coherent_noise(event_dict)         # (Optional) remove noise

            base_name  = fname[:-5]                              # Strip ".json" from filename
            base_outname = os.path.join(PLOTS_OUTPUT_DIR, base_name)  # e.g. "Plots/20230718_0001"

            first_time = (i == 0)  # True only for the first event in each file

            adc, is_ok = prepare_data(
                equalize_bool=input_equalize_bool,
                filter_bool=input_filter_bool,
                data_event_dict=event_nc,
                evt_title=raw_data_folder_name,
                j_file_nr=file_count,
                j_evt_nr=i,
                base_output_name=base_outname,
                first_time=first_time,
                input_evtdisplay_bool=input_evtdisplay_bool,
                input_chndisplay_bool=input_chndisplay_bool
            )

            if is_ok:
                print(f"{output_align}Event {i} => KEPT.")             # If not cosmic
            else:
                print(f"{output_align}Event {i} => EXCLUDED (cosmic).")# If cosmic

    return evt_count  # Return how many events were processed total

###############################################################################
# MAIN
###############################################################################
DEFAULTS = {
    "raw_data_folder_name": "20230717",  # Default folder name or date
    "input_evtdisplay_bool": True,       # Whether to show or save event-level 2D plots
    "input_chndisplay_bool": True,       # Whether to produce single-channel plots
    "input_equalize_bool": False,        # Whether to apply equalization factors
    "input_filter_bool": True            # Whether to do cosmic filtering
}

def main(raw_data_folder_name,
         input_evtdisplay_bool,
         input_chndisplay_bool,
         input_equalize_bool,
         input_filter_bool,
         terminal_bool):
    """
    Main entry point after CLI parsing. Reads data, processes events,
    and logs the final event count.
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
if __name__ == "__main__":  # If script is run directly
    raw_data_folder_, evt_disp_, chn_disp_, eq_bool_, filt_bool_, term_bool_ = parse_args(sys.argv, DEFAULTS)  # Parse CLI
    main(raw_data_folder_, evt_disp_, chn_disp_, eq_bool_, filt_bool_, term_bool_)  # Call main



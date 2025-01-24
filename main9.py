#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:35:56 2025

@author: alexlupu
"""

"""
Full code: 
 - Phase A: scans all JSON, detects cosmic, saves cosmic events to cosmic_events.csv
   with columns [#Folder_name, EvtID, Raw_file_nr, Raw_Evt_nr, File_nr, Evt_nr].
 - Phase B: re-opens cosmic_events.csv, reads those columns (File_nr, Raw_Evt_nr, etc.)
   => single_event(...) channel plotting.
 - find_charge_cluster(...) no longer returns 0 if overshadowed,
   it just skips adding neighbor's partial charge.
 - If two close peaks are found, we also print EvtID in the debug message.

Additionally:
 - store_cosmic_charges_from_csv() re-opens cosmic_events.csv,
   for each row => re-opens the JSON => compute_channel_charges(...),
   writing results to cosmic_charges.csv.

We have changed the legend label to "ADC*tick" in nicer_single_evt_display(...)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from scipy.signal import find_peaks

###############################################################################
# SETTINGS & HARD-CODED PATH
###############################################################################
NTT = 645
NC  = 128
NCC = 48
AMP_THRESHOLD         = 18
WIDTH_SPACE_THRESHOLD = 40
WIDTH_PEAK_THRESHOLD  = 5
IGNORED_FOLDER_PREFIX = "."

output_align = "[ANALYSIS]"

DATA_FOLDER       = "/Users/alexlupu/github/dune-apex-umn/DATA/20230722/jsonData/"
PLOTS_OUTPUT_DIR  = "/Users/alexlupu/github/dune-apex-umn/Plots"
COSMIC_CSV_PATH   = "/Users/alexlupu/github/dune-apex-umn/cosmic_events.csv"
COSMIC_CHARGES_CSV_PATH = "/Users/alexlupu/github/dune-apex-umn/cosmic_charges.csv"

plt.rcParams.update({'font.size': 20})

###############################################################################
# DEFAULTS FOR MAIN
###############################################################################
DEFAULTS = {
    "raw_data_folder_name": "20230722",
    "input_evtdisplay_bool": True,
    "input_chndisplay_bool": True,
    "input_equalize_bool": False,
    "input_filter_bool": True
}

###############################################################################
# UTILS
###############################################################################
def parse_args(argv, defaults):
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
        d["raw_data_folder_name"],
        d["input_evtdisplay_bool"],
        d["input_chndisplay_bool"],
        d["input_equalize_bool"],
        d["input_filter_bool"],
        True
    )

def find_time_now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def most_frequent(arr):
    c = Counter(arr)
    return c.most_common(1)[0][0]

###############################################################################
# COSMIC DETECTION PARAMETERS
###############################################################################
def get_cosmic_rejection_parameters(year_str):
    """
    Typically we slice the first 4 characters if year_str is a string, e.g. "2023".
    Returns (peak_width, peak_height, min_strips).
    """
    return (5, AMP_THRESHOLD, NCC - 1)

def ch_eq_year(year_str):
    # placeholder => no real calibration
    return np.ones(NC, dtype=float)

###############################################################################
# PEAK-FINDING
###############################################################################
def find_peak_range(data_1d, pk_indices):
    # Initialize four empty lists for storing peak boundaries and integrals
    pk_start, pk_top, pk_end, pk_int = [], [], [], []

    # Get the number of samples in data_1d (makes function flexible, not tied to NTT)
    n_samp = len(data_1d)                               

    # Loop over each peak index in pk_indices
    for pk in pk_indices:
        
        # Start the integral accumulator with the value at the peak center
        temp_int = data_1d[pk]
        
        # Set 'left' pointer to the peak index, then move left while data > 0
        left = pk
        while left > 0 and data_1d[left] > 0:
            left -= 1
            temp_int += data_1d[left]
        
        # Set 'right' pointer to the peak index, then move right while data >= 0
        right = pk
        while right < n_samp - 1 and data_1d[right] >= 0:
            right += 1
            temp_int += data_1d[right]
        
        # Append the computed boundaries (left, pk, right) to corresponding lists
        pk_start.append(left)
        pk_top.append(pk)
        pk_end.append(right)
        
        # Save the integral of the region; here we do no time conversion => ADC*ticks
            #pk_int.append(round(temp_int / 2.0, 1)) # this would give you ADC*us
        pk_int.append(round(temp_int, 1))               

    # Return four parallel lists: start, top, end, and integral for each peak
    return (pk_start, pk_top, pk_end, pk_int)


def find_peaks_50l(data_1d, chn, peak_height, peak_width):
    # 1) Use scipy.signal.find_peaks to get candidate peaks
    pks, props = find_peaks(data_1d, height=peak_height, width=peak_width)
    
    # 2) Initialize containers for filtered peak indices and properties
    filtered_pks = []
    filtered_props = {"peak_heights": []}
    
    # 3) If we found at least one peak, proceed
    if pks.size > 0:
        # Sort peak indices by descending height (props["peak_heights"])
        sort_idx_desc = np.argsort(props["peak_heights"])[::-1]
        
        # 4) Loop over peak indices in order of descending height
        for idx in sort_idx_desc:
            current_peak = pks[idx]
            
            # 5) Check if current_peak is at least AMP_THRESHOLD samples away
            #    from each peak already accepted into filtered_pks
            if all(abs(current_peak - fp) > AMP_THRESHOLD for fp in filtered_pks):
                # 6) If sufficiently far from every previously accepted peak,
                #    append it to filtered_pks and record its height
                filtered_pks.append(current_peak)
                filtered_props["peak_heights"].append(props["peak_heights"][idx])

    # 7) Convert filtered_pks and filtered_props["peak_heights"] to NumPy arrays
    filtered_pks = np.array(filtered_pks)
    filtered_props["peak_heights"] = np.array(filtered_props["peak_heights"])
    
    # 8) pk_ranges is initially empty
    pk_ranges = ([], [], [], [])
    
    # 9) If we have any filtered peaks, compute their (start, top, end, integral)
    #    by calling find_peak_range(...)
    if filtered_pks.size > 0:
        pk_ranges = find_peak_range(data_1d, filtered_pks)

    # 10) Return the final filtered peaks, their properties, and the ranges
    return filtered_pks, filtered_props, pk_ranges

###############################################################################
# COSMIC DETECTION
###############################################################################
def detect_cosmic_line(adc, find_peaks_func, peak_height, peak_width, debug=False, event_id=None):
    # 1) First, call check_peaks_per_channel(...) to ensure no channel has invalid or suspicious multiple peaks.
    if not check_peaks_per_channel(adc, find_peaks_func, peak_height, peak_width,
                                   debug=debug, event_id=event_id):
        # If that check fails, return False immediately => "not cosmic"
        return False

    # 2) Next, call detect_main_line(...) to ensure there's a coherent time alignment across multiple channels (a "line" in time).
    if not detect_main_line(adc, find_peaks_func, peak_height, peak_width, debug=debug):
        # If that check fails, return False => "not cosmic"
        return False

    # 3) If both checks pass, return True => "cosmic"
    return True


def check_peaks_per_channel(adc, find_peaks_func, peak_height, peak_width, debug=False, event_id=None):
    from scipy.signal import find_peaks
    
    # 1) Iterate over channels in the "collection-plane" subset (range(NCC))
    for chn in range(NCC):
#???? What does that mean to be a 1D waveform?
        # 2) Extract the 1D waveform for this channel from the adc array
        data = adc[:, chn]
        
        # 3) Use find_peaks(...) to detect candidate peaks, with the specified height/width thresholds
        pks, props = find_peaks(data, height=peak_height, width=peak_width)

        # 4) If we detect more than one peak, check if any are too close (< WIDTH_SPACE_THRESHOLD)
        if len(pks) > 1:
            # Sort peak indices in ascending order
            pks_sorted = np.sort(pks)
            # Check differences between adjacent peaks
            if np.any(np.diff(pks_sorted) < WIDTH_SPACE_THRESHOLD):
                # 5) If debug mode is on, print a diagnostic message indicating event/channel info
                if debug:
                    print(f"{output_align} EvtID {event_id}, ch{chn}: 2 peaks < {WIDTH_SPACE_THRESHOLD}")
                # 6) Return False => "channel fails" => not cosmic
                return False
        
        # 7) If the desired peak width is bigger than WIDTH_PEAK_THRESHOLD, fail immediately
        if peak_width > WIDTH_PEAK_THRESHOLD:
            if debug:
                print(f"{output_align}[DEBUG] ch{chn}: peak_width > {WIDTH_PEAK_THRESHOLD}")
            return False
    
    # 8) If all channels pass (no too-close peaks, no excessive width), return True
    return True


def detect_main_line(adc, find_peaks_func, peak_height, peak_width, debug=False):
    # 1) Set up a tolerance for time alignment and counters
    time_tolerance = 50       # maximum allowed difference in peak time
    not_in_range = 0          # counts how many channels are out of alignment
    total_peaks = 0           # counts how many channels actually have a peak

    # 2) channel_peaks will hold, for each channel, the best peak’s time index (or an empty array)
    channel_peaks = []

    # 3) Loop over each collection-plane channel (0..NCC-1)
    for chn in range(NCC):
        data = adc[:, chn]
        # call find_peaks_func(...) which returns (peaks, properties, pk_ranges)
        pks, props, _ = find_peaks_func(data, chn, peak_height, peak_width)

        # We'll pick "best_time" = the time index of the tallest valid peak
        best_time = None
        
        # 4) If we found any peaks (pks) and we have "peak_heights" in props
        if len(pks) > 0 and "peak_heights" in props:
            # Sort the peaks by descending height
            idx_sort = np.argsort(props["peak_heights"])[::-1]
            
            # 5) Loop from tallest to smallest until we find one >= AMP_THRESHOLD
            for idx in idx_sort:
                if props["peak_heights"][idx] >= AMP_THRESHOLD:
                    best_time = pks[idx]
                    break

        # 6) If we found a "best_time", store it in channel_peaks; increment total_peaks
        if best_time is not None:
            channel_peaks.append(np.array([best_time]))
            total_peaks += 1
        else:
            # If no valid peak, store an empty array
            channel_peaks.append(np.array([]))

    # 7) If fewer than (NCC - 1) channels have a valid peak, fail => not cosmic
    if total_peaks < NCC - 1:
        return False

    # 8) Check alignment across channels
    #    We iterate from channel 1..NCC-1 and compare to the previous channel
    for chn in range(1, NCC):
        prev_pk = channel_peaks[chn - 1]
        curr_pk = channel_peaks[chn]
        
        # 9) If the current channel has a peak, check its alignment with the previous channel's peak
        if len(curr_pk) > 0:
            t = curr_pk[0]
            # If the previous channel has no peak or the time difference is > time_tolerance,
            # increment not_in_range
            if len(prev_pk) == 0 or abs(t - prev_pk[0]) > time_tolerance:
                not_in_range += 1
            
            # If more than 5 channels are out of sync => fail
            if not_in_range > 5:
                return False

    # 10) If we never exceeded out-of-range threshold => pass => True
    return True


###############################################################################
# SHOW_EVENT (2D event-level)
###############################################################################
def show_event(adc, run_time, event_id,
               binary_file_id, binary_event_id,
               converted_file_id, converted_event_id,
               saveFileName, isbatchbool):
    import csv
    
    # 1) Build a descriptive title string, incorporating event IDs and filenames
    title_str = (f"{run_time} - Evt ID: {event_id} "
                 f"({binary_file_id},{binary_event_id}) "
                 f"({converted_file_id},{converted_event_id})")

    # 2) Define a directory and file path where we track "valid-cosmic-rays" metadata
    csv_file_dir = "/Users/alexlupu/github/dune-apex-umn/get-started-2025/valid-cosmic-rays"
    Path(csv_file_dir).mkdir(parents=True, exist_ok=True)
    csv_file_path = os.path.join(csv_file_dir, "plot_names_20230725.csv")

    # 3) Check if the CSV file already exists
    file_exists = os.path.exists(csv_file_path)
    
    # 4) Open that CSV file in append mode, so we can add a row if we haven't already
    with open(csv_file_path, "a", newline='') as csvf:
        writer = csv.writer(csvf)
        
        # 5) If the file didn't exist previously, write a header row
        if not file_exists:
            writer.writerow(["#Folder_name", "EvtID", "Raw_file_nr", "Raw_Evt_nr", "File_nr", "Evt_nr"])
        
        # 6) Write one row of metadata about this event:
        #    e.g. run_time, event_id, binary_file_id, binary_event_id, etc.
        writer.writerow([
            run_time,
            event_id,
            binary_file_id,
            binary_event_id,
            converted_file_id,
            converted_event_id
        ])

    # 7) Create a new figure (size 10" x 6") with 100 DPI
    plt.figure(figsize=(10, 6), dpi=100)
    
    # 8) Produce a 2D color plot (pcolor) of the ADC array
    #    vmin=-100, vmax=100 sets the color scale range;
    #    'YlGnBu_r' is a reverse “yellow-green-blue” colormap
    plt.pcolor(adc, vmin=-100, vmax=100, cmap='YlGnBu_r')
    
    # 9) Add a colorbar on the side, labeling it "ADC"
    plt.colorbar(label="ADC")
    
    # 10) Set the plot's title to the descriptive string from earlier
    plt.title(title_str)
    plt.xlabel("Strips")
    plt.ylabel("Time ticks [0.5 µs/tick]")
    
    # 11) Adjust x-ticks: place them at intervals of 10 columns, rotate them 45° for readability
    plt.xticks(np.arange(0, adc.shape[1] + 1, 10), rotation=45)
    
    # 12) Use tight_layout() to reduce excess white space and avoid label overlap
    plt.tight_layout()

    # 13) Save the figure to 'saveFileName' with bounding box = 'tight' 
    #     so it trims extra margins
    plt.savefig(saveFileName, bbox_inches='tight')
    
    # 14) If not in batch mode, display the figure interactively
    if not isbatchbool:
        plt.show()
    
    # 15) Clear the current figure to free up memory
    plt.clf()
    
    # 16) Close the figure completely
    plt.close()


###############################################################################
# CLUSTERING LOGIC (NO ZERO RETURN)
###############################################################################
def find_charge_cluster(adc, chn_c, peak_charge, r, s, candidate_event_C):
    # 1) Initialize the cluster charge to the main peak’s charge
    peak_charge_cluster = peak_charge
    
    # 2) Flag to indicate if a neighbor’s bigger peak overshadowed this one
    overshadowed = False

    # 3) If the channel index is outside the "collection-plane" range, just return what we have
    if chn_c >= NCC:
        return round(peak_charge_cluster, 1)

    # 4) Only proceed if 'chn_c' is strictly less than (NCC - 1), 
    #    so we have a right neighbor
    if chn_c < (NCC - 1):

        # 5) We'll look at the channel to the immediate right
        c_strip_right = chn_c + 1
        charge_right = 0

        # 6) If 'r[c_strip_right]' is empty, there's no peak info there => skip
        if len(r[c_strip_right]) == 0:
            pass
        else:
            # 7) Loop over all possible peaks in that right channel
            for c_strip_right_event in range(len(s[c_strip_right])):
#????
                # 8) Check if the right channel’s peak (top index) is within 
                #    the left channel’s [start, end) range in time
                if (r[c_strip_right][0][1][c_strip_right_event] 
                        > r[chn_c][0][0][candidate_event_C] 
                    and r[c_strip_right][0][1][c_strip_right_event] 
                        < r[chn_c][0][2][candidate_event_C]):

                    # 9) If right neighbor’s peak integral is bigger, set overshadowed = True
                    if (r[c_strip_right][0][3][c_strip_right_event] 
                            > r[chn_c][0][3][candidate_event_C]):
                        overshadowed = True
                        break
                else:
                    # 10) If it doesn’t overlap or overshadow, sum up the ADC from that 
                    #     neighbor’s strip in the [start, end) range
                    for tick in range(r[chn_c][0][0][candidate_event_C],
                                      r[chn_c][0][2][candidate_event_C]):
                        if tick < adc.shape[0] and c_strip_right < adc.shape[1]:
                            charge_right += adc[tick, c_strip_right]
#???? How does this help avoid double counting?
            # 11) If not overshadowed and we found any positive neighbor charge, 
            #     add half of it to the cluster
            if (not overshadowed) and (charge_right > 0):
                charge_right /= 2.0
                peak_charge_cluster += charge_right

        # 12) If still not overshadowed, try the left neighbor (chn_c - 1), if it exists
        if not overshadowed and chn_c > 0:
            c_strip_left = chn_c - 1
            charge_left = 0

            # 13) Sum the left neighbor’s ADC in the same [start, end) time range
            for tick in range(r[chn_c][0][0][candidate_event_C],
                              r[chn_c][0][2][candidate_event_C]):
                if tick < adc.shape[0] and c_strip_left < adc.shape[1]:
                    charge_left += adc[tick, c_strip_left]

            # 14) If positive, add half that left neighbor’s charge
            if charge_left > 0:
                charge_left /= 2.0
                peak_charge_cluster += charge_left

    # 15) Return the total cluster charge, rounded
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
    
    # 1) Initialize flags controlling data validity and cosmic status
    is_event_ok = True
    is_data_dimension_incorrect = False

    # 2) Create an empty 2D NumPy array for the waveforms
    #    shape (NTT, NC) => NTT time samples, NC channels
    adc = np.empty((NTT, NC))
#???? What is the waveform?
    # 3) Loop over all channels from 0..NC-1
    for chn in range(NC):
        # Extract the waveform list for channel chn from data_event_dict
        wave = data_event_dict.get(f"chn{chn}", [])
        
        # 4) Check if the waveform length matches the expected NTT
        if len(wave) != NTT:
            # If it’s the first time we see a mismatch, optionally print a warning
            if first_time:
                print(f"{output_align}!! Event {j_evt_nr}: strip {chn} length {len(wave)} != {NTT}")
            is_data_dimension_incorrect = True
            break
        
        # 5) Compute the baseline using the most frequent sample value
        freq_val = most_frequent(wave)
        
        # 6) Convert wave to a NumPy array, subtract baseline
        wave_bs  = np.array(wave) - freq_val
#????
        # 7) If equalization is enabled, multiply by a channel-specific scale factor
        if equalize_bool:
            eq_factors = ch_eq_year(str(evt_title)[:4])
            wave_bs = np.round(wave_bs * eq_factors[chn], decimals=1)

        # 8) Store the processed waveform in column chn of adc
        adc[:, chn] = wave_bs

    # 9) If any channel had the wrong length, return (adc, False) => “invalid event”
    if is_data_dimension_incorrect:
        return adc, False

    # 10) If no filtering is requested, just return (adc, True)
    if not filter_bool:
        return adc, True

    # 11) Otherwise, we proceed to cosmic detection:
    current_event_id = data_event_dict.get("eventId", "???")
    
    # 12) Retrieve cosmic detection parameters (peak_width, peak_height, etc.) based on evt_title
    pk_width, pk_height, _ = get_cosmic_rejection_parameters(str(evt_title)[:4])

    # 13) Call detect_cosmic_line(...) to see if this event is cosmic
    is_cosmic = detect_cosmic_line(
        adc,
        find_peaks_50l,
        pk_height,
        pk_width,
        debug=first_time,
        event_id=current_event_id
    )

    # 14) If cosmic, optionally create a 2D color plot (and mark is_event_ok as False)
    if is_cosmic:
        if first_time:
            print(f"{output_align}[DEBUG] cosmic => adjacency => Evt({j_file_nr},{j_evt_nr}).")

        # 15) If input_evtdisplay_bool is True, generate a 2D event-level plot
        if input_evtdisplay_bool:
            event_id          = current_event_id
            binary_file_id    = data_event_dict.get("binaryFileID", "???")
            binary_event_id   = data_event_dict.get("binaryEventID", "???")
            converted_file_id = data_event_dict.get("convertedFileID", "???")
            converted_event_id= data_event_dict.get("convertedEventID", "???")

            # Build the PDF filename based on file/event indices
            save_file_name = f"{base_output_name}_file{j_file_nr}_evt{j_evt_nr}.pdf"
            
            # Call show_event(...) to produce and optionally show the plot
            show_event(
                adc,
                str(evt_title),
                event_id,
                binary_file_id,
                binary_event_id,
                converted_file_id,
                converted_event_id,
                save_file_name,
                isbatchbool=False
            )
        
        # 16) Mark the event as "not OK" => meaning it’s recognized as cosmic
        #   ChatGPT says this to "exclude" the file, but it saves the file to a different folder
        is_event_ok = False

    # 17) Return the ADC array and a boolean indicating if this event is “OK” (non-cosmic)
    return adc, is_event_ok


###############################################################################
# PHASE A: SCAN & SAVE COSMIC EVENTS
###############################################################################
def scan_and_save_cosmic_events(raw_data_folder_name,
                                input_evtdisplay_bool,
                                input_chndisplay_bool,
                                input_equalize_bool,
                                input_filter_bool):

    # 1) path_json is the directory where JSON files are located.
    #    We ensure that directory exists.
    path_json = DATA_FOLDER
    Path(path_json).mkdir(parents=True, exist_ok=True)

    # 2) Initialize an empty list to collect rows of cosmic event data
    cosmic_rows = []

    # 3) List all files in path_json, sort them, initialize counters
    allfiles = sorted(os.listdir(path_json))
    file_count = 0   # how many JSON files processed
    evt_count = 0    # how many events processed overall

    # 4) Print a timestamp/log message that we’re starting to scan
    print(f"\n[{find_time_now()}] > Start scanning => {COSMIC_CSV_PATH}")

    # 5) Loop over each filename in the sorted list
    for fname in allfiles:
        # 5a) Skip files that aren’t .json or that start with an ignored prefix
        if not fname.lower().endswith(".json"):
            continue
        if fname.startswith(IGNORED_FOLDER_PREFIX):
            continue

        # 5b) Increment the JSON file counter
        file_count += 1

        # 5c) Build the full path to this JSON file
        fullpath = os.path.join(path_json, fname)

        # 6) Open and load the JSON data
        with open(fullpath, "r") as f:
            data_all = json.load(f)
        events = data_all["all"]  # Typically a list of event dictionaries

        # 7) Create a directory for potential output plots if needed
        base_out_dir = PLOTS_OUTPUT_DIR
        Path(base_out_dir).mkdir(parents=True, exist_ok=True)

        # 8) Prepare a base name for this file (minus ".json") and build an output path
        base_name   = fname[:-5]
        base_outname= os.path.join(base_out_dir, base_name)

        # 9) Loop over each event in the JSON “events” list
        for i, event_dict in enumerate(events):
            
            # 10) Call prepare_data(...) to build the ADC array and see if it’s cosmic
            adc, is_ok = prepare_data(
                input_equalize_bool,
                input_filter_bool,
                event_dict,
                raw_data_folder_name,
                file_count,
                i,
                base_outname,
                first_time=(i==0),                 # True if it’s the first event of this file
                input_evtdisplay_bool=input_evtdisplay_bool
            )

            # 11) If the event is cosmic (is_ok == False), store metadata in cosmic_rows
            if not is_ok:
                event_id      = event_dict.get("eventId", "???")
                binary_evt_id = event_dict.get("binaryEventID", "???")

                cosmic_rows.append({
                    "#Folder_name": raw_data_folder_name, # e.g., “20230722”
                    "EvtID":        event_id,
                    "Raw_file_nr":  file_count,           # which file we’re on
                    "Raw_Evt_nr":   i,                    # which event within that file
                    "File_nr":      fname,                # the JSON filename
                    "Evt_nr":       binary_evt_id         # possibly the “binaryEventID”
                })

    # 12) After processing all files and events, log a “done” message
    print(f"{output_align} Done scanning => {COSMIC_CSV_PATH}")

    # 13) If we collected any cosmic events in cosmic_rows, we append them to cosmic_events.csv
    if cosmic_rows:
        cosmic_df = pd.DataFrame(cosmic_rows)
        # reorder columns so they appear in a consistent order
        cosmic_df = cosmic_df[
            ["#Folder_name","EvtID","Raw_file_nr","Raw_Evt_nr","File_nr","Evt_nr"]
        ]

        # 14) Check if cosmic_events.csv already exists, so we know if we should write a header
        csv_exists = os.path.exists(COSMIC_CSV_PATH)

        # 15) Append the new cosmic rows to cosmic_events.csv
        cosmic_df.to_csv(
            COSMIC_CSV_PATH,
            mode='a',        # append mode
            index=False,
            header=not csv_exists
        )
    else:
        # 16) If no cosmic events were found at all, print a message
        print(f"{output_align} No cosmic events found; nothing written to {COSMIC_CSV_PATH}.")


###############################################################################
# CHANNEL-CHARGE CALCULATION (SHARED)
###############################################################################
def compute_channel_charges(adc, evt_title):
    """
    Finds largest peak for each channel, merges adjacency => returns dict of:
    results[chn] = (largest_pk_idx, cluster_charge).
    Also returns (r, s) for advanced usage (peak highlighting).
    """
    
    # 1) Convert evt_title to string, slice out the first 4 chars to get a year, 
    #    then call get_cosmic_rejection_parameters(...) to retrieve pk_width, pk_height
    year_str = str(evt_title)
    pk_width, pk_height, _ = get_cosmic_rejection_parameters(year_str[:4])
#???? How is "s" useful?
    # 2) Prepare two lists-of-lists:
    #    - r[chn] will hold arrays of [pk_start, pk_top, pk_end, pk_int]
    #    - s[chn] tracks how many peaks were found in channel chn
    r = [[] for _ in range(NC)]
    s = [[] for _ in range(NC)]
#???? This should be changed to NCC?
    # 3) Loop over all channels in the ADC array
    for chn in range(NC):
        # Extract the 1D waveform for channel 'chn'
        data_1d = adc[:, chn]
        
        # 4) Call find_peaks_50l(...) to detect peaks for this channel
        pks, props, pk_ranges = find_peaks_50l(data_1d, chn, pk_height, pk_width)
        
        # 5) pk_ranges is a tuple (pk_start, pk_top, pk_end, pk_int) for the peaks
        (pk_start, pk_top, pk_end, pk_int) = pk_ranges
        
        # 6) Store that peak info in r[chn][0], so we can use it for adjacency logic
        r[chn].append([pk_start, pk_top, pk_end, pk_int])
        
        # 7) s[chn] is set to range(len(pk_start)), effectively storing the count of peaks
        s[chn] = range(len(pk_start))

    # 8) Prepare 'results' as a dictionary: results[chn] = (largest_pk_idx, cluster_charge)
    results = {}

    # 9) Loop again over each channel to find the largest peak and compute its cluster
    for chn in range(NC):
        # If no peaks were found in channel chn, set largest_pk_idx=None, cluster_charge=0
        if len(r[chn]) == 0 or len(r[chn][0][0]) == 0:
            largest_pk_idx = None
            cluster_charge = 0
        else:
            # Extract arrays of start, top, end, integral for all peaks in channel 'chn'
            pk_start = r[chn][0][0]
            pk_top   = r[chn][0][1]
            pk_end   = r[chn][0][2]
            pk_int   = r[chn][0][3]

            # idx_max => the index of the peak with the largest integral
            idx_max = np.argmax(pk_int)
            peak_charge   = pk_int[idx_max]

            # 10) Call find_charge_cluster(...) to possibly merge neighbor charge
            cluster_charge= find_charge_cluster(adc, chn, peak_charge, r, s, idx_max)

            # 11) The largest peak’s time index
            largest_pk_idx= pk_top[idx_max]

        # 12) Store the final (largest peak index, cluster charge) in the dictionary
        results[chn] = (largest_pk_idx, cluster_charge)

    # 13) Return both the 'results' dictionary and the (r, s) data structure
    #     which can be used for plotting or advanced logic
    return results, (r, s)


###############################################################################
# PHASE B: READ CSV => PLOT COSMIC EVENTS
###############################################################################
def plot_cosmic_events_from_csv():
    # 1) Check if cosmic_events.csv exists; if not, there's nothing to plot
    if not os.path.exists(COSMIC_CSV_PATH):
        print(f"{output_align} No CSV found at {COSMIC_CSV_PATH}. Nothing to plot.")
        return

    # 2) Read cosmic_events.csv into a pandas DataFrame, then convert to a list of dicts
    cosmic_df = pd.read_csv(COSMIC_CSV_PATH)
    rows = cosmic_df.to_dict(orient="records")

    # 3) Print how many cosmic events were found in the CSV
    cosmic_count = len(rows)
    print(f"{output_align} Found {cosmic_count} cosmic events in {COSMIC_CSV_PATH}. Plotting...")

    # 4) Loop over each row in cosmic_events.csv
    for row in rows:
        # Retrieve necessary metadata from the row
        folder_name  = row["#Folder_name"]
        evt_id_str   = row["EvtID"]
        raw_file_nr  = row["Raw_file_nr"]
        raw_evt_nr   = row["Raw_Evt_nr"]
        json_fname   = row["File_nr"]
        binary_evt   = row["Evt_nr"]

        # 5) Build the path to the original JSON file
        fullpath = os.path.join(DATA_FOLDER, json_fname)
        
        # 6) Check if the file actually exists; if not, skip
        if not os.path.exists(fullpath):
            print(f"{output_align} Missing file {fullpath}; skip.")
            continue

        # 7) Re-open the JSON, grab all events
        with open(fullpath, "r") as f:
            data_all = json.load(f)
        events = data_all["all"]

        # 8) Convert raw_evt_nr to an integer => index of the event in `events`
        raw_evt_idx = int(raw_evt_nr)
        if raw_evt_idx < 0 or raw_evt_idx >= len(events):
            print(f"{output_align} Invalid event_idx {raw_evt_idx} in {json_fname}; skip.")
            continue

        # 9) Retrieve the specific event dictionary
        event_dict = events[raw_evt_idx]

        # 10) Build an empty ADC array, shape (NTT, NC). We'll fill it with waveforms
        adc = np.empty((NTT, NC))
        is_bad_dim = False

        # 11) Loop over all channels, extract wave data, do baseline subtraction
        for chn in range(NC):
            wave = event_dict.get(f"chn{chn}", [])
            # If any channel’s wave doesn't match NTT in length => mark bad dimension
            if len(wave) != NTT:
                is_bad_dim = True
                break
            freq_val = most_frequent(wave)
            wave_bs  = np.array(wave) - freq_val
            adc[:, chn] = wave_bs

        # 12) If we found a dimension mismatch in any channel => skip this event
        if is_bad_dim:
            print(f"{output_align} Bad wave size in {json_fname} Evt {raw_evt_idx}; skip.")
            continue

        # 13) Build a label for logging/plot usage, define an output subdirectory
        cosmic_label = f"EvtID {evt_id_str} (COSMIC) => {json_fname}"
        out_subdir   = os.path.join(PLOTS_OUTPUT_DIR, "cosmic_from_csv", f"{json_fname}_Evt{raw_evt_idx}")

        # 14) Finally, call single_event(...) => produce channel-level plots for this cosmic event
        single_event(adc, str(folder_name), out_subdir)

    # 15) After looping over all rows, print a completion message
    print(f"{output_align} Done plotting cosmic events from CSV.")


###############################################################################
# SINGLE-EVENT CHANNEL PLOTTING HELPER
###############################################################################
def nicer_single_evt_display(chn, y, peaks, save_file_name, evt_title, yrange, peak_ranges, charge, isbatchbool):
    # 1) Create a large figure and axes (32" wide x 9" tall)
    fig, ax = plt.subplots(figsize=(32, 9))
    
    # 2) Convert the channel number to a zero-padded string, e.g. "007"
    str_chn = str(chn).zfill(3)
    
    # 3) Build a legend label that shows channel and the computed cluster charge
    label = f"Chn_{str_chn}; Charge cluster: {charge} ADC*tick"

    # 4) Ensure 'peaks' is an integer array, so indexing is valid
    peaks = peaks.astype(int)
    
    # 5) Plot the entire waveform 'y' with the label
    ax.plot(y, label=label)
    
    # 6) If we have any peak indices, mark them on the plot with an "x"
    if len(peaks) > 0:
        ax.plot(peaks, y[peaks], "x")

    # 7) If we have a well-structured peak_ranges (start, top, end, integral),
    #    fill between 0 and the waveform from pk_start..pk_end for each peak
    if len(peak_ranges) == 4 and len(peak_ranges[0]) == len(peak_ranges[2]):
        pk_start = np.array(peak_ranges[0], dtype=int)
        pk_end   = np.array(peak_ranges[2], dtype=int)
        
        # 8) Loop over each start/end pair, fill under the curve in gold
        for s_tick, e_tick in zip(pk_start, pk_end):
            region_x = np.arange(s_tick, e_tick)
            region_y = y[region_x]
            # Only fill for "collection-plane" channels (chn < NCC)
            if chn < NCC:
                ax.fill_between(region_x, 0, region_y, color='gold', alpha=0.5)

    # 9) Set axis labels and legend
    ax.set_ylabel('ADC (baseline-subtracted)', labelpad=10, fontsize=24)
    ax.set_xlabel('time ticks [0.5 µs/tick]', fontsize=24)
    ax.legend(fontsize=24)
    
    # 10) Enable a grid for clarity (major and minor tick lines)
    ax.grid(True, which='major', axis='both', linewidth=1, color='gray')
    ax.grid(True, which='minor', axis='x', linewidth=0.5, linestyle='dashed', color='gray')

    # 11) Set major x-ticks every 50 samples, minor x-ticks every 10
    ax.set_xticks(np.arange(0, NTT + 1, 50))
    ax.set_xticks(np.arange(0, NTT + 1, 10), minor=True)

    # 12) If yrange is requested, apply different y-limits depending on channel range
    if yrange:
        if chn < NCC:
            ax.set_ylim(-20, 300)
        else:
            ax.set_ylim(-250, 250)

    # 13) Set the x-axis range to [0..len(y)], remove the right spine
    ax.set_xlim(0, len(y))
    ax.yaxis.set_label_coords(-.04, 0.5)
    ax.spines['right'].set_visible(False)

    # 14) Set the figure title to something like "20230722 - Strip 005"
    plt.title(f"{evt_title} - Strip {chn + 1}", fontsize=30)

    # 15) Adjust subplot boundaries and use tight_layout() to prevent overlapping
    plt.subplots_adjust(left=0.06, right=0.94, top=0.96, bottom=0.08)
    plt.tight_layout()

    # 16) Construct an output filename, e.g. "myPlot_Strip005.pdf"
    str_strip = str(chn + 1).zfill(3)
    out_fname = f"{save_file_name}_Strip{str_strip}.pdf"
    
    # 17) Save the figure with 100 DPI
    plt.savefig(out_fname, dpi=100)
    
    # 18) If we’re not in batch mode, display the figure on screen
    if not isbatchbool:
        plt.show()
    
    # 19) Clear the figure from memory, and fully close it
    plt.clf()
    plt.close()

def single_event(adc, evt_title, out_dir):
    # 1) Ensure the output directory 'out_dir' exists, creating it if necessary
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 2) Compute channel-level charges (and additional peak info) for this event’s 'adc'
    #    - 'channel_info' is a dict: channel_info[chn] = (largest_pk_idx, cluster_charge)
    #    - 'r' holds arrays of [pk_start, pk_top, pk_end, pk_int] for each channel
    #    - 's' holds the number of peaks found per channel
    channel_info, (r, s) = compute_channel_charges(adc, str(evt_title))

    # 3) Loop over each channel in the ADC array
    for chn in range(NC):

        # Extract the 1D waveform for this channel
        data_1d = adc[:, chn]

        # 4) Retrieve the largest peak’s index and the computed cluster charge
        largest_pk_idx, cluster_charge = channel_info[chn]

        # 5) If no largest peak or if the array 'r[chn]' is empty => no peaks => plot empty
        if largest_pk_idx is None or len(r[chn]) == 0:
            out_prefix = os.path.join(out_dir, f"Chn{chn}")
            # Call nicer_single_evt_display(...) with an empty 'peaks' array
            nicer_single_evt_display(
                chn=chn,
                y=data_1d,
                peaks=np.array([], dtype=int),
                save_file_name=out_prefix,
                evt_title=str(evt_title),
                yrange=True,
                peak_ranges=([], [], [], []),
                charge=0,
                isbatchbool=True
            )
            # Move to the next channel
            continue

        # 6) Extract the arrays of start, top, end, and integral for all channel chn’s peaks
        pk_start = r[chn][0][0]
        pk_top   = r[chn][0][1]
        pk_end   = r[chn][0][2]
        pk_int   = r[chn][0][3]

        # 7) Identify which peak in pk_top matches 'largest_pk_idx'
        idx_where = np.where(pk_top == largest_pk_idx)[0]
        
        # 8) If we can’t find that index in pk_top, treat it as no peak
        if len(idx_where) == 0:
            single_peak_array = np.array([], dtype=int)
            single_pk_range   = ([], [], [], [])
        else:
            # If found, build a one-element array with that peak index
            i_pk = idx_where[0]
            single_peak_array = np.array([largest_pk_idx], dtype=int)

            # Build (start, top, end, integral) for that single largest peak
            single_pk_range   = (
                [pk_start[i_pk]],
                [pk_top[i_pk]],
                [pk_end[i_pk]],
                [pk_int[i_pk]]
            )

        # 9) Build the file prefix for saving the single-channel PDF
        out_prefix = os.path.join(out_dir, f"Chn{chn}")

        # 10) Finally, call nicer_single_evt_display(...) to plot channel chn
        #     with the largest-peak index, highlight the region, and label the cluster charge
        nicer_single_evt_display(
            chn=chn,
            y=data_1d,
            peaks=single_peak_array,
            save_file_name=out_prefix,
            evt_title=str(evt_title),
            yrange=True,
            peak_ranges=single_pk_range,
            charge=cluster_charge,
            isbatchbool=True
        )

###############################################################################
# STORE COSMIC CHARGES FROM CSV
###############################################################################
def store_cosmic_charges_from_csv():
    """
    1) Checks for cosmic_events.csv.
    2) Reads it, loops each cosmic event row.
    3) Re-opens the JSON file + event index => reconstructs ADC.
    4) Calls compute_channel_charges(...) => obtains channel-level charges.
    5) Appends them to cosmic_charges.csv.
    """
    
    # 1) If cosmic_events.csv doesn’t exist, there’s no cosmic events to process => exit
    if not os.path.exists(COSMIC_CSV_PATH):
        print(f"{output_align} No CSV found at {COSMIC_CSV_PATH}. Nothing to process.")
        return

    # 2) Read cosmic_events.csv into a DataFrame, convert rows to a list of dicts
    cosmic_df = pd.read_csv(COSMIC_CSV_PATH)
    rows = cosmic_df.to_dict(orient="records")

    # 3) Print how many cosmic events we found
    cosmic_count = len(rows)
    print(f"{output_align} Found {cosmic_count} cosmic events in {COSMIC_CSV_PATH}. Computing charges...")

    # 4) Prepare a list to hold all channel-level charge rows
    charges_rows = []

    # 5) Loop over each cosmic event row from cosmic_events.csv
    for row in rows:
        # Retrieve identifying info like folder_name, event IDs, JSON filename, etc.
        folder_name  = row["#Folder_name"]
        evt_id_str   = row["EvtID"]
        raw_file_nr  = row["Raw_file_nr"]
        raw_evt_nr   = row["Raw_Evt_nr"]
        json_fname   = row["File_nr"]
        evt_nr       = row["Evt_nr"]

        # 6) Build the full path to the corresponding JSON file
        fullpath = os.path.join(DATA_FOLDER, json_fname)
        # If it’s missing, skip
        if not os.path.exists(fullpath):
            print(f"{output_align} Missing file {fullpath}; skip.")
            continue

        # 7) Load that JSON, extract the list of events => data_all["all"]
        with open(fullpath, "r") as jsonf:
            data_all = json.load(jsonf)
        events = data_all["all"]

        # Convert the event index from string to int
        raw_evt_idx = int(raw_evt_nr)
        if raw_evt_idx < 0 or raw_evt_idx >= len(events):
            print(f"{output_align} Invalid event_idx {raw_evt_idx} in {json_fname}; skip.")
            continue

        # 8) Retrieve the specific event dictionary
        event_dict = events[raw_evt_idx]

        # 9) Build an empty ADC array => shape (NTT, NC), fill it with channel waveforms
        adc = np.empty((NTT, NC))
        is_bad_dim = False
        for chn in range(NC):
            wave = event_dict.get(f"chn{chn}", [])
            if len(wave) != NTT:
                is_bad_dim = True
                break
            freq_val = most_frequent(wave)
            wave_bs  = np.array(wave) - freq_val
            adc[:, chn] = wave_bs

        # 10) If any channel’s data dimension was incorrect => skip
        if is_bad_dim:
            print(f"{output_align} Bad wave size in {json_fname}, Evt {raw_evt_idx}; skipping.")
            continue

        # 11) Use compute_channel_charges(...) to get cluster charges for each channel
        #     Note: we pass str(folder_name) so we can do year_str[:4] if needed
        channel_info, _ = compute_channel_charges(adc, str(folder_name))

        # 12) For each channel, retrieve the cluster charge => store it in charges_rows
        for chn in range(NC):
            _, cluster_charge = channel_info[chn]
            charges_rows.append({
                "#Folder_name": folder_name,
                "EvtID": evt_id_str,
                "Raw_file_nr": raw_file_nr,
                "Raw_Evt_nr": raw_evt_nr,
                "File_nr": json_fname,
                "Evt_nr": evt_nr,
                "channel": chn,
                "cluster_charge": cluster_charge
            })

    # 13) If we found at least one channel's charge to store, append to cosmic_charges.csv
    if charges_rows:
        charges_df = pd.DataFrame(charges_rows)
        
        # Reorder columns for consistency
        column_order = [
            "#Folder_name","EvtID","Raw_file_nr","Raw_Evt_nr","File_nr","Evt_nr",
            "channel","cluster_charge"
        ]
        charges_df = charges_df[column_order]

        csv_exists = os.path.exists(COSMIC_CHARGES_CSV_PATH)
        charges_df.to_csv(
            COSMIC_CHARGES_CSV_PATH,
            mode='a',
            index=False,
            header=not csv_exists
        )
        print(f"{output_align} Done storing channel charges in {COSMIC_CHARGES_CSV_PATH}.")
    else:
        # 14) If no valid channels or events => no new rows => print a message
        print(f"{output_align} No charges to store. Possibly all events were skipped or not found.")


###############################################################################
# MAIN
###############################################################################
def main():
    raw_data_folder_name  = DEFAULTS["raw_data_folder_name"]
    input_evtdisplay_bool = DEFAULTS["input_evtdisplay_bool"]
    input_chndisplay_bool = DEFAULTS["input_chndisplay_bool"]
    input_equalize_bool   = DEFAULTS["input_equalize_bool"]
    input_filter_bool     = DEFAULTS["input_filter_bool"]

    scan_and_save_cosmic_events(
        raw_data_folder_name,
        input_evtdisplay_bool,
        input_chndisplay_bool,
        input_equalize_bool,
        input_filter_bool
    )

    plot_cosmic_events_from_csv()

    store_cosmic_charges_from_csv()

if __name__ == "__main__":
    main()

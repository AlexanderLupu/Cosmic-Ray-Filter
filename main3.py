#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:53:10 2025

@author: alexlupu
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from collections import Counter
from scipy.signal import find_peaks

###############################################################################
# 1) SETTINGS & HARD-CODED PATH
###############################################################################
NTT = 645       # each channel has 645 samples
NC = 128        # total channels
NCC = 48        # "collection" plane
AMP_THRESHOLD = 25
IGNORED_FOLDER_PREFIX = "."
output_align = "[ANALYSIS]"

DATA_FOLDER = "/Users/alexlupu/github/dune-apex-umn/DATA/20230717/jsonData/"

plt.rcParams.update({'font.size': 20})

###############################################################################
# 2) MINIMAL UTILS
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
# 3) COSMIC REJECTION PARAMETERS
###############################################################################
def get_cosmic_rejection_parameters(year_str):
    """
    e.g. (peak_width=5, peak_height=18, min_strips_w_peaks=20)
    """
    return (5, 18, 20)

def ch_eq_year(year_str):
    return np.ones(NC, dtype=float)

###############################################################################
# 4) FIND_PEAKS_50L
###############################################################################
def find_peak_range(data_1d, pk_indices):
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
    pks, props = find_peaks(data_1d, height=peak_height, width=peak_width)
    pk_ranges = ([], [], [], [])
    if pks.size > 0:
        pk_ranges = find_peak_range(data_1d, pks)
    return pks, props, pk_ranges

###############################################################################
# 5) COSMIC DETECTION (NO REGRESSION)
###############################################################################
def detect_cosmic_line(adc, find_peaks_func, peak_height, peak_width, debug=False):
    """
    1) If any channel <40 has >3 peaks above amplitude=50 => not cosmic
    2) Check adjacency for largest peak => if fail => not cosmic
    3) Check second-largests => if form line => not cosmic
    4) else cosmic
    """
    # Step A) Check if any channel <40 has >3 high-amplitude peaks
    if not check_peaks_per_channel(adc, find_peaks_func, peak_height, peak_width, debug=debug):  # Call helper
        if debug:                                                                               # If debug mode
            print("[DEBUG] => not cosmic => a channel has >3 peaks above 50.")                  # Print reason
        return False                                                                            # Return not cosmic

    # Step B) adjacency check for main (largest) peak
    if not detect_main_line(adc, find_peaks_func, peak_height, peak_width, debug=debug):  # Call adjacency check
        return False                                                                     # If fail => not cosmic

    # Step C) check second-largests => if a line forms => not cosmic
    if detect_second_line(adc, find_peaks_func, peak_height, peak_width, debug=debug):    # If function returns True
        if debug:                                                                        # If debugging
            print("[DEBUG] => not cosmic => second-largests form line.")                 # State reason
        return False                                                                     # not cosmic

    if debug:                                                               # If we haven't returned yet, pass checks
        print("[DEBUG] => cosmic => pass all checks.")                      # Indicate cosmic
    return True                                                             # cosmic => Return True


def check_peaks_per_channel(adc, find_peaks_func, peak_height, peak_width, debug=False):
    """
    For each channel < 40, gather peaks, check how many > 50 amplitude.
    If any channel has >3 => return False => not cosmic
    """

    for chn in range(NCC):                                                   # Loop through channels < 40
        data = adc[:, chn]                                                  # Extract 1D waveform for this channel
        pks, props, _ = find_peaks_func(data, chn, peak_height, peak_width) # Find peaks
        if len(pks) > 0 and "peak_heights" in props:                        # Only proceed if we have peaks + 'peak_heights'
            n_peaks_above_50 = sum(amp > AMP_THRESHOLD for amp in props["peak_heights"])  # Count big peaks
            if n_peaks_above_50 > 2:                                        # If more than 2 big peaks
                if debug:                                                   # If debug
                    print(f"[DEBUG] Channel {chn} => {n_peaks_above_50} peaks>50 => not cosmic.")
                return False                                               # Too many => not cosmic
    return True                                                             # If we never exceed, pass => True


def detect_main_line(adc, find_peaks_func, peak_height, peak_width, debug=False):
    """
    adjacency approach for largest peak => total >=20, chn0 must have peak, out_of_range>5 => not cosmic
    """
    time_tolerance = 50      # Allowed difference in peak times for adjacency
    not_in_range = 0         # Count how many adjacency failures we have
    total_peaks = 0          # Count how many channels have a valid largest peak
    

    channel_peaks = []       # Will store an array with largest peak time or empty
    for chn in range(40):                                             # Loop over channels < 40
        data = adc[:, chn]                                           # Get channel's waveform
        pks, props, _ = find_peaks_func(data, chn, peak_height, peak_width)  # Detect peaks

        best_time = None                                             # We'll pick the first big peak as largest
        if len(pks) > 0 and "peak_heights" in props:                  # Only if we have peaks & peak heights
            idx_sort = np.argsort(props["peak_heights"])[::-1]       # Sort amplitudes descending
            for idx in idx_sort:                                     # Iterate from largest amplitude
                if props["peak_heights"][idx] >= AMP_THRESHOLD:      # If amplitude >= 50
                    best_time = pks[idx]                             # This is the largest peak time
                    break                                            # Stop searching once found
        if best_time is not None:                                    # If we found a largest peak
            channel_peaks.append(np.array([best_time]))              # Store it in array form
            total_peaks += 1                                         # Increment how many channels have a main peak
        else:
            channel_peaks.append(np.array([]))                       # Otherwise store empty array

    if total_peaks < 20:                                             # If fewer than 20 channels had a main peak
        if debug:
            print(f"[DEBUG] main_line => not cosmic => total_peaks={total_peaks}<20.")
        return False                                                 # => not cosmic

    if len(channel_peaks[0]) == 0:                                   # If channel0 has no largest peak
        if debug:
            print("[DEBUG] main_line => not cosmic => channel0 has no valid peak.")
        return False                                                 # => not cosmic

    for chn in range(1, NCC):                                         # Adjacency check from chn=1..39
        prev_pk = channel_peaks[chn - 1]                             # The previous channel's largest peak time
        curr_pk = channel_peaks[chn]                                 # The current channel's largest peak time
        if len(curr_pk) > 0:                                         # If the current channel has a largest peak
            t = curr_pk[0]                                          # Extract that time
            if len(prev_pk) == 0:                                   # If the previous channel had none
                not_in_range += 1                                   # Then adjacency fails => increment
            else:                                                   # Else compare times
                diff = abs(t - prev_pk[0])                          # The difference in time indices
                if diff > time_tolerance:                           # If difference is bigger than 50
                    not_in_range += 1                               # => adjacency fail
            if not_in_range > 5:                                    # If more than 5 adjacency fails
                if debug:
                    print(f"[DEBUG] main_line => not cosmic => not_in_range={not_in_range}")
                return False                                        # => not cosmic

    if debug:
        print(f"[DEBUG] main_line => pass => total_peaks={total_peaks}, not_in_range={not_in_range}")
    return True                                                      # If we pass all checks => True


# def detect_second_line(adc, find_peaks_func, peak_height, peak_width, debug=False):
#     """
#     gather second-largests => if chain>=5 => return True => not cosmic
#     """
#     time_tolerance = 50                      # Allowed difference to consider adjacency
#     min_consecutive = 3                      # Need at least 5 consecutive channels forming a line
#     second_peaks_list = []                   # Will store (channel, second-larg peak time)

#     for chn in range(NCC):                                                    # Loop channels < 48 #very selective filter
#         data = adc[:, chn]                                                   # Extract waveform
#         pks, props, _ = find_peaks_func(data, chn, peak_height, peak_width)  # Detect peaks
#         if len(pks) < 2 or "peak_heights" not in props:                      # Need at less than 2 peaks
#             continue
#         idx_sort = np.argsort(props["peak_heights"])[::-1]                   # Sort them descending
#         second_time = pks[idx_sort[1]]                                       # Grab second-larg time
#         second_peaks_list.append((chn, second_time))                         # Store it

#     if len(second_peaks_list) < min_consecutive:                             # If fewer than 5 total second-larg
#         if debug:
#             print("[DEBUG] second_line => not forming line => return False.")
#         return False                                                         # => no second line

#     second_peaks_list.sort(key=lambda x: x[0])                               # Sort by channel
#     consecutive = 1                                                         # We'll track consecutive adjacency
#     for i in range(1, len(second_peaks_list)):
#         pchn, ptime = second_peaks_list[i - 1]                               # Previous channel/time
#         cchn, ctime = second_peaks_list[i]                                   # Current channel/time
#         if cchn == pchn + 1:                                                # If consecutive channel
#             if abs(ctime - ptime) <= time_tolerance:                        # Check if times are close
#                 consecutive += 1                                            # Consecutive adjacency
#             else:
#                 consecutive = 1                                             # Reset if times differ too much
#         else:
#             consecutive = 1                                                 # Reset if channels not consecutive

#         if consecutive >= min_consecutive:                                  # If we reach 5 in a row
#             if debug:
#                 print("[DEBUG] second_line => chain of >=5 => return True => not cosmic.")
#             return True                                                     # => second line => not cosmic

#     return False                                                             # If we never found a chain => False


###############################################################################
# 6) SHOW_EVENT
###############################################################################
def show_event(adc, run_time, event_id,
               binary_file_id, binary_event_id,
               converted_file_id, converted_event_id,
               saveFileName, isbatchbool):
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
# 7) PREPARE_DATA
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

    if is_data_dimension_incorrect:
        return adc, False
    if not filter_bool:
        return adc, True

    pk_width, pk_height, min_strips = get_cosmic_rejection_parameters(evt_title[:4])

    # Call detect_cosmic_line => returns True if "cosmic => exclude"
    is_cosmic = detect_cosmic_line(
        adc,
        find_peaks_50l,
        pk_height,
        pk_width,
        debug=first_time
    )

    if is_cosmic:
        if first_time:
            print(f"{output_align}[DEBUG] cosmic => adjacency checks => Evt({j_file_nr},{j_evt_nr}).")
        if input_evtdisplay_bool:
            event_id          = data_event_dict.get("eventId", "???")
            binary_file_id    = data_event_dict.get("binaryFileID", "???")
            binary_event_id   = data_event_dict.get("binaryEventID", "???")
            converted_file_id = data_event_dict.get("convertedFileID", "???")
            converted_event_id= data_event_dict.get("convertedEventID", "???")

            save_file_name = f"{base_output_name}_file{j_file_nr}_evt{j_evt_nr}.pdf"
            show_event(adc, evt_title,
                       event_id,
                       binary_file_id,
                       binary_event_id,
                       converted_file_id,
                       converted_event_id,
                       save_file_name,
                       isbatchbool=False)
        # cosmic => exclude
        is_event_ok = False

    return adc, is_event_ok

###############################################################################
# 8) READ_DATA
###############################################################################
def remove_coherent_noise(data_event_dict):
    return data_event_dict

def read_data(input_evtdisplay_bool,
              input_chndisplay_bool,
              input_equalize_bool,
              input_filter_bool,
              raw_data_folder_name):

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

            base_name  = fname[:-5]
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
                print(f"{output_align}Event {i} => KEPT.")
            else:
                print(f"{output_align}Event {i} => EXCLUDED (cosmic).")

    return evt_count

###############################################################################
# 9) MAIN
###############################################################################
DEFAULTS = {
    "raw_data_folder_name": "20230717",
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
# 10) SCRIPT ENTRY
###############################################################################
if __name__ == "__main__":
    raw_data_folder_, evt_disp_, chn_disp_, eq_bool_, filt_bool_, term_bool_ = parse_args(sys.argv, DEFAULTS)
    main(raw_data_folder_, evt_disp_, chn_disp_, eq_bool_, filt_bool_, term_bool_)

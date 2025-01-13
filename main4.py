#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:39:35 2025

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
# SETTINGS & HARD-CODED PATH
###############################################################################
NTT = 645       # each channel has 645 samples
NC = 128        # total channels
NCC = 48        # "collection" plane
AMP_THRESHOLD = 18
WIDTH_THRESHOLD = 40
IGNORED_FOLDER_PREFIX = "."
output_align = "[ANALYSIS]"

DATA_FOLDER = "/Users/alexlupu/github/dune-apex-umn/DATA/20230717/jsonData/"

plt.rcParams.update({'font.size': 20})

###############################################################################
# MINIMAL UTILS
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
    # Returns: (peak_width, peak_height, min_strips)
    return (5, 25, 20)

def ch_eq_year(year_str):
    return np.ones(NC, dtype=float)

###############################################################################
# FIND_PEAKS_50L
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

    # Filter out peaks that are within 50 units of each other
    filtered_pks = []
    filtered_props = {"peak_heights": []}
    if pks.size > 0:
        sorted_indices = np.argsort(props["peak_heights"])[::-1]  # Sort by height descending
        for idx in sorted_indices:
            current_peak = pks[idx]
            if all(abs(current_peak - fp) > 50 for fp in filtered_pks):
                filtered_pks.append(current_peak)
                filtered_props["peak_heights"].append(props["peak_heights"][idx])

    # Convert filtered lists back to numpy arrays
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
    # 1) Check each channel individually for too many tall peaks
    #    AND ensure no two peaks are within 50 samples
    if not check_peaks_per_channel(adc, find_peaks_func, peak_height, peak_width, debug=debug):
        return False

    # 2) Attempt to detect a main line of peaks across channels
    if not detect_main_line(adc, find_peaks_func, peak_height, peak_width, debug=debug):
        return False

    return True

###############################################################################
# NEW CHECK_PEAKS_PER_CHANNEL
# (We now directly call raw scipy.signal.find_peaks and verify no two peaks < 50 apart)
###############################################################################
def check_peaks_per_channel(adc, find_peaks_func, peak_height, peak_width, debug=False):
    for chn in range(NCC):
        data = adc[:, chn]

        # 1) Find *all* peaks in this channel (raw find_peaks)
        pks, props = find_peaks(data, height=peak_height, width=peak_width)

        # 2) If any two peaks are within 50 samples => exclude event immediately
        if len(pks) > 1:
            pks_sorted = np.sort(pks)
            if np.any(np.diff(pks_sorted) < WIDTH_THRESHOLD):
                if debug:
                    print(f"{output_align}[DEBUG] ch{chn}: two peaks < 50 samples => exclude event.")
                return False

        # 3) Also check amplitude threshold logic
        if len(pks) > 0 and "peak_heights" in props:
            n_peaks_above_amp_threshold = sum(amp > AMP_THRESHOLD for amp in props["peak_heights"])
            if n_peaks_above_amp_threshold > 2:
                if debug:
                    print(f"{output_align}[DEBUG] ch{chn}: more than 2 tall peaks => exclude event.")
                return False
    return True

def detect_main_line(adc, find_peaks_func, peak_height, peak_width, debug=False):
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

    if total_peaks < 20:
        return False

    if len(channel_peaks[0]) == 0:
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
# SHOW_EVENT
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

    # detect_cosmic_line -> calls check_peaks_per_channel() with raw find_peaks
    # and detect_main_line() with find_peaks_50l
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
        is_event_ok = False

    return adc, is_event_ok

###############################################################################
# READ_DATA
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
# MAIN
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
# SCRIPT ENTRY
###############################################################################
if __name__ == "__main__":
    raw_data_folder_, evt_disp_, chn_disp_, eq_bool_, filt_bool_, term_bool_ = parse_args(sys.argv, DEFAULTS)
    main(raw_data_folder_, evt_disp_, chn_disp_, eq_bool_, filt_bool_, term_bool_)

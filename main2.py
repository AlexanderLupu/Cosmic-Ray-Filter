#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 12:14:38 2025

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
from scipy.stats import linregress
from scipy.signal import find_peaks

###############################################################################
# 1) SETTINGS & HARD-CODED PATH
###############################################################################
NTT = 645       # each channel has 645 samples
NC = 128        # total channels
NCC = 48        # "collection" plane
IGNORED_FOLDER_PREFIX = "."
output_align = "[ANALYSIS]"

DATA_FOLDER = "/Users/alexlupu/github/dune-apex-umn/DATA/20230623/jsonData/"

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
    (peak_width=5, peak_height=18, min_strips_w_peaks=20)
    This just returns some defaults for your find_peaks logic.
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
# 5) DETECT_COSMIC_LINE
###############################################################################
def detect_cosmic_line(adc, find_peaks_func, peak_height, peak_width, debug=False):
    """
    Variation:
      - min required peaks across channels <40 => 20
      - amplitude threshold => e.g. 50
      - time_tolerance => 50
      - if channel 0 has no valid peaks => not cosmic
      - if out_of_range >5 => not cosmic
      - else cosmic
    """

    time_tolerance = 50
    not_in_range = 0
    total_peaks = 0
    amp_threshold = 50  # new amplitude threshold

    # 1) Gather peaks for channels < 40
    channel_peaks = []  # each entry is a sorted array of valid peak times
    for chn in range(40):
        data = adc[:, chn]
        pks, props, _ = find_peaks_func(data, chn, peak_height, peak_width)

        # NEW: filter out peaks whose amplitude < amp_threshold
        valid_times = []
        if len(pks) > 0 and "peak_heights" in props:
            for i, pk in enumerate(pks):
                amp = props["peak_heights"][i]
                if amp >= amp_threshold:
                    valid_times.append(pk)  # keep this peak
        else:
            # no 'peak_heights' => no valid peaks
            valid_times = []

        # store them sorted
        valid_times = np.sort(valid_times)
        channel_peaks.append(valid_times)
        total_peaks += len(valid_times)

    # 2) check if total_peaks < 20 => not cosmic
    if total_peaks < 20:
        if debug:
            print(f"[DEBUG] not cosmic => total_peaks={total_peaks} < 20.")
        return False

    # 3) if channel 0 has no peaks => not cosmic
    if len(channel_peaks[0]) == 0:
        if debug:
            print("[DEBUG] not cosmic => channel0 has no valid peaks.")
        return False

    # 4) time consistency across channels
    for chn in range(1, 40):
        prev_peaks = channel_peaks[chn - 1]
        curr_peaks = channel_peaks[chn]

        for t in curr_peaks:
            if len(prev_peaks) == 0:
                not_in_range += 1
                if not_in_range > 5:
                    if debug:
                        print(f"[DEBUG] => not cosmic (not_in_range={not_in_range})")
                    return False
                continue

            diffs = np.abs(prev_peaks - t)
            min_diff = np.min(diffs)
            if min_diff > time_tolerance:
                not_in_range += 1
                if not_in_range > 5:
                    if debug:
                        print(f"[DEBUG] => not cosmic (not_in_range={not_in_range})")
                    return False

    if debug:
        print(f"[DEBUG] => cosmic! total_peaks={total_peaks}, not_in_range={not_in_range}")
    return True

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

    # Call new detect_cosmic_line => returns True if cosmic
    is_cosmic = detect_cosmic_line(adc, find_peaks_50l, pk_height, pk_width, debug=first_time)
    if is_cosmic:
        if first_time:
            print(f"{output_align}[DEBUG] cosmic => amplitude check + time consistency => Evt({j_file_nr},{j_evt_nr}).")
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
        is_event_ok = False  # cosmic => exclude
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
    "raw_data_folder_name": "20230623",
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

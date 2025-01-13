#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:23:17 2025

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
    # If arr is large, Counter(...) might be expensive.
    # You could approximate with median or some heuristic if you want speed.
    c = Counter(arr)
    return c.most_common(1)[0][0]

###############################################################################
# 3) COSMIC REJECTION PARAMETERS
###############################################################################
def get_cosmic_rejection_parameters(year_str):
    return (5, 18, 20)  # (peak_width=5, peak_height=18, min_strips_w_peaks=20)# minimum amount of strips with at least one peak

def ch_eq_year(year_str):
    # If equalization is trivial (all 1.0), you can skip it altogether
    return np.ones(NC, dtype=float)

###############################################################################
# 4) FIND_PEAKS_50L
###############################################################################
def find_peak_range(data_1d, pk_indices):
    """
    Doing a naive approach: for each peak, expand left & right while data>0.
    This is O(N) per peak. 645 samples => not huge, but repeated many times.

    Speed-ups:
      - If you only need the 'largest peak' time, skip the integral or do partial.
      - If you only want # of peaks, you can trust find_peaks(...) alone.
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
    pks, props = find_peaks(data_1d, height=peak_height, width=peak_width)
    # If you only want to count peaks, you might skip find_peak_range entirely:
    #   return pks, props, None
    # But if you do want integrals or range, keep it:
    pk_ranges = ([], [], [], [])
    if pks.size > 0:
        pk_ranges = find_peak_range(data_1d, pks)
    return pks, props, pk_ranges

###############################################################################
# 5) DETECT COSMIC LINE
###############################################################################
def detect_cosmic_line(adc, find_peaks_func, peak_height, peak_width, debug=False):
    """
    We gather largest peak from each channel <40.
    Potential speed-ups:
      - If you only need the time of the largest peak, skip storing all peaks.
      - If your data is not too noisy, you can short-circuit if no peaks found.
    """
    n_channels = adc.shape[1]
    chn_peak_times = []

    for chn in range(n_channels):
        # pks, props, _ = find_peaks_func(adc[:, chn], chn, peak_height, peak_width)
        
        pe, prop = find_peaks(data, height=peak_height, width=peak_width)
        
        # if pe > 5
        
    #     if len(pks) > 0:
    #         if "peak_heights" in props:
    #             idx_max = np.argmax(props["peak_heights"])
    #             best_time = pks[idx_max]
    #         else:
    #             best_time = pks[0]
    #         chn_peak_times.append((chn, best_time))

    # # Only channels < 40
    # chn_peak_times = [(c, t) for (c, t) in chn_peak_times if c < 40]
    # if len(chn_peak_times) < 5:
    #     return False

    # arr = np.array(chn_peak_times)
    
    # #slope, intercept, r_value, _, _ = linregress(arr[:, 0], arr[:, 1])
    # r_squared = r_value**2

    # if debug:
    #     print(f"[DEBUG] cosmic_line => #pts={len(arr)}, R^2={r_squared:.3f}")

    # return (r_squared > 0.99)

###############################################################################
# 6) SHOW_EVENT
###############################################################################
def show_event(adc, run_time, event_id,
               binary_file_id, binary_event_id,
               converted_file_id, converted_event_id,
               saveFileName, isbatchbool):
    # Plotting can be expensive if done for many events.
    # If speed is crucial, skip or reduce figure size/quality.

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

    # Building 'adc' with baseline
    # Speed tip: If baseline is always the same freq_val across the entire channel,
    # we can do wave - wave.mean() or wave - wave.median() if close enough.

    is_event_ok = True
    is_data_dimension_incorrect = False

    adc = np.empty((NTT, NC))
    for chn in range(NC):
        ch_key = f"chn{chn}"
        wave = data_event_dict[ch_key]
        
        # if wave is not an actual list of length 645, skip
        if len(wave) != NTT:
            if first_time:
                print(f"{output_align}!! Event {j_evt_nr}: strip {chn} length {len(wave)} != {NTT}")
            is_data_dimension_incorrect = True
            break

        # This is O(N) => 645 steps
        # freq_val = most_frequent(wave)
        # wave_bs = np.array(wave) - freq_val

        # # If eq_factors is all 1.0, we can skip the loop
        # if equalize_bool:
        #     eq_factors = ch_eq_year(evt_title[:4])
        #     wave_bs = np.round(wave_bs * eq_factors[chn], decimals=1)
        # adc[:, chn] = wave_bs

    if is_data_dimension_incorrect:
        return adc, False
    if not filter_bool:
        return adc, True

    # cosmic filter
    pk_width, pk_height, min_strips = get_cosmic_rejection_parameters(evt_title[:4])

    # 1) count channels with peaks
    # we do a loop over all 48 collection channels => 48 x find_peaks
    strips_with_peaks = 0
    for chn in range(NCC):
        pks, _, _ = find_peaks_50l(adc[:, chn], chn, pk_height, pk_width)
        if len(pks) > 0:
            strips_with_peaks += 1

    # 2) line detection => extra loop over all channels, but that’s okay
    is_line = detect_cosmic_line(adc, find_peaks_50l, pk_height, pk_width, debug=first_time)

    if strips_with_peaks >= min_strips and is_line:
        if first_time:
            print(f"{output_align}[DEBUG] cosmic => #strips={strips_with_peaks} >= {min_strips}, line => exclude. Evt({j_file_nr},{j_evt_nr}).")

        if input_evtdisplay_bool:
            # If speed matters, skip plotting or do it only once per file, etc.
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
# 8) READ_DATA
###############################################################################
def remove_coherent_noise(data_event_dict):
    # If there's no real algorithm, skip it entirely to save time
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

    # If speed is crucial, skip or reduce the # of files
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

        # Possibly do a quick check if file is large => memory-map or partial load
        with open(fullpath, "r") as f:
            data_all = json.load(f)
        events = data_all["all"]

        for i, event_dict in enumerate(events):
            evt_count += 1

            # If speed is crucial, skip remove_coherent_noise or do partial
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
                base_output_name=base_outname,
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

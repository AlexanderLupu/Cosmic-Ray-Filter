#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:37:19 2025

@author: alexlupu
"""

import os
import sys
import csv
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
NTT = 645        # Number of samples (time ticks) per channel
NC = 128         # Total number of channels
NCC = 48         # "Collection" plane channels (subset)
AMP_THRESHOLD = 18
WIDTH_SPACE_THRESHOLD = 40
WIDTH_PEAK_THRESHOLD = 5
IGNORED_FOLDER_PREFIX = "."
output_align = "[ANALYSIS]"

DATA_FOLDER       = "/Users/alexlupu/github/dune-apex-umn/DATA/20230722/jsonData/"
PLOTS_OUTPUT_DIR  = "/Users/alexlupu/github/dune-apex-umn/Plots"
COSMIC_CSV_PATH   = "/Users/alexlupu/github/dune-apex-umn/cosmic_events.csv"

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
    Return (peak_width, peak_height, min_strips). 
    Hard-coded to (5, AMP_THRESHOLD, 47).
    """
    return (5, AMP_THRESHOLD, 47)

def ch_eq_year(year_str):
    """
    Return array of equalization factors (dummy: all 1.0 here).
    """
    return np.ones(NC, dtype=float)

###############################################################################
# PEAK-FINDING
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

    filtered_pks = []
    filtered_props = {"peak_heights": []}
    if pks.size > 0:
        sorted_indices = np.argsort(props["peak_heights"])[::-1]
        for idx in sorted_indices:
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
def detect_cosmic_line(adc, find_peaks_func, peak_height, peak_width, debug=False):
    """
    1) check_peaks_per_channel => ensures no channel has close/excessive peaks
    2) detect_main_line => checks for a time-coherent line
    """
    if not check_peaks_per_channel(adc, find_peaks_func, peak_height, peak_width, debug=debug):
        return False
    if not detect_main_line(adc, find_peaks_func, peak_height, peak_width, debug=debug):
        return False
    return True

def check_peaks_per_channel(adc, find_peaks_func, peak_height, peak_width, debug=False):
    from scipy.signal import find_peaks
    for chn in range(NCC):
        data = adc[:, chn]
        pks, props = find_peaks(data, height=peak_height, width=peak_width)

        if len(pks) > 1:
            pks_sorted = np.sort(pks)
            if np.any(np.diff(pks_sorted) < WIDTH_SPACE_THRESHOLD):
                if debug:
                    print(f"{output_align}[DEBUG] ch{chn}: two peaks < {WIDTH_SPACE_THRESHOLD}")
                return False

        if peak_width > WIDTH_PEAK_THRESHOLD:
            if debug:
                print(f"{output_align}[DEBUG] ch{chn}: peak_width > {WIDTH_PEAK_THRESHOLD}")
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
# SHOW_EVENT (2D event-level display)
###############################################################################
def show_event(adc, run_time, event_id,
               binary_file_id, binary_event_id,
               converted_file_id, converted_event_id,
               saveFileName, isbatchbool):
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

    pk_width, pk_height, _ = get_cosmic_rejection_parameters(evt_title[:4])
    is_cosmic = detect_cosmic_line(adc, find_peaks_50l, pk_height, pk_width, debug=first_time)

    if is_cosmic:
        if first_time:
            print(f"{output_align}[DEBUG] cosmic => adjacency => Evt({j_file_nr},{j_evt_nr}).")
        if input_evtdisplay_bool:
            event_id          = data_event_dict.get("eventId", "???")
            binary_file_id    = data_event_dict.get("binaryFileID", "???")
            binary_event_id   = data_event_dict.get("binaryEventID", "???")
            converted_file_id = data_event_dict.get("convertedFileID", "???")
            converted_event_id= data_event_dict.get("convertedEventID", "???")

            save_file_name = f"{base_output_name}_file{j_file_nr}_evt{j_evt_nr}.pdf"
            show_event(adc,
                       evt_title,
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
# PHASE A: SCAN & SAVE COSMIC EVENTS TO CSV
###############################################################################
def scan_and_save_cosmic_events(raw_data_folder_name,
                                input_evtdisplay_bool,
                                input_chndisplay_bool,
                                input_equalize_bool,
                                input_filter_bool):
    path_json = DATA_FOLDER
    Path(path_json).mkdir(parents=True, exist_ok=True)

    csv_exists = os.path.exists(COSMIC_CSV_PATH)
    with open(COSMIC_CSV_PATH, "a", newline='') as csvf:
        writer = csv.writer(csvf)
        if not csv_exists:
            writer.writerow(["filename","event_index","file_id","event_id"])

        allfiles = sorted(os.listdir(path_json))
        file_count = 0
        evt_count = 0

        print(f"\n[{find_time_now()}] > Start scanning => {COSMIC_CSV_PATH}")

        for fname in allfiles:
            if not fname.lower().endswith(".json"):
                continue
            if fname.startswith(IGNORED_FOLDER_PREFIX):
                continue

            file_count += 1
            fullpath = os.path.join(path_json, fname)

            with open(fullpath, "r") as f:
                data_all = json.load(f)
            events = data_all["all"]

            base_out_dir = PLOTS_OUTPUT_DIR
            Path(base_out_dir).mkdir(parents=True, exist_ok=True)
            base_name   = fname[:-5]
            base_outname= os.path.join(base_out_dir, base_name)

            for i, event_dict in enumerate(events):
                evt_count += 1
                adc, is_ok = prepare_data(
                    input_equalize_bool,
                    input_filter_bool,
                    event_dict,
                    raw_data_folder_name,
                    file_count,
                    i,
                    base_outname,
                    first_time=(i==0),
                    input_evtdisplay_bool=input_evtdisplay_bool
                )

                # If not is_ok => cosmic
                if not is_ok:
                    file_id = event_dict.get("binaryFileID", "???")
                    evt_id  = event_dict.get("eventId", "???")
                    writer.writerow([fname, i, file_id, evt_id])

        print(f"{output_align} Done scanning => {COSMIC_CSV_PATH}")

###############################################################################
# SINGLE-CHANNEL DISPLAY (NICER)
###############################################################################
def nicer_single_evt_display(chn, y, peaks, save_file_name, evt_title, yrange, peak_ranges, charge, isbatchbool):
    fig, ax = plt.subplots(figsize=(32, 9))
    str_chn = str(chn).zfill(3)
    label = 'Chn_' + str_chn

    peaks = peaks.astype(int)

    if chn < NCC:
        ps = [p * 0.5 for p in peaks]
        p_list = [y[p] for p in peaks]
        label += f";\nPeaks: {ps} µs, {p_list} ADC\nCharge: {charge} ADC*µs"
    else:
        ps = [p * 0.5 for p in peaks]
        label += f";\nDetected peaks at: {ps} µs"

    ax.plot(y, label=label)
    zeros = y[peaks] if chn < NCC else np.zeros(len(peaks))
    ax.plot(peaks, zeros, "x")

    if len(peak_ranges) == 4 and len(peak_ranges[0]) == len(peak_ranges[2]):
        pk_start = np.array(peak_ranges[0], dtype=int)
        pk_end   = np.array(peak_ranges[2], dtype=int)
        for s, e in zip(pk_start, pk_end):
            region_x = np.arange(s, e)
            region_y = y[region_x]
            if chn < NCC:
                ax.fill_between(region_x, 0, region_y, color='gold', alpha=0.5)

    ax.set_ylabel('ADC (baseline-subtracted)', labelpad=10, fontsize=24)
    ax.set_xlabel('time ticks [0.5 µs/tick]', fontsize=24)
    ax.legend(fontsize=24)
    ax.grid(True, which='major', axis='both', linewidth=1, color='gray')
    ax.grid(True, which='minor', axis='x', linewidth=0.5, linestyle='dashed', color='gray')
    ax.set_xticks(np.arange(0, NTT + 1, 50))
    ax.set_xticks(np.arange(0, NTT + 1, 10), minor=True)

    if yrange:
        if chn < NCC:
            ax.set_ylim(-20, 300)
        else:
            ax.set_ylim(-250, 250)

    ax.set_xlim(0, len(y))
    ax.yaxis.set_label_coords(-.04, 0.5)
    ax.spines['right'].set_visible(False)

    plt.title(f"{evt_title} - Strip {chn + 1}", fontsize=30)
    plt.subplots_adjust(left=0.06, right=0.94, top=0.96, bottom=0.08)
    plt.tight_layout()

    str_strip = str(chn + 1).zfill(3)
    out_fname = f"{save_file_name}_Strip{str_strip}.pdf"
    plt.savefig(out_fname, dpi=100)
    if not isbatchbool:
        plt.show()
    plt.clf()
    plt.close()

def single_event(adc, evt_title, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pk_width, pk_height, _ = get_cosmic_rejection_parameters(evt_title[:4])

    for chn in range(adc.shape[1]):
        data_1d = adc[:, chn]
        pks, props, pk_ranges = find_peaks_50l(data_1d, chn, pk_height, pk_width)
        charge = np.sum(data_1d)

        out_prefix = os.path.join(out_dir, f"Chn{chn}")
        nicer_single_evt_display(
            chn=chn,
            y=data_1d,
            peaks=pks,
            save_file_name=out_prefix,
            evt_title=evt_title,
            yrange=True,
            peak_ranges=pk_ranges,
            charge=charge,
            isbatchbool=True
        )

###############################################################################
# PHASE B: READ CSV => PLOT COSMIC EVENTS
###############################################################################
def plot_cosmic_events_from_csv():
    if not os.path.exists(COSMIC_CSV_PATH):
        print(f"{output_align} No CSV found at {COSMIC_CSV_PATH}. Nothing to plot.")
        return

    with open(COSMIC_CSV_PATH, "r") as csvf:
        reader = csv.DictReader(csvf)
        rows = list(reader)

    cosmic_count = len(rows)
    print(f"{output_align} Found {cosmic_count} cosmic events in {COSMIC_CSV_PATH}. Plotting...")

    for row in rows:
        fname     = row["filename"]
        event_idx = int(row["event_index"])
        file_id   = row["file_id"]
        evt_id    = row["event_id"]

        fullpath = os.path.join(DATA_FOLDER, fname)
        if not os.path.exists(fullpath):
            print(f"{output_align} Missing file {fullpath}; skip.")
            continue

        with open(fullpath, "r") as f:
            data_all = json.load(f)
        events = data_all["all"]

        if event_idx < 0 or event_idx >= len(events):
            print(f"{output_align} Invalid event_idx {event_idx} in {fname}; skip.")
            continue

        event_dict = events[event_idx]
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

        if is_bad_dim:
            print(f"{output_align} Bad wave size in {fname} Evt {event_idx}; skip.")
            continue

        cosmic_label = f"FileID {file_id}, EvtID {evt_id} (COSMIC)"
        out_subdir   = os.path.join(PLOTS_OUTPUT_DIR, "cosmic_from_csv", f"{fname}_Evt{event_idx}")
        single_event(adc, cosmic_label, out_subdir)

    print(f"{output_align} Done plotting cosmic events from CSV.")

###############################################################################
# MAIN
###############################################################################
def main():
    raw_data_folder_name  = DEFAULTS["raw_data_folder_name"]
    input_evtdisplay_bool = DEFAULTS["input_evtdisplay_bool"]
    input_chndisplay_bool = DEFAULTS["input_chndisplay_bool"]
    input_equalize_bool   = DEFAULTS["input_equalize_bool"]
    input_filter_bool     = DEFAULTS["input_filter_bool"]

    # Phase A: read .json => find cosmic => save them in cosmic_events.csv
    scan_and_save_cosmic_events(
        raw_data_folder_name,
        input_evtdisplay_bool,
        input_chndisplay_bool,
        input_equalize_bool,
        input_filter_bool
    )

    # Phase B: read cosmic_events.csv => produce single-channel plots only for cosmic
    plot_cosmic_events_from_csv()

if __name__ == "__main__":
    main()

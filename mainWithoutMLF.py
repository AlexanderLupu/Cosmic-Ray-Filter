#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:23:27 2025

@author: alexlupu
"""

"""
Full code (without detect_main_line):
 - Phase A: scans all JSON, detects cosmic, saves cosmic events to cosmic_events.csv
   with columns [#Folder_name, EvtID, Raw_file_nr, Raw_Evt_nr, File_nr, Evt_nr].
 - Phase B: re-opens cosmic_events.csv, reads those columns => single_event(...) channel plotting.
 - find_charge_cluster(...) no longer returns 0 if overshadowed,
   it just skips adding neighbor's partial charge.
 - If two close peaks are found, we also print EvtID in the debug message.

Additionally:
 - store_cosmic_charges_from_csv() re-opens cosmic_events.csv,
   for each row => re-opens the JSON => compute_channel_charges(...),
   writing results to cosmic_charges.csv.

We have removed detect_main_line(...) and simplified detect_cosmic_line(...) 
to use only check_peaks_per_channel(...). The legend label is "ADC*tick."
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
        pk_int.append(round(temp_int, 1))  # ADC*tick
    return (pk_start, pk_top, pk_end, pk_int)

def find_peaks_50l(data_1d, chn, peak_height, peak_width):
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
def check_peaks_per_channel(adc, find_peaks_func, peak_height, peak_width, debug=False, event_id=None):
    from scipy.signal import find_peaks
    
    for chn in range(NCC):
        data = adc[:, chn]
        pks, props = find_peaks(data, height=peak_height, width=peak_width)
        if len(pks) > 1:
            pks_sorted = np.sort(pks)
            if np.any(np.diff(pks_sorted) < WIDTH_SPACE_THRESHOLD):
                if debug:
                    print(f"{output_align} EvtID {event_id}, ch{chn}: 2 peaks < {WIDTH_SPACE_THRESHOLD}")
                return False
        if peak_width > WIDTH_PEAK_THRESHOLD:
            if debug:
                print(f"{output_align}[DEBUG] ch{chn}: peak_width > {WIDTH_PEAK_THRESHOLD}")
            return False
    return True

def detect_cosmic_line(adc, find_peaks_func, peak_height, peak_width, debug=False, event_id=None):
    """
    Simplified: we only check channel-level peak structure. 
    The 'detect_main_line(...)' check has been removed.
    """
    return check_peaks_per_channel(
        adc, find_peaks_func, peak_height, peak_width, debug=debug, event_id=event_id
    )

###############################################################################
# SHOW_EVENT (2D event-level)
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
    plt.xticks(np.arange(0, adc.shape[1] + 1, 10), rotation=45)
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
            eq_factors = ch_eq_year(str(evt_title)[:4])
            wave_bs = np.round(wave_bs * eq_factors[chn], decimals=1)

        adc[:, chn] = wave_bs

    if is_data_dimension_incorrect:
        return adc, False

    if not filter_bool:
        return adc, True

    current_event_id = data_event_dict.get("eventId", "???")
    pk_width, pk_height, _ = get_cosmic_rejection_parameters(str(evt_title)[:4])

    # We call detect_cosmic_line, which now only does check_peaks_per_channel
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
                str(evt_title),
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
# PHASE A: SCAN & SAVE COSMIC EVENTS
###############################################################################
def scan_and_save_cosmic_events(raw_data_folder_name,
                                input_evtdisplay_bool,
                                input_chndisplay_bool,
                                input_equalize_bool,
                                input_filter_bool):
    path_json = DATA_FOLDER
    Path(path_json).mkdir(parents=True, exist_ok=True)

    cosmic_rows = []

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

            if not is_ok:
                event_id = event_dict.get("eventId", "???")
                binary_evt_id = event_dict.get("binaryEventID", "???")

                cosmic_rows.append({
                    "#Folder_name": raw_data_folder_name,
                    "EvtID": event_id,
                    "Raw_file_nr": file_count,
                    "Raw_Evt_nr": i,
                    "File_nr": fname,
                    "Evt_nr": binary_evt_id,
                })

    print(f"{output_align} Done scanning => {COSMIC_CSV_PATH}")

    if cosmic_rows:
        cosmic_df = pd.DataFrame(cosmic_rows)
        cosmic_df = cosmic_df[
            ["#Folder_name","EvtID","Raw_file_nr","Raw_Evt_nr","File_nr","Evt_nr"]
        ]
        csv_exists = os.path.exists(COSMIC_CSV_PATH)
        cosmic_df.to_csv(
            COSMIC_CSV_PATH,
            mode='a',
            index=False,
            header=not csv_exists
        )
    else:
        print(f"{output_align} No cosmic events found; nothing written to {COSMIC_CSV_PATH}.")

###############################################################################
# CHANNEL-CHARGE CALCULATION (SHARED)
###############################################################################
def compute_channel_charges(adc, evt_title):
    year_str = str(evt_title)
    pk_width, pk_height, _ = get_cosmic_rejection_parameters(year_str[:4])

    r = [[] for _ in range(NC)]
    s = [[] for _ in range(NC)]

    for chn in range(NC):
        data_1d = adc[:, chn]
        pks, props, pk_ranges = find_peaks_50l(data_1d, chn, pk_height, pk_width)
        (pk_start, pk_top, pk_end, pk_int) = pk_ranges
        r[chn].append([pk_start, pk_top, pk_end, pk_int])
        s[chn] = range(len(pk_start))

    results = {}
    for chn in range(NC):
        if len(r[chn]) == 0 or len(r[chn][0][0]) == 0:
            largest_pk_idx = None
            cluster_charge = 0
        else:
            pk_start = r[chn][0][0]
            pk_top   = r[chn][0][1]
            pk_end   = r[chn][0][2]
            pk_int   = r[chn][0][3]

            idx_max = np.argmax(pk_int)
            peak_charge   = pk_int[idx_max]
            cluster_charge= find_charge_cluster(adc, chn, peak_charge, r, s, idx_max)
            largest_pk_idx= pk_top[idx_max]

        results[chn] = (largest_pk_idx, cluster_charge)

    return results, (r, s)

###############################################################################
# PHASE B: READ CSV => PLOT COSMIC EVENTS
###############################################################################
def plot_cosmic_events_from_csv():
    if not os.path.exists(COSMIC_CSV_PATH):
        print(f"{output_align} No CSV found at {COSMIC_CSV_PATH}. Nothing to plot.")
        return

    cosmic_df = pd.read_csv(COSMIC_CSV_PATH)
    rows = cosmic_df.to_dict(orient="records")

    cosmic_count = len(rows)
    print(f"{output_align} Found {cosmic_count} cosmic events in {COSMIC_CSV_PATH}. Plotting...")

    for row in rows:
        folder_name  = row["#Folder_name"]
        evt_id_str   = row["EvtID"]
        raw_file_nr  = row["Raw_file_nr"]
        raw_evt_nr   = row["Raw_Evt_nr"]
        json_fname   = row["File_nr"]
        binary_evt   = row["Evt_nr"]

        fullpath = os.path.join(DATA_FOLDER, json_fname)
        if not os.path.exists(fullpath):
            print(f"{output_align} Missing file {fullpath}; skip.")
            continue

        with open(fullpath, "r") as f:
            data_all = json.load(f)
        events = data_all["all"]

        raw_evt_idx = int(raw_evt_nr)
        if raw_evt_idx < 0 or raw_evt_idx >= len(events):
            print(f"{output_align} Invalid event_idx {raw_evt_idx} in {json_fname}; skip.")
            continue

        event_dict = events[raw_evt_idx]
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
            print(f"{output_align} Bad wave size in {json_fname} Evt {raw_evt_idx}; skip.")
            continue

        cosmic_label = f"EvtID {evt_id_str} (COSMIC) => {json_fname}"
        out_subdir   = os.path.join(PLOTS_OUTPUT_DIR, "cosmic_from_csv", f"{json_fname}_Evt{raw_evt_idx}")

        single_event(adc, str(folder_name), out_subdir)

    print(f"{output_align} Done plotting cosmic events from CSV.")

###############################################################################
# SINGLE-EVENT CHANNEL PLOTTING HELPER
###############################################################################
def nicer_single_evt_display(chn, y, peaks, save_file_name, evt_title, yrange, peak_ranges, charge, isbatchbool):
    fig, ax = plt.subplots(figsize=(32, 9))
    str_chn = str(chn).zfill(3)
    label = f"Chn_{str_chn}; Charge cluster: {charge} ADC*tick"

    peaks = peaks.astype(int)
    ax.plot(y, label=label)
    if len(peaks) > 0:
        ax.plot(peaks, y[peaks], "x")

    if len(peak_ranges) == 4 and len(peak_ranges[0]) == len(peak_ranges[2]):
        pk_start = np.array(peak_ranges[0], dtype=int)
        pk_end   = np.array(peak_ranges[2], dtype=int)
        for s_tick, e_tick in zip(pk_start, pk_end):
            region_x = np.arange(s_tick, e_tick)
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
    channel_info, (r, s) = compute_channel_charges(adc, str(evt_title))

    for chn in range(NC):
        data_1d = adc[:, chn]
        largest_pk_idx, cluster_charge = channel_info[chn]

        if largest_pk_idx is None or len(r[chn]) == 0:
            out_prefix = os.path.join(out_dir, f"Chn{chn}")
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
            continue

        pk_start = r[chn][0][0]
        pk_top   = r[chn][0][1]
        pk_end   = r[chn][0][2]
        pk_int   = r[chn][0][3]

        idx_where = np.where(pk_top == largest_pk_idx)[0]
        if len(idx_where) == 0:
            single_peak_array = np.array([], dtype=int)
            single_pk_range   = ([], [], [], [])
        else:
            i_pk = idx_where[0]
            single_peak_array = np.array([largest_pk_idx], dtype=int)
            single_pk_range   = (
                [pk_start[i_pk]],
                [pk_top[i_pk]],
                [pk_end[i_pk]],
                [pk_int[i_pk]]
            )

        out_prefix = os.path.join(out_dir, f"Chn{chn}")
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
    if not os.path.exists(COSMIC_CSV_PATH):
        print(f"{output_align} No CSV found at {COSMIC_CSV_PATH}. Nothing to process.")
        return

    cosmic_df = pd.read_csv(COSMIC_CSV_PATH)
    rows = cosmic_df.to_dict(orient="records")

    cosmic_count = len(rows)
    print(f"{output_align} Found {cosmic_count} cosmic events in {COSMIC_CSV_PATH}. Computing charges...")

    charges_rows = []

    for row in rows:
        folder_name  = row["#Folder_name"]
        evt_id_str   = row["EvtID"]
        raw_file_nr  = row["Raw_file_nr"]
        raw_evt_nr   = row["Raw_Evt_nr"]
        json_fname   = row["File_nr"]
        evt_nr       = row["Evt_nr"]

        fullpath = os.path.join(DATA_FOLDER, json_fname)
        if not os.path.exists(fullpath):
            print(f"{output_align} Missing file {fullpath}; skip.")
            continue

        with open(fullpath, "r") as jsonf:
            data_all = json.load(jsonf)
        events = data_all["all"]

        raw_evt_idx = int(raw_evt_nr)
        if raw_evt_idx < 0 or raw_evt_idx >= len(events):
            print(f"{output_align} Invalid event_idx {raw_evt_idx} in {json_fname}; skip.")
            continue

        event_dict = events[raw_evt_idx]
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
            print(f"{output_align} Bad wave size in {json_fname}, Evt {raw_evt_idx}; skipping.")
            continue

        channel_info, _ = compute_channel_charges(adc, str(folder_name))

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

    if charges_rows:
        charges_df = pd.DataFrame(charges_rows)
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

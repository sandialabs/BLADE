#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BLADE Main Script (version 1.1)

This script implements the BLADE (Bolide Light-curve Analysis and Discrimination Explorer)
framework for automated processing and classification of digitized CNEOS fireball light curves.

Related publication:
"BLADE: An Automated Framework for Classifying Light Curves from the Center for
 Near-Earth Object Studies (CNEOS) Fireball Database", The Astronomical Journal
 doi: 10.3847/1538-3881/adeb55

License: MIT License
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from scipy.signal import savgol_filter, find_peaks

# User-adjustable parameters
METADATA_FILE        = "LC_processing.csv"
LIGHT_CURVE_FOLDER   = "Processed_LC"
OUTPUT_FOLDER        = "Output_LC_BLADE"
PICKS_FOLDER         = os.path.join(OUTPUT_FOLDER, "automated_picks")
PROGRESS_FILE        = os.path.join(OUTPUT_FOLDER, "processing_progress.txt")
SUMMARY_FILE         = os.path.join(OUTPUT_FOLDER, "summary.csv")

TIME_TOLERANCE       = 5       # seconds for matching filenames to metadata
FRAG_THRESHOLD       = 15      # sample-index threshold for continuous vs discrete fragmentation
PROMINENCE_THRESHOLD = 0.10    # in normalized-intensity units
HEIGHT_THRESHOLD     = 0.10    # in normalized-intensity units
MIN_DISTANCE         = 5       # minimum samples between peaks

# Enable or disable all plotting (True/False)
ENABLE_PLOTTING      = True

# Ensure output directories exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PICKS_FOLDER, exist_ok=True)

# Load and clean metadata
try:
    metadata_df = pd.read_csv(METADATA_FILE)
except Exception as e:
    print(f"Error loading metadata file '{METADATA_FILE}': {e}")
    metadata_df = None

REQUIRED_METADATA_COLS = [
    "UTC Year", "UTC Month", "UTC Day",
    "UTC Hour", "UTC Minute", "UTC Second",
    "Velocity [km/s]", "Entry Angle [deg]", "Bolide Altitude [km]",
    "Bolide Latitude [deg]", "Bolide Longitude [deg]", "Azimuth [deg]",
    "Total impact energy [kt]"
]

if metadata_df is not None:
    metadata_cleaned = metadata_df.dropna(subset=REQUIRED_METADATA_COLS, how="any").copy()
    metadata_cleaned["Event_Time"] = pd.to_datetime(
        metadata_cleaned["UTC Year"].astype(int).astype(str) + "-" +
        metadata_cleaned["UTC Month"].astype(int).astype(str).str.zfill(2) + "-" +
        metadata_cleaned["UTC Day"].astype(int).astype(str).str.zfill(2) + " " +
        metadata_cleaned["UTC Hour"].astype(int).astype(str).str.zfill(2) + ":" +
        metadata_cleaned["UTC Minute"].astype(int).astype(str).str.zfill(2) + ":" +
        metadata_cleaned["UTC Second"].astype(int).astype(str).str.zfill(2)
    )
else:
    metadata_cleaned = None


def preprocess_light_curve(intensities: np.ndarray) -> tuple:
    """
    Smooth the raw intensity series with an adaptive Savitzky-Golay filter,
    then normalize to [0, 1].
    """
    noise_level = np.std(np.diff(intensities, prepend=intensities[0]))
    if noise_level > 1.0:
        window_length = 15
        polyorder     = 2
    elif noise_level > 0.5:
        window_length = 21
        polyorder     = 3
    else:
        window_length = 31
        polyorder     = 3

    window_length = min(window_length, len(intensities) - 1)
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < 3:
        window_length = 3

    try:
        smoothed = savgol_filter(intensities, window_length=window_length, polyorder=polyorder)
    except Exception:
        smoothed = intensities.copy()

    min_val = np.min(smoothed)
    max_val = np.max(smoothed)
    if max_val != min_val:
        normalized = (smoothed - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(smoothed)

    return smoothed, normalized


def calculate_altitude_vector(times: np.ndarray, peak_time: float,
                              peak_altitude_m: float, velocity_m_s: float,
                              entry_angle_deg: float) -> np.ndarray:
    """
    Compute altitude (in meters) at each timestamp, assuming a linear vertical
    velocity component:
      alt(t) = peak_altitude_m - velocity_m_s * (t - peak_time) * sin(entry_angle)
    """
    times_shifted = times - peak_time
    theta = np.radians(entry_angle_deg)
    altitudes_m = peak_altitude_m - velocity_m_s * times_shifted * np.sin(theta)
    return altitudes_m


def detect_peaks(normalized_intensities: np.ndarray) -> tuple:
    """
    Identify peaks using SciPy's find_peaks with configured thresholds.
    """
    peaks, properties = find_peaks(
        normalized_intensities,
        prominence=PROMINENCE_THRESHOLD,
        height=HEIGHT_THRESHOLD,
        distance=MIN_DISTANCE
    )
    return peaks, properties


def classify_event(times: np.ndarray, normalized_intensities: np.ndarray,
                   peaks: np.ndarray, gradient: np.ndarray) -> str:
    """
    Classification rules:
      - No peaks: "No Significant Peaks"
      - Multiple peaks:
          if any adjacent-peak index difference < FRAG_THRESHOLD: "Continuous Fragmentation"
          else: "Discrete Fragmentation"
      - Single peak:
          rise_time  = t_peak - t_start
          fall_time  = t_end - t_peak
          max_grad   = max abs gradient
          if (rise_time < 0.5 * fall_time) and (max_grad > 0.5): "Airburst"
          else: "Single Peak"
    """
    n_peaks = len(peaks)
    if n_peaks == 0:
        return "No Significant Peaks"
    elif n_peaks > 1:
        intervals = np.diff(peaks)
        if np.min(intervals) < FRAG_THRESHOLD:
            return "Continuous Fragmentation"
        else:
            return "Discrete Fragmentation"
    else:
        idx = peaks[0]
        t_peak = times[idx]
        rise_time = t_peak - times[0]
        fall_time = times[-1] - t_peak
        max_grad = np.max(np.abs(gradient))
        if (rise_time < 0.5 * fall_time) and (max_grad > 0.5):
            return "Airburst"
        else:
            return "Single Peak"


def process_light_curve(file_path: str, metadata: pd.DataFrame) -> tuple:
    """
    Process a single digitized light-curve CSV, generating all required outputs.
    """
    try:
        file_name = os.path.basename(file_path)
        match = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{6})\.csv$", file_name)
        if not match:
            raise ValueError(f"Invalid filename format: {file_name}")

        year, month, day, hhmmss = match.groups()
        event_time = datetime(
            year=int(year), month=int(month), day=int(day),
            hour=int(hhmmss[:2]), minute=int(hhmmss[2:4]), second=int(hhmmss[4:])
        )

        folder_name = f"{year}{month}{day}_{hhmmss}"
        event_folder = os.path.join(OUTPUT_FOLDER, folder_name)
        os.makedirs(event_folder, exist_ok=True)

        # Match metadata row within TIME_TOLERANCE seconds
        matched_row = None
        if metadata is not None and not metadata.empty:
            start_window = event_time - timedelta(seconds=TIME_TOLERANCE)
            end_window   = event_time + timedelta(seconds=TIME_TOLERANCE)
            subset = metadata[
                (metadata["Event_Time"] >= start_window) &
                (metadata["Event_Time"] <= end_window)
            ]
            if not subset.empty:
                matched_row = subset.iloc[0]

        # Read raw CSV
        lc_df = pd.read_csv(file_path)
        if "Time [s]" not in lc_df.columns or "Intensity [W/sr]" not in lc_df.columns:
            raise KeyError("CSV missing 'Time [s]' or 'Intensity [W/sr]' columns.")

        times = lc_df["Time [s]"].to_numpy(dtype=float)
        intensities = lc_df["Intensity [W/sr]"].to_numpy(dtype=float)

        # Plot raw Intensity vs Time
        if ENABLE_PLOTTING:
            plt.figure(figsize=(8, 6))
            plt.plot(times, intensities, color="blue")
            plt.xlabel("Time [s]", fontsize=14)
            plt.ylabel("Intensity [W/sr]", fontsize=14)
            plt.title(event_time.strftime("%Y-%m-%d %H:%M:%S UTC"), fontsize=16)
            plt.grid(linestyle=":", linewidth=0.5)
            raw_plot_path = os.path.join(event_folder, f"{folder_name}_raw_intensity.png")
            plt.tight_layout()
            plt.savefig(raw_plot_path, dpi=300)
            plt.close()

        # Smooth and normalize
        smoothed, normalized = preprocess_light_curve(intensities)

        # Plot normalized Intensity vs Time
        if ENABLE_PLOTTING:
            plt.figure(figsize=(8, 6))
            plt.plot(times, normalized, color="blue")
            plt.xlabel("Time [s]", fontsize=14)
            plt.ylabel("Normalized Intensity", fontsize=14)
            plt.title(event_time.strftime("%Y-%m-%d %H:%M:%S UTC"), fontsize=16)
            plt.grid(linestyle=":", linewidth=0.5)
            norm_plot_path = os.path.join(event_folder, f"{folder_name}_normalized_intensity.png")
            plt.tight_layout()
            plt.savefig(norm_plot_path, dpi=300)
            plt.close()

        # Compute gradient safely
        with np.errstate(divide="ignore", invalid="ignore"):
            gradient = np.gradient(normalized, times)
            gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)

        # Detect peaks
        peaks, properties = detect_peaks(normalized)
        peak_times = times[peaks]
        prominences = properties.get("prominences", np.array([0.0] * len(peaks)))

        # Create 2x1 figure: top=normalized with picks; bottom=bar plot of prominence
        if ENABLE_PLOTTING:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            # Top subplot: normalized curve + red dots at picks
            axes[0].plot(times, normalized, color="blue", label="Normalized Intensity")
            if peaks.size > 0:
                axes[0].scatter(peak_times, normalized[peaks], color="red", s=50, label="Picks")
            axes[0].set_ylabel("Normalized Intensity", fontsize=14)
            axes[0].set_title(event_time.strftime("%Y-%m-%d %H:%M:%S UTC"), fontsize=16)
            axes[0].grid(linestyle=":", linewidth=0.5)
            axes[0].legend(loc="upper right")

            # Bottom subplot: bar plot of prominence vs time (no title)
            if peaks.size > 0:
                full_width = times[-1] - times[0]
                bar_width  = full_width * 0.01 if full_width > 0 else 0.1
                axes[1].bar(peak_times, prominences, width=bar_width, color="green", edgecolor="black")
            axes[1].set_xlabel("Time [s]", fontsize=14)
            axes[1].set_ylabel("Prominence", fontsize=14)
            axes[1].grid(linestyle=":", linewidth=0.5)

            multi_plot_path = os.path.join(event_folder, f"{folder_name}_lc_prominence.png")
            plt.tight_layout()
            plt.savefig(multi_plot_path, dpi=300)
            plt.close()

        # Overlay plot: gradient + normalized + picks (no prominence)
        if ENABLE_PLOTTING:
            plt.figure(figsize=(10, 6))
            plt.plot(times, gradient, label="Gradient", color="green", linestyle="--")
            plt.plot(times, normalized, label="Normalized Intensity", color="blue")
            if peaks.size > 0:
                plt.scatter(peak_times, normalized[peaks], color="red", s=50, label="Detected peaks")
            plt.xlabel("Time [s]", fontsize=14)
            plt.ylabel("Normalized / Gradient", fontsize=14)
            plt.title(event_time.strftime("%Y-%m-%d %H:%M:%S UTC"), fontsize=16)
            plt.grid(linestyle=":", linewidth=0.5)
            plt.legend(loc="upper right", fontsize=12)

            overlay_plot_path = os.path.join(event_folder, f"{folder_name}_auto_picks.png")
            plt.tight_layout()
            plt.savefig(overlay_plot_path, dpi=300)
            plt.close()

        # Classify the event
        classification = classify_event(times, normalized, peaks, gradient)

        # Compute altitudes if metadata is present
        full_alt_km = None
        peak_alt_km = [None] * len(peaks)

        if matched_row is not None:
            altitude_km     = matched_row["Bolide Altitude [km]"]
            velocity_km_s   = matched_row["Velocity [km/s]"]
            entry_angle_deg = matched_row["Entry Angle [deg]"]

            if pd.notna(altitude_km) and pd.notna(velocity_km_s) and pd.notna(entry_angle_deg):
                peak_alt_m   = float(altitude_km) * 1000.0
                velocity_m_s = float(velocity_km_s) * 1000.0
                entry_angle  = float(entry_angle_deg)

                global_peak_idx = int(np.argmax(smoothed))
                peak_time = times[global_peak_idx]

                altitudes_m = calculate_altitude_vector(
                    times, peak_time, peak_alt_m, velocity_m_s, entry_angle
                )
                full_alt_km = altitudes_m / 1000.0

                for i, p in enumerate(peaks):
                    peak_alt_km[i] = float(altitudes_m[p] / 1000.0)

                # Plot Altitude vs Intensity (Y-X)
                if ENABLE_PLOTTING:
                    plt.figure(figsize=(6, 10))
                    plt.plot(intensities, full_alt_km, color="blue")
                    plt.xlabel("Intensity [W/sr]", fontsize=14)
                    plt.ylabel("Altitude [km]", fontsize=14)
                    plt.title(event_time.strftime("%Y-%m-%d %H:%M:%S UTC"), fontsize=16)
                    plt.grid(linestyle=":", linewidth=0.5)
                    alt_yx_path = os.path.join(event_folder, f"{folder_name}_alt_vs_intensity_yx.png")
                    plt.tight_layout()
                    plt.savefig(alt_yx_path, dpi=300)
                    plt.close()

                    # Plot Intensity vs Altitude (X-Y inverted)
                    plt.figure(figsize=(8, 6))
                    plt.plot(full_alt_km, intensities, color="blue")
                    plt.gca().invert_xaxis()
                    plt.xlabel("Altitude [km]", fontsize=14)
                    plt.ylabel("Intensity [W/sr]", fontsize=14)
                    plt.title(event_time.strftime("%Y-%m-%d %H:%M:%S UTC"), fontsize=16)
                    plt.grid(linestyle=":", linewidth=0.5)
                    alt_xy_path = os.path.join(event_folder, f"{folder_name}_intensity_vs_alt_xy.png")
                    plt.tight_layout()
                    plt.savefig(alt_xy_path, dpi=300)
                    plt.close()

        # Save picks CSV in both event folder and automated picks folder
        picks_header = "Peak Time [s],Original Intensity,Normalized Intensity,Peak Altitude [km],Prominence\n"
        picks_lines = []
        for i, p in enumerate(peaks):
            tpk  = times[p]
            orig = intensities[p]
            norm = normalized[p]
            alt  = peak_alt_km[i] if peak_alt_km[i] is not None else ""
            prom = prominences[i]
            picks_lines.append(f"{tpk:.6f},{orig:.6f},{norm:.6f},{alt},{prom:.6f}\n")

        # Write to event folder
        event_picks_csv = os.path.join(event_folder, f"{folder_name}_picks.csv")
        with open(event_picks_csv, "w") as fout:
            fout.write(picks_header)
            fout.writelines(picks_lines)

        # Write to automated picks folder
        auto_picks_csv = os.path.join(PICKS_FOLDER, f"{folder_name}_picks.csv")
        with open(auto_picks_csv, "w") as fout:
            fout.write(picks_header)
            fout.writelines(picks_lines)

        return True, file_name, classification

    except Exception as e:
        print(f"Error processing '{file_path}': {e}")
        return False, os.path.basename(file_path), ""


# Load the list of already processed files
completed_files = set()
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as pf:
        completed_files = set(line.strip() for line in pf if line.strip())

# Initialize summary CSV if not already present
if not os.path.exists(SUMMARY_FILE):
    with open(SUMMARY_FILE, "w") as sf:
        sf.write("File Name,Classification\n")

# Iterate over all light-curve CSV files
light_curve_files = [
    os.path.join(LIGHT_CURVE_FOLDER, fn)
    for fn in os.listdir(LIGHT_CURVE_FOLDER)
    if fn.lower().endswith(".csv")
]

processed_count = 0
skipped_count   = 0

for lc_path in sorted(light_curve_files):
    lc_name = os.path.basename(lc_path)
    if lc_name in completed_files:
        print(f"[SKIP] Already processed: {lc_name}")
        skipped_count += 1
        continue

    success, fname, classification = process_light_curve(lc_path, metadata_cleaned)
    if success:
        with open(PROGRESS_FILE, "a") as pf:
            pf.write(fname + "\n")
        completed_files.add(fname)
        with open(SUMMARY_FILE, "a") as sf:
            sf.write(f"{fname},{classification}\n")
        print(f"[DONE] {fname} -> {classification}")
        processed_count += 1
    else:
        print(f"[ERROR] Failed processing {fname}")
        skipped_count += 1

print("\nProcessing complete.")
print(f"  Files processed: {processed_count}")
print(f"  Files skipped:   {skipped_count}")

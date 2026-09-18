# -*- coding: utf-8 -*-
"""
BLADE LC Processor Script (version 1.0)

This script is an optional code complementing the BLADE 
(Bolide Light-curve Analysis and Discrimination Explorer)
framework for automated processing and classification of digitized CNEOS fireball light curves.

Related publication:
Silber and Sawal (2025) "BLADE: An Automated Framework for Classifying Light Curves from the Center for
 Near-Earth Object Studies Fireball Database", The Astronomical Journal, doi: 10.3847/1538-3881/adeb55"

License: MIT License
"""


import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Optional: import cartopy for map visualization (if installed)
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    cartopy_installed = True
except ImportError:
    cartopy_installed = False
    print("Cartopy not installed. Trajectory map will not include a geographical map.")

# ------------------------------
# PATH CONFIGURATION
# ------------------------------

# Metadata CSV (must be in same working directory or provide full path)
metadata_file = "LC_processing.csv"

# Folder containing individual light-curve CSV files named as: YYYYMMDD_HHMMSS.csv
light_curve_folder = "Processed_LC"

# Top-level output folder; each event will get its own subfolder here
output_folder = "Output_LC_plots"
os.makedirs(output_folder, exist_ok=True)

# ------------------------------
# LOAD METADATA (DO NOT DROP ROWS)
# ------------------------------

try:
    metadata = pd.read_csv(metadata_file)
except Exception as e:
    print("Error loading metadata file '{}': {}".format(metadata_file, e))
    exit()

# Build a single datetime column ("Event_Time") for matching
metadata["Event_Time"] = pd.to_datetime(
    metadata["UTC Year"].astype(int).astype(str) + "-" +
    metadata["UTC Month"].astype(int).astype(str).str.zfill(2) + "-" +
    metadata["UTC Day"].astype(int).astype(str).str.zfill(2) + " " +
    metadata["UTC Hour"].astype(int).astype(str).str.zfill(2) + ":" +
    metadata["UTC Minute"].astype(int).astype(str).str.zfill(2) + ":" +
    metadata["UTC Second"].astype(int).astype(str).str.zfill(2)
)

total_metadata_entries = metadata.shape[0]

# ------------------------------
# DEFINE LIGHT-CURVE PROCESSING FUNCTION
# ------------------------------

def analyze_light_curve_event(file_path, metadata, time_tolerance=5):
    """
    Processes one light-curve CSV. Filename must be YYYYMMDD_HHMMSS.csv.
    Returns: (status, file_name, reason)
      status=True if fully processed,
      else False and reason is one of:
        "no match", "trajectory data missing", "invalid filename"
    Steps:
    1. Parse timestamp from filename.
    2. Find matching metadata row within +/- time_tolerance seconds.
    3. If no match: plot intensity vs time; reason="no match".
    4. If matched but any of Velocity, Entry Angle, or Altitude is missing: reason="trajectory data missing".
    5. If all present: compute altitudes (km), save CSV, create plots; reason="processed".
    """

    file_name = os.path.basename(file_path)

    # (a) PARSE FILENAME & BUILD EVENT TIME
    match = re.match(r"(\d{4})(\d{2})(\d{2})_(\d{6})\.csv", file_name)
    if not match:
        print("Invalid filename '{}': expected YYYYMMDD_HHMMSS.csv".format(file_name))
        return False, file_name, "invalid filename"

    year, month, day, hhmmss = match.groups()
    try:
        event_time = datetime(
            year=int(year),
            month=int(month),
            day=int(day),
            hour=int(hhmmss[:2]),
            minute=int(hhmmss[2:4]),
            second=int(hhmmss[4:])
        )
    except ValueError:
        print("Invalid date/time in filename '{}'.".format(file_name))
        return False, file_name, "invalid filename"

    # Create subfolder name: "yyyyddmm_hhmmss"
    folder_name = "{}{}{}_{}".format(year, day, month, hhmmss)
    folder_path = os.path.join(output_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # (b) MATCH METADATA WITHIN +/- time_tolerance
    t_start = event_time - timedelta(seconds=time_tolerance)
    t_end = event_time + timedelta(seconds=time_tolerance)
    matched_df = metadata[
        (metadata["Event_Time"] >= t_start) &
        (metadata["Event_Time"] <= t_end)
    ]

    # If no metadata match, plot Intensity vs Time; reason="no match"
    if matched_df.empty:
        light_curve_data = pd.read_csv(file_path)
        times = light_curve_data["Time [s]"].values
        intensities = light_curve_data["Intensity [W/sr]"].values

        plt.figure()
        plt.plot(times, intensities, color="blue")
        plt.xlabel("Time [s]")
        plt.ylabel("Intensity [W/sr]")
        plt.title("{}-{}-{} {}:{}:{} UTC\n(no metadata match)".format(
            year, month, day, hhmmss[:2], hhmmss[2:4], hhmmss[4:]
        ))
        plt.grid(linestyle=":", linewidth=0.5)
        png_path = os.path.join(folder_path, "{}_intensity_vs_time.png".format(folder_name))
        plt.savefig(png_path, dpi=300)
        plt.close()

        return False, file_name, "no match"

    # Use the first matched row
    matched_row = matched_df.iloc[0]

    # (c) CHECK TRAJECTORY DATA: Velocity, Entry Angle, and Altitude must all exist
    velocity_km_s = matched_row.get("Velocity [km/s]", np.nan)
    entry_angle = matched_row.get("Entry Angle [deg]", np.nan)
    altitude_km = matched_row.get("Bolide Altitude [km]", np.nan)
    if pd.isna(velocity_km_s) or pd.isna(entry_angle) or pd.isna(altitude_km):
        light_curve_data = pd.read_csv(file_path)
        times = light_curve_data["Time [s]"].values
        intensities = light_curve_data["Intensity [W/sr]"].values

        plt.figure()
        plt.plot(times, intensities, color="blue")
        plt.xlabel("Time [s]")
        plt.ylabel("Intensity [W/sr]")
        plt.title("{}-{}-{} {}:{}:{} UTC\n(trajectory data missing)".format(
            year, month, day, hhmmss[:2], hhmmss[2:4], hhmmss[4:]
        ))
        plt.grid(linestyle=":", linewidth=0.5)
        png_path = os.path.join(folder_path, "{}_intensity_vs_time.png".format(folder_name))
        plt.savefig(png_path, dpi=300)
        plt.close()

        return False, file_name, "trajectory data missing"

    # All trajectory fields exist, proceed with full processing
    velocity_m_s = velocity_km_s * 1000.0  # convert to m/s
    azimuth = matched_row["Azimuth [deg]"]
    latitude = matched_row["Bolide Latitude [deg]"]
    longitude = matched_row["Bolide Longitude [deg]"]
    peak_altitude_m = altitude_km * 1000.0   # convert km to m

    # Read light-curve CSV
    light_curve_data = pd.read_csv(file_path)
    times = light_curve_data["Time [s]"].values
    intensities = light_curve_data["Intensity [W/sr]"].values

    # Compute altitudes (m -> km)
    peak_idx = np.argmax(intensities)
    times_shifted = times - times[peak_idx]
    altitudes_m = peak_altitude_m - velocity_m_s * times_shifted * np.sin(np.radians(entry_angle))
    altitudes_km = altitudes_m / 1000.0

    # Save Time, Intensity, Altitude[km] to CSV
    alt_df = pd.DataFrame({
        "Time [s]": times,
        "Intensity [W/sr]": intensities,
        "Altitude [km]": altitudes_km
    })
    csv_out_path = os.path.join(folder_path, "time_intensity_altitude.csv")
    alt_df.to_csv(csv_out_path, index=False)

    # Compute horizontal trajectory (optional map)
    earth_radius = 6371e3  # meters
    azimuth_rad = np.radians(azimuth)
    horizontal_distance = velocity_m_s * times_shifted * np.cos(np.radians(entry_angle))

    latitudes = []
    longitudes = []
    for dist in horizontal_distance:
        dlat = (dist / earth_radius) * np.cos(azimuth_rad)
        dlon = (dist / (earth_radius * np.cos(np.radians(latitude)))) * np.sin(azimuth_rad)
        latitudes.append(latitude + np.degrees(dlat))
        longitudes.append(longitude + np.degrees(dlon))

    # Create plot titles using UTC date/time
    title_info = "{}-{}-{} {}:{}:{} UTC".format(
        year, month, day, hhmmss[:2], hhmmss[2:4], hhmmss[4:]
    )

    # 1) Intensity vs. Time
    plt.figure()
    plt.plot(times, intensities, color="blue")
    plt.xlabel("Time [s]")
    plt.ylabel("Intensity [W/sr]")
    plt.title("Light Curve: " + title_info)
    plt.grid(linestyle=":", linewidth=0.5)
    png1 = os.path.join(folder_path, "{}_intensity_vs_time.png".format(folder_name))
    plt.savefig(png1, dpi=300)
    plt.close()

    # 2) Altitude vs. Intensity (Y vs. X)
    plt.figure(figsize=(6, 10))
    plt.plot(intensities, altitudes_km, color="green")
    plt.xlabel("Intensity [W/sr]")
    plt.ylabel("Altitude [km]")
    plt.title("Altitude vs. Intensity: " + title_info)
    plt.grid(linestyle=":", linewidth=0.5)
    png2 = os.path.join(folder_path, "{}_altitude_vs_intensity_yx.png".format(folder_name))
    plt.savefig(png2, dpi=300)
    plt.close()

    # 3) Altitude vs. Intensity (X vs. Y), inverted x-axis
    plt.figure()
    plt.plot(altitudes_km, intensities, color="red")
    plt.gca().invert_xaxis()
    plt.xlabel("Altitude [km]")
    plt.ylabel("Intensity [W/sr]")
    plt.title("Intensity vs. Altitude: " + title_info)
    plt.grid(linestyle=":", linewidth=0.5)
    png3 = os.path.join(folder_path, "{}_altitude_vs_intensity_xy.png".format(folder_name))
    plt.savefig(png3, dpi=300)
    plt.close()

    # 4) Optional: Cartopy trajectory map
    if cartopy_installed:
        plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([
            min(longitudes) - 10, max(longitudes) + 10,
            min(latitudes)  - 10, max(latitudes)  + 10
        ])
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAND, edgecolor="black")
        ax.add_feature(cfeature.LAKES, edgecolor="black")
        ax.add_feature(cfeature.RIVERS)
        plt.plot(longitudes, latitudes, color="blue", label="Trajectory", transform=ccrs.PlateCarree())
        plt.scatter(longitude, latitude, color="red", s=50, label="Peak Brightness", transform=ccrs.PlateCarree())
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        plt.title("Ground-Track: " + title_info)
        plt.legend(loc="lower left")
        png4 = os.path.join(folder_path, "{}_trajectory_map.png".format(folder_name))
        plt.savefig(png4, dpi=300)
        plt.close()

    print("Processed '{}' -> saved all outputs to '{}'".format(file_name, folder_path))
    return True, file_name, "processed"

# ------------------------------
# MAIN LOOP: PROCESS ALL LIGHT-CURVE FILES
# ------------------------------

processed_count = 0
skipped_no_match = 0
skipped_trajectory_missing = 0
processed_events = []
skipped_events = []

light_curve_files = [
    os.path.join(light_curve_folder, fname)
    for fname in os.listdir(light_curve_folder)
    if fname.lower().endswith(".csv")
]
num_lightcurves = len(light_curve_files)

for lc_path in light_curve_files:
    success, fname, reason = analyze_light_curve_event(lc_path, metadata, time_tolerance=5)
    if success:
        processed_count += 1
        processed_events.append(fname)
    else:
        skipped_events.append(fname)
        if reason == "no match":
            skipped_no_match += 1
        elif reason == "trajectory data missing":
            skipped_trajectory_missing += 1

# ------------------------------
# WRITE SUMMARY FILE
# ------------------------------

summary_path = os.path.join(output_folder, "processing_summary.txt")
with open(summary_path, "w") as f:
    f.write("Processing Summary\n")
    f.write("==================\n")
    f.write("Total metadata entries:                   {}\n".format(total_metadata_entries))
    f.write("Total light-curve files found:            {}\n\n".format(num_lightcurves))
    f.write("Total fully processed:                    {}\n".format(processed_count))
    f.write("Skipped (no metadata match):              {}\n".format(skipped_no_match))
    f.write("Skipped (trajectory data missing):        {}\n\n".format(skipped_trajectory_missing))
    f.write("Fully Processed Files:\n")
    for evt in processed_events:
        f.write("  {}\n".format(evt))
    f.write("\nPlotted-only Files:\n")
    for evt in skipped_events:
        f.write("  {}\n".format(evt))

print("\nAll done! Summary written to '{}'.".format(summary_path))
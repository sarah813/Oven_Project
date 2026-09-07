"""Analyze and plot CO/CO2 measurements from the oven exhaust."""

from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

# Plot switches
SHOW_CO_EXHAUST_PLOT = True
SHOW_CO2_CARBON_PLOT = True
SHOW_CO2_CO_PERCENTAGE_PLOT = True
SHOW_CO2_FLOW_VS_CO_CONCENTRATION_PLOT = True
SHOW_ALL_DATA_BAR_PLOT = True
SAVE_PLOTS = True

OUTPUT_DIR = Path("oven_outputs")
PLOT_DPI = 300

# Molar volume at the selected gas temperature and pressure.
MOLAR_VOLUME_L_PER_MOL = 24.47
CARBON_MOLAR_MASS_G_PER_MOL = 12.011

# -----------------------------
# Plot 1: CO concentration and CO2 throughput during exhaust measurements
# -----------------------------

def calculate_co_throughput_l_min(exhaust_flow_l_min, co_concentration_ppm):
    """Calculate volumetric CO throughput from exhaust flow and CO concentration."""
    return np.asarray(exhaust_flow_l_min) * np.asarray(co_concentration_ppm) * 1e-6


def volumetric_to_molar_throughput(volumetric_flow_l_min, molar_volume_l_mol):
    """Convert volumetric throughput [L/min] to molar throughput [mol/min]."""
    if molar_volume_l_mol <= 0:
        raise ValueError("Molar volume must be greater than zero.")
    return np.asarray(volumetric_flow_l_min) / molar_volume_l_mol


# The dates below are placeholders so that measurements from "the next day"
# appear with the correct overnight spacing. Only the day and clock time are
# shown in the plot.
measurement_day = datetime(2026, 1, 1)

# For ppm ranges, the upper value is used, matching the original notes.
# The first two rows use 0.30 L/min because the supplied example calculations
# were explicitly based on 0.30 L/min, although the heading states 0.25 L/min.
co_exhaust_measurements = [
    # time, CO2 flow [L/min], CO [ppm], note
    (measurement_day.replace(day=1,hour=9, minute=45), 0.30, 100, "Initial flow"),
    (measurement_day.replace(day= 1, hour=13, minute=35), 0.30, 70, "End of process"),
    (measurement_day.replace(day= 1, hour=13, minute=30), 0.50, 80, "Flow increased"),
    (measurement_day.replace(day= 1, hour=14, minute=30), 0.30, 60, "Flow decreased"),
    (measurement_day.replace(day= 1, hour=14, minute=50), 0.25, 70, "Back to 0.25 L/min"),
    (measurement_day.replace(day= 2, hour=9, minute=0), 0.25, 42, ""),
    (measurement_day.replace(day= 2, hour=9, minute=15), 1.00, 125, ""),
    (measurement_day.replace(day=2, hour=10, minute=0), 1.00, 70, ""),
    (measurement_day.replace(day=2, hour=10, minute=3), 0.90, 65, ""),
    (measurement_day.replace(day=2, hour=10, minute=6), 0.73, 60, ""),
    (measurement_day.replace(day=2, hour=10, minute=9), 0.57, 55, ""),
    (measurement_day.replace(day=2, hour=10, minute=12), 0.41, 48, ""),
    (measurement_day.replace(day=2, hour=10, minute=15), 0.25, 37, ""),
    (measurement_day.replace(day=2, hour=12, minute=30), 1.00, 42, ""),
    (measurement_day.replace(day=2, hour=13, minute=45), 1.00, 40, ""),
    (measurement_day.replace(day=2, hour=14, minute=45), 1.00, 40, ""),
    (measurement_day.replace(day=2, hour=16, minute=40), 1.00, 9, "T=840°C"),
]

# Keep the plot chronological even if the notes were entered out of order.
co_exhaust_measurements.sort(key=lambda row: row[0])

# Arrays containing every recorded measurement from both days.
all_measurement_times = [row[0] for row in co_exhaust_measurements]
all_co2_flow_l_min = np.array([row[1] for row in co_exhaust_measurements])
all_co_ppm = np.array([row[2] for row in co_exhaust_measurements])
all_co_throughput_l_min = calculate_co_throughput_l_min(
    all_co2_flow_l_min,
    all_co_ppm,
)
all_co_molar_throughput_mol_min = volumetric_to_molar_throughput(
    all_co_throughput_l_min,
    MOLAR_VOLUME_L_PER_MOL,
)
all_carbon_molar_throughput_mol_min = all_co_molar_throughput_mol_min / 2
all_carbon_mass_throughput_g_min = (
    all_carbon_molar_throughput_mol_min * CARBON_MOLAR_MASS_G_PER_MOL
)

# Show only measurements from day 2 between 10:00 and 10:15, inclusive.
day_2_measurements = [
    row
    for row in co_exhaust_measurements
    if (
        row[0].day == 2
        and datetime.min.time().replace(hour=10)
        <= row[0].time()
        <= datetime.min.time().replace(hour=10, minute=15)
    )
]

measurement_times = [row[0] for row in day_2_measurements]
co2_flow_l_min = np.array([row[1] for row in day_2_measurements])
co_ppm = np.array([row[2] for row in day_2_measurements])

# Assuming the CO2 flow approximates the total exhaust flow.
co_throughput_l_min = calculate_co_throughput_l_min(co2_flow_l_min, co_ppm)
co_molar_throughput_mol_min = volumetric_to_molar_throughput(
    co_throughput_l_min,
    MOLAR_VOLUME_L_PER_MOL,
)
carbon_molar_throughput_mol_min = co_molar_throughput_mol_min / 2
carbon_mass_throughput_g_min = (
    carbon_molar_throughput_mol_min * CARBON_MOLAR_MASS_G_PER_MOL
)
co_percentage_of_co2 = co_throughput_l_min / co2_flow_l_min * 100

fig_co_exhaust, ax_co2 = plt.subplots(figsize=(13.5, 5.5))
ax_co = ax_co2.twinx()
ax_co_throughput = ax_co2.twinx()
ax_co_molar = ax_co2.twinx()
ax_co_throughput.spines["right"].set_position(("axes", 1.09))
ax_co_molar.spines["right"].set_position(("axes", 1.19))

co2_line = ax_co2.plot(
    measurement_times,
    co2_flow_l_min,
    color="#7FA9C4",
    marker="s",
    linestyle="-",
    linewidth=0.7,
    markersize=7,
    label="CO2 throughput [L/min]",
)
co_line = ax_co.plot(
    measurement_times,
    co_ppm,
    color="#C98C8C",
    marker="o",
    linestyle="none",
    markersize=7,
    label="CO concentration [ppm]",
)
co_throughput_line = ax_co_throughput.plot(
    measurement_times,
    co_throughput_l_min,
    color="#914D4D",
    marker="x",
    linestyle="none",
    markersize=8,
    markeredgewidth=1.8,
    label="CO concentration [L/min]",
)
co_molar_line = ax_co_molar.plot(
    measurement_times,
    co_molar_throughput_mol_min,
    color="#76638A",
    marker="o",
    markerfacecolor="none",
    markeredgewidth=1.8,
    linestyle="none",
    markersize=8,
    label="CO molar throughput [mol/min]",
)
carbon_molar_line = ax_co_molar.plot(
    measurement_times,
    carbon_molar_throughput_mol_min,
    color="#5F8F65",
    marker="o",
    markerfacecolor="none",
    markeredgewidth=1.8,
    linestyle="-",
    linewidth=0.7,
    markersize=8,
    label="C molar throughput [mol/min]",
)

ax_co2.set_xlabel("Measurement time")
ax_co2.set_ylabel("CO2 throughput [L/min]", color="#6289A3")
ax_co.set_ylabel("CO concentration [ppm]", color="#A66F6F")
ax_co_throughput.set_ylabel("CO concentration [L/min]", color="#914D4D")
ax_co_molar.set_ylabel("CO / C molar throughput [mol/min]", color="black")
ax_co2.tick_params(axis="y", labelcolor="#6289A3")
ax_co.tick_params(axis="y", labelcolor="#A66F6F")
ax_co_throughput.tick_params(axis="y", labelcolor="#914D4D")
ax_co_molar.tick_params(axis="y", labelcolor="black")
ax_co_molar.spines["right"].set_color("black")
ax_co2.set_ylim(bottom=0)
ax_co.set_ylim(bottom=0)
ax_co_throughput.set_ylim(bottom=0)
ax_co_molar.set_ylim(bottom=0)
ax_co_molar.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax_co2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax_co2.grid(True, alpha=0.3)
ax_co2.set_title("CO Concentration and CO2 Throughput Through the Oven — Day 2")

lines = (
    list(co2_line)
    + co_line
    + co_throughput_line
    + co_molar_line
    + carbon_molar_line
)
ax_co2.legend(lines, [line.get_label() for line in lines], loc="lower left")

fig_co_exhaust.autofmt_xdate(rotation=0, ha="center")
fig_co_exhaust.subplots_adjust(
    left=0.07,
    right=0.78,
    bottom=0.16,
    top=0.90,
)

# -----------------------------
# Plot 2: CO2 flow and carbon molar throughput
# -----------------------------

fig_co2_carbon, ax_co2_simple = plt.subplots(figsize=(10, 5.5))
ax_carbon_simple = ax_co2_simple.twinx()
ax_carbon_mass_simple = ax_co2_simple.twinx()
ax_carbon_mass_simple.spines["right"].set_position(("axes", 1.14))

co2_simple_line = ax_co2_simple.plot(
    measurement_times,
    co2_flow_l_min,
    color="#7FA9C4",
    marker="o",
    linestyle="none",
    markersize=7,
    label="CO2 throughput [L/min]",
)
carbon_simple_line = ax_carbon_simple.plot(
    measurement_times,
    carbon_molar_throughput_mol_min,
    color="#5F8F65",
    marker="o",
    markerfacecolor="none",
    markeredgewidth=1.8,
    linestyle="none",
    markersize=8,
    label="C molar throughput [mol/min]",
)
carbon_mass_simple_line = ax_carbon_mass_simple.plot(
    measurement_times,
    carbon_mass_throughput_g_min,
    color="#356B3D",
    marker="s",
    linestyle="none",
    markersize=7,
    label="C mass throughput [g/min]",
)

ax_co2_simple.set_xlabel("Measurement time")
ax_co2_simple.set_ylabel("CO2 throughput [L/min]", color="#6289A3")
ax_carbon_simple.set_ylabel("C molar throughput [mol/min]", color="#5F8F65")
ax_carbon_mass_simple.set_ylabel("C mass throughput [g/min]", color="#356B3D")
ax_co2_simple.tick_params(axis="y", labelcolor="#6289A3")
ax_carbon_simple.tick_params(axis="y", labelcolor="#5F8F65")
ax_carbon_mass_simple.tick_params(axis="y", labelcolor="#356B3D")
ax_co2_simple.set_ylim(bottom=0)
ax_carbon_simple.set_ylim(bottom=0)
ax_carbon_mass_simple.set_ylim(bottom=0)
ax_carbon_simple.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax_carbon_mass_simple.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax_co2_simple.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax_co2_simple.grid(True, alpha=0.3)
ax_co2_simple.set_title("CO2 Throughput and Carbon Molar Throughput - Day 2")

simple_lines = co2_simple_line + carbon_simple_line + carbon_mass_simple_line
ax_co2_simple.legend(
    simple_lines,
    [line.get_label() for line in simple_lines],
    loc="lower left",
)

fig_co2_carbon.autofmt_xdate(rotation=0, ha="center")
fig_co2_carbon.tight_layout(rect=[0, 0, 0.86, 1])

# -----------------------------
# Plot 3: CO2 flow and CO percentage of CO2 flow
# -----------------------------

fig_co2_co_percentage, ax_co2_percentage = plt.subplots(figsize=(10, 5.5))
ax_co_percentage = ax_co2_percentage.twinx()

co2_percentage_line = ax_co2_percentage.plot(
    measurement_times,
    co2_flow_l_min,
    color="#7FA9C4",
    marker="s",
    linestyle="-",
    linewidth=0.7,
    markersize=7,
    label="CO2 throughput [L/min]",
)
co_percentage_line = ax_co_percentage.plot(
    measurement_times,
    co_percentage_of_co2,
    color="#914D4D",
    marker="o",
    markerfacecolor="none",
    markeredgewidth=1.8,
    linestyle="-",
    linewidth=0.7,
    markersize=8,
    label="CO concentration [% of CO2 flow]",
)

ax_co2_percentage.set_xlabel("Measurement time")
ax_co2_percentage.set_ylabel("CO2 throughput [L/min]", color="#6289A3")
ax_co_percentage.set_ylabel("CO concentration [% of CO2 flow]", color="#914D4D")
ax_co2_percentage.tick_params(axis="y", labelcolor="#6289A3")
ax_co_percentage.tick_params(axis="y", labelcolor="#914D4D")
ax_co2_percentage.set_ylim(bottom=0)
ax_co_percentage.set_ylim(0, 0.01)
ax_co2_percentage.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax_co2_percentage.grid(True, alpha=0.3)
ax_co2_percentage.set_title("CO2 Throughput and CO Percentage - Day 2")

percentage_lines = co2_percentage_line + co_percentage_line
ax_co2_percentage.legend(
    percentage_lines,
    [line.get_label() for line in percentage_lines],
    loc="lower left",
)

fig_co2_co_percentage.autofmt_xdate(rotation=0, ha="center")
fig_co2_co_percentage.tight_layout()

# -----------------------------
# Plot 4: CO2 flow versus CO concentration
# -----------------------------

fig_co2_vs_co_concentration, ax_co2_vs_co = plt.subplots(figsize=(11, 5.5))
ax_co_production = ax_co2_vs_co.twinx()
ax_c_consumption = ax_co2_vs_co.twinx()
ax_c_consumption.spines["right"].set_position(("axes", 1.15))

co_concentration_scatter = ax_co2_vs_co.plot(
    co2_flow_l_min,
    co_ppm,
    color="#C98C8C",
    marker="o",
    linestyle="none",
    markersize=8,
    label="CO concentration [ppm]",
)
co_production_scatter = ax_co_production.plot(
    co2_flow_l_min,
    co_molar_throughput_mol_min,
    color="#76638A",
    marker="o",
    markerfacecolor="none",
    markeredgewidth=1.8,
    linestyle="none",
    markersize=8,
    label="CO production [mol/min]",
)
c_consumption_scatter = ax_c_consumption.plot(
    co2_flow_l_min,
    carbon_molar_throughput_mol_min,
    color="#5F8F65",
    marker="s",
    linestyle="-",
    linewidth=0.7,
    markersize=7,
    label="C consumption [mol/min]",
)

ax_co2_vs_co.set_xlabel("CO2 flow [L/min]")
ax_co2_vs_co.set_ylabel("CO concentration [ppm]", color="#A66F6F")
ax_co_production.set_ylabel("CO production [mol/min]", color="#76638A")
ax_c_consumption.set_ylabel("C consumption [mol/min]", color="#5F8F65")
ax_co2_vs_co.tick_params(axis="y", labelcolor="#A66F6F")
ax_co_production.tick_params(axis="y", labelcolor="#76638A")
ax_c_consumption.tick_params(axis="y", labelcolor="#5F8F65")
ax_co2_vs_co.set_xlim(left=0)
ax_co2_vs_co.set_ylim(bottom=0)
ax_co_production.set_ylim(bottom=0)
ax_c_consumption.set_ylim(bottom=0)
ax_co_production.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax_c_consumption.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax_co2_vs_co.grid(True, alpha=0.3)
ax_co2_vs_co.set_title(
    "CO Concentration, CO Production, and C Consumption vs CO2 Flow"
)

co2_flow_plot_lines = (
    co_concentration_scatter
    + co_production_scatter
    + c_consumption_scatter
)
ax_co2_vs_co.legend(
    co2_flow_plot_lines,
    [line.get_label() for line in co2_flow_plot_lines],
    loc="upper left",
)

fig_co2_vs_co_concentration.tight_layout(rect=[0, 0, 0.86, 1])

# -----------------------------
# Plot 5: selected values as overlapping bars with separate axes
# -----------------------------

measurement_numbers = np.arange(1, len(all_measurement_times) + 1)
measurement_labels = [
    f"M{number}\nDay {timestamp.day} {timestamp:%H:%M}"
    for number, timestamp in zip(measurement_numbers, all_measurement_times)
]

fig_all_data_bars, ax_all_co2_bars = plt.subplots(figsize=(13, 6.5))
ax_all_co_bars = ax_all_co2_bars.twinx()
ax_all_carbon_bars = ax_all_co2_bars.twinx()
ax_all_carbon_bars.spines["right"].set_position(("axes", 1.12))

# Keep all axes transparent so overlapping bars from the other axes remain visible.
ax_all_co_bars.patch.set_visible(False)
ax_all_carbon_bars.patch.set_visible(False)

co2_bars = ax_all_co2_bars.bar(
    measurement_numbers,
    all_co2_flow_l_min,
    width=0.82,
    color="#7FA9C4",
    alpha=0.78,
    label="CO2 volumetric flow rate [L/min]",
    zorder=1,
)
co_bars = ax_all_co_bars.bar(
    measurement_numbers,
    all_co_ppm,
    width=0.58,
    color="#C98C8C",
    alpha=0.82,
    label="CO outlet concentration [ppm]",
    zorder=2,
)
carbon_bars = ax_all_carbon_bars.bar(
    measurement_numbers,
    all_carbon_molar_throughput_mol_min,
    width=0.34,
    color="#5F8F65",
    alpha=0.88,
    label="C molar throughput [mol/min]",
    zorder=3,
)

ax_all_co2_bars.set_title(
    "Oven Exhaust Measurements and Carbon Molar Throughput"
)
ax_all_co2_bars.set_xlabel("Measurement")
ax_all_co2_bars.set_ylabel("CO2 volumetric flow rate [L/min]", color="#6289A3")
ax_all_co_bars.set_ylabel("CO outlet concentration [ppm]", color="#A66F6F")
ax_all_carbon_bars.set_ylabel("C molar throughput [mol/min]", color="#5F8F65")

ax_all_co2_bars.tick_params(axis="y", labelcolor="#6289A3")
ax_all_co_bars.tick_params(axis="y", labelcolor="#A66F6F")
ax_all_carbon_bars.tick_params(axis="y", labelcolor="#5F8F65")

ax_all_co2_bars.set_ylim(bottom=0)
ax_all_co_bars.set_ylim(bottom=0)
ax_all_carbon_bars.set_ylim(bottom=0)
ax_all_carbon_bars.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

ax_all_co2_bars.set_xticks(measurement_numbers)
ax_all_co2_bars.set_xticklabels(measurement_labels, rotation=45, ha="right")
ax_all_co2_bars.grid(True, axis="y", alpha=0.25)
ax_all_co2_bars.legend(
    handles=[co2_bars, co_bars, carbon_bars],
    loc="upper left",
)

fig_all_data_bars.subplots_adjust(
    left=0.08,
    right=0.78,
    bottom=0.24,
    top=0.90,
)


# Save every enabled plot whenever the script runs.
plots_to_save = [
    (
        SHOW_CO_EXHAUST_PLOT,
        fig_co_exhaust,
        "co_exhaust_day_2_10_00_to_10_15.png",
    ),
    (
        SHOW_CO2_CARBON_PLOT,
        fig_co2_carbon,
        "co2_and_carbon_throughput_day_2.png",
    ),
    (
        SHOW_CO2_CO_PERCENTAGE_PLOT,
        fig_co2_co_percentage,
        "co2_throughput_and_co_percentage_day_2.png",
    ),
    (
        SHOW_CO2_FLOW_VS_CO_CONCENTRATION_PLOT,
        fig_co2_vs_co_concentration,
        "co2_flow_vs_co_concentration_day_2.png",
    ),
    (
        SHOW_ALL_DATA_BAR_PLOT,
        fig_all_data_bars,
        "all_exhaust_measurements_bar_plots.png",
    ),
]

if SAVE_PLOTS:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for plot_enabled, figure, filename in plots_to_save:
        if plot_enabled:
            output_path = OUTPUT_DIR / filename
            figure.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
            print(f"Plot saved to: {output_path}")

# Close disabled figures before showing the enabled plots.
if not SHOW_CO_EXHAUST_PLOT:
    plt.close(fig_co_exhaust)
if not SHOW_CO2_CARBON_PLOT:
    plt.close(fig_co2_carbon)
if not SHOW_CO2_CO_PERCENTAGE_PLOT:
    plt.close(fig_co2_co_percentage)
if not SHOW_CO2_FLOW_VS_CO_CONCENTRATION_PLOT:
    plt.close(fig_co2_vs_co_concentration)
if not SHOW_ALL_DATA_BAR_PLOT:
    plt.close(fig_all_data_bars)

# Create all figures first, then display them together.
plt.show()

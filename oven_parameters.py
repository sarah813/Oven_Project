import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path

# -----------------------------
# Plot switches
# -----------------------------

SHOW_OVEN_TEMPERATURE_PLOT = False
SHOW_CO_EXHAUST_PLOT = True
SHOW_CO2_CARBON_PLOT = True
SHOW_CO2_CO_PERCENTAGE_PLOT = True
SHOW_CO2_FLOW_VS_CO_CONCENTRATION_PLOT = True
SHOW_ALL_DATA_BAR_PLOT = True
SHOW_PLA_MASS_PLOT = False
SHOW_SINTERING_TEMPERATURE_PLOT = False
SAVE_PLOTS = True

OUTPUT_DIR = Path("oven_outputs")
PLOT_DPI = 300

# Molar volume at the selected gas temperature and pressure.
MOLAR_VOLUME_L_PER_MOL = 24.47
CARBON_MOLAR_MASS_G_PER_MOL = 12.011

# -----------------------------
# Temperature profile
# -----------------------------

start_temp = 15
target_temp = 482
pla_pyrolysis_temp = 280

ramp_rate = 55.6
hold_time = 4
cooling_rate = 60

ramp_time = (target_temp - start_temp) / ramp_rate
t_280_up = (pla_pyrolysis_temp - start_temp) / ramp_rate

t_cooling_start = ramp_time + hold_time
cooling_time = (target_temp - start_temp) / cooling_rate
end_time = t_cooling_start + cooling_time

t_280_down = t_cooling_start + (target_temp - pla_pyrolysis_temp) / cooling_rate

time = np.linspace(0, end_time, 3000)

temp = np.zeros_like(time)

for i, t in enumerate(time):
    if t <= ramp_time:
        temp[i] = start_temp + ramp_rate * t
    elif t <= t_cooling_start:
        temp[i] = target_temp
    else:
        temp[i] = target_temp - cooling_rate * (t - t_cooling_start)

# -----------------------------
# Plot 1 (Temperature profile)
# -----------------------------

fig_oven_temperature = plt.figure(figsize=(9, 5))
plt.plot(time, temp, color="tab:blue", label="Oven temperature profile")

plt.axvspan(
    t_280_up,
    t_280_down,
    color="red",
    alpha=0.15,
    label="PLA pyrolysis / debinding region"
)

plt.xlabel("Time [h]")
plt.ylabel("Temperature [°C]")
plt.title("Temperature Profile for PLA Debinding in a Nabertherm Top 140 Oven")
plt.grid(True)
plt.legend()
plt.tight_layout()

# -----------------------------
# Plot 4: CO concentration and CO2 throughput during exhaust measurements
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
# Plot 5: CO2 flow and carbon molar throughput
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
# Plot 6: CO2 flow and CO percentage of CO2 flow
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
# Plot 7: CO2 flow versus CO concentration
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
# Plot 8: selected values as overlapping bars with separate axes
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

# -----------------------------
# PLA mass model
# -----------------------------

def gaussian(T, center, sigma, area):
    return area / (sigma * np.sqrt(2 * np.pi)) * np.exp(
        -0.5 * ((T - center) / sigma) ** 2
    )

# Approximate PLA decomposition modeled as two Gaussian DTG peaks:
# dtg1 = main PLA pyrolysis peak centered at 362.5 °C
#        width (sigma) = 23 °C
#        total mass loss contribution = 90.04 wt%
#
# dtg2 = secondary residue / char decomposition peak centered at 462.9 °C
#        width (sigma) = 25 °C
#        total mass loss contribution = 8.327 wt%
#
# Values are based on literature TGA data and simplified for visualization.

dtg1 = gaussian(temp, 362.5, 23, 90.04)
dtg2 = gaussian(temp, 462.9, 25, 8.327)

dtg_total = dtg1 + dtg2

dT_dt = np.gradient(temp, time)
mass_loss_rate = dtg_total * np.maximum(dT_dt, 0)
mass_loss_rate[time > t_cooling_start] = 0

dt = time[1] - time[0]
pla_mass_remaining = 100 - np.cumsum(mass_loss_rate) * dt
pla_mass_remaining = np.clip(pla_mass_remaining, 0, 100)

# Convert pure-PLA loss to Cu-PLA composite loss.
pla_volume_fraction = 0.1105
cu_volume_fraction = 0.8895
rho_pla = 1.24
rho_cu = 8.96
w_pla = (pla_volume_fraction * rho_pla) / (
    pla_volume_fraction * rho_pla + cu_volume_fraction * rho_cu
)
composite_mass_remaining = 100 - (100 - pla_mass_remaining) * w_pla

# -----------------------------
# Plot 2: mass remaining vs time, with temperature comparison
# -----------------------------

heating_mask = time <= t_cooling_start
temp_heating = temp[heating_mask]
composite_mass_remaining_heating = composite_mass_remaining[heating_mask]

# Approximate F1 black curve visually traced from the reference image.
f1_temp_points = np.array([0, 150, 200, 230, 250, 280, 320, 360, 390, 420, 500, 600])
f1_mass_points = np.array([100, 100, 99.9, 98.8, 96.5, 92.0, 90.8, 90.2, 88.6, 88.0, 88.0, 88.2])
f1_reference = np.interp(temp_heating, f1_temp_points, f1_mass_points)

fig_pla_mass, (ax1, ax3) = plt.subplots(1, 2, figsize=(13, 5))

# main x-axis: time
ax1.plot(
    time,
    composite_mass_remaining,
    color="tab:green",
    linewidth=2,
    label="Cu-PLA mass remaining"
)

ax1.set_xlabel("Time [h]")
ax1.set_ylabel("Cu-PLA mass remaining [wt%]")
ax1.set_title("Approximate Cu-PLA Mass Remaining Over Time")
ax1.grid(True, alpha=0.3)

# second x-axis: temperature
ax2 = ax1.twiny()

# make the second x-axis match the same range as time
ax2.set_xlim(ax1.get_xlim())

# choose some time positions for temperature labels
time_ticks = np.linspace(0, end_time, 8)
temp_labels = np.interp(time_ticks, time, temp)

ax2.set_xticks(time_ticks)
ax2.set_xticklabels([f"{T:.0f}" for T in temp_labels])
ax2.set_xlabel("Oven temperature [°C]")

ax1.legend(loc="upper right")

# comparison panel: temperature on x-axis
ax3.plot(
    temp_heating,
    composite_mass_remaining_heating,
    color="tab:green",
    linewidth=2,
    label="Modelled Cu-PLA mass"
)
ax3.plot(
    temp_heating,
    f1_reference,
    color="black",
    linewidth=2,
    label="Reference F1 (approx.)"
)
ax3.set_xlabel("Temperature [°C]")
ax3.set_ylabel("Relative mass [%]")
ax3.set_title("Comparison with Reference F1")
ax3.set_xlim(0, 600)
ax3.set_ylim(84, 100.5)
ax3.grid(True, alpha=0.3)
ax3.legend(loc="lower left")

fig_pla_mass.tight_layout()

# -----------------------------
# Plot 3: Sintering temperature profile
# -----------------------------

start_temp_sinter = 15
sinter_temp = 1052

ramp_rate_sinter = 111.1
hold_time_sinter = 5
cooling_rate_sinter = 60

ramp_time_sinter = (sinter_temp - start_temp_sinter) / ramp_rate_sinter
t_hold_start_sinter = ramp_time_sinter
t_cooling_start_sinter = ramp_time_sinter + hold_time_sinter
cooling_time_sinter = (sinter_temp - start_temp_sinter) / cooling_rate_sinter
end_time_sinter = t_cooling_start_sinter + cooling_time_sinter

time_sinter = np.linspace(0, end_time_sinter, 3000)
temp_sinter = np.zeros_like(time_sinter)

for i, t in enumerate(time_sinter):
    if t <= ramp_time_sinter:
        temp_sinter[i] = start_temp_sinter + ramp_rate_sinter * t
    elif t <= t_cooling_start_sinter:
        temp_sinter[i] = sinter_temp
    else:
        temp_sinter[i] = sinter_temp - cooling_rate_sinter * (t - t_cooling_start_sinter)

fig_sintering_temperature = plt.figure(figsize=(9, 5))
plt.plot(time_sinter, temp_sinter, color="tab:orange", label="Oven temperature profile")

#plt.axvspan(t_hold_start_sinter,t_cooling_start_sinter,color="red",alpha=0.15,label="Sintering hold")

plt.xlabel("Time [h]")
plt.ylabel("Temperature [°C]")
plt.title("Temperature Profile for Copper Sintering")
plt.grid(True)
plt.legend()
plt.tight_layout()

print(f"Ramp time: {ramp_time_sinter:.2f} h")
print(f"Hold time: {hold_time_sinter:.2f} h")
print(f"Cooling time: {cooling_time_sinter:.2f} h")
print(f"Total process time: {end_time_sinter:.2f} h")

# Save every enabled plot whenever the script runs.
plots_to_save = [
    (
        SHOW_OVEN_TEMPERATURE_PLOT,
        fig_oven_temperature,
        "oven_temperature_profile.png",
    ),
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
    (
        SHOW_PLA_MASS_PLOT,
        fig_pla_mass,
        "pla_mass_model.png",
    ),
    (
        SHOW_SINTERING_TEMPERATURE_PLOT,
        fig_sintering_temperature,
        "sintering_temperature_profile.png",
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
if not SHOW_OVEN_TEMPERATURE_PLOT:
    plt.close(fig_oven_temperature)
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
if not SHOW_PLA_MASS_PLOT:
    plt.close(fig_pla_mass)
if not SHOW_SINTERING_TEMPERATURE_PLOT:
    plt.close(fig_sintering_temperature)

# Create all figures first, then display them together.
plt.show()

import numpy as np
import matplotlib.pyplot as plt

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

plt.figure(figsize=(9, 5))
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
plt.show()

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

fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(13, 5))

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

fig.tight_layout()
plt.show()

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

plt.figure(figsize=(9, 5))
plt.plot(time_sinter, temp_sinter, color="tab:orange", label="Oven temperature profile")

#plt.axvspan(t_hold_start_sinter,t_cooling_start_sinter,color="red",alpha=0.15,label="Sintering hold")

plt.xlabel("Time [h]")
plt.ylabel("Temperature [°C]")
plt.title("Temperature Profile for Copper Sintering")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Ramp time: {ramp_time_sinter:.2f} h")
print(f"Hold time: {hold_time_sinter:.2f} h")
print(f"Cooling time: {cooling_time_sinter:.2f} h")
print(f"Total process time: {end_time_sinter:.2f} h")

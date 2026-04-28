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
# Plot 1 (keep original)
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
mass_remaining = 100 - np.cumsum(mass_loss_rate) * dt
mass_remaining = np.clip(mass_remaining, 0, 100)

# -----------------------------
# Plot 2: mass remaining vs time, with temperature as second x-axis
# -----------------------------

fig, ax1 = plt.subplots(figsize=(9, 5))

# main x-axis: time
ax1.plot(
    time,
    mass_remaining,
    color="tab:green",
    linewidth=2,
    label="PLA mass remaining"
)

ax1.set_xlabel("Time [h]")
ax1.set_ylabel("PLA mass remaining [wt%]")
ax1.set_title("Approximate PLA Mass Remaining Over Time")
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

fig.tight_layout()
plt.show()
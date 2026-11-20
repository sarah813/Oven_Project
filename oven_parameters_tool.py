"""Interactive tool for multi-step debinding, sintering, and gas programs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from datetime import datetime, timedelta
from math import isfinite
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter


MAX_STEPS = 8
MAX_GAS_PHASES = 8
PROFILE_COLOR = "#333333"
COPPER_RANGE_COLOR = "#B87333"
COPPER_SINTERING_RANGE_C = (750.0, 1083.0)
COPPER_MELTING_TEMPERATURE_C = 1083.0
PYROLYSIS_RANGE_COLOR = "#59A14F"
PYROLYSIS_RANGE_C = (200.0, 500.0)
GAS_COLORS = {
    "Nitrogen": "#70B7E6",
    "CO2": "#633A7A",
    "Air": "#236B36",
    "Argon": "#613B20",
}
GAS_TYPES = tuple(GAS_COLORS)
RANGE_BASES = ("Time", "Temperature", "Whole phase")
STAGES = ("Debinding", "Sintering", "Cooling", "All")
CYCLE_MODES = ("Debinding only", "Sintering only", "Debinding + sintering")

# Measured cooling data used to shape/validate the predicted cooling segment.
# The three high-temperature readings below were taken after sintering and are
# included before the previously recorded 556 C-to-189 C measurements.  The
# time values are retained as elapsed minutes for traceability; the plotted
# points are positioned by inverting the fitted temperature curve.
SINTERING_COOLING_TIME_MIN = np.array(
    [0.0, 72.0, 88.0, 104.0, 360.0, 382.0, 397.0, 412.0, 427.0, 505.0, 638.0, 688.0, 868.0],
    dtype=float,
)
SINTERING_COOLING_TEMPERATURE_C = np.array(
    [1045.0, 855.0, 820.0, 788.0, 556.0, 533.0, 513.0, 497.0, 481.0, 405.0, 301.0, 270.0, 189.0],
    dtype=float,
)
DEBINDING_COOLING_TIME_MIN = np.array(
    [0.0, 215.0, 250.0, 251.0, 269.0, 278.0, 397.0, 449.0],
    dtype=float,
)
DEBINDING_COOLING_TEMPERATURE_C = np.array(
    [500.0, 258.0, 240.0, 239.0, 230.0, 226.0, 170.0, 79.0],
    dtype=float,
)
# Dimensionless decay constant obtained by fitting the measured cooling data
# with an exponential.  The curve is rescaled to the selected cooling duration.
# Fit to the post-debinding measurements above; the near-linear value reflects
# the observed 500 C to 79 C decrease over 449 minutes.
# Re-fitted after adding the 1045/855/820 C readings; the larger value captures
# the faster initial drop while retaining a smooth monotonic tail.
COOLING_SHAPE_K = 3.05


def smooth_cooling_temperature(
    elapsed_h, start_temperature_c, ambient_c, duration_h, shape_k=COOLING_SHAPE_K
):
    """Approximate cooling with a normalized exponential function.

    ``T(t) = T_ambient + (T_start - T_ambient)
    [exp(-k t/tau) - exp(-k)]/[1 - exp(-k)]``

    where ``tau`` is the total cooling duration and ``k`` is fitted to the
    supplied measurements.  This is smooth, monotonic, and reaches ambient
    temperature exactly at the end of the programmed cooling period.
    """
    tau = max(float(duration_h), 1e-12)
    u = np.clip(np.asarray(elapsed_h, dtype=float) / tau, 0.0, 1.0)
    end_factor = np.exp(-shape_k)
    shape = (np.exp(-shape_k * u) - end_factor) / (1.0 - end_factor)
    values = ambient_c + (start_temperature_c - ambient_c) * shape
    return float(values) if np.ndim(values) == 0 else values


@dataclass(frozen=True)
class ThermalStep:
    """One ramp followed by one optional temperature hold."""

    target_temperature_c: float
    ramp_rate_c_per_h: float
    hold_h: float


@dataclass(frozen=True)
class GasPhase:
    """A gas selection active over a time or temperature interval."""

    gas: str
    basis: str
    start: float
    end: float
    flow_rate_l_per_min: float = 1
    stage: str = "All"


@dataclass(frozen=True)
class TemperatureProfileParameters:
    """Complete user-configurable oven program."""

    start_temperature_c: float = 15.0
    debinding_steps: tuple[ThermalStep, ...] = (
        ThermalStep(482.0, 55.6, 4.0),
    )
    sintering_steps: tuple[ThermalStep, ...] = (
        ThermalStep(1052.0, 111.1, 5.0),
    )
    cooling_rate_c_per_h: float = 60.0
    cycle_mode: str = "Debinding + sintering"
    start_time: str = ""
    gas_phases: tuple[GasPhase, ...] = (
        GasPhase("Air", "Time", 0.0, 40.0, 500.0, "All"),
    )


def _require_finite(value: float, label: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{label} must be a finite number.")


def validate_parameters(parameters: TemperatureProfileParameters) -> None:
    """Raise ValueError when the selected program cannot be constructed."""

    _require_finite(parameters.start_temperature_c, "Starting temperature")
    _require_finite(parameters.cooling_rate_c_per_h, "Cooling rate")
    if parameters.cooling_rate_c_per_h <= 0:
        raise ValueError("Cooling rate must be greater than zero.")

    if parameters.cycle_mode not in CYCLE_MODES:
        raise ValueError("Choose a valid cycle type.")
    if parameters.start_time.strip():
        try:
            datetime.fromisoformat(parameters.start_time.strip())
        except ValueError as error:
            raise ValueError("Start time must use YYYY-MM-DD HH:MM format.") from error
    if parameters.cycle_mode in ("Debinding only", "Debinding + sintering") and not parameters.debinding_steps:
        raise ValueError("At least one debinding step is required.")
    if parameters.cycle_mode in ("Sintering only", "Debinding + sintering") and not parameters.sintering_steps:
        raise ValueError("At least one sintering step is required.")

    selected_stages = {
        "Debinding only": (("Debinding", parameters.debinding_steps),),
        "Sintering only": (("Sintering", parameters.sintering_steps),),
        "Debinding + sintering": (
            ("Debinding", parameters.debinding_steps),
            ("Sintering", parameters.sintering_steps),
        ),
    }[parameters.cycle_mode]
    for stage_name, steps in selected_stages:
        for index, step in enumerate(steps, start=1):
            prefix = f"{stage_name} step {index}"
            _require_finite(
                step.target_temperature_c,
                f"{prefix} target temperature",
            )
            _require_finite(step.ramp_rate_c_per_h, f"{prefix} ramp rate")
            _require_finite(step.hold_h, f"{prefix} hold time")
            if step.ramp_rate_c_per_h <= 0:
                raise ValueError(f"{prefix} ramp rate must be greater than zero.")
            if step.hold_h < 0:
                raise ValueError(f"{prefix} hold time cannot be negative.")

    if not parameters.gas_phases:
        raise ValueError("At least one atmosphere phase is required.")

    for index, phase in enumerate(parameters.gas_phases, start=1):
        prefix = f"Atmosphere phase {index}"
        if phase.gas not in GAS_TYPES:
            raise ValueError(f"{prefix} has an unsupported gas.")
        if phase.basis not in RANGE_BASES:
            raise ValueError(f"{prefix} must use a time, temperature, or whole-phase range.")
        if phase.stage not in STAGES:
            raise ValueError(f"{prefix} must use a valid process stage.")
        if phase.gas != "Air":
            _require_finite(phase.flow_rate_l_per_min, f"{prefix} flow rate")
        if phase.gas != "Air" and phase.flow_rate_l_per_min <= 0:
            raise ValueError(f"{prefix} flow rate must be greater than zero.")
        if phase.basis != "Whole phase":
            _require_finite(phase.start, f"{prefix} start")
            _require_finite(phase.end, f"{prefix} end")
            if phase.end <= phase.start:
                raise ValueError(f"{prefix} end must be greater than its start.")


def build_temperature_profile(parameters: TemperatureProfileParameters):
    """Return profile vertices and detailed timing for all selected steps."""

    validate_parameters(parameters)

    times_h = [0.0]
    temperatures_c = [parameters.start_temperature_c]
    step_timings = []
    current_time_h = 0.0
    current_temperature_c = parameters.start_temperature_c
    debinding_end_h = 0.0

    selected_stages = {
        "Debinding only": (("Debinding", parameters.debinding_steps),),
        "Sintering only": (("Sintering", parameters.sintering_steps),),
        "Debinding + sintering": (
            ("Debinding", parameters.debinding_steps),
            ("Sintering", parameters.sintering_steps),
        ),
    }[parameters.cycle_mode]
    for stage_name, steps in selected_stages:
        for index, step in enumerate(steps, start=1):
            ramp_start_h = current_time_h
            ramp_h = (
                abs(step.target_temperature_c - current_temperature_c)
                / step.ramp_rate_c_per_h
            )
            current_time_h += ramp_h
            times_h.append(current_time_h)
            temperatures_c.append(step.target_temperature_c)

            hold_start_h = current_time_h
            current_time_h += step.hold_h
            times_h.append(current_time_h)
            temperatures_c.append(step.target_temperature_c)

            step_timings.append(
                {
                    "stage": stage_name,
                    "index": index,
                    "ramp_start_h": ramp_start_h,
                    "ramp_end_h": hold_start_h,
                    "ramp_h": ramp_h,
                    "hold_start_h": hold_start_h,
                    "hold_end_h": current_time_h,
                    "hold_h": step.hold_h,
                    "target_temperature_c": step.target_temperature_c,
                    "ramp_rate_c_per_h": step.ramp_rate_c_per_h,
                }
            )
            current_temperature_c = step.target_temperature_c

        if stage_name == "Debinding":
            debinding_end_h = current_time_h

    sintering_end_h = current_time_h
    cooling_start_temperature_c = current_temperature_c
    if parameters.cycle_mode == "Debinding only":
        cooling_measurement_times_min = DEBINDING_COOLING_TIME_MIN
        cooling_measurement_temperatures_c = DEBINDING_COOLING_TEMPERATURE_C
        cooling_shape_k = 0.001
    else:
        cooling_measurement_times_min = SINTERING_COOLING_TIME_MIN
        cooling_measurement_temperatures_c = SINTERING_COOLING_TEMPERATURE_C
        cooling_shape_k = COOLING_SHAPE_K
    cooling_h = (
        abs(cooling_start_temperature_c - parameters.start_temperature_c)
        / parameters.cooling_rate_c_per_h
    )
    # Sample the fitted smooth cooling function densely enough for interval
    # detection while retaining a compact, predictable profile representation.
    cooling_samples_h = np.linspace(0.0, cooling_h, 121)[1:]
    cooling_samples_c = smooth_cooling_temperature(
        cooling_samples_h,
        cooling_start_temperature_c,
        parameters.start_temperature_c,
        cooling_h,
        cooling_shape_k,
    )
    for elapsed_h, predicted_temperature_c in zip(
        cooling_samples_h, cooling_samples_c
    ):
        times_h.append(sintering_end_h + elapsed_h)
        temperatures_c.append(float(predicted_temperature_c))

    total_time_h = sintering_end_h + cooling_h
    timing = {
        "steps": step_timings,
        "cycle_mode": parameters.cycle_mode,
        "debinding_end_h": debinding_end_h,
        "sintering_end_h": sintering_end_h,
        "cooling_h": cooling_h,
        "total_h": total_time_h,
    }
    return times_h, temperatures_c, timing


def temperature_range_intervals(times_h, temperatures_c, lower_c, upper_c):
    """Return profile-time intervals whose temperature lies in a range."""

    intervals = []
    for time_start, time_end, temp_start, temp_end in zip(
        times_h[:-1],
        times_h[1:],
        temperatures_c[:-1],
        temperatures_c[1:],
    ):
        if temp_start == temp_end:
            if lower_c <= temp_start <= upper_c:
                intervals.append((time_start, time_end))
            continue

        fractions = [0.0, 1.0]
        for boundary_c in (lower_c, upper_c):
            fraction = (boundary_c - temp_start) / (temp_end - temp_start)
            if 0.0 < fraction < 1.0:
                fractions.append(fraction)
        fractions.sort()
        for fraction_start, fraction_end in zip(fractions[:-1], fractions[1:]):
            midpoint = (fraction_start + fraction_end) / 2
            midpoint_temperature = temp_start + midpoint * (temp_end - temp_start)
            if lower_c <= midpoint_temperature <= upper_c:
                intervals.append(
                    (
                        time_start + fraction_start * (time_end - time_start),
                        time_start + fraction_end * (time_end - time_start),
                    )
                )

    merged = []
    for interval_start, interval_end in intervals:
        if merged and abs(merged[-1][1] - interval_start) < 1e-10:
            merged[-1] = (merged[-1][0], interval_end)
        else:
            merged.append((interval_start, interval_end))
    return merged


def stage_time_ranges(timing, stage):
    """Return absolute time bounds for an atmosphere phase's stage selection."""

    if stage == "Debinding":
        return [(0.0, timing["debinding_end_h"])]
    if stage == "Sintering":
        return [
            (timing["debinding_end_h"], timing["sintering_end_h"]),
        ]
    if stage == "Cooling":
        return [(timing["sintering_end_h"], timing["total_h"])]
    return [(0.0, timing["total_h"])]


def calculate_composite_mass_remaining(times_h, temperatures_c, timing):
    """Estimate composite mass and PLA-binder fraction remaining."""

    # Use a dense grid so the calculation remains stable when a hold has zero
    # duration and the profile contains repeated time vertices.
    model_times_h = np.linspace(0.0, timing["total_h"], 2400)
    model_temperatures_c = np.interp(
        model_times_h,
        times_h,
        temperatures_c,
    )

    def gaussian(temperature_c, center_c, sigma_c, area_wt):
        return area_wt / (sigma_c * np.sqrt(2 * np.pi)) * np.exp(
            -0.5 * ((temperature_c - center_c) / sigma_c) ** 2
        )

    dtg_total = (
        gaussian(model_temperatures_c, 362.5, 23.0, 90.04)
        + gaussian(model_temperatures_c, 462.9, 25.0, 8.327)
    )
    d_temperature_dt = np.gradient(model_temperatures_c, model_times_h)
    mass_loss_rate = dtg_total * np.maximum(d_temperature_dt, 0.0)
    dt_h = model_times_h[1] - model_times_h[0]
    pla_mass_remaining_wt = 100.0 - np.cumsum(mass_loss_rate) * dt_h
    pla_mass_remaining_wt = np.clip(pla_mass_remaining_wt, 0.0, 100.0)
    # Density 5.49 g/cm3 implies approximately 10.15 wt% PLA and 89.85 wt%
    # copper when using rho_Cu=8.96 and rho_PLA=1.24 g/cm3.
    pla_mass_fraction = 0.1015
    composite_mass_remaining_wt = 100.0 - (
        100.0 - pla_mass_remaining_wt
    ) * pla_mass_fraction
    if timing.get("cycle_mode") == "Sintering only":
        # A sintering-only run starts after debinding, so the PLA/binder loss
        # has already happened.  The remaining composite is therefore about
        # the copper fraction (approximately 90 wt%), not 100 wt% PLA.
        composite_mass_remaining_wt = np.full_like(model_temperatures_c, 90.0)
        pla_mass_remaining_wt = np.full_like(model_temperatures_c, 2.0)
    else:
        # A small residual binder fraction is retained: real debinding is not
        # perfectly complete, so the PLA curve should approach, but not equal,
        # zero.
        pla_mass_remaining_wt = np.maximum(pla_mass_remaining_wt, 2.0)

    return model_times_h, composite_mass_remaining_wt, pla_mass_remaining_wt


def create_temperature_figure(
    parameters: TemperatureProfileParameters,
    figure: Figure | None = None,
):
    """Create or redraw the profile, steps, and atmosphere regions."""

    times_h, temperatures_c, timing = build_temperature_profile(parameters)
    if parameters.cycle_mode == "Debinding only":
        measurement_temperatures_c = DEBINDING_COOLING_TEMPERATURE_C
        cooling_shape_k = 0.001
    else:
        measurement_temperatures_c = SINTERING_COOLING_TEMPERATURE_C
        cooling_shape_k = COOLING_SHAPE_K
    if figure is None:
        figure = Figure(
            figsize=(11.5, 6.6),
            dpi=100,
            constrained_layout=True,
            facecolor="#FFFDFE",
        )
    else:
        figure.clear()
        figure.set_constrained_layout(True)

    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(4.2, 1.15),
        hspace=0.025,
    )
    ax = figure.add_subplot(grid[0, 0])
    ax_mass = figure.add_subplot(grid[1, 0], sharex=ax)
    ax.set_facecolor("#FFFEFF")
    ax_mass.set_facecolor("#FFFEFF")

    profile_line = Line2D(
        [0],
        [0],
        color=PROFILE_COLOR,
        linewidth=1.8,
        label="Oven temperature profile (cooling approximation)",
    )
    mass_line = Line2D(
        [0],
        [0],
        color="#2E8B57",
        linewidth=1.5,
        label="Composite mass remaining (Cu + PLA) [wt%]",
    )
    pla_line = Line2D(
        [0],
        [0],
        color="#8E4585",
        linewidth=1.5,
        linestyle="--",
        label="PLA/binder remaining [% of initial PLA]",
    )
    gas_handles = []
    gas_shaded_intervals = []
    time_range_ends = [timing["total_h"]]
    temperature_range_values = [
        *temperatures_c,
        *COPPER_SINTERING_RANGE_C,
        *PYROLYSIS_RANGE_C,
    ]
    heating_end_index = max(
        index
        for index, time_h in enumerate(times_h)
        if time_h <= timing["sintering_end_h"] + 1e-9
    )

    # Draw light gas regions behind the temperature profile.
    for index, phase in enumerate(parameters.gas_phases, start=1):
        color = GAS_COLORS[phase.gas]
        gas_display = (
            f"{phase.gas} - ambient"
            if phase.gas == "Air"
            else f"{phase.gas} - {phase.flow_rate_l_per_min:g} L/min"
        )
        stage_ranges = stage_time_ranges(timing, phase.stage)
        if phase.basis == "Whole phase":
            # A whole-phase atmosphere uses the selected stage bounds
            # directly; the From/To fields are intentionally ignored.
            phase_intervals = stage_ranges
            for interval_start_h, interval_end_h in phase_intervals:
                ax.axvspan(
                    interval_start_h,
                    interval_end_h,
                    color=color,
                    alpha=0.22,
                    linewidth=0,
                    zorder=0,
                )
                gas_shaded_intervals.append(
                    (interval_start_h, interval_end_h, phase.gas, color, gas_display)
                )
            unit = ""
            time_range_ends.extend(end for _, end in phase_intervals)
        elif phase.basis == "Time":
            interval_start_h = phase.start
            interval_end_h = phase.end
            ax.axvspan(interval_start_h, interval_end_h, color=color, alpha=0.22, linewidth=0, zorder=0)
            gas_shaded_intervals.append((interval_start_h, interval_end_h, phase.gas, color, gas_display))
            unit = "h"
            time_range_ends.append(phase.end)
        else:
            # Temperature ranges are evaluated only within the selected
            # stage, so a 0-800 C phase does not reappear during cooling.
            profile_intervals = []
            stage_profile_intervals = (
                [(0.0, timing["sintering_end_h"])]
                if phase.stage == "All"
                else stage_ranges
            )
            all_profile_intervals = temperature_range_intervals(
                times_h,
                temperatures_c,
                phase.start,
                phase.end,
            )
            for profile_start, profile_end in all_profile_intervals:
                for stage_start, stage_end in stage_profile_intervals:
                    overlap_start = max(profile_start, stage_start)
                    overlap_end = min(profile_end, stage_end)
                    if overlap_end > overlap_start:
                        profile_intervals.append((overlap_start, overlap_end))
            for interval_start_h, interval_end_h in profile_intervals:
                ax.axvspan(
                    interval_start_h,
                    interval_end_h,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.22,
                    linewidth=0,
                    zorder=0,
                )
                gas_shaded_intervals.append((interval_start_h, interval_end_h, phase.gas, color, gas_display))
            unit = "degC"
            temperature_range_values.extend([phase.start, phase.end])

        gas_handles.append(
            Patch(
                facecolor=color,
                edgecolor=color,
                alpha=0.45,
                label=(
                    f"Gas {index}: {phase.gas}, "
                    + (
                        "whole phase, "
                        if phase.basis == "Whole phase"
                        else f"{phase.start:g}-{phase.end:g} {unit}, "
                    )
                    + (
                        f"{phase.flow_rate_l_per_min:g} L/min ({phase.stage})"
                        if phase.gas != "Air"
                        else f"ambient ({phase.stage})"
                    )
                ),
            )
        )

    # Put the gas schedule in a dedicated strip above the axes.  This keeps
    # atmosphere labels out of the temperature profile and its annotations.
    gas_label_height = 1.045
    seen_gas_intervals = set()
    labeled_gases = set()
    for interval_start_h, interval_end_h, gas_name, color, gas_display in gas_shaded_intervals:
        interval_key = (round(interval_start_h, 8), round(interval_end_h, 8), gas_name)
        if interval_key in seen_gas_intervals:
            continue
        seen_gas_intervals.add(interval_key)
        if gas_display in labeled_gases:
            continue
        labeled_gases.add(gas_display)
        ax.text(
            (interval_start_h + interval_end_h) / 2,
            gas_label_height,
            gas_display,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=color,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
            clip_on=False,
            zorder=6,
        )
    if gas_shaded_intervals:
        ax.text(
            -0.055,
            gas_label_height,
            "Gas:",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=PROFILE_COLOR,
            clip_on=False,
            zorder=7,
        )

    # Shade the time spent inside the copper sintering range with vertical,
    # lightly striped regions.
    copper_intervals = temperature_range_intervals(
        times_h,
        temperatures_c,
        *COPPER_SINTERING_RANGE_C,
    )
    for interval_start, interval_end in copper_intervals:
        ax.axvspan(
            interval_start,
            interval_end,
            facecolor="none",
            edgecolor=COPPER_RANGE_COLOR,
            linewidth=0,
            hatch="/",
            alpha=0.75,
            zorder=1,
        )
        for boundary_h in (interval_start, interval_end):
            ax.axvline(
                boundary_h,
                color=COPPER_RANGE_COLOR,
                linestyle="-",
                linewidth=1.0,
                alpha=0.9,
                zorder=3,
            )
    for reference_temperature_c in COPPER_SINTERING_RANGE_C:
        is_melting_line = (
            reference_temperature_c == COPPER_MELTING_TEMPERATURE_C
        )
        ax.axhline(
            reference_temperature_c,
            color=COPPER_RANGE_COLOR,
            linewidth=1.8 if is_melting_line else 1.4,
            linestyle="-" if is_melting_line else "--",
            alpha=0.90,
            zorder=2,
        )
        ax.text(
            0.99,
            reference_temperature_c,
            (
                f"{reference_temperature_c:g} C (Cu melting)"
                if is_melting_line
                else f"{reference_temperature_c:g} C"
            ),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=COPPER_RANGE_COLOR,
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.76,
            },
            zorder=7,
        )

    # Shade the time spent in the PLA pyrolysis range as a vertical area.
    # Pyrolysis is treated as a debinding/heating event only; do not shade the
    # same temperature range again during the final cooling cycle.
    pyrolysis_intervals = temperature_range_intervals(
        times_h,
        temperatures_c,
        *PYROLYSIS_RANGE_C,
    )[:1]
    for interval_start, interval_end in pyrolysis_intervals:
        ax.axvspan(
            interval_start,
            interval_end,
            facecolor="none",
            edgecolor=PYROLYSIS_RANGE_COLOR,
            linewidth=1.0,
            alpha=0.95,
            hatch="/",
            zorder=1,
        )
    if pyrolysis_intervals:
        interval_start, interval_end = pyrolysis_intervals[0]
        ax.text(
            (interval_start + interval_end) / 2,
            sum(PYROLYSIS_RANGE_C) / 2 + 140,
        "Pyrolysis\n200-500 C",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=PYROLYSIS_RANGE_COLOR,
            rotation=0 if interval_end - interval_start >= 2.0 else 90,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.76,
            },
            zorder=7,
        )

    ax.plot(
        times_h,
        temperatures_c,
        color=PROFILE_COLOR,
        linewidth=1.8,
        solid_capstyle="round",
        zorder=4,
    )
    mass_times_h, mass_remaining_wt, pla_remaining_wt = calculate_composite_mass_remaining(
        times_h,
        temperatures_c,
        timing,
    )
    ax_mass.plot(
        mass_times_h,
        mass_remaining_wt,
        color="#2E8B57",
        linewidth=2.0,
        linestyle="-",
        zorder=3,
    )
    ax_mass.plot(
        mass_times_h,
        pla_remaining_wt,
        color="#8E4585",
        linewidth=1.8,
        linestyle="--",
        zorder=3,
    )
    ax_mass.set_ylabel("Mass remaining [wt%]", color="#2E8B57")
    ax_mass.tick_params(axis="y", labelcolor="#2E8B57")
    ax_mass.set_ylim(0, 100.5)
    ax_mass.grid(True, alpha=0.20)

    # Keep ramp and plateau annotations compact and non-overlapping.
    for step in timing["steps"]:
        ramp_mid_x = (step["ramp_start_h"] + step["ramp_end_h"]) / 2
        ramp_mid_y = float(np.interp(ramp_mid_x, times_h, temperatures_c))
        # Keep the debinding ramp annotation below the curve while it passes
        # through the pyrolysis label area.
        ramp_offset = (
            (0, -16)
            if 200.0 <= ramp_mid_y <= 500.0 or ramp_mid_y >= 700.0
            else (0, 12)
        )
        ax.annotate(
            f"{step['ramp_rate_c_per_h']:g} C/h",
            (ramp_mid_x, ramp_mid_y),
            xytext=ramp_offset,
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=7.5,
            color=PROFILE_COLOR,
            zorder=7,
        )
        if step["hold_h"] > 0:
            plateau_x = (step["hold_start_h"] + step["hold_end_h"]) / 2
            ax.annotate(
                f"{step['target_temperature_c']:g} C",
                (plateau_x, step["target_temperature_c"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=PROFILE_COLOR,
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
                zorder=7,
            )

    maximum_time_h = max(time_range_ends)
    profile_max_temperature_c = max(temperatures_c)
    minimum_temperature_c = min(0, *temperature_range_values)
    maximum_temperature_c = max(temperature_range_values)
    temperature_span_c = max(
        100,
        maximum_temperature_c - minimum_temperature_c,
    )

    # Keep the elapsed-time scale directly under the temperature profile.
    # The mass panel shares the x limits, but should not own the time labels.
    ax_mass.set_xlabel("")
    ax_mass.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel("Temperature [C]")
    ax.set_title("Oven Parameters Tool")
    ax.set_xlim(0, maximum_time_h * 1.02)
    ax.set_ylim(
        minimum_temperature_c,
        maximum_temperature_c + temperature_span_c * 0.14,
    )
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Elapsed time [h]", labelpad=4, fontsize=8.5)
    if parameters.start_time.strip():
        start_dt = datetime.fromisoformat(parameters.start_time.strip())
        real_time_axis = ax.secondary_xaxis(
            -0.10,
            functions=(lambda value: value, lambda value: value),
        )
        real_time_axis.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _position: (
                    start_dt + timedelta(hours=float(value))
                ).strftime("%d.%m %H:%M")
            )
        )
        real_time_axis.set_xlabel("Real time", labelpad=1, fontsize=8.5)
        real_time_axis.tick_params(axis="x", labelsize=7.5, pad=1)
    copper_range_handle = Patch(
        facecolor="white",
        edgecolor=COPPER_RANGE_COLOR,
        hatch="/",
        alpha=1.0,
        label="Copper sintering range: 750-1083 C",
    )
    copper_melting_handle = Line2D(
        [0],
        [0],
        color=COPPER_RANGE_COLOR,
        linewidth=1.8,
        linestyle="-",
        label="Copper melting point: 1083 C",
    )
    pyrolysis_range_handle = Patch(
        facecolor="white",
        edgecolor=PYROLYSIS_RANGE_COLOR,
        hatch="/",
        alpha=1.0,
        label="PLA pyrolysis range: 200-500 C",
    )
    ax.legend(
        handles=[
            profile_line,
            mass_line,
            pla_line,
            copper_range_handle,
            copper_melting_handle,
            pyrolysis_range_handle,
            *gas_handles,
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        fontsize=8.5,
        title="Profile ranges and atmosphere",
        title_fontsize=9,
    )
    start_dt = None
    if parameters.start_time.strip():
        start_dt = datetime.fromisoformat(parameters.start_time.strip())
    finish_text = (
        (start_dt + timedelta(hours=timing["total_h"])).strftime("%Y-%m-%d %H:%M")
        if start_dt is not None
        else "not specified"
    )
    sintering_start_text = (
        (start_dt + timedelta(hours=timing["debinding_end_h"])).strftime("%Y-%m-%d %H:%M")
        if start_dt is not None
        else "not specified"
    )
    cooling_start_text = (
        (start_dt + timedelta(hours=timing["sintering_end_h"])).strftime("%Y-%m-%d %H:%M")
        if start_dt is not None
        else "not specified"
    )
    summary = (
        f"Total oven time: {timing['total_h']:.2f} h\n"
        f"Start: {parameters.start_time or 'not specified'}\n"
        f"Sintering start: {sintering_start_text}\n"
        f"Cooling start: {cooling_start_text}\n"
        f"Finish: {finish_text}\n"
        f"Maximum temperature: {profile_max_temperature_c:g} C\n"
        f"Debinding duration: {timing['debinding_end_h']:.2f} h\n"
        f"Sintering duration: {timing['sintering_end_h'] - timing['debinding_end_h']:.2f} h\n"
        f"Cooling duration: {timing['cooling_h']:.2f} h\n"
        f"Gas phases: {len(parameters.gas_phases)}"
    )
    # Keep the summary in the right-hand margin directly below the legend.
    # The legend grows with the number of gas phases, so move the summary down
    # slightly for larger legends while keeping it in the same column.
    summary_block = "Program summary\n\n" + summary
    summary_y = max(0.36, 0.66 - 0.025 * max(0, len(gas_handles) - 1))
    figure.text(
        0.75,
        summary_y,
        summary_block,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#333333",
        linespacing=1.35,
        zorder=8,
        clip_on=False,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#FFF9FC",
            "edgecolor": "#D9B8CD",
            "linewidth": 0.9,
            "alpha": 0.96,
        },
    )
    return figure, timing


class TemperatureProfileTool:
    """Tkinter interface for building and exporting custom oven programs."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Oven Parameters Tool")
        self.root.geometry("1550x850")
        self.root.minsize(1120, 700)

        # Soft pastel styling keeps the dense parameter panel approachable.
        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#FFF9FC")
        style.configure("TLabelframe", background="#FFF9FC", bordercolor="#E8C9DC")
        style.configure(
            "TLabelframe.Label",
            background="#FFF9FC",
            foreground="#6B3A5C",
            font=("Segoe UI", 9, "bold"),
        )
        style.configure("TLabel", background="#FFF9FC", foreground="#3F3040")
        style.configure(
            "TButton",
            background="#F3D5E8",
            foreground="#4E2942",
            padding=(8, 4),
            borderwidth=0,
        )
        style.map("TButton", background=[("active", "#E9BFD9")])
        style.configure("TCombobox", fieldbackground="#FFFFFF", background="#F8EAF3")
        root.configure(background="#FFF9FC")

        self.start_temperature_var = tk.StringVar(value="15")
        self.cooling_rate_var = tk.StringVar(value="60")
        self.cycle_mode_var = tk.StringVar(value="Debinding + sintering")
        self.start_time_var = tk.StringVar(value="")
        self.debinding_count_var = tk.StringVar(value="1")
        self.sintering_count_var = tk.StringVar(value="1")
        self.gas_count_var = tk.StringVar(value="1")
        self.debinding_step_vars = []
        self.sintering_step_vars = []
        self.gas_phase_vars = []
        self.preset_name_var = tk.StringVar()
        self.preset_choice_var = tk.StringVar()
        self.preset_names = []
        self.preset_path = Path(__file__).with_name("oven_presets.json")
        self.status = tk.StringVar(
            value="Choose step counts, atmosphere phases, and generate a profile."
        )

        controls_container = ttk.Frame(root)
        controls_container.pack(side=tk.LEFT, fill=tk.Y)
        self.controls_canvas = tk.Canvas(
            controls_container,
            width=450,
            highlightthickness=0,
            background="#FFF9FC",
        )
        controls_scrollbar = ttk.Scrollbar(
            controls_container,
            orient=tk.VERTICAL,
            command=self.controls_canvas.yview,
        )
        self.controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        controls_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.controls = ttk.Frame(self.controls_canvas, padding=14)
        self.controls_window = self.controls_canvas.create_window(
            (0, 0),
            window=self.controls,
            anchor="nw",
        )
        self.controls.bind(
            "<Configure>",
            lambda _event: self.controls_canvas.configure(
                scrollregion=self.controls_canvas.bbox("all")
            ),
        )
        self.controls_canvas.bind(
            "<Configure>",
            lambda event: self.controls_canvas.itemconfigure(
                self.controls_window,
                width=event.width,
            ),
        )

        plot_panel = ttk.Frame(root, padding=(0, 10, 10, 10))
        plot_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_controls()

        self.figure = Figure(
            figsize=(11.5, 6.6),
            dpi=100,
            constrained_layout=True,
            facecolor="#FFFDFE",
        )
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_panel)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_panel, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.generate_profile(show_error=False)

    def _build_controls(self) -> None:
        ttk.Label(
            self.controls,
            text="Oven Parameters Tool",
            font=("Segoe UI", 15, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        self._build_preset_section()

        general_frame = ttk.LabelFrame(
            self.controls,
            text="General",
            padding=10,
        )
        general_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(general_frame, text="Cycle type").grid(
            row=0, column=0, sticky="w", padx=3, pady=3
        )
        self.cycle_mode_box = ttk.Combobox(
            general_frame,
            textvariable=self.cycle_mode_var,
            values=CYCLE_MODES,
            state="readonly",
            width=24,
        )
        self.cycle_mode_box.grid(row=0, column=1, columnspan=2, sticky="ew", padx=3, pady=3)
        self.cycle_mode_box.bind("<<ComboboxSelected>>", self.update_stage_visibility)
        self._add_labeled_entry(
            general_frame,
            1,
            "Starting temperature",
            self.start_temperature_var,
            "C",
        )
        self._add_labeled_entry(
            general_frame,
            2,
            "Start time (optional)",
            self.start_time_var,
            "",
        )

        self.debinding_frame = self._create_stage_section(
            "Debinding",
            self.debinding_count_var,
            self.rebuild_debinding_rows,
        )
        self.debinding_rows = ttk.Frame(self.debinding_frame)
        self.debinding_rows.pack(fill=tk.X, pady=(8, 0))
        self.rebuild_debinding_rows()

        self.sintering_frame = self._create_stage_section(
            "Sintering",
            self.sintering_count_var,
            self.rebuild_sintering_rows,
        )
        self.sintering_rows = ttk.Frame(self.sintering_frame)
        self.sintering_rows.pack(fill=tk.X, pady=(8, 0))
        self.rebuild_sintering_rows()
        self.update_stage_visibility()

        self.atmosphere_frame = ttk.LabelFrame(
            self.controls,
            text="Atmosphere",
            padding=10,
        )
        self.atmosphere_frame.pack(fill=tk.X, pady=(0, 10))
        atmosphere_count_row = ttk.Frame(self.atmosphere_frame)
        atmosphere_count_row.pack(fill=tk.X)
        ttk.Label(atmosphere_count_row, text="Number of gas phases").pack(
            side=tk.LEFT
        )
        gas_count_box = ttk.Combobox(
            atmosphere_count_row,
            textvariable=self.gas_count_var,
            values=tuple(str(value) for value in range(1, MAX_GAS_PHASES + 1)),
            state="readonly",
            width=5,
        )
        gas_count_box.pack(side=tk.RIGHT)
        gas_count_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.rebuild_gas_rows(),
        )

        self.gas_rows = ttk.Frame(self.atmosphere_frame)
        self.gas_rows.pack(fill=tk.X, pady=(8, 0))
        self.rebuild_gas_rows()

        ttk.Button(
            self.controls,
            text="Generate profile",
            command=self.generate_profile,
        ).pack(fill=tk.X, pady=(8, 5))
        ttk.Button(
            self.controls,
            text="Save PNG...",
            command=self.save_png,
        ).pack(fill=tk.X, pady=5)
        ttk.Label(
            self.controls,
            textvariable=self.status,
            wraplength=450,
            foreground="#444444",
        ).pack(anchor="w", pady=(10, 4))

    def update_stage_visibility(self, _event=None):
        """Show only the step editor(s) used by the selected cycle type."""
        mode = self.cycle_mode_var.get()
        if mode == "Debinding only":
            self.debinding_frame.pack(fill=tk.X, pady=(0, 10), before=self.atmosphere_frame if hasattr(self, "atmosphere_frame") else None)
            self.sintering_frame.pack_forget()
        elif mode == "Sintering only":
            self.debinding_frame.pack_forget()
            self.sintering_frame.pack(fill=tk.X, pady=(0, 10), before=self.atmosphere_frame if hasattr(self, "atmosphere_frame") else None)
        else:
            self.debinding_frame.pack(fill=tk.X, pady=(0, 10), before=self.atmosphere_frame if hasattr(self, "atmosphere_frame") else None)
            self.sintering_frame.pack(fill=tk.X, pady=(0, 10), before=self.atmosphere_frame if hasattr(self, "atmosphere_frame") else None)
        if hasattr(self, "gas_rows"):
            self.rebuild_gas_rows()
        self._refresh_scroll_region()

    def allowed_atmosphere_stages(self):
        if self.cycle_mode_var.get() == "Debinding only":
            return ("Debinding", "Cooling")
        if self.cycle_mode_var.get() == "Sintering only":
            return ("Sintering", "Cooling")
        return STAGES

    def _create_stage_section(self, title, count_variable, callback):
        frame = ttk.LabelFrame(self.controls, text=title, padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))
        count_row = ttk.Frame(frame)
        count_row.pack(fill=tk.X)
        ttk.Label(count_row, text="Number of steps").pack(side=tk.LEFT)
        count_box = ttk.Combobox(
            count_row,
            textvariable=count_variable,
            values=tuple(str(value) for value in range(1, MAX_STEPS + 1)),
            state="readonly",
            width=5,
        )
        count_box.pack(side=tk.RIGHT)
        count_box.bind("<<ComboboxSelected>>", lambda _event: callback())
        return frame

    def _add_labeled_entry(self, parent, row, label, variable, unit):
        ttk.Label(parent, text=label).grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=3,
        )
        entry = ttk.Entry(parent, textvariable=variable, width=12)
        entry.grid(row=row, column=1, sticky="ew", pady=3)
        entry.bind("<Return>", lambda _event: self.generate_profile())
        ttk.Label(parent, text=unit).grid(
            row=row,
            column=2,
            sticky="w",
            padx=(6, 0),
            pady=3,
        )
        parent.columnconfigure(1, weight=1)

    def _ensure_step_variables(self, variables, count, stage):
        while len(variables) < count:
            index = len(variables)
            if stage == "Debinding":
                target = 482 + index * 75
                ramp = 55.6
                hold = 4
            else:
                target = 1052 + index * 15
                ramp = 111.1
                hold = 5
            variables.append(
                {
                    "target": tk.StringVar(value=f"{target:g}"),
                    "ramp": tk.StringVar(value=f"{ramp:g}"),
                    "hold": tk.StringVar(value=f"{hold:g}"),
                }
            )

    def _rebuild_step_rows(self, container, variables, count, stage):
        self._ensure_step_variables(variables, count, stage)
        for child in container.winfo_children():
            child.destroy()

        headers = ("Step", "Target [C]", "Ramp [C/h]", "Hold [h]")
        for column, header in enumerate(headers):
            ttk.Label(
                container,
                text=header,
                font=("Segoe UI", 9, "bold"),
            ).grid(row=0, column=column, sticky="w", padx=3)

        for index in range(count):
            ttk.Label(container, text=str(index + 1)).grid(
                row=index + 1,
                column=0,
                sticky="w",
                padx=3,
                pady=3,
            )
            for column, key in enumerate(("target", "ramp", "hold"), start=1):
                entry = ttk.Entry(
                    container,
                    textvariable=variables[index][key],
                    width=11,
                )
                entry.grid(
                    row=index + 1,
                    column=column,
                    sticky="ew",
                    padx=3,
                    pady=3,
                )
                entry.bind("<Return>", lambda _event: self.generate_profile())

        for column in range(1, 4):
            container.columnconfigure(column, weight=1)
        self._refresh_scroll_region()

    def _build_preset_section(self) -> None:
        """Create controls for naming, saving, loading, and deleting presets."""

        preset_frame = ttk.LabelFrame(
            self.controls,
            text="Presets",
            padding=10,
        )
        preset_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(preset_frame, text="Preset name").grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=3,
        )
        preset_name_entry = ttk.Entry(
            preset_frame,
            textvariable=self.preset_name_var,
            width=20,
        )
        preset_name_entry.grid(
            row=0,
            column=1,
            columnspan=2,
            sticky="ew",
            pady=3,
        )
        preset_name_entry.bind("<Return>", lambda _event: self.save_preset())

        ttk.Button(
            preset_frame,
            text="Save preset",
            command=self.save_preset,
        ).grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(6, 3))
        ttk.Button(
            preset_frame,
            text="Delete preset",
            command=self.delete_preset,
        ).grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(6, 3))
        ttk.Button(
            preset_frame,
            text="Change preset",
            command=self.update_preset,
        ).grid(row=1, column=2, sticky="ew", padx=(4, 0), pady=(6, 3))

        ttk.Label(preset_frame, text="Saved presets").grid(
            row=2,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=3,
        )
        self.preset_box = ttk.Combobox(
            preset_frame,
            textvariable=self.preset_choice_var,
            state="readonly",
            width=20,
        )
        self.preset_box.grid(
            row=2,
            column=1,
            sticky="ew",
            pady=3,
        )
        ttk.Button(
            preset_frame,
            text="Load",
            command=self.load_preset,
        ).grid(row=2, column=2, sticky="ew", padx=(4, 0), pady=3)
        for column in range(3):
            preset_frame.columnconfigure(column, weight=1)
        self.refresh_preset_names()

    @staticmethod
    def _parameters_to_dict(parameters: TemperatureProfileParameters) -> dict:
        return {
            "start_temperature_c": parameters.start_temperature_c,
            "cooling_rate_c_per_h": parameters.cooling_rate_c_per_h,
            "cycle_mode": parameters.cycle_mode,
            "start_time": parameters.start_time,
            "debinding_steps": [
                {
                    "target_temperature_c": step.target_temperature_c,
                    "ramp_rate_c_per_h": step.ramp_rate_c_per_h,
                    "hold_h": step.hold_h,
                }
                for step in parameters.debinding_steps
            ],
            "sintering_steps": [
                {
                    "target_temperature_c": step.target_temperature_c,
                    "ramp_rate_c_per_h": step.ramp_rate_c_per_h,
                    "hold_h": step.hold_h,
                }
                for step in parameters.sintering_steps
            ],
            "gas_phases": [
                {
                    "gas": phase.gas,
                    "basis": phase.basis,
                    "start": phase.start,
                    "end": phase.end,
                    "flow_rate_l_per_min": phase.flow_rate_l_per_min,
                    "stage": phase.stage,
                }
                for phase in parameters.gas_phases
            ],
        }

    @staticmethod
    def _parameters_from_dict(data: dict) -> TemperatureProfileParameters:
        def read_steps(key):
            return tuple(
                ThermalStep(
                    target_temperature_c=float(item["target_temperature_c"]),
                    ramp_rate_c_per_h=float(item["ramp_rate_c_per_h"]),
                    hold_h=float(item["hold_h"]),
                )
                for item in data[key]
            )

        return TemperatureProfileParameters(
            start_temperature_c=float(data["start_temperature_c"]),
            cooling_rate_c_per_h=float(data["cooling_rate_c_per_h"]),
            cycle_mode=data.get("cycle_mode", "Debinding + sintering"),
            start_time=data.get("start_time", ""),
            debinding_steps=read_steps("debinding_steps"),
            sintering_steps=read_steps("sintering_steps"),
            gas_phases=tuple(
                GasPhase(
                    gas=item["gas"],
                    basis=item["basis"],
                    start=float(item["start"]),
                    end=float(item["end"]),
                    flow_rate_l_per_min=float(
                        item.get("flow_rate_l_per_min", 1.0)
                    ),
                    stage=("All" if item.get("stage", "All") == "Both" else item.get("stage", "All")),
                )
                for item in data["gas_phases"]
            ),
        )

    def _read_preset_file(self) -> dict:
        if not self.preset_path.exists():
            return {}
        with self.preset_path.open("r", encoding="utf-8") as preset_file:
            data = json.load(preset_file)
        if not isinstance(data, dict):
            raise ValueError("The preset file must contain a JSON object.")
        return data

    def _write_preset_file(self, presets: dict) -> None:
        with self.preset_path.open("w", encoding="utf-8") as preset_file:
            json.dump(presets, preset_file, indent=2, ensure_ascii=False)
            preset_file.write("\n")

    def refresh_preset_names(self) -> None:
        try:
            presets = self._read_preset_file()
            self.preset_names = sorted(presets)
            self.preset_box.configure(values=self.preset_names)
            if self.preset_choice_var.get() not in self.preset_names:
                self.preset_choice_var.set("")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.preset_names = []
            self.preset_box.configure(values=())
            self.preset_choice_var.set("")
            self.status.set(f"Could not read presets: {error}")

    def save_preset(self) -> None:
        name = self.preset_name_var.get().strip()
        if not name:
            messagebox.showerror("Preset name required", "Enter a name for this preset.")
            return

        parameters = self.generate_profile(show_error=True)
        if parameters is None:
            return

        try:
            presets = self._read_preset_file()
            if name in presets and not messagebox.askyesno(
                "Overwrite preset?",
                f'A preset named "{name}" already exists. Overwrite it?',
            ):
                return
            presets[name] = self._parameters_to_dict(parameters)
            self._write_preset_file(presets)
            self.preset_name_var.set(name)
            self.preset_choice_var.set(name)
            self.refresh_preset_names()
            self.status.set(f'Saved preset "{name}".')
        except (OSError, ValueError, json.JSONDecodeError) as error:
            messagebox.showerror("Could not save preset", str(error))

    def delete_preset(self) -> None:
        name = self.preset_choice_var.get().strip()
        if not name:
            messagebox.showerror("Preset not selected", "Select a saved preset first.")
            return
        if not messagebox.askyesno(
            "Delete preset?",
            f'Delete the preset "{name}"?',
        ):
            return

        try:
            presets = self._read_preset_file()
            presets.pop(name, None)
            self._write_preset_file(presets)
            self.preset_choice_var.set("")
            self.preset_name_var.set("")
            self.refresh_preset_names()
            self.status.set(f'Deleted preset "{name}".')
        except (OSError, ValueError, json.JSONDecodeError) as error:
            messagebox.showerror("Could not delete preset", str(error))

    def update_preset(self) -> None:
        """Replace the selected preset with the currently visible settings."""

        name = self.preset_choice_var.get().strip()
        if not name:
            messagebox.showerror("Preset not selected", "Select a saved preset first.")
            return

        parameters = self.generate_profile(show_error=True)
        if parameters is None:
            return

        try:
            presets = self._read_preset_file()
            if name not in presets:
                raise ValueError(f'Preset "{name}" was not found.')
            presets[name] = self._parameters_to_dict(parameters)
            self._write_preset_file(presets)
            self.preset_name_var.set(name)
            self.status.set(f'Updated preset "{name}".')
        except (OSError, ValueError, json.JSONDecodeError) as error:
            messagebox.showerror("Could not update preset", str(error))

    def _apply_parameters(self, parameters: TemperatureProfileParameters) -> None:
        self.start_temperature_var.set(f"{parameters.start_temperature_c:g}")
        self.cooling_rate_var.set(f"{parameters.cooling_rate_c_per_h:g}")
        self.cycle_mode_var.set(parameters.cycle_mode)
        self.start_time_var.set(parameters.start_time)
        self.update_stage_visibility()

        self.debinding_count_var.set(str(len(parameters.debinding_steps)))
        self.rebuild_debinding_rows()
        for variables, step in zip(
            self.debinding_step_vars,
            parameters.debinding_steps,
        ):
            variables["target"].set(f"{step.target_temperature_c:g}")
            variables["ramp"].set(f"{step.ramp_rate_c_per_h:g}")
            variables["hold"].set(f"{step.hold_h:g}")

        self.sintering_count_var.set(str(len(parameters.sintering_steps)))
        self.rebuild_sintering_rows()
        for variables, step in zip(
            self.sintering_step_vars,
            parameters.sintering_steps,
        ):
            variables["target"].set(f"{step.target_temperature_c:g}")
            variables["ramp"].set(f"{step.ramp_rate_c_per_h:g}")
            variables["hold"].set(f"{step.hold_h:g}")

        self.gas_count_var.set(str(len(parameters.gas_phases)))
        self.rebuild_gas_rows()
        for variables, phase in zip(self.gas_phase_vars, parameters.gas_phases):
            variables["gas"].set(phase.gas)
            variables["stage"].set(phase.stage)
            variables["basis"].set(phase.basis)
            variables["start"].set(f"{phase.start:g}")
            variables["end"].set(f"{phase.end:g}")
            variables["flow"].set(f"{phase.flow_rate_l_per_min:g}")

        self.rebuild_gas_rows()
        self.generate_profile(show_error=False)

    def load_preset(self) -> None:
        name = self.preset_choice_var.get().strip()
        if not name:
            messagebox.showerror("Preset not selected", "Select a saved preset first.")
            return
        try:
            presets = self._read_preset_file()
            if name not in presets:
                raise ValueError(f'Preset "{name}" was not found.')
            parameters = self._parameters_from_dict(presets[name])
            validate_parameters(parameters)
            self._apply_parameters(parameters)
            self.preset_name_var.set(name)
            self.status.set(f'Loaded preset "{name}".')
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            messagebox.showerror("Could not load preset", str(error))

    def rebuild_debinding_rows(self) -> None:
        count = int(self.debinding_count_var.get())
        self._rebuild_step_rows(
            self.debinding_rows,
            self.debinding_step_vars,
            count,
            "Debinding",
        )

    def rebuild_sintering_rows(self) -> None:
        count = int(self.sintering_count_var.get())
        self._rebuild_step_rows(
            self.sintering_rows,
            self.sintering_step_vars,
            count,
            "Sintering",
        )

    def _ensure_gas_variables(self, count):
        default_gases = ("Air", "Nitrogen", "CO2")
        while len(self.gas_phase_vars) < count:
            index = len(self.gas_phase_vars)
            self.gas_phase_vars.append(
                {
                    "gas": tk.StringVar(
                        value=default_gases[index % len(default_gases)]
                    ),
                    "stage": tk.StringVar(value="All"),
                    "basis": tk.StringVar(value="Time"),
                    "start": tk.StringVar(value=f"{index * 10:g}"),
                    "end": tk.StringVar(value=f"{(index + 1) * 10:g}"),
                    "flow": tk.StringVar(value="500"),
                }
            )

    def rebuild_gas_rows(self) -> None:
        count = int(self.gas_count_var.get())
        self._ensure_gas_variables(count)
        for child in self.gas_rows.winfo_children():
            child.destroy()

        headers = ("Phase", "Gas", "Stage", "Defined by", "From", "To")
        for column, header in enumerate(headers):
            ttk.Label(
                self.gas_rows,
                text=header,
                font=("Segoe UI", 9, "bold"),
            ).grid(row=0, column=column, sticky="w", padx=3)

        for index in range(count):
            variables = self.gas_phase_vars[index]
            row = index * 2 + 1
            ttk.Label(self.gas_rows, text=str(index + 1)).grid(
                row=row,
                column=0,
                sticky="w",
                padx=3,
                pady=3,
            )

            gas_cell = ttk.Frame(self.gas_rows)
            gas_cell.grid(
                row=row,
                column=1,
                sticky="ew",
                padx=3,
                pady=3,
            )
            swatch = tk.Label(
                gas_cell,
                width=2,
                background=GAS_COLORS[variables["gas"].get()],
            )
            swatch.pack(side=tk.LEFT, padx=(0, 3))
            gas_box = ttk.Combobox(
                gas_cell,
                textvariable=variables["gas"],
                values=GAS_TYPES,
                state="readonly",
                width=9,
            )
            gas_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
            gas_box.bind(
                "<<ComboboxSelected>>",
                lambda _event, gas_var=variables["gas"], label=swatch: (
                    label.configure(background=GAS_COLORS[gas_var.get()])
                ),
            )

            stage_box = ttk.Combobox(
                self.gas_rows,
                textvariable=variables["stage"],
                values=self.allowed_atmosphere_stages(),
                state="readonly",
                width=10,
            )
            allowed_stages = self.allowed_atmosphere_stages()
            if variables["stage"].get() not in allowed_stages:
                variables["stage"].set(allowed_stages[0])
            stage_box.grid(
                row=row,
                column=2,
                sticky="ew",
                padx=3,
                pady=3,
            )

            variables["stage_box"] = stage_box

            basis_box = ttk.Combobox(
                self.gas_rows,
                textvariable=variables["basis"],
                values=RANGE_BASES,
                state="readonly",
                width=13,
            )
            basis_box.grid(
                row=row,
                column=3,
                sticky="ew",
                padx=3,
                pady=3,
            )
            variables["basis_box"] = basis_box

            start_cell = ttk.Frame(self.gas_rows)
            start_cell.grid(
                row=row,
                column=4,
                sticky="ew",
                padx=3,
                pady=3,
            )
            start_entry = ttk.Entry(
                start_cell,
                textvariable=variables["start"],
                width=7,
            )
            start_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            variables["start_entry"] = start_entry
            start_unit = ttk.Label(start_cell, width=3)
            start_unit.pack(side=tk.LEFT, padx=(3, 0))

            end_cell = ttk.Frame(self.gas_rows)
            end_cell.grid(
                row=row,
                column=5,
                sticky="ew",
                padx=3,
                pady=3,
            )
            end_entry = ttk.Entry(
                end_cell,
                textvariable=variables["end"],
                width=7,
            )
            end_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            variables["end_entry"] = end_entry
            end_unit = ttk.Label(end_cell, width=3)
            end_unit.pack(side=tk.LEFT, padx=(3, 0))

            flow_cell = ttk.Frame(self.gas_rows)
            flow_cell.grid(
                row=row + 1,
                column=1,
                columnspan=5,
                sticky="ew",
                padx=3,
                pady=(0, 4),
            )
            ttk.Label(flow_cell, text="Flow rate [L/min]").pack(
                side=tk.LEFT,
                padx=(0, 6),
            )
            flow_entry = ttk.Entry(
                flow_cell,
                textvariable=variables["flow"],
                width=10,
            )
            flow_entry.pack(side=tk.LEFT)

            def update_flow_state(
                _event=None,
                gas_var=variables["gas"],
                entry=flow_entry,
            ):
                if gas_var.get() == "Air":
                    entry.delete(0, tk.END)
                    entry.insert(0, "")
                    entry.configure(state="disabled")
                else:
                    entry.configure(state="normal")

            gas_box.bind("<<ComboboxSelected>>", update_flow_state, add="+")

            def update_units(
                _event=None,
                basis_var=variables["basis"],
                labels=(start_unit, end_unit),
            ):
                unit = (
                    "h"
                    if basis_var.get() == "Time"
                    else "degC"
                    if basis_var.get() == "Temperature"
                    else ""
                )
                for label in labels:
                    label.configure(text=unit)

            def update_stage_controls(
                _event=None,
                stage_var=variables["stage"],
                basis_var=variables["basis"],
                basis_widget=basis_box,
                start_widget=start_entry,
                end_widget=end_entry,
            ):
                # The selected stage limits where the range is applied.  A
                # whole-phase atmosphere does not need From/To values.
                basis_widget.configure(state="readonly")
                entry_state = (
                    "disabled" if basis_var.get() == "Whole phase" else "normal"
                )
                start_widget.configure(state=entry_state)
                end_widget.configure(state=entry_state)

            basis_box.bind("<<ComboboxSelected>>", update_units)
            basis_box.bind("<<ComboboxSelected>>", update_stage_controls, add="+")
            stage_box.bind("<<ComboboxSelected>>", update_stage_controls)
            update_units()
            update_stage_controls()
            update_flow_state()
            start_entry.bind("<Return>", lambda _event: self.generate_profile())
            end_entry.bind("<Return>", lambda _event: self.generate_profile())
            flow_entry.bind("<Return>", lambda _event: self.generate_profile())

        for column in range(1, 6):
            self.gas_rows.columnconfigure(column, weight=1)
        self._refresh_scroll_region()

    def _refresh_scroll_region(self):
        self.root.after_idle(
            lambda: self.controls_canvas.configure(
                scrollregion=self.controls_canvas.bbox("all")
            )
        )

    @staticmethod
    def _read_float(variable, label):
        raw_value = variable.get().strip().replace(",", ".")
        try:
            return float(raw_value)
        except ValueError as error:
            raise ValueError(f"{label} must be a number.") from error

    def _read_steps(self, variables, count, stage):
        steps = []
        for index in range(count):
            item = variables[index]
            prefix = f"{stage} step {index + 1}"
            steps.append(
                ThermalStep(
                    target_temperature_c=self._read_float(
                        item["target"],
                        f"{prefix} target temperature",
                    ),
                    ramp_rate_c_per_h=self._read_float(
                        item["ramp"],
                        f"{prefix} ramp rate",
                    ),
                    hold_h=self._read_float(
                        item["hold"],
                        f"{prefix} hold time",
                    ),
                )
            )
        return tuple(steps)

    def read_parameters(self) -> TemperatureProfileParameters:
        """Read and validate every currently visible control."""

        debinding_count = int(self.debinding_count_var.get())
        sintering_count = int(self.sintering_count_var.get())
        gas_count = int(self.gas_count_var.get())

        gas_phases = []
        for index in range(gas_count):
            item = self.gas_phase_vars[index]
            gas_name = item["gas"].get()
            gas_phases.append(
                GasPhase(
                    gas=gas_name,
                    stage=item["stage"].get(),
                    basis=item["basis"].get(),
                    start=self._read_float(
                        item["start"],
                        f"Atmosphere phase {index + 1} start",
                    ),
                    end=self._read_float(
                        item["end"],
                        f"Atmosphere phase {index + 1} end",
                    ),
                    flow_rate_l_per_min=(
                        0.0
                        if gas_name == "Air"
                        else self._read_float(
                            item["flow"],
                            f"Atmosphere phase {index + 1} flow rate",
                        )
                    ),
                )
            )

        parameters = TemperatureProfileParameters(
            start_temperature_c=self._read_float(
                self.start_temperature_var,
                "Starting temperature",
            ),
            cycle_mode=self.cycle_mode_var.get(),
            start_time=self.start_time_var.get().strip(),
            debinding_steps=self._read_steps(
                self.debinding_step_vars,
                debinding_count,
                "Debinding",
            ),
            sintering_steps=self._read_steps(
                self.sintering_step_vars,
                sintering_count,
                "Sintering",
            ),
            cooling_rate_c_per_h=60.0,
            gas_phases=tuple(gas_phases),
        )
        validate_parameters(parameters)
        return parameters

    def generate_profile(self, show_error: bool = True):
        """Redraw the chart using the currently visible controls."""

        try:
            parameters = self.read_parameters()
            _, timing = create_temperature_figure(parameters, self.figure)
            self.canvas.draw_idle()
            self.status.set(
                f"Profile generated. Total process time: "
                f"{timing['total_h']:.2f} h"
            )
            return parameters
        except ValueError as error:
            self.status.set(str(error))
            if show_error:
                messagebox.showerror("Invalid profile parameters", str(error))
            return None

    def save_png(self) -> None:
        """Validate the program and save the current chart as a PNG."""

        parameters = self.generate_profile()
        if parameters is None:
            return

        default_output_dir = Path.cwd() / "oven_outputs"
        initial_directory = (
            default_output_dir if default_output_dir.exists() else Path.cwd()
        )
        output_path = filedialog.asksaveasfilename(
            title="Save temperature profile",
            initialdir=initial_directory,
            initialfile="custom_debinding_sintering_profile.png",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not output_path:
            return

        self.figure.savefig(output_path, dpi=300, bbox_inches="tight")
        self.status.set(f"Saved profile to {output_path}")


def main() -> None:
    root = tk.Tk()
    TemperatureProfileTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()

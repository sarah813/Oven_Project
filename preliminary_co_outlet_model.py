"""Preliminary CO-outlet model for thermal processing of Cu/PLA filament.

This is an engineering test model, not a validated prediction.  Parameters in
the ASSUMPTIONS section are deliberately easy to find and replace.  The oven
temperature program and CO2 flow schedule are measured/program inputs loaded
from a preset created by ``oven_parameters_tool.py``.

The Boudouard reaction used here is the chemically balanced reaction

    CO2 + C -> 2 CO

so the gas-flow values in the oven preset are interpreted as CO2 feed rates.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from oven_parameters_tool import (
    GasPhase,
    TemperatureProfileParameters,
    ThermalStep,
    build_temperature_profile,
)


# =============================================================================
# USER-SELECTED PROGRAM INPUTS
# These describe the furnace program. They are not kinetic/material properties.
# =============================================================================

PRESET_FILE = Path(__file__).with_name("oven_presets.json")
PRESET_NAME = "test_n2"
TIME_STEP_MIN = 1.0
OUTPUT_DIR = Path(__file__).with_name("oven_outputs")


# =============================================================================
# PRELIMINARY MODEL ASSUMPTIONS / PLACEHOLDERS
# Replace these when filament-specific measurements are available.
# =============================================================================

# Placeholder initial PLA mass (not total Cu/PLA specimen mass).
PLA_INITIAL_G = 1.0

# Example only: fraction of decomposed PLA retained as reactive carbonaceous
# residue. This is not a measured char yield for the filament.
CHAR_YIELD = 0.05

# Assumed alpha(T): a smooth transition from zero conversion at the lower
# bound to complete conversion at the upper bound. This is NOT a TGA fit.
ALPHA_START_TEMPERATURE_C = 250.0
ALPHA_END_TEMPERATURE_C = 480.0

# Placeholder first-order Arrhenius kinetics for carbon consumption:
#     dm_C/dt = k(T) * m_C,available
#     k(T) = A * exp(-Ea / (R*T))
# Neither A nor Ea has been calibrated for this furnace/material system.
REACTION_PREEXPONENTIAL_PER_MIN = 2.0e5
REACTION_ACTIVATION_ENERGY_J_PER_MOL = 100_000.0
IDEAL_GAS_CONSTANT_J_PER_MOL_K = 8.314462618

# The residue is provisionally treated as pure reactive carbon. Replace with
# measured residue composition/reactive fraction when known.
CARBON_MOLAR_MASS_G_PER_MOL = 12.011

# Assumed CO2 flow-meter reference conditions used to convert L/min to mol/min.
# These are easy to change to the actual meter's standard/reference conditions.
FLOW_REFERENCE_TEMPERATURE_C = 25.0
FLOW_REFERENCE_PRESSURE_ATM = 1.0
IDEAL_GAS_CONSTANT_L_ATM_PER_MOL_K = 0.082057366


@dataclass(frozen=True)
class ModelAssumptions:
    """Adjustable, explicitly unvalidated inputs for the preliminary model."""

    pla_initial_g: float = PLA_INITIAL_G
    char_yield: float = CHAR_YIELD
    alpha_start_temperature_c: float = ALPHA_START_TEMPERATURE_C
    alpha_end_temperature_c: float = ALPHA_END_TEMPERATURE_C
    reaction_preexponential_per_min: float = REACTION_PREEXPONENTIAL_PER_MIN
    reaction_activation_energy_j_per_mol: float = (
        REACTION_ACTIVATION_ENERGY_J_PER_MOL
    )
    flow_reference_temperature_c: float = FLOW_REFERENCE_TEMPERATURE_C
    flow_reference_pressure_atm: float = FLOW_REFERENCE_PRESSURE_ATM


@dataclass(frozen=True)
class SimulationResults:
    """Time-resolved model results; rates are per minute."""

    time_min: np.ndarray
    temperature_c: np.ndarray
    alpha: np.ndarray
    pla_remaining_g: np.ndarray
    pla_decomposed_step_g: np.ndarray
    pla_decomposition_rate_g_per_min: np.ndarray
    carbon_generated_step_g: np.ndarray
    carbon_generated_cumulative_g: np.ndarray
    carbon_available_g: np.ndarray
    carbon_consumed_step_g: np.ndarray
    carbon_consumption_rate_g_per_min: np.ndarray
    carbon_consumed_cumulative_g: np.ndarray
    co2_flow_l_per_min: np.ndarray
    co2_inlet_mol_per_min: np.ndarray
    co_production_mol_per_min: np.ndarray
    predicted_co_ppm: np.ndarray


def load_oven_preset(
    preset_path: Path,
    preset_name: str,
) -> TemperatureProfileParameters:
    """Load one JSON preset saved by the oven parameter tool."""

    with preset_path.open("r", encoding="utf-8") as preset_file:
        presets = json.load(preset_file)

    if not isinstance(presets, dict):
        raise ValueError("The preset JSON root must be an object.")
    if preset_name not in presets:
        available = ", ".join(sorted(presets)) or "none"
        raise KeyError(
            f'Preset "{preset_name}" was not found. Available presets: {available}'
        )

    data = presets[preset_name]

    def read_steps(key: str) -> tuple[ThermalStep, ...]:
        return tuple(
            ThermalStep(
                target_temperature_c=float(item["target_temperature_c"]),
                ramp_rate_c_per_h=float(item["ramp_rate_c_per_h"]),
                hold_h=float(item["hold_h"]),
            )
            for item in data[key]
        )

    parameters = TemperatureProfileParameters(
        start_temperature_c=float(data["start_temperature_c"]),
        cooling_rate_c_per_h=float(data["cooling_rate_c_per_h"]),
        debinding_steps=read_steps("debinding_steps"),
        sintering_steps=read_steps("sintering_steps"),
        gas_phases=tuple(
            GasPhase(
                gas=str(item["gas"]),
                basis=str(item["basis"]),
                start=float(item["start"]),
                end=float(item["end"]),
                flow_rate_l_per_min=float(item.get("flow_rate_l_per_min", 1.0)),
                stage=str(item.get("stage", "Both")),
            )
            for item in data["gas_phases"]
        ),
    )

    # The shared builder performs the oven tool's normal validation.
    build_temperature_profile(parameters)
    return parameters


def make_temperature_profile(
    parameters: TemperatureProfileParameters,
    time_step_min: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Interpolate the JSON ramp/hold program onto a near-uniform time grid."""

    if time_step_min <= 0.0:
        raise ValueError("TIME_STEP_MIN must be greater than zero.")

    vertex_times_h, vertex_temperatures_c, timing = build_temperature_profile(
        parameters
    )
    total_min = timing["total_h"] * 60.0
    time_min = np.arange(0.0, total_min, time_step_min)
    if time_min.size == 0 or time_min[-1] < total_min:
        time_min = np.append(time_min, total_min)

    temperature_c = np.interp(
        time_min / 60.0,
        np.asarray(vertex_times_h),
        np.asarray(vertex_temperatures_c),
    )
    return time_min, temperature_c, timing


def assumed_alpha_from_temperature(
    temperature_c: np.ndarray,
    assumptions: ModelAssumptions = ModelAssumptions(),
) -> np.ndarray:
    """Return an unvalidated, irreversible approximation of PLA conversion.

    A cubic smooth-step avoids discontinuous decomposition at the two assumed
    bounds. ``maximum.accumulate`` represents irreversible decomposition, so
    alpha does not decrease when the furnace cools.
    """

    if (
        assumptions.alpha_end_temperature_c
        <= assumptions.alpha_start_temperature_c
    ):
        raise ValueError(
            "ALPHA_END_TEMPERATURE_C must exceed ALPHA_START_TEMPERATURE_C."
        )

    scaled_temperature = np.clip(
        (np.asarray(temperature_c) - assumptions.alpha_start_temperature_c)
        / (
            assumptions.alpha_end_temperature_c
            - assumptions.alpha_start_temperature_c
        ),
        0.0,
        1.0,
    )
    equilibrium_alpha = (
        3.0 * scaled_temperature**2 - 2.0 * scaled_temperature**3
    )
    return np.maximum.accumulate(equilibrium_alpha)


def _time_is_in_stage(time_h: float, stage: str, timing: dict) -> bool:
    """Apply the same stage interpretation used by the oven parameter tool."""

    tolerance = 1e-9
    if stage == "Debinding":
        return -tolerance <= time_h <= timing["debinding_end_h"] + tolerance
    if stage == "Sintering":
        return (
            timing["debinding_end_h"] - tolerance
            <= time_h
            <= timing["sintering_end_h"] + tolerance
        )
    return -tolerance <= time_h <= timing["total_h"] + tolerance


def _is_co2_label(gas: str) -> bool:
    """Recognize CO2, CO₂, and the legacy mojibake spelling in old presets."""

    compact = gas.strip().casefold().replace(" ", "")
    return compact.startswith("co") and compact != "co"


def co2_flow_profile(
    parameters: TemperatureProfileParameters,
    time_min: np.ndarray,
    temperature_c: np.ndarray,
    timing: dict,
) -> np.ndarray:
    """Build CO2 flow [L/min] from active gas phases in the JSON preset."""

    flow_l_per_min = np.zeros_like(time_min, dtype=float)
    for phase in parameters.gas_phases:
        if not _is_co2_label(phase.gas):
            continue

        for index, (current_time_min, current_temperature_c) in enumerate(
            zip(time_min, temperature_c)
        ):
            time_h = current_time_min / 60.0
            if not _time_is_in_stage(time_h, phase.stage, timing):
                continue

            if phase.basis == "Temperature":
                active = phase.start <= current_temperature_c <= phase.end
            elif phase.stage == "Both":
                active = phase.start <= time_h <= phase.end
            else:
                # Time ranges for an individual stage are relative to its start.
                stage_start_h = (
                    0.0
                    if phase.stage == "Debinding"
                    else timing["debinding_end_h"]
                )
                relative_time_h = time_h - stage_start_h
                active = phase.start <= relative_time_h <= phase.end

            if active:
                flow_l_per_min[index] += phase.flow_rate_l_per_min

    return flow_l_per_min


def volumetric_flow_to_molar_flow(
    flow_l_per_min: np.ndarray,
    assumptions: ModelAssumptions = ModelAssumptions(),
) -> np.ndarray:
    """Convert reference volumetric flow [L/min] to molar flow [mol/min]."""

    reference_temperature_k = assumptions.flow_reference_temperature_c + 273.15
    if reference_temperature_k <= 0.0 or assumptions.flow_reference_pressure_atm <= 0.0:
        raise ValueError("Gas reference temperature and pressure must be positive.")

    molar_volume_l_per_mol = (
        IDEAL_GAS_CONSTANT_L_ATM_PER_MOL_K
        * reference_temperature_k
        / assumptions.flow_reference_pressure_atm
    )
    return np.asarray(flow_l_per_min) / molar_volume_l_per_mol


def placeholder_carbon_rate_constant(
    temperature_c: float,
    assumptions: ModelAssumptions = ModelAssumptions(),
) -> float:
    """Return the assumed first-order carbon rate constant [1/min]."""

    temperature_k = temperature_c + 273.15
    if temperature_k <= 0.0:
        raise ValueError("Reaction temperature must be above absolute zero.")
    return assumptions.reaction_preexponential_per_min * np.exp(
        -assumptions.reaction_activation_energy_j_per_mol
        / (IDEAL_GAS_CONSTANT_J_PER_MOL_K * temperature_k)
    )


def simulate(
    parameters: TemperatureProfileParameters,
    time_step_min: float = TIME_STEP_MIN,
    assumptions: ModelAssumptions = ModelAssumptions(),
) -> SimulationResults:
    """Run the preliminary PLA/char/Boudouard mass balance."""

    if assumptions.pla_initial_g <= 0.0:
        raise ValueError("PLA_INITIAL_G must be greater than zero.")
    if not 0.0 <= assumptions.char_yield <= 1.0:
        raise ValueError("CHAR_YIELD must lie between zero and one.")
    if assumptions.reaction_preexponential_per_min < 0.0:
        raise ValueError("The reaction pre-exponential factor cannot be negative.")
    if assumptions.reaction_activation_energy_j_per_mol < 0.0:
        raise ValueError("The reaction activation energy cannot be negative.")

    time_min, temperature_c, timing = make_temperature_profile(
        parameters,
        time_step_min,
    )
    alpha = assumed_alpha_from_temperature(temperature_c, assumptions)
    alpha_change = np.diff(alpha, prepend=alpha[0])
    alpha_change = np.maximum(alpha_change, 0.0)

    pla_decomposed_step_g = assumptions.pla_initial_g * alpha_change
    pla_remaining_g = assumptions.pla_initial_g * (1.0 - alpha)
    carbon_generated_step_g = pla_decomposed_step_g * assumptions.char_yield
    carbon_generated_cumulative_g = np.cumsum(carbon_generated_step_g)

    co2_flow_l_per_min = co2_flow_profile(
        parameters,
        time_min,
        temperature_c,
        timing,
    )
    co2_inlet_mol_per_min = volumetric_flow_to_molar_flow(
        co2_flow_l_per_min,
        assumptions,
    )

    sample_count = len(time_min)
    pla_decomposition_rate_g_per_min = np.zeros(sample_count)
    carbon_available_g = np.zeros(sample_count)
    carbon_consumed_step_g = np.zeros(sample_count)
    carbon_consumption_rate_g_per_min = np.zeros(sample_count)
    carbon_consumed_cumulative_g = np.zeros(sample_count)
    co_production_mol_per_min = np.zeros(sample_count)
    predicted_co_ppm = np.zeros(sample_count)

    carbon_inventory_g = 0.0
    cumulative_carbon_consumed_g = 0.0

    for index in range(sample_count):
        if index == 0:
            # The first row is the initial state and has no preceding interval.
            carbon_inventory_g += carbon_generated_step_g[index]
            carbon_available_g[index] = carbon_inventory_g
            continue

        dt_min = time_min[index] - time_min[index - 1]
        if dt_min <= 0.0:
            raise ValueError("The generated time grid must increase monotonically.")

        pla_decomposition_rate_g_per_min[index] = (
            pla_decomposed_step_g[index] / dt_min
        )
        carbon_inventory_g += carbon_generated_step_g[index]

        # Unconstrained kinetic demand from the placeholder first-order model.
        rate_constant_per_min = placeholder_carbon_rate_constant(
            temperature_c[index],
            assumptions,
        )
        kinetic_carbon_demand_g = (
            carbon_inventory_g
            * (1.0 - np.exp(-rate_constant_per_min * dt_min))
        )

        # CO2 + C -> 2 CO: one mole of feed CO2 can consume at most one mole C.
        co2_supplied_step_mol = co2_inlet_mol_per_min[index] * dt_min
        co2_limited_carbon_g = co2_supplied_step_mol * CARBON_MOLAR_MASS_G_PER_MOL

        carbon_consumed_g = min(
            carbon_inventory_g,
            kinetic_carbon_demand_g,
            co2_limited_carbon_g,
        )
        carbon_inventory_g -= carbon_consumed_g
        # Suppress harmless negative round-off in the inventory.
        carbon_inventory_g = max(carbon_inventory_g, 0.0)
        cumulative_carbon_consumed_g += carbon_consumed_g

        carbon_consumed_step_g[index] = carbon_consumed_g
        carbon_consumption_rate_g_per_min[index] = carbon_consumed_g / dt_min
        carbon_consumed_cumulative_g[index] = cumulative_carbon_consumed_g
        carbon_available_g[index] = carbon_inventory_g

        carbon_consumed_mol = carbon_consumed_g / CARBON_MOLAR_MASS_G_PER_MOL
        co_production_mol_per_min[index] = 2.0 * carbon_consumed_mol / dt_min

        # Total outlet molar flow accounts for the net +1 mole created per mole
        # of reaction: CO2 + C -> 2 CO. No carrier-gas flow is included because
        # the current JSON preset does not specify one concurrently.
        outlet_total_mol_per_min = (
            co2_inlet_mol_per_min[index] + carbon_consumed_mol / dt_min
        )
        if outlet_total_mol_per_min > 0.0:
            predicted_co_ppm[index] = (
                co_production_mol_per_min[index]
                / outlet_total_mol_per_min
                * 1.0e6
            )

    return SimulationResults(
        time_min=time_min,
        temperature_c=temperature_c,
        alpha=alpha,
        pla_remaining_g=pla_remaining_g,
        pla_decomposed_step_g=pla_decomposed_step_g,
        pla_decomposition_rate_g_per_min=pla_decomposition_rate_g_per_min,
        carbon_generated_step_g=carbon_generated_step_g,
        carbon_generated_cumulative_g=carbon_generated_cumulative_g,
        carbon_available_g=carbon_available_g,
        carbon_consumed_step_g=carbon_consumed_step_g,
        carbon_consumption_rate_g_per_min=carbon_consumption_rate_g_per_min,
        carbon_consumed_cumulative_g=carbon_consumed_cumulative_g,
        co2_flow_l_per_min=co2_flow_l_per_min,
        co2_inlet_mol_per_min=co2_inlet_mol_per_min,
        co_production_mol_per_min=co_production_mol_per_min,
        predicted_co_ppm=predicted_co_ppm,
    )


def write_results_csv(results: SimulationResults, output_path: Path) -> None:
    """Write all requested per-step quantities with units in column names."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    field_arrays = {
        "time_min": results.time_min,
        "temperature_c": results.temperature_c,
        "alpha_fraction": results.alpha,
        "pla_remaining_g": results.pla_remaining_g,
        "pla_decomposed_step_g": results.pla_decomposed_step_g,
        "pla_decomposition_rate_g_per_min": (
            results.pla_decomposition_rate_g_per_min
        ),
        "carbon_generated_step_g": results.carbon_generated_step_g,
        "carbon_generated_cumulative_g": results.carbon_generated_cumulative_g,
        "carbon_available_g": results.carbon_available_g,
        "carbon_consumed_step_g": results.carbon_consumed_step_g,
        "carbon_consumption_rate_g_per_min": (
            results.carbon_consumption_rate_g_per_min
        ),
        "carbon_consumed_cumulative_g": results.carbon_consumed_cumulative_g,
        "co2_flow_l_per_min": results.co2_flow_l_per_min,
        "co2_inlet_mol_per_min": results.co2_inlet_mol_per_min,
        "co_production_mol_per_min": results.co_production_mol_per_min,
        "predicted_co_ppm": results.predicted_co_ppm,
    }

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(field_arrays)
        writer.writerows(zip(*field_arrays.values()))


def draw_temperature_and_alpha(
    figure: Figure,
    results: SimulationResults,
) -> None:
    """Draw temperature and assumed conversion in a supplied figure."""

    figure.clear()
    time_h = results.time_min / 60.0
    temperature_axis = figure.add_subplot(111)
    alpha_axis = temperature_axis.twinx()
    temperature_line = temperature_axis.plot(
        time_h,
        results.temperature_c,
        color="#3B6EA8",
        linewidth=2.0,
        label="Furnace temperature",
    )[0]
    alpha_line = alpha_axis.plot(
        time_h,
        results.alpha,
        color="#D97706",
        linewidth=2.0,
        label="Assumed PLA conversion alpha",
    )[0]
    temperature_axis.set(
        xlabel="Time [h]",
        ylabel="Temperature [degC]",
        title="Furnace Temperature and Preliminary PLA Conversion",
    )
    alpha_axis.set_ylabel("PLA decomposition conversion, alpha [-]")
    alpha_axis.set_ylim(-0.02, 1.02)
    temperature_axis.grid(alpha=0.25)
    temperature_axis.legend(
        [temperature_line, alpha_line],
        [temperature_line.get_label(), alpha_line.get_label()],
        loc="best",
    )
    figure.tight_layout()


def draw_carbon_balance(figure: Figure, results: SimulationResults) -> None:
    """Draw cumulative generation/consumption and current carbon inventory."""

    figure.clear()
    time_h = results.time_min / 60.0
    carbon_axis = figure.add_subplot(111)
    carbon_axis.plot(
        time_h,
        results.carbon_generated_cumulative_g,
        label="C generated (cumulative)",
        linewidth=2.0,
    )
    carbon_axis.plot(
        time_h,
        results.carbon_available_g,
        label="C available (inventory)",
        linewidth=2.0,
    )
    carbon_axis.plot(
        time_h,
        results.carbon_consumed_cumulative_g,
        label="C consumed (cumulative)",
        linewidth=2.0,
    )
    carbon_axis.set(
        xlabel="Time [h]",
        ylabel="Carbon mass [g]",
        title="Preliminary Carbon Mass Balance",
    )
    carbon_axis.grid(alpha=0.25)
    carbon_axis.legend(loc="best")
    figure.tight_layout()


def draw_co_curve(figure: Figure, results: SimulationResults) -> None:
    """Draw the predicted outlet CO concentration curve."""

    figure.clear()
    time_h = results.time_min / 60.0
    co_axis = figure.add_subplot(111)
    co_axis.plot(
        time_h,
        results.predicted_co_ppm,
        color="#A33A3A",
        linewidth=2.0,
    )
    co_axis.set(
        xlabel="Time [h]",
        ylabel="Predicted outlet CO [ppm]",
        title="Preliminary Predicted CO Outlet Concentration",
    )
    co_axis.grid(alpha=0.25)
    figure.tight_layout()


def create_plots(
    results: SimulationResults,
    output_dir: Path,
    preset_name: str,
) -> list[Path]:
    """Create and save the three requested figures as separate PNG files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in preset_name
    )

    figure_1 = plt.figure(figsize=(10, 5.5))
    draw_temperature_and_alpha(figure_1, results)
    plot_1_path = output_dir / f"{safe_name}_01_temperature_and_alpha.png"
    figure_1.savefig(plot_1_path, dpi=200, bbox_inches="tight")

    figure_2 = plt.figure(figsize=(10, 5.5))
    draw_carbon_balance(figure_2, results)
    plot_2_path = output_dir / f"{safe_name}_02_carbon_balance.png"
    figure_2.savefig(plot_2_path, dpi=200, bbox_inches="tight")

    figure_3 = plt.figure(figsize=(10, 5.5))
    draw_co_curve(figure_3, results)
    plot_3_path = output_dir / f"{safe_name}_03_predicted_co_ppm.png"
    figure_3.savefig(plot_3_path, dpi=200, bbox_inches="tight")

    return [plot_1_path, plot_2_path, plot_3_path]


def print_assumption_summary(
    results: SimulationResults,
    preset_name: str,
    output_paths: list[Path],
    assumptions: ModelAssumptions = ModelAssumptions(),
    time_step_min: float = TIME_STEP_MIN,
) -> None:
    """Print a compact warning and run summary for traceability."""

    print("PRELIMINARY MODEL - NOT A VALIDATED PREDICTION")
    print(f'Oven preset: "{preset_name}"')
    print(f"Time step: {time_step_min:g} min")
    print(f"Assumed initial PLA mass: {assumptions.pla_initial_g:g} g")
    print(f"Assumed char yield: {assumptions.char_yield:.1%}")
    print(
        "Unvalidated alpha(T) range: "
        f"{assumptions.alpha_start_temperature_c:g} to "
        f"{assumptions.alpha_end_temperature_c:g} degC"
    )
    print(
        "Placeholder carbon kinetics: "
        f"A={assumptions.reaction_preexponential_per_min:.3g} 1/min, "
        "Ea="
        f"{assumptions.reaction_activation_energy_j_per_mol / 1000:g} kJ/mol"
    )
    print(
        "CO2 flow reference: "
        f"{assumptions.flow_reference_temperature_c:g} degC, "
        f"{assumptions.flow_reference_pressure_atm:g} atm"
    )
    print(f"Maximum predicted CO: {np.max(results.predicted_co_ppm):.3g} ppm")
    print("Outputs:")
    for path in output_paths:
        print(f"  {path}")


class PreliminaryCOModelTool:
    """Tkinter interface for running and inspecting the preliminary CO model."""

    def __init__(
        self,
        root: tk.Tk,
        preset_path: Path = PRESET_FILE,
        initial_preset: str = PRESET_NAME,
        output_dir: Path = OUTPUT_DIR,
    ) -> None:
        self.root = root
        self.root.title("Preliminary CO Outlet Model")
        self.root.geometry("1500x850")
        self.root.minsize(1080, 680)

        self.output_dir = output_dir
        self.current_results: SimulationResults | None = None
        self.current_assumptions: ModelAssumptions | None = None
        self.preset_path_var = tk.StringVar(value=str(preset_path))
        self.preset_choice_var = tk.StringVar(value=initial_preset)
        self.time_step_var = tk.StringVar(value=f"{TIME_STEP_MIN:g}")
        self.pla_initial_var = tk.StringVar(value=f"{PLA_INITIAL_G:g}")
        self.char_yield_percent_var = tk.StringVar(value=f"{CHAR_YIELD * 100:g}")
        self.alpha_start_var = tk.StringVar(
            value=f"{ALPHA_START_TEMPERATURE_C:g}"
        )
        self.alpha_end_var = tk.StringVar(value=f"{ALPHA_END_TEMPERATURE_C:g}")
        self.preexponential_var = tk.StringVar(
            value=f"{REACTION_PREEXPONENTIAL_PER_MIN:g}"
        )
        self.activation_energy_var = tk.StringVar(
            value=f"{REACTION_ACTIVATION_ENERGY_J_PER_MOL / 1000:g}"
        )
        self.flow_temperature_var = tk.StringVar(
            value=f"{FLOW_REFERENCE_TEMPERATURE_C:g}"
        )
        self.flow_pressure_var = tk.StringVar(
            value=f"{FLOW_REFERENCE_PRESSURE_ATM:g}"
        )
        self.status = tk.StringVar(
            value="Select an oven preset, then calculate the CO curve."
        )
        self.summary = tk.StringVar(value="No simulation calculated yet.")

        controls_container = ttk.Frame(root)
        controls_container.pack(side=tk.LEFT, fill=tk.Y)
        self.controls_canvas = tk.Canvas(
            controls_container,
            width=440,
            highlightthickness=0,
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
        self.notebook = ttk.Notebook(plot_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.figures: dict[str, Figure] = {}
        self.plot_canvases: dict[str, FigureCanvasTkAgg] = {}
        for key, tab_label in (
            ("co", "CO output"),
            ("temperature", "Temperature and alpha"),
            ("carbon", "Carbon balance"),
        ):
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=tab_label)
            figure = Figure(figsize=(10, 6), dpi=100)
            canvas = FigureCanvasTkAgg(figure, master=tab)
            toolbar = NavigationToolbar2Tk(canvas, tab, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.figures[key] = figure
            self.plot_canvases[key] = canvas

        self._build_controls()
        self.refresh_presets(show_error=False)
        self.root.after_idle(lambda: self.calculate(show_error=False))

    @staticmethod
    def _add_labeled_entry(
        parent: ttk.LabelFrame,
        row: int,
        label: str,
        variable: tk.StringVar,
        unit: str,
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=3,
        )
        ttk.Entry(parent, textvariable=variable, width=14).grid(
            row=row,
            column=1,
            sticky="ew",
            pady=3,
        )
        ttk.Label(parent, text=unit).grid(
            row=row,
            column=2,
            sticky="w",
            padx=(6, 0),
            pady=3,
        )
        parent.columnconfigure(1, weight=1)

    def _build_controls(self) -> None:
        ttk.Label(
            self.controls,
            text="Preliminary CO Outlet Model",
            font=("Segoe UI", 15, "bold"),
        ).pack(anchor="w", pady=(0, 6))
        ttk.Label(
            self.controls,
            text=(
                "Engineering test only - alpha(T), char yield, and reaction "
                "kinetics are unvalidated assumptions. Reaction: CO2 + C -> 2 CO."
            ),
            foreground="#9A3412",
            wraplength=400,
            justify=tk.LEFT,
        ).pack(anchor="w", fill=tk.X, pady=(0, 12))

        preset_frame = ttk.LabelFrame(
            self.controls,
            text="Oven profile from JSON",
            padding=10,
        )
        preset_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(preset_frame, text="Preset file").grid(
            row=0, column=0, sticky="w", pady=3
        )
        ttk.Entry(
            preset_frame,
            textvariable=self.preset_path_var,
            state="readonly",
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Button(
            preset_frame,
            text="Browse...",
            command=self.browse_preset_file,
        ).grid(row=1, column=2, sticky="ew", padx=(5, 0), pady=(0, 4))
        ttk.Label(preset_frame, text="Saved profile").grid(
            row=2, column=0, sticky="w", pady=3
        )
        self.preset_box = ttk.Combobox(
            preset_frame,
            textvariable=self.preset_choice_var,
            state="readonly",
        )
        self.preset_box.grid(row=3, column=0, columnspan=2, sticky="ew", pady=3)
        self.preset_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.calculate(show_error=False),
        )
        ttk.Button(
            preset_frame,
            text="Refresh",
            command=self.refresh_presets,
        ).grid(row=3, column=2, sticky="ew", padx=(5, 0), pady=3)
        preset_frame.columnconfigure(0, weight=1)
        preset_frame.columnconfigure(1, weight=1)

        material_frame = ttk.LabelFrame(
            self.controls,
            text="Material assumptions (unvalidated)",
            padding=10,
        )
        material_frame.pack(fill=tk.X, pady=(0, 10))
        self._add_labeled_entry(
            material_frame, 0, "Initial PLA mass", self.pla_initial_var, "g"
        )
        self._add_labeled_entry(
            material_frame,
            1,
            "Char yield",
            self.char_yield_percent_var,
            "%",
        )
        self._add_labeled_entry(
            material_frame,
            2,
            "Alpha transition start",
            self.alpha_start_var,
            "degC",
        )
        self._add_labeled_entry(
            material_frame,
            3,
            "Alpha transition end",
            self.alpha_end_var,
            "degC",
        )

        kinetics_frame = ttk.LabelFrame(
            self.controls,
            text="Kinetic placeholders (unvalidated)",
            padding=10,
        )
        kinetics_frame.pack(fill=tk.X, pady=(0, 10))
        self._add_labeled_entry(
            kinetics_frame,
            0,
            "Pre-exponential A",
            self.preexponential_var,
            "1/min",
        )
        self._add_labeled_entry(
            kinetics_frame,
            1,
            "Activation energy Ea",
            self.activation_energy_var,
            "kJ/mol",
        )

        gas_frame = ttk.LabelFrame(
            self.controls,
            text="Flow conversion assumptions",
            padding=10,
        )
        gas_frame.pack(fill=tk.X, pady=(0, 10))
        self._add_labeled_entry(
            gas_frame,
            0,
            "Reference temperature",
            self.flow_temperature_var,
            "degC",
        )
        self._add_labeled_entry(
            gas_frame,
            1,
            "Reference pressure",
            self.flow_pressure_var,
            "atm",
        )
        self._add_labeled_entry(
            gas_frame,
            2,
            "Simulation time step",
            self.time_step_var,
            "min",
        )

        ttk.Button(
            self.controls,
            text="Calculate / update CO curve",
            command=self.calculate,
        ).pack(fill=tk.X, pady=(2, 5))
        ttk.Button(
            self.controls,
            text="Export CSV and three PNG plots",
            command=self.export_results,
        ).pack(fill=tk.X, pady=(0, 10))

        summary_frame = ttk.LabelFrame(
            self.controls,
            text="Current result",
            padding=10,
        )
        summary_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            summary_frame,
            textvariable=self.summary,
            justify=tk.LEFT,
            wraplength=390,
        ).pack(anchor="w", fill=tk.X)
        ttk.Label(
            self.controls,
            textvariable=self.status,
            foreground="#3B6EA8",
            justify=tk.LEFT,
            wraplength=400,
        ).pack(anchor="w", fill=tk.X)

    def browse_preset_file(self) -> None:
        selected_path = filedialog.askopenfilename(
            title="Select oven preset JSON",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            initialdir=Path(self.preset_path_var.get()).parent,
        )
        if selected_path:
            self.preset_path_var.set(selected_path)
            self.refresh_presets()

    def refresh_presets(self, show_error: bool = True) -> None:
        try:
            preset_path = Path(self.preset_path_var.get())
            with preset_path.open("r", encoding="utf-8") as preset_file:
                presets = json.load(preset_file)
            if not isinstance(presets, dict):
                raise ValueError("The preset JSON root must be an object.")
            names = sorted(presets)
            self.preset_box.configure(values=names)
            if self.preset_choice_var.get() not in names:
                self.preset_choice_var.set(names[0] if names else "")
            self.status.set(f"Loaded {len(names)} saved oven profile(s).")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.preset_box.configure(values=())
            self.preset_choice_var.set("")
            self.status.set(f"Could not load presets: {error}")
            if show_error:
                messagebox.showerror("Could not load presets", str(error))

    def _read_assumptions(self) -> tuple[ModelAssumptions, float]:
        assumptions = ModelAssumptions(
            pla_initial_g=float(self.pla_initial_var.get()),
            char_yield=float(self.char_yield_percent_var.get()) / 100.0,
            alpha_start_temperature_c=float(self.alpha_start_var.get()),
            alpha_end_temperature_c=float(self.alpha_end_var.get()),
            reaction_preexponential_per_min=float(self.preexponential_var.get()),
            reaction_activation_energy_j_per_mol=(
                float(self.activation_energy_var.get()) * 1000.0
            ),
            flow_reference_temperature_c=float(self.flow_temperature_var.get()),
            flow_reference_pressure_atm=float(self.flow_pressure_var.get()),
        )
        time_step_min = float(self.time_step_var.get())
        return assumptions, time_step_min

    def calculate(self, show_error: bool = True) -> bool:
        """Run the model with visible inputs and update all three plot tabs."""

        try:
            preset_name = self.preset_choice_var.get().strip()
            if not preset_name:
                raise ValueError("Select a saved oven profile.")
            assumptions, time_step_min = self._read_assumptions()
            parameters = load_oven_preset(
                Path(self.preset_path_var.get()),
                preset_name,
            )
            results = simulate(parameters, time_step_min, assumptions)

            draw_co_curve(self.figures["co"], results)
            draw_temperature_and_alpha(self.figures["temperature"], results)
            draw_carbon_balance(self.figures["carbon"], results)
            for canvas in self.plot_canvases.values():
                canvas.draw_idle()

            self.current_results = results
            self.current_assumptions = assumptions
            self.summary.set(
                f"Preset: {preset_name}\n"
                f"Duration: {results.time_min[-1] / 60.0:.2f} h\n"
                f"Peak predicted CO: {np.max(results.predicted_co_ppm):,.1f} ppm\n"
                "Final carbon inventory: "
                f"{results.carbon_available_g[-1]:.6g} g"
            )
            self.status.set("Calculation complete. Plot tabs are up to date.")
            return True
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as error:
            self.status.set(f"Could not calculate: {error}")
            if show_error:
                messagebox.showerror("Could not calculate", str(error))
            return False

    def export_results(self) -> None:
        """Recalculate, then export the CSV and the three standalone plots."""

        if not self.calculate():
            return
        assert self.current_results is not None

        preset_name = self.preset_choice_var.get().strip()
        safe_name = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in preset_name
        )
        csv_path = self.output_dir / f"{safe_name}_preliminary_co_model.csv"
        try:
            write_results_csv(self.current_results, csv_path)
            plot_paths = create_plots(
                self.current_results,
                self.output_dir,
                preset_name,
            )
            self.status.set(
                f"Exported CSV and {len(plot_paths)} plots to {self.output_dir}."
            )
        except OSError as error:
            self.status.set(f"Could not export results: {error}")
            messagebox.showerror("Could not export results", str(error))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset-file",
        type=Path,
        default=PRESET_FILE,
        help="JSON file written by oven_parameters_tool.py",
    )
    parser.add_argument(
        "--preset",
        default=PRESET_NAME,
        help="Name of the preset object to simulate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for CSV and PNG outputs",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run without the desktop interface and immediately export results",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="With --batch, show the exported figures interactively",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if not arguments.batch:
        root = tk.Tk()
        PreliminaryCOModelTool(
            root,
            preset_path=arguments.preset_file,
            initial_preset=arguments.preset,
            output_dir=arguments.output_dir,
        )
        root.mainloop()
        return

    parameters = load_oven_preset(arguments.preset_file, arguments.preset)
    assumptions = ModelAssumptions()
    results = simulate(parameters, TIME_STEP_MIN, assumptions)

    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in arguments.preset
    )
    csv_path = arguments.output_dir / f"{safe_name}_preliminary_co_model.csv"
    write_results_csv(results, csv_path)
    plot_paths = create_plots(results, arguments.output_dir, arguments.preset)
    output_paths = [csv_path, *plot_paths]
    print_assumption_summary(
        results,
        arguments.preset,
        output_paths,
        assumptions,
        TIME_STEP_MIN,
    )

    if arguments.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()

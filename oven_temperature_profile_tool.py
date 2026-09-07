"""Interactive tool for creating two-stage debinding and sintering profiles."""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


@dataclass(frozen=True)
class TemperatureProfileParameters:
    """User-configurable temperatures, ramp rates, and hold durations."""

    start_temperature_c: float = 15.0
    debinding_temperature_c: float = 482.0
    debinding_ramp_rate_c_per_h: float = 55.6
    debinding_hold_h: float = 4.0
    sintering_temperature_c: float = 1052.0
    sintering_ramp_rate_c_per_h: float = 111.1
    sintering_hold_h: float = 5.0
    cooling_rate_c_per_h: float = 60.0


FIELD_SPECS = (
    (
        "start_temperature_c",
        "Starting temperature",
        "°C",
    ),
    (
        "debinding_temperature_c",
        "Debinding temperature",
        "°C",
    ),
    (
        "debinding_ramp_rate_c_per_h",
        "Ramp to debinding temperature",
        "°C/h",
    ),
    (
        "debinding_hold_h",
        "Debinding hold time",
        "h",
    ),
    (
        "sintering_temperature_c",
        "Sintering temperature",
        "°C",
    ),
    (
        "sintering_ramp_rate_c_per_h",
        "Ramp from debinding to sintering",
        "°C/h",
    ),
    (
        "sintering_hold_h",
        "Sintering hold time",
        "h",
    ),
    (
        "cooling_rate_c_per_h",
        "Cooling rate",
        "°C/h",
    ),
)


def validate_parameters(parameters: TemperatureProfileParameters) -> None:
    """Raise ValueError when a profile cannot be constructed safely."""

    for field in fields(parameters):
        value = getattr(parameters, field.name)
        if not isfinite(value):
            raise ValueError(f"{field.name} must be a finite number.")

    if parameters.debinding_temperature_c <= parameters.start_temperature_c:
        raise ValueError(
            "Debinding temperature must be higher than the starting temperature."
        )
    if parameters.sintering_temperature_c <= parameters.debinding_temperature_c:
        raise ValueError(
            "Sintering temperature must be higher than the debinding temperature."
        )
    if parameters.debinding_ramp_rate_c_per_h <= 0:
        raise ValueError("The debinding ramp rate must be greater than zero.")
    if parameters.sintering_ramp_rate_c_per_h <= 0:
        raise ValueError("The sintering ramp rate must be greater than zero.")
    if parameters.cooling_rate_c_per_h <= 0:
        raise ValueError("The cooling rate must be greater than zero.")
    if parameters.debinding_hold_h < 0:
        raise ValueError("The debinding hold time cannot be negative.")
    if parameters.sintering_hold_h < 0:
        raise ValueError("The sintering hold time cannot be negative.")


def build_temperature_profile(parameters: TemperatureProfileParameters):
    """Return the stage vertices and timing details for a two-stage profile."""

    validate_parameters(parameters)

    debinding_ramp_h = (
        parameters.debinding_temperature_c - parameters.start_temperature_c
    ) / parameters.debinding_ramp_rate_c_per_h
    debinding_hold_end_h = debinding_ramp_h + parameters.debinding_hold_h

    sintering_ramp_h = (
        parameters.sintering_temperature_c
        - parameters.debinding_temperature_c
    ) / parameters.sintering_ramp_rate_c_per_h
    sintering_ramp_end_h = debinding_hold_end_h + sintering_ramp_h
    sintering_hold_end_h = sintering_ramp_end_h + parameters.sintering_hold_h

    cooling_h = (
        parameters.sintering_temperature_c - parameters.start_temperature_c
    ) / parameters.cooling_rate_c_per_h
    total_h = sintering_hold_end_h + cooling_h

    times_h = [
        0.0,
        debinding_ramp_h,
        debinding_hold_end_h,
        sintering_ramp_end_h,
        sintering_hold_end_h,
        total_h,
    ]
    temperatures_c = [
        parameters.start_temperature_c,
        parameters.debinding_temperature_c,
        parameters.debinding_temperature_c,
        parameters.sintering_temperature_c,
        parameters.sintering_temperature_c,
        parameters.start_temperature_c,
    ]
    timing = {
        "debinding_ramp_h": debinding_ramp_h,
        "debinding_hold_end_h": debinding_hold_end_h,
        "sintering_ramp_h": sintering_ramp_h,
        "sintering_ramp_end_h": sintering_ramp_end_h,
        "sintering_hold_end_h": sintering_hold_end_h,
        "cooling_h": cooling_h,
        "total_h": total_h,
    }
    return times_h, temperatures_c, timing


def create_temperature_figure(
    parameters: TemperatureProfileParameters,
    figure: Figure | None = None,
):
    """Create or redraw the combined temperature-profile figure."""

    times_h, temperatures_c, timing = build_temperature_profile(parameters)
    if figure is None:
        figure = Figure(figsize=(10.5, 6.2), dpi=100)
    else:
        figure.clear()

    ax = figure.add_subplot(111)
    ax.plot(
        times_h,
        temperatures_c,
        color="#1f77b4",
        linewidth=2.4,
        marker="o",
        markersize=4,
        label="Oven temperature profile",
    )
    ax.axvspan(
        timing["debinding_ramp_h"],
        timing["debinding_hold_end_h"],
        color="#e15759",
        alpha=0.18,
        label="Debinding hold",
    )
    ax.axvspan(
        timing["sintering_ramp_end_h"],
        timing["sintering_hold_end_h"],
        color="#f28e2b",
        alpha=0.22,
        label="Sintering hold",
    )

    ax.annotate(
        (
            f"{parameters.debinding_temperature_c:g} °C, "
            f"{parameters.debinding_hold_h:g} h"
        ),
        xy=(
            (
                timing["debinding_ramp_h"]
                + timing["debinding_hold_end_h"]
            )
            / 2,
            parameters.debinding_temperature_c,
        ),
        xytext=(0, 11),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )
    ax.annotate(
        (
            f"{parameters.sintering_temperature_c:g} °C, "
            f"{parameters.sintering_hold_h:g} h"
        ),
        xy=(
            (
                timing["sintering_ramp_end_h"]
                + timing["sintering_hold_end_h"]
            )
            / 2,
            parameters.sintering_temperature_c,
        ),
        xytext=(0, 11),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )

    summary = (
        f"Ramp 1: {timing['debinding_ramp_h']:.2f} h\n"
        f"Ramp 2: {timing['sintering_ramp_h']:.2f} h\n"
        f"Cooling: {timing['cooling_h']:.2f} h\n"
        f"Total: {timing['total_h']:.2f} h"
    )
    ax.text(
        0.985,
        0.03,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#bbbbbb",
            "alpha": 0.92,
        },
    )

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Combined Debinding and Sintering Temperature Profile")
    ax.set_xlim(left=0)
    ax.set_ylim(
        bottom=min(0, parameters.start_temperature_c),
        top=parameters.sintering_temperature_c * 1.12,
    )
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper left")
    figure.tight_layout()
    return figure, timing


class TemperatureProfileTool:
    """Tkinter interface for previewing and saving custom profiles."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Oven Temperature Profile Tool")
        self.root.geometry("1280x760")
        self.root.minsize(1050, 650)

        defaults = TemperatureProfileParameters()
        self.variables = {
            name: tk.StringVar(value=f"{getattr(defaults, name):g}")
            for name, _, _ in FIELD_SPECS
        }
        self.status = tk.StringVar(value="Enter values and generate a profile.")

        controls = ttk.Frame(root, padding=14)
        controls.pack(side=tk.LEFT, fill=tk.Y)
        plot_panel = ttk.Frame(root, padding=(0, 10, 10, 10))
        plot_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(
            controls,
            text="Profile parameters",
            font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))

        for row, (name, label, unit) in enumerate(FIELD_SPECS, start=1):
            ttk.Label(controls, text=label).grid(
                row=row,
                column=0,
                sticky="w",
                padx=(0, 8),
                pady=4,
            )
            entry = ttk.Entry(controls, textvariable=self.variables[name], width=12)
            entry.grid(row=row, column=1, sticky="ew", pady=4)
            entry.bind("<Return>", lambda _event: self.generate_profile())
            ttk.Label(controls, text=unit).grid(
                row=row,
                column=2,
                sticky="w",
                padx=(6, 0),
                pady=4,
            )

        button_row = len(FIELD_SPECS) + 1
        ttk.Button(
            controls,
            text="Generate profile",
            command=self.generate_profile,
        ).grid(row=button_row, column=0, columnspan=3, sticky="ew", pady=(16, 5))
        ttk.Button(
            controls,
            text="Save PNG…",
            command=self.save_png,
        ).grid(row=button_row + 1, column=0, columnspan=3, sticky="ew", pady=5)
        ttk.Label(
            controls,
            textvariable=self.status,
            wraplength=300,
            foreground="#444444",
        ).grid(
            row=button_row + 2,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(14, 0),
        )

        self.figure = Figure(figsize=(10.5, 6.2), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_panel)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_panel, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.generate_profile(show_error=False)

    def read_parameters(self) -> TemperatureProfileParameters:
        """Read and validate all entry values."""

        values = {}
        for name, label, _unit in FIELD_SPECS:
            raw_value = self.variables[name].get().strip().replace(",", ".")
            try:
                values[name] = float(raw_value)
            except ValueError as error:
                raise ValueError(f"{label} must be a number.") from error
        parameters = TemperatureProfileParameters(**values)
        validate_parameters(parameters)
        return parameters

    def generate_profile(self, show_error: bool = True):
        """Redraw the chart from the current entries."""

        try:
            parameters = self.read_parameters()
            _, timing = create_temperature_figure(parameters, self.figure)
            self.canvas.draw_idle()
            self.status.set(
                f"Profile generated. Total process time: {timing['total_h']:.2f} h"
            )
            return parameters
        except ValueError as error:
            self.status.set(str(error))
            if show_error:
                messagebox.showerror("Invalid profile parameters", str(error))
            return None

    def save_png(self) -> None:
        """Validate the entries and save the current profile as a PNG."""

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

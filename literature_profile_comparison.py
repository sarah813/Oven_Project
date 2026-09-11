"""Compare reported two-stage debinding and sintering programs.

Debinding and sintering share one axis but are displayed as consecutive stage
groups. Missing ramp durations are not inferred: reported holds are horizontal
segments, maximum-only values are dots, and temperature ranges are vertical
segments.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


START_TEMPERATURE_C = 15.0
SINTERING_OFFSET_H = 19.0
OUTPUT_DIR = Path("oven_outputs")
OUTPUT_PATH = OUTPUT_DIR / "literature_temperature_profile_comparison.png"
OUTPUT_PDF_PATH = OUTPUT_DIR / "literature_temperature_profile_comparison.pdf"
PLOT_DPI = 300


COLORS = {
    "schuessler": "#4C78A8",
    "yan": "#F58518",
    "pellegrini": "#54A24B",
    "jones_a": "#E45756",
    "jones_b": "#B279A2",
    "lawal": "#72B7B2",
    "downard": "#9D755D",
    "cheng": "#E6AB02",
}


LEGEND_LABELS = {
    "schuessler": (
        "Schuessler 2023 — debind: Ar or vacuum (25 mbar); "
        "sinter: 50% Ar / 50% H₂ or vacuum"
    ),
    "yan": (
        "Yan 2018 — debind: air or carbon-embedded; "
        "sinter: atmosphere n/r"
    ),
    "pellegrini": (
        "Pellegrini 2024 — debind: open atmosphere; sinter: CO₂"
    ),
    "jones_a": (
        "Jones 2026 exp. A — debind: open atmosphere; "
        "sinter: CO₂-assisted"
    ),
    "jones_b": (
        "Jones 2026 exp. B — debind: open atmosphere; "
        "sinter: CO₂-assisted"
    ),
    "lawal": (
        "Lawal 2026 — debind/sinter: Ar (500 mL/min) "
        "or reducing carbon"
    ),
    "downard": (
        "Downard 2024 — debind: open atmosphere; sinter: CO₂"
    ),
    "cheng": (
        "Cheng et al. 2022 — debind/sinter: atmosphere not reported"
    ),
}


def plot_timed_stage(
    ax,
    *,
    start_h,
    target_temperature_c,
    heating_rate_c_per_h,
    hold_h,
    color,
    cooling_rate_c_per_h=None,
    linewidth=2.4,
):
    """Draw a stage whose heating rate is known and return its end time."""

    heating_h = (
        target_temperature_c - START_TEMPERATURE_C
    ) / heating_rate_c_per_h
    heating_end_h = start_h + heating_h
    times_h = [start_h, heating_end_h]
    temperatures_c = [START_TEMPERATURE_C, target_temperature_c]

    if hold_h > 0:
        times_h.append(heating_end_h + hold_h)
        temperatures_c.append(target_temperature_c)

    if cooling_rate_c_per_h is not None:
        cooling_h = (
            target_temperature_c - START_TEMPERATURE_C
        ) / cooling_rate_c_per_h
        times_h.append(times_h[-1] + cooling_h)
        temperatures_c.append(START_TEMPERATURE_C)

    ax.plot(
        times_h,
        temperatures_c,
        color=color,
        linewidth=linewidth,
        linestyle="-",
        solid_capstyle="round",
    )
    return times_h[-1]


def plot_reported_hold(
    ax,
    *,
    start_h,
    hold_h,
    temperature_c,
    color,
    linewidth=3.2,
    zorder=4,
):
    """Draw a hold without implying an unreported ramp duration."""

    ax.plot(
        [start_h, start_h + hold_h],
        [temperature_c, temperature_c],
        color=color,
        linewidth=linewidth,
        linestyle="-",
        solid_capstyle="round",
        zorder=zorder,
    )


def add_temperature_label(
    ax,
    *,
    x,
    temperature_c,
    color,
    text=None,
    offset=(0, 9),
    horizontal_alignment="center",
):
    """Place a readable, color-matched label beside a reported maximum."""

    ax.annotate(
        text or f"{temperature_c:g} °C",
        (x, temperature_c),
        xytext=offset,
        textcoords="offset points",
        ha=horizontal_alignment,
        va="bottom" if offset[1] >= 0 else "top",
        fontsize=8.5,
        fontweight="bold",
        color=color,
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
        },
        zorder=8,
    )


def create_comparison_figure():
    """Create one plot with debinding followed by sintering."""

    figure, ax = plt.subplots(figsize=(18, 9), constrained_layout=True)

    # Debinding stage: timed profiles.
    plot_timed_stage(
        ax,
        start_h=0,
        target_temperature_c=483,
        heating_rate_c_per_h=0.92 * 60,
        hold_h=0,
        color=COLORS["pellegrini"],
    )
    plot_timed_stage(
        ax,
        start_h=0,
        target_temperature_c=482,
        heating_rate_c_per_h=55.6,
        hold_h=4,
        color=COLORS["jones_a"],
    )
    plot_timed_stage(
        ax,
        start_h=0,
        target_temperature_c=482,
        heating_rate_c_per_h=111,
        hold_h=4,
        color=COLORS["jones_b"],
    )
    plot_timed_stage(
        ax,
        start_h=0,
        target_temperature_c=482,
        heating_rate_c_per_h=55,
        hold_h=4,
        cooling_rate_c_per_h=120,
        color=COLORS["lawal"],
    )

    # Cheng, Loh & Zhang (2022), transcribed from Fig. 4. Times are shown in
    # hours here (the source figure reports minutes): 0, 32, 160, 256, 416 min.
    cheng_debinding_times_h = [0, 32 / 60, 160 / 60, 256 / 60, 416 / 60]
    cheng_debinding_temperatures_c = [15, 200, 200, 480, 480]
    ax.plot(
        cheng_debinding_times_h,
        cheng_debinding_temperatures_c,
        color=COLORS["cheng"],
        linewidth=2.4,
        linestyle="-",
        solid_capstyle="round",
    )

    # Schuessler: four reported 2 h debinding holds; ramps are not reported.
    for index, temperature_c in enumerate([240, 320, 400, 600]):
        plot_reported_hold(
            ax,
            start_h=index * 2,
            hold_h=2,
            temperature_c=temperature_c,
            color=COLORS["schuessler"],
        )

    # Yan: only the 200–400 °C binder-removal range is reported. Use a pale
    # area across step 1 instead of implying a time-resolved profile.
    ax.fill_between(
        [-0.5, SINTERING_OFFSET_H - 1.5],
        [200, 200],
        [400, 400],
        color=COLORS["yan"],
        alpha=0.10,
        linewidth=0,
        zorder=1,
    )

    # Downard: only the maximum debinding temperature is reported.
    ax.scatter(
        0.7,
        483,
        color=COLORS["downard"],
        marker="o",
        s=48,
        zorder=6,
    )

    # Color-matched debinding maximum labels.
    for index, temperature_c in enumerate([240, 320, 400, 600]):
        add_temperature_label(
            ax,
            x=index * 2 + 1,
            temperature_c=temperature_c,
            color=COLORS["schuessler"],
            offset=(0, 7),
        )
    add_temperature_label(
        ax,
        x=15.7,
        temperature_c=300,
        color=COLORS["yan"],
        text="200–400 °C range",
        offset=(0, 0),
        horizontal_alignment="right",
    )
    add_temperature_label(
        ax,
        x=(483 - START_TEMPERATURE_C) / (0.92 * 60),
        temperature_c=483,
        color=COLORS["pellegrini"],
        offset=(-4, 26),
        horizontal_alignment="right",
    )
    add_temperature_label(
        ax,
        x=(482 - START_TEMPERATURE_C) / 55.6 + 2,
        temperature_c=482,
        color=COLORS["jones_a"],
        offset=(0, 10),
    )
    add_temperature_label(
        ax,
        x=(482 - START_TEMPERATURE_C) / 111 + 2,
        temperature_c=482,
        color=COLORS["jones_b"],
        offset=(0, 10),
    )
    add_temperature_label(
        ax,
        x=(482 - START_TEMPERATURE_C) / 55 + 4,
        temperature_c=482,
        color=COLORS["lawal"],
        offset=(5, -10),
        horizontal_alignment="left",
    )
    add_temperature_label(
        ax,
        x=32 / 60,
        temperature_c=200,
        color=COLORS["cheng"],
        offset=(0, 8),
    )
    add_temperature_label(
        ax,
        x=(160 + 256) / 2 / 60,
        temperature_c=480,
        color=COLORS["cheng"],
        offset=(0, 10),
    )
    add_temperature_label(
        ax,
        x=0.7,
        temperature_c=483,
        color=COLORS["downard"],
        offset=(6, 7),
        horizontal_alignment="left",
    )

    # Sintering stage: timed profiles, shifted right as a second stage group.
    plot_timed_stage(
        ax,
        start_h=SINTERING_OFFSET_H,
        target_temperature_c=1038,
        heating_rate_c_per_h=111,
        hold_h=5,
        color=COLORS["jones_a"],
    )
    plot_timed_stage(
        ax,
        start_h=SINTERING_OFFSET_H,
        target_temperature_c=1038,
        heating_rate_c_per_h=222,
        hold_h=5,
        color=COLORS["jones_b"],
    )
    plot_timed_stage(
        ax,
        start_h=SINTERING_OFFSET_H,
        target_temperature_c=1052,
        heating_rate_c_per_h=111,
        hold_h=5,
        cooling_rate_c_per_h=120,
        color=COLORS["lawal"],
    )

    # Cheng et al. sintering segment: 480 °C to 1075 °C from 416 to 512 min,
    # followed by a hold to 800 min in the source figure.
    cheng_sintering_times_h = [
        SINTERING_OFFSET_H,
        SINTERING_OFFSET_H + (512 - 416) / 60,
        SINTERING_OFFSET_H + (800 - 416) / 60,
    ]
    ax.plot(
        cheng_sintering_times_h,
        [480, 1075, 1075],
        color=COLORS["cheng"],
        linewidth=2.4,
        linestyle="-",
        solid_capstyle="round",
    )

    # Missing sintering ramps: draw only the reported hold segments.
    plot_reported_hold(
        ax,
        start_h=SINTERING_OFFSET_H,
        hold_h=3,
        temperature_c=1075,
        color=COLORS["schuessler"],
    )
    plot_reported_hold(
        ax,
        start_h=SINTERING_OFFSET_H,
        hold_h=5,
        temperature_c=1057,
        color=COLORS["pellegrini"],
        linewidth=7,
        zorder=4,
    )
    plot_reported_hold(
        ax,
        start_h=SINTERING_OFFSET_H,
        hold_h=5,
        temperature_c=1057,
        color=COLORS["downard"],
        linewidth=2.4,
        zorder=5,
    )

    # Yan: only the maximum sintering temperature is reported.
    ax.scatter(
        SINTERING_OFFSET_H + 0.35,
        1000,
        color=COLORS["yan"],
        marker="o",
        s=48,
        zorder=6,
    )

    # Color-matched sintering maximum labels.
    add_temperature_label(
        ax,
        x=SINTERING_OFFSET_H + 1.5,
        temperature_c=1075,
        color=COLORS["schuessler"],
        offset=(0, 12),
    )
    add_temperature_label(
        ax,
        x=SINTERING_OFFSET_H + 0.35,
        temperature_c=1000,
        color=COLORS["yan"],
        offset=(7, -9),
        horizontal_alignment="left",
    )
    add_temperature_label(
        ax,
        x=SINTERING_OFFSET_H + (512 - 416) / 60 + 2.4,
        temperature_c=1075,
        color=COLORS["cheng"],
        offset=(0, 10),
    )
    add_temperature_label(
        ax,
        x=SINTERING_OFFSET_H + 1.8,
        temperature_c=1057,
        color=COLORS["pellegrini"],
        offset=(0, -13),
    )
    add_temperature_label(
        ax,
        x=SINTERING_OFFSET_H + 4.2,
        temperature_c=1057,
        color=COLORS["downard"],
        offset=(0, 9),
    )
    add_temperature_label(
        ax,
        x=SINTERING_OFFSET_H + (1038 - START_TEMPERATURE_C) / 111 + 2.5,
        temperature_c=1038,
        color=COLORS["jones_a"],
        offset=(0, -10),
    )
    add_temperature_label(
        ax,
        x=SINTERING_OFFSET_H + (1038 - START_TEMPERATURE_C) / 222 + 2.5,
        temperature_c=1038,
        color=COLORS["jones_b"],
        offset=(0, -10),
    )
    add_temperature_label(
        ax,
        x=SINTERING_OFFSET_H + (1052 - START_TEMPERATURE_C) / 111 + 5,
        temperature_c=1052,
        color=COLORS["lawal"],
        offset=(5, 9),
        horizontal_alignment="left",
    )

    # Make the two steps visually sequential while keeping a single plot.
    divider_h = SINTERING_OFFSET_H - 1.5
    ax.axvline(divider_h, color="#777777", linewidth=1.2, alpha=0.65)
    ax.axvspan(-0.5, divider_h, color="#E45756", alpha=0.025, zorder=0)
    ax.axvspan(divider_h, 42.5, color="#4C78A8", alpha=0.025, zorder=0)
    ax.text(
        8.3,
        1125,
        "STEP 1 — DEBINDING",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#555555",
    )
    ax.text(
        30.5,
        1125,
        "STEP 2 — SINTERING",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#555555",
    )

    # Reset the displayed time labels at the start of the second stage group.
    debinding_ticks = [0, 5, 10, 15]
    sintering_relative_ticks = [0, 5, 10, 15, 20]
    sintering_ticks = [
        SINTERING_OFFSET_H + value for value in sintering_relative_ticks
    ]
    ax.set_xticks(
        debinding_ticks + sintering_ticks,
        [str(value) for value in debinding_ticks + sintering_relative_ticks],
    )

    ax.set_xlim(-0.5, 42.5)
    ax.set_ylim(0, 1165)
    ax.set_xlabel("Time within each process stage [h]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(
        "Reported Two-Step Debinding and Sintering Profiles",
        loc="left",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.25)

    # Legend encodes papers/experiments by color only.
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[key],
            linewidth=3,
            linestyle="-",
            label=LEGEND_LABELS[key],
        )
        for key in [
            "schuessler",
            "yan",
            "pellegrini",
            "jones_a",
            "jones_b",
            "lawal",
            "downard",
            "cheng",
        ]
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        fontsize=8.5,
        frameon=True,
        title="Paper / experiment and atmosphere",
        title_fontsize=9,
    )

    figure.suptitle(
        "Literature Comparison for Copper-Filled Material Extrusion",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.015,
        (
            "The x-axis resets for step 2; the gap is for visual separation, not "
            "elapsed process time. Missing ramps are omitted: horizontal lines "
            "show holds, dots show maximum-only data, and Yan's pale area "
            "shows its debinding range. Timed curves use a 15 °C start. "
            "n/r = not reported."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )
    return figure


def main():
    figure = create_comparison_figure()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=PLOT_DPI, bbox_inches="tight")
    figure.savefig(OUTPUT_PDF_PATH, bbox_inches="tight")
    print(f"Plot saved to: {OUTPUT_PATH}")
    print(f"Plot saved to: {OUTPUT_PDF_PATH}")
    plt.show()


if __name__ == "__main__":
    main()

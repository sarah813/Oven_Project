# Oven Parameters Tool

The Oven Parameters Tool creates and visualizes oven programs containing debinding, sintering, atmosphere phases, and cooling.

## Requirements

- Python 3.10 or newer
- NumPy
- Matplotlib
- Tkinter (usually included with Python; on Linux install the system `python3-tk` package)

Install the Python packages with:

```bash
python -m pip install numpy matplotlib
```

## Start the tool

Run this command from the project directory:

```bash
python oven_parameters_tool.py
```

The graphical interface opens with the profile plot on the right and the program controls on the left.

## Basic workflow

1. Select the **Cycle type**:
   - **Debinding only**: debinding followed by cooling.
   - **Sintering only**: sintering followed by cooling. Debinding is assumed to have happened previously.
   - **Debinding + sintering**: the complete cycle.
2. Enter the **Starting temperature**.
3. Optionally enter a **Start time** in the format `YYYY-MM-DD HH:MM`. The plot then shows elapsed time and real clock time, and the summary reports the finish time.
4. Configure the debinding and/or sintering steps:
   - **Target [C]** is the step temperature.
   - **Ramp [C/h]** is the heating rate.
   - **Hold [h]** is the plateau duration.
5. Configure the atmosphere phases.
6. Click **Generate profile** to redraw the profile and update the summary.

The summary shows total duration, start time, sintering start, cooling start, finish, maximum temperature, stage durations, and the number of gas phases.

## Atmosphere phases

Each gas phase has a gas, process stage, definition method, range, and flow rate.

### Stage

Choose where the gas is active:

- **Debinding**
- **Sintering**
- **Cooling**
- **All**

The available stage choices are restricted automatically for the selected cycle type.

### Defined by

- **Time**: enter the start and end in elapsed hours.
- **Temperature**: enter the temperature limits. The gas is active where the profile crosses that temperature range within the selected stage.
- **Whole phase**: the gas covers the complete selected stage. The From/To fields are disabled.

For nitrogen, CO2, and argon, enter a positive flow rate in L/min. Air is treated as ambient and does not require a flow rate.

The plot displays atmosphere bands above the temperature axes. Repeated labels for the same gas are suppressed where possible.

## Presets

Use the preset controls to:

- **Save preset**: store the current settings under a new name.
- **Change preset**: overwrite an existing preset.
- **Load**: load a saved preset.
- **Delete preset**: remove a saved preset.

Presets are stored in `oven_presets.json`. Keep this file with the Python script if the project is moved.

## Plot and cooling model

The upper plot contains:

- The programmed heating and hold profile.
- A smooth cooling approximation.
- The PLA thermal-debinding range, 200-500 C.
- The copper sintering range, 750-1083 C.
- The copper melting reference at 1083 C.
- Atmosphere shading and labels.

The cooling curve uses the measured cooling data incorporated in the program. For sintering cycles it uses the fitted normalized exponential shape with the current coefficient `COOLING_SHAPE_K` in `oven_parameters_tool.py`. Debinding-only cycles use their separate cooling shape.

The lower plot shows two mass estimates:

- **Composite mass remaining (Cu + PLA)**: total remaining mass relative to the starting composite.
- **PLA/binder remaining**: remaining PLA as a percentage of the initial PLA amount.

These are engineering estimates based on the implemented decomposition model, not a live mass measurement. A sintering-only cycle starts after debinding, so the composite curve starts near the copper fraction and the PLA curve is near its residual value.

## Export a PNG

Click **Save PNG...** after generating a profile. The image is saved at 300 dpi with the current plot and annotations.

## Generate plots for every saved cycle

The notebook [`thermal_cycle_plots.ipynb`](thermal_cycle_plots.ipynb) loads all entries in `oven_presets.json` and creates one reference-style PNG per preset.

Run it in Jupyter or VS Code, then execute all cells. The images are written to a `thermal_cycle_plots/` directory.

## Troubleshooting

- If the chart does not change after editing fields, click **Generate profile**.
- If a saved preset is not visible, verify that it was saved to `oven_presets.json` and reload the application.
- If Tkinter cannot be imported, install the operating-system Tk package and restart Python.
- If a start time is rejected, use exactly `YYYY-MM-DD HH:MM`, for example `2026-11-20 18:00`.

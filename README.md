# Geographic Externality Tax Visualizer

This repo provides a standalone Python script (`get_visualizer.py`) that creates GIS-esque depictions of a GET-style tax surface with **no third-party dependencies**.

## Where to run it

Run it from the repo root directory:

```bash
cd /workspace/geographic-externality-tax
python3 get_visualizer.py --help
```

If Python is installed, this should print CLI options immediately.

## Model rendered

The script visualizes these two equations:

- Effective distance
  - `d_f(x) = Σ α_j · d(x, h_j)`
- Tax intensity
  - `T(x) = base_rate · (1 + beta · d_f(x)/d_ref) · RL`

Where:
- `x` = transaction point
- `h_j` = hearths
- `α_j` = normalized hearth weights
- `RL` = revenue-to-local-investment ratio

## Modes

1. `generic` (arbitrary 2D map)
2. `usa` (contiguous USA lon/lat approximation polygon)

## Examples

### Generic 2D heatmap

```bash
python3 get_visualizer.py \
  --mode generic \
  --hearths 0:0,5:5 \
  --weights 0.7,0.3 \
  --base-rate 0.05 \
  --beta 1.4 \
  --rl 1.2 \
  --resolution 140 \
  --output generic_map.svg
```

### USA-style map

```bash
python3 get_visualizer.py \
  --mode usa \
  --hearths=-95:39,-122:37,-74:40 \
  --weights 0.5,0.2,0.3 \
  --base-rate 0.05 \
  --beta 1.0 \
  --rl 1.1 \
  --resolution 140 \
  --output usa_map.svg
```

## How to see it in action

The command generates an SVG file in the current folder.

- If you are local on your own machine: double-click the SVG file (it opens in a browser).
- From terminal on Linux:
  ```bash
  xdg-open usa_map.svg
  ```
- From terminal on macOS:
  ```bash
  open usa_map.svg
  ```

You can also host the folder briefly and open in any browser:

```bash
python3 -m http.server 8000
```

Then visit:

- `http://localhost:8000/usa_map.svg`
- `http://localhost:8000/generic_map.svg`


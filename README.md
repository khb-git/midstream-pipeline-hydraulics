# Midstream Pipeline Hydraulics

A Python implementation of steady-state pipeline hydraulics for both
liquid and gas transmission lines, with support for multi-segment routes,
pump stations, elevation profiles, and MAOP constraint checking.

This is the midstream complement to a broader oil & gas portfolio.
For upstream (decline curve analysis & probabilistic reserves), see
[decline-curve-analysis](https://github.com/khb-git/decline-curve-analysis).

## Why this project

Pipeline hydraulics is the foundational engineering discipline of midstream
oil & gas. Every operating decision — what to pump at what rate, where to
locate pump stations, how to handle elevation gain, whether a given flow
is achievable given MAOP limits — flows from solving the steady-state
hydraulic profile.

The headline analysis: given a pipeline route (geometry, elevation, fluid)
and operating conditions, produce the pressure profile along the route,
flag MAOP violations and minimum-pressure violations, and quantify the
total pumping power required.

## Features

- **Friction factor**: Colebrook-White (implicit, solved with Brent's
  method) and Swamee-Jain (explicit ~1% approximation), with automatic
  laminar/transitional/turbulent classification
- **Liquid hydraulics**: Darcy-Weisbach with elevation handling for
  incompressible flow
- **Gas hydraulics**: Weymouth and Panhandle B equations with elevation
  correction terms
- **Pipeline network model**: ordered segments + pump stations, MAOP
  per-segment
- **Hydraulic profile calculation**: walks a pipeline computing pressure
  at intermediate points; flags MAOP and NPSH violations
- **Hydraulic gradient visualization**: the canonical two-panel pipeline
  chart (pressure profile on top, terrain profile on bottom)
- **Fluid library**: realistic properties for WTI, Brent, WCS heavy,
  dilbit, gasoline, diesel

## Quick start

```python
from src.io.examples import cushing_to_houston_example
from src.network.profile import compute_profile
from src.viz.gradient import plot_hydraulic_gradient

# Load a pre-configured example pipeline
pipeline = cushing_to_houston_example()
print(pipeline.summary())

# Compute the hydraulic profile at 270k bpd
profile = compute_profile(pipeline, flow_rate_m3_s=0.5)
print(f"Outlet pressure: {profile.outlet_pressure_psi:.0f} psi")
print(f"Total pumping power: {profile.total_pump_power_kw:,.0f} kW")
print(f"Feasible operation: {profile.is_feasible}")

# Visualize the hydraulic gradient
fig = plot_hydraulic_gradient(profile, pipeline, min_delivery_pressure_psi=100)
fig.savefig("gradient.png")
```

## Installation

```bash
git clone https://github.com/khb-git/midstream-pipeline-hydraulics.git
cd midstream-pipeline-hydraulics
pip install -e ".[dev,viz]"
pytest
```

## Project structure

```
midstream-pipeline-hydraulics/
├── src/
│   ├── hydraulics/
│   │   ├── friction.py          # Colebrook-White, Swamee-Jain
│   │   ├── pressure_drop.py     # Darcy-Weisbach (liquid)
│   │   └── gas_flow.py          # Weymouth, Panhandle B
│   ├── fluids/
│   │   └── properties.py        # Fluid library + property correlations
│   ├── network/
│   │   ├── pipeline.py          # Pipeline, PipeSegment, PumpStation
│   │   └── profile.py           # Multi-segment profile calculation
│   ├── viz/
│   │   └── gradient.py          # Hydraulic gradient diagram
│   ├── optimization/            # (planned) Pump station placement
│   └── io/
│       └── examples.py          # Pre-configured example pipelines
├── tests/
├── notebooks/                   # Tutorial / demo notebooks
└── pyproject.toml
```

## Methodology

### Friction factor

Colebrook-White (1939) is the industry-standard turbulent friction
correlation:

```
1/√f = −2·log₁₀(ε/(3.7·D) + 2.51/(Re·√f))
```

It's implicit in f and is solved iteratively with Brent's method on the
residual. For laminar flow (Re < 2300) we use Hagen-Poiseuille (f = 64/Re).
In the transitional regime (2300 < Re < 4000) we linearly interpolate to
avoid discontinuities. The Swamee-Jain (1976) explicit approximation is
also available — it's ~1% accurate across the turbulent range and useful
where iteration is undesirable (e.g., inside an optimization loop).

### Pressure drop (liquid)

Darcy-Weisbach for steady-state incompressible flow:

```
ΔP = f·(L/D)·(ρ·v²/2) + ρ·g·Δz
```

The first term is friction loss; the second is hydrostatic elevation
change. Elevation gain costs pressure; elevation loss recovers it.

### Pressure drop (gas)

Gas density changes significantly with pressure, so we use the
Weymouth or Panhandle B forms of the general gas-flow equation:

```
Q ∝ √(P₁² − P₂² − ΔH) · D^a / √(G·T·L·Z)
```

with `a = 8/3` for Weymouth and `a = 2.53` for Panhandle B. The choice
depends on flow regime: Weymouth for short high-pressure lines, Panhandle
B for long transmission lines.

### Hydraulic profile

A pipeline is modeled as an ordered sequence of segments with pump
stations at segment boundaries. The profile is computed by walking
the pipeline from inlet to outlet:

1. Start with the inlet pressure
2. For each segment, discretize into small sub-segments and apply
   Darcy-Weisbach to each
3. At segment boundaries with pump stations, add the station's head boost
4. Track MAOP violations (pressure exceeds segment's MAOP) and NPSH
   violations (station inlet pressure below minimum)

## Validation

The test suite (50+ tests) covers:

- **Friction factor correctness**: laminar formula, smooth-pipe high-Re
  behavior, fully-rough independence from Re, agreement between
  Colebrook and Swamee-Jain
- **Pressure drop physics**: pure elevation gives ρgh, friction scales
  with length linearly and with velocity ~quadratically
- **Pipeline profile consistency**: multi-step profile calculation matches
  single-shot Darcy-Weisbach for a single segment
- **Pump station behavior**: stations increase outlet pressure; power
  calculation matches ρgQH/η
- **Gas flow scaling**: flow scales with diameter to the 8/3 power
  (Weymouth) and 2.53 power (Panhandle B); flow decreases with length

Run with:

```bash
pytest -v
```

## Roadmap

- [ ] Pump station placement optimization (LP-based)
- [ ] Batched-product transport modeling (multi-fluid sequences)
- [ ] Compressor station modeling for gas lines
- [ ] Demo notebook with interactive examples
- [ ] Streamlit dashboard for sensitivity analysis

## References

- Crane Co. (2013). *Technical Paper No. 410: Flow of Fluids Through
  Valves, Fittings, and Pipe*.
- Menon, E.S. (2005). *Gas Pipeline Hydraulics*. CRC Press.
- GPSA. (2012). *Engineering Data Book, Volume II*, Section 17.
- Colebrook, C.F. (1939). "Turbulent Flow in Pipes." *J. Inst. Civil Eng.*
  11, 133-156.
- Swamee, P.K. & Jain, A.K. (1976). "Explicit Equations for Pipe-Flow
  Problems." *J. Hyd. Div. ASCE* 102(5), 657-664.

## License

MIT

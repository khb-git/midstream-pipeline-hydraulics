"""
Hydraulic gradient and pipeline visualization.

The hydraulic gradient diagram is the canonical pipeline engineering chart:
- Top panel: pressure (or hydraulic head) along the pipeline, with
  pump stations marked, MAOP shown as upper limit, minimum delivery
  pressure as lower limit
- Bottom panel: terrain elevation profile

This visualization makes it immediately obvious where pressure problems
will arise (e.g., dropping below minimum at a peak, exceeding MAOP after
a station).
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from pipeline_hydraulics.network.profile import HydraulicProfile
from pipeline_hydraulics.network.pipeline import Pipeline


PLOT_STYLE = {
    "pressure_color": "#2980b9",
    "elevation_color": "#8e6e3e",
    "maop_color": "#c0392b",
    "min_pressure_color": "#27ae60",
    "station_color": "#34495e",
    "violation_color": "#e74c3c",
}


def plot_hydraulic_gradient(
    profile: HydraulicProfile,
    pipeline: Optional[Pipeline] = None,
    min_delivery_pressure_psi: float = 50.0,
    maop_psi: Optional[float] = None,
    figsize: tuple = (14, 8),
) -> Figure:
    """
    Draw the hydraulic gradient diagram: pressure vs distance with terrain.

    Args:
        profile: HydraulicProfile from compute_profile()
        pipeline: Pipeline (used for MAOP if not provided explicitly)
        min_delivery_pressure_psi: Minimum acceptable pressure (psi)
        maop_psi: MAOP for the constraint line; if None, uses pipeline's
            first segment MAOP
        figsize: Figure size

    Returns:
        Matplotlib Figure with two stacked panels
    """
    df = profile.to_dataframe()

    fig, (ax_p, ax_e) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Top panel: pressure
    ax_p.plot(
        df["distance_mi"], df["pressure_psi"],
        color=PLOT_STYLE["pressure_color"],
        lw=2.0,
        label="Pressure",
        zorder=3,
    )

    # MAOP line
    if maop_psi is None and pipeline is not None and pipeline.segments:
        maop_psi = pipeline.segments[0].maop_pa / 6894.76
    if maop_psi is not None:
        ax_p.axhline(
            maop_psi, color=PLOT_STYLE["maop_color"], lw=1.5, ls="--",
            label=f"MAOP ({maop_psi:.0f} psi)",
        )
        ax_p.fill_between(
            [df["distance_mi"].min(), df["distance_mi"].max()],
            maop_psi, maop_psi * 1.2,
            color=PLOT_STYLE["maop_color"], alpha=0.08,
        )

    # Minimum pressure line
    ax_p.axhline(
        min_delivery_pressure_psi,
        color=PLOT_STYLE["min_pressure_color"], lw=1.5, ls=":",
        label=f"Min pressure ({min_delivery_pressure_psi:.0f} psi)",
    )

    # Pump stations
    stations = df[df["is_station"] & df["station"].notna()]
    for _, st in stations.iterrows():
        ax_p.axvline(
            st["distance_mi"], color=PLOT_STYLE["station_color"],
            lw=0.8, ls="-", alpha=0.5, zorder=1,
        )
        ax_p.annotate(
            st["station"],
            xy=(st["distance_mi"], st["pressure_psi"]),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=8,
            color=PLOT_STYLE["station_color"],
            rotation=0,
            ha="left",
        )
        ax_p.scatter(
            st["distance_mi"], st["pressure_psi"],
            color=PLOT_STYLE["station_color"], s=40, zorder=5,
            edgecolors="white", linewidths=1,
        )

    # Violations
    if profile.maop_violations:
        ax_p.scatter(
            [v.distance_m / 1609.344 for v in profile.maop_violations],
            [v.pressure_psi for v in profile.maop_violations],
            color=PLOT_STYLE["violation_color"], s=20, zorder=6, marker="x",
            label="MAOP violation",
        )

    ax_p.set_ylabel("Pressure (psi)", fontsize=11)
    ax_p.set_title(
        f"Hydraulic Gradient — {profile.pipeline_name}",
        fontsize=12, fontweight="bold",
    )
    ax_p.legend(loc="upper right", framealpha=0.95, fontsize=9)
    ax_p.grid(True, alpha=0.3)
    ax_p.set_ylim(bottom=0)

    # Bottom panel: terrain elevation
    ax_e.fill_between(
        df["distance_mi"], df["elevation_m"], df["elevation_m"].min() - 20,
        color=PLOT_STYLE["elevation_color"], alpha=0.4,
    )
    ax_e.plot(df["distance_mi"], df["elevation_m"],
              color=PLOT_STYLE["elevation_color"], lw=1.2)
    ax_e.set_xlabel("Distance (miles)", fontsize=11)
    ax_e.set_ylabel("Elevation (m)", fontsize=11)
    ax_e.grid(True, alpha=0.3)
    ax_e.set_ylim(df["elevation_m"].min() - 20, df["elevation_m"].max() + 50)

    plt.tight_layout()
    return fig


def plot_pump_power_breakdown(
    profile: HydraulicProfile,
    pipeline: Pipeline,
    figsize: tuple = (10, 5),
) -> Figure:
    """Bar chart showing power consumption by pump station."""
    fig, ax = plt.subplots(figsize=figsize)
    df = profile.to_dataframe()
    stations_df = df[df["is_station"] & df["station"].notna() & (df["station"] != "Origin")]

    if len(stations_df) == 0:
        ax.text(0.5, 0.5, "No pump stations along the route",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    # Re-derive power per station from the pipeline definition
    powers = []
    names = []
    for seg_idx, station in pipeline.stations.items():
        if seg_idx == 0:
            continue
        # Use max head as nominal; profile constraints may have reduced it
        head_m = station.max_discharge_head_m
        power = station.power_required_kw(
            profile.flow_rate_m3_s, head_m, pipeline.fluid.density_kg_m3,
        )
        powers.append(power)
        names.append(station.name)

    ax.bar(names, powers, color=PLOT_STYLE["station_color"], alpha=0.85)
    ax.set_ylabel("Shaft power (kW)", fontsize=11)
    ax.set_title("Pump station power consumption", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig

#!/usr/bin/env python3
"""
Generate figures for Epistemic Frustration paper.

Figures:
1. Exploration-exploitation lifecycle with dimensional collapse
2. Epistemic frustration: high-D optimum projecting to contradictory low-D
3. Agent-based simulation of coordination-first transition
4. Exploitation trap as closed loop
5. DOF criterion: when error model grows faster than constraints
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Ellipse
from scipy.ndimage import gaussian_filter1d
import os

# Ensure figures directory exists
os.makedirs('../paper/figures', exist_ok=True)

# Set style - larger fonts for better readability when scaled
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 150


def fig1_lifecycle():
    """
    Figure 1: Exploration-exploitation lifecycle showing dimensional collapse.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    t = np.linspace(0, 10, 200)

    # Panel A: Effective dimensionality over time
    ax = axes[0]

    # Dimensionality starts high, consolidates, then collapses
    d_eff = 50 * np.exp(-0.1 * t) + 10 * np.exp(-0.5 * (t - 5)**2) + 5
    d_eff = gaussian_filter1d(d_eff, 3)

    ax.plot(t, d_eff, 'b-', linewidth=2.5, label='Effective dimensionality')
    ax.fill_between(t, 0, d_eff, alpha=0.2, color='blue')

    # Phase boundaries
    ax.axvline(2.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(5.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(8, color='gray', linestyle='--', alpha=0.5)

    ax.text(1.25, 56, 'Exploration', ha='center', fontsize=9)
    ax.text(4, 54, 'Consolidation', ha='center', fontsize=9)
    ax.text(6.75, 56, 'Exploitation', ha='center', fontsize=9)
    ax.text(9, 54, 'Brittleness', ha='center', fontsize=9)

    ax.set_xlabel('System maturity')
    ax.set_ylabel('Effective dimensionality ($D_{eff}$)')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 62)
    ax.set_title('A. Dimensional collapse')

    # Panel B: Efficiency vs resilience trade-off
    ax = axes[1]

    efficiency = 1 - np.exp(-0.5 * t)
    resilience = np.exp(-0.3 * t) + 0.1 * np.sin(t)
    resilience = np.clip(resilience, 0.05, 1)

    ax.plot(t, efficiency, 'g-', linewidth=2.5, label='Efficiency')
    ax.plot(t, resilience, 'r-', linewidth=2.5, label='Resilience')

    ax.axvline(2.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(5.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(8, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('System maturity')
    ax.set_ylabel('Normalized level')
    ax.legend(loc='center right')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.1)
    ax.set_title('B. Efficiency-resilience trade-off')

    # Panel C: Epistemic regime transition
    ax = axes[2]

    epistemic_first = np.exp(-0.4 * t)
    coordination_first = 1 - epistemic_first

    ax.fill_between(t, 0, epistemic_first, alpha=0.4, color='blue', label='Epistemic-first')
    ax.fill_between(t, epistemic_first, 1, alpha=0.4, color='orange', label='Coordination-first')

    ax.axvline(2.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(5.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(8, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('System maturity')
    ax.set_ylabel('Epistemic regime fraction')
    ax.legend(loc='center right')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.set_title('C. Epistemic regime transition')

    plt.tight_layout()
    plt.savefig('../paper/figures/fig1_lifecycle.pdf', bbox_inches='tight')
    plt.savefig('../paper/figures/fig1_lifecycle.png', bbox_inches='tight', dpi=200)
    plt.close()
    print("Generated: fig1_lifecycle.pdf (Figure 1)")


def fig2_frustration():
    """
    Figure 2: Epistemic frustration - high-D optimum projecting to contradictory low-D.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    # Panel A: High-D constraint satisfaction
    ax = axes[0]

    # Show multiple constraints as lines in 2D (representing high-D)
    theta = np.linspace(0, 2*np.pi, 7)[:-1]
    for i, t in enumerate(theta):
        x = np.cos(t)
        y = np.sin(t)
        ax.arrow(0, 0, 0.8*x, 0.8*y, head_width=0.08, head_length=0.05,
                fc=plt.cm.tab10(i), ec=plt.cm.tab10(i), linewidth=2)
        ax.text(1.1*x, 1.1*y, f'$C_{i+1}$', ha='center', va='center', fontsize=12)

    # Feasible region (small area in center)
    circle = Circle((0, 0), 0.15, fill=True, alpha=0.3, color='green')
    ax.add_patch(circle)
    ax.plot(0, 0, 'go', markersize=8)
    ax.text(0, -0.4, 'Feasible\nregion', ha='center', fontsize=11, color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='none'))

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('A. High-D constraint space')

    # Panel B: Projection to 1D creates contradiction
    ax = axes[1]

    # Show projection to a 1D axis
    ax.axhline(0, color='black', linewidth=1)

    # Different constraints project to different optimal points
    positions = [-0.8, -0.3, 0.2, 0.6, 0.9]
    colors = [plt.cm.tab10(i) for i in range(5)]
    labels = ['$C_1$', '$C_2$', '$C_3$', '$C_4$', '$C_5$']

    for pos, col, lab in zip(positions, colors, labels):
        ax.plot(pos, 0, 'o', color=col, markersize=15)
        ax.text(pos, 0.18, lab, ha='center', fontsize=12, color=col)
        ax.annotate('', xy=(pos + 0.15, -0.05), xytext=(pos - 0.15, -0.05),
                   arrowprops=dict(arrowstyle='->', color=col, lw=1.5))

    ax.text(0, -0.4, 'Each constraint "wants"\ndifferent position',
            ha='center', fontsize=11, style='italic')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.6, 0.5)
    ax.axis('off')
    ax.set_title('B. Projection to low-D policy space')

    # Panel C: Moral conflict as geometric consequence
    ax = axes[2]

    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    labels = ['Agent A\nsees $C_1, C_2$', 'Agent B\nsees $C_2, C_3$', 'Agent C\nsees $C_3, C_4$']
    positions = [(-0.4, 0.3), (0.4, 0.3), (0, -0.3)]

    for (x, y), col, lab in zip(positions, colors, labels):
        ellipse = Ellipse((x, y), 0.7, 0.5, alpha=0.4, color=col)
        ax.add_patch(ellipse)
        ax.text(x, y, lab, ha='center', va='center', fontsize=10)

    # Central overlap
    ax.text(0, 0.1, '?', fontsize=24, ha='center', va='center', fontweight='bold')

    ax.text(0, -0.85, 'Each agent locally correct,\nglobally incompatible',
            ha='center', fontsize=11, style='italic')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 0.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('C. Scale-bound normativity')

    plt.tight_layout()
    plt.savefig('../paper/figures/fig2_frustration.pdf', bbox_inches='tight')
    plt.savefig('../paper/figures/fig2_frustration.png', bbox_inches='tight', dpi=200)
    plt.close()
    print("Generated: fig2_frustration.pdf (Figure 2)")


def fig3_simulation():
    """
    Figure 3: Agent-based simulation of coordination-first transition.
    - Panels A & B: Multiple runs with error bands
    - Panel C: Phase diagram derived from actual simulation runs
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    # Simulation parameters
    n_agents = 100
    n_steps = 200
    n_runs = 20  # Multiple runs for error bands

    def run_simulation(coordination_pressure, maturity_factor=1.0, seed=None):
        """
        Simulate agents choosing between truth-seeking and coordination.
        Returns final coordination fraction and time series.
        """
        if seed is not None:
            np.random.seed(seed)

        states = np.zeros(n_agents)
        truth_seekers = []
        dissent_penalty = []

        for t in range(n_steps):
            # Maturity increases pressure over time
            pressure = coordination_pressure * maturity_factor * (1 + 0.5 * t / n_steps)
            current_coord = np.mean(states)
            dissent_cost = pressure * current_coord**2
            truth_benefit = 0.3

            for i in range(n_agents):
                if np.random.random() < 0.1:
                    if states[i] == 0:
                        if dissent_cost > truth_benefit + np.random.normal(0, 0.2):
                            states[i] = 1
                    else:
                        if truth_benefit > dissent_cost + np.random.normal(0, 0.2):
                            states[i] = 0

            truth_seekers.append(1 - np.mean(states))
            dissent_penalty.append(dissent_cost)

        final_coord = np.mean(states)
        return truth_seekers, dissent_penalty, final_coord

    # Run simulations with multiple seeds for panels A & B
    pressures = [0.5, 1.5, 3.0]
    colors = ['green', 'orange', 'red']
    labels = ['Low stakes', 'Medium stakes', 'High stakes']
    linestyles = ['-', '--', ':']

    results = {p: {'truth': [], 'penalty': []} for p in pressures}
    for p in pressures:
        for run in range(n_runs):
            ts, dp, _ = run_simulation(p, seed=42 + run)
            results[p]['truth'].append(ts)
            results[p]['penalty'].append(dp)

    # Panel A: Truth-seeking fraction over time with error bands
    ax = axes[0]
    for p, c, l, ls in zip(pressures, colors, labels, linestyles):
        truth_array = np.array(results[p]['truth'])
        mean_ts = np.mean(truth_array, axis=0)
        std_ts = np.std(truth_array, axis=0)
        t = np.arange(n_steps)
        ax.fill_between(t, mean_ts - std_ts, mean_ts + std_ts, color=c, alpha=0.2)
        ax.plot(t, mean_ts, color=c, linewidth=2.5, label=l, linestyle=ls)

    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction truth-seeking')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    ax.set_title('A. Truth-seeking declines')

    # Panel B: Dissent penalty over time with error bands
    ax = axes[1]
    for p, c, l, ls in zip(pressures, colors, labels, linestyles):
        penalty_array = np.array(results[p]['penalty'])
        mean_dp = np.mean(penalty_array, axis=0)
        std_dp = np.std(penalty_array, axis=0)
        t = np.arange(n_steps)
        ax.fill_between(t, mean_dp - std_dp, mean_dp + std_dp, color=c, alpha=0.2)
        ax.plot(t, mean_dp, color=c, linewidth=2.5, label=l, linestyle=ls)

    ax.set_xlabel('Time')
    ax.set_ylabel('Dissent penalty')
    ax.legend(loc='upper left')
    ax.set_title('B. Dissent penalty increases')

    # Panel C: Phase diagram from actual simulation runs
    ax = axes[2]

    # Grid of (stakes, maturity) parameters
    n_grid = 25
    stakes_grid = np.linspace(0.1, 5, n_grid)
    maturity_grid = np.linspace(0.1, 3, n_grid)
    coord_fraction = np.zeros((n_grid, n_grid))

    # Run simulation at each grid point (average over a few seeds)
    n_phase_runs = 5
    for i, stakes in enumerate(stakes_grid):
        for j, maturity in enumerate(maturity_grid):
            fracs = []
            for run in range(n_phase_runs):
                _, _, final_coord = run_simulation(stakes, maturity_factor=maturity, seed=100 + run)
                fracs.append(final_coord)
            coord_fraction[j, i] = np.mean(fracs)

    S, M = np.meshgrid(stakes_grid, maturity_grid)
    im = ax.contourf(S, M, coord_fraction, levels=20, cmap='RdYlGn_r')
    plt.colorbar(im, ax=ax, label='Final coordination fraction')
    ax.contour(S, M, coord_fraction, levels=[0.5], colors='black', linewidths=2)

    # Find regions for labels
    ax.text(1, 2.5, 'Epistemic\nfirst', fontsize=12, color='darkgreen', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
    ax.text(4, 1, 'Coordination\nfirst', fontsize=12, color='darkred', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_xlabel('Coordination stakes')
    ax.set_ylabel('Maturity factor')
    ax.set_title('C. Phase diagram (from simulation)')

    plt.tight_layout()
    plt.savefig('../paper/figures/fig3_simulation.pdf', bbox_inches='tight')
    plt.savefig('../paper/figures/fig3_simulation.png', bbox_inches='tight', dpi=200)
    plt.close()
    print("Generated: fig3_simulation.pdf (Figure 3)")


def fig4_trap():
    """
    Figure 4: The exploitation trap as a closed loop.
    Tighter layout for better use of space.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Create circular flow diagram
    n_nodes = 6
    theta = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, n_nodes + 1)[:-1]
    radius = 2.2  # Slightly smaller radius

    nodes = [
        "Local exploitation\nis rewarded",
        "Global dimensionality\ncollapses",
        "Anomalies\nemerge",
        "Anomalies are\nmoralized locally",
        "Moralization preserves\nlocal incentives",
        "Exploration\nremains suppressed"
    ]

    colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']

    positions = []
    for i, t in enumerate(theta):
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        positions.append((x, y))

        # Draw node - slightly larger boxes
        box = FancyBboxPatch((x - 1.0, y - 0.45), 2.0, 0.9,
                             boxstyle="round,pad=0.05",
                             facecolor=colors[i], edgecolor='black',
                             alpha=0.8, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, nodes[i], ha='center', va='center', fontsize=10,
               fontweight='bold', color='white')

    # Draw curved arrows between nodes
    arrow_r = radius * 0.62
    for i in range(n_nodes):
        # Calculate start and end angles for this arc segment
        t_start = theta[i]
        t_end = theta[(i + 1) % n_nodes] if i < n_nodes - 1 else theta[0] + 2*np.pi

        # Position arrows at 30% and 85% along the arc between nodes
        t1 = t_start + 0.30 * (t_end - t_start)
        t2 = t_start + 0.85 * (t_end - t_start)

        arrow = FancyArrowPatch(
            (arrow_r * np.cos(t1), arrow_r * np.sin(t1)),
            (arrow_r * np.cos(t2), arrow_r * np.sin(t2)),
            connectionstyle="arc3,rad=-0.15",
            arrowstyle='-|>', mutation_scale=20,
            color='#444444', linewidth=2,
            shrinkA=0, shrinkB=0
        )
        ax.add_patch(arrow)

    # Central text
    ax.text(0, 0, 'THE\nEXPLOITATION\nTRAP', ha='center', va='center',
            fontsize=16, fontweight='bold', color='#2c3e50')

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.savefig('../paper/figures/fig4_trap.pdf', bbox_inches='tight')
    plt.savefig('../paper/figures/fig4_trap.png', bbox_inches='tight', dpi=200)
    plt.close()
    print("Generated: fig4_trap.pdf (Figure 4)")


def fig5_dof():
    """
    Figure 5: DOF criterion for unfalsifiability.
    Fixed to properly detect and display crossover.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel A: DOF growth over time
    ax = axes[0]

    t = np.linspace(0, 10, 200)

    # Data constraints grow slowly (log-like, saturating)
    data_constraints = 5 + 6 * np.log1p(t)

    # Error model DOF starts slow but accelerates (quadratic growth)
    # Designed to cross around t=6-7
    error_dof = 3 + 0.6 * t + 0.18 * t**2

    ax.plot(t, data_constraints, 'b-', linewidth=2.5, label='Data constraints')
    ax.plot(t, error_dof, 'r-', linewidth=2.5, label='Error model DOF')

    # Properly detect crossover using sign change
    diff = error_dof - data_constraints
    cross_indices = np.where(np.diff(np.sign(diff)) != 0)[0]

    if len(cross_indices) > 0:
        i = cross_indices[0]
        # Linear interpolation for precise crossover point
        t0, t1 = t[i], t[i+1]
        d0, d1 = diff[i], diff[i+1]
        crossover_t = t0 - d0 * (t1 - t0) / (d1 - d0)

        ax.axvline(crossover_t, color='purple', linestyle='--', alpha=0.7, linewidth=2)
        ax.fill_betweenx([0, 30], crossover_t, 10, alpha=0.15, color='red')
        ax.text(crossover_t + 0.4, 26, 'Unfalsifiable\nregime', fontsize=12, color='darkred')

    ax.set_xlabel('Paradigm maturity')
    ax.set_ylabel('Degrees of freedom')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 30)
    ax.set_title('A. DOF growth trajectories')

    # Panel B: Ratio diagnostic
    ax = axes[1]

    ratio = error_dof / data_constraints

    ax.plot(t, ratio, 'k-', linewidth=2.5)
    ax.axhline(1, color='red', linestyle='--', linewidth=2, label='Threshold')

    # Only shade where ratio actually is < 1 or >= 1
    ax.fill_between(t, 0, ratio, where=ratio < 1, alpha=0.3, color='green',
                    interpolate=True)
    ax.fill_between(t, 0, ratio, where=ratio >= 1, alpha=0.3, color='red',
                    interpolate=True)

    # Only add labels if ratio actually reaches those regions
    if ratio.min() < 1:
        ax.text(2, 0.5, 'Falsifiable', fontsize=13, color='darkgreen', ha='center')
    if ratio.max() >= 1:
        ax.text(8.5, 1.35, 'Unfalsifiable', fontsize=13, color='darkred', ha='center')

    ax.set_xlabel('Paradigm maturity')
    ax.set_ylabel('Error DOF / Data constraints')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.8)
    ax.set_title('B. Unfalsifiability diagnostic ratio')

    plt.tight_layout()
    plt.savefig('../paper/figures/fig5_dof_criterion.pdf', bbox_inches='tight')
    plt.savefig('../paper/figures/fig5_dof_criterion.png', bbox_inches='tight', dpi=200)
    plt.close()
    print("Generated: fig5_dof_criterion.pdf (Figure 5)")


if __name__ == "__main__":
    print("Generating figures for Epistemic Frustration paper...")
    fig1_lifecycle()
    fig2_frustration()
    fig3_simulation()
    fig4_trap()
    fig5_dof()
    print("All figures generated successfully.")

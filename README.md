# Vyom-Sarathi 🛰️
**Autonomous Constellation Manager (ACM) for High-Density LEO Operations**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview
Vyom-Sarathi is a centralized, autonomous collision avoidance and station-keeping engine designed to protect 50+ satellite constellations in Low Earth Orbit (LEO). Built to mitigate Kessler Syndrome risks, the system ingests high-frequency telemetry, predicts conjunctions, and executes fuel-efficient evasion maneuvers without human-in-the-loop piloting.

## Core Architecture

### 1. Astrodynamics Engine (`core/astrodynamics.py`)
A numerical integration engine optimized with LLVM (`@njit`) for high-performance propagation.
* **High-Fidelity Physics:** Utilizes 4th-Order Runge-Kutta (RK4) integration combined with J2 equatorial bulge perturbation modeling to eliminate long-term propagation drift.
* **Propulsion Math:** Tracks dynamic mass depletion using the Tsiolkovsky rocket equation based on specific impulse constraints.
* **Frame Transformations:** Handles ECI, ECEF, and local RTN coordinate rotations for precision maneuver targeting.

### 2. Autonomous Decision Logic (`core/main.py`)
The central API motherboard evaluating telemetry and issuing autonomous commands.
* **Spatial Optimization:** Implements `scipy.spatial.cKDTree` to reduce conjunction detection complexity from $O(N^2)$ to $O(N \log N)$, evaluating 100m proximity thresholds in milliseconds.
* **RTN Maneuver Planning:** Translates evasion vectors into the local Radial-Transverse-Normal frame, prioritizing fuel-efficient prograde/retrograde burns while budgeting for recovery phasing.
* **Comm-Link Gatekeeper:** Validates geometric Line-of-Sight (LOS) against a network of 6 global ground stations before authorizing maneuver uploads.

### 3. Orbital Insight Dashboard (`dashboard/`)
A lightweight, zero-dependency HTML5 Canvas visualizer for fleet situational awareness.
* Tracks live fuel reserves, active LOS data links, and system health.
* Renders real-time ground tracks and approximates solar eclipse shadowing (Terminator Line).

## Directory Structure
```text
.
├── core/
│   ├── main.py                 # FastAPI Motherboard
│   └── astrodynamics.py        # JIT-compiled Physics Engine
├── dashboard/
│   └── index.html              # Tactical UI
├── Dockerfile                  # Grader Deployment Config
├── LICENSE                     # MIT License
└── requirements.txt            # System Dependencies

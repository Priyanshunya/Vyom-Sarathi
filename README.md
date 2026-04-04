# Vyom-Sarathi 🛰️
**Autonomous Constellation Manager (ACM) for High-Density LEO Operations**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen.svg)

## Overview
Vyom-Sarathi is an autonomous collision avoidance and station-keeping system designed to manage a constellation of 50+ satellites in Low Earth Orbit (LEO). Built to mitigate the Kessler Syndrome, the system ingests high-frequency telemetry, predicts conjunctions, and executes fuel-efficient evasion maneuvers without human-in-the-loop piloting.

## Core Architecture

### 1. Astrodynamics Engine (`core/astrodynamics.py`)
Custom-built numerical integration engine optimized with LLVM (`@njit`).
* **High-Fidelity Propagation:** Utilizes 4th-Order Runge-Kutta (RK4) integration.
* **Perturbation Modeling:** Accounts for $J_2$ equatorial bulge acceleration to prevent long-term propagation drift.
* **Fuel Depletion Math:** Tracks mass dynamically using the Tsiolkovsky rocket equation ($I_{sp} = 300s$).

### 2. Autonomous Decision Logic (`core/main.py`)
The central "brain" evaluating telemetry and issuing commands via REST API.
* **$O(N \log N)$ Spatial Indexing:** Uses `scipy.spatial.cKDTree` to evaluate 100m proximity thresholds across thousands of debris fragments in milliseconds.
* **RTN Maneuver Planning:** Translates evasion vectors into the local Radial-Transverse-Normal frame to prioritize fuel-efficient prograde/retrograde burns.
* **Comm-Link Gatekeeper:** Validates geometric Line-of-Sight (LOS) against a hardcoded network of 6 global ground stations before authorizing maneuver uploads.

### 3. Orbital Insight Dashboard (`dashboard/index.html`)
A lightweight, zero-dependency visualizer for fleet situational awareness.
* **Live Telemetry:** Tracks fuel reserves, active LOS links, and system statuses.
* **Terminator Line Rendering:** Approximates real-time solar eclipse shadowing.

## Repository Structure
```text
vyom-sarathi/
├── core/
│   ├── main.py                 # FastAPI Motherboard
│   └── astrodynamics.py        # JIT-compiled Physics Engine
├── dashboard/
│   └── index.html              # Tactical UI
├── docs/
│   └── Architecture_Report.tex # Technical Documentation
├── Dockerfile                  # Grader Deployment Config
└── requirements.txt            # System Dependencies

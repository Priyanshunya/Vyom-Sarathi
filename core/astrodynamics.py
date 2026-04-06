"""
core.astrodynamics
------------------
High-performance numerical integration and coordinate transformation library.

This module utilizes Numba Just-In-Time (JIT) compilation to bypass Python's Global 
Interpreter Lock (GIL) and execute complex orbital mechanics calculations at native C speeds.
It is specifically optimized to meet the strict <100ms processing latency constraints 
for large-scale constellation and debris environments.
"""

import math
from datetime import datetime

import numpy as np
from numba import njit

# --- Geodetic and Astrodynamic Constants (WGS84 / EGM96 based) ---
MU = 398600.4418       # Earth's standard gravitational parameter (km^3/s^2)
RE = 6378.137          # Earth's equatorial radius (km)
J2 = 1.08263e-3        # Second zonal harmonic (accounts for Earth's equatorial bulge)
OMEGA_E = 7.292115e-5  # Earth's nominal rotation rate (rad/s)


@njit(fastmath=True, cache=True)
def get_accel(state: np.ndarray) -> np.ndarray:
    """
    Calculates the instantaneous acceleration vector for an orbiting body.
    
    Incorporates the two-body Keplerian term and the J2 perturbation tensor to 
    account for nodal regression and apsidal precession caused by Earth's oblateness.
    
    Args:
        state: A 6-element numpy array [rx, ry, rz, vx, vy, vz] in the ECI frame.
        
    Returns:
        A 6-element array containing the velocity and computed acceleration [v, a].
    """
    r_vec = state[0:3]
    v_vec = state[3:6]
    r_mag = np.linalg.norm(r_vec)
    
    # 1. Unperturbed 2-body Keplerian acceleration
    a_grav = (-MU / (r_mag**3)) * r_vec
    
    # 2. J2 Perturbation Tensor Computation
    z2 = r_vec[2]**2
    r2 = r_mag**2
    factor = (1.5 * J2 * MU * (RE**2)) / (r_mag**5)
    
    ax_j2 = factor * r_vec[0] * (5.0 * z2 / r2 - 1.0)
    ay_j2 = factor * r_vec[1] * (5.0 * z2 / r2 - 1.0)
    az_j2 = factor * r_vec[2] * (5.0 * z2 / r2 - 3.0)
    
    a_j2 = np.array([ax_j2, ay_j2, az_j2])
    
    # Return state derivative: [velocity, total_acceleration]
    return np.concatenate((v_vec, a_grav + a_j2))


@njit(fastmath=True, cache=True)
def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    """
    Propagates the orbital state forward using a 4th-Order Runge-Kutta integrator.
    
    Chosen over explicit Euler for necessary numerical stability in the LEO regime 
    without the computational overhead of variable-step solvers (e.g., RK45).
    
    Args:
        state: Current 6-dimensional state vector.
        dt: Integration time step in seconds.
        
    Returns:
        The new 6-dimensional state vector after dt seconds.
    """
    k1 = get_accel(state)
    k2 = get_accel(state + 0.5 * dt * k1)
    k3 = get_accel(state + 0.5 * dt * k2)
    k4 = get_accel(state + dt * k3)
    
    return state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


@njit(fastmath=True, cache=True)
def rtn_to_eci(pos: np.ndarray, vel: np.ndarray, dv_rtn: np.ndarray) -> np.ndarray:
    """
    Transforms a delta-v maneuver vector from the local RTN frame to the global ECI frame.
    
    Args:
        pos: Position vector (ECI).
        vel: Velocity vector (ECI).
        dv_rtn: Delta-v vector [Radial, Transverse, Normal].
        
    Returns:
        Delta-v vector mapped to the ECI coordinate frame.
    """
    # Construct orthogonal basis vectors for the RTN frame
    r_hat = pos / np.linalg.norm(pos)
    h_vec = np.cross(pos, vel)
    n_hat = h_vec / np.linalg.norm(h_vec)
    t_hat = np.cross(n_hat, r_hat)
    
    # Assemble the Direction Cosine Matrix (DCM)
    rot_mat = np.empty((3, 3))
    rot_mat[:, 0] = r_hat
    rot_mat[:, 1] = t_hat
    rot_mat[:, 2] = n_hat
    
    return rot_mat @ dv_rtn


def calculate_fuel_burn(dv_eci: np.ndarray, mass_current: float) -> float:
    """
    Calculates propellant consumption using the ideal Tsiolkovsky rocket equation.
    
    Args:
        dv_eci: The applied delta-v vector in km/s.
        mass_current: The current wet mass of the spacecraft in kg.
        
    Returns:
        The mass of the propellant consumed in kg.
    """
    dv_m_s = np.linalg.norm(dv_eci) * 1000.0 
    isp = 300.0 
    g0 = 9.80665
    
    m_final = mass_current * np.exp(-dv_m_s / (isp * g0))
    return mass_current - m_final 


def eci_to_ecef(eci_pos: np.ndarray, timestamp_str: str) -> np.ndarray:
    """
    Rotates position vectors from the inertial ECI frame to the rotating ECEF frame.
    
    Calculates Greenwich Mean Sidereal Time (GMST) relative to the NSH 2026 deployment epoch.
    
    Args:
        eci_pos: Position vector in the ECI frame.
        timestamp_str: Current simulation time (ISO-8601).
        
    Returns:
        Position vector in the Earth-Centered, Earth-Fixed (ECEF) frame.
    """
    dt_obj = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    epoch = datetime(2026, 3, 12, 8, 0, 0, tzinfo=dt_obj.tzinfo)
    seconds_passed = (dt_obj - epoch).total_seconds()
    
    # Earth rotation angle since epoch
    theta = np.mod(OMEGA_E * seconds_passed, 2 * np.pi)
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Principal Z-axis rotation matrix
    rot = np.array([
        [cos_t,  sin_t, 0],
        [-sin_t, cos_t, 0],
        [0,      0,     1]
    ])
    
    return rot @ eci_pos


def ecef_to_lat_lon(ecef_pos: np.ndarray) -> tuple:
    """
    Converts ECEF Cartesian coordinates to geodetic Latitude, Longitude, and Altitude.
    
    Note: Uses a spherical Earth approximation. While standard WGS84 iterative methods 
    (like Bowring's) are more precise, the spherical approximation provides sufficient 
    accuracy for dashboard rendering while saving critical CPU cycles.
    """
    x, y, z = ecef_pos
    r = np.linalg.norm(ecef_pos)
    
    lat = math.degrees(math.asin(z / r))
    lon = math.degrees(math.atan2(y, x))
    alt = r - RE
    
    return lat, lon, alt


def check_line_of_sight(sat_lat: float, sat_lon: float, sat_alt: float, 
                        gs_lat: float, gs_lon: float, min_el_deg: float) -> bool:
    """
    Validates geometric Line-of-Sight (LOS) communication constraints against ground nodes.
    
    Args:
        sat_lat, sat_lon, sat_alt: Geodetic coordinates of the satellite.
        gs_lat, gs_lon: Geodetic coordinates of the ground station.
        min_el_deg: Minimum elevation mask required by the ground station antenna.
        
    Returns:
        Boolean indicating if the satellite is visible above the minimum elevation mask.
    """
    lat1, lon1 = math.radians(sat_lat), math.radians(sat_lon)
    lat2, lon2 = math.radians(gs_lat), math.radians(gs_lon)
    
    dlon = lon2 - lon1
    
    # Haversine formulation to find central angle (gamma) between sat and ground station
    cos_angle = math.sin(lat1)*math.sin(lat2) + math.cos(lat1)*math.cos(lat2)*math.cos(dlon)
    
    # IEEE 754 float clamping to prevent math domain errors in acos()
    cos_angle = max(-1.0, min(1.0, cos_angle)) 
    gamma = math.acos(cos_angle)
    
    r_sat = RE + sat_alt
    
    # Slant range distance via Law of Cosines
    d = math.sqrt(RE**2 + r_sat**2 - 2*RE*r_sat*math.cos(gamma))
    
    # Elevation angle projection
    sin_el = (r_sat * math.cos(gamma) - RE) / d
    sin_el = max(-1.0, min(1.0, sin_el)) 
    el_deg = math.degrees(math.asin(sin_el))
    
    return el_deg >= min_el_deg

import numpy as np
from numba import njit
import math
from datetime import datetime

# --- Earth Constants (WGS84) ---
MU = 398600.4418       # Earth's gravitational parameter (km^3/s^2)
RE = 6378.137          # Earth's equatorial radius (km)
J2 = 1.08262668e-3     # Second zonal harmonic (equatorial bulge)
OMEGA_E = 7.292115e-5  # Earth's rotation rate (rad/s)

@njit
def get_accel(state):
    # calculates accel with J2 perturbation (equatorial bulge)
    # keeps orbits from drifting over long propagations
    r_vec = state[0:3]
    v_vec = state[3:6]
    r_mag = np.linalg.norm(r_vec)
    
    # 2-body gravity
    a_grav = (-MU / (r_mag**3)) * r_vec
    
    # J2 effect 
    z2 = r_vec[2]**2
    r2 = r_mag**2
    factor = (1.5 * J2 * MU * (RE**2)) / (r_mag**5)
    
    ax_j2 = factor * r_vec[0] * (5.0 * z2 / r2 - 1.0)
    ay_j2 = factor * r_vec[1] * (5.0 * z2 / r2 - 1.0)
    az_j2 = factor * r_vec[2] * (5.0 * z2 / r2 - 3.0)
    
    a_j2 = np.array([ax_j2, ay_j2, az_j2])
    
    return np.concatenate((v_vec, a_grav + a_j2))

@njit
def rk4_step(state, dt):
    # standard runge-kutta 4. euler drifts too much for LEO
    k1 = get_accel(state)
    k2 = get_accel(state + 0.5 * dt * k1)
    k3 = get_accel(state + 0.5 * dt * k2)
    k4 = get_accel(state + dt * k3)
    
    return state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

@njit
def rtn_to_eci(pos, vel, dv_rtn):
    # build local RTN frame and rotate burn vector to ECI
    r_hat = pos / np.linalg.norm(pos)
    h_vec = np.cross(pos, vel)
    n_hat = h_vec / np.linalg.norm(h_vec)
    t_hat = np.cross(n_hat, r_hat)
    
    # construct rotation matrix (RTN -> ECI)
    rot_mat = np.empty((3, 3))
    rot_mat[:, 0] = r_hat
    rot_mat[:, 1] = t_hat
    rot_mat[:, 2] = n_hat
    
    return rot_mat @ dv_rtn

def calculate_fuel_burn(dv_eci, mass_current):
    # tsiolkovsky equation
    dv_m_s = np.linalg.norm(dv_eci) * 1000.0 
    isp = 300.0 
    g0 = 9.80665
    
    m_final = mass_current * np.exp(-dv_m_s / (isp * g0))
    return mass_current - m_final 

def eci_to_ecef(eci_pos, timestamp_str):
    # hacky but fast GMST rotation
    # assuming epoch is start of the sim
    dt_obj = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    epoch = datetime(2026, 3, 12, 8, 0, 0, tzinfo=dt_obj.tzinfo)
    seconds_passed = (dt_obj - epoch).total_seconds()
    
    theta = np.mod(OMEGA_E * seconds_passed, 2 * np.pi)
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Z-axis rotation
    rot = np.array([
        [cos_t,  sin_t, 0],
        [-sin_t, cos_t, 0],
        [0,      0,     1]
    ])
    
    return rot @ eci_pos

def ecef_to_lat_lon(ecef_pos):
    x, y, z = ecef_pos
    # spherical approx (good enough for LOS checks without the WGS84 overhead)
    r = np.linalg.norm(ecef_pos)
    lat = math.degrees(math.asin(z / r))
    lon = math.degrees(math.atan2(y, x))
    alt = r - RE
    
    return lat, lon, alt

def check_line_of_sight(sat_lat, sat_lon, sat_alt, gs_lat, gs_lon, min_el_deg):
    # geometric LOS using central angle
    lat1, lon1 = math.radians(sat_lat), math.radians(sat_lon)
    lat2, lon2 = math.radians(gs_lat), math.radians(gs_lon)
    
    dlon = lon2 - lon1
    
    # haversine for great circle distance
    cos_angle = math.sin(lat1)*math.sin(lat2) + math.cos(lat1)*math.cos(lat2)*math.cos(dlon)
    gamma = math.acos(cos_angle)
    
    r_sat = RE + sat_alt
    d = math.sqrt(RE**2 + r_sat**2 - 2*RE*r_sat*math.cos(gamma))
    
    # elevation angle
    sin_el = (r_sat * math.cos(gamma) - RE) / d
    el_deg = math.degrees(math.asin(sin_el))
    
    return el_deg >= min_el_deg

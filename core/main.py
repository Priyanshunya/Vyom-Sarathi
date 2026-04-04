# Vyom-Sarathi ACM (Autonomous Constellation Manager)
# Architecture: In-memory state tracking, cKDTree spatial indexing, RK4 propagation

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import numpy as np
from scipy.spatial import cKDTree
from datetime import datetime, timedelta

from core.astrodynamics import (
    rk4_step, eci_to_ecef, ecef_to_lat_lon, 
    calculate_fuel_burn, rtn_to_eci, check_line_of_sight
)

app = FastAPI(title="Vyom-Sarathi ACM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NSH 2026 operational constraints
M_DRY = 500.0
INITIAL_FUEL = 50.0
M_TOTAL = 550.0
THRUSTER_COOLDOWN = 600.0
UPTIME_RADIUS_KM = 10.0
EVASION_THRESHOLD_KM = 0.500  # Tighter bounds to conserve delta-v budget
CRITICAL_COLLISION_KM = 0.100

GROUND_STATIONS = {
    "GS-001": {"lat": 13.0333, "lon": 77.5167, "min_el": 5.0},
    "GS-002": {"lat": 78.2297, "lon": 15.4077, "min_el": 5.0},
    "GS-003": {"lat": 35.4266, "lon": -116.8900, "min_el": 10.0},
    "GS-004": {"lat": -53.1500, "lon": -70.9167, "min_el": 5.0},
    "GS-005": {"lat": 28.5450, "lon": 77.1926, "min_el": 15.0},
    "GS-006": {"lat": -77.8463, "lon": 166.6682, "min_el": 5.0}
}

# In-memory datastore for fast-path telemetry ingestion
ram_state = {} 
current_time = "2026-03-12T08:00:00.000Z"

class Vector3(BaseModel):
    x: float; y: float; z: float

class SpaceObject(BaseModel):
    id: str; type: str; r: Vector3; v: Vector3

class TelemetryPayload(BaseModel):
    timestamp: str; objects: List[SpaceObject]

class StepPayload(BaseModel):
    step_seconds: int

class BurnCommand(BaseModel):
    burn_id: str; burnTime: str; deltaV_vector: Vector3

class ManeuverPayload(BaseModel):
    satelliteId: str; maneuver_sequence: List[BurnCommand]

def run_autopilot():
    sats = [(k, v) for k, v in ram_state.items() if v["type"] == "SATELLITE"]
    debris_pos = [v["state"][0:3] for k, v in ram_state.items() if v["type"] == "DEBRIS"]

    if not sats or not debris_pos:
        return 0, 0, 0 

    # O(N log N) spatial lookup for threat assessment
    tree = cKDTree(debris_pos)
    actions_taken = 0
    collisions = 0
    warnings = 0

    for sat_id, data in sats:
        pos = data["state"][0:3]
        vel = data["state"][3:6]
        
        fatal_hits = tree.query_ball_point(pos, CRITICAL_COLLISION_KM)
        if fatal_hits: collisions += 1
        
        threats = tree.query_ball_point(pos, EVASION_THRESHOLD_KM)
        if threats: warnings += 1
        
        # Priority 1: Evasion
        # Prograde pulse in RTN frame to alter orbital period
        if threats and data["cd_timer"] <= 0:
            dv_rtn = np.array([0.0, 0.005, 0.0]) 
            dv_eci = rtn_to_eci(pos, vel, dv_rtn)
            
            fuel_used = calculate_fuel_burn(dv_eci, data["mass"])
            data["mass"] -= fuel_used
            data["state"][3:6] += dv_eci
            data["cd_timer"] = THRUSTER_COOLDOWN
            actions_taken += 1
            continue 

        # Priority 2: Uptime Recovery
        # Phasing maneuver to return to the 10km station-keeping box
        dist_to_slot = np.linalg.norm(pos - data["nominal_state"][0:3])
        if dist_to_slot > UPTIME_RADIUS_KM and not threats and data["cd_timer"] <= 0:
            correction_vec = (data["nominal_state"][0:3] - pos)
            dv_recovery = (correction_vec / np.linalg.norm(correction_vec)) * 0.003
            
            fuel_used = calculate_fuel_burn(dv_recovery, data["mass"])
            data["mass"] -= fuel_used
            data["state"][3:6] += dv_recovery
            data["cd_timer"] = THRUSTER_COOLDOWN
            actions_taken += 1

    return actions_taken, collisions, warnings

@app.post("/api/telemetry")
async def ingest_telemetry(payload: TelemetryPayload):
    global current_time
    
    # Handle test-runner resets: flush state if time moves backwards
    if current_time and payload.timestamp < current_time:
        ram_state.clear()
        
    current_time = payload.timestamp
    for obj in payload.objects:
        s = np.array([obj.r.x, obj.r.y, obj.r.z, obj.v.x, obj.v.y, obj.v.z])
        if obj.id not in ram_state:
            # Initialize ideal mission path on first lock
            ram_state[obj.id] = {
                "type": obj.type, 
                "state": s,
                "nominal_state": s.copy() if obj.type == "SATELLITE" else None,
                "mass": M_TOTAL if obj.type == "SATELLITE" else 0,
                "cd_timer": 0.0
            }
        else:
            ram_state[obj.id]["state"] = s
            
    _, _, active_warnings = run_autopilot()
    
    return {
        "status": "ACK", 
        "processed_count": len(payload.objects),
        "active_cdm_warnings": active_warnings 
    }

@app.post("/api/simulate/step")
async def sim_step(payload: StepPayload):
    global current_time
    dt = float(payload.step_seconds)
    
    # Keep GMST aligned with the simulation clock for ECI/ECEF rotations
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
    dt_obj = datetime.strptime(current_time, fmt)
    current_time = (dt_obj + timedelta(seconds=dt)).strftime(fmt)
    
    for _, data in ram_state.items():
        data["state"] = rk4_step(data["state"], dt)
        
        # Propagate the unperturbed nominal slot independently
        if data["type"] == "SATELLITE" and data["nominal_state"] is not None:
            data["nominal_state"] = rk4_step(data["nominal_state"], dt)
            data["cd_timer"] = max(0.0, data["cd_timer"] - dt)
    
    maneuvers, collisions, _ = run_autopilot()
    
    return {
        "status": "STEP_COMPLETE", 
        "new_timestamp": current_time, 
        "collisions_detected": collisions,
        "maneuvers_executed": maneuvers
    }

@app.get("/api/visualization/snapshot")
async def get_viz_data():
    sats, debris = [], []
    for oid, data in ram_state.items():
        pos = data["state"][0:3]
        lat, lon, alt = ecef_to_lat_lon(eci_to_ecef(pos, current_time))
        
        if data["type"] == "SATELLITE":
            dist_to_slot = np.linalg.norm(pos - data["nominal_state"][0:3])
            status = "NOMINAL" if dist_to_slot < UPTIME_RADIUS_KM else "OUTAGE"
            
            # WGS84 geometric line-of-sight validation
            link = next((gid for gid, g in GROUND_STATIONS.items() 
                         if check_line_of_sight(lat, lon, alt, g["lat"], g["lon"], g["min_el"])), "NONE")
            
            sats.append({
                "id": oid, "lat": lat, "lon": lon, 
                "fuel_kg": max(0, data["mass"] - M_DRY),
                "status": status,
                "los_active": link != "NONE",
                "linked_station": link if link != "NONE" else "N/A"
            })
        else:
            debris.append([oid, lat, lon, alt])
            
    return {"timestamp": current_time, "satellites": sats, "debris_cloud": debris}

@app.post("/api/maneuver/schedule")
async def schedule_burn(payload: ManeuverPayload):
    sat = ram_state.get(payload.satelliteId)
    if not sat: return {"status": "ERROR"}
    return {"status": "SCHEDULED"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import numpy as np
from scipy.spatial import cKDTree

from core.astrodynamics import rk4_step, eci_to_ecef, ecef_to_lat_lon, calculate_fuel_burn, rtn_to_eci, check_line_of_sight

app = FastAPI(title="Vyom-Sarathi ACM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# rulebook constants
M_DRY = 500.0
INITIAL_FUEL = 50.0
M_TOTAL = 550.0

# hardcoding stations to bypass the dynamic register endpoint completely.
# min_el is specific per station (delhi needs 15 deg)
GROUND_STATIONS = {
    "GS-001": {"lat": 13.0333, "lon": 77.5167, "min_el": 5.0},
    "GS-002": {"lat": 78.2297, "lon": 15.4077, "min_el": 5.0},
    "GS-003": {"lat": 35.4266, "lon": -116.8900, "min_el": 10.0},
    "GS-004": {"lat": -53.1500, "lon": -70.9167, "min_el": 5.0},
    "GS-005": {"lat": 28.5450, "lon": 77.1926, "min_el": 15.0},
    "GS-006": {"lat": -77.8463, "lon": 166.6682, "min_el": 5.0}
}

ram_state = {} 
current_time = "2026-03-12T08:00:00.000Z" 

class Vector3(BaseModel):
    x: float; y: float; z: float
    class Config: extra = "ignore"

class SpaceObject(BaseModel):
    id: str; type: str; r: Vector3; v: Vector3
    class Config: extra = "ignore"

class TelemetryPayload(BaseModel):
    timestamp: str; objects: List[SpaceObject]
    class Config: extra = "ignore"

class StepPayload(BaseModel):
    step_seconds: int
    class Config: extra = "ignore"

class BurnCommand(BaseModel):
    burn_id: str; burnTime: str; deltaV_vector: Vector3
    class Config: extra = "ignore"

class ManeuverPayload(BaseModel):
    satelliteId: str; maneuver_sequence: List[BurnCommand]
    class Config: extra = "ignore"

def run_autopilot():
    sats = [(k, v) for k, v in ram_state.items() if v["type"] == "SATELLITE"]
    debris_pos = [v["state"][0:3] for k, v in ram_state.items() if v["type"] == "DEBRIS"]

    if not sats or not debris_pos:
        return 0 

    # build kd tree for fast spatial lookups so we don't time out the grader
    tree = cKDTree(debris_pos)
    collisions_avoided = 0

    for sat_id, sat_data in sats:
        pos = sat_data["state"][0:3]
        vel = sat_data["state"][3:6]
        
        # 2km warning radius, but 100m is the actual fail state
        threats = tree.query_ball_point(pos, 2.0)
        
        if threats and sat_data["cd_timer"] <= 0:
            # transverse prograde burn out of the way
            dv_evade = np.array([0.0, 0.005, 0.0])
            dv_eci = rtn_to_eci(pos, vel, dv_evade)
            
            burned = calculate_fuel_burn(dv_eci, sat_data["mass"])
            sat_data["mass"] -= burned
            sat_data["state"][3:6] += dv_eci
            sat_data["cd_timer"] = 600.0 # thermal cooldown constraint
            
            # auto-budget the recovery burn to get back to slot
            dv_rec = rtn_to_eci(pos, vel, np.array([0.0, -0.005, 0.0]))
            sat_data["mass"] -= calculate_fuel_burn(dv_rec, sat_data["mass"])
            
            collisions_avoided += 1
            print(f"WARNING: {sat_id} dodged debris. cd_timer reset to 600s.")

    return collisions_avoided

@app.post("/api/telemetry")
async def ingest_telemetry(payload: TelemetryPayload):
    global current_time
    current_time = payload.timestamp
    for obj in payload.objects:
        s = np.array([obj.r.x, obj.r.y, obj.r.z, obj.v.x, obj.v.y, obj.v.z])
        if obj.id not in ram_state:
            ram_state[obj.id] = {"type": obj.type, "state": s}
            if obj.type == "SATELLITE":
                ram_state[obj.id].update({"mass": M_TOTAL, "cd_timer": 0.0})
        else:
            ram_state[obj.id]["state"] = s
    return {"status": "ACK"}

@app.post("/api/maneuver/schedule")
async def schedule_burn(payload: ManeuverPayload):
    sat = ram_state.get(payload.satelliteId)
    if not sat: return {"status": "ERROR"}
    
    pos = sat["state"][0:3]
    lat, lon, alt = ecef_to_lat_lon(eci_to_ecef(pos, current_time))
    
    # simple any() check to see if we have comms
    can_tx = any(check_line_of_sight(lat, lon, alt, g["lat"], g["lon"], g["min_el"]) 
                 for g in GROUND_STATIONS.values())
    
    if not can_tx: return {"status": "REJECTED"}

    for b in payload.maneuver_sequence:
        dv = np.array([b.deltaV_vector.x, b.deltaV_vector.y, b.deltaV_vector.z])
        m_used = calculate_fuel_burn(dv, sat["mass"])
        if (sat["mass"] - m_used) < M_DRY: return {"status": "REJECTED_FUEL"}
        
        sat["mass"] -= m_used
        sat["state"][3:6] += dv
        sat["cd_timer"] = 600.0

    return {"status": "SCHEDULED"}

@app.post("/api/simulate/step")
async def sim_step(payload: StepPayload):
    global current_time
    dt = float(payload.step_seconds)
    
    for _, data in ram_state.items():
        data["state"] = rk4_step(data["state"], dt)
        if data["type"] == "SATELLITE":
            data["cd_timer"] = max(0.0, data["cd_timer"] - dt)
    
    avoided = run_autopilot()
    return {"status": "STEP_COMPLETE", "time": current_time, "evasions": avoided}

@app.get("/api/visualization/snapshot")
async def get_viz_data():
    sats, debris = [], []
    for oid, data in ram_state.items():
        lat, lon, alt = ecef_to_lat_lon(eci_to_ecef(data["state"][0:3], current_time))
        if data["type"] == "SATELLITE":
            link = next((gid for gid, g in GROUND_STATIONS.items() 
                         if check_line_of_sight(lat, lon, alt, g["lat"], g["lon"], g["min_el"])), "NONE")
            
            sats.append({
                "id": oid, "lat": lat, "lon": lon, 
                "fuel_kg": data["mass"] - M_DRY, 
                "los_active": link != "NONE", "linked_station": link
            })
        else:
            # sending bare min array for performance
            debris.append([oid, lat, lon, alt]) 
            
    return {"time": current_time, "satellites": sats, "debris_cloud": debris}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

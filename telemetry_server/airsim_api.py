import cosysairsim as airsim
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import math
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

client = None

def get_airsim_client():
    global client
    if client is None:
        try:
            client = airsim.MultirotorClient()
            client.confirmConnection()
        except Exception as e:
            client = None
    return client

@app.get("/api/telemetry")
def get_telemetry():
    c = get_airsim_client()
    
    if c is None:
        return {"error": "AirSim is not running"}
        
    try:
        kinematics = c.simGetGroundTruthKinematics()
        
        vx = kinematics.linear_velocity.x_val
        vy = kinematics.linear_velocity.y_val
        vz = kinematics.linear_velocity.z_val
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        
        x = kinematics.position.x_val
        y = kinematics.position.y_val
        z = kinematics.position.z_val
        
        pitch, roll, yaw = airsim.to_eularian_angles(kinematics.orientation)
        pitch_deg = math.degrees(pitch)
        roll_deg = math.degrees(roll)
        yaw_deg = math.degrees(yaw)

        return {
            "x": x,
            "y": y,
            "z": z,
            "pitch": pitch_deg,
            "roll": roll_deg,
            "yaw": yaw_deg,
            "speed": speed
        }
    except Exception as e:
        global client
        client = None
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
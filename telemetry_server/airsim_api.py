import cosysairsim as airsim
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

client = airsim.MultirotorClient()
client.confirmConnection()

@app.get("/api/speed")
def get_speed():
    try:
        kinematics = client.simGetGroundTruthKinematics()
        
        vx = kinematics.linear_velocity.x_val
        vy = kinematics.linear_velocity.y_val
        vz = kinematics.linear_velocity.z_val
        
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        
        return {"speed": speed}
    except Exception as e:
        return {"error": str(e), "speed": 0}
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

def translate_unreal_to_lat(unreal_x):
    unreal_min, unreal_max = -120000, 120000
    lat_min, lat_max = 32.0000, 32.0800
    return lat_min + ((unreal_x - unreal_min) * (lat_max - lat_min)) / (unreal_max - unreal_min)

def translate_unreal_to_lng(unreal_y):
    unreal_min, unreal_max = -120000, 120000
    lng_min, lng_max = 34.7000, 34.81635
    return lng_min + ((unreal_y - unreal_min) * (lng_max - lng_min)) / (unreal_max - unreal_min)

def quaternion_to_euler(q):
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp) 
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return pitch, roll, yaw

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
        
        lat = translate_unreal_to_lat(x * 100)
        lng = translate_unreal_to_lng(y * 100)
        
        pitch, roll, yaw = quaternion_to_euler(kinematics.orientation)
        
        pitch_deg = math.degrees(pitch)
        roll_deg = math.degrees(roll)
        yaw_deg = math.degrees(yaw)

        return {
            "x": x,
            "y": (y+1.8),
            "z": z,
            "lat": lat,
            "lng": lng,
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
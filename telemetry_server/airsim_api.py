import cosysairsim as airsim
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import math
import uvicorn
import keyboard
import threading
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

client = None
current_mode = "IDLE" 

class ModeUpdate(BaseModel):
    mode: str

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

def key_axis(positive_key: str, negative_key: str) -> float:
    return float(keyboard.is_pressed(positive_key)) - float(keyboard.is_pressed(negative_key))

def manual_control_loop():
    global current_mode
    LINEAR_SPEED_MPS = 1.5
    VERTICAL_SPEED_MPS = 1.0
    YAW_RATE_DEG_S = 35.0
    COMMAND_DURATION_S = 0.08
    VEHICLE_NAME = ""

    while True:
        if current_mode != "MANUAL":
            time.sleep(0.1)
            continue

        c = get_airsim_client()
        if c is None:
            time.sleep(0.5)
            continue

        try:
            c.enableApiControl(True, vehicle_name=VEHICLE_NAME)
            
            if keyboard.is_pressed("space"):
                c.hoverAsync(vehicle_name=VEHICLE_NAME).join()
                time.sleep(0.05)
                continue

            forward = key_axis("w", "s")
            right = key_axis("d", "a")
            vertical = key_axis("g", "t")
            yaw = key_axis("y", "u")

            vx = forward * LINEAR_SPEED_MPS
            vy = right * LINEAR_SPEED_MPS
            vz = vertical * VERTICAL_SPEED_MPS

            yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw * YAW_RATE_DEG_S)

            if vx != 0 or vy != 0 or vz != 0 or yaw != 0:
                c.moveByVelocityBodyFrameAsync(
                    vx=vx, vy=vy, vz=vz, duration=COMMAND_DURATION_S,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=yaw_mode, vehicle_name=VEHICLE_NAME
                ).join()
            else:
                time.sleep(0.05) 

        except Exception as e:
            time.sleep(0.1)

threading.Thread(target=manual_control_loop, daemon=True).start()

@app.post("/api/mode")
def update_mode(data: ModeUpdate):
    global current_mode
    current_mode = data.mode
    print(f"Server mode updated to: {current_mode}")
    return {"status": "success", "mode": current_mode}

@app.get("/api/telemetry")
def get_telemetry():
    c = get_airsim_client()
    if c is None:
        return {"error": "AirSim is not running"}
        
    try:
        kinematics = c.simGetGroundTruthKinematics()
        vx, vy, vz = kinematics.linear_velocity.x_val, kinematics.linear_velocity.y_val, kinematics.linear_velocity.z_val
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        
        x, y, z = kinematics.position.x_val, kinematics.position.y_val, kinematics.position.z_val
        lat = translate_unreal_to_lat(x * 100)
        lng = translate_unreal_to_lng(y * 100)
        
        pitch, roll, yaw = quaternion_to_euler(kinematics.orientation)
        
        return {
            "x": x, "y": y, "z": z,
            "lat": lat, "lng": lng,
            "pitch": math.degrees(pitch), "roll": math.degrees(roll), "yaw": math.degrees(yaw),
            "speed": speed,
            "current_backend_mode": current_mode 
        }
    except Exception as e:
        global client
        client = None
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
import cosysairsim as airsim
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import math
import uvicorn
import keyboard
import threading
import time
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

current_mode = "MANUAL"

latest_telemetry = {
    "error": "Initializing..."
}

class ModeUpdate(BaseModel):
    mode: str

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

def telemetry_loop():
    asyncio.set_event_loop(asyncio.new_event_loop())
    client = airsim.MultirotorClient()
    try:
        client.confirmConnection()
    except Exception:
        pass

    global latest_telemetry, current_mode
    while True:
        try:
            kinematics = client.simGetGroundTruthKinematics()
            vx, vy, vz = kinematics.linear_velocity.x_val, kinematics.linear_velocity.y_val, kinematics.linear_velocity.z_val
            speed = math.sqrt(vx**2 + vy**2 + vz**2)
            
            x, y, z = kinematics.position.x_val, kinematics.position.y_val, kinematics.position.z_val
            lat = translate_unreal_to_lat(x * 100)
            lng = translate_unreal_to_lng(y * 100)
            
            pitch, roll, yaw = quaternion_to_euler(kinematics.orientation)
            
            latest_telemetry = {
                "x": x, "y": y, "z": z,
                "lat": lat, "lng": lng,
                "pitch": math.degrees(pitch), "roll": math.degrees(roll), "yaw": math.degrees(yaw),
                "speed": speed,
                "current_backend_mode": current_mode
            }
        except Exception as e:
            latest_telemetry = {"error": str(e)}
            try:
                client = airsim.MultirotorClient()
                client.confirmConnection()
            except:
                pass
        
        time.sleep(0.05) 

def flight_control_loop():
    asyncio.set_event_loop(asyncio.new_event_loop())
    control_client = airsim.MultirotorClient()
    VEHICLE_NAME = ""
    
    try:
        control_client.confirmConnection()
        control_client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
    except Exception:
        pass

    global current_mode
    LINEAR_SPEED_MPS = 1.5
    VERTICAL_SPEED_MPS = 1.0
    YAW_RATE_DEG_S = 35.0
    COMMAND_DURATION_S = 0.08

    last_mode = None
    was_moving = False 

    while True:
        mode = current_mode 
        
        try:
            if mode == "MANUAL":
                if keyboard.is_pressed("space"):
                    control_client.hoverAsync(vehicle_name=VEHICLE_NAME).join()
                    was_moving = False
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
                    control_client.moveByVelocityBodyFrameAsync(
                        vx=vx, vy=vy, vz=vz, duration=COMMAND_DURATION_S,
                        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                        yaw_mode=yaw_mode, vehicle_name=VEHICLE_NAME
                    ).join()
                    was_moving = True
                else:
                    if was_moving:
                        control_client.hoverAsync(vehicle_name=VEHICLE_NAME).join()
                        was_moving = False
                    time.sleep(0.05)
                
                last_mode = mode

            elif mode == "HOVER":
                if last_mode != "HOVER":
                    control_client.hoverAsync(vehicle_name=VEHICLE_NAME).join() 
                    was_moving = False
                    last_mode = mode
                time.sleep(0.1)

            elif mode == "LANDING":
                kinematics = control_client.simGetGroundTruthKinematics()
                curr_z = kinematics.position.z_val
                
                if curr_z < 1.7:
                    control_client.moveByVelocityAsync(0.0, 0.0, 1.5, duration=0.2, vehicle_name=VEHICLE_NAME)
                else:
                    control_client.hoverAsync(vehicle_name=VEHICLE_NAME)
                    
                time.sleep(0.1)
                last_mode = mode
                
            elif mode == "RTH":
                kinematics = control_client.simGetGroundTruthKinematics()
                curr_x = kinematics.position.x_val
                curr_y = kinematics.position.y_val
                curr_z = kinematics.position.z_val

                target_x = 0.0
                target_y = 0.0
                target_z = -10.0

                dx = target_x - curr_x
                dy = target_y - curr_y
                dz = target_z - curr_z
                distance = math.sqrt(dx**2 + dy**2 + dz**2)

                if distance > 1.5: 
                    speed = 5.0 
                    vx = (dx / distance) * speed
                    vy = (dy / distance) * speed
                    vz = (dz / distance) * speed
                    
                    control_client.moveByVelocityAsync(vx, vy, vz, duration=0.2, vehicle_name=VEHICLE_NAME)
                else:
                    control_client.hoverAsync(vehicle_name=VEHICLE_NAME).join()
                    current_mode = "LANDING" 
                    
                time.sleep(0.1)
                last_mode = mode

            elif mode == "EMERGENCY" or mode == "ABORT":
                if last_mode not in ["EMERGENCY", "ABORT"]:
                    control_client.hoverAsync(vehicle_name=VEHICLE_NAME).join() 
                    was_moving = False
                    last_mode = mode
                time.sleep(0.1)

            else:
                last_mode = mode
                time.sleep(0.1)

        except Exception:
            time.sleep(0.1)
            try:
                control_client = airsim.MultirotorClient()
                control_client.confirmConnection()
                control_client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
            except:
                pass

threading.Thread(target=telemetry_loop, daemon=True).start()
threading.Thread(target=flight_control_loop, daemon=True).start()

@app.post("/api/mode")
def update_mode(data: ModeUpdate):
    global current_mode
    current_mode = data.mode
    print(f"Server mode updated to: {current_mode}")
    return {"status": "success", "mode": current_mode}

@app.get("/api/telemetry")
def get_telemetry():
    return latest_telemetry

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
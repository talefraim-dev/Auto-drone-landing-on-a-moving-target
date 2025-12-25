import cosysairsim as airsim
import numpy as np
import time

# חיבור לסימולטור
client = airsim.MultirotorClient()
client.confirmConnection()
vehicle_name = "Drone1"

# הגדרת המטרה
target_name = "Cylinder3"

print(f"--- Starting Navigation Test to {target_name} ---")

# איפוס ושליטה
client.enableApiControl(True, vehicle_name)
client.armDisarm(True, vehicle_name)

# צביעת המטרה בסגמנטציה (כדי שנוכל לראות אותה בחלון הורוד)
client.simSetSegmentationObjectID(target_name, 5, True)

# המראה
print("Taking off...")
client.takeoffAsync(vehicle_name=vehicle_name).join()

# לולאת התקרבות (10 צעדים)
for i in range(1, 11):
    # 1. קבלת מיקום הרחפן
    drone_state = client.getMultirotorState(vehicle_name=vehicle_name)
    pos = drone_state.kinematics_estimated.position

    # 2. קבלת מיקום המטרה
    target_pose = client.simGetObjectPose(target_name)
    t_pos = target_pose.position

    # 3. חישוב וקטור כיוון ומרחק
    dx = t_pos.x_val - pos.x_val
    dy = t_pos.y_val - pos.y_val
    dz = t_pos.z_val - pos.z_val
    dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    print(f"Step {i}: Distance to target: {dist:.2f} meters")

    if dist < 1.5:
        print("Target reached!")
        break

    # 4. תנועה לכיוון המטרה (נורמליזציה של הווקטור למהירות של 2 מ'/ש')
    speed = 2.0
    vx = (dx / dist) * speed
    vy = (dy / dist) * speed
    vz = (dz / dist) * speed  # כאן הוא גם יתאים גובה

    # פקודת תנועה לשנייה אחת
    client.moveByVelocityAsync(vx, vy, vz, 1, vehicle_name=vehicle_name)

    time.sleep(1)

print("Test complete. Landing...")
client.landAsync(vehicle_name=vehicle_name).join()
client.armDisarm(False, vehicle_name)
client.enableApiControl(False, vehicle_name)
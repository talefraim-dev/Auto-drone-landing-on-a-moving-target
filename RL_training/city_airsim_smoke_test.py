import cosysairsim as airsim


def main():
    print("[INFO] Connecting to Cosys-AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()

    print("[INFO] Enabling API control...")
    client.enableApiControl(True, vehicle_name="Drone1")
    client.armDisarm(True, vehicle_name="Drone1")

    print("[INFO] Taking off...")
    client.takeoffAsync(vehicle_name="Drone1").join()

    print("[INFO] Getting state...")
    state = client.getMultirotorState(vehicle_name="Drone1")
    print(state)

    print("[INFO] Requesting camera image...")
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ], vehicle_name="Drone1")

    if not responses:
        print("[ERROR] No response from simGetImages.")
    elif len(responses[0].image_data_uint8) == 0:
        print("[ERROR] Empty image received.")
    else:
        print("[OK] Camera image received.")
        print("[INFO] Width:", responses[0].width)
        print("[INFO] Height:", responses[0].height)

    print("[INFO] Landing...")
    client.landAsync(vehicle_name="Drone1").join()

    client.armDisarm(False, vehicle_name="Drone1")
    client.enableApiControl(False, vehicle_name="Drone1")

    print("[DONE] Smoke test finished.")


if __name__ == "__main__":
    main()
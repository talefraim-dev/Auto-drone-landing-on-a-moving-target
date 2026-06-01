import cosysairsim as airsim

client = airsim.MultirotorClient()
client.confirmConnection()

names = client.simListSceneObjects(".*")

print("\n=== Scene objects containing X6 / BMW / Car / Target / BP ===")
for name in names:
    lower = name.lower()
    if any(key in lower for key in ["x6", "bmw", "car", "target", "bp"]):
        print(name)

print(f"\nTotal objects: {len(names)}")
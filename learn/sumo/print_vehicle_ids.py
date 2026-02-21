"Learn to start sumo with traci and print vehicle ids"

import yaml
import traci

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    sumo_config_file_path = config["sumo"]["config_file"]

traci.start(["sumo", "-c", sumo_config_file_path])
print("SUMO started and TraCI connected.")

for step in range(10):
    traci.simulationStep()
    vehicle_ids  = traci.vehicle.getIDList()
    print(f"Step {step}: Vehicle IDs: {vehicle_ids}")

traci.close()



import traci
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    sumo_config_file_path = config["sumo"]["config_file"]

traci.start(["sumo", "-c", sumo_config_file_path])


# Print all junctions id
junctions = traci.junction.getIDList()
for junction in junctions:
    print(f"Id: {junction}")


# Print traffic controlled junctions
tls_junctions = traci.trafficlight.getIDList()
for junction in tls_junctions:
    print(f"Traffic controlled junction Id: {junction}")

traci.close()

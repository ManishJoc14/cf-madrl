import traci
import sumolib
import yaml
import matplotlib.pyplot as plt


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    sumo_config_file_path = config["sumo"]["config_file"]
    sumo_net_file_path = config["sumo"]["net_file"]

traci.start(["sumo", "-c", sumo_config_file_path])

# controlled traffic junction ids
tls_ids = traci.trafficlight.getIDList()

# Choose first jucntion
tls_id = tls_ids[0]

# Find incoming lanes only
incoming_lanes = []
net = sumolib.net.readNet(sumo_net_file_path)
for edge in net.getNode(tls_id).getIncoming():
    for lane in edge.getLanes():
        incoming_lanes.append(lane.getID())

print(f"Incoming lanes of Junction {tls_id}: {incoming_lanes}")

junction_steps = {tls_id: [] for tls_id in tls_ids}
junction_waiting_vehicles = {tls_id: [] for tls_id in tls_ids}
junction_waiting_times = {tls_id: [] for tls_id in tls_ids}
junction_incoming_lanes = {tls_ids: [] for tls_id in tls_ids}

# find incoming lanes
for tls_id in tls_ids:
    for edge in net.getNode(tls_id).getIncoming():
        junction_incoming_lanes[tls_id] = [lane.getID() for lane in edge.getLanes()]

for step in range(100):
    # lane_ids = traci.trafficlight.getControlledLanes(tls_id) # it gives both incoming and outgoing lanes
    for tls_id in tls_ids:

        total_waiting = 0
        total_wait_time = 0
        for lane_id in junction_incoming_lanes[tls_id]:
            total_waiting += traci.lane.getLastStepHaltingNumber(lane_id)
            total_wait_time += traci.lane.getWaitingTime(lane_id)

        junction_steps[tls_id].append(step)
        junction_waiting_vehicles[tls_id].append(total_waiting)
        junction_waiting_times[tls_id].append(total_wait_time)

    traci.simulationStep()

traci.close()


# Visualization
plt.figure(figsize=(12, 8))

# Number of Waiting Vehicles
plt.subplot(2, 1, 1)
for tls_id in tls_ids:
    plt.plot(
        junction_steps[tls_id],
        junction_waiting_vehicles[tls_id],
        label=f"Junction {tls_id}",
    )
plt.xlabel("Simulation Step")
plt.ylabel("Number of Waiting Vehicles")
plt.title(f"Waiting Vehicles per Step for All Junctions")
plt.legend()

# Total Waiting Time
plt.subplot(2, 1, 2)
for tls_id in tls_ids:
    plt.plot(
        junction_steps[tls_id],
        junction_waiting_times[tls_id],
        label=f"Junction {tls_id}",
    )
plt.xlabel("Simulation Step")
plt.ylabel("Total Waiting Time")
plt.title(f"Total Waiting Time per Step for ALl Junctions")
plt.legend()

plt.tight_layout()
plt.show()
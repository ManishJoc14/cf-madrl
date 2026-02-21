import traci
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    sumo_config_file_path = config["sumo"]["config_file"]

traci.start(["sumo-gui", "-c", sumo_config_file_path])

# Ids of controlled junctions
tls_junctions = traci.trafficlight.getIDList()

# Select first controlled junction
tls_id = tls_junctions[0]

# Program logic
logics = traci.trafficlight.getAllProgramLogics(tls_id)
print(f"\nLogics of Junction {tls_id}: {logics}")

# Tls states
tls_states = traci.trafficlight.getRedYellowGreenState(tls_id)
print(f"Tls state of junction {tls_id}: {tls_states}")

# Index of the current phase
phase = traci.trafficlight.getPhase(tls_id)
print(f"Current Phase Index of junction {tls_id}: {phase}")

num_phases = len(logics[0].phases)
print(f"Possible phase indexes for junction {tls_id}: {list(range(num_phases))}")

phase_durations = [
    (1, 10),
    (2, 20),
    (3, 20),
    (0, 10),
]  # (phase_index, duration_in_steps)
step = 0

for phase_index, duration in phase_durations:
    traci.trafficlight.setPhase(tls_id, phase_index)
    print(f"Step {step}: Set phase {phase_index} for {duration} steps")
    for _ in range(duration):
        traci.simulationStep()
        step += 1

traci.close()

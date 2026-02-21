"""Test mode with simulated scenarios"""
import time


TEST_SCENARIOS = [
    ("Heavy lanes 0&3", [9, 1, 1, 8], [28, 3, 4, 25], "Expect phase 2"),
    ("Heavy lanes 1&2", [1, 8, 9, 1], [2, 27, 30, 3], "Expect phase 0"),
    ("Lane 0 heavy", [10, 0, 0, 0], [30, 0, 0, 0], "Expect phase 2"),
    ("Lane 3 heavy", [0, 0, 0, 10], [0, 0, 0, 30], "Expect phase 2"),
    ("Lane 1 heavy", [0, 10, 0, 0], [0, 30, 0, 0], "Expect phase 0"),
    ("Lane 2 heavy", [0, 0, 10, 0], [0, 0, 30, 0], "Expect phase 0"),
    ("Balanced", [5, 5, 5, 5], [15, 15, 15, 15], "Either phase"),
    ("No congestion", [0, 0, 0, 0], [0, 0, 0, 0], "Maintain current"),
]


def run_test_mode(engine):
    """Run test mode with simulated scenarios"""
    current_phase = 0
    step = 0
    
    try:
        while True:
            scenario_idx = step % len(TEST_SCENARIOS)
            name, queues, waits, insight = TEST_SCENARIOS[scenario_idx]
            
            print(f"\n{'=' * 60}")
            print(f"Step {step:03} | {name}")
            print(f"         | Queues: {queues} | Waits: {waits}")
            print(f"         | {insight}")
            print(f"{'=' * 60}")
            
            phase, duration, yellow = engine.get_action(queues, waits, current_phase)
            
            if yellow:
                print(f"[YELLOW] {current_phase} -> {phase} (3s)")
                time.sleep(3)
            
            print(f"[GREEN] Phase {phase} for {duration}s")
            current_phase = phase
            time.sleep(duration)
            
            step += 1
    
    except KeyboardInterrupt:
        print("\nTest mode stopped.")

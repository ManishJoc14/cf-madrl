
Ideal Project Structure: 

```
cf_madrl/
│
├── main.py                    # Entry point (train / eval)
├── config.yaml                # All configuration settings
│
├── utils.py                   # Logging, config, metrics, console
├── federation.py              # Clustering + FedAvg
├── agent_manager.py           # RLlib PPO agent manager
├── multi_agent_sumo_env.py    # SUMO multi-agent environment
│
├── visualize.py               # Plots generation
├── generate_plots.py          # Plot generation script
│
├── logs/                      # Logs directory    
│   ├── training_logs.json
│   └── evaluation_logs.json
│
│── plots/                     # Generated plots
│    ├── training/
│    └── evaluation/
│
└──sumo_files/                # SUMO configuration files
└──learn/                     # Learn anything related to project
└──tests/                     # To test the code
```
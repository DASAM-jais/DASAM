This repository contains parts of the agent-based model code supporting the paper titled “Intervention Strategy for Online Reviews in Sharing Accommodation Platforms.” The code implements a data-driven agent-based shared accommodation model (DASAM).

- **DASAM_base.py** includes classes and functions for reproducing the simulation model process. It serves as the mapping from the mathematical model to the computational model.
- **Repeated_Experiments.py** contains the experimental design and code for running experiments in parallel. It executes the experiments based on the number of cores available on the computer minus two.
- The **Results_Analysis** folder includes some preprocessing, validation, and confirmation processes for analyzing the experimental results. Additionally, we provide a sample of the summarized data, which includes the first 1,000 rows of the original dataset.

We hope that you can gain a better understanding of the simulation's complexity and the decisions made by agents through the provided code.


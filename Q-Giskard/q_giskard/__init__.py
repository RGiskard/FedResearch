"""
GISKARD acronym and module mapping:

G: Gate-based Quantum Circuits
   → gates/ (gates.py)

I: Inference Engine
   → federation/client.py (GiskardClient)

S: State Preparation & Superposition
   → preprocessing/encoding.py

K: Kinetic Gradient Computation
   → kinetics/kinetics.py

A: Aggregation & Averaging
   → federation/server.py

R: Runner Orchestration
   → runner.py

D: Deployment Entry Point
   → main.py

Additional:
- preprocessing/decoding.py   (quantum→classical decoding)
- utils/utils.py             (helpers: serialize, deserialize, metrics, logging)
"""

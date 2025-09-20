# Perudo
Perudo project template


1. Save files under the structure shown above.
2. Create a virtualenv and `pip install -r requirements.txt`.
3. Run tournaments:
python eval/tournament.py --agent1 baseline --agent2 mc --games 200 --mc-n 300 --palifico --exact


Notes:
- The Monte-Carlo agent is single-machine friendly; for speed, you can parallelize evaluate_action in mc_agent using multiprocessing.
- Use this modular layout to later plug ISMCTS, opponent modeling, or NN-based rollout policies.

from datasets import load_dataset

dataset = load_dataset("./data/kepler_flare/kepler_flare.py", trust_remote_code=True)

dataset.push_to_hub("Maxwell-Jia/kepler-flare")
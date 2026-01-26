from datasets import load_dataset

print("Téléchargement du dataset night2day...")
dataset = load_dataset("huggan/night2day")
dataset.save_to_disk("./data/night2day")
print("Dataset téléchargé dans ./data/night2day")

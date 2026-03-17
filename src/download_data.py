import kagglehub
import shutil
import os

path = kagglehub.dataset_download("aiswaryaramachandran/hindienglish-corpora")
print("Path to dataset files:", path)

os.makedirs("../data", exist_ok=True)
for f in os.listdir(path):
    shutil.copy(os.path.join(path, f), os.path.join("../data", f))
    print(f"Copied: {f}")
print("Done!")
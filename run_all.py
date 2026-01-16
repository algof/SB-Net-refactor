import subprocess
from pathlib import Path

work_dir = Path(__file__).parent

print("===== Running CTU-13 data maker =====")
subprocess.run([
    "jupyter", "nbconvert",
    "--to", "notebook",
    "--execute",
    "--inplace",
    r"Datasets\CTU-13\data_maker_CTU_13.ipynb"]
)
print("")

print("===== Running NCC data maker =====")
subprocess.run([
    "jupyter", "nbconvert",
    "--to", "notebook",
    "--execute",
    "--inplace",
    r"Datasets\NCC\data_maker_NCC.ipynb"]
)
print("")

print("===== Running NCC-2 data maker =====")
subprocess.run([
    "jupyter", "nbconvert",
    "--to", "notebook",
    "--execute",
    "--inplace",
    r"Datasets\NCC-2\data_maker_NCC_2.ipynb"]
)
print("")

print("===== Running train combiner =====")
subprocess.run(["python", "train_combiner.py"], cwd=work_dir / "Datasets")
print("")

print("===== Running train_train and train_test maker =====")
subprocess.run([
    "jupyter", "nbconvert",
    "--to", "notebook",
    "--execute",
    "--inplace",
    "train_train_test_maker.ipynb"]
)
print("")

print("===== Running rank aggregation =====")
subprocess.run([
    "jupyter", "nbconvert",
    "--to", "notebook",
    "--execute",
    "--inplace",
    "rank_aggregation.ipynb"]
)
print("")

print("===== Calculating borda score =====")
subprocess.run(["python", "borda_score.py"])
print("")

print("===== Running looping features =====")
subprocess.run(["python", "looping_features.py"])
print("")

print("===== Running looping classification =====")
subprocess.run(["python", "looping_classification.py"])
print("")

print("===== Running final classification test =====")
subprocess.run(["python", "final_classification_test.py"])
print("")
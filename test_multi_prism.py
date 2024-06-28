from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import jax.numpy as jnp


path = Path("/home/spol/Documents/repos/hypermagnetics/")
folder = path / "data" / "Meshes_complete_2D"

for filepath in list(folder.iterdir())[:300]:
    with open(filepath, "r") as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            columns = line.split()
            r = columns[:3]
            size = columns[3:]

            # Convert the coordinates and size to integers
            r = [float(coord) for coord in r]
            size = [float(dim) for dim in size]

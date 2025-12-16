import jax.random as jr
import jax.numpy as jnp


class Cell:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.width = x1 - x0
        self.height = y1 - y0

    def center(self):
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def split(self):
        """Split into four subcells."""
        mx = (self.x0 + self.x1) / 2
        my = (self.y0 + self.y1) / 2
        return [
            Cell(self.x0, self.y0, mx, my),
            Cell(mx, self.y0, self.x1, my),
            Cell(self.x0, my, mx, self.y1),
            Cell(mx, my, self.x1, self.y1),
        ]
    
class Cell3D:
    def __init__(self, x0, y0, z0, x1, y1, z1):
        self.x0, self.y0, self.z0, self.x1, self.y1, self.z1 = x0, y0, z0, x1, y1, z1
        self.width = x1 - x0
        self.height = y1 - y0
        self.depth = z1 - z0
    def center(self):
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2, (self.z0 + self.z1) / 2)

    def split(self):
        """Split into eight subcells."""
        mx = (self.x0 + self.x1) / 2
        my = (self.y0 + self.y1) / 2
        mz = (self.z0 + self.z1) / 2
        return [
            Cell3D(self.x0, self.y0, self.z0, mx, my, mz),
            Cell3D(mx, self.y0, self.z0, self.x1, my, mz),
            Cell3D(self.x0, my, self.z0, mx, self.y1, mz),
            Cell3D(mx, my, self.z0, self.x1, self.y1, mz),
            Cell3D(self.x0, self.y0, mz, mx, my, self.z1),
            Cell3D(mx, self.y0, mz, self.x1, my, self.z1),
            Cell3D(self.x0, my, mz, mx, self.y1, self.z1),
            Cell3D(mx, my, mz, self.x1, self.y1, self.z1),
        ]


def random_quadtree(x0, y0, x1, y1, n_cells, key):
    """Generate a random 2D quadtree until about n_cells are reached."""
    cells = [Cell(x0, y0, x1, y1)]
    while len(cells) < n_cells:
        # Pick a random cell to split
        # idx = random.randrange(len(cells))
        # Pick a random cell to split weighted by area
        areas = jnp.array([c.width * c.height for c in cells])
        idx_key, key = jr.split(key, 2)
        idx = jr.choice(idx_key, jnp.array(range(len(cells))), p=areas / jnp.sum(areas))
        chosen = cells.pop(idx)
        new_cells = chosen.split()
        cells.extend(new_cells)
    return cells, key

def random_quadtree3D(x0, y0, z0, x1, y1, z1, n_cells, key):
    """Generate a random 3D quadtree until about n_cells are reached."""
    cells = [Cell3D(x0, y0, z0, x1, y1, z1)]
    while len(cells) < n_cells:
        # Pick a random cell to split
        # idx = random.randrange(len(cells))
        # Pick a random cell to split weighted by area
        volumes = jnp.array([c.width * c.height * c.depth for c in cells])
        idx_key, key = jr.split(key, 2)
        idx = jr.choice(idx_key, jnp.array(range(len(cells))), p=volumes / jnp.sum(volumes))
        chosen = cells.pop(idx)
        new_cells = chosen.split()
        cells.extend(new_cells)
    return cells, key

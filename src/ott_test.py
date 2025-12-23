import jax
import jax.numpy as jnp
from ott.tools import sliced
from ott.solvers.linear import sinkhorn
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.tools.sinkhorn_divergence import sinkhorn_divergence
from ott.geometry import costs
from time import time

# Beispieldaten erzeugen (z.B. zwei Punktwolken)
key = jax.random.PRNGKey(0)
key, subkey1, subkey2 = jax.random.split(key, 3)

# Create batches of point clouds: (batch_size, n_points, n_dims)
n_points = 1000
n_dims = 10

key, subkey1, subkey2 = jax.random.split(key, 3)
# Batch of point clouds
x = jax.random.normal(subkey1, (n_points, n_dims))
y = jax.random.normal(subkey2, (n_points, n_dims)) + \
    3.0  # Verschiebung für Variation


# 1. Sliced Wasserstein Distanz für einen einzelnen Pair (erste aus dem Batch)
# n_proj: Anzahl der zufälligen Richtungen (Slices)
_n_proj = 2000

start = time()

swd_value = sliced.sliced_wasserstein(
    x, y,
    n_proj=_n_proj,
    rng=key,
    cost_fn=costs.Euclidean()
)
print(f"SWD: {swd_value[0]**2}")
end = time()
print(f"Time without vmap: {end - start:.4f} seconds")

# 2. Sinkhorn für einen einzelnen Pair (erste aus dem Batch)
# n_proj: Anzahl der zufälligen Richtungen (Slices)
# start = time()

# geom = pointcloud.PointCloud(x, y, cost_fn=costs.Euclidean())
# prob = linear_problem.LinearProblem(geom)
# solver = sinkhorn.Sinkhorn()
# out = solver(prob)


# print(f"Wasserstein Distanz: {out.reg_ot_cost}")

# end = time()
# print(f"Time without vmap: {end - start:.4f} seconds")

# 3. Sinkhorn Divergenz für einen einzelnen Pair (erste aus dem Batch)
# start = time()

# sink_v2_squared = sinkhorn_divergence(
#     pointcloud.PointCloud,
#     x, y,
#     epsilon=0.01
# )
# end = time()
# print(f"Wasserstein Distanz: {sink_v2_squared}")
# print(f"Time without vmap: {end - start:.4f} seconds")

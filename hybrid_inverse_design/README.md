# Global-to-Local Optimization Framework

This project implements a global-to-local optimization strategy tailored for metasurface design and other multi-parameter optimization problems. The approach combines:

- A custom global optimizer that discretizes the parameter space, estimates function minima using gradients, and discards non-promising regions.
- Clustering of promising regions using DBSCAN to identify distinct areas of interest.
- Local optimization within each cluster using the L-BFGS-B algorithm.
- Visualization tools for analyzing optimization progress.

---

Project Structure

- 'stogo_search(...)' — Global optimization via adaptive space partitioning and pruning.
- 'cluster_promising_regions(...)' — Clusters promising subregions from the global search.
- 'local_optimization(...)' — Applies L-BFGS-B local search in each cluster with multiple starting points.
- 'plot_best_fom_iter(...)' — Visualizes cumulative best FoM per iteration (global + local).
- 'plot_best_fom_calc(...)' — Visualizes cumulative best FoM per function evaluation.
- '_compute_gradient(...)' — Computes numerical gradients via central differences.

---

Algorithm Overview

Global Optimization ('stogo_search')
1. Initial discretization: Divide the domain into a grid of subregions.
2. Estimate region quality: Evaluate the function and its gradient at the center of each subregion.
3. Filter: Estimate local minima and discard regions unlikely to contain the global minimum.
4. Cluster: Use DBSCAN to group promising regions and track their shrinkage.
5. Stop early if all clusters have shrunk below a threshold percentage of the original domain or subregions are too small.

Clustering ('cluster_promising_regions')
- Uses DBSCAN on region centers to detect spatial clusters.
- Creates bounding boxes (with padding) around each cluster for local optimization.

Local Optimization ('local_optimization')
- Runs 'scipy.optimize.minimize' ('L-BFGS-B') from a single/multiple random starting point(s) in each cluster.
- Records and optionally saves optimization paths.

---

Visualization

Use:
- 'plot_best_fom_iter(...)' — to view how the best FoM evolves per iteration.
- 'plot_best_fom_calc(...)' — to analyze improvement per function call.

Plots are saved in .svg format.

---

Parameters

	Global Search

| Parameter               | Description                                                  |
|------------------------|--------------------------------------------------------------|
| 'bounds'               | Domain bounds, e.g. '[(xmin, xmax), (ymin, ymax)]'            |
| 'max_iters'            | Max number of global iterations                              |
| 'min_iters'            | Minimum iterations before shrinkage-based early stopping     |
| 'subdivisions'         | Number of subdivisions per promising region (after iter 0)   |
| 'initial_subdivisions' | Grid resolution for iteration 0                              |
| 'shrinkage_threshold'  | Stop if cluster size drops below this fraction of domain     |
| 'min_region_size_ratio'| Stop if subregion area drops below this ratio of domain area |

	Local Optimization

| Parameter     | Description                                             |
|---------------|---------------------------------------------------------|
| 'n_starts'    | Number of random local starts per cluster               |

---

Output Files

| File                          | Description                              |
|-------------------------------|------------------------------------------|
| 'stogo_iterations.csv'        | FoM and estimated minimum at each point  |
| 'local_optimization_paths.csv'| Local optimization path per run          |
| 'best_fom_per_iter.svg'       | Best FoM evolution vs. iteration         |
| 'best_fom_per_calc.svg'       | Best FoM evolution vs. function call     |

---

Dependencies

- 'numpy'
- 'scipy'
- 'matplotlib'
- 'pandas'
- 'scikit-learn'

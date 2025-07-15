import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams['figure.autolayout'] = True
plt.rcParams.update({'font.size': 16})
from sklearn.cluster import DBSCAN

# Compute gradient of objective function (for global search)
def _compute_gradient(fom_func, params, epsilon=1e-3):
    '''Calculation of the numerical gradient of a function (fom_func) at specific coordinates (params).
        Params:
            fom_func: figure-of-merit function;
            params: coordinates of function;
            epsilon: perturbation factor.
    '''
    grad = np.zeros_like(params)
    for i in range(len(params)):
        delta = np.zeros_like(params)
        delta[i] = epsilon
        grad[i] = (fom_func(params + delta) - fom_func(params - delta)) / (2 * epsilon)
    return grad

# Global Search + Space Discretization/Discarding
def stogo_search(objective_function, bounds, max_iters, min_iters, subdivisions, initial_subdivisions,
                 shrinkage_threshold, min_region_size_ratio, minimize = True, save_results=True):
    '''Run global search of the function space, by dividing it into multiple subregions ('subdivisions'),
       comparing the subestimated minima/maxima of each subregion with the best FoM so far, and discarding
       non-promising regions.
       Args:
            - objective_function: figure-of-merit function;
            - bounds: array of tuples, e.g. [(x_min, x_max), (y_min, y_max)];
            - max_iters: maximum amount of iterations/subdivisions;
            - min_iters: minimum amount of iterations/subdivisions;
            - subdivisions: number of subdivisions (in x and y) of the promissing regions per iteration;
            - initial_subdivisions: number of divisions (in x and y) of the search space in iteration 0;
            - shrinkage_threshold: clusters' shrinkage ratio between iterations (as percentage of the 
            largest initial space size);
            - min_region_size_ratio: minimum side of a subregion before autoshutoff;
            
            - save_results: option to save results (default = True).
       Results:
            - csv file with coordinates, estimated best and best FoM at each iteration.
    '''
    best_fom = float('inf')
    best_params = None
    promising_regions = [bounds]
    results = []
    cluster_areas = []

    full_area = (bounds[0][1] - bounds[0][0]) * (bounds[1][1] - bounds[1][0])
    original_width = bounds[0][1] - bounds[0][0]
    original_height = bounds[1][1] - bounds[1][0]

    for iteration in range(max_iters):
        new_regions = []
        # Space division
        current_subdivisions = initial_subdivisions if iteration == 0 else subdivisions

        for region in promising_regions:
            sub_intervals_x = np.linspace(region[0][0], region[0][1], current_subdivisions + 1)
            sub_intervals_y = np.linspace(region[1][0], region[1][1], current_subdivisions + 1)

            for i in range(current_subdivisions):
                for j in range(current_subdivisions):
                    sub_region = [(sub_intervals_x[i], sub_intervals_x[i+1]),
                                  (sub_intervals_y[j], sub_intervals_y[j+1])]
                    new_regions.append(sub_region)

        # Subregions gradient and FoM at center
        promising_regions = []
        for region in new_regions:
            center = np.array([(r[0] + r[1]) / 2 for r in region])
            fom_value = objective_function(center)
            gradient = _compute_gradient(objective_function, center)

            width_x = region[0][1] - region[0][0]
            width_y = region[1][1] - region[1][0]
            max_distance = 0.5 * np.sqrt((width_x / 2)**2 + (width_y / 2)**2)
            estimated_min = fom_value - np.linalg.norm(gradient) * max_distance

            results.append([iteration, center[0], center[1], fom_value, estimated_min])

            if estimated_min < best_fom:
                promising_regions.append(region)
                if fom_value < best_fom:
                    best_fom = fom_value
                    best_params = center

        if not promising_regions:
            break

        # Perform clustering on promising regions
        centers = np.array([
            [(region[0][0] + region[0][1]) / 2, (region[1][0] + region[1][1]) / 2]
            for region in promising_regions
        ])
        widths = np.array([region[0][1] - region[0][0] for region in promising_regions])
        heights = np.array([region[1][1] - region[1][0] for region in promising_regions])
        avg_width = np.mean(widths)
        avg_height = np.mean(heights)

        eps = max(avg_width, avg_height) * 1.1
        clustering = DBSCAN(eps=eps, min_samples=1).fit(centers)
        labels = clustering.labels_

        # Check shrinkage
        all_shrunk = True
        for label in np.unique(labels):
            cluster_points = centers[labels == label]
            min_x, max_x = np.min(cluster_points[:, 0]), np.max(cluster_points[:, 0])
            min_y, max_y = np.min(cluster_points[:, 1]), np.max(cluster_points[:, 1])
            cluster_width = max_x - min_x
            cluster_height = max_y - min_y

            width_ratio = cluster_width / original_width
            height_ratio = cluster_height / original_height
            max_ratio = max(width_ratio, height_ratio)

            if max_ratio >= shrinkage_threshold:
                all_shrunk = False
                break

        if iteration >= min_iters and all_shrunk:
            print(f"Stopping early at iteration {iteration} due to shrinkage threshold.")
            break

        # Stop if subregions too small
        region_area = (bounds[0][1] - bounds[0][0]) / current_subdivisions * \
                      (bounds[1][1] - bounds[1][0]) / current_subdivisions
        if region_area / full_area < min_region_size_ratio:
            print(f"[Early stop] Subregions too small: area ratio = {region_area / full_area:.4f}")
            break

        if not promising_regions:
            print("[Early stop] No promising regions")
            break

    if save_results:
        df = pd.DataFrame(results, columns=["Iteration", "X", "Y", "FoM", "Estimated Min"])
        df.to_csv("stogo_iterations.csv", index=False)

    return promising_regions


def cluster_promising_regions(promising_regions):
    centers = np.array([
        [(region[0][0] + region[0][1]) / 2, (region[1][0] + region[1][1]) / 2]
        for region in promising_regions
    ])

    widths = np.array([region[0][1] - region[0][0] for region in promising_regions])
    heights = np.array([region[1][1] - region[1][0] for region in promising_regions])
    avg_width = np.mean(widths)
    avg_height = np.mean(heights)
    eps = max(avg_width, avg_height) * 1.1

    clustering = DBSCAN(eps=eps, min_samples=1).fit(centers)
    labels = clustering.labels_

    clustered_regions = []
    for label in np.unique(labels):
        cluster_points = centers[labels == label]
        min_x, max_x = np.min(cluster_points[:, 0]), np.max(cluster_points[:, 0])
        min_y, max_y = np.min(cluster_points[:, 1]), np.max(cluster_points[:, 1])
        padding_x = avg_width / 2
        padding_y = avg_height / 2
        bounding_box = [(min_x - padding_x, max_x + padding_x), (min_y - padding_y, max_y + padding_y)]
        clustered_regions.append(bounding_box)

    return clustered_regions

def local_optimization(objective_function, clustered_regions, n_starts=1):
    best_result = None
    best_fom = float('inf')
    local_paths = []
    run_index = 0

    for region in clustered_regions:
        for _ in range(n_starts):
            x_start = np.random.uniform(region[0][0], region[0][1])
            y_start = np.random.uniform(region[1][0], region[1][1])
            initial_guess = np.array([x_start, y_start])
            path = []

            def callback(xk):
                fom = objective_function(xk)
                path.append([run_index, xk[0], xk[1], fom])

            result = minimize(
                objective_function,
                initial_guess,
                method='L-BFGS-B',
                bounds=region,
                callback=callback
            )

            final_fom = result.fun
            path.append([run_index, result.x[0], result.x[1], final_fom])

            if final_fom < best_fom:
                best_fom = final_fom
                best_result = result.x

            local_paths.extend(path)
            run_index += 1

    df_paths = pd.DataFrame(local_paths, columns=["Run", "X", "Y", "FoM"])
    df_paths.to_csv("local_optimization_paths.csv", index=False)

    return best_result, best_fom

# VISUALIZATION FUNCTIONS

# Plot cumulative best FoM solution at each iteration
def plot_best_fom_iter(df_global, df_local, save = False):
    
    # Compute cumulative best FoM for global optimization
    global_best_fom = []
    current_best = float('inf')
    global_iterations = sorted(df_global["Iteration"].unique())
    for i in global_iterations:
        iter_best = df_global[df_global["Iteration"] == i]["FoM"].min()
        current_best = min(current_best, iter_best)
        global_best_fom.append(current_best)

    runs = df_local["Run"].unique()
    run_iters = [df_local[df_local["Run"] == r].reset_index(drop=True) for r in runs]

    local_best_fom = []
    current_best = global_best_fom[-1]
    t = 0
    while True:
        all_done = True
        for run in run_iters:
            if t < len(run):
                current_best = min(current_best, run.loc[t, "FoM"])
                all_done = False
        if all_done:
            break
        local_best_fom.append(current_best)
        t += 1

    global_x = list(range(len(global_best_fom)))
    local_x = list(range(global_x[-1] + 1, global_x[-1] + 1 + len(local_best_fom)))

    plt.figure(figsize=(7, 5))
    plt.plot(global_x, global_best_fom, 'o-k', label="Global Optimization")
    plt.plot(local_x, local_best_fom, 'o-m', label="Local Optimization")
    plt.axvline(global_x[-1], linestyle='--', color='gray')
    plt.xlabel("Iteration")
    plt.ylabel("Best FoM so far")
    plt.yscale("log")
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig("best_fom_per_iter.svg", dpi = 300)

# Plot cumulative best FoM solution evolution through function evaluations
def plot_best_fom_calc(df_global, df_local, save = False):
    # Sort by iteration
    df_global = df_global.sort_values(by="Iteration")

    # Compute cumulative best FoM for global optimization
    best_global_fom = []
    current_best = float('inf')
    for _, row in df_global.iterrows():
        current_best = min(current_best, row["FoM"])
        best_global_fom.append(current_best)

    # Number of global function evaluations
    global_steps = list(range(len(best_global_fom)))

    # Compute cumulative best FoM for local optimization
    best_local_fom = []
    current_best = best_global_fom[-1]
    for _, row in df_local.iterrows():
        current_best = min(current_best, row["FoM"])
        best_local_fom.append(current_best)

    local_steps = list(range(len(global_steps), len(global_steps) + len(best_local_fom)))

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(global_steps, best_global_fom, '-o', label="Global Optimization", color='black', markersize=4)
    plt.plot(local_steps, best_local_fom, '-o', label="Local Optimization", color='m', markersize=3)
    plt.axvline(x=len(global_steps) - 1, color='gray', linestyle='--', label='Global/Local Transition')
    plt.xlabel("Function Evaluations")
    plt.ylabel("Best FoM")
    plt.yscale("log")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig("best_fom_per_eval.svg", dpi = 300)

# Plot regions of global iterations
def plot_global_iterations(objective_function, csv_file, bounds, save = False):
    df = pd.read_csv(csv_file)
    iterations = df["Iteration"].unique()
    
    x_vals = np.linspace(bounds[0][0], bounds[0][1], 100)
    y_vals = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = objective_function((X, Y))
    
    Z_log = np.log1p(np.abs(Z))
    
    for iteration in iterations:
        plt.figure(figsize=(8, 6))
        plt.imshow(Z_log, extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]], origin='lower', cmap="viridis", aspect='auto')
        
        iter_data = df[df["Iteration"] == iteration]
        plt.scatter(iter_data["X"], iter_data["Y"], c='r', edgecolors='k')
        plt.colorbar(label="FoM")

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Iteration {iteration}")
        
        if save:
            plt.savefig(f'iteration_{iteration}.svg')

# Plot final clusters
def plot_clustered_regions(objective_function, clustered_regions, bounds, save = False):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the full bounds as a background
    ax.set_xlim(bounds[0][0], bounds[0][1])
    ax.set_ylim(bounds[1][0], bounds[1][1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('clustered regions')

    
    x_vals = np.linspace(bounds[0][0], bounds[0][1], 100)
    y_vals = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = objective_function((X, Y))
    
    Z_log = np.log1p(np.abs(Z))
    plt.imshow(Z_log, extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]], origin='lower', cmap="viridis", aspect='auto')

    # Plot each cluster as a rectangle
    for idx, region in enumerate(clustered_regions):
        x_min, x_max = region[0]
        y_min, y_max = region[1]
        width = x_max - x_min
        height = y_max - y_min
        
        rect = patches.Rectangle((x_min, y_min), width, height, 
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    # plt.gca().set_aspect('equal', adjustable='box')
    if save:
        plt.savefig("clustered regions.svg", dpi = 300)


import numpy as np

from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


class GridPointOptimizer:
    """
    GRIDPOINT Warehouse Location Optimizer

    Mathematical objective:

        Minimize:
            J = sum(demand_i * distance_i)

    where:
        demand_i   = daily orders at neighborhood i
        distance_i = distance from neighborhood i
                     to its assigned warehouse

    The optimizer uses:
        1. Nearest-warehouse assignment
        2. Weighted geometric median (Weiszfeld algorithm)
        3. Multiple random restarts
        4. Convergence detection
    """

    def __init__(
        self,
        n_warehouses=3,
        max_iterations=100,
        tolerance=1e-5,
        restarts=10,
        random_state=42
    ):
        self.k = n_warehouses
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.restarts = restarts
        self.random_state = random_state

        self.warehouses = None
        self.assignments = None
        self.cost = None

    # --------------------------------------------------
    # DISTANCE CALCULATION

    def calculate_distance_matrix(self, points, warehouses):
        """
        Calculate Euclidean distance between every
        neighborhood and every warehouse.

        Output shape:

            (number of neighborhoods,
             number of warehouses)
        """

        differences = (
            points[:, np.newaxis, :]
            - warehouses[np.newaxis, :, :]
        )

        distances = np.sqrt(
            np.sum(differences ** 2, axis=2)
        )

        return distances

    # --------------------------------------------------
    # INITIALIZATION

    def initialize_warehouses(self, points, demands, rng):
        """
        Demand-weighted initialization.

        High-demand neighborhoods have a higher probability
        of being selected as initial warehouse locations.
        """

        probabilities = demands / demands.sum()

        indices = rng.choice(
            len(points),
            size=self.k,
            replace=False,
            p=probabilities
        )

        return points[indices].copy()

    # --------------------------------------------------
    # ASSIGNMENT

    def assign_neighborhoods(self, points, warehouses):
        """
        Assign every neighborhood to its nearest warehouse.
        """

        distances = self.calculate_distance_matrix(
            points,
            warehouses
        )

        assignments = np.argmin(
            distances,
            axis=1
        )

        return assignments, distances

    # --------------------------------------------------
    # WEISZFELD ALGORITHM
    def assign_with_constraints(
        self,
        points,
        demands,
        warehouses,
        capacities,
        max_radius
    ):
        """
        Solve the constrained assignment problem
        using Mixed-Integer Linear Programming (MILP).

        Objective:

            minimize sum(
                demand_i * distance_ij * x_ij
            )

        Subject to:

            Every neighborhood is assigned exactly once.

            Warehouse capacity is respected.

            Assignments outside the service radius
            are prohibited.
        """

        n = len(points)
        k = len(warehouses)

        distances = self.calculate_distance_matrix(
            points,
            warehouses
        )

        # ------------------------------------------
        # Number of binary decision variables

        num_variables = n * k

        # Objective coefficients
        c = np.zeros(num_variables)

        # Upper bounds for variables.
        # 1 = assignment allowed
        # 0 = assignment prohibited
        upper_bounds = np.ones(num_variables)

        # ------------------------------------------
        # Build objective

        for i in range(n):

            for j in range(k):

                index = i * k + j

                c[index] = (
                    demands[i]
                    * distances[i, j]
                )

                # Service-radius constraint
                if distances[i, j] > max_radius:

                    upper_bounds[index] = 0

        # ------------------------------------------
        # Build constraints

        num_constraints = n + k

        A = lil_matrix(
            (num_constraints, num_variables)
        )

        lower = np.zeros(num_constraints)
        upper = np.zeros(num_constraints)

        for i in range(n):

            for j in range(k):

                index = i * k + j

                A[i, index] = 1

            lower[i] = 1
            upper[i] = 1

        for j in range(k):

            row = n + j

            for i in range(n):

                index = i * k + j

                A[row, index] = demands[i]

            lower[row] = 0

            upper[row] = capacities[j]

        # ------------------------------------------
        # Solve MILP

        constraint = LinearConstraint(
            A.tocsr(),
            lower,
            upper
        )

        result = milp(
            c=c,
            integrality=np.ones(
                num_variables
            ),
            bounds=Bounds(
                np.zeros(num_variables),
                upper_bounds
            ),
            constraints=constraint
        )

        # ------------------------------------------
        # Check feasibility
        if not result.success:

            raise ValueError(
                "No feasible assignment exists "
                "for the given warehouse capacities "
                "and service radius."
            )

        # ------------------------------------------
        # Decode solution
        solution = np.rint(
            result.x
        ).astype(int)

        assignments = np.full(
            n,
            -1,
            dtype=int
        )

        for i in range(n):

            for j in range(k):

                index = i * k + j

                if solution[index] == 1:

                    assignments[i] = j

                    break

        # ------------------------------------------
        # Calculate remaining capacity

        remaining_capacity = np.array(
            capacities,
            dtype=float
        )

        for i in range(n):

            warehouse_id = assignments[i]

            remaining_capacity[
                warehouse_id
            ] -= demands[i]

        return (
            assignments,
            distances,
            remaining_capacity,
            result.fun
        )
    def evaluate_constrained_solution(
        self,
        points,
        demands,
        warehouses,
        capacities,
        max_radius
    ):
        """
        Evaluate a warehouse configuration using
        exact constrained assignment.
        """

        (
            assignments,
            distances,
            remaining_capacity,
            optimal_cost
        ) = self.assign_with_constraints(
            points,
            demands,
            warehouses,
            capacities,
            max_radius
        )

        total_demand = np.sum(
            demands
        )

        average_distance = (
            optimal_cost /
            total_demand
        )

        return {
            "assignments": assignments,

            "total_cost":
                optimal_cost,

            "average_distance":
                average_distance,

            "remaining_capacity":
                remaining_capacity,

            "unassigned_demand": 0
        }
    def weighted_geometric_median(
        self,
        points,
        demands,
        max_iterations=100,
        tolerance=1e-5
    ):
        """
        Calculate the weighted geometric median.

        Objective:

            minimize sum(
                demand_i * distance(point_i, x)
            )

        Uses the Weiszfeld iterative algorithm.
        """

        # Start at the weighted centroid.
        total_demand = np.sum(demands)

        current = (
            np.sum(
                points * demands[:, np.newaxis],
                axis=0
            )
            / total_demand
        )

        for _ in range(max_iterations):

            distances = np.linalg.norm(
                points - current,
                axis=1
            )

            # Avoid division by zero.
            if np.any(distances < 1e-12):

                index = np.argmin(distances)

                current = points[index].copy()

                continue

            weights = demands / distances

            new_position = (
                np.sum(
                    points * weights[:, np.newaxis],
                    axis=0
                )
                / np.sum(weights)
            )

            movement = np.linalg.norm(
                new_position - current
            )

            current = new_position

            if movement < tolerance:
                break

        return current

    # --------------------------------------------------
    # UPDATE WAREHOUSES

    def update_warehouses(
        self,
        points,
        demands,
        assignments,
        old_warehouses
    ):
        """
        Move every warehouse to the weighted geometric
        median of its assigned neighborhoods.
        """

        new_warehouses = old_warehouses.copy()

        for warehouse_id in range(self.k):

            mask = (
                assignments == warehouse_id
            )

            # No neighborhoods assigned.
            if not np.any(mask):

                continue

            cluster_points = points[mask]

            cluster_demands = demands[mask]

            new_position = self.weighted_geometric_median(
                cluster_points,
                cluster_demands,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance
            )

            new_warehouses[warehouse_id] = (
                new_position
            )

        return new_warehouses

    # --------------------------------------------------
    # OBJECTIVE FUNCTION
    def calculate_cost(
        self,
        points,
        demands,
        assignments,
        warehouses
    ):
        """
        Calculate:

            J = sum(demand_i * distance_i)
        """

        total_cost = 0.0

        for i in range(len(points)):

            warehouse_id = assignments[i]

            warehouse = warehouses[
                warehouse_id
            ]

            distance = np.linalg.norm(
                points[i] - warehouse
            )

            total_cost += (
                demands[i] * distance
            )

        return total_cost

    # --------------------------------------------------
    # SOLUTION EVALUATION
    def evaluate_solution(
        self,
        points,
        demands,
        warehouses
    ):
        """
        Evaluate any given warehouse configuration.
        """

        assignments, distances = (
            self.assign_neighborhoods(
                points,
                warehouses
            )
        )

        total_weighted_distance = 0.0

        total_demand = np.sum(demands)

        for i in range(len(points)):

            warehouse_id = assignments[i]

            distance = distances[
                i,
                warehouse_id
            ]

            total_weighted_distance += (
                demands[i] * distance
            )

        average_distance = (
            total_weighted_distance
            / total_demand
        )

        return {
            "assignments": assignments,
            "total_weighted_distance":
                total_weighted_distance,
            "average_distance":
                average_distance
        }

    # --------------------------------------------------
    # MAIN OPTIMIZATION
    def fit(self, points, demands):

        points = np.asarray(
            points,
            dtype=float
        )

        demands = np.asarray(
            demands,
            dtype=float
        )

        if len(points) < self.k:

            raise ValueError(
                "Number of neighborhoods must be "
                "greater than or equal to the "
                "number of warehouses."
            )

        if np.any(demands <= 0):

            raise ValueError(
                "All demand values must be positive."
            )

        rng = np.random.default_rng(
            self.random_state
        )

        best_cost = float("inf")

        best_warehouses = None

        best_assignments = None

        best_iterations = 0

        # ------------------------------------------
        # MULTIPLE RANDOM RESTARTS
        for restart in range(self.restarts):

            warehouses = (
                self.initialize_warehouses(
                    points,
                    demands,
                    rng
                )
            )

            previous_cost = float("inf")

            # --------------------------------------
            # ITERATIVE OPTIMIZATION
            for iteration in range(
                self.max_iterations
            ):

                # Step 1:
                # Assign neighborhoods
                assignments, distances = (
                    self.assign_neighborhoods(
                        points,
                        warehouses
                    )
                )

                # Step 2:
                # Calculate objective
                cost = self.calculate_cost(
                    points,
                    demands,
                    assignments,
                    warehouses
                )

                # Step 3:
                # Move warehouses toward the
                # weighted geometric median
                new_warehouses = (
                    self.update_warehouses(
                        points,
                        demands,
                        assignments,
                        warehouses
                    )
                )

                # Calculate movement
                movement = np.max(
                    np.linalg.norm(
                        new_warehouses
                        - warehouses,
                        axis=1
                    )
                )

                warehouses = new_warehouses

                # ----------------------------------
                # CONVERGENCE
                cost_change = abs(
                    previous_cost - cost
                )

                if (
                    cost_change < self.tolerance
                    or movement < self.tolerance
                ):

                    break

                previous_cost = cost

            # --------------------------------------
            # FINAL EVALUATION

            assignments, _ = (
                self.assign_neighborhoods(
                    points,
                    warehouses
                )
            )

            final_cost = self.calculate_cost(
                points,
                demands,
                assignments,
                warehouses
            )

            # --------------------------------------
            # KEEP BEST RESTART

            if final_cost < best_cost:

                best_cost = final_cost

                best_warehouses = (
                    warehouses.copy()
                )

                best_assignments = (
                    assignments.copy()
                )

                best_iterations = (
                    iteration + 1
                )

        self.warehouses = best_warehouses

        self.assignments = best_assignments

        self.cost = best_cost

        return {
            "warehouses":
                best_warehouses,

            "assignments":
                best_assignments,

            "cost":
                best_cost,

            "iterations":
                best_iterations
        }
    def analyze_warehouse_counts(
                self,
                points,
                demands,
                max_warehouses,
                warehouse_capacity,
                max_radius,
                delivery_cost_per_order_km,
                warehouse_operating_cost
            ):
                """
                Evaluate different numbers of warehouses.
    
                For each k:
    
                    Total Cost =
                        Delivery Cost
                        +
                        Infrastructure Cost
    
                Returns a list of scenario results.
                """
    
                results = []
    
                total_demand = np.sum(demands)
    
                for k in range(1, max_warehouses + 1):
    
                    # ------------------------------------------
                    # Check basic capacity feasibility
    
                    total_capacity = (
                        k * warehouse_capacity
                    )
    
                    if total_capacity < total_demand:
    
                        results.append({
                            "warehouses": k,
                            "feasible": False,
                            "delivery_distance": np.nan,
                            "delivery_cost": np.nan,
                            "infrastructure_cost":
                                k * warehouse_operating_cost,
                            "total_cost": np.nan
                        })
    
                        continue
    
                    # ------------------------------------------
                    # Create optimizer for this k
    
                    optimizer = GridPointOptimizer(
                        n_warehouses=k,
                        max_iterations=self.max_iterations,
                        tolerance=self.tolerance,
                        restarts=self.restarts,
                        random_state=self.random_state
                    )
    
                    try:
    
                        # --------------------------------------
                        # Optimize warehouse locations
    
                        result = optimizer.fit(
                            points,
                            demands
                        )
    
                        warehouses = (
                            result["warehouses"]
                        )
    
                        # --------------------------------------
                        # Constrained assignment
    
                        capacities = np.full(
                            k,
                            warehouse_capacity
                        )
    
                        constrained = (
                            optimizer
                            .evaluate_constrained_solution(
                                points,
                                demands,
                                warehouses,
                                capacities,
                                max_radius
                            )
                        )
    
                        delivery_distance = (
                            constrained["total_cost"]
                        )
    
                        delivery_cost = (
                            delivery_distance
                            * delivery_cost_per_order_km
                        )
    
                        infrastructure_cost = (
                            k
                            * warehouse_operating_cost
                        )
    
                        total_cost = (
                            delivery_cost
                            + infrastructure_cost
                        )
    
                        results.append({
    
                            "warehouses": k,
    
                            "feasible": True,
    
                            "delivery_distance":
                                delivery_distance,
    
                            "delivery_cost":
                                delivery_cost,
    
                            "infrastructure_cost":
                                infrastructure_cost,
    
                            "total_cost":
                                total_cost
                        })
    
                    except ValueError:
    
                        results.append({
    
                            "warehouses": k,
    
                            "feasible": False,
    
                            "delivery_distance":
                                np.nan,
    
                            "delivery_cost":
                                np.nan,
    
                            "infrastructure_cost":
                                k * warehouse_operating_cost,
    
                            "total_cost":
                                np.nan
                        })
    
                return results

# ======================================================
# TEST THE OPTIMIZER

if __name__ == "__main__":

    # --------------------------------------------------
    # SAMPLE NEIGHBORHOODS
    points = np.array([
        [2, 3],
        [3, 4],
        [4, 3],
        [5, 5],
        [8, 8],
        [9, 7],
        [8, 6],
        [20, 20],
        [21, 19],
        [19, 21]
    ])

    # Daily orders
    demands = np.array([
        100,
        150,
        120,
        200,
        500,
        450,
        300,
        800,
        700,
        600
    ])

    # --------------------------------------------------
    # BASELINE

    baseline_warehouses = np.array([
        [5, 5],
        [12, 12],
        [18, 18]
    ])

    optimizer = GridPointOptimizer(
        n_warehouses=3,
        max_iterations=100,
        tolerance=1e-5,
        restarts=10,
        random_state=42
    )

    baseline = optimizer.evaluate_solution(
        points,
        demands,
        baseline_warehouses
    )

    # --------------------------------------------------
    # OPTIMIZATION

    result = optimizer.fit(
        points,
        demands
    )

    optimized = optimizer.evaluate_solution(
        points,
        demands,
        result["warehouses"]
    )

    # --------------------------------------------------
    # COMPARISON

    baseline_cost = (
        baseline["total_weighted_distance"]
    )

    optimized_cost = (
        optimized["total_weighted_distance"]
    )

    improvement = (
        (baseline_cost - optimized_cost)
        / baseline_cost
    ) * 100

    print("\n================================")
    print("       GRIDPOINT RESULTS")
    print("================================")

    print("\nBASELINE")
    print("--------------------------------")

    print(
        f"Weighted Delivery Distance: "
        f"{baseline_cost:.2f}"
    )

    print(
        f"Average Delivery Distance: "
        f"{baseline['average_distance']:.2f}"
    )

    print("\nOPTIMIZED")
    print("--------------------------------")

    print(
        f"Weighted Delivery Distance: "
        f"{optimized_cost:.2f}"
    )

    print(
        f"Average Delivery Distance: "
        f"{optimized['average_distance']:.2f}"
    )

    print(
        f"\nImprovement: "
        f"{improvement:.2f}%"
    )

    print(
        f"Optimization Iterations: "
        f"{result['iterations']}"
    )

    print(
        f"Random Restarts: "
        f"{optimizer.restarts}"
    )

    print("\nWAREHOUSE LOCATIONS")
    print("--------------------------------")

    for i, warehouse in enumerate(
        result["warehouses"]
    ):

        print(
            f"Warehouse {i + 1}: "
            f"({warehouse[0]:.2f}, "
            f"{warehouse[1]:.2f})"
        )

    print("\nASSIGNMENTS")
    print("--------------------------------")

    for i, warehouse_id in enumerate(
        result["assignments"]
    ):

        print(
            f"Neighborhood {i + 1} "
            f"-> Warehouse {warehouse_id + 1}"
        )

    print("\n================================")
    print("\n\n================================")
    print("   CONSTRAINED TEST")
    print("================================")

    capacities = np.array([
        2500,
        2000,
        2000
    ])

    max_radius = 10

    constrained = optimizer.evaluate_constrained_solution(
        points,
        demands,
        result["warehouses"],
        capacities,
        max_radius
    )

    print(
        f"\nMaximum Service Radius: "
        f"{max_radius}"
    )

    print("\nWarehouse Capacities:")

    for i, capacity in enumerate(capacities):

        used = (
            capacity
            - constrained["remaining_capacity"][i]
        )

        print(
            f"Warehouse {i + 1}: "
            f"{used:.0f} / {capacity:.0f}"
        )

    print(
        f"\nUnassigned Demand: "
        f"{constrained['unassigned_demand']:.0f}"
    )

    print(
        f"Constrained Delivery Distance: "
        f"{constrained['total_cost']:.2f}"
    )

    print(
        f"Constrained Average Distance: "
        f"{constrained['average_distance']:.2f}"
    )

    print("\nAssignments:")

    for i, warehouse_id in enumerate(
        constrained["assignments"]
    ):

        if warehouse_id == -1:

            print(
                f"Neighborhood {i + 1} "
                f"-> UNASSIGNED"
            )

        else:

            print(
                f"Neighborhood {i + 1} "
                f"-> Warehouse "
                f"{warehouse_id + 1}"
            )
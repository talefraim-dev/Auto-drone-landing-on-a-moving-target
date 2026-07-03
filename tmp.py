from typing import List, Tuple, Set
from scipy.optimize import linprog


Supplier = Tuple[str, float]
Consumer = Tuple[str, float]
Edge = Tuple[str, str]
Allocation = Tuple[str, str, float]


def find_valid_edges(
    suppliers: List[Supplier],
    consumers: List[Consumer],
    candidate_edges: List[Edge]
) -> List[Edge]:
    """
    Validates the candidate edges.

    An edge is valid when:
    1. Its first endpoint is an existing supplier.
    2. Its second endpoint is an existing consumer.
    3. Both supplier and consumer have a positive capacity.

    The actual capacity restrictions are enforced later by the
    linear-programming constraints.
    """

    supplier_capacity = dict(suppliers)
    consumer_capacity = dict(consumers)

    valid_edges = []
    seen_edges: Set[Edge] = set()

    for supplier, consumer in candidate_edges:
        edge = (supplier, consumer)

        if edge in seen_edges:
            continue

        if supplier not in supplier_capacity:
            raise ValueError(
                f"Unknown supplier in edge {edge}: {supplier}"
            )

        if consumer not in consumer_capacity:
            raise ValueError(
                f"Unknown consumer in edge {edge}: {consumer}"
            )

        if supplier_capacity[supplier] <= 0:
            continue

        if consumer_capacity[consumer] <= 0:
            continue

        valid_edges.append(edge)
        seen_edges.add(edge)

    return valid_edges


def optimal_supply_matching(
    suppliers: List[Supplier],
    consumers: List[Consumer],
    edges: List[Edge]
) -> List[Allocation]:
    """
    Finds an optimal allocation between suppliers and consumers
    using linear programming.

    Returns:
        A list of tuples:
        (supplier, consumer, supplied_amount)
    """

    if not suppliers or not consumers or not edges:
        return []

    supplier_capacity = dict(suppliers)
    consumer_capacity = dict(consumers)

    if len(supplier_capacity) != len(suppliers):
        raise ValueError("Supplier names must be unique.")

    if len(consumer_capacity) != len(consumers):
        raise ValueError("Consumer names must be unique.")

    for supplier, capacity in suppliers:
        if capacity < 0:
            raise ValueError(
                f"Supplier {supplier} has a negative capacity."
            )

    for consumer, capacity in consumers:
        if capacity < 0:
            raise ValueError(
                f"Consumer {consumer} has a negative capacity."
            )

    valid_edges = find_valid_edges(
        suppliers,
        consumers,
        edges
    )

    if not valid_edges:
        return []

    number_of_variables = len(valid_edges)

    # linprog minimizes a function.
    # Therefore, maximizing sum(x) is equivalent to minimizing -sum(x).
    objective = [-1.0] * number_of_variables

    constraint_matrix = []
    constraint_bounds = []

    # Supplier constraints:
    # Sum of all quantities leaving supplier u <= supplier capacity.
    for supplier, capacity in suppliers:
        row = []

        for edge_supplier, _ in valid_edges:
            if edge_supplier == supplier:
                row.append(1.0)
            else:
                row.append(0.0)

        constraint_matrix.append(row)
        constraint_bounds.append(capacity)

    # Consumer constraints:
    # Sum of all quantities entering consumer v <= consumer capacity.
    for consumer, capacity in consumers:
        row = []

        for _, edge_consumer in valid_edges:
            if edge_consumer == consumer:
                row.append(1.0)
            else:
                row.append(0.0)

        constraint_matrix.append(row)
        constraint_bounds.append(capacity)

    # Every x(u, v) must be non-negative.
    variable_bounds = [(0.0, None)] * number_of_variables

    result = linprog(
        c=objective,
        A_ub=constraint_matrix,
        b_ub=constraint_bounds,
        bounds=variable_bounds,
        method="highs"
    )

    if not result.success:
        raise RuntimeError(
            f"Failed to find an optimal solution: {result.message}"
        )

    solution = []
    epsilon = 1e-9

    for edge, supplied_amount in zip(valid_edges, result.x):
        if supplied_amount > epsilon:
            supplier, consumer = edge

            solution.append(
                (
                    supplier,
                    consumer,
                    round(float(supplied_amount), 6)
                )
            )

    return solution


def test_optimal_supply_matching() -> None:
    """
    Creates the graph shown in the question,
    runs the algorithm and prints the optimal solution.
    """

    suppliers = [
        ("u1", 3.5),
        ("u2", 2.0),
        ("u3", 4.0),
        ("u4", 1.5)
    ]

    consumers = [
        ("v1", 2.5),
        ("v2", 3.0),
        ("v3", 2.0),
        ("v4", 1.5),
        ("v5", 2.0)
    ]

    candidate_edges = [
        ("u1", "v1"),
        ("u1", "v2"),
        ("u1", "v4"),
        ("u2", "v2"),
        ("u2", "v5"),
        ("u3", "v3"),
        ("u3", "v4"),
        ("u4", "v2"),
        ("u4", "v5")
    ]

    valid_edges = find_valid_edges(
        suppliers,
        consumers,
        candidate_edges
    )

    print("Valid edges:")
    print(set(valid_edges))
    print()

    optimal_solution = optimal_supply_matching(
        suppliers,
        consumers,
        valid_edges
    )

    print("Optimal solution:")
    print(set(optimal_solution))
    print()

    total_supplied = sum(
        supplied_amount
        for _, _, supplied_amount in optimal_solution
    )

    total_consumed = sum(
        supplied_amount
        for _, _, supplied_amount in optimal_solution
    )

    difference = total_supplied - total_consumed

    print(f"Total supplied by all suppliers: {total_supplied}")
    print(f"Total consumed by all consumers: {total_consumed}")
    print(f"Difference between supplied and consumed: {difference}")


if __name__ == "__main__":
    test_optimal_supply_matching()
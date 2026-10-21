import math

def compute_partial_derivatives(
    func_name: str,
    point: tuple[float, ...]
) -> tuple[float, ...]:

    dimensions = {
        "poly2d": 2,
        "exp_sum": 2,
        "product_sin": 2,
        "poly3d": 3,
        "squared_error": 2,
    }

    if func_name not in dimensions:
        raise ValueError(f"Fonction inconnue : {func_name}")

    if len(point) != dimensions[func_name]:
        raise ValueError(
            f"{func_name} exige {dimensions[func_name]} coordonnées."
        )

    x, y = float(point[0]), float(point[1])

    match func_name:
        case "poly2d":
            return (2 * x * y + y ** 2, x ** 2 + 2 * x * y)

        case "exp_sum":
            valeur = math.exp(x + y)
            return (valeur, valeur)

        case "product_sin":
            return (math.sin(y), x * math.cos(y))

        case "poly3d":
            z = float(point[2])
            return (2 * x * y, x ** 2 + z ** 2, 2 * y * z)

        case "squared_error":
            return (2 * (x - y), 2 * (y - x))
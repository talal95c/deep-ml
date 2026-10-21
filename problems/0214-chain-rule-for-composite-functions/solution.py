import math

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """Calcule la dérivée de fonctions composées de droite à gauche."""

    def simple_der(function: str, x: float) -> float:
        match function:
            case "square":
                return 2 * x
            case "sin":
                return math.cos(x)
            case "exp":
                return math.exp(x)
            case "log":
                if x <= 0:
                    raise ValueError("Le logarithme exige x > 0.")
                return 1 / x
            case _:
                raise ValueError(f"Fonction inconnue : {function}")

    def simple_cal(function: str, x: float) -> float:
        match function:
            case "square":
                return x ** 2
            case "sin":
                return math.sin(x)
            case "exp":
                return math.exp(x)
            case "log":
                if x <= 0:
                    raise ValueError("Le logarithme exige x > 0.")
                return math.log(x)
            case _:
                raise ValueError(f"Fonction inconnue : {function}")

    gradient = 1.0
    valeur = x

    for function in reversed(functions):
        gradient *= simple_der(function, valeur)
        valeur = simple_cal(function, valeur)

    return gradient
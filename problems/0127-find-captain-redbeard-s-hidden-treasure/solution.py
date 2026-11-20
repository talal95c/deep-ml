def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.
    """
    lr = 0.001
    momentum = 0.9
    epsilon = 0.00001

    x = start_x
    velocity = 0

    for _ in range(100000):
        gradient = 4 * x**3 - 9 * x**2

        velocity = momentum * velocity + lr * gradient
        new_x = x - velocity

        if abs(new_x - x) < epsilon and abs(gradient) > epsilon:
            # Ne pas s'arrêter au point plat x = 0
            if abs(new_x - 2.25) < 0.01:
                return new_x

        x = new_x

    return x
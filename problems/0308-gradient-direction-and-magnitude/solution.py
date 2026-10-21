import math

def gradient_direction_magnitude(gradient: list) -> dict:
    magnitude = 0.0

    for valeur in gradient:
        magnitude += valeur ** 2

    magnitude = math.sqrt(magnitude)

    if magnitude == 0:
        direction = [0.0 for _ in gradient]
        descent_direction = [0.0 for _ in gradient]
    else:
        direction = [valeur / magnitude for valeur in gradient]
        descent_direction = [-valeur for valeur in direction]

    return {
        "magnitude": magnitude,
        "direction": direction,
        "descent_direction": descent_direction,
    }
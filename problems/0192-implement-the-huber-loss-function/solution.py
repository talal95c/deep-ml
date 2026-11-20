def huber_loss(y_true, y_pred, delta=1.0):

    if isinstance(y_true, (int, float)):
        error = abs(y_true - y_pred)

        if error > delta:
            return delta * (error - delta / 2)
        else:
            return 0.5 * error**2

    if len(y_true) != len(y_pred):
        return None

    total = 0

    for i in range(len(y_true)):
        error = abs(y_true[i] - y_pred[i])

        if error > delta:
            total += delta * (error - delta / 2)
        else:
            total += 0.5 * error**2

    return total / len(y_true)
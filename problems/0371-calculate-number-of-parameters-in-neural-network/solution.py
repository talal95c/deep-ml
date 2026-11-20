
def calculate_parameters(layers: list[dict]) -> int:
    total = 0

    for layer in layers:
        bias = layer.get("bias", True)

        if layer["type"] == "dense":
            total += layer["input_size"] * layer["output_size"]

            if bias:
                total += layer["output_size"]

        elif layer["type"] == "conv2d":
            total += (
                layer["in_channels"]
                * layer["out_channels"]
                * layer["kernel_size"]**2
            )

            if bias:
                total += layer["out_channels"]

    return total

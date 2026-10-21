"""Strided logical parameter views over unique optimizer storage blocks."""
import itertools


def positions(view):
    shape = view["shape"]
    strides = view["strides"]
    offset = int(view.get("offset", 0))
    return [offset + sum(i * s for i, s in zip(coord, strides))
            for coord in itertools.product(*(range(n) for n in shape))]


def gather(blocks, view):
    weight = blocks[int(view["block"])]["weight"]
    return [float(weight[i]) for i in positions(view)]


def scatter_add(gradients, view, values):
    target = gradients[int(view["block"])]
    indices = positions(view)
    if len(indices) != len(values):
        raise ValueError("gradient and logical view have different sizes")
    for index, value in zip(indices, values):
        target[index] += float(value)


def zeros(blocks):
    return [[0.0] * len(block["weight"]) for block in blocks]


def trainable_values(blocks, gradients):
    return [v for block, grad in zip(blocks, gradients)
            if bool(block.get("trainable", True)) for v in grad]

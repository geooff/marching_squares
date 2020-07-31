from opensimplex import OpenSimplex
import numpy as np
from scipy.interpolate import interp1d
from PIL import Image
import imageio
from math import ceil, floor
import datetime

noise = OpenSimplex()

image_size = (100, 100)  # The resolution of the final image
gif_length = 100

feature_size = 25
resolution = 2  # This number should be odd
noise_speed = 0.05
record = True

grid = tuple(int(x / resolution) for x in image_size)


def contour_lookup(resolution):
    resolution = resolution - 1
    return dict(
        [
            (0, None),
            (1, [[(0, ceil(resolution / 2)), (ceil(resolution / 2), resolution)]]),
            (
                2,
                [
                    [
                        (ceil(resolution / 2), resolution),
                        (resolution, ceil(resolution / 2)),
                    ]
                ],
            ),
            (3, [[(0, ceil(resolution / 2)), (resolution, ceil(resolution / 2))]]),
            (4, [[(ceil(resolution / 2), 0), (resolution, ceil(resolution / 2))]]),
            (
                5,
                [
                    [(ceil(resolution / 2), 0), (0, ceil(resolution / 2))],
                    [
                        (resolution, ceil(resolution / 2)),
                        (ceil(resolution / 2), resolution),
                    ],
                ],
            ),
            (6, [[(ceil(resolution / 2), 0), (ceil(resolution / 2), resolution)]]),
            (7, [[(ceil(resolution / 2), 0), (0, ceil(resolution / 2))]]),
            (8, [[(ceil(resolution / 2), 0), (0, ceil(resolution / 2))]]),
            (9, [[(ceil(resolution / 2), 0), (ceil(resolution / 2), resolution)]]),
            (
                10,
                [
                    [(ceil(resolution / 2), 0), (resolution, ceil(resolution / 2))],
                    [(ceil(resolution / 2), resolution), (0, ceil(resolution / 2))],
                ],
            ),
            (11, [[(ceil(resolution / 2), 0), (resolution, ceil(resolution / 2))]]),
            (
                12,
                [
                    [
                        (0, ceil(resolution / 2)),
                        (ceil(resolution / 2), ceil(resolution / 2)),
                    ]
                ],
            ),
            (
                13,
                [
                    [
                        (resolution, ceil(resolution / 2)),
                        (ceil(resolution / 2), resolution),
                    ]
                ],
            ),
            (14, [[(0, ceil(resolution / 2)), (ceil(resolution / 2), resolution)]]),
            (15, None),
        ]
    )


def make_square(binary_string, resolution):
    lookup = contour_lookup(resolution)
    output = np.zeros((resolution, resolution))
    lines = lookup[int(binary_string, 2)]

    if lines is None:
        return output

    for points in lines:
        x_coords, y_coords = zip(*points)
        x_range = sorted(x_coords)

        if x_coords[0] == x_coords[1]:  # case one: vertical line
            output[:, y_coords[0]] = 1
            return output

        elif y_coords[0] == y_coords[1]:  # case two: horizontal line
            output[x_coords[0]] = 1
            return output

        else:
            x_new = list(range(x_range[0], (x_range[1]) + 1))
            f = interp1d(x_coords, y_coords, kind="linear")
            y_new = [int(y) for y in f(x_new)]

            for y, x in zip(x_new, y_new):
                output[x, y] = 1

    return output


# Refactor this to use numpy array not list of lists
def make_binary_noise(step_number, grid, resolution, feature_size):
    def make_noise(x, y, step_number, resolution, feature_size):
        # Return Simplex Noise of elements mapped to grey-scale
        return noise.noise3d(
            (x * resolution) / feature_size,
            (y * resolution) / feature_size,
            step_number,
        )

    output = []
    for x in range(grid[0]):
        tmp = []
        for y in range(grid[1]):
            value = make_noise(x, y, step_number, resolution, feature_size)
            colour = int(ceil(value))
            tmp.append(colour)
        output.append(tmp)
    return output
    # return [
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # ]


# Refactor this to use numpy array not list of lists
def binary_to_nibble(data, resolution, feature_size):
    output = []
    for x_idx, row in enumerate(data):
        if x_idx == len(data) - 1:
            continue
        tmp = []
        for y_idx, value in enumerate(row):
            if y_idx == len(row) - 1:
                continue
            tmp.append(
                str(data[x_idx][y_idx])
                + str(data[x_idx + 1][y_idx])
                + str(data[x_idx + 1][y_idx + 1])
                + str(data[x_idx][y_idx + 1])
            )
        output.append(tmp)
    return output


def marching_squares_step(step_number, resolution, feature_size, image_size):
    def _preview_image(image_size, data, resolution):
        im = Image.new("1", image_size)
        for x_idx, row in enumerate(data):
            for y_idx, value in enumerate(row):
                im.putpixel((x_idx, y_idx), int(value))
        im.show()

    def return_image(image_size, data, resolution):
        im = Image.new("1", image_size)
        for x_idx, row in enumerate(data):
            for y_idx, value in enumerate(row):
                im.putpixel((x_idx, y_idx), int(value))
        return im

    surface = np.zeros(image_size)
    binary_noise = make_binary_noise(step_number, grid, resolution, feature_size)
    nibble_noise = binary_to_nibble(binary_noise, resolution, feature_size)

    for x_idx, row in enumerate(nibble_noise):
        for y_idx, value in enumerate(row):
            surface[
                x_idx * resolution : (x_idx + 1) * resolution,
                y_idx * resolution : (y_idx + 1) * resolution,
            ] = make_square(value, resolution)

    # _preview_image(image_size, surface, resolution)
    return return_image(image_size, surface, resolution)


if record:
    # Generate frame names
    names = [f"tmp/{i}.gif" for i in range(gif_length)]
    images = []

    for idx, name in enumerate(names):
        im = marching_squares_step(
            idx * noise_speed, resolution, feature_size, image_size
        )
        im.save(name)

    for filename in names:
        images.append(imageio.imread(filename))
    imageio.mimsave(f"size-{feature_size}_speed-{noise_speed}.gif", images)

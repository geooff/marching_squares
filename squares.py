from datetime import datetime
from typing import Tuple

# Import math libs
from opensimplex import OpenSimplex
import numpy as np
from math import ceil
from scipy.interpolate import interp1d

# Import image manipulation libs
from PIL import Image
import imageio

image_size = (400, 400)  # The resolution of the final image
gif_length = 10

resolution = 5
feature_size = 50
noise_speed = 0.05

grid = tuple(int(x / resolution) for x in image_size)


# fmt: off
contour_lookup =  dict(
    [
        (0, None),
        (1, [[(0, ceil((resolution - 1) / 2)), (ceil((resolution - 1) / 2), (resolution - 1))]]),
        (2, [[(ceil((resolution - 1) / 2), (resolution - 1)),((resolution - 1), ceil((resolution - 1) / 2))]]),
        (3, [[(0, ceil((resolution - 1) / 2)), ((resolution - 1), ceil((resolution - 1) / 2))]]),
        (4, [[(ceil((resolution - 1) / 2), 0), ((resolution - 1), ceil((resolution - 1) / 2))]]),
        (5, [[(ceil((resolution - 1) / 2), 0), (0, ceil((resolution - 1) / 2))],[((resolution - 1), ceil((resolution - 1) / 2)),(ceil((resolution - 1) / 2), (resolution - 1))]]),
        (6, [[(ceil((resolution - 1) / 2), 0), (ceil((resolution - 1) / 2), (resolution - 1))]]),
        (7, [[(ceil((resolution - 1) / 2), 0), (0, ceil((resolution - 1) / 2))]]),
        (8, [[(ceil((resolution - 1) / 2), 0), (0, ceil((resolution - 1) / 2))]]),
        (9, [[(ceil((resolution - 1) / 2), 0), (ceil((resolution - 1) / 2), (resolution - 1))]]),
        (10, [[(ceil((resolution - 1) / 2), 0), ((resolution - 1), ceil((resolution - 1) / 2))],[(ceil((resolution - 1) / 2), (resolution - 1)), (0, ceil((resolution - 1) / 2))]]),
        (11, [[(ceil((resolution - 1) / 2), 0), ((resolution - 1), ceil((resolution - 1) / 2))]]),
        (12, [[(0, ceil((resolution - 1) / 2)),(ceil((resolution - 1) / 2), ceil((resolution - 1) / 2))]]),
        (13, [[((resolution - 1), ceil((resolution - 1) / 2)),(ceil((resolution - 1) / 2), (resolution - 1))]]),
        (14, [[(0, ceil((resolution - 1) / 2)), (ceil((resolution - 1) / 2), (resolution - 1))]]),
        (15, None),
        ]
)
# fmt: on


def make_binary_noise(
    step_number: int,
    grid: Tuple[int, int],
    resolution: int,
    feature_size: int,
    noise_speed: float,
):
    """Return an array of binary noise generated by Open Simplex Noise

    Args:
        step_number (int): The frame of the output to generate (Feed to noise function as 3rd dimension)
        grid (Tuple[int, int]): Two dimensional grid used for noise generation
        resolution (int): Scaling factor to be used in converting grid to image
        feature_size (int): Scaling factor used for (x,y) scaling in noise generation
        noise_speed (float): Scaling factor used for z-scaling in noise generation
    """

    def _make_noise(x, y, step_number, resolution, feature_size, noise_speed):
        # Return Simplex Noise for given coordinates. Spoof time dimension using step_number as the z-dimension
        return noise.noise3d(
            (x * resolution) / feature_size,
            (y * resolution) / feature_size,
            step_number * noise_speed,
        )

    # Init our OpenSimplex class
    noise = OpenSimplex()

    # Generate blank (all zeros) grid of size specified
    output = np.zeros(grid, dtype=int)

    # Iterate through the grid we just created using numpys built in fxn
    for ix, iy in np.ndindex(output.shape):
        # Generate noise using our noise function
        value = _make_noise(ix, iy, step_number, resolution, feature_size, noise_speed)
        # Noise ranges from (-1 -> 1), take the ceiling to normalize to values of [0,1]
        output[ix, iy] = int(ceil(value))
    return output


def make_square(binary_string: str, resolution: int):
    """Generate Isoband given binary string

    Args:
        binary_string (str): Four character binary string to be converted to isoband
        resolution (int): Scaling factor to be used in converting grid to image

    Returns:
        np.array: Array containing output binary pixel data
    """
    # Generate empty output array
    output = np.zeros((resolution, resolution))

    # Convert our binary string into the decimal isoband
    isoband = contour_lookup[int(binary_string, 2)]

    # Return empty grid if isoband is undefined or blank
    if isoband is None:
        return output

    for points in isoband:
        x_coords, y_coords = zip(*points)
        x_range = sorted(x_coords)

        if x_coords[0] == x_coords[1]:  # case one: vertical line
            output[:, x_coords[0]] = 1
            return output

        elif y_coords[0] == y_coords[1]:  # case two: horizontal line
            output[y_coords[0]] = 1
            return output

        # TODO: Implement np.array.diagonal() to simplify this
        else:
            x_new = list(range(x_range[0], (x_range[1]) + 1))
            f = interp1d(x_coords, y_coords, kind="linear")
            y_new = [int(y) for y in f(x_new)]

            for y, x in zip(x_new, y_new):
                output[x, y] = 1

    return output


def binary_to_nibble(data, resolution):
    offset_grid = tuple(int(x - 1) for x in grid)
    output = np.zeros(offset_grid, dtype=np.dtype(("U", 4)))
    for ix, iy in np.ndindex(output.shape):

        if ix == len(data) - 1 or iy == len(data) - 1:
            continue

        output[ix, iy] = (
            str(data[ix][iy])
            + str(data[ix + 1][iy])
            + str(data[ix + 1][iy + 1])
            + str(data[ix][iy + 1])
        )
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
    binary_noise = make_binary_noise(
        step_number, grid, resolution, feature_size, noise_speed
    )
    nibble_noise = binary_to_nibble(binary_noise, resolution)

    for x_idx, row in enumerate(nibble_noise):
        for y_idx, value in enumerate(row):
            surface[
                x_idx * resolution : (x_idx + 1) * resolution,
                y_idx * resolution : (y_idx + 1) * resolution,
            ] = make_square(value, resolution)

    return return_image(image_size, surface, resolution)


# Start timer
start = datetime.now()

# Generate frame names
names = [f"tmp/{i}.gif" for i in range(gif_length)]
images = []

for idx, name in enumerate(names):
    im = marching_squares_step(idx, resolution, feature_size, image_size)
    im.save(name)

for filename in names:
    images.append(imageio.imread(filename))
imageio.mimsave(f"size-{feature_size}_speed-{noise_speed}.gif", images)

end = datetime.now()
print(f"Finished execution, ran for {str(end-start)}")

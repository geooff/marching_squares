from datetime import datetime

# Import math libs
from opensimplex import OpenSimplex
import numpy as np
from math import ceil
from scipy.interpolate import interp1d

# Import image manipulation libs
from PIL import Image
import imageio

# Import argparse for terminal input parsing
import argparse


# fmt: off
def get_contour_lookup(resolution):
    return dict(
        [
            (0, None),
            (1, [[(ceil((resolution - 1) / 2), 0), ((resolution - 1), ceil((resolution - 1) / 2))]]),
            (2, [[(ceil((resolution - 1) / 2), (resolution - 1)),((resolution - 1), ceil((resolution - 1) / 2))]]),
            (3, [[(0, ceil((resolution - 1) / 2)), (resolution, ceil((resolution - 1) / 2))]]),
            (4, [[(0, ceil((resolution - 1) / 2)), (ceil((resolution - 1) / 2), (resolution - 1))]]),
            (5, [[(ceil((resolution - 1) / 2), 0), (0, ceil((resolution - 1) / 2))],[((resolution - 1), ceil((resolution - 1) / 2)),(ceil((resolution - 1) / 2), (resolution - 1))]]),
            (6, [[(ceil((resolution - 1) / 2), 0), (ceil((resolution - 1) / 2), resolution)]]),
            (7, [[(ceil((resolution - 1) / 2), 0), (0, ceil((resolution - 1) / 2))]]),
            (8, [[(ceil((resolution - 1) / 2), 0), (0, ceil((resolution - 1) / 2))]]),
            (9, [[(ceil((resolution - 1) / 2), 0), (ceil((resolution - 1) / 2), resolution)]]),
            (10, [[(ceil((resolution - 1) / 2), 0), ((resolution - 1), ceil((resolution - 1) / 2))],[(ceil((resolution - 1) / 2), (resolution - 1)), (0, ceil((resolution - 1) / 2))]]),
            (11, [[(0, ceil((resolution - 1) / 2)), (ceil((resolution - 1) / 2), (resolution - 1))]]),
            (12, [[(0, ceil((resolution - 1) / 2)),(resolution, ceil((resolution - 1) / 2))]]),
            (13, [[((resolution - 1), ceil((resolution - 1) / 2)),(ceil((resolution - 1) / 2), (resolution - 1))]]),
            (14, [[(ceil((resolution - 1) / 2), 0), ((resolution - 1), ceil((resolution - 1) / 2))]]),
            (15, None),
            ]
    )
# fmt: on


def make_binary_noise(
    step_number,
    grid,
    resolution,
    feature_size,
    noise_speed,
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


def make_square(binary_string, resolution):
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
    isoband_num = int(binary_string, 2)
    isoband = contour_lookup[isoband_num]

    # Return empty grid if isoband is undefined or blank
    if isoband is None:
        return output

    for points in isoband:
        x_coords, y_coords = zip(*points)
        x_range = sorted(x_coords)

        if isoband_num in (6, 9):  # case one: vertical line
            output[x_coords[0], y_coords[0] : y_coords[1]] = 1
            return output

        elif isoband_num in (3, 12):  # case two: horizontal line
            output[x_coords[0] : x_coords[1], y_coords[0]] = 1
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
    """Calculate the Isobands for a 2D field of binary noise

    Args:
        data (nd.array): Array of binary data
        resolution (int): Scaling factor used in converting grid to image

    Returns:
        str: binary nibble formatted as string containing output
    """
    # Calculate the grid size and create of the output array
    offset_grid = tuple(int(x - 1) for x in grid)
    output = np.zeros(offset_grid, dtype=np.dtype(("U", 4)))

    # Iterate over output array, calculate binary nibbles for each idx
    for ix, iy in np.ndindex(output.shape):

        # Skip the ends of rows and cols where isoband is undefined
        if ix == len(data) - 1 or iy == len(data) - 1:
            continue

        # Calculate nibble by using clockwise index
        output[ix, iy] = (
            str(data[ix][iy])
            + str(data[ix + 1][iy])
            + str(data[ix + 1][iy + 1])
            + str(data[ix][iy + 1])
        )
    return output


def marching_squares_step(
    step_number, resolution, feature_size, image_size, noise_speed, SHOW_GRID=True
):
    """Generate a frame of marching squares

    Args:
        step_number (int): The frame number of the animation
        resolution (int): Scaling factor used in converting grid to image
        feature_size (int): Scaling factor used for (x,y) scaling in noise generation
        image_size (Tuple[int, int]): Output image dimensions
        SHOW_GRID (bool, optional): Shows binary values at grid points. Defaults to True.
    """

    def return_image(image_size, data):
        """Generate an image given a grid of binary data

        Args:
            image_size (Tuple[int, int]): Output image dimensions
            data (np.array): Array containing output binary pixel data

        Returns:
            Image: Output image type to be saved
        """
        im = Image.new("1", image_size)
        for x_idx, row in enumerate(data):
            for y_idx, value in enumerate(row):
                im.putpixel((x_idx, y_idx), int(value))
        return im

    # Generate the output array
    surface = np.zeros(image_size)

    # Generate a grid of binary noise
    binary_noise = make_binary_noise(
        step_number, grid, resolution, feature_size, noise_speed
    )
    # Using our binary noise generate isobands
    nibble_noise = binary_to_nibble(binary_noise, resolution)

    # Iterate over grid of isobands converting isobands to pixels
    for x_idx, row in enumerate(nibble_noise):
        for y_idx, value in enumerate(row):
            surface[
                x_idx * resolution : (x_idx + 1) * resolution,
                y_idx * resolution : (y_idx + 1) * resolution,
            ] = make_square(value, resolution)

    # Print grid, useful for debugging
    if SHOW_GRID:
        for ix, iy in np.ndindex(binary_noise.shape):
            surface[ix * resolution, iy * resolution] = binary_noise[ix, iy]

    return return_image(image_size, surface)


# Start timer
start = datetime.now()

parser = argparse.ArgumentParser(description="Generate marching squares sequence.")
parser.add_argument(
    "--image_size",
    type=int,
    nargs=2,
    default=(400, 400),
    help="The resolution of the final image (x, y)",
)
parser.add_argument(
    "--length",
    type=int,
    help="The total frames (length) of the final gif",
)
parser.add_argument(
    "--resolution",
    type=int,
    default=5,
    help="Scaling factor to be used in converting grid to image",
)
parser.add_argument(
    "--feature_size",
    type=int,
    default=50,
    help="Scaling factor used for (x,y) scaling in noise generation",
)
parser.add_argument(
    "--noise_frequency",
    type=int,
    default=0.05,
    help="Scaling factor used for z-scaling in noise generation",
)
args = parser.parse_args()


grid = tuple(int(x / args.resolution) for x in args.image_size)
contour_lookup = get_contour_lookup(args.resolution)

# Generate frame names
names = [f"tmp/{i}.gif" for i in range(args.length)]
images = []

# Generate each frame of our output gif
for idx, name in enumerate(names):
    im = marching_squares_step(
        idx, args.resolution, args.feature_size, args.image_size, args.noise_frequency
    )
    im.save(name)

# Generate the gif itself from each frame
for filename in names:
    images.append(imageio.imread(filename))
imageio.mimsave(f"{datetime.now()}.gif", images)

# End timer and log
end = datetime.now()
print(f"Finished execution, ran for {str(end-start)}")

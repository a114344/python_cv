import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import filters


def compute_harris_response(img, sigma=3):
    """Compute Harris corner detector response function for each pixel in a

        grayscale image.
    """
    # derivatives
    imx = np.zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), (0, 1), imx)
    imy = np.zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), (1, 0), imy)

    # Compute components of Harris matrix
    Wxx = filters.gaussian_filter(imx * imx, sigma)
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)

    # determinant and trace
    Wdet = Wxx * Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return Wdet / Wtr


def get_harris_points(harrisimg, min_dist=10, threshold=0.1):
    """Return corners for a Harris response image. min_dist

       is the minumum number of pixels separating corners and image

       boundary.
    """

    # Find top corner candidates above threshold
    corner_threshold = harrisimg.max() * threshold
    harrisimg_t = (harrisimg > corner_threshold) * 1

    # Get coordinates of candidates
    coords = np.array(harrisimg_t.nonzero()).T

    # ...and their values
    candidate_values = [harrisimg[coord[0], coord[1]] for coord in coords]

    # Sort candidates
    index = np.argsort(candidate_values)

    # Store allowed point locations in an array
    allowed_locations = np.zeros(harrisimg.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    # Select the best points taking min_dist into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i, 0] - min_dist):
                              (coords[i, 0] + min_dist),
                              (coords[i, 1] - min_dist):
                              (coords[i, 1] + min_dist)] = 0
        return filtered_coords


def plot_harris_points(image, filtered_coords):
    """Plots corners found in image with get_harris_points function.
    """
    plt.figure()
    plt.gray()
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords],
             [p[0] for p in filtered_coords], '*')
    plt.axis('off')
    plt.show()

    return True


def get_descriptors(image, filtered_coords, width=5):
    """For each point return pixel values around the point using

       neighborhood of width 2 * width + 1. Functiona assumes

       points are extracted with min_distance > width.
    """
    desc = []
    for coords in filtered_coords:
        patch = image[coords[0] - width: coords[0] + width + 1,
                      coords[1] - width: coords[1] + width + 1].flatten()
        desc.append(patch)

    return desc


def match(desc1, desc2, threshold=0.5):
    """For each corner point descriptor in the first image, select its

       match to the second image using normalized cross-correlation.
    """
    n = len(desc1[0])

    # pair-wise distances
    d = -np.ones((len(desc1), len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
            ncc_value = sum(d1 * d2) / (n-1)
            if ncc_value > threshold:
                d[i, j] = ncc_value
    ndx = np.argsort(-d)
    matchscores = ndx[:, 0]

    return matchscores


def match_twosided(desc1, desc2, threshold=0.5):
    """Two sided symmetric version of the match function.
    """

    matches_12 = match(desc1, desc2, threshold)
    matches_21 = match(desc2, desc1, threshold)

    ndx_12 = np.where(matches_12 >= 0)[0]

    # Remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1

    return matches_12


def append_images(img1, img2):
    """Return an image that append sthe two images side by side
    """

    # Select the image wiht the fewest rows and fill in enough empty rows
    rows1, rows2 = (img1.shape[0], img2.shape[0])

    if rows1 < rows2:
        img1 = np.concatenate((img1,
                               np.zeros((rows2-rows1,
                                         img1.shape[1]))),
                              axis=0)
    elif rows1 > rows2:
        img2 = np.concatenate((img2,
                               np.zeros((rows1-rows2,
                                         img2.shape[1]))),
                              axis=0)
    # No else needed because if neither of the above evaluate to True
    # no filling is required.

    return np.concatenate((img1, img2), axis=1)


def plot_matches(img1, img2, locs1, locs2, matchscores, show_below=True):
    """Show a figure with lines joining the accepted matches.

       Input: img1, img2 (images as arrays),

              locs1, locs2 (feature locations),

              matchscores (as output from match() function),

              show_below (if images should be shown below matches).
    """

    img3 = append_images(img1, img2)

    if show_below:
        img3 = np.vstack((img3, img3))

    plt.imshow(img3)

    cols1 = img1.shape[1]
    for index, match in enumerate(matchscores):
        if match > 0:
            plt.plot([locs1[index][1],
                      locs2[match][1] + cols1],
                     [locs1[index][0],
                     locs2[match][0]], 'c')
    plt.axis = ('off')

    # finish me

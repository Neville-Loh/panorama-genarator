from typing import List, Optional
import os

from image_stiching.feature_descriptor.feature_descriptor import match_corner_by_ncc, reject_outlier_pairs
from image_stiching.harris_conrner_detection.harris import compute_harris_corner
from image_stiching.homography.homography import fit_transform_homography
from image_stiching.performance_evaulation.timer import measure_elapsed_time
from matplotlib import pyplot as plt

from image_stiching.util.save_object import load_object_at_location, save_object_at_location, get_file_name_from_path


@measure_elapsed_time
def stitch(
        left_px_array: List[List[int]],
        right_px_array: List[List[int]],
        n_corner: Optional[int] = 1000,
        alpha: Optional[float] = 0.04,
        gaussian_window_size: Optional[int] = 7,
        plot_harris_corner: Optional[bool] = False,
        feature_descriptor_patch_size: Optional[int] = 15,
        feature_descriptor_threshold: Optional[float] = 0.9,
        enable_outlier_rejection: Optional[bool] = True,
        outlier_rejection_m: Optional[float] = 1,
        plot_result: Optional[bool] = False,
        left_source_path: Optional[str] = None,
        right_source_path: Optional[str] = None,
        ransac_iteration_input: Optional[int] = 20000,
        ransac_threshold_input: Optional[float] = 1,
        cache_result: Optional[bool] = True,
        save_output_as_file: Optional[bool] = False,
) -> None:
    """
    Stitch two images together.

    parameters:
    -----------
    left_px_array: List[List[int]]
        The greyscale pixel array of the left image.
    right_px_array: List[List[int]]
        The greyscale pixel array of the right image.
    n_corner: Optional[int]
        The number of corners to detect in the left image, default is 1000.
    alpha: Optional[float]
        The alpha value for the Harris corner detector.
    gaussian_window_size: Optional[int]
        The size of the gaussian window for the Harris corner detector, default is 7.
    plot_harris_corner: Optional[bool]
        Whether to plot the detected corners.
    feature_descriptor_path_size: Optional[int]
        The size of the path for the feature descriptor, default is 15.
    feature_descriptor_threshold: Optional[float]
        The threshold for the feature descriptor, default is 0.9.
    enable_outlier_rejection: Optional[bool]
        Whether to enable outlier rejection, default is True.
    outlier_rejection_m: Optional[float]
        The standard deviation for the outlier rejection to include, default is 1.
    plot_result: Optional[bool]
        Whether to plot the result, default is False.

    returns:
    --------
    List[Pair]
        The list of pairs of the matched points.
    """

    def compute_pairs():
        left_corners = compute_harris_corner(left_px_array,
                                             n_corner=n_corner,
                                             alpha=alpha,
                                             gaussian_window_size=gaussian_window_size,
                                             plot_image=plot_harris_corner)

        right_corners = compute_harris_corner(right_px_array,
                                              n_corner=1000,
                                              alpha=0.04,
                                              gaussian_window_size=7,
                                              plot_image=False)

        # get the best matches for each corner in the left image
        return match_corner_by_ncc((left_px_array, left_corners),
                                   (right_px_array, right_corners),
                                   feature_descriptor_patch_size=feature_descriptor_patch_size,
                                   threshold=feature_descriptor_threshold)

    if cache_result:
        try:
            pairs = load_object_at_location(
                os.path.join(
                    ".",
                    "cache",
                    f"%s_%s_cache.pkl" % (
                        get_file_name_from_path(left_source_path),
                        get_file_name_from_path(right_source_path)
                    )
                )
            )
        except FileNotFoundError:

            # compute the harris corner
            pairs = compute_pairs()

            # Save the result
            save_object_at_location(
                os.path.join(
                    ".",
                    "cache",
                    f"%s_%s_cache.pkl" % (
                        get_file_name_from_path(left_source_path),
                        get_file_name_from_path(right_source_path)
                    )
                ),
                pairs
            )

    else:
        pairs = compute_pairs()

    # remove the outliers
    if enable_outlier_rejection:
        pairs = reject_outlier_pairs(pairs, width_offset=len(left_px_array[0]), m=outlier_rejection_m)

    # compute homography
    image = fit_transform_homography(list(pairs),
                                     source_left_image_path=left_source_path,
                                     source_right_image_path=right_source_path,
                                     ransac_iteration=ransac_iteration_input,
                                     ransac_threshold=ransac_threshold_input)

    plt.imshow(image)
    plt.show()

    if save_output_as_file:
        plt.imsave("output.png", image)
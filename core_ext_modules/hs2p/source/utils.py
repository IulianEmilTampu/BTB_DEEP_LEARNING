import cv2
import time
import h5py
import numpy as np
import pandas as pd
import seaborn as sns

from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, Dict, List


def compute_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def initialize_df(
    slides,
    masks,
    spacings,
    seg_params,
    filter_params,
    vis_params,
    patch_params,
    use_heatmap_args=False,
    slide_ids:list=None #IET
):
    """
    initiate a pandas df describing a list of slides to process
    args:
            slides (df or list): list of slide filepath
                    if df, these paths assumed to be stored under the 'slide_path' column
            masks (list): list of slides' segmentation masks filepath
            spacings (list): list of slides' spacing at level 0
            seg_params (dict): segmentation paramters
            filter_params (dict): filter parameters
            vis_params (dict): visualization paramters
            patch_params (dict): patching paramters
            use_heatmap_args (bool): whether to include heatmap arguments such as ROI coordinates
    """
    total = len(slides)
    if isinstance(slides, pd.DataFrame):
        slide_ids = list(slides.slide_id.values)
        slide_paths = list(slides.slide_path.values)
        if "segmentation_mask_path" in slides.columns:
            mask_paths = list(slides.segmentation_mask_path.values)
        else:
            mask_paths = masks.copy()
        if "spacing" in slides.columns:
            slide_spacings = list(slides.spacing.values)
        else:
            slide_spacings = spacings.copy()
    else:
        slide_ids = [Path(s).stem for s in slides] if not slide_ids else slide_ids # IET
        slide_paths = slides.copy()
        mask_paths = masks.copy()
        slide_spacings = spacings.copy()
    default_df_dict = {
        "slide_id": slide_ids,
        "slide_path": slide_paths,
    }
    if len(mask_paths) > 0:
        default_df_dict.update({"segmentation_mask_path": mask_paths})
    if len(slide_spacings) > 0:
        default_df_dict.update({"spacing": slide_spacings})

    # initiate empty labels in case not provided
    if use_heatmap_args:
        default_df_dict.update({"label": np.full((total), -1)})

    default_df_dict.update(
        {
            "process": np.full((total), 1, dtype=np.uint8),
            "status": np.full((total), "tbp"),
            "has_patches": np.full((total), "tbd"),
            # seg params
            "seg_level": np.full((total), int(seg_params["seg_level"]), dtype=np.int8),
            "sthresh": np.full((total), int(seg_params["sthresh"]), dtype=np.uint8),
            "mthresh": np.full((total), int(seg_params["mthresh"]), dtype=np.uint8),
            "close": np.full((total), int(seg_params["close"]), dtype=np.uint32),
            "use_otsu": np.full((total), bool(seg_params["use_otsu"]), dtype=bool),
            # filter params
            "a_t": np.full((total), int(filter_params["a_t"]), dtype=np.float32),
            "a_h": np.full((total), int(filter_params["a_h"]), dtype=np.float32),
            "max_n_holes": np.full(
                (total), int(filter_params["max_n_holes"]), dtype=np.uint32
            ),
            # vis params
            "vis_level": np.full((total), int(vis_params["vis_level"]), dtype=np.int8),
            "line_thickness": np.full(
                (total), int(vis_params["line_thickness"]), dtype=np.uint32
            ),
            # patching params
            "use_padding": np.full(
                (total), bool(patch_params["use_padding"]), dtype=bool
            ),
            "contour_fn": np.full((total), patch_params["contour_fn"]),
            "tissue_thresh": np.full((total), patch_params["tissue_thresh"]),
        }
    )

    if use_heatmap_args:
        # initiate empty x,y coordinates in case not provided
        default_df_dict.update(
            {
                "x1": np.empty((total)).fill(np.NaN),
                "x2": np.empty((total)).fill(np.NaN),
                "y1": np.empty((total)).fill(np.NaN),
                "y2": np.empty((total)).fill(np.NaN),
            }
        )

    if isinstance(slides, pd.DataFrame):
        temp_copy = pd.DataFrame(
            default_df_dict
        )  # temporary dataframe w/ default params
        # find key in provided df
        # if exist, fill empty fields w/ default values, else, insert the default values as a new column
        for key in default_df_dict.keys():
            if key in slides.columns:
                mask = slides[key].isna()
                slides.loc[mask, key] = temp_copy.loc[mask, key]
            else:
                slides.insert(len(slides.columns), key, default_df_dict[key])
    else:
        slides = pd.DataFrame(default_df_dict)

    return slides


def save_hdf5(output_path, asset_dict, attr_dict=None, mode="a"):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = file.create_dataset(
                key,
                shape=data_shape,
                maxshape=maxshape,
                chunks=chunk_shape,
                dtype=data_type,
            )
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0] :] = val
    file.close()
    return output_path


def find_common_spacings(spacings_1, spacings_2, tolerance: float = 0.05):
    common_spacings = []
    for s1 in spacings_1:
        for s2 in spacings_2:
            # check how far appart these two spacings are
            if abs(s1 - s2) / s1 <= tolerance:
                common_spacings.append((s1, s2))
    return common_spacings


def save_patch(
    wsi,
    spacing: float,
    save_dir,
    asset_dict,
    attr_dict=None,
    fmt="png",
    save_patches_in_common_dir: bool = False,
):
    coords = asset_dict["coords"]
    patch_size = attr_dict["coords"]["patch_size"]
    wsi_name = attr_dict["coords"]["wsi_name"]

    npatch = len(coords)
    start_time = time.time()

    for coord in coords:
        x, y = coord
        patch = wsi.get_patch(
            x, y, patch_size, patch_size, spacing=spacing, center=False
        )
        pil_patch = Image.fromarray(patch).convert("RGB")
        save_name = f"{int(x)}_{int(y)}.{fmt}"
        if save_patches_in_common_dir:
            save_name = f"{wsi_name}_{save_name}"
        save_path = Path(save_dir, save_name)
        pil_patch.save(save_path)

    end_time = time.time()
    patch_saving_mins, patch_saving_secs = compute_time(start_time, end_time)
    return npatch, patch_saving_mins, patch_saving_secs


def initialize_hdf5_bag(first_patch, save_coord=False):
    (
        x,
        y,
        cont_idx,
        patch_size,
        patch_level,
        downsample,
        downsampled_level_dim,
        level_dimensions,
        img_patch,
        name,
        save_path,
    ) = tuple(first_patch.values())
    file_path = Path(save_path, f"{name}.h5")
    file = h5py.File(file_path, "w")
    img_patch = np.array(img_patch)[np.newaxis, ...]
    dtype = img_patch.dtype

    # Initialize a resizable dataset to hold the output
    img_shape = img_patch.shape
    maxshape = (None,) + img_shape[
        1:
    ]  # maximum dimensions up to which dataset maybe resized (None means unlimited)
    dset = file.create_dataset(
        "imgs", shape=img_shape, maxshape=maxshape, chunks=img_shape, dtype=dtype
    )

    dset[:] = img_patch
    dset.attrs["patch_size"] = patch_size
    dset.attrs["patch_level"] = patch_level
    dset.attrs["wsi_name"] = name
    dset.attrs["downsample"] = downsample
    dset.attrs["level_dimensions"] = level_dimensions
    dset.attrs["downsampled_level_dim"] = downsampled_level_dim

    if save_coord:
        coord_dset = file.create_dataset(
            "coords", shape=(1, 2), maxshape=(None, 2), chunks=(1, 2), dtype=np.int32
        )
        coord_dset[:] = (x, y)

    file.close()
    return file_path


def overlay_mask_on_slide(
    wsi_object,
    mask_object,
    vis_level: int,
    pixel_mapping: Dict[str, int],
    color_mapping: Optional[Dict[str, List[int]]] = None,
    alpha: float = 0.5,
    downsample: int = -1,
):
    """
    Show a mask overlayed on a slide
    """

    x_mask = mask_object.level_dimensions[0][0]
    mask_min_level = int(
        np.argmin([abs(x_wsi - x_mask) for x_wsi, _ in wsi_object.level_dimensions])
    )

    assert x_mask == wsi_object.level_dimensions[mask_min_level][0]

    if vis_level < 0:
        if len(wsi_object.level_dimensions) == 1:
            vis_level = 0
            assert mask_min_level == 0
        else:
            vis_level = wsi_object.get_best_level_for_downsample_custom(downsample)
            vis_level = max(vis_level, mask_min_level)
    else:
        vis_level = max(vis_level, mask_min_level)

    mask_vis_level = vis_level - mask_min_level

    if mask_vis_level >= len(mask_object.spacings):
        # mask_vis_level doens't exist, need to get last available level
        mask_vis_level = len(mask_object.spacings) - 1
        x_mask_vis_level = mask_object.level_dimensions[mask_vis_level][0]
        vis_level = int(
            np.argmin(
                [
                    abs(x_wsi - x_mask_vis_level)
                    for x_wsi, _ in wsi_object.level_dimensions
                ]
            )
        )
        assert x_mask_vis_level == wsi_object.level_dimensions[vis_level][0]

    slide_vis_spacing = wsi_object.spacings[vis_level]
    slide_width, slide_height = wsi_object.level_dimensions[vis_level]
    slide = wsi_object.wsi.get_patch(
        0, 0, slide_width, slide_height, spacing=slide_vis_spacing, center=False
    )
    slide = Image.fromarray(slide).convert("RGBA")

    mask_vis_spacing = mask_object.spacings[mask_vis_level]
    mask_width, mask_height = mask_object.level_dimensions[mask_vis_level]
    mask = mask_object.wsi.get_patch(
        0, 0, mask_width, mask_height, spacing=mask_vis_spacing, center=False
    )
    if mask.shape[-1] == 1:
        mask = np.squeeze(mask, axis=-1)
    mask = Image.fromarray(mask)

    # Mask data is present in the R channel
    mask = mask.split()[0]

    # Create alpha mask
    mask_arr = np.array(mask)
    alpha_int = int(round(255 * alpha))
    if color_mapping is not None:
        alpha_content = np.zeros_like(mask_arr)
        for k, v in pixel_mapping.items():
            if color_mapping[k] is not None:
                alpha_content += mask_arr == v
        alpha_content = np.less(alpha_content, 1).astype("uint8") * alpha_int + (
            255 - alpha_int
        )
    else:
        alpha_content = np.less_equal(mask_arr, 0).astype("uint8") * alpha_int + (
            255 - alpha_int
        )
    alpha_content = Image.fromarray(alpha_content)

    preview_palette = np.zeros(shape=768, dtype=int)

    if color_mapping is None:
        ncat = len(pixel_mapping)
        if ncat <= 10:
            color_palette = sns.color_palette("tab10")[:ncat]
        elif ncat <= 20:
            color_palette = sns.color_palette("tab20")[:ncat]
        else:
            raise ValueError(
                f"Implementation supports up to 20 categories (provided pixel_mapping has {ncat})"
            )
        color_mapping = {
            k: tuple(255 * x for x in color_palette[i])
            for i, k in enumerate(pixel_mapping.keys())
        }

    p = [0] * 3 * len(color_mapping)
    for k, v in pixel_mapping.items():
        if color_mapping[k] is not None:
            p[v * 3 : v * 3 + 3] = color_mapping[k]
    n = len(p)
    preview_palette[0:n] = np.array(p).astype(int)

    mask.putpalette(data=preview_palette.tolist())
    mask_rgb = mask.convert(mode="RGB")

    overlayed_image = Image.composite(image1=slide, image2=mask_rgb, mask=alpha_content)
    return overlayed_image


def get_masked_tile(
    wsi_object,
    mask_object,
    tile: Image.Image,
    x: int,
    y: int,
    spacing: float,
    patch_size: Tuple[int],
    upsample: bool = True,
    eps: float = 1e-5,
):
    wsi_spacing_level = wsi_object.get_best_level_for_spacing(
        spacing, ignore_warning=True
    )

    x_mask = mask_object.level_dimensions[0][0]
    mask_min_level = int(
        np.argmin([abs(x_wsi - x_mask) for x_wsi, _ in wsi_object.level_dimensions])
    )
    mask_spacing_level = mask_object.get_best_level_for_spacing(
        spacing, ignore_warning=True
    )

    assert x_mask == wsi_object.level_dimensions[mask_min_level][0]

    # if mask doesn't start at same spacing as wsi, need to downsample tile & scale (x, y) coordinates!
    # need to scale x, y from slide level 0 to mask level 0 referential
    mask_scale = tuple(
        a / b
        for a, b in zip(
            wsi_object.level_downsamples[mask_min_level],
            wsi_object.level_downsamples[0],
        )
    )
    x_scaled, y_scaled = int(x * 1.0 / mask_scale[0]), int(y * 1.0 / mask_scale[1])
    # need to scale tile size from wsi_spacing_level to mask_spacing_level
    ts_scale = tuple(
        a / b
        for a, b in zip(
            wsi_object.level_downsamples[mask_min_level + mask_spacing_level],
            wsi_object.level_downsamples[wsi_spacing_level],
        )
    )
    ts_x, ts_y = int(patch_size[0] * 1.0 / ts_scale[0]), int(
        patch_size[1] * 1.0 / ts_scale[1]
    )
    # read annotation tile from mask
    masked_tile = mask_object.wsi.get_patch(
        x_scaled, y_scaled, ts_x, ts_y, spacing=spacing, center=False
    )
    if masked_tile.shape[-1] == 1:
        masked_tile = np.squeeze(masked_tile, axis=-1)
    masked_tile = Image.fromarray(masked_tile)
    masked_tile = masked_tile.split()[0]
    if ts_scale[0] > (1 + eps) or ts_scale[1] > (1 + eps):
        # 2 possible ways to go:
        # - upsample annotation tile to match true tile size
        # - read tile from slide to match annotation tile size
        if upsample:
            # option 1
            masked_tile = masked_tile.resize(
                tuple(int(e * ts_scale[i]) for i, e in enumerate(masked_tile.size)),
                Image.NEAREST,
            )
        else:
            # option 2
            tile_spacing = wsi_object.spacings[mask_min_level + mask_spacing_level]
            tile = wsi_object.get_patch(
                x, y, ts_x, ts_y, spacing=tile_spacing, center=False
            )
            tile = Image.fromarray(tile).convert("RGB")
    return tile, masked_tile


def overlay_mask_on_tile(
    tile: Image.Image,
    mask: Image.Image,
    pixel_mapping: Dict[str, int],
    color_mapping: Optional[Dict[str, List[int]]] = None,
    alpha=0.5,
):

    # Create alpha mask
    mask_arr = np.array(mask)
    alpha_int = int(round(255 * alpha))
    if color_mapping is not None:
        alpha_content = np.zeros_like(mask_arr)
        for k, v in pixel_mapping.items():
            if color_mapping[k] is not None:
                alpha_content += mask_arr == v
        alpha_content = np.less(alpha_content, 1).astype("uint8") * alpha_int + (
            255 - alpha_int
        )
    else:
        alpha_content = np.less_equal(mask_arr, 0).astype("uint8") * alpha_int + (
            255 - alpha_int
        )
    alpha_content = Image.fromarray(alpha_content)

    preview_palette = np.zeros(shape=768, dtype=int)

    if color_mapping is None:
        ncat = len(pixel_mapping)
        if ncat <= 10:
            color_palette = sns.color_palette("tab10")[:ncat]
        elif ncat <= 20:
            color_palette = sns.color_palette("tab20")[:ncat]
        else:
            raise ValueError(
                f"Implementation supports up to 20 categories (provided pixel_mapping has {ncat})"
            )
        color_mapping = {
            k: tuple(255 * x for x in color_palette[i])
            for i, k in enumerate(pixel_mapping.keys())
        }

    p = [0] * 3 * len(color_mapping)
    for k, v in pixel_mapping.items():
        if color_mapping[k] is not None:
            p[v * 3 : v * 3 + 3] = color_mapping[k]
    n = len(p)
    preview_palette[0:n] = np.array(p).astype(int)

    mask.putpalette(data=preview_palette.tolist())
    mask_rgb = mask.convert(mode="RGB")

    overlayed_image = Image.composite(image1=tile, image2=mask_rgb, mask=alpha_content)
    return overlayed_image


def DrawGrid(img, coord, shape, thickness=2, color=(0, 0, 0, 255)):
    cv2.rectangle(
        img,
        tuple(np.maximum([0, 0], coord - thickness // 2)),
        tuple(coord - thickness // 2 + np.array(shape)),
        color,
        thickness=thickness,
    )
    return img


def DrawMapFromCoords(
    canvas,
    wsi_object,
    coords,
    patch_size,
    vis_level: int,
    indices: Optional[List[int]] = None,
    draw_grid: bool = True,
    thickness: int = 2,
    verbose: bool = False,
    mask_object=None,
    pixel_mapping: Optional[Dict[str, int]] = None,
    color_mapping: Optional[Dict[str, int]] = None,
    alpha: Optional[float] = None,
):

    downsamples = wsi_object.level_downsamples[vis_level]
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)

    patch_size = tuple(
        np.ceil((np.array(patch_size) / np.array(downsamples))).astype(np.int32)
    )
    if verbose:
        print(f"downscaled patch size: {patch_size}")

    for idx in range(total):

        patch_id = indices[idx]
        coord = coords[patch_id]
        x, y = coord
        vis_spacing = wsi_object.get_level_spacing(vis_level)

        if mask_object is not None:
            # ensure mask and slide have at least one common spacing
            common_spacings = find_common_spacings(wsi_object.spacings, mask_object.spacings, tolerance=0.1)
            assert (
                len(common_spacings) >= 1
            ), f"The provided segmentation mask (spacings={mask_object.spacings}) has no common spacing with the slide (spacings={wsi_object.spacings}). A minimum of 1 common spacing is required."

            # check if this spacing is present in common spacings
            is_in_common_spacings = vis_spacing in [s for s,_ in common_spacings]
            if not is_in_common_spacings:
                # find spacing that is common to slide and mask and that is the closest to seg_spacing
                closest = np.argmin([abs(vis_spacing - s) for s,_ in common_spacings])
                closest_common_spacing = common_spacings[closest][0]
                vis_spacing = closest_common_spacing
                vis_level = wsi_object.get_best_level_for_spacing(vis_spacing)

        width, height = patch_size
        tile = wsi_object.wsi.get_patch(
            x, y, width, height, spacing=vis_spacing, center=False
        )

        if mask_object is not None:
            tile, masked_tile = get_masked_tile(
                wsi_object,
                mask_object,
                Image.fromarray(tile).convert("RGB"),
                x,
                y,
                vis_spacing,
                patch_size,
            )
            tile = overlay_mask_on_tile(
                tile, masked_tile, pixel_mapping, color_mapping, alpha=alpha
            )
            tile = np.array(tile)

        coord = np.ceil(
            tuple(coord[i] / downsamples[i] for i in range(len(coord)))
        ).astype(np.int32)
        canvas_crop_shape = canvas[
            coord[1] : coord[1] + patch_size[1],
            coord[0] : coord[0] + patch_size[0],
            :3,
        ].shape[:2]
        canvas[
            coord[1] : coord[1] + patch_size[1],
            coord[0] : coord[0] + patch_size[0],
            :3,
        ] = tile[: canvas_crop_shape[0], : canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size, thickness=thickness)

    return Image.fromarray(canvas)


def VisualizeCoords(
    hdf5_file_path: Path,
    wsi_object,
    downscale: int = 16,
    draw_grid: bool = False,
    thickness: int = 2,
    bg_color: Tuple[int] = (0, 0, 0),
    verbose: bool = False,
    key: str = "coords",
    heatmap: Optional[Image.Image] = None,
    mask_object=None,
    pixel_mapping: Optional[Dict[str, int]] = None,
    color_mapping: Optional[Dict[str, int]] = None,
    alpha: Optional[float] = None,
    display_slide: bool = True,
):
    vis_level = wsi_object.get_best_level_for_downsample_custom(downscale)
    h5_file = h5py.File(hdf5_file_path, "r")
    dset = h5_file[key]
    coords = dset[:]
    w, h = wsi_object.level_dimensions[0]

    if len(coords) == 0:
        return heatmap

    if verbose:
        print(f"original size: {w} x {h}")

    w, h = wsi_object.level_dimensions[vis_level]

    patch_size = dset.attrs["patch_size"]
    patch_level = dset.attrs["patch_level"]
    if verbose:
        print(f"downscaled size for stiching: {w} x {h}")
        print(f"number of patches: {len(coords)}")
        print(f"patch size: {patch_size}")
        print(f"patch level: {patch_level}")

    patch_size = tuple(
        (
            np.array((patch_size, patch_size))
            * wsi_object.level_downsamples[patch_level]
        ).astype(np.int32)
    )
    if verbose:
        print(f"ref patch size: {patch_size}")

    if w * h > Image.MAX_IMAGE_PIXELS:
        raise Image.DecompressionBombError(
            "Visualization Downscale %d is too large" % downscale
        )

    if heatmap is None:
        if mask_object is not None:
            heatmap = overlay_mask_on_slide(
                wsi_object,
                mask_object,
                vis_level,
                pixel_mapping,
                color_mapping,
                alpha=alpha,
            )
        elif display_slide:
            vis_spacing = wsi_object.spacings[vis_level]
            heatmap = wsi_object.wsi.get_patch(
                0, 0, w, h, spacing=vis_spacing, center=False
            )
            heatmap = Image.fromarray(heatmap).convert("RGB")
        else:
            heatmap = Image.new(size=(w, h), mode="RGB", color=bg_color)

    heatmap = np.array(heatmap)
    heatmap = DrawMapFromCoords(
        heatmap,
        wsi_object,
        coords,
        patch_size,
        vis_level,
        indices=None,
        draw_grid=draw_grid,
        thickness=thickness,
        verbose=verbose,
        mask_object=mask_object,
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
        alpha=alpha,
    )

    h5_file.close()
    return heatmap

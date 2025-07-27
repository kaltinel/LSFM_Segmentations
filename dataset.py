"""Handles dataset creation for the PyTorch models - namely segemntation mask ones.

References:
    https://pytorch.org/docs/stable/data.html
    https://github.com/aladdinpersson/Machine-\
    Learning-Collection/blob/master/ML/Pytorch/Basics/custom_dataset/custom_dataset.py
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    https://shashikachamod4u.medium.com/excel-csv-to-pytorch-dataset-def496b6bcc1
"""

import pathlib
import torch
from skimage import io
from skimage import transform as trans
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import transform as trans
from sklearn.model_selection import train_test_split
import json
import nibabel as nib
from PIL import Image


# Fix COLOR_DICT to use integer keys as expected by the original framework
COLOR_DICT = {
    0: (0, 0, 0),  # background - black
    1: (255, 0, 0),  # largest - red
    2: (0, 255, 0),  # second_largest - green
}


class OneHotEncodeMask(object):
    def __init__(self, color_dict: dict):
        """Converts an RGB mask from the input dataset sample,
        into one-hot-encoded multidimensional array, with shape CxHxW,
        C being the number of classes. Conversion is done on the whole batch.

        Args:
            color_dict: Dictionary that maps RGB colors to classes with
                        integer keys (0, 1, 2, ...) mapped to RGB color tuples.
        """
        self.color_dict = color_dict

    def __call__(self, sample: dict):
        """Performs the conversion.

        Args:
            sample: Input dataset sample, containing image and the mask.

        Returns:
            Dataset sample containing unchanged image and one-hot-encoded mask.
        """
        img, mask = sample["image"], sample["mask"]

        num_classes = len(self.color_dict)
        shape = mask.shape[:2] + (num_classes,)
        arr = np.zeros(shape, dtype=np.int8)

        # Use more robust color matching with tolerance for compression artifacts
        for i, color in enumerate(self.color_dict.values()):
            # Calculate Euclidean distance to handle slight variations from resizing/compression
            color_array = np.array(color)
            mask_reshaped = mask.reshape((-1, 3))

            # Use a small tolerance for color matching (handles compression artifacts)
            distances = np.linalg.norm(mask_reshaped - color_array, axis=1)
            tolerance = 10  # Allow small variations due to resizing/compression
            matches = distances <= tolerance

            arr[:, :, i] = matches.reshape(shape[:2])

        return {"image": img, "mask": arr}


class Resize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        img, mask = sample["image"], sample["mask"]
        new_h, new_w = self.output_size, self.output_size

        img = trans.resize(img, (new_h, new_w))
        # It is expected that masks are always first one-hot-encoded.
        # preserve_range=True is called here to preserve them in the 0-1 range, which is benefecial
        # for models that predict pixel probabilities.
        mask = trans.resize(mask, (new_h, new_w), preserve_range=True)

        return {"image": img, "mask": mask}


class ColorJitter(object):
    def __call__(self, sample):
        img, mask = sample["image"], sample["mask"]

        # Handle different image formats
        if len(img.shape) == 2:  # Grayscale image (H, W)
            # Convert to RGB by stacking the grayscale channel 3 times
            img = np.stack([img] * 3, axis=-1)  # (H, W, 3)
        elif len(img.shape) == 3 and img.shape[2] == 1:  # (H, W, 1)
            # Convert single channel to 3 channels
            img = np.repeat(img, 3, axis=2)  # (H, W, 3)

        # Ensure image has 3 dimensions (H, W, C)
        if len(img.shape) != 3:
            raise ValueError(
                f"Image must have 2 or 3 dimensions, got shape: {img.shape}"
            )

        jitter = transforms.ColorJitter(brightness=0.5, hue=0.3)
        img = img.transpose(2, 0, 1)  # Convert to (C, H, W) for PyTorch
        jitted_img = jitter(torch.from_numpy(img).double())
        jitted_img = jitted_img.numpy().transpose(1, 2, 0)  # Convert back to (H, W, C)

        return {"image": jitted_img, "mask": mask}


class GaussianBlur(object):
    def __call__(self, sample):
        img, mask = sample["image"], sample["mask"]

        # Handle different image formats
        if len(img.shape) == 2:  # Grayscale image (H, W)
            # Convert to RGB by stacking the grayscale channel 3 times
            img = np.stack([img] * 3, axis=-1)  # (H, W, 3)
        elif len(img.shape) == 3 and img.shape[2] == 1:  # (H, W, 1)
            # Convert single channel to 3 channels
            img = np.repeat(img, 3, axis=2)  # (H, W, 3)

        # Ensure image has 3 dimensions (H, W, C)
        if len(img.shape) != 3:
            raise ValueError(
                f"Image must have 2 or 3 dimensions, got shape: {img.shape}"
            )

        blurrer = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        # Convert to (C, H, W) for PyTorch transforms
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).double()
        blurred_img = blurrer(img_tensor)
        blurred_img = blurred_img.numpy().transpose(
            1, 2, 0
        )  # Convert back to (H, W, C)

        return {"image": blurred_img, "mask": mask}


class ToTensor(object):
    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]

        # Handle different image formats
        if len(image.shape) == 2:  # Grayscale image (H, W)
            # Convert to RGB by stacking the grayscale channel 3 times
            image = np.stack([image] * 3, axis=-1)  # (H, W, 3)
        elif len(image.shape) == 3 and image.shape[2] == 1:  # (H, W, 1)
            # Convert single channel to 3 channels
            image = np.repeat(image, 3, axis=2)  # (H, W, 3)

        # Ensure image has 3 dimensions (H, W, C)
        if len(image.shape) != 3:
            raise ValueError(
                f"Image must have 2 or 3 dimensions, got shape: {image.shape}"
            )

        # Ensure mask has 3 dimensions (H, W, C)
        if len(mask.shape) != 3:
            raise ValueError(f"Mask must have 3 dimensions, got shape: {mask.shape}")

        # Transpose from (H, W, C) to (C, H, W) for PyTorch
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        return {
            "image": torch.from_numpy(image).double(),
            "mask": torch.from_numpy(mask).double(),
        }


class SegMaskDataset(Dataset):

    def __init__(
        self,
        dataset_dir,
        mask_dir,
        train_image,
        eval_image,
        train=True,
        transforms=None,
        mask_in_png=True,
        store_original_size=False,
        store_original_name=False,
    ):
        """Performs initial loading of a image segmentation dataset as a collection of .png
        images and segmentation masks. It is assumed that input images and segmentation
        masks have the same size!
        """
        self.dataset_dir = dataset_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.train_image = train_image
        self.eval_image = eval_image
        self.train = train
        # Sometimes, input dataset comes with images in bmp and masks in png,
        # so it can't be assumed that they will have the same exact name and extension.
        self.mask_in_png = mask_in_png
        self.store_original_size = store_original_size
        self.store_original_name = store_original_name

    def __len__(self):
        if self.train == True:
            return len(self.train_image)
        else:
            return len(self.eval_image)

    def __getitem__(self, idx):
        """Returns a single element of a loaded dataset."""
        if self.train:
            im = io.imread(pathlib.Path(self.dataset_dir, self.train_image[idx]))
            if self.mask_in_png:
                train_image = self.train_image[idx].split(".")[0] + ".png"
                im_mask = io.imread(pathlib.Path(self.mask_dir, train_image))
            else:
                im_mask = io.imread(pathlib.Path(self.mask_dir, self.train_image[idx]))
            sample = {"image": im, "mask": im_mask}
        else:
            im = io.imread(pathlib.Path(self.dataset_dir, self.eval_image[idx]))
            if self.mask_in_png:
                eval_image = self.eval_image[idx].split(".")[0] + ".png"
                im_mask = io.imread(pathlib.Path(self.mask_dir, eval_image))
            else:
                im_mask = io.imread(pathlib.Path(self.mask_dir, self.eval_image[idx]))
            sample = {"image": im, "mask": im_mask}

        if self.transforms:
            sample = self.transforms(sample)

        if self.store_original_size:
            sample["original_height"] = im.shape[0]
            sample["original_width"] = im.shape[1]

        if self.store_original_name:
            sample["original_name"] = pathlib.Path(
                self.dataset_dir, self.train_image[idx]
            ).stem

        return sample


def create_rgb_mask_from_annotation_largest_2(annotation_slice, background_value=0):
    """
    Creates an RGB mask with only the 2 largest annotated regions.

    Args:
        annotation_slice: 2D numpy array with integer region labels
        background_value: Value representing background (default: 0)

    Returns:
        RGB image (H, W, 3) with largest regions colored according to COLOR_DICT
        and background black
    """
    h, w = annotation_slice.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Get unique region IDs (excluding background)
    unique_regions = np.unique(annotation_slice)
    unique_regions = unique_regions[unique_regions != background_value]

    if len(unique_regions) == 0:
        return rgb_mask  # Return black mask if no regions

    # Calculate size of each region
    region_sizes = []
    for region_id in unique_regions:
        region_mask = annotation_slice == region_id
        region_size = np.sum(region_mask)
        region_sizes.append((region_id, region_size))

    # Sort by size (largest first) and take top 2
    region_sizes.sort(key=lambda x: x[1], reverse=True)
    top_2_regions = region_sizes[:2]

    # Use COLOR_DICT as single source of truth for colors
    color_keys = ["largest", "second_largest"]
    color_names = ["Red", "Green"]

    # Apply colors to the regions
    for i, (region_id, region_size) in enumerate(top_2_regions):
        if i < 2:  # Only process top 2
            region_mask = annotation_slice == region_id
            # Fix: Use integer keys instead of string keys
            rgb_mask[region_mask] = COLOR_DICT[
                i + 1
            ]  # 1 for largest, 2 for second_largest
            print(f"Region {region_id}: {region_size} pixels, colored {color_names[i]}")

    return rgb_mask


def analyze_region_distribution(annotation_slice, background_value=0):
    """
    Analyzes the distribution of regions in an annotation slice.

    Args:
        annotation_slice: 2D numpy array with integer region labels
        background_value: Value representing background (default: 0)

    Returns:
        Dictionary with region statistics
    """
    unique_regions = np.unique(annotation_slice)
    unique_regions = unique_regions[unique_regions != background_value]

    region_stats = {}
    total_annotated_pixels = 0

    for region_id in unique_regions:
        region_mask = annotation_slice == region_id
        region_size = np.sum(region_mask)
        region_stats[region_id] = region_size
        total_annotated_pixels += region_size

    # Sort by size
    sorted_regions = sorted(region_stats.items(), key=lambda x: x[1], reverse=True)

    print(f"Total annotated pixels: {total_annotated_pixels}")
    print(f"Number of regions: {len(unique_regions)}")
    print("\nTop 10 largest regions:")
    for i, (region_id, size) in enumerate(sorted_regions[:10]):
        percentage = (
            (size / total_annotated_pixels) * 100 if total_annotated_pixels > 0 else 0
        )
        print(f"{i+1}. Region {region_id}: {size} pixels ({percentage:.2f}%)")

    return {
        "region_stats": region_stats,
        "sorted_regions": sorted_regions,
        "total_annotated_pixels": total_annotated_pixels,
    }


def normalize_image_for_training(image_slice):
    """
    Normalize image slice for training (0-255 uint8).

    Args:
        image_slice: 2D numpy array

    Returns:
        Normalized image as uint8
    """
    # Normalize to 0-1 range
    img_norm = (image_slice - image_slice.min()) / (
        image_slice.max() - image_slice.min()
    )
    # Convert to 0-255 range
    img_uint8 = (img_norm * 255).astype(np.uint8)
    # Convert to RGB if grayscale
    if len(img_uint8.shape) == 2:
        img_uint8 = np.stack([img_uint8] * 3, axis=-1)

    return img_uint8


def process_and_save_largest_regions_masks(
    annotation_data, output_dir, slice_range=None, axis=2
):
    """
    Process annotation data and save masks with only the 2 largest regions.

    Args:
        annotation_data: 3D numpy array with region annotations
        output_dir: Directory to save the generated mask images
        slice_range: Optional tuple (start, end) to process specific slices
        axis: Which axis to slice along (0, 1, or 2)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine slice range
    if slice_range is None:
        start_slice = 0
        end_slice = annotation_data.shape[axis]
    else:
        start_slice, end_slice = slice_range

    print(f"Processing slices {start_slice} to {end_slice-1} along axis {axis}")

    for slice_idx in range(start_slice, end_slice):
        # Extract slice based on axis
        if axis == 0:
            annotation_slice = annotation_data[slice_idx, :, :]
        elif axis == 1:
            annotation_slice = annotation_data[:, slice_idx, :]
        else:  # axis == 2
            annotation_slice = annotation_data[:, :, slice_idx]

        # Transpose if needed (common for visualization)
        annotation_slice = np.transpose(annotation_slice)

        # Generate mask with 2 largest regions
        print(f"\nProcessing slice {slice_idx}:")
        rgb_mask = create_rgb_mask_from_annotation_largest_2(annotation_slice)

        # Save as PNG
        output_filename = f"largest_2_regions_slice_{slice_idx:04d}.png"
        output_filepath = output_path / output_filename

        plt.imsave(output_filepath, rgb_mask)
        print(f"Saved to {output_filepath}")

    print(f"\nCompleted processing. Masks saved to {output_dir}")


def create_rgb_mask_from_annotation(annotation_slice, background_value=0):
    """
    Creates an RGB mask with the 2 largest annotated regions.
    This is a wrapper around the existing create_rgb_mask_from_annotation_largest_2 function.

    Args:
        annotation_slice: 2D numpy array with integer region labels
        background_value: Value representing background (default: 0)

    Returns:
        RGB image (H, W, 3) with largest regions colored according to COLOR_DICT
        and background black
    """
    return create_rgb_mask_from_annotation_largest_2(annotation_slice, background_value)


def generate_training_dataset(
    template_data,
    annotation_data,
    output_dir="training_dataset",
    image_size=128,
    n_slices_per_axis=10,
    test_split=0.2,
):
    """
    Generate a complete U-Net training dataset from 3D brain data.

    Args:
        template_data: 3D template image data
        annotation_data: 3D annotation data
        output_dir: Directory to save dataset
        image_size: Target image size for training
        n_slices_per_axis: Number of slices to extract per axis
        test_split: Fraction of data for testing

    Returns:
        tuple: (train_samples, test_samples, dirs) where dirs contains the created directory paths
    """

    output_dir = Path(output_dir)

    # Create directory structure matching the existing U-Net setup
    dirs = {
        "train_images": output_dir / "train" / "JPEGImages",
        "train_masks": output_dir / "train" / "SegmentationClass",
        "test_images": output_dir / "test" / "JPEGImages",
        "test_masks": output_dir / "test" / "SegmentationClass",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # Extract slices from all three axes
    axes_info = [
        (0, "sagittal", template_data.shape[0]),
        (1, "coronal", template_data.shape[1]),
        (2, "axial", template_data.shape[2]),
    ]

    sample_id = 0

    for axis, axis_name, max_slices in axes_info:
        # Select evenly spaced slices
        slice_indices = np.linspace(
            max_slices // 4, 3 * max_slices // 4, n_slices_per_axis, dtype=int
        )

        for slice_idx in slice_indices:
            # Extract image and annotation slices
            if axis == 0:
                img_slice = template_data[slice_idx, :, :]
                ann_slice = annotation_data[slice_idx, :, :]
            elif axis == 1:
                img_slice = template_data[:, slice_idx, :]
                ann_slice = annotation_data[:, slice_idx, :]
            else:  # axis == 2
                img_slice = template_data[:, :, slice_idx]
                ann_slice = annotation_data[:, :, slice_idx]

            # Skip if slice is mostly empty
            if np.sum(ann_slice > 0) < 100:  # Minimum annotation pixels
                continue

            # Process image and mask
            processed_img = normalize_image_for_training(img_slice)
            rgb_mask = create_rgb_mask_from_annotation(ann_slice)

            # Resize to target size
            processed_img = trans.resize(
                processed_img, (image_size, image_size), preserve_range=True
            ).astype(np.uint8)
            rgb_mask = trans.resize(
                rgb_mask, (image_size, image_size), preserve_range=True
            ).astype(np.uint8)

            # Save sample info
            sample_info = {
                "id": sample_id,
                "axis": axis_name,
                "slice_idx": int(slice_idx),
                "filename": f"{axis_name}_{slice_idx:03d}_{sample_id:04d}",
            }
            all_samples.append(sample_info)
            sample_id += 1

    # Split into train/test
    train_samples, test_samples = train_test_split(
        all_samples, test_size=test_split, random_state=42
    )

    return train_samples, test_samples, dirs


def save_samples(
    samples, template_data, annotation_data, dirs, split="train", image_size=128
):
    """
    Save processed samples to disk in the format expected by the existing U-Net setup.

    Args:
        samples: List of sample dictionaries with metadata
        template_data: 3D template image data
        annotation_data: 3D annotation data
        dirs: Dictionary containing directory paths
        split: Either 'train' or 'test'
        image_size: Target image size for resizing
    """

    if split == "train":
        img_dir = dirs["train_images"]
        mask_dir = dirs["train_masks"]
    else:
        img_dir = dirs["test_images"]
        mask_dir = dirs["test_masks"]

    axes_info = [(0, "sagittal"), (1, "coronal"), (2, "axial")]
    axis_map = {name: idx for idx, name in axes_info}

    for sample in samples:
        axis_idx = axis_map[sample["axis"]]
        slice_idx = sample["slice_idx"]
        filename = sample["filename"]

        # Extract slices
        if axis_idx == 0:
            img_slice = template_data[slice_idx, :, :]
            ann_slice = annotation_data[slice_idx, :, :]
        elif axis_idx == 1:
            img_slice = template_data[:, slice_idx, :]
            ann_slice = annotation_data[:, slice_idx, :]
        else:
            img_slice = template_data[:, :, slice_idx]
            ann_slice = annotation_data[:, :, slice_idx]

        # Process and resize
        processed_img = normalize_image_for_training(img_slice)
        rgb_mask = create_rgb_mask_from_annotation(ann_slice)

        processed_img = trans.resize(
            processed_img, (image_size, image_size), preserve_range=True
        ).astype(np.uint8)
        rgb_mask = trans.resize(
            rgb_mask, (image_size, image_size), preserve_range=True
        ).astype(np.uint8)

        # Save image (convert to grayscale if needed for consistency)
        if len(processed_img.shape) == 3 and processed_img.shape[2] == 3:
            # Convert RGB to grayscale for template images
            gray_img = np.dot(processed_img, [0.299, 0.587, 0.114]).astype(np.uint8)
            img_pil = Image.fromarray(gray_img, mode="L")
        else:
            img_pil = Image.fromarray(processed_img, mode="L")

        img_pil.save(img_dir / f"{filename}.jpg")

        # Save mask as PNG (matching existing setup)
        mask_pil = Image.fromarray(rgb_mask, mode="RGB")
        mask_pil.save(mask_dir / f"{filename}.png")

    print(f"Saved {len(samples)} {split} samples")


def generate_complete_dataset(
    template_file,
    annotation_file,
    output_dir="training_dataset",
    image_size=128,
    n_slices_per_axis=10,
    test_split=0.2,
):
    """
    Complete pipeline to generate a U-Net compatible dataset from NIfTI files.

    Args:
        template_file: Path to template NIfTI file
        annotation_file: Path to annotation NIfTI file
        output_dir: Directory to save the generated dataset
        image_size: Target image size for training
        n_slices_per_axis: Number of slices to extract per axis
        test_split: Fraction of data for testing

    Returns:
        Dictionary with dataset statistics and paths
    """

    print("Loading data files...")

    # Load template and annotation data
    template_nii = nib.load(template_file)
    template_data = template_nii.get_fdata()

    annotation_nii = nib.load(annotation_file)
    annotation_data = annotation_nii.get_fdata().astype(np.uint16)

    print(f"Template data shape: {template_data.shape}")
    print(f"Annotation data shape: {annotation_data.shape}")

    # Generate dataset structure
    print("Generating dataset structure...")
    train_samples, test_samples, dirs = generate_training_dataset(
        template_data,
        annotation_data,
        output_dir=output_dir,
        image_size=image_size,
        n_slices_per_axis=n_slices_per_axis,
        test_split=test_split,
    )

    print(
        f"Generated {len(train_samples)} training samples and {len(test_samples)} test samples"
    )

    # Save training samples
    print("Saving training samples...")
    save_samples(
        train_samples, template_data, annotation_data, dirs, "train", image_size
    )

    # Save test samples
    print("Saving test samples...")
    save_samples(test_samples, template_data, annotation_data, dirs, "test", image_size)

    # Return summary
    dataset_info = {
        "output_dir": str(Path(output_dir).absolute()),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "image_size": image_size,
        "directories": {k: str(v) for k, v in dirs.items()},
    }

    print("Dataset generation complete!")
    print(f"Dataset saved to: {dataset_info['output_dir']}")

    return dataset_info


if __name__ == "__main__":

    def create_rgb_mask_from_annotation_largest_2(annotation_slice, background_value=0):
        """
        Creates an RGB mask with only the 2 largest annotated regions.

        Args:
            annotation_slice: 2D numpy array with integer region labels
            background_value: Value representing background (default: 0)

        Returns:
            RGB image (H, W, 3) with largest regions colored according to COLOR_DICT
            and background black
        """
        h, w = annotation_slice.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

        # Get unique region IDs (excluding background)
        unique_regions = np.unique(annotation_slice)
        unique_regions = unique_regions[unique_regions != background_value]

        if len(unique_regions) == 0:
            return rgb_mask  # Return black mask if no regions

        # Calculate size of each region
        region_sizes = []
        for region_id in unique_regions:
            region_mask = annotation_slice == region_id
            region_size = np.sum(region_mask)
            region_sizes.append((region_id, region_size))

        # Sort by size (largest first) and take top 2
        region_sizes.sort(key=lambda x: x[1], reverse=True)
        top_2_regions = region_sizes[:2]

        # Use COLOR_DICT as single source of truth for colors
        color_keys = ["largest", "second_largest"]
        color_names = ["Red", "Green"]

        # Apply colors to the regions
        for i, (region_id, region_size) in enumerate(top_2_regions):
            if i < 2:  # Only process top 2
                region_mask = annotation_slice == region_id
                # Fix: Use integer keys instead of string keys
                rgb_mask[region_mask] = COLOR_DICT[
                    i + 1
                ]  # 1 for largest, 2 for second_largest
                print(
                    f"Region {region_id}: {region_size} pixels, colored {color_names[i]}"
                )

        return rgb_mask

    def analyze_region_distribution(annotation_slice, background_value=0):
        """
        Analyzes the distribution of regions in an annotation slice.

        Args:
            annotation_slice: 2D numpy array with integer region labels
            background_value: Value representing background (default: 0)

        Returns:
            Dictionary with region statistics
        """
        unique_regions = np.unique(annotation_slice)
        unique_regions = unique_regions[unique_regions != background_value]

        region_stats = {}
        total_annotated_pixels = 0

        for region_id in unique_regions:
            region_mask = annotation_slice == region_id
            region_size = np.sum(region_mask)
            region_stats[region_id] = region_size
            total_annotated_pixels += region_size

        # Sort by size
        sorted_regions = sorted(region_stats.items(), key=lambda x: x[1], reverse=True)

        print(f"Total annotated pixels: {total_annotated_pixels}")
        print(f"Number of regions: {len(unique_regions)}")
        print("\nTop 10 largest regions:")
        for i, (region_id, size) in enumerate(sorted_regions[:10]):
            percentage = (
                (size / total_annotated_pixels) * 100
                if total_annotated_pixels > 0
                else 0
            )
            print(f"{i+1}. Region {region_id}: {size} pixels ({percentage:.2f}%)")

        return {
            "region_stats": region_stats,
            "sorted_regions": sorted_regions,
            "total_annotated_pixels": total_annotated_pixels,
        }

    def normalize_image_for_training(image_slice):
        """
        Normalize image slice for training (0-255 uint8).

        Args:
            image_slice: 2D numpy array

        Returns:
            Normalized image as uint8
        """
        # Normalize to 0-1 range
        img_norm = (image_slice - image_slice.min()) / (
            image_slice.max() - image_slice.min()
        )
        # Convert to 0-255 range
        img_uint8 = (img_norm * 255).astype(np.uint8)
        # Convert to RGB if grayscale
        if len(img_uint8.shape) == 2:
            img_uint8 = np.stack([img_uint8] * 3, axis=-1)

        return img_uint8

    def process_and_save_largest_regions_masks(
        annotation_data, output_dir, slice_range=None, axis=2
    ):
        """
        Process annotation data and save masks with only the 2 largest regions.

        Args:
            annotation_data: 3D numpy array with region annotations
            output_dir: Directory to save the generated mask images
            slice_range: Optional tuple (start, end) to process specific slices
            axis: Which axis to slice along (0, 1, or 2)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Determine slice range
        if slice_range is None:
            start_slice = 0
            end_slice = annotation_data.shape[axis]
        else:
            start_slice, end_slice = slice_range

        print(f"Processing slices {start_slice} to {end_slice-1} along axis {axis}")

        for slice_idx in range(start_slice, end_slice):
            # Extract slice based on axis
            if axis == 0:
                annotation_slice = annotation_data[slice_idx, :, :]
            elif axis == 1:
                annotation_slice = annotation_data[:, slice_idx, :]
            else:  # axis == 2
                annotation_slice = annotation_data[:, :, slice_idx]

            # Transpose if needed (common for visualization)
            annotation_slice = np.transpose(annotation_slice)

            # Generate mask with 2 largest regions
            print(f"\nProcessing slice {slice_idx}:")
            rgb_mask = create_rgb_mask_from_annotation_largest_2(annotation_slice)

            # Save as PNG
            output_filename = f"largest_2_regions_slice_{slice_idx:04d}.png"
            output_filepath = output_path / output_filename

            plt.imsave(output_filepath, rgb_mask)
            print(f"Saved to {output_filepath}")

        print(f"\nCompleted processing. Masks saved to {output_dir}")

    annotation_file = "/home/vice-calibras/mus_brain_seg/data/annotation_25.nii.gz"

    try:
        # Load annotation data
        annotation_nii = nib.load(annotation_file)
        annotation_data = annotation_nii.get_fdata().astype(np.uint16)

        print(f"Annotation data shape: {annotation_data.shape}")

        # Process a single slice for testing
        test_slice_idx = 100  # Adjust as needed
        annotation_slice = annotation_data[:, :, test_slice_idx]
        annotation_slice = np.transpose(annotation_slice)

        # Analyze region distribution
        print(f"\nAnalyzing slice {test_slice_idx}:")
        stats = analyze_region_distribution(annotation_slice)

        # Create and display the mask
        rgb_mask = create_rgb_mask_from_annotation_largest_2(annotation_slice)

        # Display results
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        axes[0].imshow(annotation_slice, cmap="nipy_spectral")
        axes[0].set_title(f"Original Annotations (Slice {test_slice_idx})")
        axes[0].axis("off")

        axes[1].imshow(rgb_mask)
        axes[1].set_title("2 Largest Regions Mask\n(Red=Largest, Green=2nd)")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(
            "Please update the annotation_file path to point to your actual data file"
        )

    # Generate a complete dataset from NIfTI files
    dataset_info = generate_complete_dataset(
        template_file="/home/vice-calibras/mus_brain_seg/data/average_template_25.nii.gz",
        annotation_file="/home/vice-calibras/mus_brain_seg/data/annotation_25.nii.gz",
        output_dir="/home/vice-calibras/mus_brain_seg/data/mouse_brain_unet_dataset",
        image_size=128,
        n_slices_per_axis=15,
        test_split=0.2,
    )

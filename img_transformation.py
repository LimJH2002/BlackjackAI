import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


class BalancedImageAugmenter:
    def __init__(self, image_dir: str, label_dir: str):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

    def get_class_distribution(self):
        """Analyze class distribution in the dataset"""
        class_counts = defaultdict(int)

        for label_file in self.label_dir.glob("*.txt"):
            with open(label_file, "r") as f:
                for line in f:
                    class_id = line.strip().split()[0]
                    class_counts[class_id] += 1

        return class_counts

    def calculate_augmentation_weights(self, class_counts):
        """Calculate how many augmentations each class needs"""
        max_count = max(class_counts.values())
        weights = {}
        for class_id, count in class_counts.items():
            # More augmentations for underrepresented classes
            ratio = max_count / count
            # Cap the maximum number of augmentations
            weights[class_id] = min(ratio, 3.0)
        return weights

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height))

    def adjust_brightness_contrast(
        self, image: np.ndarray, brightness: float, contrast: float
    ) -> np.ndarray:
        return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def get_image_classes(self, label_path: Path) -> list:
        """Get classes present in an image"""
        classes = []
        with open(label_path, "r") as f:
            for line in f:
                class_id = line.strip().split()[0]
                if class_id not in classes:
                    classes.append(class_id)
        return classes

    def adjust_label_rotation(
        self, label_content: str, angle: float, img_size: tuple
    ) -> str:
        new_labels = []
        height, width = img_size
        center_x, center_y = width / 2, height / 2
        angle_rad = np.radians(angle)

        for line in label_content.strip().split("\n"):
            if not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = parts[0]
            x, y, w, h = map(float, parts[1:])

            abs_x = x * width
            abs_y = y * height

            x_shifted = abs_x - center_x
            y_shifted = abs_y - center_y

            x_rot = x_shifted * np.cos(angle_rad) - y_shifted * np.sin(angle_rad)
            y_rot = x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad)

            x_new = (x_rot + center_x) / width
            y_new = (y_rot + center_y) / height

            x_new = max(0, min(1, x_new))
            y_new = max(0, min(1, y_new))

            new_labels.append(f"{class_id} {x_new:.6f} {y_new:.6f} {w:.6f} {h:.6f}")

        return "\n".join(new_labels)

    def augment_dataset(
        self,
        output_image_dir: str,
        output_label_dir: str,
        base_augmentation_probability: float = 0.5,
        max_augmentations_per_image: int = 2,
    ):
        """
        Augment dataset with balanced random transformations
        """
        # Analyze class distribution
        class_counts = self.get_class_distribution()
        augmentation_weights = self.calculate_augmentation_weights(class_counts)

        # Setup output directories
        output_image_path = Path(output_image_dir)
        output_label_path = Path(output_label_dir)
        output_image_path.mkdir(parents=True, exist_ok=True)
        output_label_path.mkdir(parents=True, exist_ok=True)

        # Available augmentation types
        augmentations = {
            "rotation": lambda img: self.rotate_image(
                img, random.choice([90, 180, 270])
            ),
            "brightness_contrast": lambda img: self.adjust_brightness_contrast(
                img,
                brightness=random.uniform(0.3, 1.7),
                contrast=random.uniform(0.5, 2.0),
            ),
            "grayscale": self.convert_to_grayscale,
            "blur": lambda img: self.apply_gaussian_blur(img, random.choice([3, 5])),
        }

        image_files = list(self.image_dir.glob("*.jpg")) + list(
            self.image_dir.glob("*.png")
        )
        print(f"Processing {len(image_files)} images...")

        for img_path in tqdm(image_files):
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue

            label_path = self.label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                print(f"No label file found for: {img_path}")
                continue

            # Copy original files
            shutil.copy2(img_path, output_image_path / img_path.name)
            shutil.copy2(label_path, output_label_path / label_path.name)

            # Get classes in this image
            image_classes = self.get_image_classes(label_path)

            # Calculate maximum augmentation weight for this image
            max_weight = max(
                augmentation_weights[class_id] for class_id in image_classes
            )

            # Adjust augmentation probability based on class weights
            adjusted_probability = min(base_augmentation_probability * max_weight, 1.0)

            # Determine number of augmentations for this image
            num_augmentations = random.randint(
                1,
                min(int(max_weight * max_augmentations_per_image), len(augmentations)),
            )

            if random.random() < adjusted_probability:
                # Randomly select augmentations
                selected_augmentations = random.sample(
                    list(augmentations.items()), num_augmentations
                )

                for aug_name, aug_func in selected_augmentations:
                    # Apply augmentation
                    augmented = aug_func(image)

                    # Handle rotation labels specially
                    if aug_name == "rotation":
                        with open(label_path, "r") as f:
                            label_content = f.read()
                        angle = 90  # Default rotation angle
                        rotated_labels = self.adjust_label_rotation(
                            label_content, angle, image.shape[:2]
                        )
                        # Save rotated labels
                        aug_filename = f"{img_path.stem}_{aug_name}"
                        with open(output_label_path / f"{aug_filename}.txt", "w") as f:
                            f.write(rotated_labels)
                    else:
                        # For other augmentations, copy original labels
                        aug_filename = f"{img_path.stem}_{aug_name}"
                        shutil.copy2(
                            label_path, output_label_path / f"{aug_filename}.txt"
                        )

                    # Save augmented image
                    cv2.imwrite(
                        str(output_image_path / f"{aug_filename}{img_path.suffix}"),
                        augmented,
                    )


if __name__ == "__main__":
    augmenter = BalancedImageAugmenter(
        image_dir="data/train/images", label_dir="data/train/labels"
    )

    # Create temporary output directories
    temp_image_dir = "data/train/temp_images"
    temp_label_dir = "data/train/temp_labels"

    try:
        # Perform balanced augmentation
        augmenter.augment_dataset(
            output_image_dir=temp_image_dir,
            output_label_dir=temp_label_dir,
            base_augmentation_probability=0.5,
            max_augmentations_per_image=2,
        )

        # Move files to original directory
        temp_image_path = Path(temp_image_dir)
        temp_label_path = Path(temp_label_dir)

        for img_file in temp_image_path.glob("*"):
            if img_file.name.endswith((".jpg", ".png")):
                shutil.move(str(img_file), str(augmenter.image_dir / img_file.name))

        for label_file in temp_label_path.glob("*.txt"):
            shutil.move(str(label_file), str(augmenter.label_dir / label_file.name))

        # Clean up temporary directories
        shutil.rmtree(temp_image_dir)
        shutil.rmtree(temp_label_dir)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Clean up temporary directories if they exist
        for temp_dir in [temp_image_dir, temp_label_dir]:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)

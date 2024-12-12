import os
import json
from PIL import Image

class YOLOToCOCO:
    def __init__(self):
        self.categories = []
        self.annotations = []
        self.images = []
        self.annotation_id = 1  # Unique ID for each annotation

    def add_category(self, class_id, class_name):
        """
        Add a category to the COCO dataset.
        :param class_id: Class ID (integer)
        :param class_name: Class name (string)
        """
        self.categories.append({"id": class_id, "name": class_name})

    def yolo_to_coco_bbox(self, yolo_bbox, img_width, img_height):
        """
        Converts YOLO bounding box format to COCO format.
        :param yolo_bbox: List containing [x_center, y_center, width, height] (normalized values)
        :param img_width: Width of the image
        :param img_height: Height of the image
        :return: [x_min, y_min, width, height] in COCO format
        """
        x_center, y_center, width, height = yolo_bbox
        x_min = (x_center - width / 2) * img_width
        y_min = (y_center - height / 2) * img_height
        bbox_width = width * img_width
        bbox_height = height * img_height
        return [x_min, y_min, bbox_width, bbox_height]

    def convert_yolo_to_coco(self, yolo_dir, img_dir, output_path):
        """
        Converts a YOLO dataset into COCO JSON format.
        :param yolo_dir: Directory with YOLO annotations
        :param img_dir: Directory with images
        :param output_path: Path to save the COCO JSON file
        """
        for txt_file in os.listdir(yolo_dir):
            if not txt_file.endswith('.txt'):
                continue

            img_file = os.path.splitext(txt_file)[0] + '.jpg'
            img_path = os.path.join(img_dir, img_file)

            # Load image to get dimensions
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except FileNotFoundError:
                print(f"Image {img_file} not found. Skipping.")
                continue

            # Add image entry to COCO format
            image_id = len(self.images) + 1
            self.images.append({
                "id": image_id,
                "file_name": img_file,
                "width": img_width,
                "height": img_height
            })

            # Read YOLO annotations
            annotation_path = os.path.join(yolo_dir, txt_file)
            with open(annotation_path, 'r') as f:
                yolo_annotations = f.readlines()

            # Convert annotations
            for ann in yolo_annotations:
                yolo_values = list(map(float, ann.strip().split()))
                class_id = int(yolo_values[0])
                bbox = self.yolo_to_coco_bbox(yolo_values[1:], img_width, img_height)

                self.annotations.append({
                    "id": self.annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                })
                self.annotation_id += 1

        # Save COCO JSON
        coco_data = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories
        }
        with open(output_path, 'w') as json_file:
            json.dump(coco_data, json_file, indent=4)

if __name__ == "__main__":
    yolo_to_coco = YOLOToCOCO()

    category_list = [
    "10c", "10d", "10h", "10s", "2c", "2d", "2h", "2s", "3c", "3d", "3h", "3s",
    "4c", "4d", "4h", "4s", "5c", "5d", "5h", "5s", "6c", "6d", "6h", "6s",
    "7c", "7d", "7h", "7s", "8c", "8d", "8h", "8s", "9c", "9d", "9h", "9s",
    "Ac", "Ad", "Ah", "As", "Jc", "Jd", "Jh", "Js", "Kc", "Kd", "Kh", "Ks",
    "Qc", "Qd", "Qh", "Qs"]

    for i, category_name in enumerate(category_list):
        yolo_to_coco.add_category(i + 1, category_name)

    # Convert YOLO to COCO
    yolo_to_coco.convert_yolo_to_coco(
        yolo_dir='./data/train/labels',
        img_dir='./data/train/images',
        output_path='./data/train/annotations_coco.json'
    )



import os

class YOLOToPascal:
    def yolo_to_pascal_voc(self, yolo_annotation, img_width, img_height):
        """
        Converts YOLO annotation to Pascal VOC format.
        :param yolo_annotation: List containing [class_id, x_center, y_center, width, height]
        :param img_width: Width of the image
        :param img_height: Height of the image
        :return: [class_id, x_min, y_min, x_max, y_max]
        """
        class_id, x_center, y_center, width, height = yolo_annotation
        x_min = (x_center - width / 2) * img_width
        y_min = (y_center - height / 2) * img_height
        x_max = (x_center + width / 2) * img_width
        y_max = (y_center + height / 2) * img_height
        
        return [class_id, int(x_min), int(y_min), int(x_max), int(y_max)]

    def convert_yolo_dataset(self, yolo_dir, img_dir, output_dir):
        """
        Converts a YOLO dataset into a format suitable for Faster R-CNN.
        :param yolo_dir: Directory with YOLO annotations
        :param img_dir: Directory with images
        :param output_dir: Output directory for converted annotations
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for txt_file in os.listdir(yolo_dir):
            if not txt_file.endswith('.txt'):
                continue
            
            img_file = os.path.splitext(txt_file)[0] + '.jpg'
            img_path = os.path.join(img_dir, img_file)
            # Load image to get dimensions
            img_width, img_height = 1920, 1080  # Example dimensions; replace with actual size
            
            annotation_path = os.path.join(yolo_dir, txt_file)
            with open(annotation_path, 'r') as f:
                yolo_annotations = f.readlines()
            
            # Convert and save annotations
            converted_annotations = []
            for ann in yolo_annotations:
                yolo_values = list(map(float, ann.strip().split()))
                converted_annotations.append(
                    self.yolo_to_pascal_voc(yolo_values, img_width, img_height)
                )
            
            # Save as Pascal VOC style
            output_annotation_path = os.path.join(output_dir, txt_file)
            with open(output_annotation_path, 'w') as f:
                for ann in converted_annotations:
                    f.write(' '.join(map(str, ann)) + '\n')

yolo = YOLOToPascal()
#yolo.convert_yolo_dataset('./data/train/labels', './data/train/images', './data/train/labels_pascal')
yolo.convert_yolo_dataset('./data/test/labels', './data/test/images', './data/test/labels_pascal')
yolo.convert_yolo_dataset('./data/valid/labels', './data/valid/images', './data/valid/labels_pascal')

from copy import copy

import cv2
from ultralytics import YOLOWorld


class Detector:
    def __init__(self, spot):
        self.spot = spot


class YOLODetector(Detector):
    def __init__(self, spot, yolo_world_path):
        super().__init__(spot)

        print("Initializing YOLOWorld model. ")
        if not yolo_world_path:
            raise ValueError("YOLOWorld model path must be provided.")

        self.yolo_model = YOLOWorld(yolo_world_path)
        custom_classes = ["", "bag", "cone", "pipe"]
        self.yolo_model.set_classes(custom_classes)
        print("Set classes for YOLOWorld model.")

    def set_up_detector(self, semantic_class):
        # Get list of current classes recognized by YOLO World model
        recognized_classes = self.yolo_model.model.names

        # If the classes were never set by the users, the recognized classes (model.model.names)
        # will be a dictionary. Otherwise, it'll be a list.
        if isinstance(recognized_classes, dict):
            recognized_classes = list(recognized_classes.values())

        # Check if the class exists in the list (lowercase for consistency)
        if semantic_class.lower() not in [cls.lower() for cls in recognized_classes]:
            updated_classes = recognized_classes + [semantic_class.lower()]
            self.yolo_model.model.set_classes(updated_classes)
            print(f"Updated recognized classes: {updated_classes}")

    def return_centroid(self, img_source, semantic_class, debug):
        image, img = self.spot.get_image_RGB(view=img_source)

        xy = self._get_centroid(img, semantic_class, rotate=0, debug=debug)
        if xy is None:
            print("Object not found in first image. Looking around!")
            xy, image, img, image_source = self._look_for_object(
                semantic_class, debug=debug
            )

            if xy is None:
                print("Object not found near robot")

        return xy, image, img

    def _get_centroid(self, img, semantic_class, rotate, debug):
        if rotate == 0:
            model_input = copy(img)
        elif rotate == 1:
            model_input = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 2:
            model_input = cv2.rotate(img, cv2.ROTATE_180)

        results = self.yolo_model(model_input)

        best_box = None
        best_confidence = -1.0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = r.names[class_id]
                confidence = float(box.conf[0])

                # Check if the class name matches the semantic class we're looking for and if box is not too big
                box_height = (
                    box.xyxy[0][3] - box.xyxy[0][1]
                )  # height of the bounding box
                if box_height > 0.95 * img.shape[0]:
                    continue
                box_width = box.xyxy[0][2] - box.xyxy[0][0]  # width of the bounding box
                if (
                    box_width > 0.95 * img.shape[1]
                ):  # If the box is more than 95% the width of the image, skip it
                    continue

                if class_name == semantic_class and confidence > best_confidence:
                    best_confidence = confidence
                    best_box = box

        if best_box:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            class_id = int(best_box.cls[0])

            mid_x = x1 + (x2 - x1) // 2
            mid_y = y1 + (y2 - y1) // 2

            if rotate == 0:
                centroid_x = mid_x
                centroid_y = mid_y

            elif rotate == 1:
                centroid_x = mid_y
                centroid_y = img.shape[0] - mid_x

            elif rotate == 2:
                centroid_x = img.shape[1] - mid_x
                centroid_y = img.shape[0] - mid_y

            print("The centroid of the bounding box is at:", centroid_x, centroid_y)

            return (centroid_x, centroid_y)  # Return the centroid of the bounding box

        else:
            return None

    def _look_for_object(self, semantic_class, debug):
        sources = self.spot.image_client.list_image_sources()

        for source in sources:
            if (
                "depth" in source.name or source.name == "hand_image"
            ):  # "hand_image" is only in greyscale, "hand_color_image" is RGB
                continue

            image_source = source.name
            print("Getting image from source:", image_source)
            image, img = self.spot.get_image_RGB(view=image_source)

            rotate = 0

            if (
                "frontleft_fisheye_image" in image_source
                or "frontright_fisheye_image" in image_source
            ):
                rotate = 1  # cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            elif "right_fisheye_image" in image_source:
                rotate = 2  # cv2.rotate(img, cv2.ROTATE_180)

            xy = self._get_centroid(img, semantic_class, rotate=rotate, debug=debug)
            print("Found object centroid:", xy)
            if xy is None:
                print(f"Object not found in {image_source}.")
                continue
            else:
                return xy, image, img, image_source

        return None, None, None, None

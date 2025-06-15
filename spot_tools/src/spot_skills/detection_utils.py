from copy import copy

import cv2
import numpy as np
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
        custom_classes = ["", "bag", "wood block", "pipe"]
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

    def return_centroid(self, img_source, semantic_class, debug, feedback):
        image, img = self.spot.get_image_RGB(view=img_source)

        xy = self._get_centroid(
            img, semantic_class, rotate=0, debug=debug, feedback=feedback
        )
        if xy is None:
            print("Object not found in first image. Looking around!")
            xy, image, img, image_source = self._look_for_object(
                semantic_class, debug=debug, feedback=feedback
            )

            if xy is None:
                print("Object not found near robot")

        return xy, image

    def _get_centroid(self, img, semantic_class, rotate, debug, feedback):
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
                if box_height > 0.5 * img.shape[0]:
                    continue
                box_width = box.xyxy[0][2] - box.xyxy[0][0]  # width of the bounding box
                if (
                    box_width > 0.5 * img.shape[1]
                ):  # If the box is more than half the width of the image, skip it
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

            if feedback is not None:
                annotated_img = copy(img)

                feedback.bounding_box_detection_feedback(
                    annotated_img,
                    centroid_x,
                    centroid_y,
                    semantic_class,
                    best_confidence,
                )

            return centroid_x, centroid_y  # Return the centroid of the bounding box

        else:
            if feedback is not None:
                annotated_img = copy(img)

                feedback.bounding_box_detection_feedback(
                    annotated_img,
                    None,
                    None,
                    None,
                    None,
                )

            return None

    def _look_for_object(self, semantic_class, debug, feedback):
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

            xy = self._get_centroid(
                img, semantic_class, rotate=rotate, debug=debug, feedback=feedback
            )
            print("Found object centroid:", xy)
            if xy is None:
                print(f"Object not found in {image_source}.")
                continue
            else:
                return xy, image, img, image_source

        return None, None, None, None


class SemanticDetector(Detector):
    def __init__(self, spot, labelspace_map):
        super().__init__(spot)

        if self.spot.semantic_name_to_id is None:
            raise Exception(
                "Semantic names must be mapped to their id's (semantic_name_to_id is undefined)."
            )

        self.semantic_ids_to_grab = []

    def set_up_detector(self, semantic_class):
        if self.spot.labelspace_map is not None:
            self.semantic_ids_to_grab = (
                self.spot.labelspace_map[semantic_class]
                + self.spot.labelspace_map["clothes"]
                + self.spot.labelspace_map["bag"]
            )
        else:
            self.semantic_ids_to_grab = [
                self.spot.semantic_name_to_id[semantic_class],
                self.spot.semantic_name_to_id["bag"],
                self.spot.semantic_name_to_id["clothes"],
            ]

    def return_centroid(self, img_source, semantic_class, debug, feedback):
        image, img = self.spot.get_image_alt(view=img_source)
        semantic_image = self.spot.segment_image(img)

        # Convert to grayscale if needed
        if len(semantic_image.shape) == 3:
            gray = cv2.cvtColor(semantic_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = semantic_image

        # Create a binary mask where the class of interest is white and the rest is black
        mask = np.zeros_like(gray)
        # mask[gray == class_index] = 255
        mask[np.isin(gray, self.semantic_ids_to_grab)] = 255

        xy = self._get_centroid(semantic_image, self.semantic_ids_to_grab, img)

        if xy is None:
            print("Object not found in first image. Looking around!")
            xy, image, img, image_source = self._look_for_object(
                self.semantic_ids_to_grab
            )
            if xy is None:
                print("Object not found near robot.")

        return xy, image

    def _get_centroid(self, segmented_image, class_indices, image):
        """Get the centroid of a class in a segmented image."""

        # Convert to grayscale if needed
        if len(segmented_image.shape) == 3:
            gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = segmented_image

        # Create a binary mask where the class of interest is white and the rest is black
        mask = np.zeros_like(gray)
        # mask[gray == class_index] = 255
        mask[np.isin(gray, class_indices)] = 255

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None  # No contour found for the class

        # Assuming we take the largest contour if multiple are found
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate moments for the largest contour
        M = cv2.moments(largest_contour)

        # Calculate centroid using moments
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            seg_image_size = segmented_image.shape
            image_size = image.shape
            x_scale = image_size[1] / seg_image_size[1]
            y_scale = image_size[0] / seg_image_size[0]

            return (cX * x_scale, cY * y_scale)
        else:
            return None  # Centroid calculation failed

    def _look_for_object(self, semantic_ids):
        """Look for an object in the image sources. Return the centroid of the object, and the image source."""

        sources = self.spot.image_client.list_image_sources()

        for source in sources:
            if "depth" in source.name:
                continue
            image_source = source.name
            image, img = self.spot.get_image_alt(view=image_source)

            rotate = 0
            if "front" in image_source or "hand_image" in image_source:
                rotate = 1
            elif "right_fisheye_image" in image_source:
                rotate = 2

            semantic_image = self.spot.segment_image(img, rotate=rotate, show=False)
            xy = self.get_class_centroid(semantic_image, semantic_ids, img)
            print("Found object centroid:", xy)
            if xy is None:
                print(f"Object not found in {image_source}.")
                continue
            else:
                return xy, image, img, image_source

        return None, None, None, None

import cv2
import numpy as np
import os

class ObjectDetector:
    def __init__(self, config_path, weights_path, labels_path):
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, image_path, confidence_threshold=0.5, nms_threshold=0.4):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        (H, W) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.ln)

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > confidence_threshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

        results = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in np.random.uniform(0, 255, size=3)]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = f"{self.labels[class_ids[i]]}: {confidences[i]:.4f}"
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                results.append({"label": self.labels[class_ids[i]], "confidence": confidences[i], "box": [x, y, w, h]})

        return image, results

if __name__ == "__main__":
    # These paths would typically point to pre-trained YOLO files
    # For demonstration, we'll use dummy paths. You would need to download these.
    # For example, from: https://pjreddie.com/darknet/yolo/
    config_path = "yolov3.cfg"  # Placeholder
    weights_path = "yolov3.weights" # Placeholder
    labels_path = "coco.names" # Placeholder

    # Create dummy files for demonstration purposes if they don't exist
    # In a real scenario, these would be downloaded.
    for path in [config_path, weights_path, labels_path]:
        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write("dummy content for " + path)

    detector = ObjectDetector(config_path, weights_path, labels_path)
    
    # Create a dummy image for testing
    dummy_image_path = "dummy_image.jpg"
    if not os.path.exists(dummy_image_path):
        dummy_image = np.zeros((300, 500, 3), dtype=np.uint8)
        cv2.putText(dummy_image, "Test Image", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(dummy_image_path, dummy_image)

    try:
        detected_image, detections = detector.detect_objects(dummy_image_path)
        print("Detections:", detections)
        # cv2.imshow("Detected Image", detected_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    except FileNotFoundError as e:
        print(e)
        print("Please ensure yolov3.cfg, yolov3.weights, coco.names, and dummy_image.jpg are available.")


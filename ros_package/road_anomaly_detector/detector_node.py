#!/usr/bin/env python3
"""
road_anomaly_detector / detector_node.py

Subscribes to the camera topic published by the mobile_sensor_logger package
(camera_node -> /camera/image_raw/compressed, sensor_msgs/msg/CompressedImage,
JPEG bytes) and runs a YOLO road-anomaly-detection model on each frame.

Detections are published as vision_msgs/Detection2DArray on
/road_anomaly/detections so any ROS 2 consumer (logging, visualization,
navigation) can subscribe without depending on this package directly.

Class mapping (fixed for this thesis):
    0: D00  - Longitudinal crack
    1: D10  - Transverse crack
    2: D20  - Alligator crack
    3: D40  - Pothole

Usage:
    ros2 run road_anomaly_detector detector_node --ros-args \
        -p weights_path:=/path/to/best.pt \
        -p image_topic:=/camera/image_raw/compressed \
        -p conf_threshold:=0.25

If the upstream camera node is launched with publish_raw:=true (i.e. it also
publishes /camera/image_raw as sensor_msgs/msg/Image, bgr8), set
    -p use_compressed:=false
    -p image_topic:=/camera/image_raw
to subscribe to the raw topic instead.
"""

import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import CompressedImage, Image
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from ultralytics import YOLO


CLASS_NAMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}


class RoadAnomalyDetectorNode(Node):
    def __init__(self):
        super().__init__("road_anomaly_detector")

        # ---- Parameters ----------------------------------------------------
        self.declare_parameter("weights_path", "")
        self.declare_parameter("image_topic", "/camera/image_raw/compressed")
        self.declare_parameter("use_compressed", True)
        self.declare_parameter("detections_topic", "/road_anomaly/detections")
        self.declare_parameter("conf_threshold", 0.25)
        self.declare_parameter("device", "cpu")  # "cpu", "0" for cuda:0, etc.
        self.declare_parameter("publish_annotated", False)

        weights_path = self.get_parameter("weights_path").get_parameter_value().string_value
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.use_compressed = self.get_parameter("use_compressed").get_parameter_value().bool_value
        detections_topic = self.get_parameter("detections_topic").get_parameter_value().string_value
        self.conf_threshold = self.get_parameter("conf_threshold").get_parameter_value().double_value
        device = self.get_parameter("device").get_parameter_value().string_value
        self.publish_annotated = self.get_parameter("publish_annotated").get_parameter_value().bool_value

        if not weights_path:
            self.get_logger().error(
                "No 'weights_path' parameter set. Launch with "
                "-p weights_path:=/path/to/best.pt"
            )
            raise RuntimeError("weights_path parameter is required")

        # ---- Model -----------------------------------------------------------
        self.get_logger().info(f"Loading YOLO weights from: {weights_path}")
        self.model = YOLO(weights_path)
        self.device = device
        self.get_logger().info(f"Model loaded. Running inference on device='{device}'.")

        # ---- QoS (matches typical sensor-stream defaults: best-effort, small queue) ----
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ---- Subscriber (compressed by default, matches the logger's live-node output) ----
        if self.use_compressed:
            self.sub = self.create_subscription(
                CompressedImage, image_topic, self.compressed_callback, qos
            )
            self.get_logger().info(f"Subscribed to CompressedImage on '{image_topic}'")
        else:
            self.sub = self.create_subscription(
                Image, image_topic, self.raw_callback, qos
            )
            self.get_logger().info(f"Subscribed to raw Image on '{image_topic}'")

        # ---- Publisher --------------------------------------------------------
        self.detections_pub = self.create_publisher(Detection2DArray, detections_topic, 10)
        self.get_logger().info(f"Publishing detections on '{detections_topic}'")

        if self.publish_annotated:
            self.annotated_pub = self.create_publisher(
                CompressedImage, "/road_anomaly/annotated/compressed", qos
            )

        self.frame_count = 0
        self.processed_since_log = 0

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    def compressed_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # BGR
        if frame is None:
            self.get_logger().warn("Failed to decode incoming CompressedImage frame; skipping.")
            return
        self._run_inference(frame, msg.header)

    def raw_callback(self, msg: Image):
        # Assumes bgr8 encoding, matching the logger's /camera/image_raw convention.
        if msg.encoding != "bgr8":
            self.get_logger().warn(
                f"Expected bgr8 encoding, got '{msg.encoding}'. Attempting to interpret as bgr8 anyway."
            )
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self._run_inference(frame, msg.header)

    # -------------------------------------------------------------------------
    # Core inference + publish
    # -------------------------------------------------------------------------
    def _run_inference(self, frame: np.ndarray, header):
        t0 = time.time()
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        inference_ms = (time.time() - t0) * 1000.0

        detections_msg = Detection2DArray()
        detections_msg.header = header  # preserve upstream timestamp/frame_id

        r = results[0]
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x_center, y_center, w, h = box.xywh[0].tolist()

            det = Detection2D()
            det.header = header
            det.bbox.center.position.x = float(x_center)
            det.bbox.center.position.y = float(y_center)
            det.bbox.size_x = float(w)
            det.bbox.size_y = float(h)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = CLASS_NAMES.get(cls_id, str(cls_id))
            hyp.hypothesis.score = conf
            det.results.append(hyp)

            detections_msg.detections.append(det)

        self.detections_pub.publish(detections_msg)

        if self.publish_annotated:
            annotated = r.plot()  # BGR np.ndarray with boxes drawn
            success, encoded = cv2.imencode(".jpg", annotated)
            if success:
                out_msg = CompressedImage()
                out_msg.header = header
                out_msg.format = "jpeg"
                out_msg.data = encoded.tobytes()
                self.annotated_pub.publish(out_msg)

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f"Frame {self.frame_count}: {len(detections_msg.detections)} detections, "
                f"{inference_ms:.1f} ms inference"
            )


def main(args=None):
    rclpy.init(args=args)
    node = RoadAnomalyDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

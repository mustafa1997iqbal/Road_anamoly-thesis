# road_anomaly_detector

ROS 2 (ament_python) node that subscribes to the live camera stream from the
`mobile_sensor_logger` package and runs a YOLO road-anomaly-detection model
(trained for this thesis) on each frame, publishing results as
`vision_msgs/Detection2DArray`.

This package provides the **perception** side of the pipeline. Sensor
acquisition, GPS/IMU, and ROS 2 publishing of the raw camera stream are
handled entirely by `mobile_sensor_logger` (a separate package/thesis) — this
node only consumes its `/camera/image_raw*` topics and does not touch
hardware directly.

## Class mapping

| ID | Code | Description         |
|----|------|----------------------|
| 0  | D00  | Longitudinal crack   |
| 1  | D10  | Transverse crack     |
| 2  | D20  | Alligator crack      |
| 3  | D40  | Pothole              |

## Install

```bash
mkdir -p ~/ros2_ws/src
ln -s /path/to/road_anomaly_detector ~/ros2_ws/src/
cd ~/ros2_ws
pip install ultralytics opencv-python numpy --break-system-packages   # if not already installed
colcon build --symlink-install --packages-select road_anomaly_detector
source install/setup.bash
```

## Run

### 1. Start the camera stream (from mobile_sensor_logger)

```bash
ros2 launch mobile_sensor_logger sensors.launch.py camera_mock:=false
```

By default this publishes `/camera/image_raw/compressed`
(`sensor_msgs/msg/CompressedImage`, JPEG). Add `publish_raw:=true` on that
launch if you want the raw `/camera/image_raw` (`bgr8`) topic instead.

### 2. Start the detector

```bash
ros2 launch road_anomaly_detector detector.launch.py \
    weights_path:=/absolute/path/to/best.pt \
    conf_threshold:=0.25
```

To use the Large offline-tier model instead of the Nano edge model, just
point `weights_path` at that checkpoint — the node itself is
architecture-agnostic (any `.pt` file loadable via `ultralytics.YOLO(...)`
works).

To subscribe to the raw (uncompressed) topic instead:

```bash
ros2 launch road_anomaly_detector detector.launch.py \
    weights_path:=/absolute/path/to/best.pt \
    image_topic:=/camera/image_raw \
    use_compressed:=false
```

### 3. Inspect detections

```bash
ros2 topic echo /road_anomaly/detections
```

Each `Detection2D` in the array carries:
- `bbox.center.position.{x,y}`, `bbox.size_x`, `bbox.size_y` — pixel-space bounding box (center-based, matches YOLO's native `xywh`)
- `results[0].hypothesis.class_id` — one of `D00`/`D10`/`D20`/`D40`
- `results[0].hypothesis.score` — confidence

The published `header` is copied directly from the incoming image message, so
detections stay time-synchronized with the GPS/IMU streams from
`mobile_sensor_logger` for anyone downstream who wants to geo-tag them.

### Optional: visualize

```bash
ros2 launch road_anomaly_detector detector.launch.py \
    weights_path:=/absolute/path/to/best.pt \
    publish_annotated:=true
```

Publishes an annotated JPEG stream on `/road_anomaly/annotated/compressed`
(boxes drawn on frame) for `rqt_image_view` or similar.

## Parameters

| Parameter            | Default                          | Description |
|-----------------------|-----------------------------------|-------------|
| `weights_path`         | *(required)*                     | Path to trained `.pt` weights |
| `image_topic`          | `/camera/image_raw/compressed`   | Camera topic to subscribe to |
| `use_compressed`       | `true`                           | `true` = CompressedImage, `false` = raw Image |
| `conf_threshold`       | `0.25`                           | Minimum confidence to publish a detection |
| `device`               | `cpu`                            | `cpu`, `0` (cuda:0), etc. |
| `publish_annotated`    | `false`                          | Also publish an annotated debug image stream |

## Offline / batch inference (non-ROS)

For generating the thesis's required output format
(`image_path, object_type, confidence, bbox_x, bbox_y, bbox_w, bbox_h`) over a
static folder of images rather than a live topic, see
`scripts/batch_inference_csv.py` in the main detection repository (not part
of this ROS package).

## Topic name

`camera_node.py` publishes on `TOPIC_IMAGE_COMPRESSED` from
`src/tools/ros2_common.py`, which resolves to `/camera/image_raw/compressed`
— matching this package's default `image_topic` value, so no override should
be needed.

## Frame-rate mismatch on Pi hardware

`camera_node.py` publishes at a fixed rate (30 fps by default, uncapped by
consumers) via a ROS timer. On Raspberry Pi CPU, YOLO inference — especially
above 33 ms/frame — will fall behind that rate. This node uses best-effort
QoS with `depth=1`, so it will **silently drop frames it can't keep up with**
rather than queue or block the publisher; this is the correct behavior for a
live system, but it means the detector will process a fraction of frames on
Pi hardware, not all of them. Watch the periodic log line
(`Frame N: ... detections, ... ms inference`) to see actual achieved rate,
and treat the real per-frame inference latency reported there as the number
to cite for on-device performance, not the camera's nominal fps.

## Notes on hardware / real-time performance

This node was validated against the Nano (edge-tier) and Large (offline-tier)
checkpoints described in the thesis. See the thesis Results chapter for the
speed/accuracy tradeoff between the two; the Nano checkpoint is the
recommended default for live, on-robot use given its comparable-or-better
accuracy on the self-collected bicycle-lane domain and its substantially
lower inference latency.

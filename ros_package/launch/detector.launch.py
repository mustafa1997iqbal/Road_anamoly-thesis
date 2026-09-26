from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    weights_path_arg = DeclareLaunchArgument(
        "weights_path",
        description="Absolute path to the trained YOLO .pt weights file (required).",
    )
    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="/camera/image_raw/compressed",
        description="Camera topic to subscribe to.",
    )
    use_compressed_arg = DeclareLaunchArgument(
        "use_compressed",
        default_value="true",
        description="True for CompressedImage, false for raw sensor_msgs/Image (bgr8).",
    )
    conf_threshold_arg = DeclareLaunchArgument(
        "conf_threshold",
        default_value="0.25",
        description="Minimum detection confidence to publish.",
    )
    device_arg = DeclareLaunchArgument(
        "device",
        default_value="cpu",
        description="Inference device: 'cpu', '0' (cuda:0), etc.",
    )
    publish_annotated_arg = DeclareLaunchArgument(
        "publish_annotated",
        default_value="false",
        description="Also publish an annotated JPEG stream for visualization/debugging.",
    )

    detector_node = Node(
        package="road_anomaly_detector",
        executable="detector_node",
        name="road_anomaly_detector",
        output="screen",
        parameters=[
            {
                "weights_path": LaunchConfiguration("weights_path"),
                "image_topic": LaunchConfiguration("image_topic"),
                "use_compressed": LaunchConfiguration("use_compressed"),
                "conf_threshold": LaunchConfiguration("conf_threshold"),
                "device": LaunchConfiguration("device"),
                "publish_annotated": LaunchConfiguration("publish_annotated"),
            }
        ],
    )

    return LaunchDescription(
        [
            weights_path_arg,
            image_topic_arg,
            use_compressed_arg,
            conf_threshold_arg,
            device_arg,
            publish_annotated_arg,
            detector_node,
        ]
    )

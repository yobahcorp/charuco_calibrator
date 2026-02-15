"""ROS 2 node that wraps the ChArUco calibrator."""

from __future__ import annotations

import rclpy
from rclpy.node import Node

from charuco_calibrator.config import AppConfig
from charuco_calibrator.main import run


class CalibratorParameterNode(Node):
    """Thin node that declares ROS parameters and builds an AppConfig."""

    def __init__(self) -> None:
        super().__init__("charuco_calibrator")

        # Declare parameters with defaults
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("aruco_dict", "DICT_4X4_50")
        self.declare_parameter("squares_x", 5)
        self.declare_parameter("squares_y", 7)
        self.declare_parameter("square_length", 0.04)
        self.declare_parameter("marker_length", 0.03)
        self.declare_parameter("min_corners", 6)
        self.declare_parameter("min_blur_var", 50.0)
        self.declare_parameter("min_score", 0.3)
        self.declare_parameter("capture_cooldown_ms", 1500)
        self.declare_parameter("output_dir", "calibration_output")
        self.declare_parameter("camera_name", "camera")
        self.declare_parameter("auto_capture", False)

    def build_config(self) -> AppConfig:
        """Build an AppConfig from current parameter values."""
        cfg = AppConfig()

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value

        cfg.board.aruco_dict = self.get_parameter("aruco_dict").get_parameter_value().string_value
        cfg.board.squares_x = self.get_parameter("squares_x").get_parameter_value().integer_value
        cfg.board.squares_y = self.get_parameter("squares_y").get_parameter_value().integer_value
        cfg.board.square_length = self.get_parameter("square_length").get_parameter_value().double_value
        cfg.board.marker_length = self.get_parameter("marker_length").get_parameter_value().double_value

        cfg.thresholds.min_corners = self.get_parameter("min_corners").get_parameter_value().integer_value
        cfg.thresholds.min_blur_var = self.get_parameter("min_blur_var").get_parameter_value().double_value
        cfg.thresholds.min_score = self.get_parameter("min_score").get_parameter_value().double_value
        cfg.thresholds.capture_cooldown_ms = self.get_parameter("capture_cooldown_ms").get_parameter_value().integer_value

        cfg.output.output_dir = self.get_parameter("output_dir").get_parameter_value().string_value
        cfg.output.camera_name = self.get_parameter("camera_name").get_parameter_value().string_value

        cfg.auto_capture = self.get_parameter("auto_capture").get_parameter_value().bool_value

        # The only image source is the ROS topic
        cfg.source.ros_topic = image_topic

        return cfg

    def log_config(self, cfg: AppConfig) -> None:
        """Log a summary of the active configuration."""
        self.get_logger().info(
            f"Board: {cfg.board.squares_x}x{cfg.board.squares_y}, "
            f"dict={cfg.board.aruco_dict}, "
            f"square={cfg.board.square_length}m, marker={cfg.board.marker_length}m"
        )
        self.get_logger().info(
            f"Thresholds: min_corners={cfg.thresholds.min_corners}, "
            f"min_blur_var={cfg.thresholds.min_blur_var}, "
            f"min_score={cfg.thresholds.min_score}, "
            f"cooldown={cfg.thresholds.capture_cooldown_ms}ms"
        )
        self.get_logger().info(
            f"Output: dir={cfg.output.output_dir}, camera={cfg.output.camera_name}"
        )
        self.get_logger().info(f"Image topic: {cfg.source.ros_topic}")
        self.get_logger().info(f"Auto-capture: {cfg.auto_capture}")


def main(args=None) -> None:
    rclpy.init(args=args)

    try:
        node = CalibratorParameterNode()
        cfg = node.build_config()
        node.log_config(cfg)

        # Destroy the parameter node before entering the calibration loop,
        # which creates its own ROS subscription node internally.
        node.destroy_node()

        rc = run(cfg)
    except KeyboardInterrupt:
        rc = 0
    finally:
        rclpy.shutdown()

    raise SystemExit(rc)


if __name__ == "__main__":
    main()

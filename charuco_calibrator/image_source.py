"""Image source abstraction: OpenCV camera, video file, and optional ROS 2 adapter."""

from __future__ import annotations

import abc
import time
from typing import Optional

import cv2
import numpy as np

from .config import SourceConfig


class ImageSource(abc.ABC):
    """Abstract base class for image sources."""

    @abc.abstractmethod
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read the next frame.

        Returns:
            (success, frame) where frame is BGR np.ndarray or None.
        """
        ...

    @abc.abstractmethod
    def release(self) -> None:
        """Release underlying resources."""
        ...

    @property
    @abc.abstractmethod
    def is_open(self) -> bool:
        ...


class CameraSource(ImageSource):
    """OpenCV VideoCapture-based image source (webcam or video file)."""

    def __init__(self, camera_id: int = 0, video_path: Optional[str] = None,
                 width: int = 0, height: int = 0) -> None:
        if video_path:
            self._cap = cv2.VideoCapture(video_path)
        else:
            self._cap = cv2.VideoCapture(camera_id)

        if width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        return True, frame

    def release(self) -> None:
        self._cap.release()

    @property
    def is_open(self) -> bool:
        return self._cap.isOpened()


class RosImageSource(ImageSource):
    """ROS 2 image topic subscriber with lazy rclpy import.

    Thread-safe: the ROS callback stores frames in a buffer that read() consumes.
    """

    def __init__(self, topic: str, width: int = 0, height: int = 0) -> None:
        self._topic = topic
        self._width = width
        self._height = height
        self._frame: Optional[np.ndarray] = None
        self._frame_stamp: float = 0.0
        self._lock = None  # set in _init_ros
        self._node = None
        self._sub = None
        self._bridge = None
        self._init_ros()

    def _init_ros(self) -> None:
        """Lazily import rclpy and cv_bridge."""
        import threading

        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Image
            from cv_bridge import CvBridge
        except ImportError as e:
            raise ImportError(
                f"ROS 2 dependencies not available: {e}. "
                "Install rclpy and cv_bridge, or use --camera instead."
            ) from e

        self._lock = threading.Lock()
        self._bridge = CvBridge()

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node("charuco_calibrator_source")
        self._sub = self._node.create_subscription(
            Image, self._topic, self._image_callback, 10
        )

        # Spin in a background thread
        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self._node,), daemon=True
        )
        self._spin_thread.start()

    def _image_callback(self, msg) -> None:
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if self._width > 0 and self._height > 0:
            frame = cv2.resize(frame, (self._width, self._height))
        with self._lock:
            self._frame = frame
            self._frame_stamp = time.time()

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._lock is None:
            return False, None
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self) -> None:
        if self._node is not None:
            self._node.destroy_node()

    @property
    def is_open(self) -> bool:
        return self._node is not None


def create_source(cfg: SourceConfig) -> ImageSource:
    """Factory: create the appropriate ImageSource from config."""
    if cfg.ros_topic:
        return RosImageSource(cfg.ros_topic, width=cfg.width, height=cfg.height)
    return CameraSource(
        camera_id=cfg.camera_id,
        video_path=cfg.video_path,
        width=cfg.width,
        height=cfg.height,
    )

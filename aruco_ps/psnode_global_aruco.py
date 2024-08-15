import cv2
import rclpy
from rclpy.node import Node
import numpy as np
from .perf_utils import TimerTicTok
from aruco_interface.msg import ImageMarkers
from rclpy.qos import QoSProfile

class GlobalArucoPSNode(Node):
    def __init__(self):
        super().__init__('psnode_global_aruco')

        # tic-tok timer
        self.tictok = TimerTicTok()

        qos_profile = QoSProfile(depth=10)
        self.aruco_subs = self.create_subscription(
                ImageMarkers,
                '/cam/aruco',
                self.handle_aruco,
                qos_profile)
        
    def handle_aruco(self, msg):
        self.get_logger().info(f"Received ImageMarkers: {msg}")
    


def main(args=None):
    rclpy.init(args=args)
    node = GlobalArucoPSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    


if __name__ == "__main__":
    main()

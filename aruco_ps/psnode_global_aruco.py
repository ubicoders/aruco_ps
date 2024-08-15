import cv2
import rclpy
from rclpy.node import Node
import numpy as np
from .perf_utils import TimerTicTok
from aruco_interface.msg import ImageMarkers
from rclpy.qos import QoSProfile
from .global_aruco_utils import get_aae_T,check_nan_np
# format numpy to print all 3 decimals
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

class GlobalArucoPSNode(Node):
    def __init__(self):
        super().__init__('psnode_global_aruco')

        # tic-tok timer
        self.tictok = TimerTicTok()

        qos_profile = QoSProfile(depth=10)
        self.aruco_subs = self.create_subscription(
                ImageMarkers,
                '/cam/aruco_depth',
                self.handle_aruco,
                qos_profile)
        
    def handle_aruco(self, msg):
        self.tictok.update()
        self.tictok.pprint()

        aruco_marker_list = msg.aruco_markers_0
        eul_list = []
        xyz_list = []
        tvec_list = []
        for marker in aruco_marker_list:
            mid = marker.mid
            eul = np.array([marker.eul.x, marker.eul.y, marker.eul.z])
            xyz = np.array([marker.pc_xyz.x, marker.pc_xyz.y, marker.pc_xyz.z])
            tvec = np.array([marker.tvec.x, marker.tvec.y, marker.tvec.z])
            if (check_nan_np(eul) or check_nan_np(xyz) or check_nan_np(tvec)):
                print(f"Error: NaN detected in marker {mid}")
                continue
           
            eul_list.append(eul)
            xyz_list.append(xyz)
            tvec_list.append(tvec)


        # calcualte average
        if (len(eul_list) == 0 or len(xyz_list) == 0 or len(tvec_list) == 0):
            eul_avg = np.array([0., 0., 0.])
            xyz_avg = np.array([0., 0., 0.])
            tvec_avg = np.array([0., 0., 0.])
        eul_avg = np.mean(np.array(eul_list), axis=0)
        xyz_avg = np.mean(np.array(xyz_list), axis=0)
        tvec_avg = np.mean(np.array(tvec_list), axis=0)

        if (check_nan_np(eul_avg) or check_nan_np(xyz_avg) or check_nan_np(tvec_avg)):
            print(f"Error: NaN detected in average")
            aaeT = np.eye(4)
        else:
            aaeT = get_aae_T(eul_avg, xyz_avg)
        self.get_logger().info(f"Transform: \n{aaeT}")
            
    


def main(args=None):
    rclpy.init(args=args)
    node = GlobalArucoPSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    


if __name__ == "__main__":
    main()

import cv2
import rclpy
from rclpy.node import Node
import numpy as np
from .perf_utils import TimerTicTok
from aruco_interface.msg import ImageMarkers
from rclpy.qos import QoSProfile
from .global_aruco_utils import get_aae_T, check_nan_np, cveul_aaeeul, cvXYZ_aaeXYZ
from geometry_msgs.msg import Vector3
# format numpy to print all 3 decimals
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.low_pass_vec3 = np.zeros(3)
    
    def update(self, new_vec3):
        if (check_nan_np(new_vec3)):
            return self.low_pass_vec3
        self.low_pass_vec3 = self.alpha * self.low_pass_vec3 + (1 - self.alpha) * new_vec3
        return self.low_pass_vec3



class BodyArucoPSNode(Node):
    def __init__(self):
        super().__init__('psnode_body_aruco')

        # tic-tok timer
        self.tictok = TimerTicTok()

        # subscriber to aruco depth image
        qos_profile = QoSProfile(depth=10)
        self.aruco_subs = self.create_subscription(
                ImageMarkers,
                '/cam/aruco',
                self.handle_aruco,
                qos_profile)
        
        # publisher for cam global pose
        self.obj_gp_pub = self.create_publisher(Vector3, '/cam/obj_pose', qos_profile)
        self.lp_pos = LowPassFilter(0.9)
        
        
    def handle_aruco(self, msg):
        # self.tictok.update()
        # self.tictok.pprint()

        aruco_marker_list = msg.aruco_markers_0
        tvec_list = []
        for marker in aruco_marker_list:
            mid = marker.mid
            if mid not in [100, 101, 102, 103, 104, 105]:
                continue
            tvec_aae = marker.tvec_aae
            tvec = np.array([tvec_aae.x, tvec_aae.y, tvec_aae.z])
            if (check_nan_np(tvec)):
                print(f"Error: NaN detected in marker {mid}")
                continue           
            tvec_list.append(tvec)

        # calcualte average pose
        if (len(tvec_list) == 0):
            tvec_avg = np.array([0., 0., 0.])
        

        tvec_avg = np.mean(np.array(tvec_list), axis=0)
        #self.get_logger().info(f"Average Tvec: {tvec_avg}") 
        
        T_cam2global = np.eye(4)
        T_cam2global[:3, 3] = np.array([-4.3, 0.0, 0.0])

        T_m2cam = np.eye(4)
        T_m2cam[:3, 3] = tvec_avg

        T_m2global = T_cam2global @ T_m2cam
        obj_gp = T_m2global[:3, 3]
        obj_gp_msg = Vector3()
        obj_gp_msg.x = obj_gp[0]
        obj_gp_msg.y = obj_gp[1]
        obj_gp_msg.z = obj_gp[2]
        self.obj_gp_pub.publish(obj_gp_msg)



def main(args=None):
    rclpy.init(args=args)
    node = BodyArucoPSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

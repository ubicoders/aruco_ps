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
        self.cam_gp_pub = self.create_publisher(Vector3, '/cam/obj_pose', qos_profile)

        
        self.lp_pos = LowPassFilter(0.9)
        self.lp_eul = LowPassFilter(0.9)
        
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

        # calcualte average pose and rotation
        if (len(eul_list) == 0 or len(xyz_list) == 0 or len(tvec_list) == 0):
            eul_avg = np.array([0., 0., 0.])
            xyz_avg = np.array([0., 0., 0.])
            tvec_avg = np.array([0., 0., 0.])
        eul_avg = np.mean(np.array(eul_list), axis=0)
        xyz_avg = np.mean(np.array(xyz_list), axis=0)
        tvec_avg = np.mean(np.array(tvec_list), axis=0)

        # self.get_logger().info(f"Average Euler: {eul_avg*180.0/np.pi}")
        # self.get_logger().info(f"Average XYZ: {xyz_avg}")
        self.get_logger().info(f"Average Tvec: {tvec_avg}")


        self.lp_eul.update(eul_avg)
        self.lp_pos.update(xyz_avg)

        eul_aae = cveul_aaeeul(self.lp_eul.low_pass_vec3)
        xyz_aae = cvXYZ_aaeXYZ(self.lp_pos.low_pass_vec3)
        # self.get_logger().info(f"Filtered Euler CV: {self.lp_eul.low_pass_vec3*180.0/np.pi}")
        # self.get_logger().info(f"Filtered Euler AA: {cveul_aaeeul(self.lp_eul.low_pass_vec3)*180.0/np.pi}")
        # self.get_logger().info(f"Filtered XYZ CV: {self.lp_pos.low_pass_vec3}")
        # self.get_logger().info(f"Filtered XYZ AA: {cvXYZ_aaeXYZ(self.lp_pos.low_pass_vec3)}")

        arT_board2cam = get_aae_T(eul_aae, xyz_aae)
        # self.get_logger().info(f"Transform: \n{arT_board2cam}")

        aruco_T_b2g = np.eye(4)
        aruco_T_b2g[0:3, 0:3] = np.eye(3)
        aruco_T_b2g[0:3, 3] = np.array([4.3, 0.0, -1])

        cam_global_T = aruco_T_b2g @ np.linalg.inv(arT_board2cam)
        cam_gp = cam_global_T[0:3, 3]
        cam_gp_msg = Vector3()
        cam_gp_msg.x = cam_gp[0]
        cam_gp_msg.y = cam_gp[1]
        cam_gp_msg.z = cam_gp[2]
        self.cam_gp_pub.publish(cam_gp_msg)
        # self.get_logger().info(f"Cam Global Pose: \n{cam_global_pose}")



def main(args=None):
    rclpy.init(args=args)
    node = BodyArucoPSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

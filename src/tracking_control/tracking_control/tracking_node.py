import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import math

## Functions for quaternion and rotation matrix conversion
## The code is adapted from the general_robotics_toolbox package
## Code reference: https://github.com/rpiRobotics/rpi_general_robotics_toolbox_py
def hat(k):
    """
    Returns a 3 x 3 cross product matrix for a 3 x 1 vector

             [  0 -k3  k2]
     khat =  [ k3   0 -k1]
             [-k2  k1   0]

    :type    k: numpy.array
    :param   k: 3 x 1 vector
    :rtype:  numpy.array
    :return: the 3 x 3 cross product matrix
    """

    khat=np.zeros((3,3))
    khat[0,1]=-k[2]
    khat[0,2]=k[1]
    khat[1,0]=k[2]
    khat[1,2]=-k[0]
    khat[2,0]=-k[1]
    khat[2,1]=k[0]
    return khat

def q2R(q):
    """
    Converts a quaternion into a 3 x 3 rotation matrix according to the
    Euler-Rodrigues formula.
    
    :type    q: numpy.array
    :param   q: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :rtype:  numpy.array
    :return: the 3x3 rotation matrix    
    """
    
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2*q[0]*qhat + 2*qhat2
######################

def euler_from_quaternion(q):
    w=q[0]
    x=q[1]
    y=q[2]
    z=q[3]
    # euler from quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - z * x))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    return [roll,pitch,yaw]

class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.get_logger().info('Tracking Node Started')

        # Current detected poses in world frame
        self.obs_pose = None
        self.goal_pose = None

        # Save starting position in world frame
        self.start_pose_world = None

        # State machine
        self.state = "SEARCH"

        # ROS parameters
        self.declare_parameter('world_frame_id', 'odom')

        # Controller parameters
        self.goal_stop_distance = 0.30
        self.home_stop_distance = 0.15
        self.obstacle_avoid_distance = 0.60
        self.obstacle_front_angle = 0.9   # radians (~51 deg)

        self.k_linear = 0.6
        self.k_angular = 1.8
        self.k_avoid = 1.5

        self.max_linear_speed = 0.22
        self.max_angular_speed = 1.5
        self.search_angular_speed = 0.35

        # Create a transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publisher
        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)

        # Subscribers
        self.sub_detected_goal_pose = self.create_subscription(
            PoseStamped,
            'detected_color_object_pose',
            self.detected_obs_pose_callback,
            10
        )
        self.sub_detected_obs_pose = self.create_subscription(
            PoseStamped,
            'detected_color_goal_pose',
            self.detected_goal_pose_callback,
            10
        )

        # Timer at 100 Hz
        self.timer = self.create_timer(0.01, self.timer_update)

    def clamp(self, value, min_value, max_value):
        return max(min(value, max_value), min_value)

    def detected_obs_pose_callback(self, msg):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        # Simple filtering for noisy detections
        if np.linalg.norm(center_points) > 3.0 or abs(center_points[2]) > 0.7:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                odom_id,
                msg.header.frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )
            t_R = q2R(np.array([
                transform.transform.rotation.w,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z
            ]))
            cp_world = t_R @ center_points + np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return

        self.obs_pose = cp_world

    def detected_goal_pose_callback(self, msg):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        # Simple filtering for noisy detections
        if np.linalg.norm(center_points) > 3.0 or abs(center_points[2]) > 0.7:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                odom_id,
                msg.header.frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )
            t_R = q2R(np.array([
                transform.transform.rotation.w,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z
            ]))
            cp_world = t_R @ center_points + np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return

        self.goal_pose = cp_world

    def get_current_poses(self):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        try:
            # Transform world-frame points into base_footprint frame
            transform = self.tf_buffer.lookup_transform('base_footprint', odom_id, rclpy.time.Time())
            robot_world_x = transform.transform.translation.x
            robot_world_y = transform.transform.translation.y
            robot_world_z = transform.transform.translation.z
            robot_world_R = q2R([
                transform.transform.rotation.w,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z
            ])

            obstacle_pose = None
            goal_pose = None

            if self.obs_pose is not None:
                obstacle_pose = robot_world_R @ self.obs_pose + np.array([robot_world_x, robot_world_y, robot_world_z])

            if self.goal_pose is not None:
                goal_pose = robot_world_R @ self.goal_pose + np.array([robot_world_x, robot_world_y, robot_world_z])

        except TransformException as e:
            self.get_logger().error('Transform error: ' + str(e))
            return None, None

        return obstacle_pose, goal_pose

    def get_robot_pose_in_world(self):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        try:
            # Robot pose in world frame
            transform = self.tf_buffer.lookup_transform(odom_id, 'base_footprint', rclpy.time.Time())
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z
            return np.array([x, y, z])
        except TransformException as e:
            self.get_logger().error('Robot pose transform error: ' + str(e))
            return None

    def world_point_to_base(self, point_world):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        try:
            transform = self.tf_buffer.lookup_transform('base_footprint', odom_id, rclpy.time.Time())
            R = q2R([
                transform.transform.rotation.w,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z
            ])
            t = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            return R @ point_world + t
        except TransformException as e:
            self.get_logger().error('World-to-base transform error: ' + str(e))
            return None

    def stop_robot(self):
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.linear.y = 0.0
        cmd_vel.angular.z = 0.0
        self.pub_control_cmd.publish(cmd_vel)

    def timer_update(self):
        # Save the start pose once TF is available
        if self.start_pose_world is None:
            robot_pose = self.get_robot_pose_in_world()
            if robot_pose is not None:
                self.start_pose_world = robot_pose.copy()
                self.get_logger().info(
                    f"Saved start pose: x={self.start_pose_world[0]:.2f}, y={self.start_pose_world[1]:.2f}"
                )

        current_obs_pose, current_goal_pose = self.get_current_poses()

        cmd_vel = self.controller(current_obs_pose, current_goal_pose)

        self.pub_control_cmd.publish(cmd_vel)

    def controller(self, current_obs_pose, current_goal_pose):
        cmd_vel = Twist()

        # ---------- SEARCH STATE ----------
        if self.state == "SEARCH":
            # No goal seen yet: slowly rotate to search
            if current_goal_pose is None:
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = self.search_angular_speed
                return cmd_vel
            else:
                self.state = "GO_TO_GOAL"

        # ---------- GO TO GOAL ----------
        if self.state == "GO_TO_GOAL":
            if current_goal_pose is None:
                # Lost goal: search again
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = self.search_angular_speed
                return cmd_vel

            goal_x = current_goal_pose[0]
            goal_y = current_goal_pose[1]
            goal_dist = math.sqrt(goal_x**2 + goal_y**2)
            goal_ang = math.atan2(goal_y, goal_x)

            # Reached goal
            if goal_dist < self.goal_stop_distance:
                self.get_logger().info("Reached goal. Switching to RETURN_HOME.")
                self.state = "RETURN_HOME"
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
                return cmd_vel

            # Attractive command toward goal
            linear_cmd = self.k_linear * goal_dist
            angular_cmd = self.k_angular * goal_ang

            # Obstacle avoidance
            if current_obs_pose is not None:
                obs_x = current_obs_pose[0]
                obs_y = current_obs_pose[1]
                obs_dist = math.sqrt(obs_x**2 + obs_y**2)
                obs_ang = math.atan2(obs_y, obs_x)

                # If obstacle is close and generally in front of the robot, turn away
                if obs_dist < self.obstacle_avoid_distance and abs(obs_ang) < self.obstacle_front_angle:
                    if obs_ang >= 0.0:
                        # obstacle on left -> turn right
                        angular_cmd -= self.k_avoid * (1.0 / max(obs_dist, 0.1))
                    else:
                        # obstacle on right -> turn left
                        angular_cmd += self.k_avoid * (1.0 / max(obs_dist, 0.1))

                    # slow down near obstacle
                    linear_cmd *= 0.4

            cmd_vel.linear.x = self.clamp(linear_cmd, 0.0, self.max_linear_speed)
            cmd_vel.angular.z = self.clamp(angular_cmd, -self.max_angular_speed, self.max_angular_speed)
            return cmd_vel

        # ---------- RETURN HOME ----------
        if self.state == "RETURN_HOME":
            if self.start_pose_world is None:
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
                return cmd_vel

            home_pose_base = self.world_point_to_base(self.start_pose_world)
            if home_pose_base is None:
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
                return cmd_vel

            home_x = home_pose_base[0]
            home_y = home_pose_base[1]
            home_dist = math.sqrt(home_x**2 + home_y**2)
            home_ang = math.atan2(home_y, home_x)

            if home_dist < self.home_stop_distance:
                self.get_logger().info("Returned home. Stopping robot.")
                self.state = "DONE"
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
                return cmd_vel

            linear_cmd = self.k_linear * home_dist
            angular_cmd = self.k_angular * home_ang

            # Optional obstacle avoidance on the way home too
            if current_obs_pose is not None:
                obs_x = current_obs_pose[0]
                obs_y = current_obs_pose[1]
                obs_dist = math.sqrt(obs_x**2 + obs_y**2)
                obs_ang = math.atan2(obs_y, obs_x)

                if obs_dist < self.obstacle_avoid_distance and abs(obs_ang) < self.obstacle_front_angle:
                    if obs_ang >= 0.0:
                        angular_cmd -= self.k_avoid * (1.0 / max(obs_dist, 0.1))
                    else:
                        angular_cmd += self.k_avoid * (1.0 / max(obs_dist, 0.1))
                    linear_cmd *= 0.4

            cmd_vel.linear.x = self.clamp(linear_cmd, 0.0, self.max_linear_speed)
            cmd_vel.angular.z = self.clamp(angular_cmd, -self.max_angular_speed, self.max_angular_speed)
            return cmd_vel

        # ---------- DONE ----------
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        return cmd_vel

        ############################################

def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)
    # Create the node
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    # Destroy the node explicitly
    tracking_node.destroy_node()
    # Shutdown the ROS client library for Python
    rclpy.shutdown()
if __name__ == '__main__':
    main()

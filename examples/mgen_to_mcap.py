#!/usr/bin/env python3
"""
Convert a single MobilityGen replay directory into a ROS 2 MCAP bag.
Produces:
  - /pelvis_pose                        (geometry_msgs/PoseStamped)
  - /pelvis_odom                        (nav_msgs/Odometry)
  - /left_camera_pose                   (geometry_msgs/PoseStamped)
  - /right_camera_pose                  (geometry_msgs/PoseStamped)
  - /tf                                 (tf2_msgs/TFMessage)
  - /tf_static                          (tf2_msgs/TFMessage)
  - /joint_states                       (sensor_msgs/JointState)
  - /cmd_vel                            (geometry_msgs/msg/Twist)
  - /rgb/image_raw/<camera>             (sensor_msgs/Image, rgb8)
  - /rgb/camera_info/<camera>           (sensor_msgs/CameraInfo)
  - /segmentation/instance_id/<camera>  (sensor_msgs/Image, mono8)
  - /segmentation/semantic/<camera>     (sensor_msgs/Image, mono8)
  - /depth/image_raw/<camera>           (sensor_msgs/Image, mono16)
  - /depth/camera_info/<camera>         (sensor_msgs/CameraInfo)
  - /normals/image/<camera>             (sensor_msgs/Image, rgb8)
"""
import os
import glob
import argparse
import numpy as np
from PIL import Image
import math

import rclpy
from rclpy.serialization import serialize_message
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata

from sensor_msgs.msg import Image as ImgMsg, CameraInfo
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R


def usd_to_ros_pose(pos, ori):
    """
    Simply pass through the USD position, and reorder the USD quaternion [qx, qy, qz, qw]
    into ROS’s (w, x, y, z) tuple.
    
    Args:
      pos: 3‑float iterable [x_s, y_s, z_s]
      ori: 4‑float iterable [qw, qx, qy, qz]
    
    Returns:
      ( (x, y, z), (x, y, z, w) )
    """
    # passthrough position
    x_ros, y_ros, z_ros = pos

    # build the incoming rotation
    qw, qx, qy, qz = ori
    
    return (x_ros, y_ros, z_ros), (qx, qy, qz, qw)


# def quat_mul(q1, q2):
#     """
#     Hamilton product of q1 * q2.
#     q = (x,y,z,w)
#     """
#     x1, y1, z1, w1 = q1
#     x2, y2, z2, w2 = q2
#     x =  w1*x2 + x1*w2 + y1*z2 - z1*y2
#     y =  w1*y2 - x1*z2 + y1*w2 + z1*x2
#     z =  w1*z2 + x1*y2 - y1*x2 + z1*w2
#     w =  w1*w2 - x1*x2 - y1*y2 - z1*z2
#     return (x, y, z, w)


# def usd_to_ros_camera_frame(pos, ori):
#     """
#     Apply a fixed +90° rotation about the Z‐axis (instead of a matrix),
#     by quaternion multiplication.
#     """
#     # 1) passthrough translation
#     x_ros, y_ros, z_ros = pos

#     # 2) unpack USD→ROS quaternion from sim (qx,qy,qz,qw)
#     qx, qy, qz, qw = ori

#     # 3) build the “fix” quaternion: rotation of +90° about Z
#     half = math.pi/2  # 90°/2
#     q_fix = (0.0, 0.0, math.sin(half), math.cos(half))

#     # 4) multiply: q_ros = q_fix * q_usd
#     qx1, qy1, qz1, qw1 = quat_mul(q_fix, (qx, qy, qz, qw))

#     # return (x_ros, y_ros, z_ros), (qx1, qy1, qz1, qw1)
#     return (x_ros, y_ros, z_ros), (qx, qy, qz, qw)


# def usd_to_ros_camera_frame(pos, ori):
#     # 1) USD→ROS “passthrough” (no axis‐swap yet)
#     x, y, z = pos
#     R_usd = R.from_quat(ori).as_matrix()

#     # 2) your fix‐rotation, purely 3×3
#     R_fix = np.array([
#         [ 0,  1,  0],
#         [-1,  0,  0],
#         [ 0,  0,  1],
#     ], dtype=float)

#     # 3) apply fix only to the rotation
#     R_ros = R_fix.dot(R_usd)
#     qx, qy, qz, qw = R.from_matrix(R_ros).as_quat()

#     # 4) leave translation untouched
#     return (x, y, z), (qx, qy, qz, qw)


def usd_to_ros_camera_frame(pos, ori):
    # Pass‐through translation + USD→ROS reorder
    (x, y, z), (qx, qy, qz, qw) = usd_to_ros_pose(pos, ori)

    # Extract the camera’s current Euler angles (roll, pitch, yaw)
    ex, ey, ez = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=True)

    # Apply formulas *from zero*:
    # new_roll  = current_pitch
    # new_pitch = current_roll   
    # new_yaw   = - 90° - (180 - current_yaw)
    new_ex =   ey
    new_ey =   ex
    new_ez = - 90 - (180 - ez)

    # Re‑build the quaternion
    qx2, qy2, qz2, qw2 = R.from_euler('xyz', [new_ex, new_ey, new_ez], degrees=True).as_quat()

    return (x, y, z), (qx2, qy2, qz2, qw2)



def make_image_msg(npdata, encoding, stamp):
    bridge = CvBridge()
    # Choose the correct cv_bridge conversion
    if encoding == 'rgb8':
        return bridge.cv2_to_imgmsg(npdata, encoding='rgb8')
    elif encoding == 'mono16':
        return bridge.cv2_to_imgmsg(npdata, encoding='mono16')
    elif encoding == 'rgb32f':
        return bridge.cv2_to_imgmsg(npdata, encoding='32FC3')
    elif encoding == 'mono8':
        # ensure we have uint8, not int32
        npdata = npdata.astype('uint8')
        return bridge.cv2_to_imgmsg(npdata, encoding='mono8')
    else:
        return bridge.cv2_to_imgmsg(npdata, encoding=encoding)


# CameraInfo helper (plumb bob, no distortion)
def make_camera_info(width, height, frame_id):
    ci = CameraInfo()
    ci.header.frame_id  = frame_id
    # ci.header.stamp     = ts      
    ci.height           = height
    ci.width            = width
    ci.distortion_model = 'plumb_bob'
    ci.d = [0.0]*5

    fx = fy = 500.0
    cx, cy = width/2.0, height/2.0

    ci.k = [
        fx,  0.0, cx,
       0.0,  fy, cy,
       0.0, 0.0, 1.0,
    ]
    ci.r = [
       1.0, 0.0, 0.0,
       0.0, 1.0, 0.0,
       0.0, 0.0, 1.0,
    ]
    ci.p = [
        fx, 0.0, cx, 0.0,
       0.0, fy,   cy, 0.0,
       0.0, 0.0, 1.0, 0.0,
    ]
    return ci


def loop_mod(common_path, folders, topic_base, read_fn, encoding, ext, writer, ts, nanosec):
    """
    Publish each sub-folder (camera) under its own topic.
      - folders: list of folder paths
      - topic_base: e.g. '/rgb/image_raw'
      - read_fn: function(path) -> numpy array
      - encoding: image encoding string
      - ext: file extension to replace '.npy' in common step ('.jpg', '.png', '.npy')
      - writer: rosbag2 writer
      - ts, nanosec: timestamp
    """
    for folder in folders:
        raw_cam = os.path.basename(folder)              # "robot.front_camera.left.rgb_image"
        cam     = raw_cam.replace('.', '_')             # sanitized for topic
        parts   = raw_cam.split('.')                    # ["robot","front_camera","left","rgb_image"]
        camera  = parts[2]                              # "left" or "right"
        frame_id = f"{camera}_frame"                    # "left_frame" or "right_frame"

        fname = os.path.basename(common_path).replace('.npy', ext)
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            continue

        data = read_fn(path)
        img_msg = make_image_msg(data, encoding, ts)
        img_msg.header.stamp    = ts
        img_msg.header.frame_id = frame_id                # use the camera frame
        topic = f"{topic_base}/{cam}"
        writer.write(topic, serialize_message(img_msg), nanosec)


def main():
    parser = argparse.ArgumentParser(description='Convert MobilityGen replay to MCAP bag')
    parser.add_argument('--input', required=True, type=str, help='path to one recording under replays')
    parser.add_argument('--output', required=True, type=str, help='output bag filename (must end in .mcap, e.g. ~/bags/my_run.mcap)')
    parser.add_argument('--hz', type=float, default=1.0, help='data rate in Hz (default: 1.0)')
    args = parser.parse_args()

    # Expand '~' if present
    base = os.path.expanduser(args.input)
    outbag = os.path.expanduser(args.output)

    # Initialize ROS 2 (for message definitions)
    rclpy.init()

    # Set up rosbag2 writer for MCAP
    writer = SequentialWriter()
    # storage_opts = StorageOptions(uri=outbag, storage_id='sqlite3')
    storage_opts = StorageOptions(uri=outbag, storage_id='mcap')
    conv_opts = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    writer.open(storage_opts, conv_opts)

    # Declare all topics
    topics = [
        TopicMetadata(name='/pelvis_pose', type='geometry_msgs/msg/PoseStamped', serialization_format='cdr'),
        TopicMetadata(name='/pelvis_odom', type='nav_msgs/msg/Odometry', serialization_format='cdr'),
        TopicMetadata(name='/cmd_vel',     type='geometry_msgs/msg/Twist', serialization_format='cdr'),
        TopicMetadata(name='/tf',        type='tf2_msgs/msg/TFMessage',   serialization_format='cdr'),
        TopicMetadata(name='/tf_static', type='tf2_msgs/msg/TFMessage',   serialization_format='cdr'),
        TopicMetadata(name='/rgb/camera_info/left',  type='sensor_msgs/msg/CameraInfo', serialization_format='cdr'),
        TopicMetadata(name='/rgb/camera_info/right', type='sensor_msgs/msg/CameraInfo', serialization_format='cdr'),
        TopicMetadata(name='/depth/camera_info/left',  type='sensor_msgs/msg/CameraInfo', serialization_format='cdr'),
        TopicMetadata(name='/depth/camera_info/right', type='sensor_msgs/msg/CameraInfo', serialization_format='cdr'),
        TopicMetadata(name='/left_camera_pose', type='geometry_msgs/msg/PoseStamped', serialization_format='cdr'),
        TopicMetadata(name='/right_camera_pose', type='geometry_msgs/msg/PoseStamped', serialization_format='cdr'),
        TopicMetadata(name='/joint_states',     type='sensor_msgs/msg/JointState',   serialization_format='cdr'),
    ]
    # Add image topics dynamically below once camera names are known
    for meta in topics:
        writer.create_topic(meta)

    # Gather file lists
    common_list     = sorted(glob.glob(os.path.join(base, 'state', 'common', '*.npy')))
    rgb_folders     = sorted(glob.glob(os.path.join(base, 'state', 'rgb', '*')))
    seg_folders     = sorted(glob.glob(os.path.join(base, 'state', 'segmentation', '*')))
    depth_folders   = sorted(glob.glob(os.path.join(base, 'state', 'depth', '*')))
    normals_folders = sorted(glob.glob(os.path.join(base, 'state', 'normals', '*')))
    inst_folders = [f for f in seg_folders if 'instance_id_segmentation' in os.path.basename(f)]
    sem_folders  = [f for f in seg_folders if f.endswith('.segmentation_image')]

    # Pick a sample image to read width/height
    sample_rgb = Image.open(os.path.join(rgb_folders[0], os.path.basename(common_list[0]).replace('.npy','.jpg')))  
    W, H = sample_rgb.size


    # Create camera info’s
    rgb_ci_left   = make_camera_info(W, H, 'left_frame')
    rgb_ci_right  = make_camera_info(W, H, 'right_frame')
    depth_ci_left = make_camera_info(W, H, 'left_frame')
    depth_ci_right= make_camera_info(W, H, 'right_frame')
        
    for folder in inst_folders:
        cam = os.path.basename(folder).replace('.', '_')
        writer.create_topic(TopicMetadata(
            name=f"/segmentation/instance_id/{cam}",
            type="sensor_msgs/msg/Image",
            serialization_format="cdr"
        ))
    for folder in sem_folders:
        cam = os.path.basename(folder).replace('.', '_')
        writer.create_topic(TopicMetadata(
            name=f"/segmentation/semantic/{cam}",
            type="sensor_msgs/msg/Image",
            serialization_format="cdr"
        ))


    # Create image topics for each camera folder
    for folder_list, topic_base in [
        (rgb_folders, '/rgb/image_raw'),
        (depth_folders, '/depth/image_raw'),
        (normals_folders, '/normals/image'),
    ]:
        for folder in folder_list:
            raw_cam = os.path.basename(folder)
            cam     = raw_cam.replace('.', '_')
            writer.create_topic(TopicMetadata(
                name=f"{topic_base}/{cam}",
                type='sensor_msgs/msg/Image',
                serialization_format='cdr'
            ))
    
    # Load the very first frame’s common state to get camera poses
    first_state = np.load(common_list[0], allow_pickle=True).item()

    static_transforms = []

    def make_camera_static(name):
        # 1) grab raw USD poses
        raw_pel_p = first_state['robot.position']
        raw_pel_o = first_state['robot.orientation']
        raw_cam_p = first_state[f'robot.front_camera.{name}.position']
        raw_cam_o = first_state[f'robot.front_camera.{name}.orientation']

        # 2) convert into ROS conventions
        pel_p, pel_q = usd_to_ros_pose(raw_pel_p, raw_pel_o)
        cam_p, cam_q = usd_to_ros_camera_frame(raw_cam_p, raw_cam_o)

        # 3) build scipy Rotation objects
        R_pel = R.from_quat(pel_q)   # pel_q is (qx,qy,qz,qw)
        R_cam = R.from_quat(cam_q)

        # 4) compute relative translation in pelvis frame
        t_rel = R_pel.inv().apply(np.array(cam_p) - np.array(pel_p))

        # 5) compute relative rotation
        R_rel = R_pel.inv() * R_cam
        q_rel = R_rel.as_quat()  # [x, y, z, w]

        # 6) populate TransformStamped
        tf = TransformStamped()
        tf.header.stamp    = rclpy.time.Time(seconds=2, nanoseconds=0).to_msg()
        tf.header.frame_id = 'pelvis'
        tf.child_frame_id  = f'{name}_frame'

        tf.transform.translation.x = float(t_rel[0])
        tf.transform.translation.y = float(t_rel[1])
        tf.transform.translation.z = float(t_rel[2])

        tf.transform.rotation.x = float(q_rel[0])
        tf.transform.rotation.y = float(q_rel[1])
        tf.transform.rotation.z = float(q_rel[2])
        tf.transform.rotation.w = float(q_rel[3])

        return tf


    # def make_camera_static(name):
    #     # grab global (map) pose of camera and pelvis
    #     cam_p = np.array(first_state[f'robot.front_camera.{name}.position'])
    #     cam_q = np.array(first_state[f'robot.front_camera.{name}.orientation'])  # [qw, qx, qy, qz]

    #     pel_p = np.array(first_state['robot.position'])
    #     pel_q = np.array(first_state['robot.orientation'])  # [qw, qx, qy, qz]

    #     # build Rotation objects (note: scipy expects [x,y,z,w])
    #     R_pel = R.from_quat([pel_q[1], pel_q[2], pel_q[3], pel_q[0]])
    #     R_cam = R.from_quat([cam_q[1], cam_q[2], cam_q[3], cam_q[0]])

    #     # compute relative translation in pelvis frame
    #     t_map = cam_p - pel_p
    #     t_rel = R_pel.inv().apply(t_map)

    #     # compute relative rotation
    #     R_rel = R_pel.inv() * R_cam
    #     q_rel = R_rel.as_quat()  # returns [x,y,z,w]

    #     # fill the TransformStamped
    #     tf = TransformStamped()
    #     tf.header.stamp    = rclpy.time.Time(seconds=2, nanoseconds=0).to_msg()
    #     tf.header.frame_id = 'pelvis'
    #     tf.child_frame_id  = f'{name}_frame'

    #     tf.transform.translation.x = float(t_rel[0])
    #     tf.transform.translation.y = float(t_rel[1])
    #     tf.transform.translation.z = float(t_rel[2])

    #     tf.transform.rotation.x = float(q_rel[0])
    #     tf.transform.rotation.y = float(q_rel[1])
    #     tf.transform.rotation.z = float(q_rel[2])
    #     tf.transform.rotation.w = float(q_rel[3])

    #     return tf

    # Add left and right camera static transforms
    static_transforms.append(make_camera_static('left'))
    static_transforms.append(make_camera_static('right'))

    # Add map→odom transform
    tf_m_o = TransformStamped()
    tf_m_o.header.frame_id   = 'map'
    tf_m_o.child_frame_id    = 'odom'
    tf_m_o.transform.translation.x = 0.0
    tf_m_o.transform.translation.y = 0.0
    tf_m_o.transform.translation.z = 0.0
    tf_m_o.transform.rotation.w    = 1.0
    static_transforms.append(tf_m_o)

    # Now write tf_static once with all the transforms
    static_msg = TFMessage(transforms=static_transforms)
    writer.write('/tf_static', serialize_message(static_msg), rclpy.time.Time(seconds=2, nanoseconds=0).nanoseconds)

    # Time stepping
    hz = args.hz
    dt_ns = int(1e9 / hz)

    # Initialize odom variables 
    prev_t = None
    prev_pos = None
    prev_q = None

    # Loop through each common npy (one per timestamp)
    for idx, common_path in enumerate(common_list):
        # Build timestamp
        secs = int(idx / hz)
        nsecs = int((idx / hz - secs) * 1e9)
        t = rclpy.time.Time(seconds=secs, nanoseconds=nsecs)
        ts = t.to_msg()
        nanosec = t.nanoseconds

        # /pelvis_pose
        state = np.load(common_path, allow_pickle=True).item()
        # print(f"Keys for {common_path}:", state.keys())  # debug

        #── DEBUG: print first‐sample USD→ROS pose vs camera‐frame pose ──
        if idx == 0:
            # raw USD camera pose (map→cam)
            cam_pos = state['robot.front_camera.left.position']
            cam_ori = state['robot.front_camera.left.orientation']
            (cam_p1, cam_q1) = usd_to_ros_pose(cam_pos, cam_ori)
            cam_e1 = R.from_quat(cam_q1).as_euler('xyz', degrees=True)
            print("=== USD→ROS camera‐passthrough (first sample) ===")
            print(f"  position:   {cam_p1}")
            print(f"  quaternion: {cam_q1}")
            print(f"  euler xyz:  {cam_e1}\n")

            # now your camera‐frame conversion
            (cam_p2, cam_q2) = usd_to_ros_camera_frame(cam_pos, cam_ori)
            cam_e2 = R.from_quat(cam_q2).as_euler('xyz', degrees=True)
            print("=== usd_to_ros_camera_frame (first sample) ===")
            print(f"  position:   {cam_p2}")
            print(f"  quaternion: {cam_q2}")
            print(f"  euler xyz:  {cam_e2}\n")
        #── end DEBUG ──

        pos_key = 'robot.position'
        ori_key = 'robot.orientation'
        if pos_key not in state or ori_key not in state:
            raise RuntimeError(f"Missing {pos_key} or {ori_key} in {common_path}")

        pos = state[pos_key]
        ori = state[ori_key]

        # use the helper defined up top
        (x, y, z), (qx, qy, qz, qw) = usd_to_ros_pose(pos, ori)

        pose = PoseStamped()
        pose.header.stamp = ts
        pose.header.frame_id = 'map'
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)
        writer.write('/pelvis_pose', serialize_message(pose), nanosec)


        # /left_camera_pose 
        left_pos = state['robot.front_camera.left.position']
        left_ori = state['robot.front_camera.left.orientation']
        (lx, ly, lz), (lqx, lqy, lqz, lqw) = usd_to_ros_camera_frame(left_pos, left_ori)
        
        left_pose = PoseStamped()
        left_pose.header.stamp    = ts
        left_pose.header.frame_id = 'map'
        
        left_pose.pose.position.x    = float(lx)
        left_pose.pose.position.y    = float(ly)
        left_pose.pose.position.z    = float(lz)
        left_pose.pose.orientation.x = float(lqx)
        left_pose.pose.orientation.y = float(lqy)
        left_pose.pose.orientation.z = float(lqz)
        left_pose.pose.orientation.w = float(lqw)

        writer.write('/left_camera_pose', serialize_message(left_pose), nanosec)

        # /right_camera_pose 
        right_pos = state['robot.front_camera.right.position']
        right_ori = state['robot.front_camera.right.orientation']
        (rx, ry, rz), (rqx, rqy, rqz, rqw) = usd_to_ros_camera_frame(right_pos, right_ori)
        
        right_pose = PoseStamped()
        right_pose.header.stamp    = ts
        right_pose.header.frame_id = 'map'
        
        right_pose.pose.position.x    = float(rx)
        right_pose.pose.position.y    = float(ry)
        right_pose.pose.position.z    = float(rz)
        right_pose.pose.orientation.x = float(rqx)
        right_pose.pose.orientation.y = float(rqy)
        right_pose.pose.orientation.z = float(rqz)
        right_pose.pose.orientation.w = float(rqw)

        writer.write('/right_camera_pose', serialize_message(right_pose), nanosec)

        # /pelvis_odom
        odom = Odometry()
        odom.header.stamp = ts
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'pelvis'
        odom.pose.pose = pose.pose
        # compute twist dt
        if prev_t is not None:
            dt = (t - prev_t).nanoseconds * 1e-9
            # finite diff linear velocity
            dx = (x - prev_pos[0]) / dt
            dy = (y - prev_pos[1]) / dt
            dz = (z - prev_pos[2]) / dt
            # angular velocity from quaternion derivative
            # approximate by diff of euler yaw,pitch,roll
            prev_e = R.from_quat(prev_q).as_euler('xyz',degrees=False)
            curr_e = R.from_quat([qx,qy,qz,qw]).as_euler('xyz',degrees=False)
            domega = (curr_e - prev_e) / dt
            twist = Twist()
            twist.linear.x, twist.linear.y, twist.linear.z = dx, dy, dz
            twist.angular.x, twist.angular.y, twist.angular.z = domega
            odom.twist.twist = twist
        prev_t = t; prev_pos = (x,y,z); prev_q = [qx,qy,qz,qw]
        writer.write('/pelvis_odom', serialize_message(odom), nanosec)

        # /cmd_vel
        act = state.get('robot.action', [0,0,0,0])
        # assume [vx and omega_z]
        tv = Twist()
        tv.linear.x  = act[0]
        tv.angular.z = act[1]
        writer.write('/cmd_vel', serialize_message(tv), nanosec)


        # joint_names = [
        #     "left_hip_yaw_joint",
        #     "left_hip_roll_joint",
        #     "left_hip_pitch_joint",
        #     "left_knee_joint",
        #     "left_ankle_joint",
        #     "right_hip_yaw_joint",
        #     "right_hip_roll_joint",
        #     "right_hip_pitch_joint",
        #     "right_knee_joint",
        #     "right_ankle_joint",
        #     "torso_joint",
        #     "left_shoulder_pitch_joint",
        #     "left_shoulder_roll_joint",
        #     "left_shoulder_yaw_joint",
        #     "left_elbow_joint",
        #     "right_shoulder_pitch_joint",
        #     "right_shoulder_roll_joint",
        #     "right_shoulder_yaw_joint",
        #     "right_elbow_joint",
        # ]
        
        joint_names = [
            "torso_joint",
            "right_shoulder_pitch_joint",
            "left_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "left_shoulder_yaw_joint",
            "right_ankle_joint",
            "left_ankle_joint",
            "right_hip_roll_joint",
            "left_hip_roll_joint",
            "right_knee_joint",
            "left_knee_joint",
            "right_hip_yaw_joint",
            "left_hip_yaw_joint",
            "right_hip_pitch_joint",
            "left_hip_pitch_joint",
            "right_elbow_joint",
            "left_elbow_joint",
        ]
        
        js = JointState()
        js.header.stamp = ts
        js.name     = joint_names
        js.position = [float(x) for x in state['robot.joint_positions']]
        js.velocity = [float(x) for x in state['robot.joint_velocities']]
        js.effort   = [0.0] * len(joint_names)
        writer.write('/joint_states', serialize_message(js), nanosec)
        

        # /tf (dynamic)
        tf = TransformStamped()
        tf.header.stamp    = ts
        tf.header.frame_id = 'map'
        tf.child_frame_id  = 'pelvis'

        # instead of writing raw pos/ori:
        tx, ty, tz = usd_to_ros_pose(pos, ori)[0]
        qx, qy, qz, qw = usd_to_ros_pose(pos, ori)[1]
        tf.transform.translation.x = float(tx)
        tf.transform.translation.y = float(ty)
        tf.transform.translation.z = float(tz)
        tf.transform.rotation.x    = float(qx)
        tf.transform.rotation.y    = float(qy)
        tf.transform.rotation.z    = float(qz)
        tf.transform.rotation.w    = float(qw)


        tf_msg = TFMessage(transforms=[tf])
        writer.write('/tf', serialize_message(tf_msg), nanosec)

        # images
        # RGB
        loop_mod(common_path, rgb_folders, '/rgb/image_raw',
                lambda p: np.asarray(Image.open(p)), 'rgb8', '.jpg',
                writer, ts, nanosec)
        
        # RGB Info
        rgb_ci_left.header.stamp  = ts
        writer.write('/rgb/camera_info/left', serialize_message(rgb_ci_left), nanosec)
        rgb_ci_right.header.stamp = ts
        writer.write('/rgb/camera_info/right', serialize_message(rgb_ci_right), nanosec)
        
        # Instance Segmentation ID 
        loop_mod(common_path, inst_folders, '/segmentation/instance_id',
                lambda p: np.asarray(Image.open(p)), 'mono8', '.png',
                writer, ts, nanosec)

        # Semantic Segmentation(scaled into 0–255) 
        def read_and_scale(p):
            lbl = np.asarray(Image.open(p), dtype=np.int32)
            m   = int(lbl.max()) or 1
            scaled = ((lbl.astype(np.float32) / m) * 255.0).astype(np.uint8)
            return scaled

        loop_mod(common_path, sem_folders, '/segmentation/semantic',
                read_and_scale, 'mono8', '.png',
                writer, ts, nanosec)

        # Depth (I;16 → mono16)
        loop_mod(common_path, depth_folders, '/depth/image_raw',
                 lambda p: np.asarray(Image.open(p).convert('I;16')), 'mono16', '.png',
                 writer, ts, nanosec)
        
        # Depth Info
        depth_ci_left.header.stamp = ts
        writer.write('/depth/camera_info/left', serialize_message(depth_ci_left), nanosec)
        depth_ci_right.header.stamp = ts
        writer.write('/depth/camera_info/right', serialize_message(depth_ci_right), nanosec)


        # # Normals (npy → float32 RGB triplet)
        # def read_norm_rgb(p):
        #     arr = np.load(p).astype(np.float32)
        #     # if it has 4 channels, drop the last one
        #     if arr.ndim == 3 and arr.shape[2] == 4:
        #         arr = arr[:, :, :3]
        #     return arr


        # loop_mod(common_path, normals_folders, '/normals/image',
        #         read_norm_rgb, 'rgb32f', '.npy',
        #         writer, ts, nanosec)

        # Normals (npy → uint8 RGB)
        def read_norm(p):
            arr = np.load(p).astype(np.float32)  # (H,W,4)
            arr = arr[..., :3]                   # drop padding
            # normalize from [-1,1] to [0,1]
            view = (arr * 0.5 + 0.5).clip(0.0, 1.0)
            # to uint8 in [0,255]
            view8 = (view * 255.0).astype('uint8')
            return view8

        loop_mod(common_path, normals_folders, '/normals/image',
                read_norm, 'rgb8', '.npy',
                writer, ts, nanosec)



    print(f"Finished writing MCAP bag: {outbag}")
    rclpy.shutdown()


if __name__ == '__main__':
    main()

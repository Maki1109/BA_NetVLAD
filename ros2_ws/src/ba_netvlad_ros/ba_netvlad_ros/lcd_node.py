"""Live BA-NetVLAD loop-closure detection node for TurtleBot4.

Pipeline per incoming camera frame:
  image -> BlurGate (diagnostic only; a live camera stream has no offline
           sharp/blur *branch* split to route ORB vs NetVLAD -- this repo has
           no ORB/tracking implementation, see README) -> BANetVLAD -> 128-D
           descriptor -> cosine top-k against the DynamicCosineDict, filtered
           to entries older than `min_loop_time_s` -> tau/margin confirmation
           -> TemporalVoter m-of-r -> on acceptance, publish the matched
           keyframe's pose as a PoseWithCovarianceStamped ("we believe the
           robot is here") plus a DiagnosticArray with the raw match evidence.

NOT tested against a real TurtleBot4 or a live ROS 2 graph -- this development
environment has no ROS 2 / rclpy installed. Written to be correct against the
Humble API and buildable with colcon on the robot/workstation; verify topic
names, QoS, and every threshold against your actual camera and odometry before
trusting it near the robot.
"""
import os
import sys
import threading

import numpy as np
import cv2
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import ParameterDescriptor, FloatingPointRange, IntegerRange

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

# ba_netvlad is a plain Python package (repo root), not a ROS package -- make
# it importable without duplicating any pipeline logic here.
_REPO_ROOT = os.environ.get('BA_NETVLAD_REPO_ROOT', '')
if _REPO_ROOT and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from ba_netvlad.blur import BlurGate                     # noqa: E402
from ba_netvlad.model import load_ckpt                   # noqa: E402
from ba_netvlad.dataset import to_tensor_224              # noqa: E402
from ba_netvlad.dictionary import load_dictionary                    # noqa: E402
from ba_netvlad.matcher import TemporalVoter               # noqa: E402

from ba_netvlad_ros.dynamic_dictionary import DynamicCosineDict


def _float_range(lo, hi):
    return ParameterDescriptor(floating_point_range=[FloatingPointRange(
        from_value=float(lo), to_value=float(hi), step=0.0)])


def _int_range(lo, hi):
    return ParameterDescriptor(integer_range=[IntegerRange(
        from_value=int(lo), to_value=int(hi), step=1)])


class BANetVLADLCDNode(Node):
    def __init__(self):
        super().__init__('ba_netvlad_lcd_node')

        # -- parameters (units and valid ranges documented per-field) --------
        self.declare_parameter('image_topic', '/oakd/rgb/preview/image_raw')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('model_path', '')                       # required, no sane default
        self.declare_parameter('dict_path', '')                        # '' -> start with an empty dictionary
        self.declare_parameter('dict_autosave_path', 'live_dict.npz')
        self.declare_parameter('dict_autosave_every_n_keyframes', 20,
                               _int_range(1, 10000))
        self.declare_parameter('device', 'cpu')                        # 'cpu' or 'cuda'
        self.declare_parameter('stride', 16, _int_range(16, 32))

        # Step 1 (blur gate) -- MUST be recalibrated per camera via
        # scripts/01_calibrate_blur_gate.py; the KITTI value (1521.5) is
        # meaningless for a different lens/sensor.
        self.declare_parameter('epsilon', 0.0)                         # variance-of-Laplacian units
        self.declare_parameter('epsilon_hysteresis_ratio', 1.15, _float_range(1.0, 3.0))

        # Step 4 (matcher / temporal voting) -- start from the values you
        # tuned offline on a recorded run of THIS robot/environment, then
        # re-tune from a PR curve; the KITTI operating point does not transfer.
        self.declare_parameter('tau', 0.5, _float_range(0.0, 1.0))
        self.declare_parameter('delta', 0.05, _float_range(0.0, 1.0))
        self.declare_parameter('topk', 5, _int_range(1, 50))
        self.declare_parameter('voter_r', 5, _int_range(1, 50))
        self.declare_parameter('voter_m', 3, _int_range(1, 50))
        self.declare_parameter('margin_window_m', 3.0, _float_range(0.0, 100.0))

        # Live exclusion test -- excludes RECENT history along the trajectory,
        # not spatially nearby places (a true loop closure is spatially near
        # the query by definition). See DynamicCosineDict.eligible_mask.
        self.declare_parameter('min_loop_time_s', 30.0, _float_range(0.0, 3600.0))

        # Mapping-phase keyframe spacing -- only add a new dictionary entry
        # once the robot has moved at least this far from the last one.
        self.declare_parameter('mapping_mode', True)
        self.declare_parameter('keyframe_min_distance_m', 1.0, _float_range(0.0, 50.0))

        p = self.get_parameter
        self.image_topic = p('image_topic').value
        self.odom_topic = p('odom_topic').value
        model_path = p('model_path').value
        dict_path = p('dict_path').value
        self.dict_autosave_path = p('dict_autosave_path').value
        self.dict_autosave_every_n = p('dict_autosave_every_n_keyframes').value
        self.device = p('device').value
        stride = p('stride').value

        epsilon = p('epsilon').value
        if epsilon <= 0.0:
            self.get_logger().warn(
                'epsilon <= 0: the blur gate will report every frame as blurry. '
                'Run scripts/01_calibrate_blur_gate.py on this camera and set '
                'the "epsilon" parameter before trusting the blur diagnostic.')
        self.gate = BlurGate(epsilon=epsilon,
                             hysteresis_ratio=p('epsilon_hysteresis_ratio').value)

        self.tau = p('tau').value
        self.delta = p('delta').value
        self.topk_k = p('topk').value
        self.margin_window_m = p('margin_window_m').value
        self.min_loop_time_s = p('min_loop_time_s').value
        self.mapping_mode = p('mapping_mode').value
        self.keyframe_min_distance_m = p('keyframe_min_distance_m').value
        self.voter = TemporalVoter(r=p('voter_r').value, m=p('voter_m').value)

        if not model_path:
            raise RuntimeError('The "model_path" parameter is required (path to a .pt checkpoint).')
        self.get_logger().info(f'Loading BA-NetVLAD checkpoint: {model_path}')
        self.model = load_ckpt(model_path, stride=stride, pretrained=False).to(self.device)
        self.model.eval()

        self.pca_project = None
        pca_dim = 128
        if dict_path and os.path.exists(dict_path):
            d = load_dictionary(dict_path)
            self.dyn_dict = DynamicCosineDict.load_npz(dict_path, dim=d['descs'].shape[1])
            if d.get('pca') is not None:
                self.pca_project = d['pca']
            self.get_logger().info(f'Warm-started dictionary from {dict_path} '
                                   f'({len(self.dyn_dict)} keyframes).')
        else:
            self.dyn_dict = DynamicCosineDict(dim=pca_dim)
            self.get_logger().info('Starting with an EMPTY dictionary (mapping mode should be true).')

        self._bridge = CvBridge()
        self._infer_lock = threading.Lock()
        self._busy = False
        self._latest_odom = None            # (x, y, t_sec)
        self._last_keyframe_pose = None
        self._keyframes_since_autosave = 0
        self._n_processed = 0

        # High-rate sensor data: BEST_EFFORT, shallow queue -- inference is the
        # bottleneck, so we always want the newest frame, never a backlog.
        image_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                               history=HistoryPolicy.KEEP_LAST, depth=1)
        # Odometry is comparatively cheap and typically published RELIABLE;
        # small backlog is fine and we only ever read the latest sample.
        odom_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST, depth=10)

        self.create_subscription(Image, self.image_topic, self._on_image, image_qos)
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, odom_qos)

        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped,
                                              '~/loop_closure_pose', 10)
        self.diag_pub = self.create_publisher(DiagnosticArray, '~/loop_closure_debug', 10)

        self.create_service(Trigger, '~/save_dictionary', self._on_save_dictionary)
        self.create_service(Trigger, '~/set_mapping_mode_on', self._on_mapping_on)
        self.create_service(Trigger, '~/set_mapping_mode_off', self._on_mapping_off)

        self.get_logger().info(
            f'ba_netvlad_lcd_node ready: image={self.image_topic} odom={self.odom_topic} '
            f'mapping_mode={self.mapping_mode} dict_size={len(self.dyn_dict)}')

    # -- odometry -------------------------------------------------------------
    def _on_odom(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._latest_odom = (msg.pose.pose.position.x, msg.pose.pose.position.y, t)

    # -- services ---------------------------------------------------------------
    def _on_save_dictionary(self, request, response):
        try:
            self.dyn_dict.save_npz(self.dict_autosave_path)
            response.success = True
            response.message = f'saved {len(self.dyn_dict)} keyframes to {self.dict_autosave_path}'
        except Exception as e:  # noqa: BLE001 -- report any failure to the caller, don't crash the node
            response.success = False
            response.message = str(e)
        return response

    def _on_mapping_on(self, request, response):
        self.mapping_mode = True
        response.success = True
        response.message = 'mapping_mode = true'
        return response

    def _on_mapping_off(self, request, response):
        self.mapping_mode = False
        response.success = True
        response.message = 'mapping_mode = false'
        return response

    # -- main callback ----------------------------------------------------------
    def _on_image(self, msg: Image):
        if self._busy:
            return  # drop this frame rather than queue -- inference is the bottleneck
        odom = self._latest_odom
        if odom is None:
            return  # no pose yet; can't place a keyframe or test the time gap meaningfully

        self._busy = True
        try:
            self._process(msg, odom)
        except Exception:                    # noqa: BLE001
            self.get_logger().error('exception in image callback', exc_info=True)
        finally:
            self._busy = False

    def _process(self, msg: Image, odom):
        x, y, _odom_t = odom
        # Timestamp the EVENT at hardware acquisition time (the image's own
        # header stamp), not at the time this callback happens to run.
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        beta, is_sharp = self.gate(img)

        with torch.no_grad():
            x_in = to_tensor_224(img).unsqueeze(0).to(self.device)
            if self.pca_project is not None:
                vlad = self.model(x_in, return_vlad=True)[1].cpu().numpy()[0]
                emb = self.pca_project(vlad)
            else:
                emb = self.model(x_in).cpu().numpy()[0]
        self._n_processed += 1

        # -- mapping: grow the dictionary -----------------------------------
        if self.mapping_mode:
            last = self.dyn_dict.last_pose()
            if last is None or np.linalg.norm(np.array([x, y]) - last) >= self.keyframe_min_distance_m:
                self.dyn_dict.add(emb, x, y, t, image_path='')
                self._keyframes_since_autosave += 1
                if self._keyframes_since_autosave >= self.dict_autosave_every_n:
                    self.dyn_dict.save_npz(self.dict_autosave_path)
                    self._keyframes_since_autosave = 0

        # -- loop-closure query -----------------------------------------------
        exclude = ~self.dyn_dict.eligible_mask(t, self.min_loop_time_s)
        idx, sc = self.dyn_dict.topk(emb, k=self.topk_k, exclude=exclude)
        if len(sc) == 0 or not np.isfinite(sc[0]):
            self.voter.vote(TemporalVoter.NO_VOTE)
            return

        s1, top1 = float(sc[0]), int(idx[0])
        s_full = self.dyn_dict.scores(emb, exclude=exclude)
        s2 = self.dyn_dict.second_best_other_place(s_full, top1, self.margin_window_m)
        candidate = (s1 >= self.tau) and ((s1 - s2) >= self.delta)
        accepted = self.voter.vote(top1 if candidate else TemporalVoter.NO_VOTE)

        if not accepted:
            return

        matched_pose = self.dyn_dict.pose(top1)
        self._publish_loop_closure(msg.header, x, y, matched_pose, top1, s1, s2, beta)

    def _publish_loop_closure(self, header, x, y, matched_pose, matched_idx, s1, s2, beta):
        out = PoseWithCovarianceStamped()
        out.header = header
        out.header.frame_id = header.frame_id or 'odom'
        out.pose.pose.position.x = float(matched_pose[0])
        out.pose.pose.position.y = float(matched_pose[1])
        # Covariance is a placeholder, not a calibrated uncertainty: tighter
        # for a larger tau/delta margin. Replace with a real estimate (e.g.
        # from a geometric re-verification step) before feeding this into a
        # pose-graph optimizer -- see README's "what this does NOT do" section.
        margin = max(s1 - s2, 1e-3)
        var_xy = float(np.clip(0.5 / margin, 0.01, 5.0))
        cov = [0.0] * 36
        cov[0] = var_xy
        cov[7] = var_xy
        cov[35] = 0.5
        out.pose.covariance = cov
        self.pose_pub.publish(out)

        diag = DiagnosticArray()
        diag.header = header
        status = DiagnosticStatus(name='ba_netvlad_lcd', level=DiagnosticStatus.OK,
                                  message='loop closure accepted')
        status.values = [
            KeyValue(key='query_x', value=f'{x:.3f}'),
            KeyValue(key='query_y', value=f'{y:.3f}'),
            KeyValue(key='matched_index', value=str(matched_idx)),
            KeyValue(key='matched_x', value=f'{matched_pose[0]:.3f}'),
            KeyValue(key='matched_y', value=f'{matched_pose[1]:.3f}'),
            KeyValue(key='s1_cosine', value=f'{s1:.4f}'),
            KeyValue(key='s2_cosine', value=f'{s2:.4f}'),
            KeyValue(key='margin', value=f'{s1 - s2:.4f}'),
            KeyValue(key='blur_beta', value=f'{beta:.1f}'),
        ]
        diag.status = [status]
        self.diag_pub.publish(diag)
        self.get_logger().info(
            f'LOOP CLOSURE: query=({x:.2f},{y:.2f}) -> matched idx={matched_idx} '
            f'@({matched_pose[0]:.2f},{matched_pose[1]:.2f})  s1={s1:.3f} margin={s1-s2:.3f}')


def main(args=None):
    rclpy.init(args=args)
    node = BANetVLADLCDNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

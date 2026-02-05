import numpy as np
import genesis as gs
from pynput import keyboard
import matplotlib.pyplot as plt
import torch
import math


########################## 初期化 ##########################
gs.init(backend=gs.gpu)

########################## シーンの作成 ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        res=(960, 640),
        max_FPS=1000,
    ),
    # renderer = gs.renderers.Rasterizer(),
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=True,
    vis_options=gs.options.VisOptions(shadow=False,plane_reflection=False)
)

cam1 = scene.add_camera(
    res    = (157, 87),
    pos    = (0,0,15),
    lookat = (1, 0, 0),
    fov    = 58,
    GUI    = False
)
cam2 = scene.add_camera(
    res    = (157, 87),
    pos    = (0,0,15),
    lookat = (1, 0, 0),
    fov    = 58,
    GUI    = False
)
cam3 = scene.add_camera(
    res    = (157, 87),
    pos    = (0,0,15),
    lookat = (1, 0, 0),
    fov    = 58,
    GUI    = False
)
cam4 = scene.add_camera(
    res    = (157, 87),
    pos    = (0,0,15),
    lookat = (1, 0, 0),
    fov    = 58,
    GUI    = False
)


########################## エンティティ ##########################
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.URDF(file='/home/chujoken/Genesis/kachaka-api/ros2/kachaka_02.urdf'),
    # vis_mode='collision',
    # visualize_contact=True,
)

for l in plane.links:
    for g in l.geoms:
        g.set_friction(0.01)

for l in franka.links:
    if l.name in ['caster_sphere_left', 'caster_sphere_right']:
        for g in l.geoms:
            g.set_friction(0.01)



wall_thickness = 0.05
wall_height = 1.0
room_size = 0.5  # ルームサイズを1.0に変更
floor_thickness = 0.01

# 壁の追加
scene.add_entity(
    gs.morphs.Box(
        size=(wall_thickness, room_size * 2, wall_height),
        pos=(room_size + wall_thickness / 2, 0, wall_height / 2),
        fixed = True
    )
)

scene.add_entity(
    gs.morphs.Box(
        size=(wall_thickness, room_size * 2, wall_height),
        pos=(-room_size - wall_thickness / 2, 0, wall_height / 2),
        fixed = True
    )
)

scene.add_entity(
    gs.morphs.Box(
        size=(room_size * 2, wall_thickness, wall_height),
        pos=(0, room_size + wall_thickness / 2, wall_height / 2),
        fixed = True
    )
)

scene.add_entity(
    gs.morphs.Box(
        size=(room_size * 2, wall_thickness, wall_height),
        pos=(0, -room_size - wall_thickness / 2, wall_height / 2),
        fixed = True
    )
)

########################## ビルド ##########################
scene.build()

# cam.follow_entity(franka,fixed_axis=(None, None, 0.2), smoothing=0.5, fix_orientation=True)

jnt_names = [
    'base_r_drive_wheel_joint',
    'base_l_drive_wheel_joint',
]
dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]

############ オプション：制御ゲインの設定 ############
franka.set_dofs_kp(kp=np.array([4500, 4500]), dofs_idx_local=dofs_idx)
franka.set_dofs_kv(kv=np.array([450, 450]), dofs_idx_local=dofs_idx)
franka.set_dofs_force_range(
    lower=np.array([-87, -87]),
    upper=np.array([87, 87]),
    dofs_idx_local=dofs_idx,
)

########################## キー入力設定 ##########################
key_state = {'left': False, 'right': False, 'up': False, 'down': False}

def on_press(key):
    try:
        if key == keyboard.Key.left:
            key_state['left'] = True
        elif key == keyboard.Key.right:
            key_state['right'] = True
        elif key == keyboard.Key.up:
            key_state['up'] = True
        elif key == keyboard.Key.down:
            key_state['down'] = True
    except AttributeError:
        pass

def on_release(key):
    try:
        if key == keyboard.Key.left:
            key_state['left'] = False
        elif key == keyboard.Key.right:
            key_state['right'] = False
        elif key == keyboard.Key.up:
            key_state['up'] = False
        elif key == keyboard.Key.down:
            key_state['down'] = False
    except AttributeError:
        pass


def quat_to_rotmat(quat):
    """
    quat: Tensor of shape (4,) in [x, y, z, w] order
    return: 3x3 rotation matrix
    """
    w, x, y, z = quat
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    rot = torch.tensor([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),       2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),   1 - 2*(xx + yy)]
    ])
    return rot

def depth_reset(lidar):
    """
    lidarの配列の数値が0.1以下の場合そのインデックスだけ0にリセット,また、10以上の場合も0にリセット
    """
    for i in range(len(lidar)):
        if lidar[i] < 0.1 or lidar[i] > 10:
            lidar[i] = 0.0

    return lidar

def shift_to_center(data):
    """配列の中心を先頭にシフト"""
    center = len(data) // 2
    arr = np.array(data[center:])
    arr2 = np.array(data[:center])
    return arr, arr2

def back_camera_zero(data):
    """後ろのカメラを0にリセット"""
    for i in range(len(data)):
        data[i] = 0.0
    return data

class LaserScanBinsFast:
    def __init__(self, angle_min=-math.pi, angle_max=math.pi, ranges_size=628,
                 range_min=0.10, range_max=10.0, use_inf=True, inf_epsilon=1.0):
        self.angle_min = float(angle_min)
        self.angle_max = float(angle_max)
        self.ranges_size = int(ranges_size)
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self.use_inf = bool(use_inf)
        self.inf_epsilon = float(inf_epsilon)
        self.angle_increment = (self.angle_max - self.angle_min) / self.ranges_size

        if self.use_inf:
            self.ranges = np.full(self.ranges_size, np.inf, dtype=np.float32)
        else:
            self.ranges = np.full(self.ranges_size, self.range_max + self.inf_epsilon, dtype=np.float32)

    def push_batch(self, xs, ys):
        """(N,)のx,y配列を一括処理してrangesを更新"""
        rs = np.hypot(xs, ys)
        angs = np.arctan2(ys, xs)

        # 有効範囲フィルタ
        mask = (rs >= self.range_min) & (rs <= self.range_max) \
                 & (angs >= self.angle_min) & (angs <= self.angle_max)
        if not np.any(mask):
            return

        rs = rs[mask]
        angs = angs[mask]

        # 各点の角度→ビンindex
        idx = np.floor((angs - self.angle_min) / self.angle_increment).astype(np.int32)

        # 範囲外のインデックス除去
        mask_valid = (idx >= 0) & (idx < self.ranges_size)
        if not np.any(mask_valid):
            return
        idx = idx[mask_valid]
        rs = rs[mask_valid]

        # ビンごとに最小距離で更新（np.minimum.at が高速）
        np.minimum.at(self.ranges, idx, rs)

    def result(self):
        return self.ranges

def zero_head(arr, n):
    """配列の先頭 n 本を 0 に"""
    if n > 0: arr[:n] = 0.0
    return arr

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

########################## メインループ ##########################
velocity_cmd = np.zeros(len(dofs_idx))

trk = 5.0  # トラックの速度

while True:
    velocity_cmd[:] = 0
    if key_state['up']:
        velocity_cmd[:] = trk  # 前進（両輪前進）
    if key_state['down']:
        velocity_cmd[:] = -trk  # 後退
    if key_state['right']:
        velocity_cmd = np.array([-trk, trk])  # 左回転
    if key_state['left']:
        velocity_cmd = np.array([trk, -trk])  # 右回転


    franka.control_dofs_velocity(velocity_cmd, dofs_idx)
    scene.step()



    #ここからカメラの制御
    robot_pos = franka.get_pos()
    robot_quat = franka.get_quat()
    robot_quat = robot_quat / torch.norm(robot_quat)

    # クォータニオンから回転行列を作成
    R = quat_to_rotmat(robot_quat)

    
    # 前方方向の単位ベクトル
    forward = R[:, 0]  # Genesis環境では -X が前方になっているため

    # カメラの位置（真上に0.5mオフセット）
    cam_pos = robot_pos + torch.tensor([0.0, 0.0, 0.15], device=robot_pos.device) + 0.15 * forward


    # 各方向に対応するZ軸ベクトル（カメラの向き）
    directions = {
        'front': -R[:, 0],
        'back':  R[:, 0],
        'right': -R[:, 1],
        'left':  R[:, 1],
    }

    # 各方向の回転行列と変換行列を格納
    R_cams = {}
    T_cams = {}

    # カメラ位置ベクトル（共通）
    # cam_pos: カメラの位置（world座標系）
    # robot_pos.device: 使用デバイス（GPU/CPUに自動対応）
    for name, z_axis in directions.items():
        up_world = torch.tensor([0.0, 0.0, 1.0], device=z_axis.device)
        
        # 右手系の座標軸を構築
        x_axis = torch.nn.functional.normalize(torch.cross(up_world, z_axis, dim=0), dim=0)
        y_axis = torch.cross(z_axis, x_axis, dim=0)

        # 回転行列 R_cam（3x3）
        R_cam = torch.stack([x_axis, y_axis, z_axis], dim=1)
        R_cams[name] = R_cam

        # 4x4変換行列 T_cam
        T = torch.eye(4, device=z_axis.device)
        T[:3, :3] = R_cam
        T[:3, 3] = cam_pos  # カメラの位置
        T_cams[name] = T


    # カメラの姿勢設定
    cam1.set_pose(transform=T_cams['front'].cpu().numpy())
    cam2.set_pose(transform=T_cams['back'].cpu().numpy())

    # カメラの姿勢設定
    cam3.set_pose(transform=T_cams['right'].cpu().numpy())
    cam4.set_pose(transform=T_cams['left'].cpu().numpy())

    # 各カメラのスキャンライン取得
    res = cam1.render_pointcloud()
    front_points = res[0] if isinstance(res, tuple) else res
    rows, cols = front_points.shape[:2]

    res = cam2.render_pointcloud()
    back_points = res[0] if isinstance(res, tuple) else res
    rows, cols = back_points.shape[:2]

    res = cam3.render_pointcloud()
    right_points = res[0] if isinstance(res, tuple) else res
    rows, cols = right_points.shape[:2]

    res = cam4.render_pointcloud()
    left_points = res[0] if isinstance(res, tuple) else res
    rows, cols = left_points.shape[:2]

        # 4点群を連結（N,3）。NaN/Inf除去
    P = np.vstack([front_points.reshape(-1, 3),
                   back_points.reshape(-1, 3),
                   right_points.reshape(-1, 3),
                   left_points.reshape(-1, 3)])
    P = P[np.isfinite(P).all(axis=1)]

    # --- world → robot 座標変換 ---
    # ロボット姿勢（torch）を numpy へ
    R_np = R.cpu().numpy()          # 3x3（列がロボット軸、robot→world）
    robot_pos_np = robot_pos.cpu().numpy()  # 3,
    # 各点のロボット原点相対ベクトル（world基準）
    rel = P - robot_pos_np[None, :]
    # world→robot は R^T を右側から乗算
    Pr = rel @ R_np.T               # (N,3) @ (3,3) = (N,3)

    # ---- 高さフィルタ（例：±0.20m）----
    # ロボット座標でのカメラ高さ（cam_pos は robot_pos + [0,0,0.15]）
    target_z = 0.15
    tol = 0.02   # 許容幅は状況に合わせて（例: 2cm）
    mask_z = np.abs(Pr[:, 2] - target_z) < tol
    Pr = Pr[mask_z]
    if Pr.shape[0] == 0:
        # 何もヒットしない場合はスキップ
        continue

    # ---- XY へ投影 → 角度・距離 ----
        # --- world → robot座標に変換済みの点群 Pr (N,3) があるとする ---
    xs = Pr[:, 0]
    ys = Pr[:, 1]

    # LaserScanビン作成
    scan = LaserScanBinsFast(angle_min=-math.pi,
                             angle_max= math.pi,
                             ranges_size=628,  # 全周628本
                             range_min=0.10,
                             range_max=10.0,
                             use_inf=True,
                             inf_epsilon=1.0)

    # 一括処理
    scan.push_batch(xs, ys)

    # 出力
    lidar = scan.result()  # shape = (628,)
    lidar = np.roll(lidar, 157)

    n = len(lidar)
    step_per_deg = n / 360.0

    # 角度→インデックス範囲
    center = int(270 * step_per_deg)
    half_width = int(60 * step_per_deg)

    start = (center - half_width) % n
    end   = (center + half_width) % n

    lidar = lidar.copy()
    if start < end:
        lidar[start:end] = 0.0
    else:
        lidar[start:] = 0.0
        lidar[:end] = 0.0

    # np.savetxt("kachaka_lidar.txt", lidar, fmt="%f")

    # print("Lidar data:", len(lidar))
    # print(list(lidar))
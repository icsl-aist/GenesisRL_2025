import genesis as gs
import math
import torch


import argparse
import os
import threading
from genesis.sensors.raycaster.patterns import DepthCameraPattern, GridPattern, SphericalPattern
from genesis.utils.geom import euler_to_quat

# ===========================
# Kachakaロボット用の強化学習環境
# ===========================
class KachakaEnv:
    def __init__(self, min_goal_dist=0.5, viewer=False, cam=False):

        # ---- Genesisの初期化 ----
        gs.init(backend=gs.gpu)

        # parser = argparse.ArgumentParser(description="Genesis LiDAR/Depth Camera Visualization with Keyboard Teleop")
        # parser.add_argument("-B", "--n_envs", type=int, default=0, help="Number of environments to replicate")
        # parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
        # parser.add_argument("--use-box", action="store_true", help="Use Box as robot instead of Go2")
        # parser.add_argument(
        #     "--pattern", type=str, default="spherical", choices=("spherical", "depth", "grid"), help="Sensor pattern type"
        # )
        # args = parser.parse_args()

        # ---- シーンを構築 ----
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS= 1000,
            ),
            show_viewer=viewer,
        )

        # ---- 地面を追加 ----
        self.plane = self.scene.add_entity(gs.morphs.Plane(),surface=gs.surfaces.Rough(
        diffuse_texture=gs.textures.ColorTexture(color=(0.5, 0.5, 0.5))
    ))
        # ---- ロボット(Kachaka)を追加 ----
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file="/home/chujoken/Genesis/kachaka-api/ros2/kachaka_02.urdf")
        )

        # ---- 摩擦の設定 ----
        for l in self.plane.links:
            for g in l.geoms:
                g.set_friction(0.01)
        for l in self.robot.links:
            if l.name in ['caster_sphere_left', 'caster_sphere_right']:
                for g in l.geoms:
                    g.set_friction(0.01)

        # ---- 目標(赤い球)を追加 ----
        self.target = self.scene.add_entity(
            morph=gs.morphs.Cylinder(
                radius=0.15,
                height=0.01,
                fixed=True,
                collision=False,
                pos=(0, 0, -0.005),
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(1, 0, 0))
            ),
        )

        if cam == True:
            self.cam = cam
            self.camera = self.scene.add_camera(
                res    = (600, 600),
                pos    = (23,23,15),
                lookat = (23, 23, 0),
                fov    = 60,
                GUI    = True
            )
        

        sensor_kwargs = dict(
            # pattern= pattern
            entity_idx=self.robot.idx,
            pos_offset=(0.0, 0.0, 0.15),
            euler_offset=(0.0, 0.0, 0.0),
            return_world_frame=True,
            draw_debug=False,
        )

        # if args.pattern == "depth":
        #     sensor = self.scene.add_sensor(gs.sensors.DepthCamera(pattern=DepthCameraPattern(), **sensor_kwargs))
        #     self.scene.start_recording(
        #         data_func=(lambda: sensor.read_image()[0]) if args.n_envs > 0 else sensor.read_image,
        #         rec_options=gs.recorders.MPLImagePlot(),
        #     )
        # else:
        #     if args.pattern == "grid":
        #         pattern_cfg = GridPattern()
        #     else:
        #         if args.pattern != "spherical":
        #             gs.logger.warning(f"Unrecognized raycaster pattern: {args.pattern}. Using 'spherical' instead.")
        #         pattern_cfg = SphericalPattern(fov=(360.0,0.0),n_points=(628,1))

        self.sensor = self.scene.add_sensor(gs.sensors.Lidar(pattern=SphericalPattern(fov=(360.0,0.0),n_points=(24,1)), **sensor_kwargs))

        add_square_room(self.scene, inner_size=4.5, height=1.0, thickness=0.05, center=(0.0, 0.0), z0=0.0)
        

        # ---- 並列環境数と空間配置 ----
        self.scene.build(n_envs=1)
        self.num_envs = self.scene.n_envs
        self.target_geoms = self._get_geom_ids(self.target)

        # ---- 関節と制御パラメータ ----
        self.jnt_names = ['base_r_drive_wheel_joint', 'base_l_drive_wheel_joint']
        self.dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in self.jnt_names]
        self.robot.set_dofs_kp(kp=torch.tensor([4500.0, 4500.0], device=gs.device), dofs_idx_local=self.dofs_idx)
        self.robot.set_dofs_kv(kv=torch.tensor([ 450.0,  450.0], device=gs.device), dofs_idx_local=self.dofs_idx)
        self.robot.set_dofs_force_range(
            lower=torch.tensor([-87.0, -87.0], device=gs.device), upper=torch.tensor([87.0, 87.0], device=gs.device), dofs_idx_local=self.dofs_idx
        )

        # ---- 状態変数の初期化 ----
        self.min_goal_dist = min_goal_dist
        self.goal_threshold = 0.15  # ゴール到達判定距離
        self.lidar_dim = 628
        self.state_buffer  = torch.zeros((self.num_envs, 6), device=gs.device, dtype=torch.float32)  # 状態ベクトル(20並列)
        self.goal_pos      = torch.zeros((self.num_envs, 2), device=gs.device, dtype=torch.float32)  # ゴール位置
        self.last_pos      = torch.zeros((self.num_envs, 3), device=gs.device, dtype=torch.float32)  # 前回の位置
        self.last_heading  = torch.zeros(self.num_envs, device=gs.device, dtype=torch.float32)       # 前回のヨー角
        self.last_action   = torch.zeros((self.num_envs, 2), device=gs.device, dtype=torch.float32)  # 前回の行動
        self.prev_distances = torch.zeros(self.num_envs, device=gs.device, dtype=torch.float32)      # 差分報酬用
        self.prev_angle_errors = torch.zeros(self.num_envs, device=gs.device, dtype=torch.float32)  # 角度誤差の履歴

    # ===========================
    # 目標エンティティのジオメトリID取得
    # ===========================
    def _get_geom_ids(self, entity):
        ids = []
        for link in entity.links:
            for geom in link.geoms:
                ids.append(geom.idx)
        return set(ids)

    # ===========================
    # 環境のリセット
    # ===========================

    def reset_idx(self, envs_idx: torch.Tensor, hit: torch.Tensor):
        """envs_idxで指定した複数環境をまとめてリセット（GPU/torchオンリー）"""
        if envs_idx is None:
            return
        if isinstance(envs_idx, (list, tuple)):
            envs_idx = torch.as_tensor(envs_idx, device=self.state_buffer.device, dtype=torch.long)
        if envs_idx.numel() == 0:
            return

        device = self.state_buffer.device
        k = envs_idx.numel()
        r = self.min_goal_dist + torch.rand((k,1), device=device) * (2.2 - self.min_goal_dist)

        # ---- ゴール位置：半径 r の円周上に一括サンプル (k,2) ----
        # 収束補助として上限反復（min_goal_dist を確実に満たす）
        for _ in range(32):
            angles = 2.0 * math.pi * torch.rand((k,), device=device)          # (k,)
            xy = r * torch.stack([torch.cos(angles), torch.sin(angles)], 1)   # (k,2)
            if torch.linalg.norm(xy, dim=1).min() >= self.min_goal_dist:
                self.goal_pos[envs_idx] = xy
                break
            else:
                # フォールバック：全て +x 方向
                fallback_xy = torch.cat([r, torch.zeros_like(r)], dim=1)    # (k,2)
                self.goal_pos[envs_idx] = fallback_xy

        # ---- 目標エンティティの位置設定 (k,3) + envs_idx ----
        z = torch.full((k, 1), -0.005, device=device, dtype=self.goal_pos.dtype)
        self.target.set_pos(torch.cat([self.goal_pos[envs_idx], z], 1),
                            zero_velocity=True, envs_idx=envs_idx)

        # ---- ロボットの初期姿勢をゴール方向へ（yaw→quat, (k,4) (qw,qx,qy,qz)）----
        # dx = self.goal_pos[envs_idx, 0]
        # dy = self.goal_pos[envs_idx, 1]
        # yaw = torch.atan2(dy, dx)                     # (k,)
        # half = yaw * 0.5
        # qw = torch.cos(half)
        # qz = torch.sin(half)
        # quat = torch.stack([qw, torch.zeros_like(qw), torch.zeros_like(qw), qz], dim=1)  # (k,4)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(k, 1)

        # ---- ロボットの位置/姿勢と関節速度の初期化（対象envのみ）----
        idx_nohit = envs_idx[~hit[envs_idx]]
        m = idx_nohit.numel()
        

        if m > 0:
            quat_nohit = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(m, 1)

            self.robot.set_pos(torch.zeros((m, 3), device=device, dtype=torch.float32),envs_idx=idx_nohit)
            self.robot.set_quat(quat_nohit, envs_idx=idx_nohit)

        self.robot.set_dofs_velocity(
            torch.zeros((k, self.robot.n_dofs), device=device, dtype=torch.float32),
            envs_idx=envs_idx
        )

        with torch.no_grad():
            pos_robot  = self.robot.get_pos().detach()
            pos_target = self.target.get_pos().detach()
        gd = torch.linalg.norm(pos_robot[:, :2] - pos_target[:, :2], dim=1)

        quat_robot = self.robot.get_quat().detach()
        heading = torch.atan2(
            2 * (quat_robot[:, 0] * quat_robot[:, 3] + quat_robot[:, 1] * quat_robot[:, 2]),
            1 - 2 * (quat_robot[:, 2] ** 2 + quat_robot[:, 3] ** 2)
        )

        pos_target = self.target.get_pos().detach()

        # ---- ゴール方向と現在の向きの差分 ----
        direction_to_goal = torch.atan2(
            self.goal_pos[:, 1] - pos_robot[:, 1],
            self.goal_pos[:, 0] - pos_robot[:, 0]
        )
        direction_error = (direction_to_goal - heading + math.pi) % (2 * math.pi) - math.pi

        # ---- 履歴系バッファ（対象envのみ）----
        self.last_pos[envs_idx] = 0
        self.last_heading[envs_idx] = 0
        self.last_action[envs_idx] = 0
        # self.prev_distances[envs_idx] = 0
        self.prev_distances[envs_idx] = gd[envs_idx]
        self.prev_angle_errors[envs_idx] = torch.abs(direction_error[envs_idx])


        # ---- 観測バッファ（対象envのみ）----
        # state = [x, y, goal_dx, goal_dy, sin(yaw), cos(yaw)]
        self.state_buffer[envs_idx, 0:2] = 0
        self.state_buffer[envs_idx, 2]   = self.goal_pos[envs_idx, 0]
        self.state_buffer[envs_idx, 3]   = self.goal_pos[envs_idx, 1]
        # self.state_buffer[envs_idx, 4]   = torch.sin(yaw)
        # self.state_buffer[envs_idx, 5]   = torch.cos(yaw)
        self.state_buffer[envs_idx, 4] = 0.0
        self.state_buffer[envs_idx, 5] = 1.0


    def reset(self):
        """全環境リセット（戻り値は既存コードに合わせて state_buffer を返す）"""
        # 既存の env.py は reset で state_buffer を返す設計（copy/clone 推奨）
        all_idx = torch.arange(self.state_buffer.size(0), device=self.state_buffer.device)
        hit = torch.zeros(self.state_buffer.size(0), dtype=torch.bool, device=self.state_buffer.device)
        self.reset_idx(all_idx, hit)
        with torch.no_grad():
            # LIDARデータを読み込み、フラット化
            dist = self.sensor.read().distances.view(self.num_envs, -1)
            # 異常値は無限大でクリップ (step関数と処理を統一)
            # 1. 最大距離の設定 (例: 部屋の対角線の半分より少し大きい値など)
            MAX_LIDAR_DIST = 4.5  # 部屋のサイズが4.25mなので、適当な最大値を設定

            # 2. 異常値（inf, nan）を有限な最大距離でクリップ
            # isfiniteでない値を MAX_LIDAR_DIST に置き換える
            dist_clipped = torch.where(torch.isfinite(dist), dist, torch.full_like(dist, MAX_LIDAR_DIST))
        
            # 3. 0〜MAX_LIDAR_DIST の値を 0〜1 に正規化
            dist_normalized = dist_clipped / MAX_LIDAR_DIST

            real_vel = self.robot.get_dofs_velocity(dofs_idx_local=self.dofs_idx)

            # 6次元状態とLIDARデータを連結 (N, 634)
            new_state = torch.cat([self.state_buffer, real_vel, dist_normalized], dim=1)
        return new_state.clone()


    # def reset(self):
    #     n = self.scene.n_envs
    #     # ---- ゴールを円周上のランダムな位置に配置 ----
    #     for i in range(n):
    #         while True:
    #             angle = np.random.uniform(0, 2*np.pi)
    #             x = 2.0*np.cos(angle)
    #             y = 2.0*np.sin(angle)
    #             if np.linalg.norm([x, y]) >= self.min_goal_dist:
    #                 self.goal_pos[i] = [x, y]
    #                 break
    #     self.target.set_pos(np.hstack([self.goal_pos, np.ones((n, 1)) * 0.1]))

    #     # ---- ロボットの初期姿勢をゴール方向へ向ける ----
    #     quat_array = np.zeros((n, 4), dtype=np.float32)
    #     for i in range(n):
    #         dx = self.goal_pos[i, 0]
    #         dy = self.goal_pos[i, 1]
    #         yaw = np.arctan2(dy, dx)
    #         qz = np.sin(yaw / 2)
    #         qw = np.cos(yaw / 2)
    #         quat_array[i] = [qw, 0.0, 0.0, qz]

    #     self.robot.set_pos(np.zeros((n, 3), dtype=np.float32))
    #     self.robot.set_quat(quat_array)
    #     self.robot.set_dofs_velocity(np.zeros((n, self.robot.n_dofs), dtype=np.float32))

    #     # ---- 状態バッファ初期化 ----
    #     self.last_pos.fill(0)
    #     self.last_heading.fill(0)
    #     self.last_action.fill(0)
    #     self.prev_distances.fill(0)

    #     self.state_buffer[:, 0:2] = 0
    #     self.state_buffer[:, 2] = self.goal_pos[:, 0]
    #     self.state_buffer[:, 3] = self.goal_pos[:, 1]
    #     self.state_buffer[:, 4] = np.sin(yaw)
    #     self.state_buffer[:, 5] = np.cos(yaw)

    #     return self.state_buffer.copy()

    # ===========================
    # ステップ処理（1タイムステップ進める）
    # ===========================
    def step(self, action, B :torch.Tensor):

        n = self.scene.n_envs
        # # --- 速度制限付きゲイン ---
        # v_gain = 5
        # w_gain = 2.0
        # max_v = 5.0  # 並進速度[m/s]
        # max_w = 3  # 角速度[rad/s]

        # # --- アクション制限 ---
        # action = torch.clamp(action, -5.0, 5.0)

        # # --- 並進・旋回速度計算 ---
        # v = torch.clamp(action[:, 0] * v_gain, -max_v, max_v)
        # w = torch.clamp(action[:, 1] * w_gain, -max_w, max_w)

        # # --- 車輪速度計算 ---
        # left = v - w
        # right = v + w
        # velocity_cmd = torch.stack([left, right], dim=1)

        # # --- 実際のDOFに適用 ---
        # self.robot.control_dofs_velocity(velocity_cmd, dofs_idx_local=self.dofs_idx)
        v_gain = 12.0
        w_gain = 6.0

        v = action[:, 0] * v_gain
        w = action[:, 1] * w_gain
        left = v - w
        right = v + w
        velocity_cmd = torch.stack([left, right], dim=1)     # (N, 2)
        velocity_cmd = torch.clamp(velocity_cmd, -9.5, 9.5)
        self.robot.control_dofs_velocity(velocity_cmd, dofs_idx_local=self.dofs_idx)
        # ---- シミュレーションを1ステップ進める ----
        self.scene.step()
        if self.cam:
            self.camera.render()

        # ---- 現在位置と姿勢の取得 ----
        with torch.no_grad():
            pos_robot = self.robot.get_pos().detach()
            quat_robot = self.robot.get_quat().detach()

        # ① 取得直後に有限性チェック
        finite_mask = torch.isfinite(pos_robot).all(dim=1) & torch.isfinite(quat_robot).all(dim=1)
        if (~finite_mask).any():
            bad = torch.nonzero(~finite_mask, as_tuple=False).squeeze(-1)
            # 壊れた並列環境のみを即時リセット
            hit_for_reset = torch.zeros(bad.numel(), dtype=torch.bool, device=bad.device)
            self.reset_idx(bad, hit_for_reset)
            # リセット直後の値を取り直す
            with torch.no_grad():
                pos_robot = self.robot.get_pos().detach()
                quat_robot = self.robot.get_quat().detach()

        heading = torch.atan2(
            2 * (quat_robot[:, 0] * quat_robot[:, 3] + quat_robot[:, 1] * quat_robot[:, 2]),
            1 - 2 * (quat_robot[:, 2] ** 2 + quat_robot[:, 3] ** 2)
        )

        pos_target = self.target.get_pos().detach()
        goal_distances = torch.linalg.norm(pos_robot[:, :2] - pos_target[:, :2], dim=1)

        # ---- lidarによる障害物の判定 ----
        # collision_distances = self.sensor.read().distances
        
        #衝突判定のフラグ
        colides = torch.zeros(n, dtype=torch.bool)

        # ---- ゴール方向と現在の向きの差分 ----
        direction_to_goal = torch.atan2(
            self.goal_pos[:, 1] - pos_robot[:, 1],
            self.goal_pos[:, 0] - pos_robot[:, 0]
        )
        direction_error = (direction_to_goal - heading + math.pi) % (2 * math.pi) - math.pi

        # ===========================
        # 報酬設計
        # ===========================

        reward = torch.zeros(n, device=pos_robot.device, dtype=torch.float32)

        # 現在の角度誤差（絶対値）
        current_abs_error = torch.abs(direction_error)
        # 角度誤差の差分: prev_abs_error - current_abs_error
        angle_error_diff = 100 * (self.prev_angle_errors - current_abs_error)

        # 💡 更新: 次ステップのために現在の角度誤差を保存
        self.prev_angle_errors = current_abs_error

        # angle_reward = 12.0 * (1.0 - 2.0 * torch.abs(direction_error) / math.pi)  # 方向が合っているほど高い値（-1～1）
        
        
        # # ---- 差分報酬（近づいた分だけ加点） ----
        # goal_distance_diff = self.prev_distances - goal_distances
        # reward += goal_distance_diff * 10.0
        # self.prev_distances = goal_distances

        goal_distance_diff = self.prev_distances - goal_distances
        distance_reward = 5000.0 * goal_distance_diff

        # 次ステップ計算に向けて prev を更新
        self.prev_distances = goal_distances

        # ---- ゴール到達判定 ----
        hit = goal_distances < self.goal_threshold

        # ---- 位置・角度が変わっていなければペナルティ ----
        # not_moved = (torch.linalg.norm(pos_robot[:, :2] - self.last_pos[:, :2], dim=1) < 0.01) & \
        #             (torch.abs(heading - self.last_heading) < 0.01)
        # reward[not_moved] -= 0.1

        r_succ = self.goal_threshold     # 成功半径
        r_warm = self.goal_threshold * 3     # この距離から効き始める

        mask_warm = goal_distances < r_warm
        approach_bonus = torch.zeros(n, device=pos_robot.device, dtype=torch.float32)
        if mask_warm.any():
            B_max = 5.0   # +1.5〜+3.0 で調整
            gd_clamped = torch.maximum(goal_distances, torch.tensor(r_succ, device=goal_distances.device))

            # 進捗ボーナス計算（正規化：r_warm～r_succ の間で 0→1 に）
            bonus = B_max * (r_warm - gd_clamped) / (r_warm - r_succ)
            # ボーナスは mask_warm の要素にのみ適用
            approach_bonus[mask_warm] = bonus[mask_warm]
        # jerk = torch.linalg.norm(action - self.last_action, dim=1)
        # reward -= 0.01 * jerk

        step_penalty = 0.07

        #衝突判定
        # ---- lidarによる障害物の判定（envごとに最短距離→hit判定）----
        dist = self.sensor.read().distances            # (n_envs, n_rays[, 1])
        # torch.save(dist, "tensor_data.tet")
        dist = dist.view(dist.shape[0], -1)            # (n_envs, R) にフラット化

        # 1. 最大距離の設定 (例: 部屋の対角線の半分より少し大きい値など)
        MAX_LIDAR_DIST = 4.5  # 部屋のサイズが4.25mなので、適当な最大値を設定

        # 2. 異常値（inf, nan）を有限な最大距離でクリップ
        # isfiniteでない値を MAX_LIDAR_DIST に置き換える
        dist_clipped = torch.where(torch.isfinite(dist), dist, torch.full_like(dist, MAX_LIDAR_DIST))
        
        # 3. 0〜MAX_LIDAR_DIST の値を 0〜1 に正規化
        dist_normalized = dist_clipped / MAX_LIDAR_DIST


        threshold = 0.28  # m （必要に応じて調整）
        min_dist, _ = dist.min(dim=1)                  # (n_envs,) 各envの最短レイ
        accident = min_dist < threshold                     # (n_envs,) 閾値内に何かある

        #lidar_penalty
        lidar_penalty_term = torch.sum(torch.clamp(0.4 - dist, min=0.0), dim=1) * 0.6
        # lidar_penalty[accident] = 200.0

        MAX_STEPS = 1500.0 # run.py の max_steps と合わせる (ここでは定数として定義)
        BASE_PENALTY = 0.07 # 基本ペナルティ (最初のステップのペナルティ)
        
        # B/MAX_STEPS は 1ステップ目でほぼ0、MAX_STEPSで1に近づく
        # 例: B/MAX_STEPS に 10倍の重みをつけてペナルティに加算
        penalty_scale = B.to(reward.dtype) / MAX_STEPS 
        step_penalty = BASE_PENALTY + (penalty_scale * 0.5) # 0.07 から 0.57 程度まで増加


        # （任意）近いほど強い連続ペナルティ
        # penalty = 0.01 * torch.clamp(threshold - min_dist, min=0.0)

        reward = angle_error_diff + distance_reward + approach_bonus - step_penalty - lidar_penalty_term
        # reward = angle_error_diff

        real_vel = self.robot.get_dofs_velocity(dofs_idx_local=self.dofs_idx)



        # ---- 状態更新 ----
        self.last_pos[:] = pos_robot
        self.last_heading[:] = heading
        self.last_action[:] = action

        self.state_buffer[:, 0] = pos_robot[:, 0]
        self.state_buffer[:, 1] = pos_robot[:, 1]
        self.state_buffer[:, 2] = self.goal_pos[:, 0] - pos_robot[:, 0]
        self.state_buffer[:, 3] = self.goal_pos[:, 1] - pos_robot[:, 1]
        self.state_buffer[:, 4] = torch.sin(heading)
        self.state_buffer[:, 5] = torch.cos(heading)

        # 💡 修正箇所: 既存の6次元状態とLIDARの全データ (628次元) を連結
        new_state = torch.cat([self.state_buffer, real_vel, dist_normalized], dim=1) # (N, 6 + 628) = (N, 634)


        return new_state.clone(), reward, accident, hit, {}
    
    # ===========================
    # 四方を囲む壁
    # ===========================
    
def add_square_room(scene, inner_size: float = 1.0, height: float = 1.0,
    thickness: float = 0.05, center=(0.0, 0.0), z0: float = 0.0,
    color=(0.7, 0.7, 0.7)) -> None:
    """内寸が inner_size×inner_size の正方形の壁を 4 面追加する。


    並列環境（n_envs>1）でも Scene.build() 前に呼び出せば、
    各環境に同一配置で複製されます。


    Args:
    scene: gs.Scene インスタンス
    inner_size: 壁で囲まれた内寸（メートル）
    height: 壁の高さ（メートル）
    thickness: 壁の厚み（メートル）
    center: (cx, cy) で部屋中心の平面位置
    z0: 床面の Z 座標（通常 0.0）
    color: (r,g,b) 0-1 の色
    """
    cx, cy = center
    half = inner_size * 0.5
    hz = z0 + height * 0.5


    surf = gs.surfaces.Rough(
    diffuse_texture=gs.textures.ColorTexture(color=(1, 0.6, 0))
    )


    # 右壁（+X）: X 方向は厚み、Y 方向は内寸
    scene.add_entity(
    gs.morphs.Box(
    size=(thickness, inner_size, height),
    pos=(cx + half + thickness * 0.5, cy, hz),
    fixed=True,
    ),
    surface=surf,
    )


    # 左壁（-X）
    scene.add_entity(
    gs.morphs.Box(
    size=(thickness, inner_size, height),
    pos=(cx - half - thickness * 0.5, cy, hz),
    fixed=True,
    ),
    surface=surf,
    )


    # 上壁（+Y）: Y 方向は厚み、X 方向は内寸
    scene.add_entity(
    gs.morphs.Box(
    size=(inner_size, thickness, height),
    pos=(cx, cy + half + thickness * 0.5, hz),
    fixed=True,
    ),
    surface=surf,
    )


    # 下壁（-Y）
    scene.add_entity(
    gs.morphs.Box(
    size=(inner_size, thickness, height),
    pos=(cx, cy - half - thickness * 0.5, hz),
    fixed=True,
    ),
    surface=surf,
    )
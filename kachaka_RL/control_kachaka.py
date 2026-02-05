import numpy as np
import genesis as gs
from pynput import keyboard
import math
import random
import torch
import ast
import os
import time

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

# =========================================
# ログ読み込み・データ準備用の関数
# =========================================
def load_trajectories_from_log(log_file="log.txt"):
    """log.txtから軌跡データを読み込む"""
    trajectories = []
    if not os.path.exists(log_file):
        print(f"警告: {log_file} が見つかりません。軌跡の描画はスキップされます。")
        return []

    print(f"{log_file} から軌跡データを読み込んでいます...")
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        # utf-8で読めない場合はcp932(Shift-JIS)を試す
        with open(log_file, "r", encoding="cp932") as f:
            lines = f.readlines()

    is_data_section = False
    for line in lines:
        line = line.strip()
        # データセクションの開始マーカーを探す
        if "=== 全エピソードの軌跡データ" in line:
            is_data_section = True
            continue
        
        if is_data_section and line.startswith("[["):
            try:
                # 文字列形式のリストをPythonのリストに変換
                traj_data = ast.literal_eval(line)
                trajectories.append(traj_data)
            except Exception as e:
                print(f"軌跡データのパースエラー: {e}")
    
    print(f"{len(trajectories)} 件の軌跡データを読み込みました。")
    return trajectories

def prepare_debug_lines_tensor(trajectories_2d, device, z_height=0.05):
    """2D軌跡データをGenesisのdraw_debug_lines用の3D Tensorに変換する"""
    if not trajectories_2d:
        return None, None

    all_starts = []
    all_ends = []

    for traj in trajectories_2d:
        if len(traj) < 2:
            continue
        
        # リストをTensorに変換 (N, 2)
        points_2d = torch.tensor(traj, dtype=torch.float32, device=device)
        
        # Z軸(高さ)を追加して3Dにする (N, 3)。床と被らないように少し浮かせる。
        z_coords = torch.full((points_2d.shape[0], 1), z_height, device=device)
        points_3d = torch.cat([points_2d, z_coords], dim=1)

        # 線分の始点リストと終点リストを作成
        # starts: p_0, p_1, ..., p_{n-1}
        # ends:   p_1, p_2, ..., p_n
        starts = points_3d[:-1]
        ends = points_3d[1:]
        
        all_starts.append(starts)
        all_ends.append(ends)

    if not all_starts:
        return None, None

    # 全エピソードの線分データを結合して一つの大きなTensorにする
    final_starts = torch.cat(all_starts, dim=0)
    final_ends = torch.cat(all_ends, dim=0)
    
    return final_starts, final_ends


class KachakaEnv:
    def __init__(self, viewer=True):
        ########################## 初期化 ##########################
        gs.init(backend=gs.gpu)

        ########################## シーンの作成 ##########################
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            show_viewer=True,
        )

        ########################## エンティティ ##########################
        # 地面
        plane = self.scene.add_entity(gs.morphs.Plane(),surface=gs.surfaces.Rough(
                # diffuse_texture=gs.textures.ColorTexture(color=(0.5, 0.5, 0.5))
            ),)
        # ロボット
        kachaka = self.scene.add_entity(
            gs.morphs.URDF(file='/home/chujoken/Genesis/kachaka-api/ros2/kachaka_02.urdf'),
        )
        self.cam = self.scene.add_camera(
            res    = (600, 600),
            pos    = (23,23,15),
            lookat = (23, 23, 0),
            fov    = 60,
            GUI    = True
        )

        # self.light = self.scene.add_light(
        #     pos=(0,0,50),
        #     dir=(0,0,0)
        # )


        for l in plane.links:
            for g in l.geoms:
                g.set_friction(0.01)

        for l in kachaka.links:
            if l.name in ['caster_sphere_left', 'caster_sphere_right']:
                for g in l.geoms:
                    g.set_friction(0.01)

        # 目印球
        # center = (0.0, 0.0)  # ロボットの中心位置
        # radius =2.0  # ロボットからの距離
        # angle = random.uniform(0, 2 * math.pi)
        # x = center[0] + radius * math.cos(angle)
        # y = center[1] + radius * math.sin(angle)
        # target = scene.add_entity(
        #     morph=gs.morphs.Sphere(
        #         radius=0.1,
        #          fixed=True,
        #           collision=False,
        #           pos=(x, y, 0.1),
        #           ),
        #     surface=gs.surfaces.Rough(
        #         diffuse_texture=gs.textures.ColorTexture(color=(1, 0, 0))
        #     ),
        # )
        # add_square_room(self.scene, inner_size=4.5, height=1.0, thickness=0.05, center=(0.0, 0.0), z0=0.0)
                
        ########################## ビルド ##########################
        B = 1
        self.scene.build(
            n_envs=B,
            env_spacing=(4.6,4.6)
        )

        jnt_names = [
            'base_r_drive_wheel_joint',
            'base_l_drive_wheel_joint',
        ]
        dofs_idx = [kachaka.get_joint(name).dof_idx_local for name in jnt_names]

        ############ オプション：制御ゲインの設定 ############
        kachaka.set_dofs_kp(kp=np.array([4500, 4500]), dofs_idx_local=dofs_idx)
        kachaka.set_dofs_kv(kv=np.array([450, 450]), dofs_idx_local=dofs_idx)
        kachaka.set_dofs_force_range(
            lower=np.array([-87, -87]),
            upper=np.array([87, 87]),
            dofs_idx_local=dofs_idx,
        )

        # 衝突判定用に衝突ジオメトリ情報を取り出す
        # def get_geom_ids(entity):
        #     ids = []
        #     for link in entity.links:
        #         for geom in link.geoms:
        #             ids.append(geom.idx)
        #     return set(ids)

        # target_geoms = get_geom_ids(target)
        # def reset(self):
        #     # fixed=Trueにしたので特に処理不要ですが、念のためstepを呼んで初期化完了させます
        #     self.scene.step()


if __name__ == '__main__':
    env = KachakaEnv(viewer=True)
    # env.reset()
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

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    ########################## メインループ ##########################
    # velocity_cmd = np.zeros((1, len(env.dofs_idx)))  # B = 1 なので shape=(1, 2)
    # trk = 10.0
    # step_count = 0
    # at_target_threshold = 0.3  # 距離しきい値[m]

    # while True:
    #     velocity_cmd[:] = 0  # 初期化
    #     if key_state['up']:
    #         velocity_cmd[:] = trk
    #     if key_state['down']:
    #         velocity_cmd[:] = -trk
    #     if key_state['right']:
    #         velocity_cmd = np.array([[-trk, trk]] * 1)
    #     if key_state['left']:
    #         velocity_cmd = np.array([[trk, -trk]] * 1)

        # env.kachaka.control_dofs_velocity(velocity_cmd, env.dofs_idx)
        # env.scene.step()


    # # 2. ログ読み込み & 描画データ準備
    # raw_trajectories = load_trajectories_from_log("log.txt")
    # starts, ends = prepare_debug_lines_tensor(
    #     raw_trajectories, 
    #     device=torch.device("cuda"), 
    #     z_height=0.05
    # )

    # print("\nSimulation started. Press 'ESC' in the viewer to exit.")

    # # 3. 軌跡の描画 (ループの外で1回だけ実行！)
    # if starts is not None and ends is not None:
    #     print("Drawing trajectories...")
    #     # Tensorの行数分だけ繰り返して1本ずつ追加
    #     count = starts.shape[0]
    #     for i in range(count):
    #         try:
    #             env.scene.draw_debug_line(
    #                 start=starts[i].clone(), 
    #                 end=ends[i].clone(),     
    #                 color=(0.0, 0.0, 0.0),
    #                 radius=0.03
    #             )
    #         except Exception:
    #             # 重複エラーは無視して次へ
    #             pass
    #     print("Done drawing.")

    # 3. 描画ループ
    while True:
        # シミュレーションを進める（時間を進めて描画を更新）
        env.scene.step()
        env.cam.render()
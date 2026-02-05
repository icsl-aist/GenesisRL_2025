from env import KachakaEnv
# from ppo import PPOAgent
from ppo import PPO
import matplotlib.pyplot as plt
import argparse
import torch
import numpy as np

# ===========================
# メイン関数
# ===========================
def main():
    # ---- コマンドライン引数 ----
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer', action='store_true')  # 推論モード（学習済モデルで走らせる）
    args = parser.parse_args()

    # ---- 環境・エージェントの基本設定 ----
    state_dim = 32        # 状態の次元数（位置+目標相対+sinθ+cosθ）
    action_dim = 2       # アクションの次元数（速度 v と 旋回 ω）
    # agent = PPOAgent(state_dim, action_dim, n_envs=20, max_steps=2000)  # PPOエージェント生成
    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        n_envs=1,
        T=2048,                 # 収集ステップ数（例）
        n_epochs=10,           # SB3 デフォ寄り
        minibatch_size=4096,   # T×N=5120 なら 2048/2048/1024 などで回る
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        clip_vf=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
    )

    num_episodes = 100000  # 学習エピソード数
    max_steps = 1500      # 1エピソードあたりの最大ステップ数
    reward_history = []   # 報酬履歴を格納

    # ===========================
    # 推論モード
    # ===========================
    if args.infer:
        agent.load('./ppo_env1_lidar24_model_final.pth')
        agent.reset_buffer()
        env = KachakaEnv(min_goal_dist=1.0, viewer=True, cam=True)
        
        # --- 統計用変数 ---
        total_distances = []    # 各エピソードの総移動距離
        success_count = 0       # 成功回数
        all_trajectories = []   # 軌跡データ
        log = []
        
        S_t = float(max_steps)
        state = env.reset()

        test_episodes = 50  # テスト回数
        
        for ep_idx in range(test_episodes): 
            logA = []
            accident = torch.zeros(1, dtype=torch.bool, device=state.device)
            B = torch.zeros(agent.n_envs, dtype=torch.int, device=state.device)
            total_reward = torch.tensor(0.0, device=state.device)
            mask = accident | (B >= max_steps)
            
            # --- エピソードごとの追跡変数 ---
            current_traj = [] 
            # 距離累積用テンソル (GPU上に保持)
            episode_dist_tensor = torch.tensor(0.0, device=state.device)
            is_success = False

            # 初期位置 (Tensorのまま保持)
            # get_pos() は (n_envs, 3) を返すので [0, :2] で (x, y) を取得
            prev_pos = env.robot.get_pos()[:, :2].clone() 
            
            # 軌跡用にリストへ保存 (ここでは tolist() でPythonのfloatリストにする)
            current_traj.append(prev_pos[0].tolist())

            while not mask.any():
                B += 1
                action = agent.select_action(state)
                state, reward, accident, hit, _ = env.step(action, B)

                # --- 【修正箇所】Torchによる位置・距離計算 ---
                # 現在位置取得 (Tensor)
                curr_pos = env.robot.get_pos()[:, :2]
                
                # ベクトル差分のノルム計算 (Tensor演算)
                dist_step = torch.linalg.norm(curr_pos - prev_pos, dim=1)
                
                # 距離を加算 (Tensor)
                episode_dist_tensor += dist_step[0]
                
                # 軌跡保存 (リストへ追加)
                current_traj.append(curr_pos[0].tolist())
                
                # 次ステップのために位置を更新 (cloneして参照を切る)
                prev_pos = curr_pos.clone()
                
                # --- 判定ロジック ---
                if hit.any():
                    idx_hit = torch.nonzero(hit, as_tuple=False).squeeze(-1)
                    succ_scale = 1.0 - (B - 1.0) / S_t
                    succ_scale = max(succ_scale, 0.2)
                    reward[idx_hit] += 2000.0 * succ_scale
                    log.append(f"Episode {ep_idx+1}: ヒット (Steps: {B.item()})")
                    is_success = True
                
                elif accident.any():
                    idx_fail = torch.nonzero(accident, as_tuple=False).squeeze(-1)
                    reward[idx_fail] += -1500.0
                    log.append(f"Episode {ep_idx+1}: 衝突 (Steps: {B.item()})")

                mask = accident | (B >= max_steps) | hit
                total_reward += reward.sum()
            
            # --- エピソード終了時の集計 ---
            if is_success:
                success_count += 1
            
            # TensorからPythonのfloatへ変換して保存
            episode_dist_val = episode_dist_tensor.item()
            total_distances.append(episode_dist_val)
            all_trajectories.append(current_traj)

            env.reset()
            logA.append(f"エピソード合計報酬: {total_reward.item():.2f}")
            print(f"エピソード {ep_idx+1} 終了: 距離={episode_dist_val:.2f}m, 成功={is_success}")

        # --- 統計計算 (ここはNumpyで一括計算が楽なので使用) ---
        avg_dist = np.mean(total_distances)
        min_dist = np.min(total_distances)
        max_dist = np.max(total_distances)
        std_dist = np.std(total_distances)
        
        # --- ログ出力 ---
        summary_lines = [
            "========================================",
            f"テストエピソード数: {test_episodes}",
            f"成功回数: {success_count}",
            f"平均移動距離: {avg_dist:.4f} m",
            f"最短移動距離: {min_dist:.4f} m",
            f"最長移動距離: {max_dist:.4f} m",
            f"移動距離標準偏差: {std_dist:.4f}",
            "========================================",
        ]
        
        with open(f"log.txt", "w") as f:
            for line in log:
                f.write(line + "\n")
            f.write("\n")
            for line in summary_lines:
                f.write(line + "\n")
                print(line)

            f.write("\n=== 全エピソードの軌跡データ (X, Y) ===\n")
            for i, traj in enumerate(all_trajectories):
                f.write(f"Episode {i+1} Trajectory (Points: {len(traj)}):\n")
                f.write(str(traj) + "\n")

        print("ログの保存が完了しました: log.txt")
        return

    # ===========================
    # 学習モード
    # ===========================
    else:
        env = KachakaEnv(min_goal_dist=1.0, viewer=False)  # ビューアなしで高速学習

    save_counter = 0  # モデル保存のカウンタ

    ep = 0
    A = 0
    state = env.reset()
    log = []
    B = torch.zeros(agent.n_envs, dtype=torch.int, device=state.device)
    # B=0
    total_reward = torch.tensor(0.0, device=state.device)
    # S_t = torch.tensor(float(max_steps), device=state.device)
    S_t = float(max_steps)
    logA = []
    while ep < num_episodes:
        A += 1
        B += 1

        action = agent.select_action(state)
        reward = torch.zeros(agent.n_envs, device=state.device)
        next_state, reward, accident, hit, _ = env.step(action, B)  # 環境を1ステップ進める

        mask = accident | hit | (B >= max_steps)
        mask2 = accident | hit

        if mask.any():
        # if B >= S_t:
            if hit.any():
                idx_hit = torch.nonzero(hit, as_tuple=False).squeeze(-1)
                succ_scale = 1.0 - (B[idx_hit].to(reward.dtype) - 1.0) / S_t
                # succ_scale = 1.0 - (B - 1.0) / S_t
                succ_scale = torch.clamp(succ_scale, max=0.2)
                # succ_scale = max(succ_scale, 0.2)
                reward[idx_hit] += 2000.0 * succ_scale
            
            elif accident.any():
                idx_fail = torch.nonzero(accident, as_tuple=False).squeeze(-1)

                reward[idx_fail] += -500.0

            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            a = accident
            c = hit
            for i in idx.tolist():
                if a.any():
                    log.append(
                        f"{ep}エピソード終了: accident env={i} 報酬: {reward[i].item():.2f} ステップ数: {B[i].item()}"
                    )
                if (B >= max_steps).any():
                    log.append(
                        f"{ep}エピソード終了: max_steps env={i} 報酬: {reward[i].item():.2f} ステップ数: {B[i].item()}"
                    )
                if c.any():
                    log.append(
                        f"{ep}エピソード終了: hit env={i} 報酬: {reward[i].item():.2f} ステップ数: {B[i].item()}"
                    )
            env.reset_idx(idx,hit)
            # env.reset()
            # あなたの env では最新観測は state_buffer に反映される前提
            with torch.no_grad():
                # reset_idxでリセットした環境の最新LIDARデータを取得する必要がある
                dist = env.sensor.read().distances.view(env.num_envs, -1)
                dist_filtered = torch.where(torch.isfinite(dist), dist, torch.full_like(dist, float('inf')))
                real_vel = env.robot.get_dofs_velocity(dofs_idx_local=env.dofs_idx)
                
                # 6次元状態とLIDARデータを連結 (N, 634)
                # env.state_bufferはreset_idxで更新済み
                next_state_after_reset = torch.cat([env.state_buffer, real_vel, dist_filtered], dim=1)
            next_state = next_state_after_reset

            # 報酬合計（ログ用）
            total_reward += reward.sum()
            # if a.any():
            #     log.append(f"エピソード終了: done 報酬: {total_reward.item():.2f} ステップ数: {B}")
            # if B >= max_steps:
            #     log.append(f"エピソード終了: max_steps 報酬: {total_reward.item():.2f} ステップ数: {B}")
            # if c.any():
            #     log.append(f"エピソード終了: hit 報酬: {total_reward.item():.2f} ステップ数: {B}")
            # logA.append(f"エピソード合計報酬: {total_reward.item():.2f}")

            B[idx] = 0
            ep += 1
            # 報酬＆done を記録（GAE 用）
            agent.store_step(reward=reward, done=mask2)

            # ---- 定期的にモデル保存 ----
            if (ep + 1) % 50 == 0:
                save_counter += 1
                agent.save(f"ppo_env1_lidar24_model_episode_{save_counter}.pth")
                print(f"Model saved at episode {ep + 1}")

        else:
            # 報酬合計（ログ用）
            total_reward += reward.sum()
            # 報酬＆done を記録（GAE 用）
            agent.store_step(reward=reward, done=mask)
        state = next_state

        if A >= agent.T:
            # 末端の V(s_T) でブートストラップして更新
            last_values = agent.value(state)
            agent.update(last_values=last_values)

            reward_history.append(total_reward)
            print(f"Iter {ep}\tTotal Reward(sum over N): {total_reward:.2f}")
            A = 0
            total_reward = 0.0

    # ===========================
    # 学習後の処理
    # ===========================
    agent.save("ppo_env1_lidar24_model_final.pth")  # 最終モデル保存

    with open(f"log.txt", "w") as f:
                for line in log:
                    f.write(line + "\n")

    # ---- 報酬曲線をプロット ----
    plt.plot(torch.stack(reward_history).detach().cpu().numpy())
    plt.xlabel('update frequency')
    plt.ylabel('Total Reward')
    plt.title('Training Reward Curve')
    plt.show()


# スクリプトのエントリーポイント
if __name__ == "__main__":
    main()

# ppo.py  — SB3に近づけたPPO（GAE/Adv正規化/KL停止/ValueClip/Entropy/learnable log_std/Minibatch&Multi-epoch）
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ====== ユーティリティ ======
def mlp(sizes, activation=nn.Tanh, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else out_act
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if act is not None:
            layers += [act()]
    return nn.Sequential(*layers)


def combined_shape(T: int, N: int, *dims) -> Tuple[int, ...]:
    return (T, N) + dims


# ====== Actor-Critic ======
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden=(128, 128)):
        super().__init__()
        self.net = mlp([state_dim, *hidden, action_dim], activation=nn.ReLU, out_act=None)
        # 学習可能な log_std（各次元同一σでも、ベクトルとして持つ）
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(x)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def dist(self, x: torch.Tensor) -> Normal:
        mean, std = self.forward(x)
        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden=(128, 128)):
        super().__init__()
        self.v = mlp([state_dim, *hidden, 1], activation=nn.ReLU, out_act=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(x).squeeze(-1)


# ====== PPO（SB3風の更新手順） ======
class PPO:
    """
    SB3に近い更新：GAE、adv正規化、クリップ付きPPO損失、価値クリップ、エントロピー項、
    近似KLによる早期停止、勾配クリップ、ミニバッチ＆複数エポック。
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_envs: int,
        T: int = 256,
        actor_hidden=(512, 512),
        critic_hidden=(512, 512),
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        clip_vf: Optional[float] = 0.2,     # Noneで無効
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        minibatch_size: int = 2048,
        target_kl: Optional[float] = 0.02,  # Noneで無効
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.n_envs = n_envs
        self.T = T
        self.gamma = gamma
        self.lam = gae_lambda
        self.clip_eps = clip_eps
        self.clip_vf = clip_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.target_kl = target_kl

        self.actor = Actor(state_dim, action_dim, actor_hidden).to(self.device, self.dtype)
        self.critic = Critic(state_dim, critic_hidden).to(self.device, self.dtype)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        # ロールアウトバッファ（(T, N, …)）
        self._alloc_buffers(state_dim, action_dim)

        # 書き込みインデックス
        self.t = 0

        # 直近のロギング値
        self.last_info = {}

    def _alloc_buffers(self, state_dim: int, action_dim: int):
        T, N = self.T, self.n_envs
        dev, dt = self.device, self.dtype
        self.states   = torch.zeros(combined_shape(T, N, state_dim), device=dev, dtype=dt)
        self.actions  = torch.zeros(combined_shape(T, N, action_dim), device=dev, dtype=dt)
        self.logprobs = torch.zeros(combined_shape(T, N), device=dev, dtype=dt)
        self.rewards  = torch.zeros(combined_shape(T, N), device=dev, dtype=dt)
        self.dones    = torch.zeros(combined_shape(T, N), device=dev, dtype=dt)
        self.values   = torch.zeros(combined_shape(T, N), device=dev, dtype=dt)
        self.advantages = torch.zeros(combined_shape(T, N), device=dev, dtype=dt)
        self.returns    = torch.zeros(combined_shape(T, N), device=dev, dtype=dt)
        

    def reset_buffer(self):
        # 学習後に再利用する場合にゼロ初期化（速度優先なら省略可）
        self.states.zero_()
        self.actions.zero_()
        self.logprobs.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.values.zero_()
        self.advantages.zero_()
        self.returns.zero_()
        self.t = 0

    @torch.no_grad()
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (N, state_dim)  すべてGPU & dtype統一済み前提
        返り値: action (N, action_dim)
        副作用: バッファに state, action, logprob, value を書き込む
        """
        s = state.to(self.device, self.dtype)

        dist = self.actor.dist(s)
        a = dist.sample()
        logp = dist.log_prob(a).sum(-1)

        v = self.critic(s)

        # バッファ記録
        self.states[self.t].copy_(s)
        self.actions[self.t].copy_(a)
        self.logprobs[self.t].copy_(logp)
        self.values[self.t].copy_(v)

        return a

    def store_step(self, reward: torch.Tensor, done: torch.Tensor):
        """
        reward, done: (N,)
        """
        self.rewards[self.t].copy_(reward.to(self.device, self.dtype))
        # doneは {0,1} を想定（floatにしてmask計算で使う）
        self.dones[self.t].copy_(done.to(self.device, self.dtype))
        self.t += 1

    @torch.no_grad()
    def _compute_gae(self, last_values: torch.Tensor):
        """
        GAE & Returns 計算
        last_values: (N,)  ロールアウト最後の状態価値（done=0 の箇所をブートストラップ）
        """
        T, N = self.T, self.n_envs
        gamma, lam = self.gamma, self.lam

        adv = torch.zeros((N,), device=self.device, dtype=self.dtype)
        for t in reversed(range(T)):
            next_v = last_values if t == T - 1 else self.values[t + 1]
            not_done = 1.0 - self.dones[t]   # done=1 → ブートストラップなし
            delta = self.rewards[t] + gamma * not_done * next_v - self.values[t]
            adv = delta + gamma * lam * not_done * adv
            self.advantages[t] = adv

        self.returns.copy_(self.advantages + self.values)

        # アドバンテージ正規化
        flat_adv = self.advantages.view(-1)
        std = flat_adv.std(unbiased=False)
        if torch.isfinite(std) and std > 0:
            self.advantages = (self.advantages - flat_adv.mean()) / (std + 1e-8)
        else:
            self.advantages.zero_()

    def _iter_minibatches(self, batch_size: int):
        """
        (T×N) のインデックスをシャッフルしてミニバッチをyieldする
        """
        T, N = self.T, self.n_envs
        total = T * N
        idx = torch.randperm(total, device=self.device)
        for start in range(0, total, batch_size):
            mb = idx[start:start + batch_size]
            yield mb

    def _flatten_batch(self):
        """
        (T, N, …) → (T×N, …)
        """
        T, N = self.T, self.n_envs
        S = self.states.reshape(T * N, -1)
        A = self.actions.reshape(T * N, -1)
        OLP = self.logprobs.reshape(T * N)
        ADV = self.advantages.reshape(T * N)
        RET = self.returns.reshape(T * N)
        VAL = self.values.reshape(T * N)
        return S, A, OLP, ADV, RET, VAL

    def update(self, last_values: Optional[torch.Tensor] = None):
        """
        agent.py の train() と同じロジックで PPO 更新を行う版
        - policy clip: self.clip_eps
        - value clip: self.clip_eps（agent の clip_range と同じ値を使用）
        - advantage は正規化
        - KL による early stop は使わない
        """
        # 末端価値で GAE / Returns を計算
        with torch.no_grad():
            if last_values is None:
                last_values = self.critic(self.states[-1])
        self._compute_gae(last_values)

        # (T, N, …) → (T*N, …)
        S, A, OLP, ADV, RET, OLDV = self._flatten_batch()

        entropy_log = []
        actor_log = []
        critic_log = []

        for _ in range(self.n_epochs):
            for mb in self._iter_minibatches(self.minibatch_size):
                s      = S[mb]
                a      = A[mb]
                old_lp = OLP[mb]
                adv    = ADV[mb]
                ret    = RET[mb]
                old_v  = OLDV[mb]

                # ---- Actor 部分（PPO クリップ）----
                dist = self.actor.dist(s)
                new_lp = dist.log_prob(a).sum(-1)          # log πθ(a|s)
                ratio  = torch.exp(new_lp - old_lp)        # πθ / πθ_old

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio,
                                    1.0 - self.clip_eps,
                                    1.0 + self.clip_eps) * adv

                # agent.train() と同じ：
                # - min(surr1, surr2) のマイナス
                # - エントロピー項を引く
                actor_loss = -torch.mean(torch.min(surr1, surr2)) \
                             - self.ent_coef * dist.entropy().sum(-1).mean()

                # ---- Critic 部分（value clip）----
                v_pred = self.critic(s)

                # agent と同じ: old_values + clamp(v_pred - old_values)
                v_clipped = old_v + (v_pred - old_v).clamp(-self.clip_eps,
                                                           self.clip_eps)

                critic_loss_1 = (ret - v_pred) ** 2
                critic_loss_2 = (ret - v_clipped) ** 2

                # max( unclipped, clipped ) を 0.5 係数で平均
                critic_loss = 0.5 * torch.mean(
                    torch.max(critic_loss_1, critic_loss_2)
                )

                # ---- 総 loss & 最適化 ----
                loss = actor_loss + self.vf_coef * critic_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) +
                        list(self.critic.parameters()),
                        self.max_grad_norm,
                    )
                self.optimizer.step()

                # ---- ログ保存 ----
                entropy_log.append(dist.entropy().sum(-1).mean().item())
                actor_log.append(actor_loss.item())
                critic_log.append(critic_loss.item())

        # ログ用の平均値を保存（必要なら）
        def _mean(x): return float(sum(x) / max(len(x), 1))
        self.last_info = {
            "entropy": _mean(entropy_log),
            "policy_loss": _mean(actor_log),
            "value_loss": _mean(critic_log),
            # agent には無いが、必要なら他も追加
        }

        # バッファをリセットして次のロールアウトへ
        self.reset_buffer()


    # ===== 学習ループ側ユーティリティ =====
    @torch.no_grad()
    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state.to(self.device, self.dtype))

    def info(self):
        return self.last_info
    
        # ===== モデル保存・読み込み =====
    def save(self, path: str):
        """
        Actor / Critic の重みを保存
        """
        data = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": {
                "gamma": self.gamma,
                "lam": self.lam,
                "clip_eps": self.clip_eps,
                "clip_vf": self.clip_vf,
                "ent_coef": self.ent_coef,
                "vf_coef": self.vf_coef,
                "max_grad_norm": self.max_grad_norm,
                "n_epochs": self.n_epochs,
                "minibatch_size": self.minibatch_size,
                "target_kl": self.target_kl,
            }
        }
        torch.save(data, path)
        print(f"[PPO] モデルを保存しました → {path}")

    def load(self, path: str, strict: bool = True):
        """
        保存した重みをロード
        """
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data["actor"], strict=strict)
        self.critic.load_state_dict(data["critic"], strict=strict)
        if "optimizer" in data:
            self.optimizer.load_state_dict(data["optimizer"])
        print(f"[PPO] モデルを読み込みました ← {path}")


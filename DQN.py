import socket
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import uuid

# ==============================
#  Configuration – must match Unity
# ==============================
GRID_SIZE = 24
NUM_CHANNELS = 8
HISTORY_LENGTH = 3                # N (number of past snapshots stacked)
NUM_SPAWNERS = 5                  # number of spawners in the scene

# Grid state shape (after stacking history along channels)
GRID_SHAPE = (HISTORY_LENGTH * NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
GRID_FLAT_LEN = HISTORY_LENGTH * GRID_SIZE * GRID_SIZE * NUM_CHANNELS

# Spawner rates shape (flat vector of length HISTORY_LENGTH * NUM_SPAWNERS)
SPAWNER_FLAT_LEN = HISTORY_LENGTH * NUM_SPAWNERS

HOST = "127.0.0.1"
PORT = 50007
CHECKPOINT_DIR = "./checkpoints_v2"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

wandb.init(
    project="IndustryDQN_CNN",
    name=f"CNN_DQN_{uuid.uuid4().hex[:8]}",
    mode="online"
)

# ==============================
#  Replay Buffer (stores both grid and spawner data)
# ==============================
class ReplayBuffer:
    def __init__(self, capacity, grid_shape, spawner_len):
        self.capacity = capacity
        self.grid_shape = grid_shape          # (C, H, W)
        self.spawner_len = spawner_len

        self.grid_states = np.zeros((capacity, *grid_shape), dtype=np.float32)
        self.spawner_states = np.zeros((capacity, spawner_len), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_grid_states = np.zeros((capacity, *grid_shape), dtype=np.float32)
        self.next_spawner_states = np.zeros((capacity, spawner_len), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.pos = 0
        self.size = 0

    def push(self, grid, spawner, a, r, next_grid, next_spawner, done):
        idx = self.pos
        self.grid_states[idx] = grid
        self.spawner_states[idx] = spawner
        self.actions[idx] = a
        self.rewards[idx] = r
        self.next_grid_states[idx] = next_grid
        self.next_spawner_states[idx] = next_spawner
        self.dones[idx] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "grid_states": self.grid_states[idxs],
            "spawner_states": self.spawner_states[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "next_grid_states": self.next_grid_states[idxs],
            "next_spawner_states": self.next_spawner_states[idxs],
            "dones": self.dones[idxs],
        }

    def __len__(self):
        return self.size

# ==============================
#  Q-Network (CNN for grid + MLP for spawner rates)
# ==============================
class QNetwork(nn.Module):
    def __init__(self, grid_shape, num_spawners, history_length):
        super().__init__()
        C, H, W = grid_shape
        # CNN branch for the grid
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # MLP branch for spawner rates history
        self.fc_spawner = nn.Sequential(
            nn.Linear(history_length * num_spawners, 32),
            nn.ReLU()
        )
        # Combined layers (CNN output + spawner features + action)
        self.fc_combined = nn.Sequential(
            nn.Linear(64 + 32 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.node_id_scale = 100.0  # for action normalization

    def forward(self, grid, spawner, action_id):
        """
        grid:    (B, C, H, W)
        spawner: (B, T*num_spawners)  flattened history of spawner rates
        action_id: (B, 1) normalized float
        """
        x = self.conv(grid)                     # (B, 64, 1, 1)
        x = x.view(x.size(0), -1)                # (B, 64)
        s = self.fc_spawner(spawner)             # (B, 32)
        combined = torch.cat([x, s, action_id], dim=1)  # (B, 64+32+1)
        q = self.fc_combined(combined)           # (B, 1)
        return q

# ==============================
#  DQN Learner
# ==============================
class DqnLearner:
    def __init__(
        self,
        grid_shape,
        num_spawners,
        history_length,
        gamma=0.99,
        lr=3e-4,
        batch_size=64,
        capacity=50_000,
        warmup=1000,
        tau=0.005,
        device=None
    ):
        self.grid_shape = grid_shape
        self.num_spawners = num_spawners
        self.history_length = history_length
        self.spawner_len = history_length * num_spawners
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup = warmup
        self.tau = tau
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(grid_shape, num_spawners, history_length).to(self.device)
        self.target_net = QNetwork(grid_shape, num_spawners, history_length).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(capacity, grid_shape, self.spawner_len)

        self.train_step_count = 0
        self.known_actions = set()
        self.node_id_scale = 100.0

        print(f"[PY] DqnLearner initialized. grid_shape={grid_shape}, spawner_len={self.spawner_len}, device={self.device}")

    def _to_tensor(self, arr, dtype=torch.float32):
        return torch.as_tensor(arr, dtype=dtype, device=self.device)

    def _action_to_tensor(self, actions):
        a = np.array(actions, dtype=np.float32) / self.node_id_scale
        return self._to_tensor(a, dtype=torch.float32).unsqueeze(-1)

    def _flat_to_grid(self, flat_array):
        """
        Convert flat grid array from Unity (length = HISTORY_LENGTH * H * W * C)
        to (HISTORY_LENGTH * C, H, W) numpy array.
        """
        arr = np.array(flat_array, dtype=np.float32)
        # Reshape to (HISTORY_LENGTH, GRID_SIZE, GRID_SIZE, NUM_CHANNELS)
        hwc = arr.reshape(HISTORY_LENGTH, GRID_SIZE, GRID_SIZE, NUM_CHANNELS)
        # Transpose to (HISTORY_LENGTH, NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
        chw = np.transpose(hwc, (0, 3, 1, 2))
        # Merge first two dimensions
        merged = chw.reshape(HISTORY_LENGTH * NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
        return merged

    def _flat_to_spawner(self, flat_array):
        """Convert flat spawner array to numpy (already flat)."""
        return np.array(flat_array, dtype=np.float32)

    def predict_q(self, grid_flat, spawner_flat, a):
        """
        grid_flat: flat list from Unity (length GRID_FLAT_LEN)
        spawner_flat: flat list from Unity (length SPAWNER_FLAT_LEN)
        a: int action node_id
        returns: float Q-value
        """
        grid_img = self._flat_to_grid(grid_flat)
        spawner_vec = self._flat_to_spawner(spawner_flat)
        grid_t = self._to_tensor(grid_img).unsqueeze(0)          # (1, C', H, W)
        spawner_t = self._to_tensor(spawner_vec).unsqueeze(0)    # (1, T*num_spawners)
        a_t = self._action_to_tensor([a])                         # (1, 1)
        with torch.no_grad():
            q = self.policy_net(grid_t, spawner_t, a_t).item()
        return q

    def observe(self, grid_flat, spawner_flat, a, r, next_grid_flat, next_spawner_flat, done=False):
        grid_img = self._flat_to_grid(grid_flat)
        spawner_vec = self._flat_to_spawner(spawner_flat)
        next_grid_img = self._flat_to_grid(next_grid_flat)
        next_spawner_vec = self._flat_to_spawner(next_spawner_flat)

        self.replay.push(grid_img, spawner_vec, a, r, next_grid_img, next_spawner_vec, done)
        self.known_actions.add(int(a))

        loss_val = None
        if len(self.replay) >= self.warmup:
            loss_val = self._train_step()
        return loss_val

    def _train_step(self):
        batch = self.replay.sample(self.batch_size)

        grid = self._to_tensor(batch["grid_states"])               # (B, C', H, W)
        spawner = self._to_tensor(batch["spawner_states"])         # (B, T*num_spawners)
        actions = batch["actions"]
        rewards = self._to_tensor(batch["rewards"])                # (B,)
        next_grid = self._to_tensor(batch["next_grid_states"])     # (B, C', H, W)
        next_spawner = self._to_tensor(batch["next_spawner_states"]) # (B, T*num_spawners)
        dones = self._to_tensor(batch["dones"])                    # (B,)

        # Compute current Q(s,a)
        actions_t = self._action_to_tensor(actions)                # (B, 1)
        q_values = self.policy_net(grid, spawner, actions_t).squeeze(-1)  # (B,)

        # Compute max_{a'} Q_target(s_next, a') using double DQN
        if len(self.known_actions) == 0:
            max_next_q = torch.zeros_like(rewards)
        else:
            action_list = sorted(self.known_actions)
            A = len(action_list)
            B = grid.shape[0]

            # Prepare all candidate actions normalized
            actions_all = np.array(action_list, dtype=np.float32) / self.node_id_scale
            actions_all_t = self._to_tensor(actions_all)                # (A,)
            actions_all_t = actions_all_t.view(1, A).repeat(B, 1)       # (B, A)
            actions_all_t = actions_all_t.unsqueeze(-1).view(B * A, 1)  # (B*A, 1)

            # Expand next states for each action
            C, H, W = self.grid_shape
            next_grid_rep = next_grid.unsqueeze(1).expand(B, A, C, H, W).reshape(B * A, C, H, W)
            next_spawner_rep = next_spawner.unsqueeze(1).expand(B, A, self.spawner_len).reshape(B * A, self.spawner_len)

            with torch.no_grad():
                # Q values from policy net for all actions (for argmax)
                q_all_policy = self.policy_net(next_grid_rep, next_spawner_rep, actions_all_t).view(B, A)  # (B, A)
                best_idx = q_all_policy.argmax(dim=1)                                 # (B,)

                # Q values from target net for all actions
                q_all_target = self.target_net(next_grid_rep, next_spawner_rep, actions_all_t).view(B, A) # (B, A)
                max_next_q = q_all_target.gather(1, best_idx.unsqueeze(1)).squeeze(1) # (B,)

        targets = rewards + self.gamma * (1.0 - dones) * max_next_q

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        # Soft target update
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

        self.train_step_count += 1
        return loss.item()

    def save(self, step_count):
        path = os.path.join(CHECKPOINT_DIR, f"dqn_step{step_count:07d}.pt")
        torch.save({
            "step": step_count,
            "model_state": self.policy_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "known_actions": list(self.known_actions),
            "grid_shape": self.grid_shape,
            "num_spawners": self.num_spawners,
            "history_length": self.history_length,
        }, path)
        print(f"[PY] Checkpoint saved: {path}")

# ==============================
#  Epsilon schedule
# ==============================
def epsilon_by_episode(episode_idx, eps_start=1.0, eps_min=0.1, decay_episodes=200):
    if episode_idx <= 1:
        return eps_start
    if episode_idx >= decay_episodes:
        return eps_min
    t = (episode_idx - 1) / (decay_episodes - 1)
    return eps_start + t * (eps_min - eps_start)

# ==============================
#  Main server loop
# ==============================
def main():
    step_count = 0
    episode_idx = 1
    episode_step = 0
    episode_return = 0.0
    learner = None

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[PY] Listening on {HOST}:{PORT} ...")

        conn, addr = s.accept()
        print(f"[PY] Connected by {addr}")

        with conn:
            buf = b""
            while True:
                data = conn.recv(4096)
                if not data:
                    print("[PY] Connection closed.")
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line.decode("utf-8-sig"))
                    except Exception as e:
                        print(f"[PY] JSON parse error: {e}")
                        continue

                    msg_type = msg.get("type")

                    # ---------- Action Request ----------
                    if msg_type == "action_request":
                        grid_flat = msg.get("state")               # flattened grid history
                        spawner_flat = msg.get("spawner_rates")    # flattened spawner rates history
                        cand_ids = msg.get("candidate_node_ids", [])
                        epsilon_unity = msg.get("epsilon", 0.1)

                        if grid_flat is None or spawner_flat is None or len(cand_ids) == 0:
                            print("[PY] Invalid action_request")
                            continue

                        # Verify lengths
                        if len(grid_flat) != GRID_FLAT_LEN:
                            print(f"[PY] WARNING: grid length {len(grid_flat)} != expected {GRID_FLAT_LEN}")
                        if len(spawner_flat) != SPAWNER_FLAT_LEN:
                            print(f"[PY] WARNING: spawner length {len(spawner_flat)} != expected {SPAWNER_FLAT_LEN}")

                        if learner is None:
                            learner = DqnLearner(
                                grid_shape=GRID_SHAPE,
                                num_spawners=NUM_SPAWNERS,
                                history_length=HISTORY_LENGTH
                            )

                        cand_ids = [int(x) for x in cand_ids]

                        # Compute Q-values for each candidate
                        q_values = [learner.predict_q(grid_flat, spawner_flat, nid) for nid in cand_ids]

                        # Episode-based epsilon
                        epsilon = epsilon_by_episode(episode_idx)
                        rand_val = np.random.rand()
                        if rand_val < epsilon:
                            idx = np.random.randint(len(cand_ids))
                            is_random = True
                        else:
                            idx = int(np.argmax(q_values))
                            is_random = False

                        chosen_node_id = cand_ids[idx]

                        reply = {
                            "type": "action_reply",
                            "chosen_node_id": chosen_node_id,
                            "candidate_node_ids": cand_ids,
                            "q_values": [float(q) for q in q_values],
                            "epsilon": float(epsilon),
                            "is_random": bool(is_random),
                        }
                        conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))

                        if episode_step == 0:
                            print(f"[PY] === Episode {episode_idx} starts ===")
                        print(f"[PY] action_reply: chosen={chosen_node_id}, random={is_random}, eps={epsilon:.3f}")
                        continue

                    # ---------- Transition ----------
                    if msg_type == "transition":
                        action_id = msg.get("action_id", -1)
                        node_id = msg.get("node_id", -1)
                        reward = msg.get("reward", 0.0)

                        grid_t_flat = msg.get("state_t")
                        spawner_t_flat = msg.get("spawner_rates_t")
                        grid_tp1_flat = msg.get("state_tp1")
                        spawner_tp1_flat = msg.get("spawner_rates_tp1")

                        pl_raw = msg.get("pl_raw_delta", 0.0)
                        qd_raw = msg.get("qd_raw_delta", 0.0)
                        bt_raw = msg.get("bt_raw_delta", 0.0)

                        if (grid_t_flat is None or spawner_t_flat is None or
                            grid_tp1_flat is None or spawner_tp1_flat is None):
                            print("[PY] Invalid transition: missing state")
                            continue

                        step_count += 1
                        episode_step += 1
                        episode_return += reward

                        done = (episode_step >= 30)  # adjust episode length as needed

                        if learner is None:
                            learner = DqnLearner(
                                grid_shape=GRID_SHAPE,
                                num_spawners=NUM_SPAWNERS,
                                history_length=HISTORY_LENGTH
                            )

                        loss = learner.observe(
                            grid_t_flat, spawner_t_flat,
                            node_id, reward,
                            grid_tp1_flat, spawner_tp1_flat,
                            done
                        )

                        # Compute Q estimate for logging
                        q_est = learner.predict_q(grid_t_flat, spawner_t_flat, node_id)

                        wandb.log({
                            "env_step": step_count,
                            "train/reward": reward,
                            "train/q_est": q_est,
                            "train/loss": loss if loss is not None else 0.0,
                            "deltas/pl_raw": pl_raw,
                            "deltas/qd_raw": qd_raw,
                            "deltas/bt_raw": bt_raw,
                            "buffer/size": len(learner.replay) if learner else 0,
                        })

                        # Send q_update back to Unity (optional)
                        q_msg = {
                            "type": "q_update",
                            "node_ids": [node_id],
                            "q_values": [q_est],
                        }
                        conn.sendall((json.dumps(q_msg) + "\n").encode("utf-8"))

                        if done:
                            print(f"[PY] Episode {episode_idx} finished. Return={episode_return:.3f}")
                            wandb.log({
                                "episode": episode_idx,
                                "episodic/return": episode_return,
                                "episodic/length": episode_step,
                            })
                            episode_idx += 1
                            episode_step = 0
                            episode_return = 0.0

                        if step_count % 500 == 0 and learner is not None:
                            learner.save(step_count)

                        continue

                    print(f"[PY] Unknown message type: {msg_type}")

if __name__ == "__main__":
    main()
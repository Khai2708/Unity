import socket
import json
import os
import uuid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import wandb

HOST = "127.0.0.1"
PORT = 50007

# ==============================
# version
# ==============================
VERSION = "EIGHT_CHANNELS_WITH_SPAWNER_V2"

# ==============================
# mode
# ==============================
EVAL_ONLY = False
USE_SPAWNER_INPUT = False  # temporarily disabled — spawner rates are constant in current scenarios
USE_ADJACENCY_INPUT = False  # temporarily disabled — fixed single-topology training makes this redundant for now

# ==============================
# wandb
# ==============================
ENTITY = "dldbgus203"
PROJECT = "IndustryDQN_Factory"

os.environ.pop("WANDB_ENTITY", None)
os.environ.pop("WANDB_PROJECT", None)
os.environ.pop("WANDB_BASE_URL", None)
os.environ["WANDB_RESUME"] = "never"
os.environ["WANDB_MODE"] = "online"

run = wandb.init(
    project=PROJECT,
    name=f"IndustryDQN_{uuid.uuid4().hex[:8]}",
    resume="never",
    id=str(uuid.uuid4()),
    mode="online",
)

wandb.define_metric("env_step")
wandb.define_metric("episode")
wandb.define_metric("trainreward", step_metric="env_step")
wandb.define_metric("train/queue_reward", step_metric="env_step")
wandb.define_metric("train/assembly_reward", step_metric="env_step")
wandb.define_metric("buffer/size", step_metric="env_step")
wandb.define_metric("episodic/*", step_metric="episode")
wandb.define_metric("train/is_fault_hit", step_metric="env_step")
wandb.define_metric("train/hit_ratio_ma100", step_metric="env_step")
wandb.define_metric("episodic/hit_ratio", step_metric="episode")
wandb.define_metric("throughput/step_throughput", step_metric="env_step")
wandb.define_metric("throughput/cumulative_exits", step_metric="env_step")

CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_INTERVAL = 500

EPS_START = 0.9
EPS_MIN = 0.01
EPS_DECAY_EPISODES = 1000

REWARD_CLIP_MIN = -20.0
REWARD_CLIP_MAX = 20.0
STATE_CLIP_MIN = -10.0
STATE_CLIP_MAX = 10.0
NEXTQ_CLIP_MIN = -10.0
NEXTQ_CLIP_MAX = 10.0
TARGET_CLIP_MIN = -10.0
TARGET_CLIP_MAX = 10.0

# ==============================
# state dimensions
# Unity sends 369 numbers:
#   grid history: 3 snapshots * 15 cells * 8 features = 360
#   spawner history: 3 snapshots * 3 spawners = 9
# total dynamic = 369
# Python adds fixed adjacency 225 -> final state = 594
# ==============================
NODE_COUNT = 15
CELL_FEATURES = 6 # features per cell (removed capacity + occupancy — constant in this topology)
SNAPSHOT_GRID_DIM = NODE_COUNT * CELL_FEATURES   # 120
NUM_SPAWNERS = 3
SNAPSHOT_SPAWNER_DIM = NUM_SPAWNERS              # 3
TIME_STEPS = 3                        # t, t-3, t-5

GRID_HISTORY_DIM = SNAPSHOT_GRID_DIM * TIME_STEPS      # 360
SPAWNER_HISTORY_DIM = SNAPSHOT_SPAWNER_DIM * TIME_STEPS # 9
RAW_STATE_DIM = GRID_HISTORY_DIM + SPAWNER_HISTORY_DIM  # 369
ADJ_FEATURE_DIM = NODE_COUNT * NODE_COUNT               # 225
FIXED_STATE_DIM = ADJ_FEATURE_DIM + RAW_STATE_DIM       # 594

# Selectable nodes (unchanged)
SELECTABLE_NODE_IDS = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
NUM_ACTIONS = len(SELECTABLE_NODE_IDS)
EXCLUDED_NODE_IDS = sorted(set(range(18)) - set(SELECTABLE_NODE_IDS))

NODE_ID_TO_ACTION_IDX = {nid: idx for idx, nid in enumerate(SELECTABLE_NODE_IDS)}
ACTION_IDX_TO_NODE_ID = {idx: nid for nid, idx in NODE_ID_TO_ACTION_IDX.items()}

# ==============================
# weighted adjacency (same as before)
# ==============================
def build_weighted_adjacency():
    idx = {n: i for i, n in enumerate(SELECTABLE_NODE_IDS)}
    W = np.zeros((NODE_COUNT, NODE_COUNT), dtype=np.float32)

    # top row
    W[idx[1],  idx[2]]  = 1.0
    W[idx[2],  idx[3]]  = 1.0
    W[idx[3],  idx[4]]  = 1.0
    W[idx[3],  idx[10]] = 1.0
    W[idx[4],  idx[5]]  = 1.0

    # middle row
    W[idx[7],  idx[2]]  = 1.0
    W[idx[7],  idx[8]]  = 1.0
    W[idx[8],  idx[9]]  = 1.0
    W[idx[9],  idx[10]] = 1.0
    W[idx[10], idx[11]] = 1.0

    # bottom row
    W[idx[13], idx[14]] = 1.0
    W[idx[14], idx[15]] = 1.0
    W[idx[14], idx[9]]  = 1.0
    W[idx[15], idx[16]] = 1.0
    W[idx[16], idx[17]] = 1.0

    return W

ADJ_W = build_weighted_adjacency()
ADJ_FLAT = ADJ_W.reshape(-1).astype(np.float32)
ADJ_FLAT = np.clip(ADJ_FLAT, STATE_CLIP_MIN, STATE_CLIP_MAX)

# ==============================
# preprocess
# Input: list of 369 numbers from Unity
# ==============================
def preprocess_state_full_adj_plus_history369(state_list):
    dynamic_s = np.array(state_list, dtype=np.float32)
    dynamic_s = np.nan_to_num(dynamic_s, nan=0.0, posinf=0.0, neginf=0.0)
    dynamic_s = np.clip(dynamic_s, STATE_CLIP_MIN, STATE_CLIP_MAX)

    if dynamic_s.shape[0] != RAW_STATE_DIM:
        raise ValueError(f"raw dynamic state_dim must be {RAW_STATE_DIM}, but got {dynamic_s.shape[0]}")

    final_s = np.concatenate([ADJ_FLAT, dynamic_s], axis=0).astype(np.float32)

    if final_s.shape[0] != FIXED_STATE_DIM:
        raise ValueError(f"final state_dim must be {FIXED_STATE_DIM}, but got {final_s.shape[0]}")

    return final_s

# ==============================
# Replay Buffer (unchanged)
# ==============================
class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_candidates = [None] * capacity
        self.pos = 0
        self.size = 0

    def push(self, s, a_idx, r, s_next, done, next_candidates=None):
        idx = self.pos
        self.states[idx] = s
        self.actions[idx] = int(a_idx)
        self.rewards[idx] = r
        self.next_states[idx] = s_next
        self.dones[idx] = float(done)
        self.next_candidates[idx] = None if next_candidates is None else [int(x) for x in next_candidates]
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "states": self.states[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "next_states": self.next_states[idxs],
            "dones": self.dones[idxs],
            "next_candidates": [self.next_candidates[i] for i in idxs],
        }

    def __len__(self):
        return self.size

# ==============================
# Q Network with 3D CNN + spawner rates
# Input state = [adjacency 225 | dynamic 369]
# dynamic part is split: grid (360) and spawner (9)
# Grid is reshaped to (batch, channels=8, depth=3, height=3, width=5)
# Spawner is processed by a separate MLP
# ==============================
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        if state_dim != FIXED_STATE_DIM:
            raise ValueError(f"expected {FIXED_STATE_DIM}, got {state_dim}")

        self.adj_dim = ADJ_FEATURE_DIM
        self.grid_hist_dim = GRID_HISTORY_DIM      # 360
        self.spawner_hist_dim = SPAWNER_HISTORY_DIM # 9
        self.time_steps = TIME_STEPS               # 3
        self.grid_h = 3
        self.grid_w = 5
        self.input_channels = 6 # 6 features per cell (capacity + occupancy removed)

        # 3D CNN for grid history
        self.conv3d = nn.Sequential(
            nn.Conv3d(self.input_channels, 32, kernel_size=(2, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(2, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
        )
        # Output shape: (batch, 64, 1, 3, 5) -> flat 960
        cnn_flat_dim = 64 * 1 * 3 * 5   # 960
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # MLP for spawner history
        self.spawner_mlp = nn.Sequential(
            nn.Linear(self.spawner_hist_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # Head: concatenate adj, cnn_embed (128), spawner_embed (16)
        head_input_dim = self.adj_dim + 128 + 16   # 225 + 128 + 16 = 369
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, state):
        # state: (batch, FIXED_STATE_DIM)
        if USE_ADJACENCY_INPUT:
            adj = state[:, :self.adj_dim]  # (batch, 225)
        else:
            adj = torch.zeros(state.size(0), self.adj_dim, device=state.device)
        dynamic = state[:, self.adj_dim:]  # (batch, 369)

        # split dynamic into grid (360) and spawner (9)
        grid_hist = dynamic[:, :self.grid_hist_dim]             # (batch, 360)
        spawner_hist = dynamic[:, self.grid_hist_dim:]          # (batch, 9)

        batch = grid_hist.size(0)

        # reshape grid history to (batch, channels, depth, height, width)
        # channels = 8, depth = 3, height = 3, width = 5
        grid_hist = grid_hist.reshape(batch, self.time_steps, self.grid_h, self.grid_w, self.input_channels)
        grid_hist = grid_hist.permute(0, 4, 1, 2, 3)   # (batch, 8, 3, 3, 5)

        # 3D CNN
        cnn_out = self.conv3d(grid_hist)                # (batch, 64, 1, 3, 5)
        cnn_out = cnn_out.reshape(batch, -1)            # (batch, 960)
        cnn_embed = self.cnn_proj(cnn_out)              # (batch, 128)

        # spawner MLP
        if USE_SPAWNER_INPUT:
            spawner_embed = self.spawner_mlp(spawner_hist)  # (batch, 16)
        else:
            spawner_embed = torch.zeros(batch, 16, device=spawner_hist.device)

        # concatenate all
        fused = torch.cat([adj, cnn_embed, spawner_embed], dim=1)   # (batch, 369)
        return self.head(fused)

# ==============================
# DQN Learner (mostly unchanged)
# ==============================
class DqnLearner:
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        gamma: float = 0.995,
        lr: float = 1e-4,
        batch_size: int = 64,
        capacity: int = 300_000,
        warmup: int = 2_000,
        target_update_interval: int = 1_000,
        device: str = None,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup = warmup
        self.target_update_interval = target_update_interval

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(state_dim, num_actions).to(self.device)
        self.target_net = QNetwork(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(capacity, state_dim)

        self.train_step_count = 0
        self.last_loss = None

        self.tau = 0.005
        self.criterion = nn.SmoothL1Loss()

        print(
            f"[PY] DqnLearner init (8 channels + spawner rates, 3D CNN): "
            f"state_dim={state_dim}, num_actions={num_actions}, device={self.device}"
        )
        # Sanity check
        dummy = torch.zeros(1, state_dim, device=self.device)
        with torch.no_grad():
            q_out = self.policy_net(dummy)
        print(f"[PY] Sanity check: input shape {dummy.shape} -> output shape {q_out.shape}")

    def _to_tensor(self, arr, dtype=torch.float32):
        return torch.as_tensor(arr, dtype=dtype, device=self.device)

    def node_id_to_action_idx(self, node_id: int) -> int:
        if int(node_id) not in NODE_ID_TO_ACTION_IDX:
            raise ValueError(f"node_id={node_id} not selectable")
        return NODE_ID_TO_ACTION_IDX[int(node_id)]

    def action_idx_to_node_id(self, action_idx: int) -> int:
        return ACTION_IDX_TO_NODE_ID[int(action_idx)]

    def node_ids_to_action_indices(self, node_ids):
        if node_ids is None:
            return None
        out = []
        for nid in node_ids:
            nid = int(nid)
            if nid in NODE_ID_TO_ACTION_IDX:
                out.append(NODE_ID_TO_ACTION_IDX[nid])
        return out

    def observe(self, s, node_id, r, s_next, done=False, next_candidate_node_ids=None):
        action_idx = self.node_id_to_action_idx(int(node_id))
        next_candidate_action_indices = self.node_ids_to_action_indices(next_candidate_node_ids)
        self.replay.push(s, action_idx, r, s_next, done, next_candidate_action_indices)

        loss_val = None
        if len(self.replay) >= self.warmup:
            loss_val = self._train_step()
        return loss_val, action_idx

    def _sanitize_candidate_action_indices(self, cand_action_indices):
        if cand_action_indices is None or len(cand_action_indices) == 0:
            return list(range(self.num_actions))
        valid = []
        for a in cand_action_indices:
            a = int(a)
            if 0 <= a < self.num_actions:
                valid.append(a)
        return valid if valid else list(range(self.num_actions))

    def _train_step(self):
        batch = self.replay.sample(self.batch_size)
        states = self._to_tensor(batch["states"])
        actions = batch["actions"]
        rewards = self._to_tensor(batch["rewards"])
        next_states = self._to_tensor(batch["next_states"])
        dones = self._to_tensor(batch["dones"])
        next_candidates = batch.get("next_candidates", None)

        actions_t = self._to_tensor(actions, dtype=torch.long).unsqueeze(1)

        q_all = self.policy_net(states)
        q_values = q_all.gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            policy_next_all = self.policy_net(next_states)
            target_next_all = self.target_net(next_states)
            max_next_q = torch.zeros_like(rewards)

            for i in range(next_states.size(0)):
                cand = None if next_candidates is None else next_candidates[i]
                valid_idx = self._sanitize_candidate_action_indices(cand)
                valid_idx_t = torch.tensor(valid_idx, dtype=torch.long, device=self.device)

                q_policy_valid = policy_next_all[i, valid_idx_t]
                best_local_idx = torch.argmax(q_policy_valid)
                best_action_idx = valid_idx_t[best_local_idx]
                max_next_q[i] = target_next_all[i, best_action_idx]

        max_next_q = torch.clamp(max_next_q, NEXTQ_CLIP_MIN, NEXTQ_CLIP_MAX)
        targets = rewards + self.gamma * (1.0 - dones) * max_next_q
        targets = torch.clamp(targets, TARGET_CLIP_MIN, TARGET_CLIP_MAX)

        loss = self.criterion(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_step_count += 1
        self.last_loss = float(loss.item())

        with torch.no_grad():
            tau = self.tau
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.mul_(1.0 - tau).add_(tau * param.data)

            wandb.log({
                "debug/q_abs_max": float(q_values.abs().max().item()),
                "debug/target_abs_max": float(targets.abs().max().item()),
                "debug/reward_abs_max": float(rewards.abs().max().item()),
            })

        return self.last_loss

    def predict_q_all(self, s):
        s_t = self._to_tensor(s).unsqueeze(0)
        with torch.no_grad():
            q_all = self.policy_net(s_t).squeeze(0)
        return q_all.detach().cpu().numpy()

    def predict_q_by_node_id(self, s, node_id) -> float:
        action_idx = self.node_id_to_action_idx(int(node_id))
        q_all = self.predict_q_all(s)
        return float(q_all[action_idx])

    def save(self, step_count: int):
        path = os.path.join(CHECKPOINT_DIR, f"dqn_step{step_count:07d}.pt")
        torch.save(
            {
                "step": step_count,
                "model_state": self.policy_net.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "num_actions": self.num_actions,
                "selectable_node_ids": SELECTABLE_NODE_IDS,
                "node_id_to_action_idx": NODE_ID_TO_ACTION_IDX,
                "action_idx_to_node_id": ACTION_IDX_TO_NODE_ID,
                "adj_w": ADJ_W,
                "adj_flat": ADJ_FLAT,
                "raw_state_dim": RAW_STATE_DIM,
                "adj_feature_dim": ADJ_FEATURE_DIM,
                "fixed_state_dim": FIXED_STATE_DIM,
            },
            path,
        )
        print(f"[PY] checkpoint saved: {path}")

# ==============================
# epsilon schedule (unchanged)
# ==============================
def epsilon_by_episode(episode_idx, eps_start=EPS_START, eps_min=EPS_MIN, decay_episodes=EPS_DECAY_EPISODES):
    if episode_idx <= 1:
        return eps_start
    if episode_idx >= decay_episodes:
        return eps_min
    t = (episode_idx - 1) / float(decay_episodes - 1)
    eps = eps_start + t * (eps_min - eps_start)
    return max(eps_min, float(eps))

# ==============================
# main
# ==============================
def main():
    step_count = 0
    state_dim = FIXED_STATE_DIM
    num_actions = NUM_ACTIONS
    learner = None

    episode_idx = 1
    episode_step = 0
    episode_return = 0.0

    episode_q_sum = 0.0
    episode_q_count = 0
    episode_loss_sum = 0.0
    episode_loss_count = 0
    episode_random_count = 0

    episode_hit_count = 0
    recent_hits = deque(maxlen=100)

    # Throughput tracking variables
    episode_start_time = None
    episode_start_exits = None
    last_step_time = None
    last_step_exits = None
    step_throughputs = []   # store per-step throughput for episode average

    SCENARIOS_PER_EPISODE = 50
    SCENARIO_STEPS = 1
    MAX_STEPS_PER_EPISODE = SCENARIOS_PER_EPISODE * SCENARIO_STEPS

    print(f"[PY] VERSION = {VERSION}")
    print(f"[PY] 8 channels per cell + spawner rates")
    print(f"[PY] RAW_STATE_DIM = {RAW_STATE_DIM} (grid 360 + spawner 9)")
    print(f"[PY] ADJ_FEATURE_DIM = {ADJ_FEATURE_DIM}")
    print(f"[PY] FIXED_STATE_DIM = {FIXED_STATE_DIM}")
    print(f"[PY] NUM_ACTIONS = {NUM_ACTIONS}")
    print(f"[PY] SELECTABLE_NODE_IDS = {SELECTABLE_NODE_IDS}")
    print(f"[PY] EVAL_ONLY = {EVAL_ONLY}")

    latest_epsilon_used = None

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
                        print(f"[PY] JSON parse error: {e}, line={line[:200]}")
                        continue

                    msg_type = msg.get("type")

                    if msg_type == "action_request":
                        state_list = msg.get("state", [])
                        cand_ids_raw = msg.get("candidate_node_ids", [])

                        if len(state_list) == 0 or len(cand_ids_raw) == 0:
                            print("[PY] action_request: empty state or candidate_node_ids")
                            continue

                        try:
                            s_t = preprocess_state_full_adj_plus_history369(state_list)
                        except Exception as e:
                            print(f"[PY] action_request state preprocess error: {e}")
                            continue

                        if learner is None:
                            learner = DqnLearner(
                                state_dim=state_dim,
                                num_actions=num_actions,
                                gamma=0.99,
                                lr=1e-4,
                                batch_size=64,
                                capacity=200_000,
                                warmup=2000,
                                target_update_interval=1_000,
                            )

                        cand_ids = []
                        for x in cand_ids_raw:
                            nid = int(x)
                            if nid in NODE_ID_TO_ACTION_IDX:
                                cand_ids.append(nid)
                            else:
                                print(f"[PY][WARN] invalid candidate node_id ignored: {nid}")

                        if len(cand_ids) == 0:
                            print("[PY][WARN] no valid candidate_node_ids after filtering")
                            continue

                        try:
                            q_all = learner.predict_q_all(s_t)
                            q_values = [float(q_all[NODE_ID_TO_ACTION_IDX[nid]]) for nid in cand_ids]
                        except Exception as e:
                            print(f"[PY] predict_q_all error: {e}")
                            q_values = [0.0 for _ in cand_ids]

                        if EVAL_ONLY:
                            epsilon = 0.0
                        else:
                            epsilon = epsilon_by_episode(
                                episode_idx=episode_idx,
                                eps_start=EPS_START,
                                eps_min=EPS_MIN,
                                decay_episodes=EPS_DECAY_EPISODES,
                            )
                        latest_epsilon_used = epsilon

                        if np.random.rand() < epsilon:
                            idx = np.random.randint(len(cand_ids))
                            is_random = True
                        else:
                            idx = int(np.argmax(q_values))
                            is_random = False

                        if is_random:
                            episode_random_count += 1

                        chosen_node_id = int(cand_ids[idx])
                        chosen_action_idx = NODE_ID_TO_ACTION_IDX[chosen_node_id]

                        reply = {
                            "type": "action_reply",
                            "chosen_node_id": chosen_node_id,
                            "candidate_node_ids": cand_ids,
                            "q_values": [float(q) for q in q_values],
                            "epsilon": float(epsilon),
                            "is_random": bool(is_random),
                            "eval_only": bool(EVAL_ONLY),
                        }

                        try:
                            conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
                            if episode_step == 0:
                                print(f"[PY] === Episode {episode_idx} start === (epsilon={epsilon:.3f})")
                            print(
                                f"[PY] action_reply: chosen_node_id={chosen_node_id}, "
                                f"chosen_action_idx={chosen_action_idx}, eps={epsilon:.3f}, random={is_random}"
                            )
                        except Exception as e:
                            print(f"[PY] action_reply send error: {e}")

                        continue

                    if msg_type == "transition":
                        action_id = msg.get("action_id", -1)
                        node_id = int(msg.get("node_id", -1))
                        reward = float(msg.get("reward", 0.0))
                        is_fault_hit = float(msg.get("is_fault_hit", 0.0))

                        # Extract throughput fields
                        sim_time = msg.get("sim_time_sec", 0.0)
                        total_exits = msg.get("total_exits", 0)

                        reward = float(np.clip(reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX))

                        qd_t = float(msg.get("qd_t", 0.0))
                        ar_t = float(msg.get("ar_t", 0.0))

                        if node_id not in NODE_ID_TO_ACTION_IDX:
                            print(f"[PY][WARN] invalid node_id in transition ignored: {node_id}")
                            continue

                        next_candidate_node_ids_raw = msg.get("next_candidate_node_ids", None)
                        if isinstance(next_candidate_node_ids_raw, list):
                            next_candidate_node_ids = []
                            for x in next_candidate_node_ids_raw:
                                nid = int(x)
                                if nid in NODE_ID_TO_ACTION_IDX:
                                    next_candidate_node_ids.append(nid)
                            if len(next_candidate_node_ids) == 0:
                                next_candidate_node_ids = None
                        else:
                            next_candidate_node_ids = None

                        try:
                            s_t = preprocess_state_full_adj_plus_history369(msg.get("state_t", []))
                            s_tp1 = preprocess_state_full_adj_plus_history369(msg.get("state_tp1", []))
                        except Exception as e:
                            print(f"[PY] transition state preprocess error: {e}")
                            continue

                        if learner is None:
                            learner = DqnLearner(
                                state_dim=state_dim,
                                num_actions=num_actions,
                                gamma=0.99,
                                lr=1e-4,
                                batch_size=64,
                                capacity=200_000,
                                warmup=2000,
                                target_update_interval=1_000,
                            )

                        step_count += 1
                        episode_step += 1
                        episode_return += reward

                        episode_hit_count += int(is_fault_hit > 0.5)
                        recent_hits.append(float(is_fault_hit))

                        # Compute step throughput (except first step of episode)
                        if episode_step == 1:
                            # First step of episode: initialize episode tracking and step throughput zero
                            episode_start_time = sim_time
                            episode_start_exits = total_exits
                            last_step_time = sim_time
                            last_step_exits = total_exits
                            step_throughput = 0.0
                        else:
                            dt = sim_time - last_step_time
                            delta_exits = total_exits - last_step_exits
                            step_throughput = delta_exits / dt if dt > 0 else 0.0
                            last_step_time = sim_time
                            last_step_exits = total_exits

                        step_throughputs.append(step_throughput)

                        #scenario_done = (episode_step % SCENARIO_STEPS == 0)
                        episode_done = (episode_step >= MAX_STEPS_PER_EPISODE)
                        done = episode_done #scenario_done or 

                        if EVAL_ONLY:
                            loss_val = None
                            action_idx = learner.node_id_to_action_idx(node_id)
                        else:
                            loss_val, action_idx = learner.observe(
                                s_t,
                                node_id,
                                reward,
                                s_tp1,
                                done=done,
                                next_candidate_node_ids=next_candidate_node_ids,
                            )

                        try:
                            q_all = learner.predict_q_all(s_t)
                            q_est = float(q_all[action_idx])
                        except Exception as e:
                            print(f"[PY] predict_q error: {e}")
                            q_est = reward

                        episode_q_sum += float(q_est)
                        episode_q_count += 1

                        if loss_val is not None:
                            episode_loss_sum += float(loss_val)
                            episode_loss_count += 1

                        q_msg = {
                            "type": "q_update",
                            "node_ids": [int(node_id)],
                            "q_values": [float(q_est)],
                            "eval_only": bool(EVAL_ONLY),
                        }

                        try:
                            conn.sendall((json.dumps(q_msg) + "\n").encode("utf-8"))
                            eps_display = latest_epsilon_used if latest_epsilon_used is not None else 0.0
                            print(
                                f"[PY] step={step_count} | episode={episode_idx} "
                                f"step={episode_step}/{MAX_STEPS_PER_EPISODE} : "
                                f"action_id={action_id}, node_id={node_id}, action_idx={action_idx}, "
                                f"reward={reward:+.3f}, eps={eps_display:.3f}, eval_only={EVAL_ONLY}, "
                                f"time={sim_time:.2f}, exits={total_exits}"
                            )
                        except Exception as e:
                            print(f"[PY] q_update send error: {e}")

                        if step_count <= 3:
                            print(f"      s_t.shape      = {s_t.shape}")
                            print(f"      s_tp1.shape    = {s_tp1.shape}")
                            print(f"      s_t[:20]       = {s_t[:20]}      # adj head")
                            print(f"      s_t[225:245]   = {s_t[225:245]}  # dynamic head")

                        # Log step metrics
                        wandb.log(
                            {
                                "env_step": step_count,
                                "trainreward": reward,
                                "train/queue_reward": qd_t,
                                "train/assembly_reward": ar_t,
                                "train/is_fault_hit": float(is_fault_hit),
                                "train/hit_ratio_ma100": float(np.mean(recent_hits)) if len(recent_hits) > 0 else 0.0,
                                "buffer/size": len(learner.replay),
                                "throughput/step_throughput": step_throughput,
                                "throughput/cumulative_exits": total_exits,
                            }
                        )

                        if episode_done:
                            # Compute episode throughput metrics
                            episode_duration = sim_time - episode_start_time
                            episode_total_exits = total_exits - episode_start_exits
                            episode_throughput = episode_total_exits / episode_duration if episode_duration > 0 else 0.0
                            avg_step_throughput = sum(step_throughputs) / len(step_throughputs) if step_throughputs else 0.0

                            avg_reward = episode_return / float(max(1, episode_step))
                            avg_q = episode_q_sum / float(max(1, episode_q_count))
                            avg_loss = (episode_loss_sum / float(episode_loss_count)) if episode_loss_count > 0 else 0.0
                            random_rate = episode_random_count / float(MAX_STEPS_PER_EPISODE)
                            hit_ratio = episode_hit_count / float(max(1, episode_step))

                            print(
                                f"[PY] === Episode {episode_idx} done === "
                                f"(len={episode_step}, return={episode_return:+.3f}, avg_reward={avg_reward:+.3f}, "
                                f"avg_q={avg_q:+.3f}, avg_loss={avg_loss:.6f}, random_rate={random_rate:.2f}, "
                                f"hit_ratio={hit_ratio:.3f}, eval_only={EVAL_ONLY}, "
                                f"episode_throughput={episode_throughput:.3f} exits/s, avg_step_throughput={avg_step_throughput:.3f})"
                            )

                            wandb.log(
                                {
                                    "episode": episode_idx,
                                    "episodic/return": float(episode_return),
                                    "episodic/avg_reward": float(avg_reward),
                                    "episodic/length": int(episode_step),
                                    "episodic/avg_q_est": float(avg_q),
                                    "episodic/avg_loss": float(avg_loss),
                                    "episodic/random_rate": float(random_rate),
                                    "episodic/hit_ratio": float(hit_ratio),
                                    "episodic/epsilon_used": float(latest_epsilon_used) if latest_epsilon_used is not None else 0.0,
                                    "episodic/eval_only": float(1.0 if EVAL_ONLY else 0.0),
                                    "buffer/size": len(learner.replay),
                                    "episodic/episode_throughput": episode_throughput,
                                    "episodic/avg_step_throughput": avg_step_throughput,
                                    "episodic/episode_total_exits": episode_total_exits,
                                    "episodic/episode_duration_sec": episode_duration,
                                }
                            )

                            # Reset episode tracking variables
                            episode_idx += 1
                            episode_step = 0
                            episode_return = 0.0
                            episode_q_sum = 0.0
                            episode_q_count = 0
                            episode_loss_sum = 0.0
                            episode_loss_count = 0
                            episode_random_count = 0
                            episode_hit_count = 0
                            step_throughputs = []
                            episode_start_time = None
                            episode_start_exits = None
                            last_step_time = None
                            last_step_exits = None

                        if (not EVAL_ONLY) and (step_count % CHECKPOINT_INTERVAL == 0):
                            learner.save(step_count)

                        continue

                    print(f"[PY] Unknown msg type: {msg_type}")

if __name__ == "__main__":
    main()
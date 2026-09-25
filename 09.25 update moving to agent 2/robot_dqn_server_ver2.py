"""
robot_dqn_server.py
Robot-selection Double-DQN  –  port 50008
Server fixes: validated messages, legal-action masks, terminal handling,
connection-independent progress, checkpoint loading, truthful exploration logging.

Unity compatibility:
- Existing message fields still work. Optional: request_id, is_random, done.
- Echoed request_id needs Unity-side matching to protect against late replies.
- Nonterminal completions with no legal next robot are logged but excluded from
  replay, rather than inventing legal actions or terminal outcomes.
- Fix next-job state construction and completion busy flags in Unity separately.
- Every 50 completions is a reporting window, not automatically a terminal episode.
- Set LOAD_CHECKPOINT and EVAL_ONLY below for evaluation.
- Dependencies: numpy, torch, wandb.
Epsilon is owned by Python (linear decay by episode).
Unity sends epsilon = -1 as a sentinel to defer to Python schedule.
"""

import socket, json, os, uuid, random, collections
import math
import numbers
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# ── Config ────────────────────────────────────────────────────────────────────
HOST       = "127.0.0.1"
PORT       = 50008
VERSION    = "ROBOT_DQN_V3_SERVER_FIXES"
EVAL_ONLY  = False
# Set to a checkpoint filename to resume or evaluate. Required for EVAL_ONLY.
LOAD_CHECKPOINT = ""
WANDB_MODE = os.environ.get("WANDB_MODE", "online")  # online / offline / disabled
MAX_MESSAGE_BYTES = 1_000_000
RUN_ID = uuid.uuid4().hex[:8]
# Nonterminal transitions without a next legal action are NOT trained on.
# They are still logged. Unity must eventually provide a valid decision state.
# Do not treat temporary lack of idle robots as environment termination.

ENTITY  = os.environ.get("WANDB_ENTITY")  # None uses your signed-in account
PROJECT = "IndustryDQN_Factory"

MAX_ROBOTS = 4
STATE_DIM = MAX_ROBOTS * 4 + 2   # distance, battery%, busy, survivability per robot + repair_TU_norm, idle_count_norm
ACTION_DIM = MAX_ROBOTS


HIDDEN      = 128
LR          = 1e-3
GAMMA       = 0.99
BATCH_SIZE  = 32
REPLAY_CAP  = 10_000
WARMUP      = 256
TAU         = 0.005
GRAD_CLIP   = 1.0

# ★ Python owns epsilon — Unity sends -1 as sentinel
EPS_START          = 0.9
EPS_MIN            = 0.01
EPS_DECAY_EPISODES = 500

REWARD_CLIP = 3.0  # new composite reward (move+repair+fleet+failure) can reach ~2.0 in magnitude, added headroom
SCENARIOS_PER_EPISODE = 50

CHECKPOINT_DIR      = "./checkpoints_robot"
CHECKPOINT_INTERVAL = 500

# ── W&B ───────────────────────────────────────────────────────────────────────
def initialize_tracking():
    run = wandb.init(
        project = PROJECT,
        entity  = ENTITY or None,
        name    = f"RobotDQN_{uuid.uuid4().hex[:8]}",
        resume  = "never",
        id      = str(uuid.uuid4()),
        mode    = WANDB_MODE,
        config  = dict(
            version              = VERSION,
            state_dim            = STATE_DIM,
            action_dim           = ACTION_DIM,
            hidden               = HIDDEN,
            lr                   = LR,
            gamma                = GAMMA,
            batch_size           = BATCH_SIZE,
            replay_cap           = REPLAY_CAP,
            warmup               = WARMUP,
            tau                  = TAU,
            eps_start            = EPS_START,
            eps_min              = EPS_MIN,
            eps_decay_episodes   = EPS_DECAY_EPISODES,
            scenarios_per_ep     = SCENARIOS_PER_EPISODE,
        )
    )

    wandb.define_metric("env_step")
    wandb.define_metric("episode")
    wandb.define_metric("train/*",    step_metric="env_step")
    wandb.define_metric("buffer/*",   step_metric="env_step")
    wandb.define_metric("train_update")
    wandb.define_metric("debug/*",    step_metric="train_update")
    wandb.define_metric("episodic/*", step_metric="episode")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ── Replay buffer ─────────────────────────────────────────────────────────────
Transition = collections.namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "idle_mask_tp1", "done"]
)

class ReplayBuffer:
    def __init__(self, cap):
        self.buf = collections.deque(maxlen=cap)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, n):
        return random.sample(list(self.buf), n)
    def __len__(self):
        return len(self.buf)

# ── Network ───────────────────────────────────────────────────────────────────
class RobotQNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN,    HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, ACTION_DIM)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[RobotDQN] device={device}  state_dim={STATE_DIM}  action_dim={ACTION_DIM}")

online_net = RobotQNet().to(device)
target_net = RobotQNet().to(device)
target_net.load_state_dict(online_net.state_dict())
target_net.eval()

optimizer = optim.Adam(online_net.parameters(), lr=LR)
criterion = nn.SmoothL1Loss()
buffer    = ReplayBuffer(REPLAY_CAP)

# ── Epsilon schedule (Python-owned, linear by episode) ────────────────────────
def epsilon_by_episode(ep):
    """
    Linear decay from EPS_START to EPS_MIN over EPS_DECAY_EPISODES episodes.
    Mirrors the fault DQN schedule in double_DQNx8f1_eps_pr.py.
    """
    if ep <= 1:
        return EPS_START
    if ep >= EPS_DECAY_EPISODES:
        return EPS_MIN
    t = (ep - 1) / float(EPS_DECAY_EPISODES - 1)
    return max(EPS_MIN, EPS_START + t * (EPS_MIN - EPS_START))

# ── Masked argmax ─────────────────────────────────────────────────────────────
def masked_argmax(q_np, idle):
    if not idle:
        return -1  # no legal dispatch
    best, best_q = idle[0], q_np[idle[0]]
    for i in idle[1:]:
        if q_np[i] > best_q:
            best_q = q_np[i]
            best   = i
    return best

# ── Training step ─────────────────────────────────────────────────────────────
train_step_count = 0
last_loss        = None

def train_step():
    global train_step_count, last_loss
    if len(buffer) < max(WARMUP, BATCH_SIZE):
        return None

    batch = buffer.sample(BATCH_SIZE)

    states   = torch.tensor(np.array([t.state      for t in batch]), dtype=torch.float32, device=device)
    actions  = torch.tensor(         [t.action      for t in batch],  dtype=torch.long,    device=device)
    rewards  = torch.tensor(         [t.reward      for t in batch],  dtype=torch.float32, device=device)
    n_states = torch.tensor(np.array([t.next_state  for t in batch]), dtype=torch.float32, device=device)

    with torch.no_grad():
        online_q_next = online_net(n_states)
        target_q_next = target_net(n_states)

        next_actions = []
        bootstrap = []
        for i, t in enumerate(batch):
            if t.done:
                next_actions.append(0)  # ignored below for terminal samples
                bootstrap.append(False)
                continue
            if not t.idle_mask_tp1:
                raise ValueError("Nonterminal replay sample has no legal next action")
            mask = torch.full((ACTION_DIM,), float('-inf'), device=device)
            mask[t.idle_mask_tp1] = online_q_next[i, t.idle_mask_tp1]
            next_actions.append(mask.argmax().item())
            bootstrap.append(True)

        next_actions_t = torch.tensor(next_actions, dtype=torch.long, device=device)
        next_q = target_q_next.gather(1, next_actions_t.unsqueeze(1)).squeeze(1)
        bootstrap_t = torch.tensor(bootstrap, dtype=torch.bool, device=device)
        next_q = torch.where(bootstrap_t, next_q, torch.zeros_like(next_q))
        targets = rewards + GAMMA * next_q

    q_all    = online_net(states)
    q_values = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    loss = criterion(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), GRAD_CLIP)
    optimizer.step()

    # Soft target update
    with torch.no_grad():
        for tp, p in zip(target_net.parameters(), online_net.parameters()):
            tp.mul_(1.0 - TAU).add_(TAU * p)

    train_step_count += 1
    last_loss = float(loss.item())

    wandb.log({
        "train_update":         train_step_count,
        "debug/q_abs_max":      float(q_values.abs().max().item()),
        "debug/target_abs_max": float(targets.abs().max().item()),
        "debug/reward_abs_max": float(rewards.abs().max().item()),
    })

    return last_loss

# Training progress belongs to the run, not a TCP connection.
# episode accumulators
episode_idx          = 1
episode_step         = 0
episode_return       = 0.0
episode_q_sum        = 0.0
episode_q_count      = 0
episode_loss_sum     = 0.0
episode_loss_count   = 0
episode_random_count = 0
episode_dist_sum = 0.0
episode_batt_sum = 0.0
episode_move_sum = 0.0
episode_repair_sum = 0.0
episode_fleet_sum = 0.0
episode_move_contrib_sum = 0.0
episode_repair_contrib_sum = 0.0
episode_fleet_contrib_sum = 0.0
episode_failure_penalty_sum = 0.0
episode_success_count = 0
recent_success = collections.deque(maxlen=100)

step_count = 0
episode_random_known = 0
skipped_no_next_action = 0

PROGRESS_FIELDS = ('episode_idx', 'episode_step', 'episode_return', 'episode_q_sum', 'episode_q_count', 'episode_loss_sum', 'episode_loss_count', 'episode_random_count', 'episode_dist_sum', 'episode_batt_sum', 'episode_move_sum', 'episode_repair_sum', 'episode_fleet_sum', 'episode_move_contrib_sum', 'episode_repair_contrib_sum', 'episode_fleet_contrib_sum', 'episode_failure_penalty_sum', 'episode_success_count', 'step_count', 'episode_random_known', 'skipped_no_next_action')

# ── Checkpoint ────────────────────────────────────────────────────────────────
def save_checkpoint(step):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"robot_dqn_{RUN_ID}_step{step:07d}.pt")
    payload = {
        "step": step,
        "model_state": online_net.state_dict(),
        "target_state": target_net.state_dict(),
        "optim_state": optimizer.state_dict(),
        "train_step_count": train_step_count,
        "progress": {name: globals()[name] for name in PROGRESS_FIELDS},
        "recent_success": list(recent_success),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
    }
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)
    print(f"[RobotDQN] checkpoint saved: {path}")


def load_checkpoint(path):
    global train_step_count
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("state_dim", STATE_DIM) != STATE_DIM or checkpoint.get("action_dim", ACTION_DIM) != ACTION_DIM:
        raise ValueError("Checkpoint state/action dimensions do not match configuration")
    online_net.load_state_dict(checkpoint["model_state"])
    target_net.load_state_dict(checkpoint.get("target_state", checkpoint["model_state"]))
    if not EVAL_ONLY:
        if "optim_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optim_state"])
        progress = checkpoint.get("progress")
        if progress is not None:
            for name in PROGRESS_FIELDS:
                if name in progress:
                    globals()[name] = progress[name]
        else:
            # Older checkpoints saved only a step count. Partial episode sums are unknown.
            step = int(checkpoint.get("step", 0))
            globals()["step_count"] = step
            globals()["episode_idx"] = step // SCENARIOS_PER_EPISODE + 1
            print("[RobotDQN] Legacy checkpoint: partial episode statistics start fresh.")
        train_step_count = int(checkpoint.get("train_step_count", 0))
        recent_success.clear()
        recent_success.extend(checkpoint.get("recent_success", []))
    online_net.eval() if EVAL_ONLY else online_net.train()
    target_net.eval()
    print(f"[RobotDQN] Loaded {path}. Replay starts empty; training warmup applies.")


def finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or abs(value) > np.finfo(np.float32).max:
        raise ValueError(f"{name} must be finite and representable as float32")
    return value


def validate_state(value, name):
    if not isinstance(value, list) or len(value) != STATE_DIM:
        raise ValueError(f"{name} must be a flat list of {STATE_DIM} numbers")
    for item in value:
        finite_number(item, name)


def validate_indices(value, name):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list (empty means no available robots)")
    if any(type(i) is not int or not 0 <= i < ACTION_DIM for i in value):
        raise ValueError(f"{name} must contain integer indices 0..{ACTION_DIM - 1}")
    return list(dict.fromkeys(value))


def validate_message(msg):
    if not isinstance(msg, dict):
        raise ValueError("Message must be a JSON object")
    mtype = msg.get("type")
    if mtype not in ("robot_action_request", "robot_transition"):
        raise ValueError(f"Unknown message type: {mtype!r}")
    if "request_id" in msg and (type(msg["request_id"]) not in (int, str) or len(str(msg["request_id"])) > 128):
        raise ValueError("request_id must be an integer or a short string")
    if mtype == "robot_action_request":
        validate_state(msg.get("state"), "state")
        msg["idle_robot_indices"] = validate_indices(msg.get("idle_robot_indices"), "idle_robot_indices")
    else:
        validate_state(msg.get("state_t"), "state_t")
        validate_state(msg.get("state_tp1"), "state_tp1")
        if type(msg.get("robot_index")) is not int or not 0 <= msg["robot_index"] < ACTION_DIM:
            raise ValueError("robot_index is outside the configured action range")
        finite_number(msg.get("reward"), "reward")
        if type(msg.get("done", False)) is not bool:
            raise ValueError("done must be true or false")
        msg["idle_robots_tp1"] = validate_indices(msg.get("idle_robots_tp1"), "idle_robots_tp1")
        if "is_random" in msg and type(msg["is_random"]) is not bool:
            raise ValueError("is_random must be true or false")
        for name in ("travel_dist_norm", "battery_drain_norm", "move_cost_norm", "repair_cost_norm",
                     "fleet_balance_norm", "move_contrib", "repair_contrib", "fleet_contrib",
                     "failure_penalty", "mission_success"):
            if name in msg:
                finite_number(msg[name], name)
        for name in ("module_count", "node_id"):
            if name in msg and type(msg[name]) is not int:
                raise ValueError(f"{name} must be an integer")
    return msg


# ── Per-connection handler ────────────────────────────────────────────────────
def handle_client(conn):
    print("[RobotDQN] client connected")

    global episode_idx, episode_step, episode_return, episode_q_sum, episode_q_count, episode_loss_sum, episode_loss_count, episode_random_count, episode_dist_sum, episode_batt_sum, episode_move_sum, episode_repair_sum, episode_fleet_sum, episode_move_contrib_sum, episode_repair_contrib_sum, episode_fleet_contrib_sum, episode_failure_penalty_sum, episode_success_count, recent_success, step_count, episode_random_known, skipped_no_next_action
    request_metadata = {}  # connection-scoped; never guess task metadata

    buf = b""

    try:
        while True:
            data = conn.recv(4096)
            if not data:
                print("[RobotDQN] connection closed.")
                break
            buf += data
            if len(buf) > MAX_MESSAGE_BYTES and b"\n" not in buf:
                raise ValueError("Incoming message exceeds MAX_MESSAGE_BYTES")

            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue

                try:
                    if len(line) > MAX_MESSAGE_BYTES:
                        raise ValueError("Incoming message exceeds MAX_MESSAGE_BYTES")
                    msg = validate_message(json.loads(line))
                except (ValueError, TypeError, UnicodeDecodeError, OverflowError) as exc:
                    print(f"[RobotDQN] Rejected message: {exc}")
                    # Never fabricate an action for malformed data; Unity may time out.
                    continue

                mtype = msg.get("type", "")

                # ── action request ────────────────────────────────────────────
                if mtype == "robot_action_request":
                    state_raw    = np.array(msg["state"], dtype=np.float32)
                    idle_indices = msg["idle_robot_indices"]

                    # ★ Python owns epsilon — ignore Unity value (it sends -1 as sentinel)
                    # Always use Python's own linear schedule based on episode index
                    eps_in = 0.0 if EVAL_ONLY else epsilon_by_episode(episode_idx)

                    with torch.no_grad():
                        s_t  = torch.tensor(state_raw, device=device).unsqueeze(0)
                        q_np = online_net(s_t).squeeze(0).cpu().numpy()

                    if EVAL_ONLY or not idle_indices:
                        chosen    = masked_argmax(q_np, idle_indices)
                        is_random = False
                    elif random.random() < eps_in:
                        chosen    = random.choice(idle_indices)
                        is_random = True
                    else:
                        chosen    = masked_argmax(q_np, idle_indices)
                        is_random = False

                    request_id = msg.get("request_id")
                    if request_id is not None and chosen >= 0:
                        if len(request_metadata) >= REPLAY_CAP:
                            request_metadata.pop(next(iter(request_metadata)))
                        request_metadata[request_id] = (chosen, is_random)

                    reply = json.dumps({
                        "type":               "robot_action_reply",
                        "chosen_robot_index": chosen,
                        "q_values":           q_np.tolist(),
                        "epsilon":            eps_in,   # send back so Unity can display it
                        "is_random":          is_random,
                        **({"request_id": request_id} if request_id is not None else {}),
                    }) + "\n"
                    conn.sendall(reply.encode("utf-8"))

                    print(
                        f"[RobotDQN] action_reply: robot={chosen} "
                        f"eps={eps_in:.3f} random={is_random} "
                        f"(ep={episode_idx}, python_schedule)"
                    )

                # ── transition ────────────────────────────────────────────────
                elif mtype == "robot_transition":
                    s       = np.array(msg["state_t"],   dtype=np.float32)
                    s1      = np.array(msg["state_tp1"], dtype=np.float32)
                    a       = int(msg["robot_index"])
                    r       = float(np.clip(msg["reward"], -REWARD_CLIP, REWARD_CLIP))
                    idle_t1 = msg["idle_robots_tp1"]
                    done = msg.get("done", False)

                    dist_norm = float(msg.get("travel_dist_norm", 0.0))
                    batt_norm = float(msg.get("battery_drain_norm", 0.0))
                    move_cost_norm = float(msg.get("move_cost_norm", 0.0))
                    repair_cost_norm = float(msg.get("repair_cost_norm", 0.0))
                    fleet_balance_norm = float(msg.get("fleet_balance_norm", 0.0))
                    move_contrib = float(msg.get("move_contrib", 0.0))
                    repair_contrib = float(msg.get("repair_contrib", 0.0))
                    fleet_contrib = float(msg.get("fleet_contrib", 0.0))
                    failure_penalty = float(msg.get("failure_penalty", 0.0))
                    mission_success = float(msg.get("mission_success", 1.0))
                    module_count = int(msg.get("module_count", 1))
                    node_id = int(msg.get("node_id", -1))

                    loss_val = None
                    if not EVAL_ONLY:
                        if done or idle_t1:
                            buffer.push(s, a, r, s1, idle_t1, done)
                            loss_val = train_step()
                        else:
                            skipped_no_next_action += 1
                            if skipped_no_next_action == 1 or skipped_no_next_action % 100 == 0:
                                print(f"[RobotDQN] Skipped {skipped_no_next_action} nonterminal replay samples with no legal next action. Fix Unity decision-state timing; these are still logged.")

                    # Q estimate for the chosen robot action
                    with torch.no_grad():
                        s_t  = torch.tensor(s, device=device).unsqueeze(0)
                        q_np = online_net(s_t).squeeze(0).cpu().numpy()
                    q_est = float(q_np[a]) if 0 <= a < ACTION_DIM else 0.0

                    step_count           += 1
                    episode_step         += 1
                    episode_return       += r
                    episode_q_sum        += q_est
                    episode_q_count      += 1
                    episode_dist_sum += dist_norm
                    episode_batt_sum += batt_norm
                    episode_move_sum += move_cost_norm
                    episode_repair_sum += repair_cost_norm
                    episode_fleet_sum += fleet_balance_norm
                    episode_move_contrib_sum += move_contrib
                    episode_repair_contrib_sum += repair_contrib
                    episode_fleet_contrib_sum += fleet_contrib
                    episode_failure_penalty_sum += failure_penalty
                    episode_success_count += int(mission_success > 0.5)
                    recent_success.append(mission_success)

                    random_flag = msg.get("is_random")
                    metadata = request_metadata.pop(msg.get("request_id"), None)
                    if random_flag is None and metadata is not None and metadata[0] == a:
                        random_flag = metadata[1]
                    if random_flag is not None:
                        episode_random_known += 1
                        episode_random_count += int(random_flag)

                    if loss_val is not None:
                        episode_loss_sum   += loss_val
                        episode_loss_count += 1

                    eps_display = 0.0 if EVAL_ONLY else epsilon_by_episode(episode_idx)
                    loss_str    = f"{loss_val:.5f}" if loss_val is not None else "n/a"

                    print(
                        f"[RobotDQN] step={step_count} ep={episode_idx} "
                        f"ep_step={episode_step}/{SCENARIOS_PER_EPISODE} "
                        f"robot={a} node={node_id} r={r:+.3f} "
                        f"dist={dist_norm:.3f} batt={batt_norm:.3f} "
                        f"q={q_est:+.3f} loss={loss_str} "
                        f"eps={eps_display:.3f}"
                    )

                    # per-step W&B
                    wandb.log({
                        "env_step": step_count,
                        "train/reward": r,
                        "train/travel_dist_norm": dist_norm,
                        "train/battery_drain_norm": batt_norm,
                        "train/move_cost_norm": move_cost_norm,
                        "train/repair_cost_norm": repair_cost_norm,
                        "train/fleet_balance_norm": fleet_balance_norm,
                        "train/move_contrib": move_contrib,
                        "train/repair_contrib": repair_contrib,
                        "train/fleet_contrib": fleet_contrib,
                        "train/failure_penalty": failure_penalty,
                        "train/mission_success": mission_success,
                        "train/success_ratio_ma100": float(np.mean(recent_success)) if len(recent_success) > 0 else 0.0,
                        "train/module_count": module_count,
                        "train/q_estimate": q_est,
                        **({"train/loss": loss_val} if loss_val is not None else {}),
                        "train/skipped_no_next_action": skipped_no_next_action,
                        "train/epsilon": eps_display,
                        "buffer/size": len(buffer),
                    })

                    # ── episode boundary ──────────────────────────────────────
                    if episode_step >= SCENARIOS_PER_EPISODE:
                        avg_reward = episode_return / max(1, episode_step)
                        avg_q = episode_q_sum / max(1, episode_q_count)
                        avg_loss = (episode_loss_sum / episode_loss_count
                                    if episode_loss_count > 0 else 0.0)
                        random_rate = episode_random_count / episode_random_known if episode_random_known else None
                        random_rate_text = f"{random_rate:.2f}" if random_rate is not None else "unknown"
                        avg_dist = episode_dist_sum / max(1, episode_step)
                        avg_batt = episode_batt_sum / max(1, episode_step)
                        avg_move = episode_move_sum / max(1, episode_step)
                        avg_repair = episode_repair_sum / max(1, episode_step)
                        avg_fleet = episode_fleet_sum / max(1, episode_step)
                        avg_move_contrib = episode_move_contrib_sum / max(1, episode_step)
                        avg_repair_contrib = episode_repair_contrib_sum / max(1, episode_step)
                        avg_fleet_contrib = episode_fleet_contrib_sum / max(1, episode_step)
                        avg_failure_penalty = episode_failure_penalty_sum / max(1, episode_step)
                        success_ratio = episode_success_count / max(1, episode_step)

                        print(
                            f"[RobotDQN] === Episode {episode_idx} done === "
                            f"return={episode_return:+.3f} avg_r={avg_reward:+.3f} "
                            f"avg_q={avg_q:+.3f} avg_loss={avg_loss:.6f} "
                            f"random_rate={random_rate_text} "
                            f"avg_dist={avg_dist:.3f} avg_batt={avg_batt:.3f}"
                        )

                        wandb.log({
                            "episode": episode_idx,
                            "episodic/return": float(episode_return),
                            "episodic/avg_reward": float(avg_reward),
                            "episodic/length": int(episode_step),
                            "episodic/avg_q_est": float(avg_q),
                            "episodic/avg_loss": float(avg_loss),
                            **({"episodic/random_rate": float(random_rate)} if random_rate is not None else {}),
                            "episodic/random_metadata_coverage": episode_random_known / max(1, episode_step),
                            "episodic/epsilon_used": float(eps_display),
                            "episodic/avg_travel_dist": float(avg_dist),
                            "episodic/avg_battery_drain": float(avg_batt),
                            "episodic/avg_move_cost": float(avg_move),
                            "episodic/avg_repair_cost": float(avg_repair),
                            "episodic/avg_fleet_balance": float(avg_fleet),
                            "episodic/avg_move_contrib": float(avg_move_contrib),
                            "episodic/avg_repair_contrib": float(avg_repair_contrib),
                            "episodic/avg_fleet_contrib": float(avg_fleet_contrib),
                            "episodic/avg_failure_penalty": float(avg_failure_penalty),
                            "episodic/success_ratio": float(success_ratio),
                            "episodic/eval_only": float(1.0 if EVAL_ONLY else 0.0),
                            "buffer/size": len(buffer),
                        })

                        # reset accumulators
                        episode_idx          += 1
                        episode_step          = 0
                        episode_return        = 0.0
                        episode_q_sum         = 0.0
                        episode_q_count       = 0
                        episode_loss_sum      = 0.0
                        episode_loss_count    = 0
                        episode_random_count = 0
                        episode_random_known = 0
                        episode_dist_sum = 0.0
                        episode_batt_sum = 0.0
                        episode_move_sum = 0.0
                        episode_repair_sum = 0.0
                        episode_fleet_sum = 0.0
                        episode_move_contrib_sum = 0.0
                        episode_repair_contrib_sum = 0.0
                        episode_fleet_contrib_sum = 0.0
                        episode_failure_penalty_sum = 0.0
                        episode_success_count = 0

                    if not EVAL_ONLY and step_count % CHECKPOINT_INTERVAL == 0:
                        save_checkpoint(step_count)

                else:
                    print(f"[RobotDQN] unknown type: {mtype}")

    except Exception as e:
        print(f"[RobotDQN] handler error: {e}")
        traceback.print_exc()
    finally:
        conn.close()
        print("[RobotDQN] client disconnected")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if EVAL_ONLY and not LOAD_CHECKPOINT:
        raise ValueError("Set LOAD_CHECKPOINT before enabling EVAL_ONLY")
    if LOAD_CHECKPOINT:
        load_checkpoint(LOAD_CHECKPOINT)
    initialize_tracking()
    print("[RobotDQN] Unity must send consistent decision-time next states. This Python update cannot reconstruct missing next-job features.")
    print(f"[RobotDQN] VERSION={VERSION}  state_dim={STATE_DIM}  action_dim={ACTION_DIM}")
    print(f"[RobotDQN] EVAL_ONLY={EVAL_ONLY}  device={device}")
    print(f"[RobotDQN] Epsilon: Python-owned linear schedule "
          f"({EPS_START} → {EPS_MIN} over {EPS_DECAY_EPISODES} episodes)")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((HOST, PORT))
            srv.listen(1)
            print(f"[RobotDQN] Listening on {HOST}:{PORT} …")
            while True:
                conn, addr = srv.accept()
                print(f"[RobotDQN] connection from {addr}")
                handle_client(conn)
    except KeyboardInterrupt:
        print("[RobotDQN] Stopping.")
    finally:
        try:
            if not EVAL_ONLY and step_count:
                save_checkpoint(step_count)
        finally:
            wandb.finish()

if __name__ == "__main__":
    main()

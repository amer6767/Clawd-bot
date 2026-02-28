# ============================================================
# TERRITORIAL.IO - BRAIN MODEL TRAINING (DQN)
# ============================================================
# HOW TO USE:
#   1. Go to https://kaggle.com/code → New Notebook
#   2. Delete all existing cells
#   3. Create ONE code cell and paste this ENTIRE script into it
#   4. On the right sidebar: Session options → Accelerator → GPU T4 x2
#   5. Click Run All (▶▶)
#   6. When done (~30-60 min), go to the Output tab on the right
#   7. Download: brain_model.pth
#   8. Put it in your territorial_bot/models/ folder
# ============================================================

# ── Step 1: Install dependencies ──────────────────────────────
import subprocess
subprocess.run(["pip", "install", "-q", "torch", "numpy", "matplotlib", "tqdm"], check=True)

# ── Step 2: Imports ───────────────────────────────────────────
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

OUTPUT_DIR = '/kaggle/working'
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Step 3: Simulated game environment ────────────────────────
class TerritorialEnv:
    """
    Simplified Territorial.io simulation for RL training.
    Grid: 0=neutral, 1=own, -1=enemy
    Actions: 0=STAY, 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW
    """
    GRID_SIZE = 20
    NUM_ENEMIES = 3
    DIRECTIONS = [
        (0, 0), (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1),
    ]

    def __init__(self):
        self.grid = None
        self.own_cells = set()
        self.enemy_cells = [set() for _ in range(self.NUM_ENEMIES)]
        self.step_count = 0
        self.max_steps = 500
        self.reset()

    def reset(self) -> np.ndarray:
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)
        self.step_count = 0
        corners = [
            (1, 1), (1, self.GRID_SIZE-4),
            (self.GRID_SIZE-4, 1), (self.GRID_SIZE-4, self.GRID_SIZE-4)
        ]
        own_corner = random.choice(corners)
        self.own_cells = set()
        for dr in range(3):
            for dc in range(3):
                r, c = own_corner[0]+dr, own_corner[1]+dc
                self.grid[r, c] = 1.0
                self.own_cells.add((r, c))
        remaining = [c for c in corners if c != own_corner]
        self.enemy_cells = [set() for _ in range(self.NUM_ENEMIES)]
        for i, corner in enumerate(remaining[:self.NUM_ENEMIES]):
            for dr in range(2):
                for dc in range(2):
                    r, c = corner[0]+dr, corner[1]+dc
                    self.grid[r, c] = -1.0
                    self.enemy_cells[i].add((r, c))
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.step_count += 1
        prev_own = len(self.own_cells)
        prev_enemy = sum(len(e) for e in self.enemy_cells)

        if action != 0:
            self._expand(action)
        for i in range(self.NUM_ENEMIES):
            if self.enemy_cells[i]:
                self._enemy_expand(i)

        curr_own = len(self.own_cells)
        curr_enemy = sum(len(e) for e in self.enemy_cells)
        total = self.GRID_SIZE * self.GRID_SIZE
        own_pct = curr_own / total

        reward = (curr_own - prev_own) * 0.5 + (prev_enemy - curr_enemy) * 0.3 - 0.01
        done = False
        if curr_own == 0:
            reward = -10.0; done = True
        elif own_pct > 0.6:
            reward = +20.0; done = True
        elif all(len(e) == 0 for e in self.enemy_cells):
            reward = +15.0; done = True
        elif self.step_count >= self.max_steps:
            done = True

        return self._get_state(), reward, done, {'own_pct': own_pct}

    def _expand(self, action: int):
        dr, dc = self.DIRECTIONS[action]
        new_cells = []
        for r, c in list(self.own_cells):
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE:
                if self.grid[nr, nc] != 1.0:
                    new_cells.append((nr, nc))
        expand_count = max(1, len(new_cells) // 3)
        for r, c in random.sample(new_cells, min(expand_count, len(new_cells))):
            old = self.grid[r, c]
            self.grid[r, c] = 1.0
            self.own_cells.add((r, c))
            if old == -1.0:
                for es in self.enemy_cells:
                    es.discard((r, c))

    def _enemy_expand(self, idx: int):
        if not self.enemy_cells[idx] or not self.own_cells:
            return
        ec = np.mean(list(self.enemy_cells[idx]), axis=0)
        oc = np.mean(list(self.own_cells), axis=0)
        diff = oc - ec
        dr, dc = int(np.sign(diff[0])), int(np.sign(diff[1]))
        for r, c in list(self.enemy_cells[idx]):
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE:
                if self.grid[nr, nc] != -1.0:
                    old = self.grid[nr, nc]
                    self.grid[nr, nc] = -1.0
                    self.enemy_cells[idx].add((nr, nc))
                    if old == 1.0:
                        self.own_cells.discard((nr, nc))
                    break

    def _get_state(self) -> np.ndarray:
        total = self.GRID_SIZE * self.GRID_SIZE
        own = len(self.own_cells)
        enemy = sum(len(e) for e in self.enemy_cells)
        neutral = total - own - enemy
        state = np.zeros(64, dtype=np.float32)
        state[0] = own / total
        state[1] = enemy / total
        state[2] = neutral / total
        state[3] = own / total
        state[4] = enemy / total
        state[5] = neutral / total
        borders = sum(1 for r,c in self.own_cells
                      for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                      if (r+dr, c+dc) not in self.own_cells)
        state[6] = borders / 100.0
        own_pct = own / total
        if own_pct < 0.05:   state[8] = 1.0
        elif own_pct < 0.25: state[9] = 1.0
        else:                state[10] = 1.0
        sr = np.linspace(0, self.GRID_SIZE-1, 5, dtype=int)
        sc = np.linspace(0, self.GRID_SIZE-1, 10, dtype=int)
        idx = 14
        for r in sr:
            for c in sc:
                if idx < 64:
                    state[idx] = self.grid[r, c]
                    idx += 1
        return state

# Test environment
env = TerritorialEnv()
s = env.reset()
print(f"✅ Environment ready. State shape: {s.shape}")

# ── Step 4: DQN network (must match brain_system.py exactly) ──
class DQNNetwork(nn.Module):
    def __init__(self, state_size=64, action_size=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_size),
        )
    def forward(self, x):
        return self.net(x)

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    def push(self, *args):
        self.memory.append(args)
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

# Hyperparameters
STATE_SIZE    = 64
ACTION_SIZE   = 9
GAMMA         = 0.95
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.995
LR            = 0.001
BATCH_SIZE    = 64
MEMORY_SIZE   = 10000
TARGET_UPDATE = 10
EPISODES      = 1000
MAX_STEPS     = 500

policy_net = DQNNetwork(STATE_SIZE, ACTION_SIZE).to(device)
target_net = DQNNetwork(STATE_SIZE, ACTION_SIZE).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory    = ReplayMemory(MEMORY_SIZE)
criterion = nn.MSELoss()

print(f"✅ DQN ready. Parameters: {sum(p.numel() for p in policy_net.parameters()):,}")

# ── Step 5: Training functions ────────────────────────────────
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, ACTION_SIZE - 1)
    with torch.no_grad():
        t = torch.FloatTensor(state).unsqueeze(0).to(device)
        return policy_net(t).argmax(dim=1).item()

def train_step() -> Optional[float]:
    if len(memory) < BATCH_SIZE:
        return None
    batch = memory.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    s_t  = torch.FloatTensor(np.array(states)).to(device)
    a_t  = torch.LongTensor(actions).to(device)
    r_t  = torch.FloatTensor(rewards).to(device)
    ns_t = torch.FloatTensor(np.array(next_states)).to(device)
    d_t  = torch.BoolTensor(dones).to(device)
    q    = policy_net(s_t).gather(1, a_t.unsqueeze(1))
    with torch.no_grad():
        na   = policy_net(ns_t).argmax(dim=1)
        nq   = target_net(ns_t).gather(1, na.unsqueeze(1)).squeeze(1)
        nq[d_t] = 0.0
        tq   = r_t + GAMMA * nq
    loss = criterion(q.squeeze(1), tq)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    return loss.item()

# ── Step 6: Main training loop ────────────────────────────────
epsilon = EPSILON_START
ep_rewards, ep_own_pcts, ep_lengths = [], [], []
best_avg = float('-inf')

print(f"\n🚀 Training DQN for {EPISODES} episodes...")
for ep in tqdm(range(1, EPISODES + 1)):
    state = env.reset()
    ep_reward = 0.0
    final_own_pct = 0.0

    for step in range(MAX_STEPS):
        action = select_action(state, epsilon)
        next_state, reward, done, info = env.step(action)
        memory.push(state, action, reward, next_state, done)
        train_step()
        state = next_state
        ep_reward += reward
        final_own_pct = info['own_pct']
        if done:
            break

    ep_rewards.append(ep_reward)
    ep_own_pcts.append(final_own_pct)
    ep_lengths.append(step + 1)

    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Update target network
    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Save best model
    if ep >= 50:
        avg_reward = np.mean(ep_rewards[-50:])
        if avg_reward > best_avg:
            best_avg = avg_reward
            torch.save({
                'policy_net': policy_net.state_dict(),
                'target_net': target_net.state_dict(),
                'epsilon': epsilon,
                'episode': ep,
            }, f'{OUTPUT_DIR}/brain_model.pth')

    # Log progress
    if ep % 100 == 0:
        avg_r   = np.mean(ep_rewards[-100:])
        avg_pct = np.mean(ep_own_pcts[-100:])
        print(f"  Ep {ep:4d}/{EPISODES} | "
              f"Avg Reward: {avg_r:7.2f} | "
              f"Avg Territory: {avg_pct:.1%} | "
              f"ε: {epsilon:.3f} | "
              f"Best avg: {best_avg:.2f}")

print(f"\n✅ Training complete! Best avg reward: {best_avg:.2f}")

# ── Step 7: Plot training curves ──────────────────────────────
window = 50
def moving_avg(data, w):
    return [np.mean(data[max(0,i-w):i+1]) for i in range(len(data))]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(ep_rewards, alpha=0.3, color='blue')
axes[0].plot(moving_avg(ep_rewards, window), color='blue', linewidth=2)
axes[0].set_title('Episode Reward'); axes[0].set_xlabel('Episode')

axes[1].plot(ep_own_pcts, alpha=0.3, color='green')
axes[1].plot(moving_avg(ep_own_pcts, window), color='green', linewidth=2)
axes[1].set_title('Final Territory %'); axes[1].set_xlabel('Episode')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

axes[2].plot(ep_lengths, alpha=0.3, color='orange')
axes[2].plot(moving_avg(ep_lengths, window), color='orange', linewidth=2)
axes[2].set_title('Episode Length'); axes[2].set_xlabel('Episode')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/training_curves.png', dpi=100)
plt.show()

# ── Step 8: Final summary ─────────────────────────────────────
print("\n" + "="*50)
print("✅ FILES READY TO DOWNLOAD (check Output tab):")
print(f"   📁 {OUTPUT_DIR}/brain_model.pth      ← PUT IN territorial_bot/models/")
print(f"   📁 {OUTPUT_DIR}/training_curves.png  ← Training progress chart")
print("="*50)
print(f"Final epsilon: {epsilon:.3f}")
print(f"Best avg reward (last 50 eps): {best_avg:.2f}")
print(f"Final avg territory: {np.mean(ep_own_pcts[-100:]):.1%}")

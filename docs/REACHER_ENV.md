# Reacher Environment Deep Analysis

**Date:** 2026-04-12  
**Purpose:** Baseline analysis for counterfactual dynamics evaluation

---

## 1. Environment Overview

The Reacher environment is a **two-link planar robotic arm** task implemented in MuJoCo via dm_control. The objective is to move the fingertip (end-effector) to a randomly generated target position in 2D space.

**Key characteristics:**
- **State space:** 2 joint angles (shoulder, wrist) + 2D target position + velocities
- **Action space:** 2D continuous torques for shoulder and wrist joints
- **Episode length:** 20 seconds (default time limit) = 1000 steps @ 0.02s timestep
- **Observation:** `position` (qpos), `to_target` (fingertip→target vector), `velocity` (qvel)

---

## 2. Initialization Logic Audit

### 2.1 Joint Initialization

**Source:** `dm_control/suite/reacher.py:92` → `randomizers.randomize_limited_and_rotational_joints()`

**Randomization rules:**
- **Shoulder joint:** Unbounded hinge → sampled uniformly in `[-π, π]` radians
- **Wrist joint:** Limited hinge with range `[-160°, 160°]` → sampled uniformly in `[-2.793, 2.793]` radians

**Default initial pose:** None specified; always randomized at episode start.

### 2.2 Target Position Initialization

**Source:** `dm_control/suite/reacher.py:94-98`

```python
angle = self.random.uniform(0, 2 * np.pi)
radius = self.random.uniform(.05, .20)
physics.named.model.geom_pos['target', 'x'] = radius * np.sin(angle)
physics.named.model.geom_pos['target', 'y'] = radius * np.cos(angle)
```

**Target sampling distribution:**
- **Polar coordinates:** angle ∈ [0, 2π], radius ∈ [0.05, 0.20] meters
- **Cartesian bounds:** x, y ∈ [-0.20, 0.20] (annular region, not uniform in Cartesian space)
- **Excluded region:** Inner circle with radius < 0.05m (too close to base)

### 2.3 Reachable Workspace

**Arm geometry (from `reacher.xml`):**
- **Upper arm length:** 0.12m (shoulder to wrist)
- **Forearm length:** 0.12m (wrist to fingertip)
- **Total reach:** 0.24m (fully extended)
- **Minimum reach:** 0m (fully retracted, fingertip at base)

**Workspace coverage:**
- Training targets span radius [0.05, 0.20]m → **83% of reachable area**
- Targets beyond 0.20m are **never seen during training** (out-of-distribution)
- Targets within 0.05m are **never seen during training** (too close)

---

## 3. Success Criteria Analysis

### 3.1 Standard Tasks (easy/hard)

**Reward function:** `dm_control/suite/reacher.py:110-112`

```python
radii = physics.named.model.geom_size[['target', 'finger'], 0].sum()
return rewards.tolerance(physics.finger_to_target_dist(), (0, radii))
```

**Reward structure:**
- **Type:** Dense reward (continuous feedback)
- **Distance metric:** Euclidean distance between fingertip and target centers
- **Tolerance bounds:** `(0, radii)` where `radii = target_size + finger_size`
  - `finger_size = 0.01m` (fixed, from XML)
  - `target_size = 0.05m` (easy) or `0.015m` (hard)
- **Reward = 1.0** when distance ≤ radii (inside tolerance)
- **Reward → 0** as distance increases beyond radii (Gaussian sigmoid decay)

**Success thresholds:**
- **Easy task:** fingertip within 0.06m of target center (target_size=0.05 + finger=0.01)
- **Hard task:** fingertip within 0.025m of target center (target_size=0.015 + finger=0.01)

**No explicit termination:** Episodes run for full 20s time limit; no early stopping on success.

### 3.2 QPosMatch Task (used in eval)

**Source:** `stable_worldmodel/envs/dmcontrol/custom_tasks/reacher.py`

**Termination condition:**
```python
diff = np.abs(physics.data.qpos - self.target_qpos)
if np.all(diff < self.qpos_threshold):
    return 0.0  # success
```

**Success criteria:**
- **Metric:** Per-joint absolute error in radians
- **Threshold:** 0.05 radians (~2.86°) per joint (default)
- **Early termination:** Episode ends when both joints match target within threshold
- **Target qpos:** Set externally via `set_target_qpos()` (from dataset goal states)

**Evaluation protocol (from `config/eval/reacher.yaml`):**
- Replay expert trajectories from dataset
- Set initial state via `set_state(qpos, qvel)`
- Set goal state via `set_target_qpos(goal_qpos)`
- Plan for 50 steps with horizon=5, receding_horizon=5
- Success = reaching goal joint configuration within 0.05 rad tolerance

---

## 4. Causal Factor Extraction

### 4.1 Fixed (Invariant) Factors

**Physical parameters (from `reacher.xml`):**
- **Timestep:** 0.02s (50 Hz control)
- **Joint damping:** 0.01 (both joints)
- **Motor gear ratio:** 0.05 (both actuators)
- **Control limits:** [-1, 1] (normalized torques)
- **Arm segment lengths:** 0.12m (upper), 0.12m (forearm)
- **Finger radius:** 0.01m
- **Joint friction:** Not explicitly set (MuJoCo defaults)
- **Gravity:** Disabled (2D planar task, no gravity effects)
- **Contact dynamics:** Disabled (`<flag contact="disable"/>`)

**Geometric constraints:**
- **Wrist joint limits:** [-160°, 160°] (prevents self-collision)
- **Shoulder joint:** Unbounded (full rotation)
- **Arena bounds:** 0.3m × 0.3m square workspace

**Visual properties (modifiable via variation_space):**
- Agent color, arm/finger density, floor color, target color/shape, lighting
- These affect **visual observations only**, not dynamics

### 4.2 Randomized (Episode-Varying) Factors

**Per-episode randomization:**
1. **Initial joint angles:**
   - Shoulder: uniform in [-π, π]
   - Wrist: uniform in [-160°, 160°]
   - Initial velocities: **not randomized** (start at rest)

2. **Target position:**
   - Angle: uniform in [0, 2π]
   - Radius: uniform in [0.05, 0.20]m

3. **Target size (task-dependent):**
   - Easy: 0.05m
   - Hard: 0.015m
   - QPosMatch: 0.015m (but success measured in joint space)

**Not randomized:**
- Arm lengths, masses, inertias
- Joint damping, friction coefficients
- Motor characteristics
- Gravity (always disabled)
- External disturbances

### 4.3 Counterfactual Intervention Points

**Potential causal factors for evaluation:**

1. **Dynamics interventions:**
   - Arm density (mass distribution): `variation_space['agent']['arm_density']` ∈ [500, 1500]
   - Finger density: `variation_space['agent']['finger_density']` ∈ [500, 1500]
   - Joint damping (requires XML modification)
   - Motor gear ratio (requires XML modification)

2. **Kinematic interventions:**
   - Arm segment lengths (requires XML modification)
   - Joint limits (wrist range)

3. **Task interventions:**
   - Target position (out-of-distribution radii)
   - Initial joint configuration (extreme poses)
   - Goal tolerance thresholds

4. **Visual interventions (should NOT affect dynamics):**
   - Agent/target colors
   - Target shape (box vs sphere)
   - Lighting intensity
   - Floor texture

---

## 5. Training Dataset Characteristics

**Dataset configuration:** `config/train/data/dmc.yaml`
- **Name:** `reacher.h5`
- **Frameskip:** 5 (actions repeated for 5 steps = 0.1s)
- **Keys:** `pixels`, `action`, `observation` (qpos, to_target, qvel)
- **Sequence length:** `num_preds + history_size` (typically 16 + 1 = 17 frames)

**Expected data distribution:**
- Joint angles: shoulder ∈ [-π, π], wrist ∈ [-2.793, 2.793]
- Target positions: annular region with radius ∈ [0.05, 0.20]m
- Actions: normalized torques ∈ [-1, 1] (via `action_scale.Wrapper`)

**Out-of-distribution scenarios for counterfactual eval:**
- Targets at radius > 0.20m (beyond training distribution)
- Targets at radius < 0.05m (inside training distribution)
- Modified arm masses (density ≠ 1000 kg/m³)
- Modified joint damping (≠ 0.01)

---

## 6. Evaluation Configuration

**From `config/eval/reacher.yaml`:**
- **Environment:** `swm/ReacherDMControl-v0`
- **Task:** `qpos_match` (joint-space goal reaching)
- **Solver:** CEM (Cross-Entropy Method)
- **Planning horizon:** 5 steps
- **Receding horizon:** 5 steps (replan every 5 steps)
- **Action block:** 5 (frameskip)
- **Evaluation budget:** 50 steps
- **Number of episodes:** 50
- **Batch size:** 10

**Evaluation protocol:**
1. Load expert trajectory from dataset
2. Set initial state: `set_state(qpos, qvel)`
3. Set goal state: `set_target_qpos(goal_qpos)` (25 steps ahead in trajectory)
4. Plan actions using world model cost function
5. Execute actions in environment
6. Success = reaching goal qpos within 0.05 rad tolerance

---

## 7. Key Insights for Counterfactual Evaluation

### 7.1 Recommended Intervention Axes

**High-impact dynamics changes:**
1. **Arm mass scaling:** Test density ∈ [500, 1500] (±50% from default 1000)
   - Affects inertia, required torques, trajectory dynamics
   - Should expose model's understanding of mass-dependent dynamics

2. **Joint damping scaling:** Modify XML damping ∈ [0.005, 0.02] (±50%)
   - Affects velocity decay, oscillation behavior
   - Tests model's implicit friction/damping modeling

3. **Target position OOD:** Test radius ∈ [0.20, 0.24] (near workspace boundary)
   - Requires full arm extension, different joint configurations
   - Exposes generalization to unseen workspace regions

**Visual-only changes (negative control):**
- Agent/target color changes should NOT affect planning performance
- If performance degrades, indicates over-reliance on visual features vs. dynamics

### 7.2 Evaluation Metrics

**Closed-loop metrics:**
- Success rate (% episodes reaching goal within tolerance)
- Final joint error (L2 distance in joint space)
- Planning efficiency (steps to goal)

**Open-loop metrics:**
- Prediction error (MSE between predicted and actual next states)
- Trajectory divergence over rollout horizon

**Causal metrics:**
- Performance delta under intervention vs. baseline
- Correlation between intervention magnitude and performance drop
- Visual-only intervention should show ~0 performance change

---

## 8. Implementation Checklist

- [ ] Extract reacher dataset statistics (joint angle distributions, target positions)
- [ ] Implement mass/density intervention wrapper
- [ ] Implement joint damping intervention (XML modification)
- [ ] Implement OOD target position sampling
- [ ] Implement visual-only intervention (color/shape changes)
- [ ] Create evaluation script with intervention conditions
- [ ] Define success metrics and logging
- [ ] Run baseline evaluation (no interventions)
- [ ] Run counterfactual evaluations (each intervention type)
- [ ] Analyze results: performance vs. intervention magnitude

---

## References

- **MuJoCo XML:** `/root/le-wm/.venv/lib/python3.10/site-packages/dm_control/suite/reacher.xml`
- **Task implementation:** `dm_control/suite/reacher.py`
- **Wrapper:** `stable_worldmodel/envs/dmcontrol/reacher.py`
- **Custom task:** `stable_worldmodel/envs/dmcontrol/custom_tasks/reacher.py`
- **Eval config:** `config/eval/reacher.yaml`
- **Train config:** `config/train/data/dmc.yaml`

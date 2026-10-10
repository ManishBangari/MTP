# Simulation Assumptions and Conventions

This document explicitly logs all design decisions, conventions, and unspecified details adopted in this reproduction framework.

---

### Assumption 1: Initial Bid Generation
- **Assumption:** At $t = 1$, bids are drawn i.i.d. per agent and per run from a Gaussian distribution:
  $$z_i(1) \sim \mathcal{N}(\mu, \sigma^2), \quad z_i(1) \leftarrow \text{clip}(z_i(1), \epsilon, c)$$
  with default parameters $\mu = 90.0$ and $\sigma = 10.0$.
- **Reason:** Paper specifies bids are drawn near the equilibrium ($z^* \approx 90$ for $n=10, \gamma=0$) within $[\epsilon, c]$.
- **How to change:** Configurable in `configs/default.yaml` under `simulation.mu` and `simulation.sigma`.

---

### Assumption 2: Run-Averaging Policy
- **Assumption:**
  - **Table I, Figure 3, Figure 4:** Averaged over 10 independent runs (seeds 0 through 9).
  - **Figure 5, Figure 7:** Single run using seed 0 to preserve detailed trajectory dynamics without smoothing out oscillations.
  - **Figure 6:** Single seeded run (seed 0) by default, with configuration option to run a 10-run average (`num_runs_avg = 10`). Both versions are generated.
- **Reason:** The paper states Table I uses multi-run averages while dynamic trace figures (like Fig 5 and 7) display single-sample paths showing transient oscillations.
- **How to change:** Configurable in `configs/default.yaml` under `experiments.<experiment_name>.seeds` and `experiments.<experiment_name>.num_runs`.

---

### Assumption 3: Representative Agent Selection
- **Assumption:** In heterogeneous setups (Figs 4, 5, 6, 7), the representative agent chosen for each algorithm group is the lowest-index agent in that group (e.g. Agent 0 for Group 1, Agent $k$ for Group 2).
- **Reason:** Provides deterministic, reproducible reporting of individual agent behavior.
- **How to change:** Configurable in `kelly/experiments.py` or script options.

---

### Assumption 4: Regularization Parameter in RRM
- **Assumption:** $\lambda_i = 1.0$ in quadratic regularized dual averaging (RMQ).
- **Reason:** Standard Euclidean regularization where $\nabla R^*(y) = y$; matches DAQ when step size $\eta$ is constant.
- **How to change:** Configurable in `kelly/algorithms.py`.

---

### Assumption 5: Payoff Normalization Bounds
- **Assumption:** Payoffs are normalized as:
  $$\bar{\phi}_i(z) = \frac{\phi_i(z) - \phi_{\min}}{\phi_{\max} - \phi_{\min}}$$
  where $\phi_{\max}$ is derived analytically when opponents bid $\epsilon$ and agent $i$ plays its best response $BR_i(s_{\min})$; $\phi_{\min}$ is derived when opponents bid $c$ and agent $i$ bids at boundary $\epsilon$.
- **Reason:** Guarantees $\bar{\phi} \in [0, 1]$ across all valid profiles and matches the paper's sanity check of $\bar{\phi}_{\text{NE}} \approx 0.655$ for $n=10, \gamma=0$.
- **How to change:** Implemented in `KellyGame._compute_payoff_bounds()` in `kelly/game.py`.

---

### Assumption 6: Heterogeneity Indexing Convention
- **Assumption:** Valuations are defined as $a_i = \max(a - i \cdot \gamma, 1)$ for $i \in \{1, \dots, n\}$ (1-indexed mathematics). In zero-indexed Python code, agent $k \in \{0, \dots, n-1\}$ receives valuation $a_{k+1} = \max(a - (k+1)\gamma, 1)$.
- **Reason:** Direct fidelity to Section 1 of the specification.
- **How to change:** Defined in `KellyGame.__init__`.

---

### Assumption 7: Learning Rate Modes (`g_bound_mode`)
- **Assumption:** By default, simulation experiments for reproducing published tables and figures use `g_bound_mode = "empirical"` ($G_i = 1.0$), which faithfully matches the authors' numerical values. A `"remark1"` mode ($G_i = a_i/\epsilon + 1$) is also provided to evaluate the theoretical bounds from Remark 1.
- **Reason:** Reconciles the theoretical bounds with the numerical artifacts published in the paper.
- **How to change:** Configurable via `--mode empirical|remark1|both` in CLI or in `configs/default.yaml`.

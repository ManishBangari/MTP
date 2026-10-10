# Deviations and Known Paper Issues

This document records all deviations from the paper *"Learning in Proportional Allocation Auction Games"* (Ben Mazziane, Mboulou Moutoubi, Altman, De Pellegrini, 2026), including known typos and mathematical inconsistencies in the original manuscript.

---

### Deviation 1: Marginal Utility / Gradient Equation Typo (Eq. 13)
- **Paper Statement:** Eq. (13) states $d_i \phi_i$ with an inconsistent algebraic sign / form.
- **Implemented Correction:** The exact derivative of $\phi_i(z) = a_i \ln\left(\frac{z_i}{\sum_j z_j + \delta}\right) - z_i$ with respect to $z_i$ is:
  $$d_i \phi_i(z) = a_i \left(\frac{1}{z_i} - \frac{1}{S(z)}\right) - 1$$
  where $S(z) = \sum_j z_j + \delta$.
- **Verification:** Verified to machine precision using central finite differences in `test_1_finite_difference_gradient_and_hessian`.

---

### Deviation 2: Indexing Inconsistency in Regularized Dual Averaging (Eqs. 17–18)
- **Paper Statement:** Equations (17) and (18) index $y_i(t)$ and $z_i(t)$ inconsistently between step updates.
- **Implemented Convention:** Standard online dual averaging convention:
  $$y_i(0) = 0$$
  $$y_i(t) = y_i(t-1) + \eta_t g_t^i$$
  $$z_i(t+1) = \text{clip}(\lambda_i y_i(t), \epsilon, c)$$
  with $\lambda_i = 1.0$.
- **Verification:** Under constant step-size $\eta$ and $\lambda_i = 1$, RMQ produces trajectories identical to DAQ, verified in unit tests.

---

### Deviation 3: Step-Size Decay Exponent in OGD_V vs. Theorem 3
- **Paper Statement:** Theorem 3 requires a step-size decay schedule $\eta_t \propto t^{-\beta}$ with $\beta > 1/2$ to guarantee almost sure convergence in pseudo-monotone games.
- **Simulation Specification:** The paper's simulation setup uses $\eta_t = \frac{D_i}{G_i \sqrt{t}}$ ($\beta = 1/2$).
- **Impact:** With $\beta = 1/2$, the theoretical guarantee of Theorem 3 is not strictly met, but this matches the empirical implementation in the paper.

---

### Deviation 4: Dual Step-Size Choices in the Paper (Simulations vs. Theorem 4(2))
- **Simulation Choice:** $\eta_i = \frac{D_i}{G_i \sqrt{T}}$ with $D_i = c - \epsilon$ and $G_i = a_i/\epsilon + 1$.
- **Theorem 4(2) Choice:** $\eta_i = \frac{\epsilon c_i}{a_i \sqrt{T}}$ for regret / gap bounds.
- **Implementation Strategy:** The main simulation runs use the empirical formula $D_i/(G_i \sqrt{T})$ to match Table I and Figures 3–7, while Theorem 4(2) bound verification is evaluated as a separate mathematical check in the test suite.

---

### Deviation 5: Contraction Condition in Theorem 5 Violated by Experimental Parameters
- **Paper Statement:** Theorem 5 provides a sufficient condition for linear convergence of Best Response dynamics.
- **Parameter Value:** Bound (35) requires $\epsilon$ to be sufficiently large. With $\epsilon = 1.0$, $c = 400.0$, $a = 100.0$, and $\delta = 0.1$, the sufficient condition is violated.
- **Reporting:** As instructed, this violation is explicitly computed, tested, and documented in the verification suite and experimental reports.

---

### Deviation 6: Citation Slip in Section IV-A
- **Paper Statement:** Section IV-A references "Theorem 4" when discussing BR contraction properties, which are established in Theorem 5.
- **Correction:** Referenced correctly as Theorem 5 in this codebase.

---

### Deviation 7: Gradient Bound Divisor in Numerical Code vs. Remark 1
- **Paper Statement:** Remark 1 and eq. (16) state $G_i = \frac{a_i}{\epsilon_i} + 1 = 101$, and Section IV states $\eta^{(i)} = \frac{D_i}{G_i \sqrt{T}}$.
- **Empirical Discovery:** With $G_i = 101$, the step size is $\eta \approx 0.072$, under which gradient updates are too slow to reach $10^{-5}$ within $T = 3000$. The published Table I iterations (e.g. 240 for $\text{OGD_F}$, 311 for $\text{DAQ_F}$, 2291 for $\text{RRM_V}$) and Figure 3 trajectories are closely reproduced when $G_i = 1.0$ (effective step size $\eta = \frac{D_i}{\sqrt{T}} \approx 7.28$).
- **Handling:** We provide both modes (`mode="empirical"` and `mode="remark1"`), generate artifacts for both, and document the discrepancy transparently.

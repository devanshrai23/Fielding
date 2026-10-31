# Research Report: Enhancing Clustered Federated Learning via Model-Impact and Temporal Drift-State Awareness

## 1. Motivation and Problem Statement
The original FIELDING algorithm utilizes Clustered Federated Learning (CFL) to mitigate data drift by dynamically re-clustering clients based on their label distributions. However, empirical observations revealed that the baseline algorithm suffers from **"cluster thrashing"**—it is overly sensitive to minor data shifts and tears down clusters too frequently. This constant disruption prevents the neural networks from settling and converging, artificially capping the baseline ResNet-18 accuracy on the FMoW dataset at ~38-39%.

To resolve this, we introduced a **Two-Gate Cascade Architecture** that acts as a regulatory brake on the clustering mechanism. A re-clustering event is now only permitted if:
1. **Gate 1:** The data drift causes a mathematically significant degradation in the model's actual performance.
2. **Gate 2:** The underlying drift state of the client is changing at a historically anomalous rate.

---

## 2. Gate 1: Model-Impact-Aware Reclustering & Smoothing

### 2.1. Base Model Impact
Data drift does not necessarily correlate with model degradation. We modified the aggregator to calculate the direct impact of the drift on the client's local Top-1 accuracy before allowing a cluster teardown.

**Mathematical Definition:**
For a given client, let $A_t$ be the accuracy at the current evaluation. The raw model impact $I_{raw}$ is:
$$ I_{raw} = A_{t-1} - A_t $$

If $I_{raw} \ge \tau$ (where $\tau$ is the `model_impact_threshold`, e.g., 0.05), the impact is deemed significant.

### 2.2. Model-Impact Smoothing (Noise Reduction)
To prevent the system from reacting to transient noise (normal epoch-to-epoch accuracy jitter), we implemented a causal rolling-window smoothing mechanism.

**Mathematical Definition:**
Given a smoothing window $W$ (default $W=3$) and an accuracy history $A$, we divide the recent history into two contiguous windows:
*   **Previous Window:** $A_{prev} = [A_{t-2W+1}, \dots, A_{t-W}]$
*   **Recent Window:** $A_{recent} = [A_{t-W+1}, \dots, A_t]$

The smoothed model impact $I_{smooth}$ is calculated as:
$$ I_{smooth} = \max\left(0, \frac{1}{W} \sum A_{prev} - \frac{1}{W} \sum A_{recent}\right) $$

**Fallback Logic:** If a client has insufficient history ($< 2W$ observations), the system dynamically falls back to $I_{raw}$. 
*Result:* This optimization alone allowed models to train stably without disruption, yielding a massive performance jump from **~38% to ~62%**.

---

## 3. Gate 2: Temporal Drift-State Epsilon ($\epsilon$)

To further optimize the architecture, we introduced a temporal analysis layer. Even if the model performance drops, we only re-cluster if the client's fundamental "drift state" is changing at an anomalous velocity compared to its own history.

### 3.1. The Drift State Vector ($s_t$)
For each drifted client at evaluation point $t$, we construct a 6-dimensional state vector:
$$ s_t = [D_L, D_X, D_C, M, V, P] $$

*   **$D_L$ (Label Drift):** The Jensen-Shannon distance between the client's previous and current label probability distributions. $D_L \in [0, 1]$.
*   **$D_X, D_C$ (Feature & Concept Drift):** Explicitly set to $0.0$ to maintain scientific integrity, as FIELDING's baseline FMoW implementation does not continuously extract features/gradients during the drift-detection phase.
*   **$M$ (Drift Magnitude):** $M_t = |D_L| + |D_X| + |D_C|$. Bounded $\in [0, 1]$.
*   **$V$ (Drift Velocity):** The rate of change of magnitude: $V_t = M_t - M_{t-1}$.
*   **$P$ (Persistence):** The ratio of recent evaluation rounds (defined by `persistence_window`) in which this specific client experienced drift. $P_t \in [0, 1]$.

### 3.2. Normalization
To ensure valid geometric distance calculations, all vector components are normalized to the $[0, 1]$ scale. Because $V_t \in [-1, 1]$, it is normalized mathematically:
$$ V_{norm} = \frac{V_t + 1.0}{2.0} $$
Resulting normalized state vector:
$$ \hat{s}_t = [D_L, 0, 0, M_t, V_{norm}, P_t] $$

### 3.3. Temporal Distance ($\Delta$) and Dynamic Threshold ($\epsilon$)
We calculate the $L_1$ distance (sum of absolute differences) between consecutive normalized state vectors to measure how violently the client's data profile is shifting:
$$ \Delta_t = ||\hat{s}_t - \hat{s}_{t-1}||_1 $$

Instead of a hardcoded threshold, the system learns a dynamic, per-client upper boundary ($\epsilon_t$) based on a rolling history of the last $N$ evaluations (default $N=50$):
$$ \epsilon_t = \text{Percentile}_{95}(\Delta_{t-N}, \dots, \Delta_{t-1}) $$
*(Note: $\Delta_t$ is strictly excluded from the calculation of $\epsilon_t$ to prevent data leakage).*

### 3.4. The Final Cascade Decision Rule
Re-clustering is only permitted if the event survives both gates:
$$ \text{Decision} = \begin{cases} 
\text{RECLUSTER} & \text{if } (I_{smooth} \ge \tau) \land (\Delta_t > \epsilon_t) \\
\text{SKIP} & \text{otherwise}
\end{cases} $$

*(During the initial warm-up period where a client has $<50$ history points, Gate 2 is bypassed, and the decision relies solely on Gate 1).*

---

## 4. Newly Introduced Configuration Parameters

To ensure reproducibility and modularity, the optimizations were integrated as optional parameters in the YAML configuration:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `model_impact_aware_reclustering` | `False` | Enables Gate 1 (Model impact check). |
| `model_impact_threshold` | `0.05` | $\tau$: Accuracy drop required to trigger Gate 1. |
| `model_impact_smoothing` | `False` | Enables rolling window averaging for Gate 1. |
| `model_impact_smoothing_window` | `3` | $W$: Size of the causal smoothing window. |
| `temporal_epsilon_reclustering` | `False` | Enables Gate 2 (Temporal state vector check). |
| `temporal_epsilon_window` | `50` | $N$: History size for calculating $\epsilon_t$. |
| `epsilon_percentile` | `95` | The percentile used to define anomalous $\Delta$. |
| `persistence_window` | `5` | Window for calculating the persistence ratio $P_t$. |

---

## 5. Experimental Observations & Next Steps
Initial runs demonstrate that **Gate 1 (Smoothing)** successfully stabilizes the clusters, eliminating thrashing and raising global convergence significantly. 

The activation of **Gate 2 (Temporal Epsilon)** successfully tracks dynamic P95 boundaries (averaging $\epsilon \approx 0.74$). Initial observations show it induces higher volatility, acting as an aggressive filter. Current experiments are focused on tuning the temporal hyperparameters (`epsilon_percentile = 99`, `persistence_window = 10`) to find the optimal equilibrium between stable smoothing and dynamic adaptability.

# FDNA Monte Carlo Simulation Documentation

This document explains the arguments, classes, and workflow of the FDNA Monte Carlo simulation.

---

## 1️⃣ FDNAState

Represents the system state.

| Argument | Type                | Description                                                 |
| -------- | ------------------- | ----------------------------------------------------------- |
| `SE`     | `Dict[Node, float]` | Self-Effectiveness of each node, range 0–100.               |
| `SOD`    | `Dict[Edge, float]` | Strength of Dependency for each edge (i→j), range 0–1.      |
| `COD`    | `Dict[Edge, float]` | Criticality of Dependency for each edge (i→j), range 0–100. |
| `nodes`  | `List[Node]`        | List of system nodes.                                       |
| `edges`  | `List[Edge]`        | List of edges representing dependencies.                    |

---

## 2️⃣ FDNARunResult

Result of the operability computation.

| Argument  | Type                | Description                                                   |
| --------- | ------------------- | ------------------------------------------------------------- |
| `O`       | `Dict[Node, float]` | Operability of each node.                                     |
| `score`   | `float`             | Global system score (sum of operabilities or via `perf_map`). |
| `details` | `Dict[str,float]`   | Optional extra metrics.                                       |

---

## 3️⃣ ResilienceAction

Elementary action to restore resilience.

| Argument | Type   | Description                                                                                |
| -------- | ------ | ------------------------------------------------------------------------------------------ |
| `kind`   | `str`  | Type: `'add_edge'`, `'remove_edge'`, `'scale_sod'`, `'set_cod'`, `'boost_se'`, `'cap_se'`. |
| `params` | `dict` | Parameters specific to the action (e.g., `{i:'N1', j:'N5', sod:0.5, cod:60}`).             |

---

## 4️⃣ ResiliencePolicy

Generates resilience actions based on attacked state.

| Argument | Type                                                  | Description                                  |
| -------- | ----------------------------------------------------- | -------------------------------------------- |
| `name`   | `str`                                                 | Name of the policy.                          |
| `rules`  | `List[Callable[[FDNAState], List[ResilienceAction]]]` | Rules returning actions from attacked state. |

**Method:**

* `actions(attacked_state)` → returns a list of actions to apply.

---

## 5️⃣ AttackModel

Simulates system perturbation.

| Argument | Type                                              | Description                                                 |
| -------- | ------------------------------------------------- | ----------------------------------------------------------- |
| `name`   | `str`                                             | Attack model name.                                          |
| `fn`     | `Callable[[FDNAState, random.Random], FDNAState]` | Function taking nominal state and returning attacked state. |

Examples: `attack_k_nodes`, `attack_remove_edges`.

---

## 6️⃣ MonteCarloConfig

Monte Carlo simulation parameters.

| Argument   | Type                                            | Description                                                                              |
| ---------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `n_runs`   | `int`                                           | Number of Monte Carlo runs. Each run generates a new attacked state and computes scores. |
| `seed`     | `Optional[int]`                                 | Random seed for reproducibility.                                                         |
| `perf_map` | `Optional[Callable[[Dict[Node,float]], float]]` | Function mapping node operabilities to global system score.                              |

---

## 7️⃣ MonteCarloOutcome

Global result of a Monte Carlo simulation.

| Argument           | Type                     | Description                                                       |
| ------------------ | ------------------------ | ----------------------------------------------------------------- |
| `baseline_scores`  | `List[float]`            | System scores without resilience.                                 |
| `resilient_scores` | `List[float]`            | System scores after resilience.                                   |
| `baseline_O`       | `List[Dict[Node,float]]` | Node operabilities per run (baseline).                            |
| `resilient_O`      | `List[Dict[Node,float]]` | Node operabilities per run (resilient).                           |
| `summary`          | `Dict[str,float]`        | Summary: mean baseline/resilient scores, absolute/relative gains. |

---

## 8️⃣ compute\_operability

Arguments:

| Argument   | Type                                            | Description                                                             |
| ---------- | ----------------------------------------------- | ----------------------------------------------------------------------- |
| `state`    | `FDNAState`                                     | State to evaluate (attacked or repaired).                               |
| `perf_map` | `Optional[Callable[[Dict[Node,float]], float]]` | Function computing global score from node operabilities.                |
| `order`    | `Optional[List[Node]]`                          | Topological order for computation. If `None`, calculated automatically. |

---

## 9️⃣ Utility Functions

* `apply_actions(state, actions)` → apply `ResilienceAction` list on a `FDNAState`.
* `attack_k_nodes(k, se_down, weighted)` → disable `k` nodes.
* `attack_remove_edges(p_remove)` → randomly remove edges.

---

## 🔄 Workflow Summary

```
FDNAState --> AttackModel --> (Attacked FDNAState)
     |                               |
     |                               +--> compute_operability --> baseline score
     |
     +--> ResiliencePolicy --> ResilienceAction --> apply_actions --> compute_operability --> resilient score
```

* Each Monte Carlo run: generates a random attacked state, applies policy, computes scores.
* `n_runs` controls how many iterations are performed.

---

This document can be included in your GitHub repository as `FDNA_MonteCarlo.md`.

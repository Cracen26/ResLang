from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable, Optional
import random
import copy
import matplotlib.pyplot as plt

Node = str
Edge = Tuple[Node, Node]


# =========================================================
# 1. Core FDNA State
# =========================================================
class FDNAState:
    def __init__(self, SE: Dict[Node, float], SOD: Dict[Edge, float],
                 COD: Dict[Edge, float], nodes: List[Node], edges: List[Edge]):
        self.SE = SE
        self.SOD = SOD
        self.COD = COD
        self.nodes = nodes
        self.edges = edges

    def copy(self) -> FDNAState:
        return copy.deepcopy(self)

    @staticmethod
    def example_state() -> FDNAState:
        nodes = ['N1', 'N2', 'N3', 'N4', 'N5']
        SE = {n: 100.0 for n in nodes}
        edges = [
            ('N1', 'N2'), ('N1', 'N3'), ('N1', 'N4'),
            ('N2', 'N5'), ('N3', 'N5'), ('N4', 'N5')
        ]
        SOD = {
            ('N1','N2'): 0.75, ('N1','N3'): 0.85, ('N1','N4'): 0.45,
            ('N2','N5'): 0.45, ('N3','N5'): 0.70, ('N4','N5'): 0.45,
        }
        COD = {
            ('N1','N2'): 20.0, ('N1','N3'): 30.0, ('N1','N4'): 55.0,
            ('N2','N5'): 35.0, ('N3','N5'): 30.0, ('N4','N5'): 55.0,
        }
        return FDNAState(SE, SOD, COD, nodes, edges)


# =========================================================
# 2. FDNA Computation Engine
# =========================================================
class FDNAEngine:
    @staticmethod
    def compute_operability(
        state: FDNAState,
        perf_map: Optional[Callable[[Dict[Node, float]], float]] = None,
        order: Optional[List[Node]] = None
    ) -> Tuple[Dict[Node, float], float]:

        def predecessors(j: Node) -> List[Node]:
            return [i for (i, jj) in state.edges if jj == j]

        def roots() -> List[Node]:
            children = {j for (_, j) in state.edges}
            return [n for n in state.nodes if n not in children]

        # Determine node order (topological-like)
        if order is None:
            pending = set(state.nodes)
            order = []
            frontier = roots() or list(state.nodes)
            while frontier:
                order.extend(frontier)
                pending -= set(frontier)
                next_frontier = [n for n in list(pending)
                                 if all(p in order for p in predecessors(n))]
                if not next_frontier and pending:
                    next_frontier = list(pending)
                frontier = next_frontier

        O = {}
        for j in order:
            preds = predecessors(j)
            if not preds:
                O[j] = float(state.SE[j])
                continue
            sod_terms, cod_terms = [], []
            for i in preds:
                o_i = O.get(i, state.SE[i])
                sod_ij = state.SOD[(i, j)]
                cod_ij = state.COD[(i, j)]
                sod_terms.append(sod_ij * o_i + (1.0 - sod_ij) * state.SE[j])
                cod_terms.append(cod_ij + (100.0 - cod_ij) * (o_i / 100.0))
            sod_oj = sum(sod_terms) / len(sod_terms)
            cod_oj = min(cod_terms) if cod_terms else 100.0
            O[j] = min(sod_oj, cod_oj)

        score = perf_map(O) if perf_map else sum(O.values())
        return O, score


# =========================================================
# 3. Attack Model
# =========================================================
class AttackModel:
    def __init__(self, name: str, fn: Callable[[FDNAState, random.Random], FDNAState]):
        self.name = name
        self.fn = fn

    @staticmethod
    def attack_k_nodes(k: int, se_down: float = 0.0) -> AttackModel:
        def _fn(state: FDNAState, rng: random.Random) -> FDNAState:
            s = state.copy()
            rng.shuffle(s.nodes)
            for n in s.nodes[:k]:
                s.SE[n] = se_down
            return s
        return AttackModel(f"attack_{k}_nodes", _fn)


# =========================================================
# 4. Resilience Policy
# =========================================================
@dataclass
class ResilienceAction:
    kind: str
    params: dict


class ResiliencePolicy:
    def __init__(self, name: str, rules: List[Callable[[FDNAState], List[ResilienceAction]]]):
        self.name = name
        self.rules = rules

    def actions(self, state: FDNAState) -> List[ResilienceAction]:
        actions = []
        for rule in self.rules:
            actions.extend(rule(state))
        return actions

    @staticmethod
    def aggressive() -> ResiliencePolicy:
        def scale_sod_rule(state: FDNAState) -> List[ResilienceAction]:
            return [ResilienceAction('scale_sod', {'factor': 0.7})]

        def backup_edge_rule(state: FDNAState) -> List[ResilienceAction]:
            acts = []
            for n in ['N2','N3','N4']:
                if state.SE.get(n, 100) < 50:
                    acts.append(ResilienceAction('add_edge', {'i':'N1','j':'N5','sod':0.5,'cod':60}))
            return acts

        return ResiliencePolicy('Aggressive', [scale_sod_rule, backup_edge_rule])

    @staticmethod
    def apply_actions(state: FDNAState, actions: List[ResilienceAction]) -> FDNAState:
        s = state.copy()
        for a in actions:
            if a.kind == 'scale_sod':
                factor = a.params.get('factor', 1.0)
                for e in list(s.SOD.keys()):
                    s.SOD[e] = max(0.0, min(1.0, s.SOD[e] * factor))
            elif a.kind == 'add_edge':
                i, j = a.params['i'], a.params['j']
                sod = a.params.get('sod', 0.5)
                cod = a.params.get('cod', 100.0)
                if (i, j) not in s.edges:
                    s.edges.append((i, j))
                s.SOD[(i, j)] = sod
                s.COD[(i, j)] = cod
        return s


# =========================================================
# 5. Monte Carlo Simulation
# =========================================================
class MonteCarloSimulator:
    def __init__(self, base_state: FDNAState, attack: AttackModel,
                 policy: Optional[ResiliencePolicy], n_runs: int = 1000, seed: int = 42,
                 perf_map: Optional[Callable[[Dict[Node, float]], float]] = None):
        self.base_state = base_state
        self.attack = attack
        self.policy = policy
        self.n_runs = n_runs
        self.rng = random.Random(seed)
        self.perf_map = perf_map

        self.baseline_scores = []
        self.resilient_scores = []
        self.baseline_O = []
        self.resilient_O = []

    def run(self):
        for _ in range(self.n_runs):
            attacked = self.attack.fn(self.base_state, self.rng)
            O_base, score_base = FDNAEngine.compute_operability(attacked, self.perf_map)
            self.baseline_scores.append(score_base)
            self.baseline_O.append(O_base)

            if self.policy:
                actions = self.policy.actions(attacked)
                repaired = self.policy.apply_actions(attacked, actions)
                O_res, score_res = FDNAEngine.compute_operability(repaired, self.perf_map)
            else:
                O_res, score_res = O_base, score_base

            self.resilient_scores.append(score_res)
            self.resilient_O.append(O_res)
        return self

    def plot_histogram(self):
        plt.figure(figsize=(10, 5))
        plt.hist(self.baseline_scores, bins=30, alpha=0.6, label='Baseline', color='skyblue')
        plt.hist(self.resilient_scores, bins=30, alpha=0.6, label='Résilience', color='lightgreen')
        plt.xlabel('Score système')
        plt.ylabel('Fréquence')
        plt.title('Histogramme des scores Monte Carlo')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_boxplot(self):
        nodes = self.base_state.nodes
        n_nodes = len(nodes)

        baseline_O_matrix = [[O[n] for O in self.baseline_O] for n in nodes]
        resilient_O_matrix = [[O[n] for O in self.resilient_O] for n in nodes]

        plt.figure(figsize=(12, 6))
        positions_base = [i - 0.2 for i in range(n_nodes)]
        positions_res = [i + 0.2 for i in range(n_nodes)]

        b1 = plt.boxplot(
            baseline_O_matrix, positions=positions_base, widths=0.35,
            patch_artist=True, boxprops=dict(facecolor='skyblue')
        )
        b2 = plt.boxplot(
            resilient_O_matrix, positions=positions_res, widths=0.35,
            patch_artist=True, boxprops=dict(facecolor='lightgreen')
        )

        plt.xticks(range(n_nodes), nodes)
        plt.ylabel('Opérabilité par nœud')
        plt.title('Boxplot : baseline vs résilience')
        plt.grid(True)
        plt.legend([b1["boxes"][0], b2["boxes"][0]], ['Baseline','Résilience'])
        plt.show()

    def plot_linear_evolution(self):
        mean_baseline = []
        mean_resilience = []
        for i in range(1, self.n_runs+1):
            mean_baseline.append(sum(self.baseline_scores[:i])/i)
            mean_resilience.append(sum(self.resilient_scores[:i])/i)

        plt.figure(figsize=(10, 5))
        plt.plot(mean_baseline, label='Baseline', color='blue')
        plt.plot(mean_resilience, label='Résilience', color='green')
        plt.xlabel('Itérations Monte Carlo')
        plt.ylabel('Score moyen')
        plt.title('Évolution du score moyen')
        plt.legend()
        plt.grid(True)
        plt.show()


# =========================================================
# 6. Example Usage
# =========================================================
if __name__ == "__main__":
    base = FDNAState.example_state()
    attack = AttackModel.attack_k_nodes(k=2, se_down=0.0)
    policy = ResiliencePolicy.aggressive()

    sim = MonteCarloSimulator(base, attack, policy, n_runs=1000, seed=42,
                              perf_map=lambda O: sum(O.values()))
    sim.run()
    sim.plot_histogram()
    sim.plot_boxplot()
    sim.plot_linear_evolution()

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Callable, Optional, Iterable
import math
import random
import copy
import matplotlib.pyplot as plt


# -------------------------------
# Types de base
# -------------------------------
Node = str
# Edge key is (i, j) directed from i -> j
Edge = Tuple[Node, Node]

@dataclass
class FDNAState:
    # Self-Effectiveness par noeud [0,100]
    SE: Dict[Node, float]
    # Strength of Dependency par arête (i->j) dans [0,1]
    SOD: Dict[Edge, float]
    # Criticality of Dependency par arête (i->j) dans [0,100]
    COD: Dict[Edge, float]
    # Ensemble des noeuds et arêtes (cohérents avec SE/SOD/COD)
    nodes: List[Node]
    edges: List[Edge]

    def copy(self) -> "FDNAState":
        return copy.deepcopy(self)

@dataclass
class FDNARunResult:
    O: Dict[Node, float]               # operabilité par noeud
    score: float                       # score système (après mapping)
    details: Dict[str, float] = field(default_factory=dict)  # extra metrics

# -------------------------------
# Calcul FDNA
# -------------------------------

def _predecessors(state: FDNAState, j: Node) -> List[Node]:
    return [i for (i, jj) in state.edges if jj == j]


def _roots(state: FDNAState) -> List[Node]:
    children = {j for (_, j) in state.edges}
    return [n for n in state.nodes if n not in children]


def compute_operability(
    state: FDNAState,
    perf_map: Optional[Callable[[Dict[Node, float]], float]] = None,
    order: Optional[List[Node]] = None,
) -> FDNARunResult:
    SE = state.SE
    SOD = state.SOD
    COD = state.COD

    # Ordonnancement topologique approximatif (BFS par couches)
    if order is None:
        pending = set(state.nodes)
        order = []
        # Start with roots
        frontier = _roots(state)
        if not frontier:  # graphe cyclique ou tout le monde a un parent
            frontier = list(state.nodes)
        while frontier:
            order.extend(frontier)
            pending -= set(frontier)
            # next frontier: successeurs dont tous les parents sont déjà vus
            next_frontier = []
            for n in list(pending):
                preds = _predecessors(state, n)
                if all(p in order for p in preds):
                    next_frontier.append(n)
            if not next_frontier and pending:
                # cycle → vider le reste dans un ordre arbitraire
                next_frontier = list(pending)
            frontier = next_frontier

    O: Dict[Node, float] = {}

    for j in order:
        preds = _predecessors(state, j)
        if not preds:
            O[j] = float(SE[j])
            continue
        sod_terms = []
        cod_terms = []
        for i in preds:
            o_i = O.get(i, SE[i])
            sod_ij = SOD[(i, j)]
            cod_ij = COD[(i, j)]
            # SOD contribution (linear mix)
            sod_oji = sod_ij * o_i + (1.0 - sod_ij) * SE[j]
            sod_terms.append(sod_oji)
            # COD contribution (cap that depends on O_i, linear from COD_ij@Oi=0 to 100@Oi=100)
            cod_oji = cod_ij + (100.0 - cod_ij) * (o_i / 100.0)
            cod_terms.append(cod_oji)
        sod_oj = sum(sod_terms) / len(sod_terms)
        cod_oj = min(cod_terms) if cod_terms else 100.0
        O[j] = min(sod_oj, cod_oj)

    score = perf_map(O) if perf_map else sum(O.values())
    return FDNARunResult(O=O, score=score)

# -------------------------------
# Mécanismes de résilience (opérateur R)
# -------------------------------

@dataclass
class ResilienceAction:
    """Action atomique qui modifie l'état FDNA."""
    kind: str  # 'add_edge', 'remove_edge', 'scale_sod', 'set_cod', 'boost_se', 'cap_se'
    params: dict


@dataclass
class ResiliencePolicy:
    """Politique qui génère des actions en fonction de l'état attaqué.
       On reçoit un état *déjà affecté par l'attaque* et on renvoie une liste d'actions.
    """
    name: str
    rules: List[Callable[[FDNAState], List[ResilienceAction]]]

    def actions(self, attacked_state: FDNAState) -> List[ResilienceAction]:
        all_actions: List[ResilienceAction] = []
        for r in self.rules:
            try:
                all_actions.extend(r(attacked_state))
            except Exception:
                pass
        return all_actions


def apply_actions(state: FDNAState, actions: Iterable[ResilienceAction]) -> FDNAState:
    s = state.copy()
    for a in actions:
        if a.kind == 'add_edge':
            i, j = a.params['i'], a.params['j']
            sod, cod = a.params.get('sod', 0.5), a.params.get('cod', 100.0)
            if (i, j) not in s.edges:
                s.edges.append((i, j))
            s.SOD[(i, j)] = float(max(0.0, min(1.0, sod)))
            s.COD[(i, j)] = float(max(0.0, min(100.0, cod)))
        elif a.kind == 'remove_edge':
            i, j = a.params['i'], a.params['j']
            if (i, j) in s.edges:
                s.edges.remove((i, j))
            s.SOD.pop((i, j), None)
            s.COD.pop((i, j), None)
        elif a.kind == 'scale_sod':
            factor = float(a.params.get('factor', 1.0))
            for e in list(s.SOD.keys()):
                s.SOD[e] = max(0.0, min(1.0, s.SOD[e] * factor))
        elif a.kind == 'set_cod':
            target = a.params.get('edges', list(s.edges))
            value = float(a.params.get('value', 100.0))
            for e in target:
                s.COD[e] = max(0.0, min(100.0, value))
        elif a.kind == 'boost_se':
            targets: List[Node] = a.params.get('nodes', s.nodes)
            delta = float(a.params.get('delta', 0.0))
            cap = float(a.params.get('cap', 100.0))
            for n in targets:
                s.SE[n] = min(cap, s.SE[n] + delta)
        elif a.kind == 'cap_se':
            cap = float(a.params.get('cap', 100.0))
            for n in s.nodes:
                s.SE[n] = min(cap, s.SE[n])
        else:
            # Action inconnue: ignorer
            pass
    return s

# -------------------------------
# Modèles d'attaque (perturbations)
# -------------------------------

@dataclass
class AttackModel:
    """Génère un état *attaqué* à partir d'un état nominal.
       Exemples: mise hors service de k noeuds (SE=0), suppression d'arêtes, etc.
    """
    name: str
    fn: Callable[[FDNAState, random.Random], FDNAState]


def attack_k_nodes(k: int, se_down: float = 0.0, weighted: Optional[Dict[Node, float]] = None) -> AttackModel:
    def _fn(state: FDNAState, rng: random.Random) -> FDNAState:
        s = state.copy()
        nodes = s.nodes[:]
        if weighted:
            # tirage pondéré
            pop = []
            for n in nodes:
                w = weighted.get(n, 1.0)
                pop.extend([n] * max(1, int(1000 * w)))
            chosen = set()
            while len(chosen) < min(k, len(nodes)):
                chosen.add(rng.choice(pop))
            targets = list(chosen)
        else:
            rng.shuffle(nodes)
            targets = nodes[:k]
        for n in targets:
            s.SE[n] = max(0.0, min(100.0, se_down))
        return s
    return AttackModel(name=f"attack_{k}_nodes", fn=_fn)


def attack_remove_edges(p_remove: float = 0.2) -> AttackModel:
    def _fn(state: FDNAState, rng: random.Random) -> FDNAState:
        s = state.copy()
        s.edges = [e for e in s.edges if rng.random() > p_remove]
        # Nettoyer SOD/COD orphelins
        for e in list(s.SOD.keys()):
            if e not in s.edges:
                s.SOD.pop(e, None)
                s.COD.pop(e, None)
        return s
    return AttackModel(name=f"remove_edges_p{p_remove}", fn=_fn)

# -------------------------------
# Simulation Monte Carlo
# -------------------------------

@dataclass
class MonteCarloConfig:
    n_runs: int = 5000
    seed: Optional[int] = None
    # mapping perf → score système
    perf_map: Optional[Callable[[Dict[Node, float]], float]] = None

@dataclass
class MonteCarloOutcome:
    baseline_scores: List[float]
    resilient_scores: List[float]
    baseline_O: List[Dict[Node, float]]
    resilient_O: List[Dict[Node, float]]
    summary: Dict[str, float]


def monte_carlo(
    base_state: FDNAState,
    attack: AttackModel,
    policy: Optional[ResiliencePolicy],
    cfg: MonteCarloConfig,
) -> MonteCarloOutcome:
    rng = random.Random(cfg.seed)
    baseline_scores: List[float] = []
    resilient_scores: List[float] = []
    baseline_O: List[Dict[Node, float]] = []
    resilient_O: List[Dict[Node, float]] = []

    for _ in range(cfg.n_runs):
        attacked = attack.fn(base_state, rng)
        # Baseline (sans résilience)
        r0 = compute_operability(attacked, perf_map=cfg.perf_map)
        baseline_scores.append(r0.score)
        baseline_O.append(r0.O)
        # Avec résilience
        if policy is not None:
            acts = policy.actions(attacked)
            repaired = apply_actions(attacked, acts)
            rR = compute_operability(repaired, perf_map=cfg.perf_map)
        else:
            rR = r0
        resilient_scores.append(rR.score)
        resilient_O.append(rR.O)

    # Résumés
    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else float('nan')

    mean_base = _mean(baseline_scores)
    mean_res = _mean(resilient_scores)
    gain_abs = mean_res - mean_base
    gain_rel = (gain_abs / mean_base * 100.0) if mean_base else 0.0

    summary = {
        'mean_baseline_score': mean_base,
        'mean_resilient_score': mean_res,
        'gain_abs': gain_abs,
        'gain_rel_percent': gain_rel,
    }

    return MonteCarloOutcome(
        baseline_scores, resilient_scores, baseline_O, resilient_O, summary
    )


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
    return FDNAState(SE=SE, SOD=SOD, COD=COD, nodes=nodes, edges=edges)


def perf_sum(O: Dict[Node, float]) -> float:
    return sum(O.values())


def policy_circuit_breaker_and_backup() -> ResiliencePolicy:
    """Exemple: si un prédécesseur d'un noeud final est KO, on ajoute une
    arête de secours depuis N1 et on réduit la propagation (scale SOD)."""
    def rule_scale_sod(attacked: FDNAState) -> List[ResilienceAction]:
        # Réduire légèrement la sensibilité des dépendances
        return [ResilienceAction('scale_sod', {'factor': 0.9})]

    def rule_backup_edge(attacked: FDNAState) -> List[ResilienceAction]:
        acts: List[ResilienceAction] = []
        # Si N2 ou N3 est très dégradé, ajouter une redondance vers N5 depuis N1
        if attacked.SE.get('N2', 100) < 30 or attacked.SE.get('N3', 100) < 30:
            acts.append(ResilienceAction('add_edge', {'i':'N1','j':'N5','sod':0.4,'cod':60.0}))
        return acts

    return ResiliencePolicy(name='CB+Backup', rules=[rule_scale_sod, rule_backup_edge])

def aggressive_policy() -> ResiliencePolicy:
    def scale_sod_rule(state: FDNAState) -> list[ResilienceAction]:
        return [ResilienceAction('scale_sod', {'factor': 0.7})]  # more reduction

    def backup_edge_rule(state: FDNAState) -> list[ResilienceAction]:
        acts = []
        # Add backup edges from N1 to all downstream nodes if predecessors are low
        for n in ['N2','N3','N4']:
            if state.SE.get(n, 100) < 50:
                acts.append(ResilienceAction('add_edge', {'i':'N1','j':'N5','sod':0.5,'cod':60}))
        return acts

    return ResiliencePolicy('Aggressive', rules=[scale_sod_rule, backup_edge_rule])


if __name__ == "__main__":
    base = example_state()
    atk = attack_k_nodes(k=1, se_down=0.0)
    pol = aggressive_policy()
    cfg = MonteCarloConfig(n_runs=2000, seed=42, perf_map=perf_sum)
    out = monte_carlo(base, atk, pol, cfg)
    print(out.summary)

    # 1. Histogramme des scores système
    plt.figure(figsize=(10,5))
    plt.hist(out.baseline_scores, bins=30, alpha=0.6, label='Baseline')
    plt.hist(out.resilient_scores, bins=30, alpha=0.6, label='Avec résilience')
    plt.xlabel('Score système')
    plt.ylabel('Nombre de simulations')
    plt.title('Distribution des scores Monte Carlo')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Boxplot par nœud (opérabilité)
    nodes = base.nodes
    baseline_O_matrix = [list(d.values()) for d in out.baseline_O]
    resilient_O_matrix = [list(d.values()) for d in out.resilient_O]

    plt.figure(figsize=(12,6))
    plt.boxplot([ [row[i] for row in baseline_O_matrix] for i in range(len(nodes)) ],
                positions=[i-0.2 for i in range(len(nodes))], widths=0.35)
    plt.boxplot([ [row[i] for row in resilient_O_matrix] for i in range(len(nodes)) ],
                positions=[i+0.2 for i in range(len(nodes))], widths=0.35)
    plt.xticks(range(len(nodes)), nodes)
    plt.ylabel('Opérabilité par nœud')
    plt.title('Comparaison opérabilité par nœud : baseline vs résilience')
    plt.legend(['Baseline','Résilience'])
    plt.grid(True)
    plt.show()
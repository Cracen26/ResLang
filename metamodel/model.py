class Node:
    def __init__(self, name, effectiveness, criticality):
        self.name = name
        self.effectiveness = effectiveness
        self.criticality = criticality
        self.operability = None
        self.dependencies = []
        self.predecessors = []
        self.successors = []
        self.root = False
        self.operabilityLogs = []
        self.performance = None

    def updateOperability(self, value):
        self.operability = value

    def average(self):
        temp = []
        for dependency in self.dependencies:
            if dependency.source.operability is not None:
                temp.append(dependency.alpha * dependency.source.operability + (1 - dependency.alpha * self.effectiveness))
        return sum(temp) / len(temp) if temp else 0

    def minimum(self):
        temp = []
        for dependency in self.dependencies:
            if dependency.source.operability is not None:
                temp.append(dependency.source.operability + dependency.beta)
        return min(temp) if temp else 0

    def computeOperability(self):
        if self.root:
            return self.operability
        self.operability = min(self.average(), self.minimum())
        return self.operability

    
    def getPredecessors(self):
        if self.root:
            for dependency in self.dependencies:
                self.predecessors.append(dependency)

    def computePerformance(self, heuristic):
        self.performance = self.operability*heuristic
        return self.performance
    
    def __str__(self):
        return (f"Node({self.name}): "
            f"effectiveness={self.effectiveness}, "
            f"criticality={self.criticality}, "
            f"operability={self.operability}, "
            f"root={self.root}, "
            f"dependencies={[d.source.name for d in self.dependencies]}, "
            f"logs={[d for d in self.operabilityLogs]}")
    


        
class Dependency:
    def __init__(self, source, target, alpha, beta):
        self.source = source
        self.target = target 
        self.alpha = alpha
        self.beta = beta

    def __str__(self):
        return (f"Dependency: "
            f"source={self.source}, "
            f"target={self.target}, "
            f"alpha={self.alpha}, "
            f"beta={self.beta}, ")
    
    def __str__(self):
        return (f"Dependency({self.source.name} -> {self.target.name}): "
            f"alpha={self.alpha}, beta={self.beta}")

class System:
    def __init__(self, name, nodes, dependencies, contexts):
        self.name = name
        self.nodes = nodes
        self.dependencies = dependencies
        self.contexts = contexts

    def initRoot(self):
        for node in self.nodes:
            if len(node.dependencies) == 0:
                node.root = True
                node.operability = node.effectiveness
            node.operabilityLogs.append(Log(node.operability))

    def propagateOperability(self):
        max_iter = 100
        epsilon = 1e-3
        iteration = 0
        converged = False

        self.initRoot()

        while not converged and iteration < max_iter:
            iteration += 1
            converged = True  # assume done until change found

            for node in self.nodes:
                if node.root:
                    continue  # skip roots

                new_op = node.computeOperability()
                if node.operability is None or abs(node.operability - new_op) > epsilon:
                    node.operability = new_op
                    node.operabilityLogs.append(Log(node.operability))
                    converged = False  # still changes happening

        print(f"Converged in {iteration} iterations")

    def computePerformance(self, heuristic):
        for node in self.nodes:
            node.computePerformance(heuristic)
    
    def introduceHazard(self):
        #we apply hazard in this method and recompute operability
        pass
    def __str__(self):
        return (f"System: "
            f"name={self.name}, "
            f"nodes={[node.name for node in self.nodes]}, "
            f"dependencies={[dep for dep in self.dependencies]},"
            f"context={self.contexts}")

class Context:
    def __init__(self, name, hazards, performances):
        self.name = name
        self.hazards = hazards
        self.performances = performances

    def __str__(self):
        return (f"Context: "
            f"name={self.name},"
            f"hazards={self.hazards},"
            f"perfomance={self.performances}")
class Hazard:
    def __init__(self, name, targets, latency, impact):
        self.name = name 
        self.targets = targets
        self.impact = impact
        self.latency = latency

    def __str__(self):
        return (f"Hazard: "
                f"name={self.name},"
                f"targets={self.targets},"
                f"latency={self.latency},"
                f"impact (%)={self.impact*100}"
                )
class Performance:
    def __init__(self, name):
        self.name = name 

    def __str__(self):
        return (f"Performance: "
                f"name:{self.name}")

class Log:
    def __init__(self, value):
        self.step = 0 
        self.value = value

    def __str__(self):
        return (f"Log: "
            f"step={self.step}, "
            f"value={self.value}")

# # Create nodes
# D1 = Node('D1', effectiveness=10, criticality=0.9)
# D2 = Node('D2', effectiveness=10, criticality=0.6)
# D3 = Node('D3', effectiveness=10, criticality=0.5)
# D4 = Node('D4', effectiveness=10, criticality=0.8)

# # Create dependencies
# dep1 = Dependency(source=D1, target=D2, alpha=0.3, beta=80)
# dep2 = Dependency(source=D2, target=D3, alpha=0.4, beta=85)
# dep3 = Dependency(source=D3, target=D4, alpha=0.2, beta=90)

# # Link dependencies to nodes
# D2.dependencies.append(dep1)
# D3.dependencies.append(dep2)
# D4.dependencies.append(dep3)

# # Build the system
# system = System(nodes=[D1, D2, D3, D4], dependencies=[dep1, dep2, dep3])

# system.propagateOperability()

# # Print results
# print(system)
# for node in system.nodes:
#     print(node)
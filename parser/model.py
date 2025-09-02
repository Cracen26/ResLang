class Component:
    def __init__(self, name, effectiveness):
        self.name = name
        self.effectiveness = effectiveness
        self.criticality = None
        self.operability = None
        self.dependencies = []
        self.predecessors = []
        self.successors = []
        self.root = False
        self.performance = None

    def updateOperability(self, value):
        self.operability = value

    def computePerformance(self, heuristic):
        self.performance = self.operability*heuristic
        return self.performance
    
    def __str__(self):
        return (f"Node({self.name}): "
            f"effectiveness={self.effectiveness}, "
            f"criticality={self.criticality}, "
            f"operability={self.operability}, "
            f"root={self.root}, "
            f"dependencies={[d.source.name for d in self.dependencies]}")
    
class Dependency:
    def __init__(self, source, target, alpha, beta):
        self.source = source
        self.target = target 
        self.alpha = alpha #SOD 
        self.beta = beta    #COD

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

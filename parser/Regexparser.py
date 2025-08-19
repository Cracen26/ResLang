import re
import ast
from metamodel.model import *

def returnNode(nodes, name):
    for node in nodes:
        if node.name == name:
            return node
    return name

def parse_dsl(dsl_text: str) -> System:
    system_match = re.search(r"system\s+(\w+)\s*{(.*?)}\s*context", dsl_text, re.DOTALL)
    system_name = system_match.group(1)
    system_body = system_match.group(2)

    components = {}
    for comp in re.findall(r"(\w+)\s*\[SE=(\d+),\s*criticality=([\d.]+)\];", system_body):
        name, se, crit = comp
        components[name] = Node(name=name, effectiveness=float(se), criticality=float(crit))

    nodes= []
    for value in components.values():
        nodes.append(value)

    dependencies = []
    for dep in re.findall(r"(\w+)\s*->\s*(\w+)\s*\[alpha=([\d.]+),\s*beta=(\d+)\];", system_body):
        src, tgt, alpha, beta = dep
        dependency = Dependency(source=returnNode(nodes, src), target=returnNode(nodes, tgt), alpha=float(alpha), beta=float(beta))
        
        for node in nodes:
            if node.name == dependency.target.name:
                node.dependencies.append(dependency)

        dependencies.append(dependency)

    # Context block
    context_match = re.search(r"context\s+(\w+)\s*{(.*?)}", dsl_text, re.DOTALL)
    ctx_name = context_match.group(1)
    ctx_body = context_match.group(2)
    
    # Hazards
    hazards = []
    for hz in re.findall(
        r"hazards\s+(\w+)\s*{\s*target=\[(.*?)\],\s*latency=(\d+);?\s*}", 
        ctx_body, re.DOTALL
    ):
        name, targets_str, latency = hz
        targets = re.findall(r"\w+", targets_str)
        hazards.append(Hazard(name=name, targets=targets, latency=float(latency), impact=0.5))

    
    # Performances
    performances = []
    for perf in re.findall(r"Function\s+'([^']+)'", ctx_body):
        performances.append(Performance(name=perf))

    context = Context(name=ctx_name, hazards=hazards, performances=performances)

    sys = System(name=system_name, nodes=nodes, dependencies=dependencies, contexts=[context])

    return sys


from ..parser.Regexparser import *

def initRoot(system):
    pass

if __name__ == "__main__":
    with open("model.res", "r") as f:
        data = f.read()

    system = parse_dsl(data)

    print(system)
    # system.propagateOperability()
    # print(system)

    # viz = Visualize(1, system)
    # viz.graphPlot()
    # viz.allOperability1()

    # print(system.nodes[1].dependencies[0])
    # print('before', system.nodes[0])
    # system.propagateOperability()
    # print('after', system.nodes[0])
    
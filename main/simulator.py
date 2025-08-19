def compute_gsp(system):
    system.link_dependencies()
    for node in system.nodes:
        node.compute_operability()

    gsp = sum(node.O * node.criticality for node in system.nodes)
    return gsp

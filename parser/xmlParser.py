import xml.etree.ElementTree as ET

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



# Load the XML file
tree = ET.parse("res.xml")  # replace with your filename
root = tree.getroot()

# Define namespaces
ns = {
    "archimate": "http://www.opengroup.org/xsd/archimate/3.0/",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance"
}

# Function to get property values
def get_properties(properties_element):
    props = {}
    if properties_element is not None:
        for prop in properties_element.findall("archimate:property", ns):
            prop_name = prop.get("propertyDefinitionRef")
            value_element = prop.find("archimate:value", ns)
            value = value_element.text if value_element is not None else None
            props[prop_name] = value
    return props

# Extract Equipment and Material elements
elements = []
for elem in root.findall("archimate:elements/archimate:element", ns):
    xsi_type = elem.get("{http://www.w3.org/2001/XMLSchema-instance}type")
    if xsi_type in ["Equipment", "Material"]:
        element_info = {
            "name": elem.find("archimate:name", ns).text,
            "type": xsi_type,
            "identifier": elem.get("identifier"),
            "properties": get_properties(elem.find("archimate:properties", ns))
        }
        elements.append(element_info)

# Extract BusinessEvent
elementsBusinessEvent = []
for elem in root.findall("archimate:elements/archimate:element", ns):
    xsi_type = elem.get("{http://www.w3.org/2001/XMLSchema-instance}type")
    if xsi_type in ["BusinessEvent"]:
        element_info = {
            "name": elem.find("archimate:name", ns).text,
            "type": xsi_type,
            "identifier": elem.get("identifier"),
            "properties": get_properties(elem.find("archimate:properties", ns))
        }
        elementsBusinessEvent.append(element_info)
        
# Extract Assessment
elementsAssessment = []
for elem in root.findall("archimate:elements/archimate:element", ns):
    xsi_type = elem.get("{http://www.w3.org/2001/XMLSchema-instance}type")
    if xsi_type in ["Assessment"]:
        element_info = {
            "name": elem.find("archimate:name", ns).text,
            "type": xsi_type,
            "identifier": elem.get("identifier"),
            "properties": get_properties(elem.find("archimate:properties", ns))
        }
        elementsAssessment.append(element_info)

# Extract Requirements
elementsRequirements= []
for elem in root.findall("archimate:elements/archimate:element", ns):
    xsi_type = elem.get("{http://www.w3.org/2001/XMLSchema-instance}type")
    if xsi_type in ["Requirement"]:
        element_info = {
            "name": elem.find("archimate:name", ns).text,
            "type": xsi_type,
            "identifier": elem.get("identifier"),
            "properties": get_properties(elem.find("archimate:properties", ns))
        }
        elementsRequirements.append(element_info)

# Print results
nodes = []
for e in elements:
    node = Component(e['name'], e['properties']['propid-1'] )
    nodes.append(node)
    # print(f"Name: {e['name']}, Type: {e['type']}, ID: {e['identifier']}, Properties: {e['properties']}")

hazard = []
for e in elementsBusinessEvent:
    # hazard.append(Hazard(e['name'],))
    print(f"Name: {e['name']}, Type: {e['type']}, ID: {e['identifier']}, Properties: {e['properties']}")

targets = []
for e1 in elementsAssessment:
        pass
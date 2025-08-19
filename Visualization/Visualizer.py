import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==== Dummy Node class to simulate your model ====
class Node:
    def __init__(self, name, operability):
        self.name = name
        self.operability = operability

# ==== Sample nodes (replace this with your actual list of Node instances) ====
nodes = [
    Node("D1", 0.78),
    Node("D2", 0.65),
    Node("D3", 0.72),
    Node("D4", 0.55),
    Node("D5", 0.83),
    Node("D6", 0.61)
]

# ==== Extract names and operability values ====
node_names = [node.name for node in nodes]
operabilities = [node.operability for node in nodes]

# ==== Bar Plot ====
plt.figure(figsize=(10, 5))
plt.bar(node_names, operabilities, color='skyblue')
plt.ylim(0, 1)
plt.ylabel("Operability")
plt.title("Operability of Each Node")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ==== Line Plot ====
plt.figure(figsize=(10, 5))
plt.plot(node_names, operabilities, marker='o', linestyle='-', color='green')
plt.ylim(0, 1)
plt.ylabel("Operability")
plt.title("Operability Progression Across Nodes")
plt.grid(True)
plt.tight_layout()
plt.show()

# ==== Heatmap ====
# plt.figure(figsize=(10, 1))
# sns.heatmap([operabilities], cmap="coolwarm", annot=True, xticklabels=node_names, yticklabels=[""])
# plt.title("Node Operability Heatmap")
# plt.tight_layout()
# plt.show()

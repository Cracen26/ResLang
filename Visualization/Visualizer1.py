import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import pandas as pd
import numpy as np

class Visualize:
    def __init__(self, id, model):
        self.id = id
        self.model = model
        self.graph = None


    #this class provide visualization of the model

    def graphPlot(self):
        self.graph = nx.DiGraph()
        for node in self.model.nodes:
            self.graph.add_node(str(node.name))

        for dep in self.model.dependencies:
            self.graph.add_edge(str(dep.source.name), str(dep.target.name), label=f'a={dep.alpha}, b={dep.beta}')
        
        edge_labels = nx.get_edge_attributes(self.graph, 'label')
        pos = nx.spring_layout(self.graph)  # You can also use other layouts like shell_layout, circular_layout
        
        nx.draw(self.graph, pos, with_labels=True, node_size=1000, arrows=True)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.title("Dependency Graph")
        plt.axis('off')
        return plt.show()


    def operabilityPlot(self):
        node_names = []
        operabilities = []
        for node in self.model.nodes:
            node_names.append(str(node.name))
            operabilities.append(node.operability)
        
        # print(node_names)
        # print(operabilities)
        plt.figure(figsize=(10, 1))
        sns.heatmap([operabilities], cmap="coolwarm", annot=True, xticklabels=node_names, yticklabels=[""])
        plt.title("Node Operability Heatmap")
        plt.tight_layout()
        return plt.show()
    

    def allOperability(self):
        node_names = []
        operabilities = []
        effectiveness = []
        for node in self.model.nodes:
            node_names.append(str(node.name))
            operabilities.append(node.operability)
            effectiveness.append(node.effectiveness)

        df = pd.DataFrame({
            'nodes': node_names,
            'ops': operabilities,
            'ses': effectiveness
        })

        f, ax = plt.subplots(figsize=(100, 10))

        # Load the example car crash dataset
        # crashes = sns.load_dataset("car_crashes").sort_values("total", ascending=False)
        
        # Plot the total crashes
        sns.set_color_codes("pastel")
        sns.barplot(x="ses", y="nodes", data=df,
                    label="Effectiveness", color="b")

        # Plot the crashes where alcohol was involved
        sns.set_color_codes("muted")
        sns.barplot(x="ops", y="nodes", data=df,
                    label="Operability", color="b")

        # Add a legend and informative axis label
        ax.legend(ncol=2, loc="lower right", frameon=True)
        ax.set(xlim=(0, 24), ylabel="",
            xlabel="Operability/Effectiveness of Nodes ")
        sns.despine(left=True, bottom=True)

        return plt.show()
    
    def allOperability1(self):
        node_names = []
        operabilities = []
        effectiveness = []
        for node in self.model.nodes:
            node_names.append(str(node.name))
            operabilities.append(node.operability)
            effectiveness.append(node.effectiveness)
        plt.style.use('_mpl-gallery')

        df = pd.DataFrame({
        'Component': node_names,
        'Operability': operabilities,
        'Effectiveness': effectiveness
        })

        # Sort by operability
        df = df.sort_values('Operability', ascending=False)

        # Plot
        plt.figure(figsize=(8, 6))
        sns.set_style("whitegrid")

        # Main bar: Operability
        sns.barplot(x='Operability', y='Component', data=df, color='skyblue', label='Operability', height=0.6)

        # Overlay bar: Effectiveness (narrower and darker)
        sns.barplot(x='Effectiveness', y='Component', data=df, color='navy', label='Effectiveness', height=0.3)

        plt.xlabel('Value (%)')
        plt.ylabel('Component')
        plt.title('Component Operability & Effectiveness')
        plt.legend()
        plt.tight_layout()
        plt.show()
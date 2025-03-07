import os
import ast
from difflib import SequenceMatcher
import tokenize
from io import BytesIO
from sklearn.cluster import DBSCAN
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import graphviz
import seaborn as sns
from pathlib import Path
from markdown2 import markdown
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for Flask
import matplotlib.pyplot as plt

# Similarity functions (AST, token, and control flow)
def ast_similarity(code1, code2):
    try:
        tree1, tree2 = ast.parse(code1), ast.parse(code2)
    except SyntaxError:
        return 0.0
    nodes1 = [type(node).__name__ for node in ast.walk(tree1)]
    nodes2 = [type(node).__name__ for node in ast.walk(tree2)]
    return SequenceMatcher(None, nodes1, nodes2).ratio()

def tokenize_code(code):
    tokens = []
    g = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
    for token in g:
        if token.type == tokenize.NAME:
            tokens.append('IDENTIFIER')
        elif token.type == tokenize.OP:
            tokens.append(token.string)
        elif token.type == tokenize.NUMBER:
            tokens.append('NUMBER')
        elif token.type == tokenize.STRING:
            tokens.append('STRING')
    return tokens

def token_similarity(code1, code2):
    tokens1, tokens2 = tokenize_code(code1), tokenize_code(code2)
    set1 = set(tokens1)
    set2 = set(tokens2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if len(union) != 0 else 1.0

from networkx.algorithms.similarity import graph_edit_distance

# Function to build a control flow graph from code
class ControlFlowGraphBuilder(ast.NodeVisitor):
    def __init__(self):
        self.cfg = nx.DiGraph()
        self.current_node = None
        self.node_count = 0

    def add_node(self, label):
        node_id = self.node_count

        
        self.cfg.add_node(node_id, label=label)
        if self.current_node is not None:
            self.cfg.add_edge(self.current_node, node_id)
        self.current_node = node_id
        self.node_count += 1

    def visit_FunctionDef(self, node):
        self.add_node(f"FunctionDef: {node.name}")
        self.generic_visit(node)  # Recursively visit child nodes

    def visit_If(self, node):
        self.add_node("If")
        self.generic_visit(node)

    def visit_For(self, node):
        self.add_node("For")
        self.generic_visit(node)

    def visit_While(self, node):
        self.add_node("While")
        self.generic_visit(node)

    def visit_Return(self, node):
        self.add_node("Return")

    # Additional nodes for expressions or custom logic can be added here

def build_control_flow_graph(code):
    tree = ast.parse(code)
    builder = ControlFlowGraphBuilder()
    builder.visit(tree)
    return builder.cfg


def control_flow_similarity(code1, code2):
    cfg1 = build_control_flow_graph(code1)
    cfg2 = build_control_flow_graph(code2)
    ged = graph_edit_distance(cfg1, cfg2)
    # Normalize similarity score (inverse relation with graph edit distance)
    normalized_score = 1 / (1 + ged) if ged is not None else 0
    return normalized_score

def calculate_weighted_similarity(ast_sim, token_sim, control_flow_sim, weights=(0.4, 0.4, 0.2)):
    return (weights[0] * ast_sim) + (weights[1] * token_sim) + (weights[2] * control_flow_sim)

def load_codes_from_folder(folder_path):
    code_files = [f for f in os.listdir(folder_path) if f.endswith(".py")]
    codes = {}
    for file in code_files:
        with open(os.path.join(folder_path, file), 'r') as f:
            codes[file] = f.read()
    return codes

def calculate_pairwise_similarities(codes):
    similarity_threshold = 0.7
    files = list(codes.keys())
    similarities = []
    for i, file1 in enumerate(files):
        for j, file2 in enumerate(files):
            if i < j:
                ast_sim = ast_similarity(codes[file1], codes[file2])
                token_sim = token_similarity(codes[file1], codes[file2])
                control_flow_sim = control_flow_similarity(codes[file1], codes[file2])
                overall_sim = calculate_weighted_similarity(ast_sim, token_sim, control_flow_sim)
                if overall_sim > similarity_threshold:
                    similarities.append((file1, file2, overall_sim))
    return similarities

def cluster_codes(similarities, codes, eps=0.5, min_samples=2):
    files = list(codes.keys())
    n = len(files)
    similarity_matrix = np.zeros((n, n))
    for (file1, file2, sim) in similarities:
        i, j = files.index(file1), files.index(file2)
        similarity_matrix[i][j] = similarity_matrix[j][i] = sim
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = clustering.fit_predict(1 - similarity_matrix)
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(files[idx])
    return clusters, similarity_matrix, files

# Visualization functions
def visualize_similarity_matrix(similarity_matrix, files, output_dir):
    # Ensure diagonal values are 1
    np.fill_diagonal(similarity_matrix, 1)
    
    # Set figure size and create the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix, 
        annot=True, 
        fmt=".2f", 
        xticklabels=files, 
        yticklabels=files, 
        cmap="YlGnBu", 
        cbar=True, 
        square=True  # Ensures cells are square-shaped
    )
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Add a title and adjust layout
    plt.title("Code Similarity Heatmap", fontsize=16, pad=20)
    plt.tight_layout()  # Automatically adjust layout to prevent cutoff

    # Save the heatmap
    filename = os.path.join(output_dir, "similarity_matrix.png")
    plt.savefig(filename, dpi=300)  # High resolution for better quality
    plt.close()
    
    return filename

def visualize_ast(code, filename):
    try:
        tree = ast.parse(code)
        graph = graphviz.Digraph(filename=filename)
        def add_nodes_edges(node, parent=None):
            node_id = str(id(node))
            graph.node(node_id, label=type(node).__name__)
            if parent:
                graph.edge(str(id(parent)), node_id)
            for child in ast.iter_child_nodes(node):
                add_nodes_edges(child, node)
        add_nodes_edges(tree)
        graph.render(filename=filename, format='png', cleanup=True)
        return str(filename) + '.png'
    except SyntaxError:
        return None

def visualize_tokens(code, filename):
    tokens = tokenize_code(code)
    token_counts = {token: tokens.count(token) for token in set(tokens)}
    
    plt.figure(figsize=(10, 5))
    plt.bar(token_counts.keys(), token_counts.values(), color='skyblue')
    plt.title("Token Frequency")
    plt.xlabel("Token Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    filepath = f"{filename}.png"
    plt.savefig(filepath)
    plt.close()
    return filepath

def visualize_control_flow(code, filename):
    try:
        tree = ast.parse(code)
        graph = nx.DiGraph()

        def add_nodes_edges(node, parent=None):
            node_id = str(id(node))
            graph.add_node(node_id, label=type(node).__name__)
            if parent:
                graph.add_edge(str(id(parent)), node_id)
            for child in ast.iter_child_nodes(node):
                add_nodes_edges(child, node)

        add_nodes_edges(tree)

        pos = nx.spring_layout(graph, k=0.5)  # Adjust k to increase spacing
        labels = nx.get_node_attributes(graph, 'label')
        node_sizes = [1000 if labels[node] == 'FunctionDef' else 300 for node in graph]
        
        plt.figure(figsize=(12, 8))
        nx.draw(graph, pos, labels=labels, with_labels=True, node_size=node_sizes, node_color="lightblue", font_size=10)
        
        filepath = f"{filename}.png"
        plt.savefig(filepath)
        plt.close()
        return filepath
    except SyntaxError:
        return None
import plotly.express as px
import pandas as pd
import random
from pathlib import Path

def generate_report(clusters, similarities, codes, similarity_matrix, files, output_dir,output_dir2):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    output_dir2 = Path(output_dir2)
    
        

    # Generate the cluster table (Separate HTML)
    cluster_md = """<html>
    <head>
        <title>Cluster Table</title>
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
                border: 1px solid black;
                font-family: Arial, sans-serif;
            }
            th, td {
                border: 1px solid black;
                padding: 10px;
                text-align: center;
            }
            thead {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            tr:nth-child(odd) {
                background-color: #ffffff;
            }
            tr.green {
                background-color: #c9f7c9; /* Green for outliers */
            }
            tr.red {
                background-color: #ffcccc; /* Red for the largest cluster */
            }
            tr.yellow {
                background-color: #ffff99; /* Yellow for other clusters */
            }
        </style>
    </head>
    <body>
        <h1>Cluster Overview</h1>
        <table>
            <thead>
                <tr>
                    <th>Cluster ID</th>
                    <th>Files</th>
                    <th>Total Files</th>
                </tr>
            </thead>
            <tbody>
    """

    # Find the cluster with the most members
    max_cluster_size = max(len(files) for files in clusters.values())
    
    for cluster_id, cluster_files in clusters.items():
        # Determine the color based on the number of members in the cluster
        if cluster_id == -1:  # Outliers
            highlight_class = "green"
        elif len(cluster_files) == max_cluster_size:  # Cluster with the most members
            highlight_class = "red"
        else:  # All other clusters
            highlight_class = "yellow"
        
        cluster_md += f"""
                <tr class="{highlight_class}">
                    <td>{'Outliers' if cluster_id == -1 else cluster_id}</td>
                    <td>{', '.join(cluster_files)}</td>
                    <td>{len(cluster_files)}</td>
                </tr>
        """
    cluster_md += """
            </tbody>
        </table>
    </body>
    </html>
    """

    # Save cluster table to separate HTML file
    cluster_path = output_dir2 / "cluster.html"
    cluster_path.write_text(cluster_md)
    print(f"Cluster table generated at {cluster_path}")

    # Generate the main report
    '''report_md = "# Code Similarity Report\n\n"
    
    # Similarity Matrix Visualization
    sim_matrix_img = visualize_similarity_matrix(similarity_matrix, files, output_dir)
    report_md += f"![Similarity Matrix]({sim_matrix_img})\n\n"

    # Add reference to the cluster table
    report_md += "## Cluster Overview\n\n"
    report_md += f"View the detailed cluster table [here](cluster.html).\n\n"
    '''
    from flask import url_for
    
    from markdown2 import markdown

# Assume similarity_matrix and files are provided
# Assume output_dir and output_dir2 are paths where files are saved

# Initialize the HTML content for the report
    heat_mp = """
    <h1>Code Similarity Report</h1>

    <p style="text-align: center;">
        <img src="{{ url_for('custom_static', filename='similarity_matrix.png') }}" 
             alt="Custom Static Image" 
             style="max-width: 100%; max-height: 100vh; object-fit: contain;" />
    </p>
"""


# Generate the similarity matrix image
    sim_matrix_img = visualize_similarity_matrix(similarity_matrix, files, output_dir)

# Write the HTML file to the output directory
    heat_path = output_dir2 / "heatmap.html"
    heat_path.write_text(heat_mp)

    # Start building the HTML structure for the report
    report_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Code Similarity Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 1rem;
                background-color: #f9f9f9;
            }
            .container {
                max-width: 900px;
                margin: auto;
                background: #fff;
                padding: 2rem;
                border-radius: 8px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            }
            h1, h2 {
                text-align: center;
                color: #333;
            }
            .cluster {
                margin-bottom: 1rem;
            }
            .visualization {
                display: none;
                margin-top: 1rem;
            }
            .btn {
                display: inline-block;
                margin: 0.5rem 0;
                padding: 0.5rem 1rem;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                text-decoration: none;
            }
            .btn:hover {
                background-color: #0056b3;
            }
            img {
                max-width: 100%;
                height: auto;
                margin: 0.5rem 0;
            }
        </style>
        <script>
            function toggleVisualization(id) {
                const section = document.getElementById(id);
                section.style.display = section.style.display === "none" ? "block" : "none";
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Code Similarity Report</h1>
            <h2>Clusters</h2>
    """

    # Iterate over clusters to generate content for each cluster
    for cluster_id, cluster_files in clusters.items():
        cluster_title = "Outliers" if cluster_id == -1 else f"Cluster {cluster_id}"
        report_html += f"<h3>{cluster_title}</h3>"
        
        for file in cluster_files:
            sanitized_file = file.replace(" ", "_").replace("/", "_")
            report_html += f"""
            <div class="cluster">
                <p>
                    <strong>{file}</strong>
                    <button class="btn" onclick="toggleVisualization('{sanitized_file}-visualizations')">View Visualizations</button>
                </p>
                <div id="{sanitized_file}-visualizations" class="visualization">
            """
            
            # Add AST Visualization
            ast_img = visualize_ast(codes[file], output_dir / f"{file}_ast")
            if ast_img:
                report_html += f"""
                    <p><strong>AST:</strong></p>
                    <img src="{{{{ url_for('custom_static', filename='{sanitized_file}_ast.png') }}}}" alt="AST for {file}">
                """
            
            # Add Token Frequency Visualization
            token_img = visualize_tokens(codes[file], output_dir / f"{file}_tokens")
            if token_img:
                report_html += f"""
                    <p><strong>Token Frequency:</strong></p>
                    <img src="{{{{ url_for('custom_static', filename='{sanitized_file}_tokens.png') }}}}" alt="Token Frequency for {file}">
                """
            
            # Add Control Flow Visualization
            cfg_img = visualize_control_flow(codes[file], output_dir / f"{file}_cfg")
            if cfg_img:
                report_html += f"""
                    <p><strong>Control Flow:</strong></p>
                    <img src="{{{{ url_for('custom_static', filename='{sanitized_file}_cfg.png') }}}}" alt="CFG for {file}">
                """
            
            # Close the visualization section
            report_html += "</div></div>"

    # Close the HTML tags to finish the report
    report_html += """
        </div>
    </body>
    </html>
    """

    # Write the HTML report to file
    report_path = output_dir2 / "report.html"
    report_path.write_text(report_html)
    print(f"Report generated at {report_path}")
    # Generate and save similarity graph
    graph_img = visualize_similarity_graph(similarity_matrix, files, output_dir)
    
    generate_similarity_pdf(output_dir)

# Add graph image to the report
    # Generate and save similarity graph




import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import networkx as nx
import matplotlib
matplotlib.use("Agg")  # Fix GUI issue
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import networkx as nx
import matplotlib
matplotlib.use("Agg")  # Fix GUI issue
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_similarity_graph(similarity_matrix, files, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    G = nx.Graph()

    for file in files:
        G.add_node(file)

    similarity_values = []
    edges = []
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            similarity_score = similarity_matrix[i][j]
            if similarity_score > 0:
                G.add_edge(files[i], files[j], weight=similarity_score)
                similarity_values.append(similarity_score)
                edges.append((files[i], files[j]))

    min_sim = min(similarity_values) if similarity_values else 0
    max_sim = max(similarity_values) if similarity_values else 1
    edge_colors = [
        (similarity_matrix[i][j] - min_sim) / (max_sim - min_sim)
        for i in range(len(files)) for j in range(i + 1, len(files))
        if similarity_matrix[i][j] > 0
    ]

    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=700, node_color="lightblue", edgecolors="black")
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, edge_color=edge_colors, edge_cmap=plt.cm.viridis, width=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min_sim, vmax=max_sim))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Similarity Score")

    # Save the graph in the output directory
    graph_path = output_dir / "similarity_graph.png"
    plt.savefig(graph_path, bbox_inches="tight")
    plt.close()

    print(f"Similarity graph saved at {graph_path}")

    # HTML Content with Improved UI
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Similarity Graph</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #6a11cb, #2575fc);
            color: #fff;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        h1 {
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        p {
            font-size: 16px;
            margin-bottom: 15px;
            color: #e0e0e0;
        }
        .graph-container {
            margin-top: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            border: 4px solid #fff;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease-in-out;
        }
        img:hover {
            transform: scale(1.05);
        }
        .btn {
            display: inline-block;
            margin-top: 20px;
            padding: 12px 20px;
            background: #ff4081;
            color: #fff;
            text-decoration: none;
            font-weight: bold;
            border-radius: 8px;
            transition: background 0.3s ease-in-out;
        }
        .btn:hover {
            background: #e73370;
        }
        footer {
            margin-top: 25px;
            font-size: 14px;
            color: #ddd;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Code Similarity Graph</h1>
        <p>This graph visually represents the similarity between code submissions.</p>
        <div class="graph-container">
            <img src="{{ url_for('custom_static', filename='similarity_graph.png') }}" alt="Similarity Graph">
        </div>
        <a href="/" class="btn">Go Back</a>
        <footer>
            <p>&copy; 2025 Online Programming Assignment Portal. All rights reserved.</p>
        </footer>
    </div>
</body>
</html>"""


    html_path = Path(r"C:\Users\Rhythm\Downloads\date 6 latest frontend\06-03-2025\Online_Programming_Assignment_Portal-main\src\templates\similarity_graph.html")
    with open(html_path, "w", encoding="utf-8") as html_file:
        html_file.write(html_content)

    print(f"HTML page created at {html_path}")
    return html_path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pathlib import Path

def generate_similarity_pdf(output_dir):
    output_dir = Path(output_dir)
    pdf_path = output_dir / "similarity_report.pdf"
    graph_path = output_dir / "similarity_graph.png"

    if not graph_path.exists():
        print(f"Error: Graph image not found at {graph_path}")
        return None

    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter

    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, height - 50, "Code Similarity Report")

    # Add description
    c.setFont("Helvetica", 12)
    text = "This report provides a visual representation of code similarity between different submissions."
    c.drawString(50, height - 80, text)

    # Embed the graph image
    c.drawImage(str(graph_path), 50, height - 400, width=500, height=300)

    # Add footer
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 50, "Generated by Online Programming Assignment Portal - 2025")

    # Save the PDF
    c.save()
    print(f"PDF report saved at {pdf_path}")

    return pdf_path





    
    



'''def main():
    folder_path = os.path.join(os.getcwd(), "codes")
    output_dir = os.path.join(os.getcwd(), "report")
    codes = load_codes_from_folder(folder_path)
    similarities = calculate_pairwise_similarities(codes)
    clusters, similarity_matrix, files = cluster_codes(similarities, codes)
    print("Code Clusters based on Approach:")
    for cluster_id, cluster_files in clusters.items():
        if cluster_id == -1:
            print("Outliers:", cluster_files)
        else:
            print(f"Cluster {cluster_id}: {cluster_files}")
    generate_report(clusters, similarities, codes, similarity_matrix, files, output_dir)

if __name__ == "__main__":
    main()'''

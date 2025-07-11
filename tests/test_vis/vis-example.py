import torch
import torch.fx as fx
from vis_blk import test_flux_transformer_block
import os
import torch.nn as nn

def visualize_fx_graph(graph, output_path="fx_graph", format="html", height="800px", width="1200px", 
                       notebook=False, show_attrs=True, physics=True):
    """
    Visualize a PyTorch FX graph using pyvis and export to various formats.
    
    Args:
        graph: torch.fx.Graph object to visualize
        output_path: Path to save the output file (without extension)
        format: Output format - 'html', 'svg', or 'png'
        height: Height of the visualization
        width: Width of the visualization
        notebook: Whether to display the graph in a Jupyter notebook
        show_attrs: Whether to show node attributes in the visualization
        physics: Whether to enable the physics simulation for node layout
    
    Returns:
        Path to the saved visualization file
    """
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError("Please install pyvis first: pip install pyvis")
    
    # Create a network
    net = Network(height=height, width=width, notebook=notebook, directed=True)
    
    # Configure physics if disabled
    if not physics:
        net.toggle_physics(False)
    
    # Node and edge tracking
    node_map = {}
    
    # Add nodes to the graph
    for i, node in enumerate(graph.nodes):
        # Get operation and target info
        op_name = node.op
        if op_name == 'call_module':
            node_label = f"{node.name}\n({op_name}: {node.target})"
            node_color = "#C2FABC"  # Light green for modules
        elif op_name == 'call_function':
            fn_name = str(node.target).split('.')[-1]
            node_label = f"{node.name}\n({fn_name})"
            node_color = "#FFC0CB"  # Pink for functions
        elif op_name == 'call_method':
            node_label = f"{node.name}\n({op_name}: {node.target})"
            node_color = "#ADD8E6"  # Light blue for methods
        elif op_name == 'get_attr':
            node_label = f"{node.name}\n({op_name}: {node.target})"
            node_color = "#FFFACD"  # Light yellow for attributes
        else:  # placeholder, output
            node_label = f"{node.name}\n({op_name})"
            node_color = "#D3D3D3"  # Light gray for others
        
        # Add additional info if needed
        if show_attrs and hasattr(node, 'meta') and node.meta:
            try:
                shape_info = str(node.meta.get('tensor_meta', {}).get('shape', ''))
                if shape_info:
                    node_label += f"\nShape: {shape_info}"
            except:
                pass  # Skip if there's an error getting meta info
        
        # Add the node
        net.add_node(i, label=node_label, title=str(node), color=node_color)
        node_map[node] = i
    
    # Add edges
    for node in graph.nodes:
        target_idx = node_map[node]
        
        # Connect this node to its inputs
        for input_node in node.all_input_nodes:
            source_idx = node_map[input_node]
            net.add_edge(source_idx, target_idx, arrows='to')
    
    # Save the visualization
    if format.lower() == 'html':
        output_file = f"{output_path}.html"
        net.save_graph(output_file)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'html', 'svg', or 'png'.")
        
    print(f"Graph visualization saved to: {output_file}")
    return output_file

from torch._dynamo import export
def _find_linear_layers_with_dynamo(model: nn.Module, args, kwargs={}):
    """Use TorchDynamo to find linear layers and their connections."""
    # Export the model with TorchDynamo - using updated API
    exported_program = export(model)(*args, **kwargs)
    fx_graph = exported_program.graph_module
    graph = fx_graph.graph

def export_model_graph_to_html():
    """Example function that exports a model's FX graph to HTML visualization"""
    
    # Step 1: Get a model to visualize
    model, kwargs = test_flux_transformer_block()
    print(f"Created model: {type(model).__name__}")
    
    # Step 2: Create an FX graph from the model
    # Option A: Using symbolic_trace for simpler models
    try:
        traced_model = fx.symbolic_trace(model)
        graph = traced_model.graph
        graph_source = "symbolic_trace"
    # Option B: Using torch.export for more complex models
    except Exception as e:
        print(f"Symbolic trace failed: {e}, trying torch._dynamo.export")
        from torch._dynamo import export
        exported_program = export(model)(**kwargs)
        graph = exported_program.graph_module.graph
        graph_source = "dynamo_export"
    
    print(f"Generated FX graph using {graph_source}")
    
    # Step 3: Visualize the graph and export to HTML
    output_dir = "results/graph_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "flux_transformer_graph")
    html_file = visualize_fx_graph(
        graph=graph,
        output_path=output_path,
        format="html",  # HTML format
        height="800px", 
        width="1200px",
        show_attrs=True,
        physics=True  # Enable physics for better layout
    )
    
    print(f"\nVisualization complete!")
    print(f"To view the graph, open this file in a web browser: {html_file}")
    return html_file

def visualize_fx_graph_using_networkx(graph, output_path="fx_graph_nx", show_plot=False):
    """
    Visualize a PyTorch FX graph using NetworkX and matplotlib.
    
    Args:
        graph: torch.fx.Graph object to visualize
        output_path: Path to save the output image file (without extension)
        show_plot: Whether to display the plot interactively
    
    Returns:
        Path to the saved visualization file
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Please install required packages: pip install networkx matplotlib")
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Node and edge tracking
    node_map = {}
    node_colors = []
    node_labels = {}
    
    # Define colors for different node types
    color_map = {
        'call_module': '#C2FABC',  # Light green for modules
        'call_function': '#FFC0CB',  # Pink for functions
        'call_method': '#ADD8E6',  # Light blue for methods
        'get_attr': '#FFFACD',  # Light yellow for attributes
        'placeholder': '#D3D3D3',  # Light gray
        'output': '#D3D3D3',  # Light gray
    }
    
    # Add nodes to the graph
    for i, node in enumerate(graph.nodes):
        G.add_node(i)
        node_map[node] = i
        
        # Get operation and target info
        op_name = node.op
        if op_name == 'call_module':
            label_text = f"{node.name}\n({op_name}: {node.target})"
        elif op_name == 'call_function':
            fn_name = str(node.target).split('.')[-1]
            label_text = f"{node.name}\n({fn_name})"
        elif op_name == 'call_method':
            label_text = f"{node.name}\n({op_name}: {node.target})"
        elif op_name == 'get_attr':
            label_text = f"{node.name}\n({op_name}: {node.target})"
        else:  # placeholder, output
            label_text = f"{node.name}\n({op_name})"
        
        # Add shape info if available
        if hasattr(node, 'meta') and node.meta:
            try:
                shape_info = str(node.meta.get('tensor_meta', {}).get('shape', ''))
                if shape_info:
                    label_text += f"\nShape: {shape_info}"
            except:
                pass  # Skip if there's an error getting meta info
        
        node_labels[i] = label_text
        node_colors.append(color_map.get(op_name, '#D3D3D3'))
    
    # Add edges
    for node in graph.nodes:
        target_idx = node_map[node]
        
        # Connect this node to its inputs
        for input_node in node.all_input_nodes:
            source_idx = node_map[input_node]
            G.add_edge(source_idx, target_idx)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Generate layout (positioning of nodes)
    pos = nx.spring_layout(G, seed=42)  # for reproducibility
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_family='sans-serif')
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    output_file = f"{output_path}.png"
    plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    print(f"NetworkX graph visualization saved to: {output_file}")
    return output_file

def export_model_graph_to_networkx():
    """Example function that exports a model's FX graph to NetworkX visualization"""
    
    # Get a model and create graph (same as in export_model_graph_to_html)
    model, kwargs = test_flux_transformer_block()
    
    try:
        traced_model = fx.symbolic_trace(model)
        graph = traced_model.graph
    except Exception as e:
        from torch._dynamo import export
        exported_program = export(model)(**kwargs)
        graph = exported_program.graph_module.graph
    
    # Visualize with NetworkX
    output_dir = "results/graph_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "flux_transformer_graph_nx")
    return visualize_fx_graph_using_networkx(graph, output_path)

if __name__ == "__main__":
    html_file = export_model_graph_to_html()
    # export_model_graph_to_networkx()
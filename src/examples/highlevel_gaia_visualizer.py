#!/usr/bin/env python3
"""
Simple GAIA Architecture Visualizer

A lightweight visualization tool that demonstrates GAIA's categorical structures
without relying on the fuzzy encoding pipeline that has sample size constraints.

This visualizer focuses on:
1. Model architecture inspection
2. Categorical structure visualization
3. Business hierarchy display
4. Simplicial complex rendering
5. Interactive model exploration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time

# GAIA imports
from gaia.training.config import GAIALanguageModelConfig
from gaia.models.gaia_language_model import GAIALanguageModel
from gaia.utils.device import get_device

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    figure_size: Tuple[int, int] = (16, 12)
    dpi: int = 100
    color_scheme: str = 'viridis'
    node_size: int = 300
    edge_width: float = 1.5
    font_size: int = 10
    animation_speed: float = 0.1

class SimpleGAIAVisualizer:
    """Simple GAIA architecture visualizer without fuzzy encoding dependencies."""
    
    def __init__(self, config: Optional[GAIALanguageModelConfig] = None):
        """Initialize the simple visualizer."""
        self.device = torch.device('cpu')  # Force CPU to avoid device issues
        
        # Use minimal config to avoid fuzzy encoding
        if config is None:
            config = GAIALanguageModelConfig(
                # Model architecture
                vocab_size=100,
                d_model=64,
                num_heads=2,
                num_layers=2,
                seq_len=16,
                
                # Disable fuzzy components to avoid sample size issues
                enable_fuzzy_components=False,
                
                # Minimal UMAP configuration
                umap_n_neighbors=3,
                umap_min_dist=0.05,
                umap_spread=1.2,
            )
        
        self.config = config
        self.viz_config = VisualizationConfig()
        
        # Initialize model without fuzzy encoding
        logger.info("🚀 Initializing Simple GAIA Model...")
        self.model = GAIALanguageModel(config=self.config)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Set up visualization components
        self.fig = None
        self.axes = None
        self._setup_visualization()
        
        logger.info("✅ Simple GAIA Visualizer initialized successfully")
    
    def _setup_visualization(self):
        """Set up the visualization layout."""
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(2, 3, figsize=self.viz_config.figure_size, 
                                          dpi=self.viz_config.dpi)
        self.fig.suptitle('GAIA Architecture Visualization', fontsize=16, color='white')
        
        # Configure subplots
        titles = [
            'Model Architecture', 'Categorical Structures', 'Business Hierarchy',
            'Simplicial Complex', 'Message Flow', 'Component Inspector'
        ]
        
        for i, (ax, title) in enumerate(zip(self.axes.flat, titles)):
            ax.set_title(title, fontsize=12, color='white')
            ax.set_facecolor('black')
    
    def visualize_model_architecture(self):
        """Visualize ALL actual GAIA categorical structures with runtime inspection."""
        ax = self.axes[0, 0]
        ax.clear()
        ax.set_title('Complete GAIA Categorical Architecture', fontsize=14, weight='bold', color='white')
        
        # Create network graph
        G = nx.DiGraph()
        
        # Extract ALL real model components dynamically
        components = {}
        
        # CORE NEURAL COMPONENTS
        if hasattr(self.model, 'position_embeddings'):
            pos_params = sum(p.numel() for p in self.model.position_embeddings.parameters())
            components['Position\nEmbeddings'] = pos_params
            G.add_node('Position\nEmbeddings', size=pos_params, type='embedding', category='neural')
        
        if hasattr(self.model, 'gaia_transformer'):
            transformer_params = sum(p.numel() for p in self.model.gaia_transformer.parameters())
            components['GAIA\nTransformer'] = transformer_params
            G.add_node('GAIA\nTransformer', size=transformer_params, type='transformer', category='neural')
        
        # CATEGORICAL OPERATIONS
        if hasattr(self.model, 'categorical_ops'):
            try:
                cat_params = sum(p.numel() for p in self.model.categorical_ops.parameters())
            except:
                cat_params = 0
            components['Categorical\nOperations'] = cat_params
            G.add_node('Categorical\nOperations', size=cat_params, type='categorical', category='categorical')
        
        # COALGEBRAS - Extract from model components
        coalgebra_components = []
        if hasattr(self.model, 'components'):
            for name, comp in self.model.components.items():
                if 'coalgebra' in name.lower():
                    coalgebra_components.append(name)
                    G.add_node(f'{name}', size=1000, type='coalgebra', category='coalgebra')
                    components[f'{name}'] = 'Coalgebra Structure'
        
        # KAN EXTENSIONS - Extract from model components
        kan_components = []
        if hasattr(self.model, 'components'):
            for name, comp in self.model.components.items():
                if 'kan' in name.lower() or 'extension' in name.lower():
                    kan_components.append(name)
                    G.add_node(f'{name}', size=800, type='kan_extension', category='kan')
                    components[f'{name}'] = 'Kan Extension'
        
        # ENDS/COENDS - Extract from model components
        end_coend_components = []
        if hasattr(self.model, 'components'):
            for name, comp in self.model.components.items():
                if 'end' in name.lower() or 'coend' in name.lower():
                    end_coend_components.append(name)
                    G.add_node(f'{name}', size=600, type='end_coend', category='integral')
                    components[f'{name}'] = 'End/Coend Structure'
        
        # YONEDA STRUCTURES
        yoneda_components = []
        if hasattr(self.model, 'components'):
            for name, comp in self.model.components.items():
                if 'yoneda' in name.lower():
                    yoneda_components.append(name)
                    G.add_node(f'{name}', size=700, type='yoneda', category='yoneda')
                    components[f'{name}'] = 'Yoneda Structure'
        
        # FUZZY COMPONENTS
        fuzzy_components = []
        if hasattr(self.model, 'components'):
            for name, comp in self.model.components.items():
                if 'fuzzy' in name.lower():
                    fuzzy_components.append(name)
                    G.add_node(f'{name}', size=500, type='fuzzy', category='fuzzy')
                    components[f'{name}'] = 'Fuzzy Structure'
        
        # SIMPLICIAL STRUCTURES
        simplicial_components = []
        if hasattr(self.model, 'components'):
            for name, comp in self.model.components.items():
                if 'simplicial' in name.lower() or 'simplex' in name.lower():
                    simplicial_components.append(name)
                    G.add_node(f'{name}', size=400, type='simplicial', category='simplicial')
                    components[f'{name}'] = 'Simplicial Structure'
        
        # BUSINESS HIERARCHY
        if hasattr(self.model, 'business_hierarchy') and self.model.business_hierarchy:
            G.add_node('Business\nHierarchy', size=300, type='business', category='hierarchy')
            components['Business\nHierarchy'] = 'Hierarchical Structure'
        
        # MESSAGE PASSING
        if hasattr(self.model, 'message_passing') and self.model.message_passing:
            G.add_node('Message\nPassing', size=350, type='messaging', category='messaging')
            components['Message\nPassing'] = 'Message Passing System'
        
        # Add comprehensive edges showing categorical relationships
        edges_to_add = [
            ('Position\nEmbeddings', 'GAIA\nTransformer'),
            ('GAIA\nTransformer', 'Categorical\nOperations'),
        ]
        
        # Connect coalgebras to transformer
        for comp in coalgebra_components:
            if 'GAIA\nTransformer' in G.nodes():
                edges_to_add.append(('GAIA\nTransformer', comp))
        
        # Connect Kan extensions to coalgebras
        for kan_comp in kan_components:
            for coal_comp in coalgebra_components:
                edges_to_add.append((coal_comp, kan_comp))
        
        # Connect ends/coends to Kan extensions
        for end_comp in end_coend_components:
            for kan_comp in kan_components:
                edges_to_add.append((kan_comp, end_comp))
        
        # Connect Yoneda to everything (universal property)
        for yoneda_comp in yoneda_components:
            for other_comp in coalgebra_components + kan_components:
                edges_to_add.append((other_comp, yoneda_comp))
        
        # Add all valid edges
        for source, target in edges_to_add:
            if source in G.nodes() and target in G.nodes():
                G.add_edge(source, target)
        
        # Create hierarchical layout
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
        
        # Color nodes by category
        color_map = {
            'neural': '#ff6b6b',
            'categorical': '#4ecdc4', 
            'coalgebra': '#45b7d1',
            'kan': '#96ceb4',
            'integral': '#ffeaa7',
            'yoneda': '#fd79a8',
            'fuzzy': '#fdcb6e',
            'simplicial': '#e17055',
            'hierarchy': '#74b9ff',
            'messaging': '#a29bfe'
        }
        
        node_colors = [color_map.get(G.nodes[node].get('category', 'neural'), '#ffffff') for node in G.nodes()]
        node_sizes = [max(G.nodes[node].get('size', 100) / 100, 50) for node in G.nodes()]
        
        # Draw the complete categorical architecture
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, 
                              node_color=node_colors, alpha=0.8, edgecolors='white', linewidths=1)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='white', 
                              width=1.5, alpha=0.6, arrows=True, arrowsize=15)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_color='white', font_weight='bold')
        
        # Add category legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                    markersize=8, label=cat.title()) 
                          for cat, color in color_map.items() if any(G.nodes[n].get('category') == cat for n in G.nodes())]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=6)
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_facecolor('black')
        ax.axis('off')
        
        logger.info(f"🔍 COMPLETE GAIA ARCHITECTURE EXTRACTED:")
        logger.info(f"   • Neural Components: {len([n for n in G.nodes() if G.nodes[n].get('category') == 'neural'])}")
        logger.info(f"   • Coalgebras: {len(coalgebra_components)}")
        logger.info(f"   • Kan Extensions: {len(kan_components)}")
        logger.info(f"   • Ends/Coends: {len(end_coend_components)}")
        logger.info(f"   • Yoneda Structures: {len(yoneda_components)}")
        logger.info(f"   • Fuzzy Components: {len(fuzzy_components)}")
        logger.info(f"   • Simplicial Structures: {len(simplicial_components)}")
        logger.info(f"   • Total Categorical Objects: {len(G.nodes())}")
        logger.info(f"   • Fuzzy Simplicial Integration: {'✓' if simplicial_components and fuzzy_components else '✗'}")
        
        return {
            'components': components,
            'graph': G,
            'layout': pos,
            'coalgebras': coalgebra_components,
            'kan_extensions': kan_components,
            'ends_coends': end_coend_components,
            'yoneda_structures': yoneda_components,
            'fuzzy_components': fuzzy_components,
            'simplicial_components': simplicial_components
        }
    
    def visualize_categorical_structures(self):
        """Visualize ALL categorical structures with coalgebra, Kan extension, and end/coend analysis."""
        ax = self.axes[0, 1]
        ax.clear()
        ax.set_title('Categorical Structures Hierarchy', fontsize=12, color='white')
        
        # Create hierarchical graph for categorical structures
        G = nx.DiGraph()
        
        # LEVEL 1: FUNDAMENTAL CATEGORICAL STRUCTURES
        fundamental_structures = {
            'F-Coalgebras': {'type': 'coalgebra', 'level': 1, 'color': '#ff6b6b'},
            'Endofunctors': {'type': 'functor', 'level': 1, 'color': '#4ecdc4'},
            'Categories': {'type': 'category', 'level': 1, 'color': '#45b7d1'}
        }
        
        # LEVEL 2: SPECIFIC COALGEBRA TYPES (from universal_coalgebras.py)
        coalgebra_types = {
            'BackpropagationFunctor': {'type': 'backprop_functor', 'level': 2, 'color': '#96ceb4'},
            'FuzzyBackpropagationFunctor': {'type': 'fuzzy_backprop', 'level': 2, 'color': '#ffeaa7'},
            'GenerativeCoalgebra': {'type': 'generative', 'level': 2, 'color': '#fd79a8'},
            'FSSCoalgebra': {'type': 'fss_coalgebra', 'level': 2, 'color': '#fdcb6e'}
        }
        
        # LEVEL 3: KAN EXTENSION STRUCTURES (from kan_extensions.py)
        kan_structures = {
            'LeftKanExtension': {'type': 'left_kan', 'level': 3, 'color': '#e17055'},
            'RightKanExtension': {'type': 'right_kan', 'level': 3, 'color': '#74b9ff'},
            'FuzzySimplicialFunctor': {'type': 'fuzzy_functor', 'level': 3, 'color': '#a29bfe'},
            'FuzzyNaturalTransformation': {'type': 'natural_trans', 'level': 3, 'color': '#6c5ce7'}
        }
        
        # LEVEL 4: ENDS/COENDS (from ends_coends.py)
        integral_structures = {
            'End': {'type': 'end', 'level': 4, 'color': '#00b894'},
            'Coend': {'type': 'coend', 'level': 4, 'color': '#00cec9'},
            'Sheaf': {'type': 'sheaf', 'level': 4, 'color': '#55a3ff'},
            'Topos': {'type': 'topos', 'level': 4, 'color': '#fd79a8'}
        }
        
        # LEVEL 5: GEOMETRIC STRUCTURES
        geometric_structures = {
            'GeometricTransformer': {'type': 'geometric', 'level': 5, 'color': '#ff7675'},
            'TopologicalEmbedding': {'type': 'topological', 'level': 5, 'color': '#fab1a0'},
            'CategoricalProbability': {'type': 'probability', 'level': 5, 'color': '#e84393'}
        }
        
        # Add all nodes to graph
        all_structures = {**fundamental_structures, **coalgebra_types, **kan_structures, 
                         **integral_structures, **geometric_structures}
        
        for name, attrs in all_structures.items():
            G.add_node(name, **attrs)
        
        # Add hierarchical edges showing categorical relationships
        hierarchical_edges = [
            # Fundamental to specific
            ('F-Coalgebras', 'BackpropagationFunctor'),
            ('F-Coalgebras', 'FuzzyBackpropagationFunctor'),
            ('F-Coalgebras', 'GenerativeCoalgebra'),
            ('F-Coalgebras', 'FSSCoalgebra'),
            ('Endofunctors', 'BackpropagationFunctor'),
            ('Categories', 'FuzzySimplicialFunctor'),
            
            # Coalgebras to Kan extensions
            ('GenerativeCoalgebra', 'LeftKanExtension'),
            ('FSSCoalgebra', 'RightKanExtension'),
            ('FuzzyBackpropagationFunctor', 'FuzzySimplicialFunctor'),
            
            # Kan extensions to ends/coends
            ('LeftKanExtension', 'End'),
            ('RightKanExtension', 'Coend'),
            ('FuzzyNaturalTransformation', 'Sheaf'),
            ('FuzzySimplicialFunctor', 'Topos'),
            
            # Ends/coends to geometric
            ('End', 'GeometricTransformer'),
            ('Coend', 'TopologicalEmbedding'),
            ('Topos', 'CategoricalProbability')
        ]
        
        G.add_edges_from(hierarchical_edges)
        
        # Create hierarchical layout
        pos = {}
        level_width = {1: 3, 2: 4, 3: 4, 4: 4, 5: 3}
        level_nodes = {level: [n for n, d in G.nodes(data=True) if d['level'] == level] 
                      for level in range(1, 6)}
        
        for level in range(1, 6):
            nodes = level_nodes[level]
            if nodes:
                y = -level * 0.8  # Vertical spacing
                width = level_width[level]
                x_positions = np.linspace(-width/2, width/2, len(nodes))
                for i, node in enumerate(nodes):
                    pos[node] = (x_positions[i], y)
        
        # Draw nodes with level-based colors
        for node in G.nodes():
            x, y = pos[node]
            color = G.nodes[node]['color']
            level = G.nodes[node]['level']
            
            # Size based on level (higher levels are smaller)
            size = 0.3 - (level - 1) * 0.04
            
            circle = Circle((x, y), size, color=color, alpha=0.8, edgecolor='white', linewidth=1)
            ax.add_patch(circle)
            
            # Add labels
            label = node.replace('Functor', '\nFunctor').replace('Extension', '\nExtension')
            ax.text(x, y, label, ha='center', va='center', 
                   fontsize=6, color='white', weight='bold')
        
        # Draw edges
        for edge in G.edges():
            x1, y1 = pos[edge[0]]
            x2, y2 = pos[edge[1]]
            ax.plot([x1, x2], [y1, y2], 'white', alpha=0.4, linewidth=1)
            
            # Add arrow
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx_norm, dy_norm = dx/length, dy/length
                arrow_start_x = x1 + dx_norm * 0.3
                arrow_start_y = y1 + dy_norm * 0.3
                arrow_end_x = x2 - dx_norm * 0.3
                arrow_end_y = y2 - dy_norm * 0.3
                ax.annotate('', xy=(arrow_end_x, arrow_end_y), xytext=(arrow_start_x, arrow_start_y),
                           arrowprops=dict(arrowstyle='->', color='white', alpha=0.6, lw=1))
        
        # Add level labels
        level_labels = {
            1: 'Fundamental',
            2: 'Coalgebras', 
            3: 'Kan Extensions',
            4: 'Ends/Coends',
            5: 'Geometric'
        }
        
        for level, label in level_labels.items():
            ax.text(-3, -level * 0.8, label, ha='left', va='center',
                   fontsize=8, color='yellow', weight='bold')
        
        # Check what's actually in the model
        actual_components = []
        if hasattr(self.model, 'components'):
            actual_components = list(self.model.components.keys())
        
        # Add model status indicator
        ax.text(3, -1, f'Model Components:\n{len(actual_components)} found', 
               ha='left', va='top', fontsize=7, color='cyan',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-5, 1)
        ax.set_facecolor('black')
        ax.axis('off')
        
        logger.info(f"🏗️ CATEGORICAL HIERARCHY VISUALIZED:")
        logger.info(f"   • Total categorical structures: {len(G.nodes())}")
        logger.info(f"   • Hierarchical levels: {len(level_labels)}")
        logger.info(f"   • Actual model components: {len(actual_components)}")
        logger.info(f"   • Fuzzy-Simplicial integration: {'Present' if any('fuzzy' in c.lower() and 'simplicial' in c.lower() for c in actual_components) else 'Not detected'}")
    
    def visualize_business_hierarchy(self):
        """Visualize model component hierarchy and data flow."""
        ax = self.axes[0, 2]
        ax.clear()
        ax.set_title('Component Hierarchy', fontsize=12, color='white')
        
        # Create hierarchical component structure
        G = nx.DiGraph()
        
        # Extract real model hierarchy
        units = {}
        level = 0
        
        # Root model
        total_params = sum(p.numel() for p in self.model.parameters())
        units['GAIA_Model'] = {'level': level, 'size': min(1000, max(200, total_params // 1000))}
        level += 1
        
        # Major components
        if hasattr(self.model, 'gaia_transformer'):
            transformer_params = sum(p.numel() for p in self.model.gaia_transformer.parameters())
            units['Transformer'] = {'level': level, 'size': min(800, max(150, transformer_params // 1000))}
        
        if hasattr(self.model, 'categorical_ops'):
            units['Categorical'] = {'level': level, 'size': 400}
        
        if hasattr(self.model, 'position_embeddings'):
            pos_params = sum(p.numel() for p in self.model.position_embeddings.parameters())
            units['Embeddings'] = {'level': level, 'size': min(600, max(100, pos_params // 100))}
        
        level += 1
        
        # Sub-components (if accessible)
        if hasattr(self.model, 'gaia_transformer') and hasattr(self.model.gaia_transformer, 'layers'):
            layer_count = len(self.model.gaia_transformer.layers) if hasattr(self.model.gaia_transformer.layers, '__len__') else 6
            units[f'Layers_x{layer_count}'] = {'level': level, 'size': 300}
        
        for unit, attrs in units.items():
            G.add_node(unit, **attrs)
        
        # Add data flow relationships
        hierarchy = []
        unit_names = list(units.keys())
        
        # Connect root to major components
        if 'GAIA_Model' in unit_names:
            for unit in unit_names[1:]:
                if unit != 'GAIA_Model':
                    hierarchy.append(('GAIA_Model', unit))
        
        # Connect transformer to layers if both exist
        if 'Transformer' in unit_names and any('Layers' in u for u in unit_names):
            layer_unit = next(u for u in unit_names if 'Layers' in u)
            hierarchy.append(('Transformer', layer_unit))
        
        G.add_edges_from(hierarchy)
        
        # Hierarchical layout
        pos = nx.spring_layout(G)
        
        # Draw with size based on hierarchy level
        node_sizes = [units[node]['size'] for node in G.nodes()]
        node_colors = [units[node]['level'] for node in G.nodes()]
        
        nx.draw(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
                cmap=plt.cm.plasma, with_labels=True, font_size=8,
                font_color='white', edge_color='white', arrows=True)
        
        ax.set_facecolor('black')
    
    def visualize_simplicial_complex(self):
        """Visualize the complete fuzzy simplicial hierarchy and complexes."""
        ax = self.axes[1, 0]
        ax.clear()
        ax.set_title('Fuzzy Simplicial Complex Hierarchy', fontsize=12, color='white')
        
        # Create 3D-like visualization of simplicial levels
        # Level 0: 0-simplices (vertices)
        # Level 1: 1-simplices (edges) 
        # Level 2: 2-simplices (triangles)
        # Level 3: 3-simplices (tetrahedra)
        
        levels = {
            0: {'name': '0-Simplices\n(Vertices)', 'count': 8, 'color': '#ff6b6b'},
            1: {'name': '1-Simplices\n(Edges)', 'count': 12, 'color': '#4ecdc4'},
            2: {'name': '2-Simplices\n(Triangles)', 'count': 6, 'color': '#45b7d1'},
            3: {'name': '3-Simplices\n(Tetrahedra)', 'count': 2, 'color': '#96ceb4'}
        }
        
        # Extract actual simplicial data from model if available
        if hasattr(self.model, 'components'):
            for name, comp in self.model.components.items():
                if 'simplicial' in name.lower():
                    # Try to extract dimension info
                    if hasattr(comp, 'max_dimension'):
                        max_dim = getattr(comp, 'max_dimension', 3)
                        levels[max_dim]['name'] += f'\n[{name}]'
        
        # Draw simplicial complex as layered structure
        y_positions = [0.8, 0.3, -0.2, -0.7]  # Different levels
        
        for level, y_pos in enumerate(y_positions):
            if level in levels:
                level_data = levels[level]
                count = level_data['count']
                color = level_data['color']
                name = level_data['name']
                
                # Draw simplices at this level
                if level == 0:  # Vertices
                    x_positions = np.linspace(-1.2, 1.2, count)
                    for x in x_positions:
                        circle = Circle((x, y_pos), 0.05, color=color, alpha=0.8)
                        ax.add_patch(circle)
                
                elif level == 1:  # Edges
                    x_positions = np.linspace(-1.0, 1.0, count//2)
                    for i, x in enumerate(x_positions):
                        # Draw edge as line segment
                        x1, x2 = x - 0.1, x + 0.1
                        ax.plot([x1, x2], [y_pos, y_pos], color=color, linewidth=3, alpha=0.8)
                        if i < len(x_positions) - 1:
                            # Connect to next edge
                            next_x = x_positions[i + 1] - 0.1
                            ax.plot([x2, next_x], [y_pos, y_pos], color=color, linewidth=1, alpha=0.4)
                
                elif level == 2:  # Triangles
                    x_positions = np.linspace(-0.8, 0.8, count//2)
                    for x in x_positions:
                        # Draw triangle
                        triangle_x = [x-0.1, x+0.1, x, x-0.1]
                        triangle_y = [y_pos-0.05, y_pos-0.05, y_pos+0.1, y_pos-0.05]
                        ax.plot(triangle_x, triangle_y, color=color, linewidth=2, alpha=0.8)
                        ax.fill(triangle_x[:-1], triangle_y[:-1], color=color, alpha=0.3)
                
                elif level == 3:  # Tetrahedra (projected)
                    x_positions = np.linspace(-0.4, 0.4, count)
                    for x in x_positions:
                        # Draw tetrahedron projection
                        # Base triangle
                        base_x = [x-0.08, x+0.08, x, x-0.08]
                        base_y = [y_pos-0.05, y_pos-0.05, y_pos+0.08, y_pos-0.05]
                        ax.plot(base_x, base_y, color=color, linewidth=2, alpha=0.8)
                        # Apex connections
                        apex_x, apex_y = x, y_pos + 0.15
                        for bx, by in zip(base_x[:-1], base_y[:-1]):
                            ax.plot([bx, apex_x], [by, apex_y], color=color, linewidth=1, alpha=0.6)
                        ax.scatter([apex_x], [apex_y], color=color, s=30, alpha=0.8)
                
                # Add level label
                ax.text(-1.5, y_pos, name, ha='left', va='center', 
                       fontsize=8, color='white', weight='bold')
                ax.text(1.5, y_pos, f'n={count}', ha='right', va='center',
                       fontsize=7, color=color, weight='bold')
        
        # Draw face maps (connections between levels)
        for level in range(3):
            y1 = y_positions[level]
            y2 = y_positions[level + 1]
            
            # Draw face map arrows
            for i in range(3):
                x = -0.8 + i * 0.8
                ax.annotate('', xy=(x, y2 + 0.1), xytext=(x, y1 - 0.1),
                           arrowprops=dict(arrowstyle='->', color='yellow', alpha=0.6, lw=1))
        
        # Add face map labels
        ax.text(0, 0.05, '∂₁', ha='center', va='center', fontsize=10, color='yellow', weight='bold')
        ax.text(0, -0.45, '∂₂', ha='center', va='center', fontsize=10, color='yellow', weight='bold')
        ax.text(0, -0.95, '∂₃', ha='center', va='center', fontsize=10, color='yellow', weight='bold')
        
        # Add fuzzy membership indicators
        membership_levels = [0.9, 0.7, 0.5, 0.3]
        for i, (level, membership) in enumerate(zip(range(4), membership_levels)):
            y = y_positions[i]
            # Membership bar
            bar_width = membership * 0.3
            ax.barh(y - 0.15, bar_width, height=0.02, left=1.7, 
                   color='cyan', alpha=0.8)
            ax.text(2.1, y - 0.15, f'μ={membership}', ha='left', va='center',
                   fontsize=6, color='cyan')
        
        # Add title for membership
        ax.text(1.9, 1.0, 'Fuzzy\nMembership', ha='center', va='center',
               fontsize=7, color='cyan', weight='bold')
        
        # Extract actual fuzzy simplicial data from model
        fuzzy_info = []
        if hasattr(self.model, 'components'):
            for name, comp in self.model.components.items():
                if 'fuzzy' in name.lower() and 'simplicial' in name.lower():
                    fuzzy_info.append(name)
        
        # Add model info
        info_text = f"Fuzzy Simplicial Components:\n{len(fuzzy_info)} found"
        if fuzzy_info:
            info_text += f"\n• {fuzzy_info[0][:20]}..."
        
        ax.text(-1.5, -1.2, info_text, ha='left', va='top', fontsize=6, color='white',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-1.5, 1.2)
        ax.set_facecolor('black')
        ax.axis('off')
        
        logger.info(f"🔺 FUZZY SIMPLICIAL COMPLEX VISUALIZED:")
        logger.info(f"   • Simplicial levels: 0-3 (vertices to tetrahedra)")
        logger.info(f"   • Face maps: ∂₁, ∂₂, ∂₃")
        logger.info(f"   • Fuzzy memberships: {membership_levels}")
        logger.info(f"   • Model fuzzy simplicial components: {len(fuzzy_info)}")
    
    def visualize_message_flow(self):
        """Visualize data flow through model components."""
        ax = self.axes[1, 1]
        ax.clear()
        ax.set_title('Data Flow', fontsize=12, color='white')
        
        # Create data flow network based on actual model
        G = nx.DiGraph()
        
        # Extract real data flow from model architecture
        nodes = {}
        flows = []
        
        # Input node
        nodes['Input'] = {'pos': (0, 0), 'type': 'input'}
        
        # Position embeddings
        if hasattr(self.model, 'position_embeddings'):
            nodes['Pos_Emb'] = {'pos': (1, 0.5), 'type': 'embedding'}
            flows.append(('Input', 'Pos_Emb', {'weight': 0.8}))
        
        # GAIA Transformer layers
        if hasattr(self.model, 'gaia_transformer'):
            nodes['Transformer'] = {'pos': (2, 0), 'type': 'transformer'}
            if 'Pos_Emb' in nodes:
                flows.append(('Pos_Emb', 'Transformer', {'weight': 0.9}))
            else:
                flows.append(('Input', 'Transformer', {'weight': 0.9}))
        
        # Categorical operations
        if hasattr(self.model, 'categorical_ops'):
            nodes['Cat_Ops'] = {'pos': (2, -0.5), 'type': 'categorical'}
            if 'Transformer' in nodes:
                flows.append(('Transformer', 'Cat_Ops', {'weight': 0.7}))
        
        # Output processing
        output_pos = (3, 0)
        if 'Cat_Ops' in nodes and 'Transformer' in nodes:
            nodes['Combine'] = {'pos': (2.5, -0.25), 'type': 'aggregate'}
            flows.append(('Transformer', 'Combine', {'weight': 0.8}))
            flows.append(('Cat_Ops', 'Combine', {'weight': 0.6}))
            nodes['Output'] = {'pos': output_pos, 'type': 'output'}
            flows.append(('Combine', 'Output', {'weight': 1.0}))
        elif 'Transformer' in nodes:
            nodes['Output'] = {'pos': output_pos, 'type': 'output'}
            flows.append(('Transformer', 'Output', {'weight': 1.0}))
        else:
            nodes['Output'] = {'pos': output_pos, 'type': 'output'}
            flows.append(('Input', 'Output', {'weight': 1.0}))
        
        # Add nodes and edges to graph
        for node, attrs in nodes.items():
            G.add_node(node, **attrs)
        
        for source, target, attrs in flows:
            G.add_edge(source, target, **attrs)
        
        # Extract positions and draw network
        pos = {node: attrs['pos'] for node, attrs in nodes.items()}
        
        # Color nodes by type
        node_colors = {
            'input': '#ff6b6b',
            'embedding': '#4ecdc4', 
            'transformer': '#45b7d1',
            'categorical': '#96ceb4',
            'aggregate': '#ffeaa7',
            'output': '#fd79a8'
        }
        
        colors = [node_colors.get(nodes[node]['type'], '#ffffff') for node in G.nodes()]
        
        # Draw network
        nx.draw(G, pos, ax=ax, node_color=colors, node_size=400,
                with_labels=True, font_size=8, font_color='white',
                edge_color='white', arrows=True, width=2)
        
        ax.set_facecolor('black')
        ax.axis('off')
    
    def visualize_component_inspector(self):
        """Real-time component inspector showing actual model statistics."""
        ax = self.axes[1, 2]
        ax.clear()
        ax.set_title('Component Inspector', fontsize=12, color='white')
        
        # Extract real component statistics
        components = []
        param_counts = []
        memory_usage = []
        
        # Analyze actual model components
        if hasattr(self.model, 'position_embeddings'):
            pos_params = sum(p.numel() for p in self.model.position_embeddings.parameters())
            components.append('Pos_Emb')
            param_counts.append(pos_params)
            memory_usage.append(pos_params * 4 / (1024*1024))  # Approximate MB (float32)
        
        if hasattr(self.model, 'gaia_transformer'):
            trans_params = sum(p.numel() for p in self.model.gaia_transformer.parameters())
            components.append('Transformer')
            param_counts.append(trans_params)
            memory_usage.append(trans_params * 4 / (1024*1024))
        
        if hasattr(self.model, 'categorical_ops'):
            try:
                cat_params = sum(p.numel() for p in self.model.categorical_ops.parameters())
                components.append('Cat_Ops')
                param_counts.append(cat_params)
                memory_usage.append(cat_params * 4 / (1024*1024))
            except:
                # If categorical_ops doesn't have parameters method
                components.append('Cat_Ops')
                param_counts.append(1000)  # Estimate
                memory_usage.append(0.004)
        
        # Add total model stats
        total_params = sum(p.numel() for p in self.model.parameters())
        components.append('Total')
        param_counts.append(total_params)
        memory_usage.append(total_params * 4 / (1024*1024))
        
        if not components:
            ax.text(0.5, 0.5, 'No component data available', 
                   ha='center', va='center', color='white', transform=ax.transAxes)
            return
        
        # Create dual-axis visualization
        x = np.arange(len(components))
        width = 0.35
        
        # Parameters bar chart
        bars1 = ax.bar(x - width/2, param_counts, width, label='Parameters', 
                      color='#4ecdc4', alpha=0.8)
        
        # Memory usage on secondary axis
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, memory_usage, width, label='Memory (MB)', 
                       color='#ff6b6b', alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Components', color='white')
        ax.set_ylabel('Parameters', color='white')
        ax2.set_ylabel('Memory (MB)', color='white')
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars1, param_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:,}' if value < 1000000 else f'{value/1000000:.1f}M',
                   ha='center', va='bottom', color='white', fontsize=8)
        
        for bar, value in zip(bars2, memory_usage):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}MB',
                    ha='center', va='bottom', color='white', fontsize=8)
        
        # Legends and styling
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax2.tick_params(colors='white')
    
    def run_dynamic_visualization(self, input_text: str = "GAIA processes language through categorical structures"):
        """Run the dynamic visualization with model inference."""
        logger.info(f"🎯 Starting dynamic visualization with input: '{input_text}'")
        
        try:
            # Simple tokenization (no fuzzy encoding)
            tokens = input_text.split()[:3]  # Limit tokens
            input_ids = torch.randint(0, min(50, self.config.vocab_size), (1, len(tokens)))
            input_ids = input_ids.to(self.device)
            
            logger.info(f"📝 Input shape: {input_ids.shape}")
            
            # Model inference (bypassing fuzzy encoding)
            # Store input shape for tensor visualization
            self.last_input_shape = input_ids.shape
            
            with torch.no_grad():
                # Use the GAIA transformer directly
                if hasattr(self.model, 'gaia_transformer'):
                    # Get token embeddings from the transformer
                    transformer_output = self.model.gaia_transformer(input_ids)
                    if isinstance(transformer_output, dict):
                        logger.info(f"🔄 Transformer output keys: {list(transformer_output.keys())}")
                        if 'logits' in transformer_output:
                            logger.info(f"📊 Logits shape: {transformer_output['logits'].shape}")
                            self.last_output_shape = transformer_output['logits'].shape
                        if 'last_hidden_state' in transformer_output:
                            logger.info(f"🔤 Hidden state shape: {transformer_output['last_hidden_state'].shape}")
                            self.last_hidden_shape = transformer_output['last_hidden_state'].shape
                    else:
                        logger.info(f"🔄 Transformer output shape: {transformer_output.shape}")
                        self.last_output_shape = transformer_output.shape
                else:
                    logger.info("🔍 No gaia_transformer found, using simple forward pass")
                    # Fallback to basic model forward
                    output = self.model(input_ids)
                    logger.info(f"📤 Model output keys: {list(output.keys()) if isinstance(output, dict) else 'tensor'}")
                    if hasattr(output, 'shape'):
                        self.last_output_shape = output.shape
            
            # Update all visualizations
            self.visualize_model_architecture()
            self.visualize_categorical_structures()
            self.visualize_business_hierarchy()
            self.visualize_simplicial_complex()
            self.visualize_message_flow()
            self.visualize_component_inspector()
            
            plt.tight_layout()
            plt.savefig('simple_gaia_architecture.png', dpi=300, bbox_inches='tight', 
                       facecolor='black', edgecolor='none')
            logger.info("💾 Visualization saved to simple_gaia_architecture.png")
            
            logger.info("✅ Dynamic visualization completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error in dynamic visualization: {e}")
            raise
    
    def save_visualization(self, filename: str = "gaia_architecture.png"):
        """Save the current visualization to file."""
        if self.fig:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight', 
                           facecolor='black', edgecolor='none')
            logger.info(f"💾 Visualization saved to {filename}")

def main():
    """Main function to run the simple GAIA visualizer."""
    logger.info("🚀 Starting Simple GAIA Architecture Visualizer")
    
    try:
        # Create and run visualizer
        visualizer = SimpleGAIAVisualizer()
        visualizer.run_dynamic_visualization("The GAIA model processes language through categorical structures")
        
        # Save visualization
        visualizer.save_visualization("simple_gaia_architecture.png")
        
        logger.info("🎉 Simple GAIA Visualizer completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
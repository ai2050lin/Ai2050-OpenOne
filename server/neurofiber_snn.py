import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# NeuroFiber-SNN: Spiking Neural Network with 3D Geometry
# -----------------------------------------------------------------------------

@dataclass
class Impulse:
    timestamp: float
    weight: float

@dataclass
class NeuronState:
    id: str
    pos: Tuple[float, float, float]
    v: float
    fired: bool

class LIFNeuron:
    def __init__(self, neuron_id, pos=(0.0, 0.0, 0.0), tau=20.0, threshold=1.0, reset_potential=0.0):
        self.id = neuron_id
        self.pos = pos
        self.tau = tau
        self.dt = 1.0
        self.threshold = threshold
        self.reset = reset_potential
        self.v = self.reset
        self.spikes = [] 
        self.incoming_spikes = []

    def step(self, t):
        input_current = sum(imp.weight for imp in self.incoming_spikes)
        self.incoming_spikes = []
        
        decay = 1.0 - (self.dt / self.tau)
        self.v = self.v * decay + input_current
        
        fired = False
        if self.v >= self.threshold:
            self.v = self.reset
            self.spikes.append(t)
            fired = True
            
        return fired

class NeuroFiberNetwork:
    def __init__(self):
        self.neurons: Dict[str, List[LIFNeuron]] = {}
        self.connections = [] # (src_id, tgt_id, weight, path_coords)
        self.time = 0.0
        
    def add_layer_grid(self, name, rows, cols, center=(0,0,0), spacing=1.0):
        """Creates a 2D grid layer (e.g., Retina)"""
        layer_neurons = []
        start_x = center[0] - (cols * spacing) / 2
        start_y = center[1] - (rows * spacing) / 2
        z = center[2]
        
        for r in range(rows):
            for c in range(cols):
                nid = f"{name}_{r}_{c}"
                x = start_x + c * spacing
                y = start_y + r * spacing
                pos = (x, y, z)
                layer_neurons.append(LIFNeuron(nid, pos=pos))
        
        self.neurons[name] = layer_neurons

    def add_layer_bundle(self, name, size, center=(0,0,0), radius=2.0, length=5.0):
        """Creates a cylindrical bundle layer (e.g., Fiber)"""
        layer_neurons = []
        for i in range(size):
            nid = f"{name}_{i}"
            # Distribution in a circle (cross-section of bundle)
            theta = (2 * math.pi * i) / size
            r = radius * 0.8  # Slight offset from edge
            x = center[0] + r * math.cos(theta)
            y = center[1] + r * math.sin(theta)
            z = center[2] 
            
            # For visualization, maybe we want them spread along Z a bit? 
            # Or just a cross-section. Let's keep them as a cross-section ring for now.
            pos = (x, y, z)
            layer_neurons.append(LIFNeuron(nid, pos=pos))
            
        self.neurons[name] = layer_neurons

    def add_layer(self, name, size):
        """Wrapper for add_layer_bundle with auto-positioning"""
        # Auto-position along X axis based on existing layer count
        x_pos = len(self.neurons) * 10.0
        self.add_layer_bundle(name, size, center=(x_pos, 0, 0))

    def connect_all_to_all(self, src_name, tgt_name, weight=0.5, delay=1.0):
        """Connects every neuron in src to every neuron in tgt"""
        srcs = self.neurons[src_name]
        tgts = self.neurons[tgt_name]
        
        for s in srcs:
            for t in tgts:
                self.connections.append({
                    "src": s,
                    "tgt": t,
                    "weight": weight,
                    "delay": delay
                })

    def connect_layers(self, src_name, tgt_name, probability=0.5, weight=0.5, delay=1.0):
        srcs = self.neurons[src_name]
        tgts = self.neurons[tgt_name]
        
        for s in srcs:
            for t in tgts:
                if random.random() < probability:
                    self.connections.append({
                        "src": s,
                        "tgt": t,
                        "weight": weight,
                        "delay": delay
                    })

    def connect_one_to_one(self, src_name, tgt_name, weight=0.8, delay=1.0):
        """Connects ith neuron of src to ith neuron of tgt"""
        srcs = self.neurons[src_name]
        tgts = self.neurons[tgt_name]
        limit = min(len(srcs), len(tgts))
        
        for i in range(limit):
             self.connections.append({
                "src": srcs[i],
                "tgt": tgts[i],
                "weight": weight,
                "delay": delay
            })

    def inject_stimulus(self, layer_name, index, intensity=1.5):
        if layer_name in self.neurons:
            neurons = self.neurons[layer_name]
            if 0 <= index < len(neurons):
                neurons[index].incoming_spikes.append(Impulse(self.time, intensity))

    def step_simulation(self):
        self.time += 1.0
        fired_info = {} # Dict[str, List[int]]
        
        # Step neurons
        active_neurons = []
        for name, layer in self.neurons.items():
            fired_indices = []
            for i, neuron in enumerate(layer):
                if neuron.step(self.time):
                    fired_indices.append(i)
                    active_neurons.append(neuron)
            if fired_indices:
                fired_info[name] = fired_indices
        
        # Propagate
        for conn in self.connections:
            src = conn["src"]
            if src in active_neurons:
                tgt = conn["tgt"]
                tgt.incoming_spikes.append(Impulse(self.time + conn["delay"], conn["weight"]))
                
        return fired_info

    def get_structure(self):
        """Returns JSON-serializable structure for frontend"""
        neurons_json = []
        for name, layer in self.neurons.items():
            for neuron in layer:
                neurons_json.append({
                    "id": neuron.id,
                    "layer": name,
                    "pos": neuron.pos
                })
        
        connections_json = []
        for conn in self.connections:
            connections_json.append({
                "srcId": conn["src"].id,
                "tgtId": conn["tgt"].id,
                "weight": conn["weight"]
            })
            
        return {
            "neurons": neurons_json,
            "connections": connections_json
        }

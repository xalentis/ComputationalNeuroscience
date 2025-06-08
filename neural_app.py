# Gideon Vos June 2025
# James Cook University
# www.linkedin.com/in/gideonvos
#
# pip install matplotlib networkx numpy
# compile the C++ library first: python setup.py build_ext --inplace
# rename compiled library to neural_cpp.so

import sys
import time
import random
import traceback
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx


random.seed(42)
np.random.seed(42)

try:
    import neural_cpp 
except ImportError:
    print("Error: neural_cpp module not found!")
    sys.exit(1)


class NeuralNetworkVisualizer:
    def __init__(self, network_size=50, enable_pruning=True, network_type="fully_connected"):
        self.network_size = network_size
        self.enable_pruning = enable_pruning
        self.network_type = network_type
        self.neuron_positions = {}
        
        if network_type == "small_world":
            self.network = neural_cpp.create_small_world_network(network_size, 0.1, 6)
        elif network_type == "fully_connected":
            self.network = self.create_fully_connected_network(network_size)
        else:
            self.network = neural_cpp.create_random_network(network_size, 0.3)
        
        # parameters
        self.pruning_params = neural_cpp.PruningParams()
        self.pruning_params.activity_threshold = 0.3    # higher = more aggressive
        self.pruning_params.weight_threshold = 0.3      # higher = more aggressive
        self.pruning_params.activity_decay_rate = 0.98  # higher = faster decay
        self.pruning_params.evaluation_interval = 50    # lower = more frequent pruning
        self.pruning_params.enable_activity_pruning = True
        self.pruning_params.enable_weight_pruning = True
        self.pruning_history = []
        self.synapse_count_history = []
        self.voltage_history = []
        self.time_steps = []
        self.step_count = 0
        self.last_pruning_step = 0
        self.update_times = []
        self.render_times = []
        self.setup_visualization()
        
    def setup_visualization(self):
        self.fig = plt.figure(figsize=(20, 12))
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax1.set_title("Neural Network Connectivity")
        self.ax1.set_aspect('equal')
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax2.set_title("Neuron Voltages Over Time")
        self.ax2.set_xlabel("Time Steps")
        self.ax2.set_ylabel("Voltage (mV)")
        self.ax2.grid(True, alpha=0.3)
        self.ax3 = self.fig.add_subplot(gs[0, 2])
        self.ax3.set_title("Network Statistics")
        self.ax3.set_xlabel("Time Steps")
        self.ax3.set_ylabel("Count")
        self.ax3.grid(True, alpha=0.3)
        self.ax4 = self.fig.add_subplot(gs[1, 0])
        self.ax4.set_title("Synaptic Pruning Events")
        self.ax4.set_xlabel("Time Steps")
        self.ax4.set_ylabel("Synapses Pruned")
        self.ax4.grid(True, alpha=0.3)
        self.ax5 = self.fig.add_subplot(gs[1, 1])
        self.ax5.set_title("Performance Metrics")
        self.ax5.set_xlabel("Time Steps")
        self.ax5.set_ylabel("Time (ms)")
        self.ax5.grid(True, alpha=0.3)
        self.ax6 = self.fig.add_subplot(gs[1, 2])
        self.ax6.set_title("Current Network State")
        self.ax6.axis('off')
        self.setup_neuron_positions()
        
    def setup_neuron_positions(self):
        # circular layout
        angles = np.linspace(0, 2 * np.pi, self.network_size, endpoint=False)
        radius = 1.0
        for i in range(self.network_size):
            self.neuron_positions[i] = (
                radius * np.cos(angles[i]),
                radius * np.sin(angles[i])
            )
    
    def get_network_graph(self):
        G = nx.DiGraph()
        for i in range(self.network_size):
            try:
                neuron = self.network.get_neuron(i)
                voltage = neuron.voltage()
                is_spiking = neuron.is_spiking()
                G.add_node(i, voltage=voltage, spiking=is_spiking)
            except Exception as e:
                print(f"Warning: Could not get neuron {i}: {e}")
                G.add_node(i, voltage=-70.0, spiking=False)
        
        for i in range(self.network_size):
            try:
                neuron = self.network.get_neuron(i)
                synapse_count = neuron.get_synapse_count()
                
                for j in range(synapse_count):
                    try:
                        synapse = neuron.get_synapse(j)
                        if synapse.get_active():
                            target_id = synapse.target_id
                            weight = synapse.weight
                            synapse_type = synapse.get_type()
                            
                            edge_color = 'red' if synapse_type == neural_cpp.SynapseType.INHIBITORY else 'blue'
                            G.add_edge(i, target_id, weight=weight, color=edge_color)
                    except Exception as e:
                        print(f"Warning: Could not get synapse {j} from neuron {i}: {e}")
            except Exception as e:
                print(f"Warning: Could not process neuron {i}: {e}")
        
        return G
    
    def create_fully_connected_network(self, n_neurons):
        network = neural_cpp.NeuralNetwork()
        for i in range(n_neurons):
            network.add_neuron()
        
        # connect every neuron to every other neuron
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i != j:  # no self-connections
                    # random weight between 0.1 and 2.0
                    weight = 0.1 + np.random.random() * 1.9
                    # 20% chance of inhibitory connection
                    synapse_type = neural_cpp.SynapseType.INHIBITORY if np.random.random() < 0.2 else neural_cpp.SynapseType.EXCITATORY
                    network.connect_neurons(i, j, weight, synapse_type)
        return network
    
    def update_network(self):
        start_time = time.perf_counter()
        
        # apply random stimulation to some neurons to
        stimulation_neurons = np.random.choice(self.network_size, size=3, replace=False)
        stimulation_current = 15.0 + np.random.normal(0, 5.0)
        
        for neuron_id in stimulation_neurons:
            self.network.stimulate_neuron(int(neuron_id), stimulation_current)
        
        self.network.update()
        
        # apply pruning if enabled
        pruned_count = 0
        if self.enable_pruning and self.step_count > 0 and (self.step_count % self.pruning_params.evaluation_interval == 0):
            pruned_count = self.network.prune_synapses(self.pruning_params)
            if pruned_count > 0:
                self.pruning_history.append((self.step_count, pruned_count))
                self.last_pruning_step = self.step_count
        
        # record network statistics
        stats = self.network.get_synapse_stats()
        self.synapse_count_history.append(stats.active_synapses)
        
        # record voltages of first few neurons for plotting
        voltages = []
        for i in range(min(5, self.network_size)):
            try:
                neuron = self.network.get_neuron(i)
                voltages.append(neuron.voltage())
            except:
                voltages.append(-70.0)
        
        self.voltage_history.append(voltages)
        self.time_steps.append(self.step_count)
        
        # limit history length for performance
        max_history = 500
        if len(self.time_steps) > max_history:
            self.time_steps = self.time_steps[-max_history:]
            self.voltage_history = self.voltage_history[-max_history:]
            self.synapse_count_history = self.synapse_count_history[-max_history:]
        
        self.step_count += 1
        
        update_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        self.update_times.append(update_time)
        if len(self.update_times) > 100:
            self.update_times = self.update_times[-100:]
        
        return pruned_count
    
    def animate(self, frame):
        render_start = time.perf_counter()
        pruned_count = self.update_network() # result not used currently, call is required
        
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
            ax.clear()
        
        # network connectivity
        self.ax1.set_title(f"Neural Network Connectivity (Step {self.step_count})")
        self.ax1.set_aspect('equal')
        G = self.get_network_graph()
        
        # draw nodes colored by voltage
        node_colors = []
        node_sizes = []
        
        for i in range(self.network_size):
            if i in G.nodes:
                voltage = G.nodes[i]['voltage']
                is_spiking = G.nodes[i]['spiking']
                # color based on voltage (blue = hyperpolarized, red = depolarized)
                voltage_norm = (voltage + 80) / 60  # Normalize roughly -80 to -20
                voltage_norm = max(0, min(1, voltage_norm))
                if is_spiking:
                    node_colors.append('yellow')
                    node_sizes.append(300)
                else:
                    node_colors.append(plt.cm.RdYlBu_r(voltage_norm))
                    node_sizes.append(200)
            else:
                node_colors.append('gray')
                node_sizes.append(100)
        
        nx.draw_networkx_nodes(G, self.neuron_positions, node_color=node_colors, node_size=node_sizes, ax=self.ax1)
        
        # draw active edges
        if G.edges():
            edge_colors = [G[u][v]['color'] for u, v in G.edges()]
            edge_weights = [abs(G[u][v]['weight']) * 2 for u, v in G.edges()]
            nx.draw_networkx_edges(G, self.neuron_positions,
                                 edge_color=edge_colors,
                                 width=edge_weights,
                                 alpha=0.6,
                                 arrows=True,
                                 arrowsize=10,
                                 ax=self.ax1)
        
        nx.draw_networkx_labels(G, self.neuron_positions, 
                              font_size=8, 
                              font_color='white',
                              ax=self.ax1)
        
        # voltage traces
        self.ax2.set_title("Neuron Voltages Over Time")
        self.ax2.set_xlabel("Time Steps")
        self.ax2.set_ylabel("Voltage (mV)")
        self.ax2.grid(True, alpha=0.3)
        
        if len(self.voltage_history) > 1:
            voltage_array = np.array(self.voltage_history)
            for i in range(min(5, voltage_array.shape[1])):
                self.ax2.plot(self.time_steps, voltage_array[:, i], 
                            label=f'Neuron {i}', alpha=0.8)
            self.ax2.legend()
            self.ax2.set_ylim(-80, 20)
        
        # network statistics
        self.ax3.set_title("Network Statistics")
        self.ax3.set_xlabel("Time Steps")
        self.ax3.set_ylabel("Count")
        self.ax3.grid(True, alpha=0.3)
        
        if len(self.synapse_count_history) > 1:
            self.ax3.plot(self.time_steps, self.synapse_count_history, 'g-', label='Active Synapses', linewidth=2)
            self.ax3.legend()
        
        # pruning events
        self.ax4.set_title("Synaptic Pruning Events")
        self.ax4.set_xlabel("Time Steps")
        self.ax4.set_ylabel("Synapses Pruned")
        self.ax4.grid(True, alpha=0.3)
        
        if self.pruning_history:
            steps, counts = zip(*self.pruning_history)
            self.ax4.bar(steps, counts, alpha=0.7, color='red', width=5)
        
        # performance metrics
        self.ax5.set_title("Performance Metrics")
        self.ax5.set_xlabel("Recent Steps")
        self.ax5.set_ylabel("Time (ms)")
        self.ax5.grid(True, alpha=0.3)
        if len(self.update_times) > 1:
            recent_steps = list(range(len(self.update_times)))
            self.ax5.plot(recent_steps, self.update_times, 'b-', label=f'Update Time (avg: {np.mean(self.update_times):.2f}ms)')
            self.ax5.legend()
        self.ax6.clear()
        self.ax6.set_title("Current Network State")
        self.ax6.axis('off')
        
        try:
            stats = self.network.get_synapse_stats()
            info_text = f"""
Network Size: {self.network_size} neurons
Total Synapses: {stats.total_synapses}
Active Synapses: {stats.active_synapses}
Average Activity: {stats.avg_activity:.4f}
Average Weight: {stats.avg_weight:.4f}

Simulation Step: {self.step_count}
Last Pruning: Step {self.last_pruning_step}
Pruning Enabled: {self.enable_pruning}

Network Type: {self.network_type}
            """
            
            self.ax6.text(0.05, 0.95, info_text, transform=self.ax6.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        except Exception as e:
            self.ax6.text(0.05, 0.95, f"Error getting stats: {e}", 
                         transform=self.ax6.transAxes, fontsize=10, 
                         verticalalignment='top')
        
        render_time = (time.perf_counter() - render_start) * 1000
        self.render_times.append(render_time)
        if len(self.render_times) > 100:
            self.render_times = self.render_times[-100:]
        
        avg_update = np.mean(self.update_times) if self.update_times else 0
        avg_render = np.mean(self.render_times) if self.render_times else 0
        self.fig.suptitle(f'Neural Network Simulation (Update: {avg_update:.1f}ms, Render: {avg_render:.1f}ms)')
    
    def run(self, interval=50):        
        try:
            anim = animation.FuncAnimation(self.fig, self.animate, interval=interval, blit=False)
            plt.tight_layout()
            plt.show()
        except KeyboardInterrupt:
            print("\nStopping visualization...")
            plt.close('all')


def main():
    parser = argparse.ArgumentParser(description='Neural Network Visualization with Pruning')
    parser.add_argument('--size', type=int, default=20, help='Network size (default: 20)')
    parser.add_argument('--no-pruning', action='store_true', help='Disable synaptic pruning')
    parser.add_argument('--network-type', choices=['random', 'small_world'], default='random', help='Network topology type')
    parser.add_argument('--interval', type=int, default=50, help='Animation interval in ms (default: 50)')
    args = parser.parse_args()
    
    try:
        visualizer = NeuralNetworkVisualizer(network_size=args.size, enable_pruning=not args.no_pruning, network_type=args.network_type)
        visualizer.run(interval=args.interval)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
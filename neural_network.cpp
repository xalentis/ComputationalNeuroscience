// neural_network.cpp
#include "neural_network.h"

// Implementation is mostly in header for template optimization
// This file can contain additional utility functions if needed

extern "C" {
    // C interface for Python binding
    NeuralNetwork* create_network() {
        return new NeuralNetwork();
    }
    
    void delete_network(NeuralNetwork* net) {
        delete net;
    }
    
    uint16_t add_neuron(NeuralNetwork* net, float threshold, float resting, float decay) {
        return net->add_neuron(threshold, resting, decay);
    }
    
    void connect_neurons(NeuralNetwork* net, uint16_t source, uint16_t target, float weight, int type) {
        SynapseType synapse_type = (type == 0) ? SynapseType::EXCITATORY : SynapseType::INHIBITORY;
        net->connect_neurons(source, target, weight, synapse_type);
    }
    
    void stimulate_neuron(NeuralNetwork* net, uint16_t id, float current) {
        net->stimulate_neuron(id, current);
    }
    
    void update_network(NeuralNetwork* net) {
        net->update();
    }
    
    float get_neuron_voltage(NeuralNetwork* net, uint16_t id) {
        Neuron* neuron = net->get_neuron(id);
        return neuron ? neuron->voltage() : 0.0f;
    }
    
    bool is_neuron_spiking(NeuralNetwork* net, uint16_t id) {
        Neuron* neuron = net->get_neuron(id);
        return neuron ? neuron->is_spiking() : false;
    }
    
    size_t get_network_size(NeuralNetwork* net) {
        return net->size();
    }
    
    // Get synapse information for visualization
    int get_synapses(NeuralNetwork* net, uint16_t neuron_id, uint16_t* targets, float* weights, int* types, int max_synapses) {
        Neuron* neuron = net->get_neuron(neuron_id);
        if (!neuron) return 0;
        
        const auto& synapses = neuron->synapses();
        int count = 0;
        
        for (const auto& synapse : synapses) {
            if (count >= max_synapses) break;
            targets[count] = synapse.target_id;
            weights[count] = synapse.weight;
            types[count] = (synapse.get_type() == SynapseType::EXCITATORY) ? 0 : 1;
            count++;
        }
        
        return count;
    }
}
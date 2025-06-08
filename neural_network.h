#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

enum class SynapseType : uint8_t {
    EXCITATORY = 0,
    INHIBITORY = 1
};

class Neuron;

// Compact synapse structure with pruning support
struct Synapse {
    uint16_t target_id;        // Target neuron ID (2 bytes)
    float weight;              // Synaptic weight (4 bytes)
    float activity_trace;      // Activity-based trace for pruning (4 bytes)
    float eligibility_trace;   // For learning (4 bytes)
    uint32_t last_active_step; // Last step when synapse was active (4 bytes)
    SynapseType type : 1;      // 1 bit for type
    bool active : 1;           // 1 bit for active state
    uint8_t padding : 6;       // Padding for alignment
    
    Synapse(uint16_t id, float w, SynapseType t) 
        : target_id(id), weight(w), activity_trace(0.0f), eligibility_trace(0.0f),
          last_active_step(0), type(t), active(true), padding(0) {}
    
    // Getters for bit-field members (needed for Python binding)
    SynapseType get_type() const { return type; }
    bool get_active() const { return active; }
    void set_active(bool a) { active = a; }
    
    // Updated methods for pruning and learning
    float get_activity_trace() const { return activity_trace; }
    float get_eligibility_trace() const { return eligibility_trace; }
    
    void update_traces(float activity_decay, float eligibility_decay, uint32_t current_step, bool was_used) {
        // Decay traces
        activity_trace *= activity_decay;
        eligibility_trace *= eligibility_decay;
        
        // Update if synapse was used
        if (was_used && active) {
            activity_trace += 1.0f;
            eligibility_trace += 1.0f;
            last_active_step = current_step;
        }
    }
    
    // Hebbian-style learning update
    void update_weight(float pre_spike, float post_spike, float learning_rate) {
        if (!active) return;
        
        // Hebbian learning: strengthen if both neurons are active
        float delta_weight = learning_rate * pre_spike * post_spike * eligibility_trace;
        
        // Also include weight decay to prevent runaway growth
        float weight_decay = -0.001f * weight;
        
        weight += delta_weight + weight_decay;
        
        // Clamp weights to reasonable bounds
        if (type == SynapseType::EXCITATORY) {
            weight = std::max(0.0f, std::min(5.0f, weight));
        } else {
            weight = std::max(-5.0f, std::min(0.0f, weight));
        }
    }
};

// Compact dendrite structure
struct Dendrite {
    uint16_t source_id;        // Source neuron ID
    float conductance;         // Dendritic conductance
    
    Dendrite(uint16_t id, float cond) : source_id(id), conductance(cond) {}
};

class Neuron {
private:
    uint16_t id_;
    float voltage_;            // Current membrane voltage
    float threshold_;          // Spike threshold
    float resting_potential_;  // Resting voltage
    float decay_rate_;         // Voltage decay rate
    bool spiking_;            // Current spike state
    float spike_trace_;       // Exponentially decaying spike trace for learning
    
    std::vector<Synapse> synapses_;    // Outgoing connections
    std::vector<Dendrite> dendrites_;  // Incoming connections
    
public:
    Neuron(uint16_t id, float threshold = -55.0f, float resting = -70.0f, float decay = 0.1f)
        : id_(id), voltage_(resting), threshold_(threshold), 
          resting_potential_(resting), decay_rate_(decay), spiking_(false), spike_trace_(0.0f) {
        synapses_.reserve(8);    // Pre-allocate for efficiency
        dendrites_.reserve(8);
    }
    
    // Add synapse (outgoing connection)
    void add_synapse(uint16_t target_id, float weight, SynapseType type) {
        synapses_.emplace_back(target_id, weight, type);
    }
    
    // Add dendrite (incoming connection)
    void add_dendrite(uint16_t source_id, float conductance = 1.0f) {
        dendrites_.emplace_back(source_id, conductance);
    }
    
    // Receive input from connected neuron
    void receive_input(float input, float conductance = 1.0f) {
        voltage_ += input * conductance;
    }
    
    // Update neuron state with learning
    bool update(uint32_t current_step) {
        // Decay spike trace
        spike_trace_ *= 0.95f;
        
        // Voltage decay towards resting potential
        voltage_ += (resting_potential_ - voltage_) * decay_rate_;
        
        // Check for spike
        bool was_spiking = spiking_;
        spiking_ = voltage_ >= threshold_;
        
        // Reset voltage after spike and update spike trace
        if (spiking_ && !was_spiking) {
            voltage_ = resting_potential_ + 30.0f; // Spike peak
            spike_trace_ = 1.0f; // Reset spike trace
            
            // Update all outgoing synapses' traces
            for (auto& synapse : synapses_) {
                synapse.update_traces(0.99f, 0.95f, current_step, true);
            }
            
            return true; // New spike occurred
        } else if (was_spiking && !spiking_) {
            voltage_ = resting_potential_ - 10.0f; // Hyperpolarization
        }
        
        // Update traces even when not spiking
        for (auto& synapse : synapses_) {
            synapse.update_traces(0.99f, 0.95f, current_step, false);
        }
        
        return false;
    }
    
    // Apply learning to all synapses
    void apply_learning(const std::vector<float>& post_spike_traces, float learning_rate = 0.01f) {
        for (auto& synapse : synapses_) {
            if (synapse.target_id < post_spike_traces.size()) {
                float post_trace = post_spike_traces[synapse.target_id];
                synapse.update_weight(spike_trace_, post_trace, learning_rate);
            }
        }
    }
    
    // Getters
    uint16_t id() const { return id_; }
    float voltage() const { return voltage_; }
    bool is_spiking() const { return spiking_; }
    float spike_trace() const { return spike_trace_; }
    const std::vector<Synapse>& synapses() const { return synapses_; }
    const std::vector<Dendrite>& dendrites() const { return dendrites_; }
    
    // Non-const access for pruning (needed internally)
    std::vector<Synapse>& get_synapses_mutable() { return synapses_; }
    
    // Setters
    void set_voltage(float v) { voltage_ = v; }
};

class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Neuron>> neurons_;
    uint32_t current_step_;  // Step counter for pruning
    
public:
    NeuralNetwork() : current_step_(0) {
        neurons_.reserve(1000); // Pre-allocate for efficiency
    }
    
    // Pruning parameters structure
    struct PruningParams {
        float activity_threshold;    // Minimum activity to keep synapse
        float weight_threshold;      // Minimum absolute weight to keep
        float activity_decay_rate;   // How fast activity traces decay
        uint32_t evaluation_interval; // Steps between pruning evaluations
        bool enable_activity_pruning; // Enable activity-based pruning
        bool enable_weight_pruning;   // Enable weight-based pruning
        
        PruningParams() : activity_threshold(0.1f), weight_threshold(0.01f),
                         activity_decay_rate(0.99f), evaluation_interval(100),
                         enable_activity_pruning(true), enable_weight_pruning(true) {}
    };
    
    // Synapse statistics structure
    struct SynapseStats {
        size_t total_synapses;
        size_t active_synapses;
        float avg_activity;
        float avg_weight;
        
        SynapseStats() : total_synapses(0), active_synapses(0), avg_activity(0.0f), avg_weight(0.0f) {}
    };
    
    // Add neuron and return its ID
    uint16_t add_neuron(float threshold = -55.0f, float resting = -70.0f, float decay = 0.1f) {
        uint16_t id = static_cast<uint16_t>(neurons_.size());
        neurons_.push_back(std::make_unique<Neuron>(id, threshold, resting, decay));
        return id;
    }
    
    // Connect two neurons
    void connect_neurons(uint16_t source_id, uint16_t target_id, float weight, SynapseType type) {
        if (source_id < neurons_.size() && target_id < neurons_.size()) {
            neurons_[source_id]->add_synapse(target_id, weight, type);
            neurons_[target_id]->add_dendrite(source_id);
        }
    }
    
    // Stimulate a neuron
    void stimulate_neuron(uint16_t id, float current) {
        if (id < neurons_.size()) {
            neurons_[id]->receive_input(current);
        }
    }
    
    // Update all neurons and propagate spikes
    void update() {
        current_step_++;  // Increment step counter
        
        std::vector<std::pair<uint16_t, float>> spike_signals;
        std::vector<float> spike_traces(neurons_.size(), 0.0f);
        
        // Update all neurons and collect spikes and spike traces
        for (size_t i = 0; i < neurons_.size(); ++i) {
            bool spiked = neurons_[i]->update(current_step_);
            spike_traces[i] = neurons_[i]->spike_trace();
            
            if (spiked) {
                // Neuron spiked, prepare to send signals
                auto& synapses = neurons_[i]->get_synapses_mutable();
                for (auto& synapse : synapses) {
                    if (synapse.get_active()) {
                        float signal = synapse.weight;
                        if (synapse.type == SynapseType::INHIBITORY) {
                            signal = -std::abs(signal); // Ensure inhibitory is negative
                        } else {
                            signal = std::abs(signal);  // Ensure excitatory is positive
                        }
                        spike_signals.emplace_back(synapse.target_id, signal);
                    }
                }
            }
        }
        
        // Propagate spike signals
        for (const auto& signal : spike_signals) {
            if (signal.first < neurons_.size()) {
                neurons_[signal.first]->receive_input(signal.second);
            }
        }
        
        // Apply learning to all neurons every few steps
        if (current_step_ % 5 == 0) {
            for (auto& neuron : neurons_) {
                neuron->apply_learning(spike_traces, 0.005f); // Small learning rate
            }
        }
    }


    // Perform synaptic pruning based on parameters
    size_t prune_synapses(const PruningParams& params) {
        size_t pruned_count = 0;
        
        for (auto& neuron : neurons_) {
            auto& synapses = neuron->get_synapses_mutable();
            
            // Remove synapses that meet pruning criteria
            auto it = std::remove_if(synapses.begin(), synapses.end(),
                [&params, &pruned_count, this](const Synapse& s) {
                    bool should_prune = false;
                    
                    // Activity-based pruning - more aggressive early on
                    if (params.enable_activity_pruning) {
                        float adaptive_threshold = params.activity_threshold;
                        // Make pruning more aggressive early in simulation
                        if (current_step_ < 1000) {
                            adaptive_threshold *= 2.0f; // Higher threshold = more pruning
                        }
                        
                        if (s.get_activity_trace() < adaptive_threshold) {
                            should_prune = true;
                        }
                    }
                    
                    // Weight-based pruning
                    if (params.enable_weight_pruning && 
                        std::abs(s.weight) < params.weight_threshold) {
                        should_prune = true;
                    }
                    
                    // Additional criterion: prune very old unused synapses
                    if (current_step_ > s.last_active_step + 200) {
                        should_prune = true;
                    }
                    
                    if (should_prune) {
                        pruned_count++;
                        return true;
                    }
                    return false;
                });
            
            synapses.erase(it, synapses.end());
        }
        
        return pruned_count;
    }

    // Get current step count
    uint32_t get_current_step() const { return current_step_; }
    
    // Get total synapse count
    size_t get_total_synapses() const {
        size_t total = 0;
        for (const auto& neuron : neurons_) {
            total += neuron->synapses().size();
        }
        return total;
    }
    
    // Get synapse statistics
    SynapseStats get_synapse_stats() const {
        SynapseStats stats;
        
        for (const auto& neuron : neurons_) {
            for (const auto& synapse : neuron->synapses()) {
                stats.total_synapses++;
                if (synapse.get_active()) {
                    stats.active_synapses++;
                }
                stats.avg_activity += synapse.get_activity_trace();
                stats.avg_weight += std::abs(synapse.weight);
            }
        }
        
        if (stats.total_synapses > 0) {
            stats.avg_activity /= stats.total_synapses;
            stats.avg_weight /= stats.total_synapses;
        }
        
        return stats;
    }
    
    // Get neuron by ID
    Neuron* get_neuron(uint16_t id) {
        return (id < neurons_.size()) ? neurons_[id].get() : nullptr;
    }
    
    // Get network size
    size_t size() const { return neurons_.size(); }
    

};

#endif // NEURAL_NETWORK_H


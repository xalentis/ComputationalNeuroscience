// neural_binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <memory>
#include "neural_network.h"

namespace py = pybind11;

PYBIND11_MODULE(neural_cpp, m) {
    m.doc() = "High-performance neural network simulation library with synaptic pruning";

    // Synapse type enum
    py::enum_<SynapseType>(m, "SynapseType")
        .value("EXCITATORY", SynapseType::EXCITATORY)
        .value("INHIBITORY", SynapseType::INHIBITORY);

    // Synapse structure
    py::class_<Synapse>(m, "Synapse")
        .def(py::init<uint16_t, float, SynapseType>())
        .def_readonly("target_id", &Synapse::target_id)
        .def_readonly("weight", &Synapse::weight)
        .def("get_type", &Synapse::get_type)
        .def("get_active", &Synapse::get_active)
        .def("set_active", &Synapse::set_active)
        .def("get_activity_trace", &Synapse::get_activity_trace)
        .def("get_eligibility_trace", &Synapse::get_eligibility_trace)
        .def("update_traces", &Synapse::update_traces)
        .def("update_weight", &Synapse::update_weight);

    // Dendrite structure
    py::class_<Dendrite>(m, "Dendrite")
        .def(py::init<uint16_t, float>())
        .def_readonly("source_id", &Dendrite::source_id)
        .def_readonly("conductance", &Dendrite::conductance);

    // PruningParams structure
    py::class_<NeuralNetwork::PruningParams>(m, "PruningParams")
        .def(py::init<>())
        .def_readwrite("activity_threshold", &NeuralNetwork::PruningParams::activity_threshold,
                      "Minimum activity level to keep synapse")
        .def_readwrite("weight_threshold", &NeuralNetwork::PruningParams::weight_threshold,
                      "Minimum absolute weight to keep synapse")
        .def_readwrite("activity_decay_rate", &NeuralNetwork::PruningParams::activity_decay_rate,
                      "Decay rate for activity traces (0-1)")
        .def_readwrite("evaluation_interval", &NeuralNetwork::PruningParams::evaluation_interval,
                      "Steps between pruning evaluations")
        .def_readwrite("enable_activity_pruning", &NeuralNetwork::PruningParams::enable_activity_pruning,
                      "Enable activity-based pruning")
        .def_readwrite("enable_weight_pruning", &NeuralNetwork::PruningParams::enable_weight_pruning,
                      "Enable weight-based pruning");

    // SynapseStats structure
    py::class_<NeuralNetwork::SynapseStats>(m, "SynapseStats")
        .def_readonly("total_synapses", &NeuralNetwork::SynapseStats::total_synapses,
                     "Total number of synapses")
        .def_readonly("active_synapses", &NeuralNetwork::SynapseStats::active_synapses,
                     "Number of active synapses")
        .def_readonly("avg_activity", &NeuralNetwork::SynapseStats::avg_activity,
                     "Average activity level")
        .def_readonly("avg_weight", &NeuralNetwork::SynapseStats::avg_weight,
                     "Average absolute weight");

    // Neuron class
    py::class_<Neuron>(m, "Neuron")
        .def("add_synapse", &Neuron::add_synapse,
             py::arg("target_id"), py::arg("weight"), py::arg("type"),
             "Add outgoing synapse to target neuron")
        .def("add_dendrite", &Neuron::add_dendrite,
             py::arg("source_id"), py::arg("conductance") = 1.0f,
             "Add incoming dendrite from source neuron")
        .def("receive_input", &Neuron::receive_input,
             py::arg("input"), py::arg("conductance") = 1.0f,
             "Receive input current")
        .def("update", &Neuron::update,
             "Update neuron state, returns True if spike occurred")
        .def("id", &Neuron::id, "Get neuron ID")
        .def("voltage", &Neuron::voltage, "Get current membrane voltage")
        .def("is_spiking", &Neuron::is_spiking, "Check if neuron is currently spiking")
        .def("get_synapse_count", [](const Neuron& n) {
            return n.synapses().size();
        }, "Get number of outgoing synapses")
        .def("get_synapse", [](const Neuron& n, size_t index) {
            const auto& synapses = n.synapses();
            if (index >= synapses.size()) {
                throw py::index_error("Synapse index out of range");
            }
            return synapses[index];
        }, py::arg("index"), "Get synapse by index")
        .def("get_dendrite_count", [](const Neuron& n) {
            return n.dendrites().size();
        }, "Get number of incoming dendrites")
        .def("get_dendrite", [](const Neuron& n, size_t index) {
            const auto& dendrites = n.dendrites();
            if (index >= dendrites.size()) {
                throw py::index_error("Dendrite index out of range");
            }
            return dendrites[index];
        }, py::arg("index"), "Get dendrite by index")
        .def("set_voltage", &Neuron::set_voltage,
             py::arg("voltage"), "Set membrane voltage")
        .def("spike_trace", &Neuron::spike_trace, "Get current spike trace")
        .def("apply_learning", [](Neuron& n, const std::vector<float>& traces, float lr) {
            n.apply_learning(traces, lr);
        }, py::arg("post_spike_traces"), py::arg("learning_rate") = 0.01f,
        "Apply learning to all synapses");


    // Neural Network class - Use shared_ptr to avoid copy constructor issues
    py::class_<NeuralNetwork, std::shared_ptr<NeuralNetwork>>(m, "NeuralNetwork")
        .def(py::init<>(), "Create new neural network")
        .def("add_neuron", &NeuralNetwork::add_neuron,
             py::arg("threshold") = -55.0f,
             py::arg("resting") = -70.0f,
             py::arg("decay") = 0.1f,
             "Add neuron with specified parameters, returns neuron ID")
        .def("connect_neurons", &NeuralNetwork::connect_neurons,
             py::arg("source_id"), py::arg("target_id"), py::arg("weight"), py::arg("type"),
             "Connect two neurons with specified weight and type")
        .def("stimulate_neuron", &NeuralNetwork::stimulate_neuron,
             py::arg("id"), py::arg("current"),
             "Apply current stimulus to specified neuron")
        .def("update", &NeuralNetwork::update,
             "Update all neurons and propagate spikes")
        .def("get_neuron", &NeuralNetwork::get_neuron,
             py::arg("id"), py::return_value_policy::reference_internal,
             "Get neuron by ID")
        .def("size", &NeuralNetwork::size,
             "Get number of neurons in network")
        
        // Pruning methods
        .def("prune_synapses", &NeuralNetwork::prune_synapses,
             py::arg("params"),
             "Prune synapses based on parameters, returns number of pruned synapses")
        .def("get_current_step", &NeuralNetwork::get_current_step,
             "Get current simulation step")
        .def("get_total_synapses", &NeuralNetwork::get_total_synapses,
             "Get total number of synapses in network")
        .def("get_synapse_stats", &NeuralNetwork::get_synapse_stats,
             "Get comprehensive synapse statistics");

    // Utility functions - Return shared_ptr instead of by value
    m.def("create_small_world_network", [](int n_neurons, double p_rewire, int k_neighbors) -> std::shared_ptr<NeuralNetwork> {
        auto network = std::make_shared<NeuralNetwork>();
        
        // Add neurons
        for (int i = 0; i < n_neurons; i++) {
            network->add_neuron();
        }
        
        // Create small-world connectivity
        for (int i = 0; i < n_neurons; i++) {
            for (int j = 1; j <= k_neighbors / 2; j++) {
                int target1 = (i + j) % n_neurons;
                int target2 = (i - j + n_neurons) % n_neurons;
                
                // Random rewiring
                if (static_cast<double>(rand()) / RAND_MAX < p_rewire) {
                    target1 = rand() % n_neurons;
                    target2 = rand() % n_neurons;
                }
                
                if (target1 != i) {
                    float weight = 0.5f + static_cast<float>(rand()) / RAND_MAX * 2.0f;
                    SynapseType type = (rand() % 5 == 0) ? SynapseType::INHIBITORY : SynapseType::EXCITATORY;
                    network->connect_neurons(i, target1, weight, type);
                }
                
                if (target2 != i && target2 != target1) {
                    float weight = 0.5f + static_cast<float>(rand()) / RAND_MAX * 2.0f;
                    SynapseType type = (rand() % 5 == 0) ? SynapseType::INHIBITORY : SynapseType::EXCITATORY;
                    network->connect_neurons(i, target2, weight, type);
                }
            }
        }
        
        return network;
    }, py::arg("n_neurons"), py::arg("p_rewire") = 0.1, py::arg("k_neighbors") = 6,
       "Create small-world network with specified parameters");

    m.def("create_random_network", [](int n_neurons, double connection_prob) -> std::shared_ptr<NeuralNetwork> {
        auto network = std::make_shared<NeuralNetwork>();
        
        // Add neurons
        for (int i = 0; i < n_neurons; i++) {
            network->add_neuron();
        }
        
        // Random connections
        for (int i = 0; i < n_neurons; i++) {
            for (int j = 0; j < n_neurons; j++) {
                if (i != j && static_cast<double>(rand()) / RAND_MAX < connection_prob) {
                    float weight = 0.5f + static_cast<float>(rand()) / RAND_MAX * 2.5f;
                    SynapseType type = (rand() % 5 == 0) ? SynapseType::INHIBITORY : SynapseType::EXCITATORY;
                    network->connect_neurons(i, j, weight, type);
                }
            }
        }
        
        return network;
    }, py::arg("n_neurons"), py::arg("connection_prob") = 0.3,
       "Create random network with specified connection probability");
}
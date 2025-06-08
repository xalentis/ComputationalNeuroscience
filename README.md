# Computational Neuroscience

Gideon Vos June 2025, James Cook University Australia, www.linkedin.com/in/gideonvos

compile the C++ library first: python setup.py build_ext --inplace

rename compiled library to neural_cpp.so

pip install matplotlib networkx numpy

## Understanding the Visualization:

Node Colors: Represent membrane voltage (purple = low, yellow = high)

Edge Colors: Blue = excitatory synapses, Red = inhibitory synapses

White Outlines: Currently spiking neurons

Voltage Plot: Shows membrane potential traces for first 5 neurons

## Key Features:

Random Stimulation: Network receives random input spikes

Realistic Dynamics: Neurons exhibit spike-and-reset behavior

Network Effects: Spikes propagate through synaptic connections

Inhibitory Control: Red connections provide network stability

![Demo](https://github.com/xalentis/ComputationalNeuroscience/blob/master/visual_demo.png)

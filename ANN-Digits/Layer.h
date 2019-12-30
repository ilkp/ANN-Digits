#pragma once
#include <vector>

class Layer
{
private:
	std::vector<std::vector<float>> _weights;
	std::vector<std::vector<float>> _deltaWeights;
	std::vector<float> _biases;
	std::vector<float> _deltaBiases;
	std::vector<float> _outputs;
	std::vector<float> _error;

public:
	int _layerSize;
	Layer* _prevLayer;
	Layer* _nextLayer;
	Layer(Layer* previousLayer, int layerSize);
	~Layer();

	void propagationForward();
	void propagationBackward();
	void calculateDelta();
	void update(float learningRate, int epochs);

	void setNextLayer(Layer* nextLayer) { _nextLayer = nextLayer; }
	void setOutputs(std::vector<float> outputs) { _outputs = outputs; }
	void setError(int node, float correctOutput) { _error.at(node) = _outputs.at(node) - correctOutput; }

	float getOutput(int node) { return _outputs.at(node); }
	std::vector<float> getOutput() { return _outputs; }
	float getWeight(int node, int w) { return _weights.at(node)[w]; }
};


#pragma once

#include "Layer.h"
#include "InputData.h"

class Network
{
private:
	Layer* _inputLayer;
	Layer* _outputLayer;
	void propagateForward();
	void propagateBackward(const std::vector<float> labels);

public:
	Network();
	~Network();
	void Init(int inputSize, int hiddenSize, int outputSize, int hiddenLayers);
	void Epoch(const std::vector<InputData> inputData, float learningRate);
	std::vector<float> Test(std::vector<float> inputData);
	void Clean();
};


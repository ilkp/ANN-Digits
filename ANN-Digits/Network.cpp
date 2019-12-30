#include "Network.h"



Network::Network()
{
}


Network::~Network()
{
	Clean();
}

void Network::Init(int inputSize, int hiddenSize, int outputSize, int hiddenLayers)
{
	_inputLayer = new Layer(nullptr, inputSize);
	Layer* lastLayer = _inputLayer;
	for (int i = 0; i < hiddenLayers; i++)
	{
		Layer* hiddenLayer = new Layer(lastLayer, hiddenSize);
		lastLayer->_nextLayer = hiddenLayer;
		lastLayer = hiddenLayer;
	}
	_outputLayer = new Layer(lastLayer, outputSize);
	lastLayer->_nextLayer = _outputLayer;
}

void Network::Epoch(const std::vector<InputData> inputData, float learningRate)
{
	Layer* l;
	for (int dataElement = 0; dataElement < inputData.size(); dataElement++)
	{
		_inputLayer->setOutputs(inputData.at(dataElement)._input);
		propagateForward();
		propagateBackward(inputData.at(dataElement)._label);
	}
	l = _outputLayer;
	while (l->_prevLayer != nullptr)
	{
		l->update(learningRate, inputData.size());
		l = l->_prevLayer;
	}
}

std::vector<float> Network::Test(std::vector<float> inputData)
{
	_inputLayer->setOutputs(inputData);
	propagateForward();
	return _outputLayer->getOutput();
}

void Network::propagateForward()
{
	Layer* l = _inputLayer->_nextLayer;
	while (l != nullptr)
	{
		l->propagationForward();
		l = l->_nextLayer;
	}
}

void Network::propagateBackward(const std::vector<float> labels)
{
	Layer* l = _outputLayer->_prevLayer;
	for (int i = 0; i < _outputLayer->_layerSize; i++)
	{
		_outputLayer->setError(i, labels.at(i));
	}
	_outputLayer->calculateDelta();

	while (l->_prevLayer != nullptr)
	{
		l->propagationBackward();
		l->calculateDelta();
		l = l->_prevLayer;
	}
}

void Network::Clean()
{
	if (!_outputLayer)
	{
		return;
	}

	Layer* l = _outputLayer->_prevLayer;
	while (l->_prevLayer != nullptr)
	{
		delete(l->_nextLayer);
	}
	delete(l);
}
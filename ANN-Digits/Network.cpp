#include "Network.h"



Network::Network()
{
}


Network::~Network()
{
	Clean();
}

void Network::Init(const int inputSize, const int hiddenSize, const int outputSize, const int hiddenLayers)
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

void Network::Epoch(const InputData* const inputData, const int inputSize, const float learningRate)
{
	Layer* l;
	for (int i = 0; i < inputSize; ++i)
	{
		_inputLayer->setOutputs(inputData[i]._input);
		propagateForward();
		propagateBackward(inputData[i]._label);
	}
	l = _outputLayer;
	while (l->_prevLayer != nullptr)
	{
		l->update(learningRate, inputSize);
		l = l->_prevLayer;
	}
}

float* Network::Test(const float* const inputData)
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

void Network::propagateBackward(const float* const labels)
{
	Layer* l = _outputLayer->_prevLayer;
	for (int i = 0; i < _outputLayer->_layerSize; i++)
	{
		_outputLayer->setError(i, labels[i]);
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
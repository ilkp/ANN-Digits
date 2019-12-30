#include <math.h>
#include <random>
#include "Layer.h"

#define SIGMOID(x) 1 / (1 + exp(-x))
#define SIGMOID_REVERSED(x) x * (1 - x) 

Layer::Layer(Layer* previousLayer, int layerSize) : _prevLayer(previousLayer), _layerSize(layerSize)
{
	for (int i = 0; i < layerSize; i++)
	{
		_outputs.push_back(0.0f);
		_error.push_back(0.0f);
		if (previousLayer != nullptr)
		{
			_weights.push_back(std::vector<float>());
			_deltaWeights.push_back(std::vector<float>());
			for (int j = 0; j < previousLayer->_layerSize; j++)
			{
				_weights.at(i).push_back(1.5f * (float(rand()) / float(RAND_MAX)) - 0.75f);
				_deltaWeights.at(i).push_back(0.0f);
			}
			_biases.push_back(1.5f * (float(rand()) / float(RAND_MAX)) - 0.75f);
			_deltaBiases.push_back(0.0f);
		}
	}
}

Layer::~Layer()
{
}

// Calculates output values for each node
void Layer::propagationForward()
{
	for (int i = 0; i < _layerSize; i++)
	{
		_outputs.at(i) = 0.0f;
		for (int j = 0; j < _prevLayer->_layerSize; j++)
		{
			_outputs.at(i) += _weights.at(i).at(j) * _prevLayer->_outputs.at(j);
		}
		_outputs.at(i) += _biases.at(i);
		_outputs.at(i) = SIGMOID(_outputs.at(i));
	}
}

// Calculates error value for each node
void Layer::propagationBackward()
{
	for (int i = 0; i < _layerSize; i++)
	{
		_error.at(i) = 0.0f;
	}
	for (int i = 0; i < _nextLayer->_layerSize; i++)
	{
		for (int k = 0; k < _layerSize; k++)
		{
			_error.at(k) += _nextLayer->_error.at(i) * _nextLayer->_weights.at(i).at(k);
		}
	}
	for (int i = 0; i < _layerSize; i++)
	{
		_error.at(i) = SIGMOID_REVERSED(_outputs.at(i)) * _error.at(i);
	}
}

void Layer::calculateDelta()
{
	for (int i = 0; i < _layerSize; i++)
	{
		_deltaBiases.at(i) += _error.at(i);
		for (int j = 0; j < _prevLayer->_layerSize; j++)
		{
			_deltaWeights.at(i).at(j) += _error.at(i) * _prevLayer->_outputs.at(j);
		}
	}
}

void Layer::update(float learningRate, int epochs)
{
	for (int i = 0; i < _layerSize; i++)
	{
		_biases.at(i) -= learningRate * _deltaBiases.at(i) / epochs;
		_deltaBiases.at(i) = 0.0f;
		for (int j = 0; j < _prevLayer->_layerSize; j++)
		{
			_weights.at(i).at(j) -= learningRate * _deltaWeights.at(i).at(j) / epochs;
			_deltaWeights.at(i).at(j) = 0.0f;
		}
	}
}
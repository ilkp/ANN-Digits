#pragma once
#include <vector>
struct InputData
{
	std::vector<float> _label;
	std::vector<float> _input;
	InputData() {};
	InputData(std::vector<float> label, std::vector<float> input)
	{
		_label = label;
		_input = input;
	};
};
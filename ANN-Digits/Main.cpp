

#include <iostream>
#include <string>
#include <fstream>
#include <time.h>
#include <stdlib.h>
#include <vector>
#include <iomanip>
#include <map>
#include <time.h>
#include <chrono>
#include <ctime>
#include <direct.h>
#include "Layer.h"
#include "InputData.h"
#include "Network.h"

#define ELEMENTS 1593
#define DATA_WIDTH 256
#define HIDDEN_LAYER_SIZE 10
#define LABEL_SIZE 10

std::vector<std::vector<float>> parseFile(const std::string fileName, const int elements, const int dataWidth);
void userNetworkSettings(int* layers, int* nodes, int* epochs);
void trainNetwork(const int epochs, Network* network, const std::vector<InputData> teachingData);
void testNetwork_digits(bool printValues, bool printInstances, Network* network, const std::vector<InputData> testingData);
float MSELoss(std::vector<float> output, std::vector<float> label);
int labelToInt(std::vector<float> label);
int networkOutputToInt(std::vector<float> output);
void initializeTrainingSets(const int digitsPerSet, const std::vector<std::vector<float>> data, std::vector<InputData>* trainingData, std::vector<InputData>* testingData);
void printAnswerTableInstances(int answerTable[10][10]);
void printAnswerTableValues(float answerTable[10][10]);



int main()
{
	bool programRunning = true;
	std::string quitMessage;
	int hiddenSize, layers, epochs;
	Network network;
	srand(time(NULL));
	std::vector<std::vector<float>> data = parseFile("semeion_sample.data", 400, DATA_WIDTH);
	if (data.size() == 0)
	{
		return -1;
	}

	std::vector<InputData> trainingData;
	std::vector<InputData> testingData;
	initializeTrainingSets(20, data, &trainingData, &testingData);
	while (programRunning)
	{
		userNetworkSettings(&layers, &hiddenSize, &epochs);

		std::cout << "Initializing network with " << layers << " hidden layers with " << hiddenSize << " nodes." << std::endl;
		network.Init(DATA_WIDTH, hiddenSize, 10, layers);
		std::cout << "\n------------------------------------------\n" << std::endl;
		std::cout << "Initial neural network results.\n" << std::endl;
		testNetwork_digits(true, true, &network, testingData);
		trainNetwork(epochs, &network, trainingData);
		testNetwork_digits(true, true, &network, testingData);
		std::cout << "Quit? (y/n): ";
		quitMessage = "";
		while (quitMessage.length() == 0 || !(quitMessage.at(0) == 'y' || quitMessage.at(0) == 'n'))
		{
			std::cin >> quitMessage;
		}
		if (quitMessage.at(0) == 'y')
		{
			programRunning = false;
			network.Clean();
		}
	}
	
	return 0;
}

// Training time will increase significantly with large values.
// Good values for testing, for example:
// 2 layers
// 50-100 nodes
// 500 epoch
void userNetworkSettings(int* layers, int* nodes, int* epochs)
{
	std::cout << "Enter number of hidden layers: ";
	std::cin >> *layers;
	while (std::cin.fail())
	{
		std::cin.clear();
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		std::cin >> *layers;
	}
	std::cout << "Enter number of nodes per hidden layer: ";
	std::cin >> *nodes;
	while (std::cin.fail())
	{
		std::cin.clear();
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		std::cin >> *nodes;
	}
	std::cout << "Enter number of epochs: ";
	std::cin >> *epochs;
	while (std::cin.fail())
	{
		std::cin.clear();
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		std::cin >> *epochs;
	}
}

std::vector<std::vector<float>> parseFile(const std::string fileName, const int elements, const int dataWidth)
{
	char cCurrentPath[FILENAME_MAX];
	std::string path = _getcwd(cCurrentPath, sizeof(cCurrentPath));
	std::vector<std::vector<float>> data;
	std::string line;
	std::ifstream file;
	file.open(path + "\\" + fileName);
	if (!file.is_open())
	{
		std::cout << "Failed to open file" << std::endl;
		return data;
	}

	for (int i = 0; i < elements; i++)
	{
		std::vector<float> dataElement;
		std::getline(file, line);
		for (int j = 0; j < dataWidth; j++)
		{
			dataElement.push_back(float(line.at(j * 2) - '0'));
		}
		data.push_back(dataElement);
	}
	file.close();
	return data;
}

/*
 Each epoch goes through the entire teaching data set and calculates delta values for networks weights and biases
 After and epoch, networks values are adjusted
 Possible implementations in the future:
	Momentum
	Batch training
*/
void trainNetwork(int epochs, Network* network, const std::vector<InputData> teachingData)
{
	int seconds, minutes, hours;
	const int testEpochs = 3;
	float learningSpeed = 0.1f;
	std::chrono::duration<double> dT;
	auto start = std::chrono::system_clock::now();
	struct tm newtime;
	time_t now = std::time(0);
	localtime_s(&newtime, &now);
	std::cout << "Starting training at " <<
		newtime.tm_mday << "." << newtime.tm_mon + 1 << "." << newtime.tm_year + 1900 << ", " <<
		newtime.tm_hour << ":" << (newtime.tm_min < 10 ? "0" : "") << newtime.tm_min << std::endl;

	// Estimate and print a time needed for the whole training run
	for (int i = 0; i < testEpochs && i <= epochs; i++)
	{
		network->Epoch(teachingData, learningSpeed);
	}
	auto end = std::chrono::system_clock::now();
	dT = end - start;
	seconds = (int)(dT.count() / testEpochs * (epochs - testEpochs));
	minutes = seconds / 60;
	hours = minutes / 60;
	std::cout << "Estimated training time: " << hours << " hours, " << minutes % 60 << " minutes and " << seconds % 60 << " seconds" << std::endl;

	// Training runs
	for (int i = 0; i < epochs - testEpochs; i++)
	{
		network->Epoch(teachingData, learningSpeed);
	}
	end = std::chrono::system_clock::now();
	now = std::time(0);
	localtime_s(&newtime, &now);
	dT = end - start;
	seconds = (int)dT.count();
	minutes = seconds / 60;
	hours = minutes / 60;
	std::cout << "Finished training at " <<
		newtime.tm_mday << "." << newtime.tm_mon + 1 << "." << newtime.tm_year + 1900 << ", " <<
		newtime.tm_hour << ":" << (newtime.tm_min < 10 ? "0" : "") << newtime.tm_min <<
		" in " << hours << " hours, " << minutes % 60 << " minutes and " << seconds % 60 << " seconds\n" << std::endl;
}

// Note: hard coded for digit recognition
void testNetwork_digits(bool printValues, bool printInstances, Network* network, const std::vector<InputData> testingData)
{
	float loss = 0.0f;
	std::vector<float> networkOutput;
	int networksAnswer;
	int input;
	int answerTable[10][10];
	float answerTableValues[10][10];
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			answerTable[i][j] = 0;
			answerTableValues[i][j] = 0.0f;
		}
	}
	for (int i = 0; i < testingData.size(); i++)
	{
		networkOutput = network->Test(testingData.at(i)._input);
		networksAnswer = networkOutputToInt(networkOutput);
		input = labelToInt(testingData.at(i)._label);
		answerTable[input][networksAnswer]++;
		for (int j = 0; j < 10; j++)
		{
			answerTableValues[input][j] += networkOutput.at(j);
		}
		//loss += MSELoss(network->_outputLayer->getOutput(), testingData.at(i)._label);
	}
	loss /= testingData.size();
	if (printValues)
	{
		printAnswerTableValues(answerTableValues);
	}
	if (printInstances)
	{
		printAnswerTableInstances(answerTable);
	}
	std::cout << "------------------------------------------\n" << std::endl;
}

void printAnswerTableValues(float answerTable[10][10])
{
	std::cout << "Input\t\tNetwork's answer" << std::endl;
	std::cout << "\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\n" << std::endl;
	for (int i = 0; i < 10; i++)
	{
		std::cout << i << "\t";
		for (int j = 0; j < 10; j++)
		{
			std::cout << std::fixed << std::setprecision(4) << answerTable[i][j] / 20 << "\t";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

void printAnswerTableInstances(int answerTable[10][10])
{
	std::cout << "Input\t\tNetwork's answer" << std::endl;
	std::cout << "\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\n" << std::endl;
	for (int i = 0; i < 10; i++)
	{
		std::cout << i << "\t";
		for (int j = 0; j < 10; j++)
		{
			std::cout << answerTable[i][j] << "\t";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

int labelToInt(std::vector<float> label)
{
	for (int i = 0; i < label.size(); i++)
	{
		if (label.at(i) >= 1.0f)
		{
			return i;
		}
	}
	return -1;
}

int networkOutputToInt(std::vector<float> output)
{
	float largest = output.at(0);
	int n = 0;
	for (int i = 1; i < output.size(); i++)
	{
		if (output.at(i) > largest)
		{
			largest = output.at(i);
			n = i;
		}
	}
	return n;
}

float MSELoss(std::vector<float> output, std::vector<float> label)
{
	float loss = 0.0f;
	for (int i = 0; i < output.size(); i++)
	{
		loss += (output.at(i) - label.at(i)) * (output.at(i) - label.at(i));
	}
	loss *= 0.5f;
	return loss;
}

// digitsPerSet is the number of a single digit per set. For example semeion_sample.data has 40 of each digits, so we will want 20 digits for each data set.
void initializeTrainingSets(const int digitsPerSet, const std::vector<std::vector<float>> data, std::vector<InputData>* trainingData, std::vector<InputData>* testingData)
{
	std::vector<std::vector<float>> labels;
	labels.push_back(std::vector<float>{ 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f });
	labels.push_back(std::vector<float>{ 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f });
	labels.push_back(std::vector<float>{ 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f });
	labels.push_back(std::vector<float>{ 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f });
	labels.push_back(std::vector<float>{ 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f });
	labels.push_back(std::vector<float>{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f });
	labels.push_back(std::vector<float>{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f });
	labels.push_back(std::vector<float>{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f });
	labels.push_back(std::vector<float>{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f });
	labels.push_back(std::vector<float>{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f });


	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < digitsPerSet; j++)
		{
			trainingData->push_back(InputData{ labels.at(i), data.at(i * digitsPerSet * 2 + j) });
			testingData->push_back(InputData{ labels.at(i), data.at(i * digitsPerSet * 2 + digitsPerSet + j) });
		}
	}
}
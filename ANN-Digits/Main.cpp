#include <iostream>
#include <string>
#include <fstream>
#include <time.h>
#include <iomanip>
#include <time.h>
#include <chrono>
#include <ctime>
#include <direct.h>
#include <algorithm>
#include "Layer.h"
#include "InputData.h"
#include "Network.h"

#define ELEMENTS 400
#define DIGIT_PER_SET 20
#define DATA_WIDTH 256
#define LABEL_SIZE 10

bool parseFile(const std::string& fileName, float* const data);
void userNetworkSettings(int* const layers, int* const nodes, int* const epochs);
void trainNetwork(const int epochs, Network* const network, const InputData* const teachingSet, const int teachingSetSize);
void testNetwork_digits(const bool printValues, const bool printInstances, Network* const network, const InputData* const testingData, int testinDataSize);
//float MSELoss(std::vector<float> output, std::vector<float> label);
int labelToInteger(const float* const label);
int networkOutputToInteger(const float* const output);
void initializeTrainingSets(InputData* trainingSet, InputData* testingSet, float* const data, float labels[LABEL_SIZE][LABEL_SIZE]);
void printAnswerTableInstances(int answerTable[LABEL_SIZE][LABEL_SIZE]);
void printAnswerTableValues(float answerTable[LABEL_SIZE][LABEL_SIZE]);

int main()
{
	srand(time(NULL));
	int hiddenSize, layers, epochs;
	bool programRunning = true;
	float* data = new float[ELEMENTS * DATA_WIDTH];
	std::string quitMessage;
	Network network;

	float labels[LABEL_SIZE][LABEL_SIZE] =
	{
		{ 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f }
	};

	if (!parseFile("semeion_sample.data", data))
	{
		return -1;
	}

	InputData trainingSet[LABEL_SIZE * DIGIT_PER_SET];
	InputData testingSet[LABEL_SIZE * DIGIT_PER_SET];

	initializeTrainingSets(trainingSet, testingSet, data, labels);

	while (programRunning)
	{
		userNetworkSettings(&layers, &hiddenSize, &epochs);

		//DEBUG VALUES
		//layers = 5;
		//hiddenSize = 400;
		//epochs = 100;

		std::cout << "Initializing network with " << layers << " hidden layers with " << hiddenSize << " nodes." << std::endl;
		network.Init(DATA_WIDTH, hiddenSize, LABEL_SIZE, layers);
		std::cout << "\n------------------------------------------\n" << std::endl;
		std::cout << "Initial neural network results.\n" << std::endl;

		testNetwork_digits(true, true, &network, testingSet, LABEL_SIZE * DIGIT_PER_SET);
		trainNetwork(epochs, &network, trainingSet, ELEMENTS / 2);
		testNetwork_digits(true, true, &network, testingSet, LABEL_SIZE * DIGIT_PER_SET);

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
void userNetworkSettings(int* const layers, int* const nodes, int* const epochs)
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

bool parseFile(const std::string& fileName, float* const data)
{
	char cCurrentPath[FILENAME_MAX];
	std::string path = _getcwd(cCurrentPath, sizeof(cCurrentPath));
	std::string line;
	std::ifstream file;
	file.open(path + "\\" + fileName);
	if (!file.is_open())
	{
		std::cout << "Failed to open file" << std::endl;
		return false;
	}

	for (int i = 0; i < ELEMENTS; ++i)
	{
		std::getline(file, line);
		for (int j = 0; j < DATA_WIDTH; ++j)
		{
			data[i * DATA_WIDTH + j] = float(line.at(j) - '0');
		}
	}
	file.close();
	return true;
}

/*
 Each epoch goes through the entire teaching data set and calculates delta values for networks weights and biases
 After and epoch, networks values are adjusted
 Possible implementations in the future:
	Momentum
	Batch training
*/
void trainNetwork(const int epochs, Network* const network, const InputData* const teachingSet, const int teachingSetSize)
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
		network->Epoch(teachingSet, teachingSetSize, learningSpeed);
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
		network->Epoch(teachingSet, teachingSetSize, learningSpeed);
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
void testNetwork_digits(const bool printValues, const bool printInstances, Network* const network, const InputData* const testingSet, int testingSetSize)
{
	float loss = 0.0f;
	float* networkOutput;
	int networksAnswer;
	int input;
	int answerTable[LABEL_SIZE][LABEL_SIZE];
	float answerTableValues[LABEL_SIZE][LABEL_SIZE];
	for (int i = 0; i < LABEL_SIZE; ++i)
	{
		for (int j = 0; j < LABEL_SIZE; ++j)
		{
			answerTable[i][j] = 0;
			answerTableValues[i][j] = 0.0f;
		}
	}
	for (int i = 0; i < testingSetSize; ++i)
	{
		networkOutput = network->Test(testingSet[i]._input);
		networksAnswer = networkOutputToInteger(networkOutput);
		input = labelToInteger(testingSet[i]._label);
		answerTable[input][networksAnswer]++;
		for (int j = 0; j < LABEL_SIZE; j++)
		{
			answerTableValues[input][j] += networkOutput[j];
		}
		//loss += MSELoss(network->_outputLayer->getOutput(), testingData.at(i)._label);
	}
	loss /= testingSetSize;
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

void printAnswerTableValues(float answerTable[LABEL_SIZE][LABEL_SIZE])
{
	std::cout << "Input\t\tNetwork's answer" << std::endl;
	std::cout << "\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\n" << std::endl;
	for (int i = 0; i < LABEL_SIZE; ++i)
	{
		std::cout << i << "\t";
		for (int j = 0; j < LABEL_SIZE; ++j)
		{
			std::cout << std::fixed << std::setprecision(4) << answerTable[i][j] / 20 << "\t";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

void printAnswerTableInstances(int answerTable[LABEL_SIZE][LABEL_SIZE])
{
	std::cout << "Input\t\tNetwork's answer" << std::endl;
	std::cout << "\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\n" << std::endl;
	for (int i = 0; i < LABEL_SIZE; ++i)
	{
		std::cout << i << "\t";
		for (int j = 0; j < LABEL_SIZE; ++j)
		{
			std::cout << answerTable[i][j] << "\t";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

int labelToInteger(const float* const label)
{
	for (int i = 0; i < LABEL_SIZE; ++i)
	{
		if (label[i] >= 1.0f)
		{
			return i;
		}
	}
	return -1;
}

int networkOutputToInteger(const float* const output)
{
	float largest = output[0];
	int n = 0;
	for (int i = 1; i < LABEL_SIZE; ++i)
	{
		if (output[i] > largest)
		{
			largest = output[i];
			n = i;
		}
	}
	return n;
}

//float MSELoss(std::vector<float> output, std::vector<float> label)
//{
//	float loss = 0.0f;
//	for (int i = 0; i < output.size(); ++i)
//	{
//		loss += (output.at(i) - label.at(i)) * (output.at(i) - label.at(i));
//	}
//	loss *= 0.5f;
//	return loss;
//}

// digitsPerSet is the number of a single digit per set. For example semeion_sample.data has 40 of each digits, so we will want 20 digits for each data set.
void initializeTrainingSets(InputData* trainingSet, InputData* testingSet, float* const data, float labels[LABEL_SIZE][LABEL_SIZE])
{
	const unsigned int testingSetOffset = DATA_WIDTH * DIGIT_PER_SET;
	unsigned int digitStartPos;
	for (int i = 0; i < LABEL_SIZE; ++i)
	{
		digitStartPos = i * DATA_WIDTH * DIGIT_PER_SET * 2;

		for (int j = 0; j < DIGIT_PER_SET; ++j)
		{
			trainingSet[i * DIGIT_PER_SET + j]._input = &data[digitStartPos + j * DATA_WIDTH];
			testingSet[i * DIGIT_PER_SET + j]._input = &data[digitStartPos + testingSetOffset + j * DATA_WIDTH];

			trainingSet[i * DIGIT_PER_SET + j]._label = labels[i];
			testingSet[i * DIGIT_PER_SET + j]._label = labels[i];
		}
	}
}

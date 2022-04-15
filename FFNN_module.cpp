/**
This script defines 3 classes that serve as a library for building a simple feed forward neural network
A pibind11 wrapper is defined at the end to build a python DLL

The first class "SingleNeuron" is a Neuron class that represents a single neuron object and holds input and output connectivity/weights for each neuron
The second class "NeuralNet" contains member functions to build/instantiate a net of neurons with the inputed topology with neuron objects, feed input values through the net, get the resulting output values, and backpropogate the results
The third class defines the main callable functions which takes user input for topology, activation function type, number of epochs, and X/Y data, instantiates NeuralNet based on inputs, and trains the net with the data
*/

//Include the header file which defines our activation functions
#include "FFNN_headers.h"

//all matrices will be stored using the vector library. We only use basic mathematical functions fromt he cmath library
#include <vector>
#include <cmath>
//standard imports
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <string>

//Import pibind11 headers which we will use to build a wrapper for our main class
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
#include <pybind11/eval.h>

using namespace std;
namespace py = pybind11;

//Define a connection object to represent a single connection between two neurons with a weight and a delta weight for packprop 
struct NeuronConnection
{
	double weight;
	double deltaWeight;
};

//define a layer object to store an array of SingleNeuron objects
class SingleNeuron;
typedef vector<SingleNeuron> Layer;

//Class that defines the member functions and variables accosiated with the behavior of a single node in a network
/**
 * Sum numbers in a vector.
 *
 * This sum is the arithmetic sum, not some other kind of sum that only
 * mathematicians have heard of.
 *
 * @param values Container whose values are summed.
 * @return sum of `values`, or 0.0 if `values` is empty.
 */
class SingleNeuron
{
public:
	SingleNeuron(unsigned numOutputs, unsigned Index, string);
	//set/get functions for the member variable for the output from a neuron 
	void setOutputVal(double val) { NeuronOutput = val; }
	double getOutputVal(void) const { return NeuronOutput; }
	//initiate functions for adjusting the weights/pasing values through a neuron during FF and backpropogation
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
	//functions for the neuron activation functiona and the derivative of the activation function for training
	double activationFunction(double x);
	double activationFunctionDerivative(double x);
	//function to get a random weight  [0,1)
	static double randomWeight(void) { return rand() / double(RAND_MAX+1); }
	//function to sum the weights and added gradientweight from a neuron passed to the next layer of neurons, used during backpropogation
	double sumOutputError(const Layer &nextLayer) const;

	//Initiate learning rate for each backprop step rate 
	static double training_rate; // learning rate values between 0.0, 1.0
	//define a learning momentum that specifies how the effective learning rate changes based on the previous weight change
	static double learning_momentum; //  0.0, n

	//output member variable representing the output value of the neuron
	double NeuronOutput;
	//Neuron connection member variable defining all of the output weights/delta weights of a neuron
	vector<NeuronConnection> NeuronOutputWeights;
	//index for the neuron in a layer
	unsigned NeuronIndex;
	//Calculated gradient for backprop
	double gradient;
	//Type of activation function member variable
	string functionType;
};


//set learning rate and learning momentum
double SingleNeuron::training_rate = 0.15;
double SingleNeuron::learning_momentum = 0.5; 


//Node constructor- takes in the number of outputs, the neuron index in the layer, 
//and the activation function type, and initializes the vector of connection objects with random output weights
SingleNeuron::SingleNeuron(unsigned numOutputs, unsigned Index, string functionType_)
{
	for (unsigned c = 0; c < numOutputs; ++c)
	{
		NeuronOutputWeights.push_back(NeuronConnection());
		NeuronOutputWeights.back().weight = randomWeight();
	}
	functionType = functionType_;
	NeuronIndex = Index;
}

//activation fnction for a neuron object. The specific function used depends on the state of the member variable functionType
double SingleNeuron::activationFunction(double x)
{
	double output;
	if (functionType=="tanh") {output = tanh(x);}
	if (functionType=="relu") {output=reLu(x);}
	if (functionType=="sigmoid") { output = sigmoid(x);}
	return output;
}


//derivative of the activation function for backpop
double SingleNeuron::activationFunctionDerivative(double x)
{
	double output;
	if (functionType=="tanh") {output = 1.0 - x * x;}
	if (functionType=="relu") { output=DreLu(x);}
	if (functionType=="sigmoid") {output = Dsigmoid(x);}
	return output;
}

//member function for feeding values from the previous layer of neurons through the  neuron (i.e the activation function)
void SingleNeuron::feedForward(const Layer& prevLayer)
{
	double sum = 0.0;

	//Sum the output*weight from each neuron in the previous layer to ge the input to the neuron, 
	//then pass into activation function to get output
	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputVal() *
			prevLayer[n].NeuronOutputWeights[NeuronIndex].weight;
	}
	NeuronOutput = SingleNeuron::activationFunction(sum);
}


//member function to update the input weights to the neuron  (i.e. the output weights of a previous layer) based on the current gradient during backprop
void SingleNeuron::updateInputWeights(Layer &prevLayer)
{
	//loop through all of the neurons in the previous layer and update the outputweight and the delta output weight
	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		//retrieve the nth neuron in the previous layer
		SingleNeuron &neuron = prevLayer[n];
		//store the old deltaweight for the neuron
		double oldDeltaWeight = neuron.NeuronOutputWeights[NeuronIndex].deltaWeight;
		//update the output weight based on the current gradient, scale the amount we update by the learning rate and the product of the momentum and the previous weight
		double newDeltaWeight = training_rate * neuron.getOutputVal() * gradient + learning_momentum * oldDeltaWeight;
		//update the output weight and the delta output weight of the neuron
		neuron.NeuronOutputWeights[NeuronIndex].deltaWeight = newDeltaWeight;
		neuron.NeuronOutputWeights[NeuronIndex].weight += newDeltaWeight;
	}
}


//member function to sum the nodes contribution to the error of the next layer during backprop
double SingleNeuron::sumOutputError(const Layer &nextLayer) const
{
	double sum = 0.0;
	//loop thorugh each neuron output* the current gradient for each neuron in the next layer and sum the total error
	for (unsigned n=0; n < nextLayer.size() - 1; ++n)
	{
		sum += NeuronOutputWeights[n].weight * nextLayer[n].gradient;
	}
	return sum;
}

//Use the sumOutputError and the derivative of the activation function to calculate a new gradient for the neuron during backprop
void SingleNeuron::calcHiddenGradients(const Layer &nextLayer)
{

	double outputError = sumOutputError(nextLayer);
	gradient = outputError * SingleNeuron::activationFunctionDerivative(NeuronOutput);
}

//calculate the loss function for the output nerons and the associated gradient for the output neurons duing packprop
void SingleNeuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - NeuronOutput;
	gradient = delta * SingleNeuron::activationFunctionDerivative(NeuronOutput);
}




//Class that defines the member functions and variables associated with the behavior of a network of nodes
class NeuralNet
{
	public:
		//default constructor
		NeuralNet() {}
		//function to se the topology member vairable
		void setNetTopology(vector<double> inputTopology) {topology = inputTopology;}		
		//build net function to initialize the network of neuron objects 
		void BuildNet(string);
		//function to feed input values through the network
		void feedForward(const vector<double>& inputVals);
		//function to conduct one round of backprop
		void backProp(const vector<double>& targetVals);
		//function to retrieve the output values from the network
		void getResults(vector<double>& resultVals) const;
		//function to calculate the recent average error
		double getRecentAverageError(void) const { return recentAverageError; }

		//member variable that holds a matrix of node objects with the net topology
		vector<Layer> NetworkLayers; // NetworkLayers[layerindex][neuronindex]
		double RMSOutputError;
		double recentAverageError;
		double recentAverageSmoothingFactor;
		//member variable that holds a 1D array that specifies the width of each layer
		vector<double> topology;
};

//Function to build a network of nodes with the specified topology
void NeuralNet::BuildNet(string functionType)
{	
	//loop through each specified "layer" in the topology variable
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
	{
		//add a network of layers to our matrix of neurons
		NetworkLayers.push_back(Layer());
		//check to make sure we are not at the output layer
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
		//add specified number of neuron objects to each layer
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
		{
			NetworkLayers.back().push_back(SingleNeuron(numOutputs, neuronNum, functionType));
		}
		//add a bias node the layer and se the output value to 1
		NetworkLayers.back().back().setOutputVal(1.0);

	}
}

//member funciton to feed input values through the network
void NeuralNet::feedForward(const vector<double>  &inputVals)
{
	//Check that the vector of input values is the same size as the input layer
	assert(inputVals.size() == NetworkLayers[0].size() - 1);
	//set the output value of each input neuron as the value of the input
	for (unsigned i = 0; i < inputVals.size(); i++)
	{
		NetworkLayers[0][i].setOutputVal(inputVals[i]);
	}
	//loop through the rest of the layers
	for (unsigned layerNum = 1; layerNum < NetworkLayers.size(); ++layerNum)
	{
		//loop through each neuron in the layer and feed previous layers outputs through the neuron
		Layer& prevLayer = NetworkLayers[layerNum - 1];
		for (unsigned n = 0; n < NetworkLayers[layerNum].size() - 1; ++n)
		{
			NetworkLayers[layerNum][n].feedForward(prevLayer);
		}
	}
}

//function to retrieve the ouput values from the network (i.e. from the output nodes)
void NeuralNet::getResults(vector<double>& resultVals) const
{
	resultVals.clear();
	for (unsigned n = 0; n < NetworkLayers.back().size() - 1; ++n)
	{
		resultVals.push_back(NetworkLayers.back()[n].getOutputVal());
	}
}

//function to perform one round of backpropogation on the network based on the targets
void NeuralNet::backProp(const vector<double>& targetVals)
{
	//Calculate the root mean squared error for a neuron based on the current outputs and the targets
	Layer &outputLayer = NetworkLayers.back();
	RMSOutputError = 0.0;
	for (unsigned n = 0; n < outputLayer.size() - 1; n++)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		RMSOutputError += delta * delta;
	}
	RMSOutputError /= outputLayer.size() - 1;
	RMSOutputError = sqrt(RMSOutputError);

	//update our "recent average error" measure
	recentAverageError = (recentAverageError * recentAverageSmoothingFactor + RMSOutputError) / (recentAverageSmoothingFactor + 1.0);
	
	//loop through each output node and calculate the gradients
	for (unsigned n = 0; n < outputLayer.size() - 1; n++)
	{
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	//Loop through the rest of the layers 
	for (unsigned layerNum = NetworkLayers.size() - 2; layerNum > 0; --layerNum)
	{
		//loop through each neuron in the layer and update the gradients
		Layer& hiddenLayer = NetworkLayers[layerNum];
		Layer& nextLayer = NetworkLayers[layerNum + 1];
		for (unsigned n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
}
	//loop through all neurons in the network starting from outputs and update the connection weights
	for (unsigned layerNum = NetworkLayers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer& layer = NetworkLayers[layerNum];
		Layer& prevLayer = NetworkLayers[layerNum - 1];
		for (unsigned n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}


//class to build and trian a network based on input net parameters and input data and retrieve the resulting output
class FFNN_Builder {
	public:
		//default constrctor
		FFNN_Builder() {}

		//member functions to get/set the topology member variable
		void setTopology(vector<double>TopologyInput) {modelTopology = TopologyInput;}
		vector<double> getTopology() { return modelTopology; }

		//main function that builds the net, passes training data into the network, and trains the network
		void fitModel(vector<vector<double>>, vector<vector<double>>);

		//member function to conduct one rount of feed forward through the net and ge teh resulting outputs
		vector<double> predict(vector<double>);

		//Set/get functions for the number of epochs
		void setEpochs(int epochs_) { epochs=epochs_; }
		int getEpochs() { return epochs; }

		//set/get functions for the type of activation function (stored as a string)
		void setActivationFunction(string activationType_) {activationType=activationType_;}
		string getActivationFunction() { return activationType; }

		//member functions to retrieve our training data(necessary for out current silly serialization procedure. These will be removed when I update the pickling method)
		vector<vector<double>> getXData() { return x_data; }
		vector<vector<double>> getYData() { return y_data; }

		//member variable that holds our network object
		NeuralNet NeuralNetModel;
		//member variable that stores our topology variable (array of layer widths)
		vector<double> modelTopology;
		//member variable that stores our training data
		vector<vector<double>> x_data;
		vector<vector<double>> y_data;
		//member variable that stores the number of epochs
		int epochs=1;
		//member variable that holds the type of activation function for our network
		string activationType="sigmoid";
		
		//vector<double> inputVals, targetVals, resultVals;
};

void FFNN_Builder::fitModel(vector<vector<double>> x_data_, vector<vector<double>> y_data_) {
	
	
	//set the topology of our network object and build the network
	NeuralNetModel.setNetTopology(modelTopology);
	NeuralNetModel.BuildNet(activationType);
	//counter for the number of epochs that have passes
	int epoch = 0;
	//set our training data member variables (this is mainly for our silly serializaition method)
	x_data = x_data_;
	y_data = y_data_;
	//set our temporary variables
	vector<double> inputVals, targetVals, resultVals;
	vector<double> x_row;
	vector<double> y_row;

	//train the net on all training data for the specified number of epochs
	while (epoch < epochs)
	{
		//counter for the number samples we have passed through a training cycle
		int trainingPass = 0;
		//loop through all of the training data
		while (trainingPass != x_data.size() - 1)
		{
			//get the current row of features and target vals
			x_row = x_data[trainingPass];
			y_row = y_data[trainingPass];
			//check that the number of features matches the width of the input layer
			if (x_row.size() != modelTopology[0])
			{
				break;
			}
			//optionally print the target vals to the python console (I might add a verbose option to make it easy to turn this on and off)
			//Feed the current row of features through the network
			NeuralNetModel.feedForward(x_row);
			//get the network outputs
			NeuralNetModel.getResults(resultVals);
			//optionally print the target vals and the output vals to the python console (I might add a verbose option to make it easy to turn this on and off)
			//py::print("Output: ", resultVals);
			//py::print("Targets:", y_row);
			//pass our targetvalues into our backpropogation function for one round of backpropogation on the network
			NeuralNetModel.backProp(y_row);

			//optionally print the recent average error (again I may add a verbose option)
			//py::print("net recent average error:", NeuralNetModel.getRecentAverageError());

			//increment to the next row of data
			++trainingPass;
		}
		++epoch;

	}
	//print "done" to the python console when the training is complete
	py::print("Done");
}

//member funciton to pass new input values through the network and get the output values
vector<double> FFNN_Builder::predict(vector<double> x_test_row) {
	vector<double> y_predicted;
	NeuralNetModel.feedForward(x_test_row);
	NeuralNetModel.getResults(y_predicted);
	return y_predicted;
}

//wrapper funtion using pibind11 package to expose our FFNN_Builder class and its member functions to python via a DLL 
PYBIND11_MODULE(FFNN_pymodule, m) {
	py::class_<FFNN_Builder>(m, "FFNN_Builder")
		.def(py::init<>())
		.def("setTopology", &FFNN_Builder::setTopology)
		.def("setTopology", &FFNN_Builder::getTopology)
		.def("setTopology", &FFNN_Builder::getXData)
		.def("setTopology", &FFNN_Builder::getYData)
		.def("fitModel", &FFNN_Builder::fitModel)
		.def("predict", &FFNN_Builder::predict)
		.def("setEpochs", &FFNN_Builder::setEpochs)
		.def("setActivationFunction", &FFNN_Builder::setActivationFunction)

		//Pickle procedure needs to be updated with a procdeure that saves weights and reloads weights 
		//Currently it just saves input parameters to bodel building/training funciton and retrains the model when called
		//(pointless, i know, but i am working on a better serialization method)
		.def(py::pickle(
			[](FFNN_Builder& FFNN) { // __getstate__
				/* Return a tuple that fully encodes the state of the object */
				return py::make_tuple(FFNN.getTopology(), FFNN.getXData(), FFNN.getYData(), FFNN.getActivationFunction(), FFNN.getEpochs());
			},
			[](py::tuple t) { // __setstate__
				if (t.size() != 5)
					throw std::runtime_error("Invalid state!");

				/* Create a new C++ instance */
				FFNN_Builder FFNN;

				/* Assign any additional state */
				FFNN.setTopology(t[0].cast<vector<double>>());
				FFNN.setActivationFunction(t[3].cast<string>());
				FFNN.setEpochs(t[4].cast<int>());
				FFNN.fitModel(t[1].cast<vector<vector<double>>>(), t[2].cast<vector<vector<double>>>());

				return FFNN;
			}
			));
}





#pragma once

#include "crossbar.h"
#include <string>
#include <memory>

class crossbarLayer {
private:
	crossbar CROSSBAR;
	adc ADC;
	std::shared_ptr<activateFunc> activatePointer;
	bool useADC;
public:
	void quick_init(int inputTensorLength, int outputTensorLength, std::string activate = "none",std::string initialization = "normal") {
		int weightNum = inputTensorLength * outputTensorLength;
		std::vector<float> g, mean, sigma;
		g.resize(weightNum);
		mean.resize(weightNum);
		sigma.resize(weightNum);
		const float _N_MEAN = -10, _N_SIGMA = 0.1;
		const float setThreshold = 0, resetThreshold = 0;
		std::default_random_engine generator(time(NULL));
		std::normal_distribution<float> Ndistribution(0, 0.1);
		for (int i = 0; i < weightNum; ++i) {
			if (initialization == "normal") {
				g[i] = Ndistribution(generator);
			} else if (initialization == "zero") {
				g[i] = 0;
			}
			mean[i] = _N_MEAN;
			sigma[i] = _N_SIGMA;
		}
		CROSSBAR.init(inputTensorLength, outputTensorLength, g, mean, sigma, setThreshold, resetThreshold);
		if (activate == "relu") {
			activatePointer.reset(new reluF());
		}
		else if (activate == "softmax") {
			activatePointer.reset(new softmaxF());
		}
		else if (activate == "sigmoid") {
			activatePointer.reset(new sigmoidF());
		}
		else if (activate == "none") {
			activatePointer.reset(new noActivationF());
		}
		useADC = false;
	}

	std::vector<float> forward(const std::vector<float> &inputTensor) {
		std::vector<float> outputTensor = CROSSBAR.forward_calculate(inputTensor);
		if (useADC) {
			ADC.A2D(outputTensor);
		}
		outputTensor = activatePointer->func(outputTensor);
		return outputTensor;
	}

	std::vector<float> backward(const std::vector<float>& delta) {
		CROSSBAR.setDelta(delta);
		std::vector<float> frontDelta = CROSSBAR.getLastLayerDelta();
		CROSSBAR.calculateGrad(*activatePointer);
		CROSSBAR.updateG();
		return frontDelta;
	}
};

class crossbarModule {
private:
	std::vector<crossbarLayer> crossbarLayerVec;
	int LayerNum;

public:
	void init(std::vector<crossbarLayer> &crossbarLayerVec) {
		this->crossbarLayerVec.swap(crossbarLayerVec);
		LayerNum = this->crossbarLayerVec.size();
	}

	std::vector<float> forward(const std::vector<float> &inputTensor) {
		std::vector<float> tmpTensor = inputTensor;
		for (int i = 0; i < LayerNum; ++i) {
			std::vector<float> outputTensor = crossbarLayerVec[i].forward(tmpTensor);
			tmpTensor.swap(outputTensor);
		}
		return tmpTensor;
	}

	void backward(const std::vector<float>& delta) {
		std::vector<float> tmpDelta = delta;
		for (int i = LayerNum - 1; i != -1; --i) {
			std::vector<float> frontDelta = crossbarLayerVec[i].backward(tmpDelta);
			tmpDelta.swap(frontDelta);
		}
	}

	std::vector<float> deltaCalculate(const std::vector<float>& outputTensor, const std::vector<float>& labelTensor) {
		int len = outputTensor.size();
		std::vector<float> deltaTensor;
		deltaTensor.resize(len, 0);
		for (int i = 0; i < len; ++i) {
			deltaTensor[i] = labelTensor[i] - outputTensor[i];
		}
		return deltaTensor;
	}

	float train(const std::vector<float>& inputTensor, const std::vector<float>& labelTensor) {
		float loss = 0;
		std::vector<float> outputTensor = this->forward(inputTensor);
		// error calculate
		std::vector<float> deltaTensor = this->deltaCalculate(outputTensor, labelTensor);
		for (float& delta : deltaTensor) {
			loss += delta * delta * 0.5;
		}
		this->backward(deltaTensor);
		return loss;
	}

	std::vector<float> test(const std::vector<float>& inputTensor) {
		return this->forward(inputTensor);
	}
};
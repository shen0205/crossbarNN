#pragma once

#include <random>
#include <ctime>
#include <cmath>
#include <vector>

class activateFunc {
public:
	std::vector<float> activateGrad;
	virtual std::vector<float> func(const std::vector<float> &inputTensor) {
		std::vector<float> outputTensor;
		outputTensor.resize(inputTensor.size());
		return outputTensor;
	}
};

class softmaxF : public activateFunc {
public:
	std::vector<float> func(const std::vector<float> &inputTensor) {
		activateGrad.resize(inputTensor.size(), 0);
		float total = 0;
		float MAX = inputTensor[0];
		for (auto x : inputTensor) {
			MAX = std::max(x, MAX);
		}
		for (auto x : inputTensor) {
			total += exp(x - MAX);
		}
		std::vector<float> outputTensor;
		for (auto x : inputTensor) {
			outputTensor.push_back(exp(x - MAX) / total);
		}
		for (int i = 0, len = inputTensor.size(); i < len; ++i) {
			activateGrad[i] = (1 - outputTensor[i]) * outputTensor[i];
		}
		return outputTensor;
	}
};

class reluF : public activateFunc {
public:
	std::vector<float> func(const std::vector<float> &inputTensor) {
		activateGrad.resize(inputTensor.size(), 0);
		std::vector<float> outputTensor;
		outputTensor.resize(inputTensor.size(), 0);
		for (int i = 0, len = inputTensor.size(); i < len; ++i) {
			if (inputTensor[i] < 0) {
				activateGrad[i] = 0;
				outputTensor[i] = 0;
			}
			else {
				activateGrad[i] = 1;
				outputTensor[i] = inputTensor[i];
			}
		}
		return outputTensor;
	}
};

class rram {
private:
	float g; // conduction, Aij
	float lognormal_mean; // lognormal's mean
	float lognormal_sigma; // lognormal's sigma
	std::lognormal_distribution<float> distribution; // distribution random

public:
	void init(const float& g, const float& lognormal_mean, const float & lognormal_sigma) {
		this->g = g;
		this->lognormal_mean = lognormal_mean;
		this->lognormal_sigma = lognormal_sigma;
		this->distribution = std::lognormal_distribution<float>(lognormal_mean, lognormal_sigma);
	}
	void set(std::default_random_engine& generator) {
		this->g += distribution(generator);
	}
	void reset(std::default_random_engine& generator) {
		this->g -= distribution(generator);
	}

	float getG() {
		return this->g;
	}
};

class adc {
private:
	int bitsNum;
	float referenceVoltage;
	float scaleDivision;
public:
	void init(int bitsNum, float referenceVoltage, float scaleDivision) {
		this->bitsNum = bitsNum;
		this->referenceVoltage = referenceVoltage;
		this->scaleDivision = scaleDivision;
	}

	void A2D(float& analogCurrent) {
		if (std::abs(analogCurrent) < referenceVoltage) {
			analogCurrent = static_cast<float>(static_cast<int>(analogCurrent / this->scaleDivision)) * this->scaleDivision;
		}
		else {
			analogCurrent = referenceVoltage;
		}
	}

	void A2D(std::vector<float>& analogCurrent) {
		for (int i = 0, len = analogCurrent.size(); i < len; ++i) {
			if (std::abs(analogCurrent[i]) < referenceVoltage) {
				analogCurrent[i] = static_cast<float>(static_cast<int>(analogCurrent[i] / this->scaleDivision)) * this->scaleDivision;
			} else {
				if (analogCurrent[i] > 0) {
					analogCurrent[i] = referenceVoltage;
				} else {
					analogCurrent[i] = -referenceVoltage;
				}
			}
		}
	}
};

class crossbar {
private:
	std::vector<std::vector<rram>> RRAMCrossbar;
	std::vector<std::vector<float>> gradMatrix;
	std::vector<float> currentLayerDelta;
	std::vector<float> lastInputTensor;
	int inputTensorLength;
	int outputTensorLength;
	float setThreshold;
	float resetThreshold;
	std::default_random_engine generator;
public:
	void init(const int &inputTensorLength, const int &outputTensorLength, const std::vector<float> &g, const std::vector<float>& lognormal_mean,
		const std::vector<float> &lognormal_sigma, const float &setThreshold, const float &resetThreshold);

	std::vector<std::vector<rram>> getRRAMCrossbar() {
		return this->RRAMCrossbar;
	}

	std::vector<float> forward_calculate(const std::vector<float> &inputTensor);

	void setDelta(const std::vector<float>& delta) {
		this->currentLayerDelta = delta;
	}

	std::vector<float> getLastLayerDelta();

	void calculateGrad(activateFunc func);

	void calculateGrad();

	void updateG();
};
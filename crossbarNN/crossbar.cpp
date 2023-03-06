#include "crossbar.h"

void crossbar::init(const int& inputTensorLength, const int& outputTensorLength, const std::vector<float> &g, const std::vector<float>& lognormal_mean,
	const std::vector<float>& lognormal_sigma, const float &setThreshold = 0, const float &resetThreshold = 0) {
	this->inputTensorLength = inputTensorLength;
	this->outputTensorLength = outputTensorLength;
	this->RRAMCrossbar.resize(inputTensorLength);
	this->gradMatrix.resize(inputTensorLength);
	for (int i = 0; i < inputTensorLength; ++i) {
		RRAMCrossbar[i].resize(outputTensorLength);
		gradMatrix[i].resize(outputTensorLength, 0);
		for (int j = 0; j < outputTensorLength; ++j) {
			RRAMCrossbar[i][j].init(g[i * outputTensorLength + j], lognormal_mean[i * outputTensorLength + j], lognormal_sigma[i * outputTensorLength + j]);
		}
	}
	this->currentLayerDelta.resize(outputTensorLength, 0);
	this->lastInputTensor.resize(inputTensorLength, 0);
	this->resetThreshold = resetThreshold;
	this->setThreshold = setThreshold;
	this->generator = std::default_random_engine(time(NULL));
}

std::vector<float> crossbar::forward_calculate(const std::vector<float>& inputTensor) {
	this->lastInputTensor = inputTensor;
	std::vector<float> outputTensor;
	outputTensor.resize(this->outputTensorLength, 0);
	for (int i = 0; i < this->inputTensorLength; ++i) {
		for (int j = 0; j < this->outputTensorLength; ++j) {
			outputTensor[j] += inputTensor[i] * this->RRAMCrossbar[i][j].getG();
		}
	}
	return outputTensor;
}

std::vector<float> crossbar::getLastLayerDelta() {
	std::vector<float> lastLayerDelta;
	lastLayerDelta.resize(inputTensorLength, 0);
	for (int i = 0; i < this->inputTensorLength; ++i) {
		for (int j = 0; j < this->outputTensorLength; ++j) {
			lastLayerDelta[i] += this->RRAMCrossbar[i][j].getG() * this->currentLayerDelta[j];
		}
	}
	return lastLayerDelta;
}

void crossbar::calculateGrad(activateFunc func) {
	for (int i = 0; i < this->inputTensorLength; ++i) {
		for (int j = 0; j < this->outputTensorLength; ++j) {
			this->gradMatrix[i][j] = this->currentLayerDelta[j] * this->lastInputTensor[i] * func.activateGrad[j];
		}
	}
}

void crossbar::calculateGrad() {
	for (int i = 0; i < this->inputTensorLength; ++i) {
		for (int j = 0; j < this->outputTensorLength; ++j) {
			this->gradMatrix[i][j] = this->currentLayerDelta[j] * this->lastInputTensor[i];
		}
	}
}

void crossbar::updateG() {
	for (int i = 0; i < this->inputTensorLength; ++i) {
		for (int j = 0; j < this->outputTensorLength; ++j) {
			if (this->gradMatrix[i][j] > setThreshold) {
				this->RRAMCrossbar[i][j].set(this->generator);
			}
			else if (this->gradMatrix[i][j] < resetThreshold) {
				this->RRAMCrossbar[i][j].reset(this->generator);
			}
		}
	}
}
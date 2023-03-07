#pragma once

#include "crossbar.h"
#include <string>
#include <memory>

class crossbarModule {
private:
	crossbar CROSSBAR;
	adc ADC;
	std::shared_ptr<activateFunc> activatePointer;
	bool useADC;
public:
	void quick_init(int inputTensorLength, int outputTensorLength, std::string activate = "none",std::string initialization = "normal") {
		std::vector<float> g, mean, sigma;
		const float _N_MEAN = -10, _N_SIGMA = 1;
		const float setThreshold = 0, resetThreshold = 0;
		float setThreshold, resetThreshold;
		std::default_random_engine generator(time(NULL));
		std::normal_distribution<float> Ndistribution(0, 0.1);
		for (int i = 0, len = inputTensorLength * outputTensorLength; i < len; ++i) {
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
	}
};
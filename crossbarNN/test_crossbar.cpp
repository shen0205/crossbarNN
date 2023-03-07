//#include "readMnist.h"
//#include "crossbar.h"
//
//void test_oneLayer_NN(int epochs) {
//    // get samples
//    std::vector<std::vector<float>> train_images;
//    read_Mnist_Images("D:/graduation_design/mnist_data/MNIST/raw/train-images-idx3-ubyte", train_images);
//    std::vector<std::vector<float>> test_images;
//    read_Mnist_Images("D:/graduation_design/mnist_data/MNIST/raw/t10k-images-idx3-ubyte", test_images);
//    // get labels
//    std::vector<int> train_labels;
//    read_Mnist_Label("D:/graduation_design/mnist_data/MNIST/raw/train-labels-idx1-ubyte", train_labels);
//    std::vector<int> test_labels;
//    read_Mnist_Label("D:/graduation_design/mnist_data/MNIST/raw/t10k-labels-idx1-ubyte", test_labels);
//
//    const int Ninput = 28 * 28;
//    const int Noutput = 10;
//    const float Nmean = -10;
//    const float Nsigma = 0.1;
//    const float setThreshold = 0;
//    const float resetThreshold = 0;
//    crossbar layer;
//    std::vector<float> g, mean, sigma, inputTensor;
//    std::default_random_engine generator(time(NULL));
//    std::normal_distribution<float> Ndistribution(0, 0.1);
//    g.resize(Ninput * Noutput);
//    mean.resize(Ninput * Noutput, Nmean);
//    sigma.resize(Ninput * Noutput, Nsigma);
//    for (int i = 0; i < Ninput * Noutput; ++i) {
//        g[i] = Ndistribution(generator);
//    }
//    layer.init(Ninput, Noutput, g, mean, sigma, setThreshold, resetThreshold);
//    softmaxF softmaxLayer;
//    // train
//    for (int i = 0; i < epochs; ++i) {
//        std::cout << "epochs " << i + 1 << "/" << epochs << std::endl;
//        for (int j = 0; j < train_images.size(); ++j) {
//            inputTensor = train_images[j];
//            std::vector<float> labelTensor;
//            labelTensor.resize(Noutput);
//            for (int k = 0; k < 10; ++k) {
//                if (k == train_labels[j]) {
//                    labelTensor[k] = 1;
//                }
//                else {
//                    labelTensor[k] = 0;
//                }
//            }
//            std::vector<float> outputTensor = layer.forward_calculate(inputTensor);
//            outputTensor = softmaxLayer.func(outputTensor);
//            std::vector<float> delta;
//            delta.resize(Noutput);
//            for (int k = 0; k < Noutput; ++k) {
//                delta[k] = (labelTensor[k] - outputTensor[k]);
//            }
//            layer.setDelta(delta);
//            layer.calculateGrad(softmaxLayer);
//            layer.updateG();
//        }
//    }
//
//    //test
//    int maxIndex = 0, total = test_images.size(), right = 0;
//    for (int j = 0; j < test_images.size(); ++j) {
//        inputTensor = test_images[j];
//        std::vector<float> labelTensor;
//        labelTensor.resize(Noutput);
//        for (int k = 0; k < 10; ++k) {
//            if (k == test_labels[j]) {
//                labelTensor[k] = 1;
//            }
//            else {
//                labelTensor[k] = 0;
//            }
//        }
//        std::vector<float> outputTensor = layer.forward_calculate(inputTensor);
//        outputTensor = softmaxLayer.func(outputTensor);
//        float maxNum = outputTensor[0];
//        for (int k = 1; k < Noutput; ++k) {
//            if (outputTensor[k] > maxNum) {
//                maxNum = outputTensor[k];
//                maxIndex = k;
//            }
//        }
//        if (maxIndex == test_labels[j]) {
//            ++right;
//        }
//        if ((j + 1) % 1000 == 0) {
//            std::cout << j << std::endl << "output:";
//            for (int k = 0; k < Noutput; ++k) {
//                std::cout << outputTensor[k] << " ";
//            }
//            std::cout << std::endl;
//            std::cout << "label:";
//            for (int k = 0; k < Noutput; ++k) {
//                std::cout << labelTensor[k] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//    std::cout << "accuracy = " << static_cast<float>(right) / static_cast<float>(total) << std::endl;
//    //std::cout << "layer's param:" << std::endl;
//    //for (int i = 0; i < Ninput; ++i) {
//    //    for (int j = 0; j < Noutput; ++j) {
//    //        std::cout << layer.getRRAMCrossbar()[i][j].getG() << " ";
//    //    }
//    //    std::cout << std::endl;
//    //}
//}
//
//void test_twoLayer_NN(int epochs) {
//    // get samples
//    std::vector<std::vector<float>> train_images;
//    read_Mnist_Images("D:/graduation_design/mnist_data/MNIST/raw/train-images-idx3-ubyte", train_images);
//    std::vector<std::vector<float>> test_images;
//    read_Mnist_Images("D:/graduation_design/mnist_data/MNIST/raw/t10k-images-idx3-ubyte", test_images);
//    // get labels
//    std::vector<int> train_labels;
//    read_Mnist_Label("D:/graduation_design/mnist_data/MNIST/raw/train-labels-idx1-ubyte", train_labels);
//    std::vector<int> test_labels;
//    read_Mnist_Label("D:/graduation_design/mnist_data/MNIST/raw/t10k-labels-idx1-ubyte", test_labels);
//
//    const int input = 28 * 28;
//    const int hidden = 32;
//    const int output = 10;
//    const float Nmean = -10;
//    const float Nsigma = 0.1;
//    const float setThreshold = 0;
//    const float resetThreshold = 0;
//    crossbar layer1, layer2;
//
//    std::vector<float> g1, mean1, sigma1, g2, mean2, sigma2, inputTensor;
//    std::default_random_engine generator(time(NULL));
//    std::normal_distribution<float> Ndistribution(0, 0.1);
//    g1.resize(input * hidden);
//    mean1.resize(input * hidden, Nmean);
//    sigma1.resize(input * hidden, Nsigma);
//    for (int i = 0; i < input * hidden; ++i) {
//        g1[i] = Ndistribution(generator);
//    }
//    layer1.init(input, hidden, g1, mean1, sigma1, setThreshold, resetThreshold);
//    g2.resize(output * hidden);
//    mean2.resize(output * hidden, Nmean);
//    sigma2.resize(output * hidden, Nsigma);
//    for (int i = 0; i < output * hidden; ++i) {
//        g2[i] = Ndistribution(generator);
//    }
//    layer2.init(hidden, output, g2, mean2, sigma2, setThreshold, resetThreshold);
//
//    softmaxF softmaxLayer;
//    reluF reluLayer;
//    sigmoidF sigmoidLayer;
//    adc ADC;
//    ADC.init(24, std::pow(2, 22));
//    // train
//    for (int i = 0; i < epochs; ++i) {
//        std::cout << "epochs " << i + 1 << "/" << epochs;
//        float total_loss = 0;
//        for (int j = 0; j < train_images.size(); ++j) {
//            inputTensor = train_images[j];
//            std::vector<float> labelTensor;
//            labelTensor.resize(output);
//            for (int k = 0; k < 10; ++k) {
//                if (k == train_labels[j]) {
//                    labelTensor[k] = 1;
//                } else {
//                    labelTensor[k] = 0;
//                }
//            }
//            std::vector<float> outputTensor = layer1.forward_calculate(inputTensor);
//            ADC.A2D(outputTensor);
//            //outputTensor = sigmoidLayer.func(outputTensor);
//            outputTensor = layer2.forward_calculate(outputTensor);
//            ADC.A2D(outputTensor);
//            outputTensor = softmaxLayer.func(outputTensor);
//            std::vector<float> delta1, delta2;
//            delta2.resize(output);
//            for (int k = 0; k < output; ++k) {
//                delta2[k] = (labelTensor[k] - outputTensor[k]);
//                total_loss += delta2[k] * delta2[k] * 0.5;
//            }
//            layer2.setDelta(delta2);
//            delta1 = layer2.getLastLayerDelta();
//            layer1.setDelta(delta1);
//
//            layer2.calculateGrad(softmaxLayer);
//            layer1.calculateGrad();
//            layer2.updateG();
//            layer1.updateG();
//        }
//        std::cout << "  loss = " << total_loss / static_cast<float>(train_images.size()) << std::endl;
//    }
//    // test
//    int maxIndex = 0, total = test_images.size(), right = 0;
//    for (int j = 0; j < test_images.size(); ++j) {
//        inputTensor = test_images[j];
//        std::vector<float> labelTensor;
//        labelTensor.resize(output);
//        for (int k = 0; k < 10; ++k) {
//            if (k == test_labels[j]) {
//                labelTensor[k] = 1;
//            }
//            else {
//                labelTensor[k] = 0;
//            }
//        }
//        std::vector<float> hiddenTensor = layer1.forward_calculate(inputTensor);
//        ADC.A2D(hiddenTensor);
//        //std::vector<float> hiddenTensor2 = sigmoidLayer.func(hiddenTensor);
//        std::vector<float> outputTensor = layer2.forward_calculate(hiddenTensor);
//        ADC.A2D(outputTensor);
//        std::vector<float> outputTensor2 = softmaxLayer.func(outputTensor);
//        float maxNum = outputTensor2[0];
//        for (int k = 1; k < output; ++k) {
//            if (outputTensor2[k] > maxNum) {
//                maxNum = outputTensor2[k];
//                maxIndex = k;
//            }
//        }
//        if (maxIndex == test_labels[j]) {
//            ++right;
//        }
//        if (j % 1000 == 0) {
//            std::cout << j << std::endl;
//            std::cout << "hidden:";
//            for (int k = 0; k < hidden; ++k) {
//                std::cout << hiddenTensor[k] << " ";
//            }
//            std::cout << "output2:";
//            std::cout << std::endl;
//            for (int k = 0; k < output; ++k) {
//                std::cout << outputTensor2[k] << " ";
//            }
//            std::cout << std::endl;
//            std::cout << "label:";
//            for (int k = 0; k < output; ++k) {
//                std::cout << labelTensor[k] << " ";
//            }
//            std::cout << std::endl;
//            std::cout << "output1:";
//            for (int k = 0; k < output; ++k) {
//                std::cout << outputTensor[k] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//    std::cout << "accuracy = " << static_cast<float>(right) / static_cast<float>(total) << std::endl;
//}
//
//void test_adc() {
//    std::vector<float> analog = { 0.0000001, -0.0000002, 0.0000003, -0.0000004, 0.0000005, -0.0000006, 0.0000007, -0.0000008, 0.0000009, -1 };
//    for (auto& tmp : analog) {
//        std::cout << tmp << " ";
//    }
//    std::cout << std::endl;
//    adc ADC;
//    ADC.init(31, 1);
//    ADC.A2D(analog); 
//    for (auto& tmp : analog) {
//        std::cout << tmp << " ";
//    }
//}
//
//int main() {
//    //test_oneLayer_NN(20);
//    test_twoLayer_NN(10);
//    //test_adc();
//	return 0;
//}
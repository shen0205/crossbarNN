#include "crossbarModule.h"
#include "readMnist.h"

void testTwoLayer(int epochs) {
    // get samples
    std::vector<std::vector<float>> train_images;
    read_Mnist_Images("D:/graduation_design/mnist_data/MNIST/raw/train-images-idx3-ubyte", train_images);
    std::vector<std::vector<float>> test_images;
    read_Mnist_Images("D:/graduation_design/mnist_data/MNIST/raw/t10k-images-idx3-ubyte", test_images);
    // get labels
    std::vector<int> train_labels;
    read_Mnist_Label("D:/graduation_design/mnist_data/MNIST/raw/train-labels-idx1-ubyte", train_labels);
    std::vector<int> test_labels;
    read_Mnist_Label("D:/graduation_design/mnist_data/MNIST/raw/t10k-labels-idx1-ubyte", test_labels);

    int train_sample_num = train_images.size();
    int test_sample_num = test_images.size();
    int class_num = 10;
    
    // build layer
    crossbarLayer layer1, layer2;
    layer1.quick_init(28 * 28, 32, "relu");
    layer2.quick_init(32, 10, "softmax");
    std::vector<crossbarLayer> layers = { layer1, layer2 };

    // build net
    crossbarModule MLP;
    MLP.init(layers);

    // train
    for (int i = 0; i < epochs; ++i) {
        float total_loss = 0;
        std::cout << "epochs " << i + 1 << "/" << epochs;
        for (int j = 0; j < train_sample_num; ++j) {
            std::vector<float> labelTensor;
            labelTensor.resize(class_num, 0);
            for (int k = 0; k < class_num; ++k) {
                if (k == train_labels[j]) {
                    labelTensor[k] = 1;
                    break;
                }
            }
            total_loss += MLP.train(train_images[j], labelTensor);
        }
        std::cout << "   loss = " << total_loss / (float)train_sample_num << std::endl;
    }

    // test
    int right_num = 0;
    for (int i = 0; i < test_sample_num; ++i) {
        std::vector<float> labelTensor;
        labelTensor.resize(class_num, 0);
        for (int k = 0; k < class_num; ++k) {
            if (k == test_labels[i]) {
                labelTensor[k] = 1;
                break;
            }
        }
        std::vector<float> outputTensor = MLP.test(test_images[i]);
        int maxIndex = 0;
        float max = outputTensor[0];
        for (int j = 1; j < class_num; ++j) {
            if (outputTensor[j] > max) {
                max = outputTensor[j];
                maxIndex = j;
            }
        }
        if (maxIndex == test_labels[i]) {
            ++right_num;
        }
    }
    std::cout << "accuracy = " << (float)right_num / (float)test_sample_num;
}

int main() {
	testTwoLayer(5);
	return 0;
}
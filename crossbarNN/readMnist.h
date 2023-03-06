#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

int ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist_Label(std::string filename, std::vector<int>& labels)
{
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        std::cout << "magic number = " << magic_number << std::endl;
        std::cout << "number of images = " << number_of_images << std::endl;


        for (int i = 0; i < number_of_images; i++)
        {
            unsigned char label = 0;
            file.read((char*)&label, sizeof(label));
            labels.push_back((int)label);
        }

    }
}

void read_Mnist_Images(std::string filename, std::vector<std::vector<float>>& images)
{
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        unsigned char label;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        file.read((char*)&n_rows, sizeof(n_rows));
        file.read((char*)&n_cols, sizeof(n_cols));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        n_rows = ReverseInt(n_rows);
        n_cols = ReverseInt(n_cols);

        std::cout << "magic number = " << magic_number << std::endl;
        std::cout << "number of images = " << number_of_images << std::endl;
        std::cout << "rows = " << n_rows << std::endl;
        std::cout << "cols = " << n_cols << std::endl;

        for (int i = 0; i < number_of_images; i++)
        {
            std::vector<float>tp;
            for (int r = 0; r < n_rows; r++)
            {
                for (int c = 0; c < n_cols; c++)
                {
                    unsigned char image = 0;
                    file.read((char*)&image, sizeof(image));
                    tp.push_back((float)image / 255);
                }
            }
            images.push_back(tp);
        }
    }
    else {
        std::cout << "file open error." << std::endl;
    }
}
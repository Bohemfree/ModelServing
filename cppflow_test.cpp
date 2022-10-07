#pragma once
#include "cppflow_test.h"

int main()
{
    std::cout << TF_Version() << std::endl;
    std::string model_path = "./saved_model";
    std::string img_dir = "D:/dataset/cppflow_test/";
    std::string output_dir = "D:/dataset/cppflow_test_output";
    
    std::vector<std::string> image_paths = Model::ReadDir(img_dir);

    // Each image prediction
    for (std::string image_path : image_paths)
    {
        std::string basename = image_path.substr(image_path.find_last_of("\\/"));
        cv::Mat image = cv::imread(image_path, 0);
        if (image.empty())
        {
            std::cout << image_path << "is not image file" << std::endl;
            continue;
        }
        
        std::vector<cv::Mat> split_images = Model::SplitImage(image);
        std::vector<cv::Mat> predictions = Model::Prediction(image_path, model_path, split_images, output_dir);
        cv::Mat output_image = Model::ConcatImage(predictions, image.rows, image.cols);

        //Save image
        std::string output_path = output_dir + basename;
        cv::imwrite(output_path, output_image);
    }
    
    return 0;
}
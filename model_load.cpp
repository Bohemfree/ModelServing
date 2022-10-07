#pragma once
#include "cppflow_test.h"

// Model input shape
const int INPUT_ROWS = 512;
const int INPUT_COLS = 512;

Model::Model()
{
    Init();
}

Model::~Model()
{
    Delete();
}

void Model::Init()
{
}

void Model::Delete()
{
}

std::vector<std::string> Model::ReadDir(const std::string& _image_dir)
{
    std::vector<std::string> image_paths;

    // load image paths
    struct _finddata_t fd;	intptr_t handle;
    if ((handle = _findfirst((_image_dir + "/*.jpg").c_str(), &fd)) == -1L)
    {
        std::cout << "No file in directory!" << std::endl;
        return image_paths;
    }

    do
    {
        //std::cout << fd.name << std::endl;
        image_paths.push_back(_image_dir + "/" + fd.name);
    } while (_findnext(handle, &fd) == 0);
    _findclose(handle);

    // using lambda, sort numeric vector elements
    std::sort(image_paths.begin(), image_paths.end(),
        [](std::string s1, std::string s2) -> bool
        {
            if (s1.size() == s2.size())
                return s1 < s2;
            else
                return s1.size() < s2.size();
        });
    return image_paths;
}

std::vector<cv::Mat> Model::SplitImage(const cv::Mat& _image_data)
{
    std::vector<cv::Mat> split_images;

    // split images
    int rows = _image_data.rows;
    int cols = _image_data.cols;
    int rows_num = rows / INPUT_ROWS;
    int cols_num = cols / INPUT_COLS;
    int x = 0;
    int y = 0;
    
    if (_image_data.rows == INPUT_ROWS && _image_data.cols == INPUT_COLS)
    {
        split_images.push_back(_image_data);
        return split_images;
    }

    for (auto r = 0; r < rows_num + 1; r++)
    {
        for (auto c = 0; c < cols_num + 1; c++)
        {
            if ((cols % INPUT_COLS == 0 && c == cols_num) ||
                (rows % INPUT_ROWS == 0 && r == rows_num))
                continue;
            else
            {
                x = (c == cols_num) ? (cols - INPUT_COLS) : (c * INPUT_COLS);
                y = (r == rows_num) ? (rows - INPUT_ROWS) : (r * INPUT_ROWS);
            }

            cv::Mat tile = _image_data(cv::Rect(x, y, INPUT_COLS, INPUT_ROWS));
            split_images.push_back(tile);
        }
    }
    return split_images;
}

cv::Mat Model::ConcatImage(const std::vector<cv::Mat>& _images, const int _rows, const int _cols)
{
    cv::Mat concat_image(_rows, _cols, CV_8UC1);

    int rows_num = _rows / INPUT_ROWS;
    int cols_num = _cols / INPUT_COLS;
    int index = 0;

    if (rows_num == 1 && cols_num == 1)
    {
        _images[0].copyTo(concat_image(cv::Rect(0, 0, INPUT_ROWS, INPUT_COLS)));
        return concat_image;
    }

    for (auto r = 0; r < rows_num + 1; r++)
    {
        for (auto c = 0; c < cols_num + 1; c++)
        {
            if ((_cols % INPUT_COLS == 0 && c == cols_num) ||
                (_rows % INPUT_ROWS == 0 && r == rows_num))
                continue;
            else
            {
                int width = (c==cols_num) ? (_cols - (c * INPUT_COLS)) : INPUT_COLS;
                int height = (r == rows_num) ? (_rows - (r * INPUT_ROWS)) : INPUT_ROWS;

                auto merging_image = _images[index](cv::Rect((INPUT_COLS - width), (INPUT_ROWS - height), width, height));
                merging_image.copyTo(concat_image(cv::Rect((c * INPUT_COLS), (r * INPUT_ROWS), width, height)));
            }
            index += 1;
        }
    }

    return concat_image;
}

std::vector<cppflow::tensor> Model::Mat2Tensor(const std::vector<cv::Mat>& _Mat_vector)
{
    std::vector<cppflow::tensor> tensor_vector;
    for (cv::Mat image : _Mat_vector)
    {
        int rows = image.rows;
        int cols = image.cols;
        int channels = image.channels();
        int total = image.total();
        unsigned char* data = image.data;


        // Mat to vector
        std::vector<uint8_t> img_data;
        if (image.isContinuous())
        {
            img_data.assign(data, data + total * channels);
        }
        else
        {
            for (int i = 0; i < rows; ++i)
            {
                img_data.insert(img_data.end(), image.ptr<uchar>(i), image.ptr<uchar>(i) + cols * channels);
            }
        }


        // Mat to Tensor
        cppflow::tensor tensor_data = cppflow::tensor(img_data, { rows, cols, channels });
        tensor_data = cppflow::cast(tensor_data, TF_UINT8, TF_FLOAT);
        tensor_data = tensor_data / 255.f;
        tensor_data = cppflow::expand_dims(tensor_data, 0);
        tensor_vector.push_back(tensor_data);
    }

    return tensor_vector;
}

std::vector<cv::Mat> Model::Tensor2Mat(const std::vector<cppflow::tensor>& _tensor_vector)
{
    std::vector<cv::Mat> Mat_vector;
    for (cppflow::tensor output_tensor : _tensor_vector)
    {
        output_tensor = output_tensor * 255.f;
        output_tensor = cppflow::cast(output_tensor, TF_FLOAT, TF_UINT8);
        
        //Tensor to Mat
        std::vector<uint8_t> output_vector = output_tensor.get_data<uint8_t>();
        cv::Mat output_Mat = cv::Mat(INPUT_ROWS, INPUT_COLS, CV_8UC1);
        memcpy(output_Mat.data, output_vector.data(), output_vector.size() * sizeof(uint8_t));
        Mat_vector.push_back(output_Mat);
    }
    return Mat_vector;
}

std::vector<cv::Mat> Model::Prediction(
    const std::string& _input_path, const std::string& _model_path,
    const std::vector<cv::Mat>& _images, const std::string& _output_dir)
{
    std::vector<cppflow::tensor> tensor_images = Model().Mat2Tensor(_images);
    std::vector<cv::Mat> test_Mat_images = Model().Tensor2Mat(tensor_images);

    // Load model
    cppflow::model model(_model_path);
    
    // Predict : Since the input size of the U-Net model is 512x512 size 1 channel, the tensor shape must be 512x512.
    std::vector<cppflow::tensor> outputs;

    int pre_index = 1;
    for (cppflow::tensor input : tensor_images)
    {
        auto output = model(input);
        printf("predicting... %s (%d/%d)\n", _input_path.c_str(), pre_index, _images.size());
        outputs.push_back(output);
        pre_index++;
    }

    std::vector<cv::Mat> Mat_images = Model().Tensor2Mat(outputs);
    return Mat_images;
}
/**
* \file     model.h
* \brief    tensorflow model 기능
* \author   YJS
* \date
* -. 초기 : 2022.07.25
*/
#pragma once
/**
* \class	ModelLoad
* \brief	Model load class
* \date
* -. 초기 : 2022.07.25
*/
class Model
{
public:
    Model();
    ~Model();
    
private:
    void Init();
    void Delete();

public:
    /*
    * Search image directory
    * \param _image_dir : image directory
    * \return : jpg format image path under _image_dir 
    */
    static std::vector<std::string> ReadDir(const std::string& _image_dir);


    /*
    * Split Mat data image
    * \param _image_data : cv::Mat data for model input of grayscale channel
    * \return : cv::Mat vector split by model input size
    */
    static std::vector<cv::Mat> SplitImage(const cv::Mat& _image_data);


    /*
    * Combine the mat images that have been predicted
    * \param _images : vector<cv::Mat> data after predictions
    * \param _rows : Output image's rows
    * \param _cols : Output image's columns
    * \return : Merged cv::Mat data
    */
    static cv::Mat ConcatImage(const std::vector<cv::Mat>& _images, const int _rows, const int _cols);

    /*
    * Convert cv::Mat to cppflow::tensor format
    * \param Mat_vector_ : cv::Mat format vector
    * \return : vector of type cppflow::tensor
    */
    std::vector<cppflow::tensor> Mat2Tensor(const std::vector<cv::Mat>& _Mat_vector);


    
    //Convert cppflow::tensor to cv::Mat format
    //\param tensor_vector_ : cppflow::tensor format vector
    //\return : vector of type cv::Mat
    std::vector<cv::Mat> Tensor2Mat(const std::vector<cppflow::tensor>& _tensor_vector);


    //Predict image data using pre-trained model (model format:pb)
    //\param _input_paths : Return vector of "ReadDir", required for naming when saving the result value
    //\param _model_path : Saved model path
    //\param _images : Image data -> cppflow::tensor format data vector
    //\param _output_dir : Output path for images
    //\return : bool
    static std::vector<cv::Mat> Prediction(
        const std::string& _input_path, const std::string& _model_path, 
        const std::vector<cv::Mat>& _images, const std::string& _output_dir);
};

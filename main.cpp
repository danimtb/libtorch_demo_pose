#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, const char* argv[]) {
    // We now accept a video file path or '0' for webcam
    if (argc != 3) {
        std::cerr << "Usage: ./pose_demo <path-to-torchscript-model> <path-to-video-or-0-for-webcam>\n";
        return -1;
    }

    std::string model_path = argv[1];
    std::string video_source = argv[2];

    // 1. Load the TorchScript model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
        module.eval(); // Set model to evaluation mode
        std::cout << "Model loaded successfully.\n";
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model!\n";
        return -1;
    }

    // Determine device
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    module.to(device);
    std::cout << "Running on " << (device.is_cuda() ? "GPU" : "CPU") << ".\n";

    // 2. Open Video Capture
    cv::VideoCapture cap;
    if (video_source == "0") {
        cap.open(0); // Open default webcam
    } else {
        cap.open(video_source); // Open video file
    }

    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file!\n";
        return -1;
    }

    cv::Mat img;
    std::cout << "Starting video loop. Press 'ESC' to exit.\n";

    // 3. Video Processing Loop
    while (cap.read(img)) {
        if (img.empty()) {
            std::cout << "End of video stream.\n";
            break;
        }

        // 4. Preprocess the frame
        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(640, 640));
        cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
        resized_img.convertTo(resized_img, CV_32FC3, 1.0f / 255.0f);

        // Convert cv::Mat to torch::Tensor
        torch::Tensor img_tensor = torch::from_blob(
            resized_img.data, 
            {1, resized_img.rows, resized_img.cols, 3}, 
            torch::kFloat32
        );

        img_tensor = img_tensor.permute({0, 3, 1, 2}).contiguous();
        img_tensor = img_tensor.to(device); // Move tensor to GPU if available

        // 5. Run inference
        std::vector<torch::jit::IValue> inputs{img_tensor};
        auto output = module.forward(inputs);

        // 6. Extract the output tensor
        torch::Tensor preds = output.toTuple()->elements()[0].toTensor();
        
        // NOTE: Post-processing (NMS and drawing skeletons) goes here.
        // For now, we will just display the original unedited frame.

        // 7. Display the result
        cv::imshow("Pose Estimation Demo", img);

        // Break the loop if the user presses the ESC key (ASCII 27)
        if (cv::waitKey(1) == 27) {
            std::cout << "User exited the video loop.\n";
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
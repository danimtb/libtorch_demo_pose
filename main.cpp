#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

// COCO Keypoint pairs for skeleton lines
const std::vector<std::pair<int, int>> SKELETON = {
    {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12},
    {7, 13}, {6, 7}, {6, 8}, {7, 9}, {8, 10}, {9, 11},
    {2, 3}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}
};

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./depth_demo <model_path> <video_path>\n";
        return -1;
    }

    // 1. Load Model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
        module.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model\n";
        return -1;
    }
    std::cout << "Model loaded successfully.\n";

    // 2. Open Video
    cv::VideoCapture cap(argv[2]);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream!\n";
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) break;

        // 3. Preprocess Frame
        int img_size = 640;
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(img_size, img_size));
        cv::cvtColor(resized, resized, cv::ColorConversionCodes::COLOR_BGR2RGB);
        
        torch::Tensor input_tensor = torch::from_blob(resized.data, {1, img_size, img_size, 3}, torch::kByte);
        input_tensor = input_tensor.permute({0, 3, 1, 2}).to(torch::kFloat).div(255.0);

        // 4. Inference
        // Use toTensor() to avoid the isTuple() assertion error
        at::Tensor output = module.forward({input_tensor}).toTensor();

        // 5. Post-process (YOLOv8 output: [1, 56, 8400])
        output = output[0].transpose(0, 1); // Shape: [8400, 56]
        
        // Find best detection based on confidence (index 4)
        auto max_result = output.slice(1, 4, 5).max(0);
        float max_conf = std::get<0>(max_result).item<float>();
        int max_idx = std::get<1>(max_result).item<int>();

        if (max_conf > 0.5) {
            at::Tensor best_box = output[max_idx];
            
            // Get keypoints (indices 5 to 55)
            // Each keypoint: [x, y, conf]
            std::vector<float> kpts;
            for (int i = 5; i < 56; i++) {
                kpts.push_back(best_box[i].item<float>());
            }

            // 6. Draw Keypoints and Skeleton
            float scale_x = (float)frame.cols / img_size;
            float scale_y = (float)frame.rows / img_size;

            // Draw Lines
            for (const auto& bone : SKELETON) {
                int i1 = (bone.first - 1) * 3;
                int i2 = (bone.second - 1) * 3;
                
                if (kpts[i1 + 2] > 0.5 && kpts[i2 + 2] > 0.5) {
                    cv::line(frame, 
                             cv::Point(kpts[i1] * scale_x, kpts[i1 + 1] * scale_y),
                             cv::Point(kpts[i2] * scale_x, kpts[i2 + 1] * scale_y),
                             cv::Scalar(255, 255, 0), 2);
                }
            }

            // Draw Circles
            for (int i = 0; i < 17; i++) {
                if (kpts[i * 3 + 2] > 0.5) {
                    cv::circle(frame, 
                               cv::Point(kpts[i * 3] * scale_x, kpts[i * 3 + 1] * scale_y),
                               4, cv::Scalar(0, 0, 255), -1);
                }
            }
        }

        cv::imshow("Pose Estimation", frame);
        if (cv::waitKey(1) == 27) break; // ESC
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
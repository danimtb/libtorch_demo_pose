#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

const std::vector<std::pair<int, int>> skeleton = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, 
    {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}
};

int main(int argc, const char* argv[]) {
    if (argc != 3) return -1;

    std::string model_path = argv[1];
    std::string video_source = argv[2];

    // Force CUDA if available
    bool use_cuda = torch::cuda::is_available();
    torch::Device device(use_cuda ? torch::kCUDA : torch::kCPU);
    if (use_cuda) std::cout << "Using CUDA Device: " << device << std::endl;

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path, device);
        module.eval(); 
        // Use Half Precision for significant speedup on Jetson/GPU
        if (use_cuda) module.to(torch::kHalf); 
    } catch (const c10::Error& e) {
        return -1;
    }

    cv::VideoCapture cap(video_source == "0" ? 0 : video_source);
    cv::Mat img;

    while (cap.read(img)) {
        if (img.empty()) break;

        // 1. Preprocess on CPU (standard)
        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(640, 640));
        cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
        resized_img.convertTo(resized_img, CV_32FC3, 1.0f / 255.0f);

        // 2. Transfer to GPU
        torch::Tensor img_tensor = torch::from_blob(resized_img.data, {1, 640, 640, 3}, torch::kFloat32);
        img_tensor = img_tensor.permute({0, 3, 1, 2}).contiguous().to(device);
        
        if (use_cuda) img_tensor = img_tensor.to(torch::kHalf);

        // 3. Inference on CUDA
        auto output_ivalue = module.forward({img_tensor});
        torch::Tensor preds;
        
        if (output_ivalue.isTensor()) {
            preds = output_ivalue.toTensor();
        } else {
            preds = output_ivalue.toTuple()->elements()[0].toTensor();
        }

        // 4. Move only the results back to CPU for drawing
        preds = preds.to(torch::kCPU).to(torch::kFloat32).squeeze(0).transpose(0, 1);
        auto preds_a = preds.accessor<float, 2>();

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<std::vector<float>> keypoints_list;
        
        for (int i = 0; i < preds.size(0); ++i) {
            float score = preds_a[i][4];
            if (score > 0.5f) {
                float cx = preds_a[i][0], cy = preds_a[i][1], w = preds_a[i][2], h = preds_a[i][3];
                boxes.push_back(cv::Rect(cx - w / 2, cy - h / 2, w, h));
                scores.push_back(score);
                
                std::vector<float> kpts(51);
                for (int k = 0; k < 51; ++k) kpts[k] = preds_a[i][5 + k];
                keypoints_list.push_back(kpts);
            }
        }

        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(boxes, scores, 0.5f, 0.4f, nms_indices);

        float x_scale = (float)img.cols / 640.0f;
        float y_scale = (float)img.rows / 640.0f;

        // 5. Drawing (CPU is faster for these simple primitives than GPU Round-trip)
        for (int idx : nms_indices) {
            auto& kpts = keypoints_list[idx];
            for (const auto& bone : skeleton) {
                int a = bone.first, b = bone.second;
                if (kpts[a*3+2] > 0.5f && kpts[b*3+2] > 0.5f) {
                    cv::line(img, 
                             cv::Point(kpts[a*3]*x_scale, kpts[a*3+1]*y_scale),
                             cv::Point(kpts[b*3]*x_scale, kpts[b*3+1]*y_scale), 
                             cv::Scalar(0, 255, 0), 2);
                }
            }
        }

        cv::imshow("CUDA Pose Demo", img);
        if (cv::waitKey(1) == 27) break;
    }
    return 0;
}

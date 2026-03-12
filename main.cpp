#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <string>

const std::vector<std::pair<int, int>> skeleton = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, 
    {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}
};

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./pose_demo <path-to-torchscript-model> <path-to-video-or-0>\n";
        return -1;
    }

    std::string model_path = argv[1];
    std::string video_source = argv[2];

    bool use_cuda = torch::cuda::is_available();
    torch::Device device(use_cuda ? torch::kCUDA : torch::kCPU);
    
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path, device);
        module.eval(); 
        if (use_cuda) module.to(torch::kHalf);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model!\n";
        return -1;
    }

    cv::VideoCapture cap(video_source == "0" ? 0 : video_source);
    if (!cap.isOpened()) return -1;

    cv::Mat img;
    while (cap.read(img)) {
        if (img.empty()) break;

        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(640, 640));
        cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
        resized_img.convertTo(resized_img, CV_32FC3, 1.0f / 255.0f);

        torch::Tensor img_tensor = torch::from_blob(resized_img.data, {1, 640, 640, 3}, torch::kFloat32);
        img_tensor = img_tensor.permute({0, 3, 1, 2}).contiguous();
        if (use_cuda) img_tensor = img_tensor.to(device).to(torch::kHalf);

        // --- INFERENCE FIX START ---
        std::vector<torch::jit::IValue> inputs{img_tensor};
        auto output_ivalue = module.forward(inputs);

        torch::Tensor preds;
        if (output_ivalue.isTensor()) {
            // Case: Model returns raw [1, 56, 8400] tensor
            preds = output_ivalue.toTensor();
        } else if (output_ivalue.isTuple()) {
            // Case: Model returns ([1, 56, 8400], ...) tuple
            preds = output_ivalue.toTuple()->elements()[0].toTensor();
        } else {
            std::cerr << "Error: Unexpected output type from model!\n";
            continue;
        }
        // --- INFERENCE FIX END ---

        preds = preds.to(torch::kCPU).to(torch::kFloat32);
        preds = preds.squeeze(0).transpose(0, 1);
        auto preds_a = preds.accessor<float, 2>();

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<std::vector<float>> keypoints_list;
        
        float conf_threshold = 0.5f;
        for (int i = 0; i < preds.size(0); ++i) {
            float score = preds_a[i][4];
            if (score > conf_threshold) {
                float cx = preds_a[i][0];
                float cy = preds_a[i][1];
                float w = preds_a[i][2];
                float h = preds_a[i][3];
                
                boxes.push_back(cv::Rect(cx - w / 2, cy - h / 2, w, h));
                scores.push_back(score);
                
                std::vector<float> kpts(51);
                for (int k = 0; k < 51; ++k) {
                    kpts[k] = preds_a[i][5 + k];
                }
                keypoints_list.push_back(kpts);
            }
        }

        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(boxes, scores, conf_threshold, 0.4f, nms_indices);

        float x_scale = (float)img.cols / 640.0f;
        float y_scale = (float)img.rows / 640.0f;

        for (int idx : nms_indices) {
            auto& kpts = keypoints_list[idx];
            std::vector<cv::Point> scaled_kpts(17);
            std::vector<float> kpt_confs(17);
            
            for (int k = 0; k < 17; ++k) {
                float kx = kpts[k * 3] * x_scale;
                float ky = kpts[k * 3 + 1] * y_scale;
                float kconf = kpts[k * 3 + 2];
                
                scaled_kpts[k] = cv::Point(kx, ky);
                kpt_confs[k] = kconf;
                
                if (kconf > 0.5f) {
                    cv::circle(img, scaled_kpts[k], 5, cv::Scalar(0, 255, 0), -1);
                }
            }
            
            for (const auto& bone : skeleton) {
                if (kpt_confs[bone.first] > 0.5f && kpt_confs[bone.second] > 0.5f) {
                    cv::line(img, scaled_kpts[bone.first], scaled_kpts[bone.second], cv::Scalar(255, 0, 0), 2);
                }
            }
        }

        cv::imshow("Jetson Pose Demo", img);
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
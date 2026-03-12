Build and run the demo:

```
cd libtorch_demo_pose
conan install --build missing [-c tools.files.download:verify=False]
cmake --preset conan-release
cmake --build build/Release
cd build/release
./pose_estimation ../../models/yolov8n-pose.torchscript ../../assets/dancing.mp4
```

Download the model:

```
pip install -r requirements.txt
python download_model.py
```

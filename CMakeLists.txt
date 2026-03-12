cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(pose_estimation)

# Buscar OpenCV
find_package(OpenCV REQUIRED)

# Buscar LibTorch
# Debes pasar la ruta: cmake -DCMAKE_PREFIX_PATH=/ruta/a/libtorch ..
find_package(Torch REQUIRED)

# Configurar C++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(pose_estimation main.cpp)

# Vincular las librerías
target_link_libraries(pose_estimation libtorch::libtorch opencv::opencv)

# Asegurar que CUDA esté habilitado
set_property(TARGET pose_estimation PROPERTY CUDA_STANDARD 14)
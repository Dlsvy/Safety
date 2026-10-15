# photo3d: CUDA 빌드 torch 를 CPU 전용으로 링크하기 위한 빈 타깃
foreach(t torch::cudart torch::nvtoolsext torch::cufft torch::curand torch::cublas torch::cudnn CUDA::nvToolsExt CUDA::cudart CUDA::cuda_driver)
  if(NOT TARGET ${t})
    add_library(${t} INTERFACE IMPORTED)
  endif()
endforeach()

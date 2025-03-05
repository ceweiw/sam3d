#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "knn_query_cuda_kernel.h"


void knn_query_cuda(int m, int nsample, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor, at::Tensor offset_tensor, at::Tensor new_offset_tensor, at::Tensor idx_tensor, at::Tensor dist2_tensor)
{   //nsample相当于k  m是new_xyz_tensor的点数
    const float *xyz = xyz_tensor.data_ptr<float>(); //获取 xyz_tensor 张量的底层数据指针，并将其强制转换为 const float* 类型
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const int *offset = offset_tensor.data_ptr<int>();
    const int *new_offset = new_offset_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();
    float *dist2 = dist2_tensor.data_ptr<float>();
    knn_query_cuda_launcher(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2);
}
//这个函数的作用是获取一下相关的指针 让cuda进行计算

#include <ATen/opencl/OPENCLTypeDefault.h>

#include <ATen/opencl/OPENCLContext.h>
#include <ATen/opencl/OPENCLDevice.h>
#include <ATen/OPENCLGenerator.h>

namespace at {
/*
Allocator* CUDATypeDefault::allocator() const {
  return cuda::getCUDADeviceAllocator();
}
Device CUDATypeDefault::getDeviceFromPtr(void * data) const {
  return cuda::getDeviceFromPtr(data);
}
std::unique_ptr<Generator> CUDATypeDefault::generator() const {
  return std::unique_ptr<Generator>(new CUDAGenerator(&at::globalContext()));
}
*/
} // namespace at

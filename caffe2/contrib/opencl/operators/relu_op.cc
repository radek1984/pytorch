
#include "caffe2/operators/relu_op.h"

#include "caffe2/utils/math.h"

#include "context.h"
#include "operator.h"
#include <sys/time.h>

namespace caffe2 {

static constexpr const char* kRelu = R"CLC(
kernel void kRelu(
  global float* a,
  global float* y
  ) {
  int index = get_global_id(0);
  y[index] = fmax(a[index], 0);
}
)CLC";

template <>
template <>
bool ReluFunctor<OpenCLContext>::
operator()(const int N, const float* X, float* Y, OpenCLContext* context_) const {
  auto kernel = context_->BuildKernelCached(kRelu, "", "kRelu");
  auto xBuffer = (cl::Buffer*)X;
  auto yBuffer = (cl::Buffer*)Y;
  OPENCL_CHECK(kernel.setArg(0, *xBuffer));
  OPENCL_CHECK(kernel.setArg(1, *yBuffer));

  struct timeval tv;
  gettimeofday(&tv, NULL);
  long long start = tv.tv_sec * 1000000 + tv.tv_usec;

  cl::Event event;
  context_->enqueue(
    kernel,
    cl::NullRange,
    cl::NDRange(N),
    cl::NullRange,
    NULL,
    &event);

  gettimeofday(&tv, NULL);
  long long end = tv.tv_sec * 1000000 + tv.tv_usec;

  std::stringstream outstr;
  outstr << "ReluOp " << end << " cpu time delta: " << end - start;
  outstr << " N: " << N;
  context_->LogProfilingInfo(event, outstr.str());

  return true;
}

REGISTER_OPENCL_OPERATOR(
    Relu,
    UnaryElementwiseOp<
        TensorTypes<float>,
        OpenCLContext,
        ReluFunctor<OpenCLContext>>);

} // namespace caffe2


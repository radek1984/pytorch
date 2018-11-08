#ifndef CAFFE2_OPENCL_CONTEXT_H_
#define CAFFE2_OPENCL_CONTEXT_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"

#define CL_HPP_ENABLE_EXCEPTIONS 1
#define CL_HPP_CL_1_2_DEFAULT_BUILD 1
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif


#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "caffe2_opencl_test-jni"
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#else
#define ALOGI(...) printf(__VA_ARGS__)
#endif

#include <mutex>

#define OPENCL_CHECK(v) do {\
    cl_int _err = (v); \
    if (_err != CL_SUCCESS) { \
      CAFFE_THROW("OpenCL Error:", _err, " on line ", __LINE__);\
    }\
  } while(0)

#define CAFFE2_OPENCL_EXPORT __attribute__((__visibility__("default")))
#define CAFFE2_OPENCL_API CAFFE2_OPENCL_EXPORT

namespace caffe2 {

struct CAFFE2_OPENCL_API ThreadLocalOpenCLObjects {
 private:
  ThreadLocalOpenCLObjects();
  ThreadLocalOpenCLObjects(const ThreadLocalOpenCLObjects &) = delete;
  ThreadLocalOpenCLObjects(ThreadLocalOpenCLObjects&&) = delete;
  bool profiling_info_enabled_;
 public:
  void LogProfilingInfo(const cl::Event& ev, const std::string& str);
  void PrintProfilingLogs();
  cl::CommandQueue& GetQueue(int queue_id);
  //cl::Platform platform;
  //cl::Device device;
  //std::vector<cl::Device> devices;
  //cl::Context context;
  std::vector<cl::CommandQueue> queues;
  std::vector<std::pair<cl::Event, std::string>> events_profiling_log;
  friend class OpenCLContext;
};

class CAFFE2_OPENCL_API OpenCLContext final : public BaseContext {
 public:
  //TODO: Move members to private section used friends for access
  static cl::Context context;
  static cl::Platform platform;
  static cl::Device computing_device;
  static std::vector<cl::Device> devices;
  static bool initialized;

  explicit OpenCLContext() {}
  explicit OpenCLContext(const at::Device& device)
      : OpenCLContext(DeviceToOption(device)) {}
  explicit OpenCLContext(const DeviceOption& option) {
    // TODO: Investigate why this assert was introduced
    //
    // It is not clear why this assert was introduced
    // it fails during initialization in opencl conv operator
    // test: conv_op_test.cc:193
    // It seams that copy operator can contain
    // OpenCLContext but transfer data from/to CPU context.
    //
    // DCHECK_EQ(option.device_type(), PROTO_OPENCL);
    // invalid cast to abstract type
    OpenCLContext();

    if (!OpenCLContext::initialized) {

      const auto platform_id = 0;
      const auto device_id = 0;

      auto platforms = std::vector<cl::Platform>();
      OPENCL_CHECK(cl::Platform::get(&platforms));
      if (platforms.size() == 0 || platform_id >= platforms.size()) {
        CAFFE_THROW("Cannot find platform for OpenCL.");
      }
      platform = platforms[platform_id];

      devices = std::vector<cl::Device>();
      platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
      if (devices.size() == 0 || device_id >= devices.size()) {
        CAFFE_THROW("Cannot find OpenCL compatible device.");
      }
      computing_device = devices[device_id];
      cl_int err;
      context = cl::Context({computing_device}, nullptr, nullptr , nullptr, &err);
      OPENCL_CHECK(err);
      OpenCLContext::initialized = true;
    }
  }
  ~OpenCLContext() override {
    FinishDeviceComputation();
  }

  /*
   * Everything below is basically boiler plate for Context classes
   */

  //TODO: get rid of these two, cause we have allocator now:
  static std::pair<void*, MemoryDeleter> New(size_t nbytes);

  static void Delete(void* data);

  void LogProfilingInfo(const cl::Event& ev, const std::string& str);
  
  static constexpr DeviceType GetDeviceType() {
    return OPENCL;
  }

  template <class SrcContext, class DstContext>
  void CopyBytes(size_t nbytes, const void *src, void *dst);

  void CopyBytesSameDevice(size_t nbytes, const void* src, void* dst) override {
    cl::Event event;
    OPENCL_CHECK(
        queue().enqueueCopyBuffer(*((const cl::Buffer*)src), *((const cl::Buffer*)dst), 0, 0, nbytes,
        NULL,
        &event)
    );
  }

  void CopyBytesToCPU(size_t nbytes, const void* src, void* dst) override {
    queue().enqueueReadBuffer(*((cl::Buffer*)src), CL_TRUE, 0, nbytes, static_cast<char*>(dst));
  }

  void CopyBytesFromCPU(size_t nbytes, const void* src, void* dst) override {
    queue().enqueueWriteBuffer(*((cl::Buffer*)(dst)), CL_FALSE, 0, nbytes, static_cast<const char*>(src));
  }

  template <typename T, class SrcContext, class DstContext>
  inline void Copy(int n, const T* src, T* dst) {
    CopyBytes<SrcContext, DstContext>(
        n * sizeof(T), static_cast<const void*>(src), static_cast<void*>(dst));
  }

  at::Device device() const override {
    // Similar implementation exist for CUDA
    // return at::Device(OPENCL, gpu_id_);
    //TODO: Implement.
    assert(false && "NOT IMPLEMENTED");
  }
  at::DeviceType device_type() const override {
    // Similar implementation exist for CUDA
    // return OPENCL;
    // TODO: Implement.
    assert(false && "NOT IMPLEMENTED");
  }

  template <class SrcContext, class DstContext>
  inline void
  CopyItems(const TypeMeta& meta, size_t n, const void* src, void* dst);

  void SwitchToDevice(int queue_id) override {
    queue_id_ = queue_id;
  }

  using BaseContext::SwitchToDevice;

  inline void WaitEvent(const Event& ev) override {
    ev.Wait(OPENCL, this);
  }

  inline void Record(Event* ev, const char* err_msg = nullptr) const override {
    CAFFE_ENFORCE(ev, "Event must not be null.");
    ev->Record(OPENCL, this, err_msg);
  }

  void FinishDeviceComputation() override {
    queue().finish();
    openCL_objects_.PrintProfilingLogs();
  }

  cl::CommandQueue& queue() const { //TODO: change name to queue
    return openCL_objects_.GetQueue(queue_id_);
  }
  cl::Context& clContext() const {
    return OpenCLContext::context;
  }

  void enqueue(
      const cl::Kernel& kernel,
      const cl::NDRange& offset,
      const cl::NDRange& global,
      const cl::NDRange& local = cl::NullRange,
      const VECTOR_CLASS<cl::Event>* events = nullptr,
      cl::Event* event = nullptr) const {
    OPENCL_CHECK(queue().enqueueNDRangeKernel(kernel, offset, global, local, events, event));
  }

  cl::Kernel BuildKernel(const char* src, std::string additional_options = "", const char* fn_name = "K");

  cl::Kernel BuildKernelCachedId(const std::string &cacheId, const char *src, std::string compile_options = "", const char *function_name = "K");
  cl::Kernel BuildKernelCached(const char *src, std::string compile_options = "", const char *function_name = "K");

  template <class SrcContext, class DstContext>
  void enqueueCopyBytes(size_t nbytes, const void *src, void *dst);

  // Disabled for PyTorch
  // It causes ambiguous concretization of give tensor fill operator.
  // long int was cast both to size_t and int so
  // it was not possible to select one of Copy overloaded versions.
  //
  //
  // For compatibility with old style copy
  //template <typename T, class SrcContext, class DstContext>
  //inline void Copy(int n, const T* src, T* dst) {
  //  if (std::is_fundamental<T>::value) {
  //    CopyBytes<SrcContext, DstContext>(n * sizeof(T),
  //                                   static_cast<const void*>(src),
  //                                   static_cast<void*>(dst));
  //  } else {
  //    for (int i = 0; i < n; ++i) {
  //      dst[i] = src[i];
  //    }
  //  }
  //}

  //Definition not necessary for 20181026 version
  //template <typename T, class SrcContext, class DstContext>
  //inline void Copy(int n, const T* src, T* dst) {
  //  //dst->Resize(src.dims());
  //  //size_t n = src->size();
  //  CopyBytes<SrcContext, DstContext>(n * sizeof(T),
  //                               static_cast<const void*>(src),
  //                               static_cast<void*>(dst));
  //}

  //Not used enymore
  //template <typename T, class SrcContext, class DstContext>
  //inline void Copy(const Tensor<SrcContext>& src, Tensor<DstContext>& dst) {
  //  Copy<T, T>(src, dst);
  //}

  // By default CUDA operators have async device parts
  static bool HasAsyncPartDefault() {
    return true;
  }

  static bool SupportsAsyncScheduling() {
    return true;
  }

  static bool IsStreamFree(const DeviceOption& option, int stream_id) {
    // FIXME: Not implemented
    //       Is it necessary for OPENCL?
    CAFFE_THROW("IsStreamFree not implemented for OpenCL");
    // This is implementation from CUDA
    // auto stream = CUDAContext::cuda_stream(option.cuda_gpu_id(), stream_id);
    // return cudaStreamQuery(stream) == cudaSuccess;
  }
  std::string BuildArgumentList(std::vector<std::pair<std::string, std::string>> args);

protected:
  int queue_id_ = 0;
  static thread_local ThreadLocalOpenCLObjects openCL_objects_;
};

//typedef Tensor<OpenCLContext> TensorCL;
using TensorCL = Tensor;

} // namespace caffe2

#endif // ifndef CAFFE2_OPENCL_CONTEXT_H_



#include "caffe2/core/flags.h"
#include "caffe2/core/blob.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/relu_op.h"
#include "caffe2/core/net.h"
#include "caffe2/core/workspace.h"
#include <gtest/gtest.h>
#include "../context.h"

#include <google/protobuf/text_format.h>

namespace caffe2 {

const std::string init_net = R"(
name: "transpose_test_model_init"
op {
  output: "X"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 4
    ints: 5
  }
  arg {
  name: "values"
    floats: 1
    floats: 1
    floats: 1
    floats: 1
    floats: 1
    floats: 1
    floats: 1
    floats: 1
    floats: 1
    floats: 1
    floats: 1
    floats: 1
    floats: 1
    floats: 1
    floats: 1
    floats: 1
    floats: 1
    floats: 1
    floats: 1
    floats: 1
  }
}
)";

const std::string pred_net = R"DOC(
      name: "relu_test"
      external_input: "X"
      op {
        input: "X"
        output: "Y"
        type: "Relu"
        name: "relu"
      }

)DOC";

TEST(OpenCL, ReluOpTest) {
  caffe2::OpenCLContext context;

  Workspace ws;
  {
    caffe2::NetDef init, predict;
    ::google::protobuf::TextFormat::ParseFromString(init_net, &init);
    ::google::protobuf::TextFormat::ParseFromString(pred_net, &predict);

    init.mutable_device_option()->set_device_type(PROTO_OPENCL);
    predict.mutable_device_option()->set_device_type(PROTO_OPENCL);

    //init net:
    NetBase *res = ws.CreateNet(init);
    EXPECT_NE(res, nullptr);
    ws.RunNet(res->Name());

    //predict net:
    res = ws.CreateNet(predict);
    EXPECT_NE(res, nullptr);
    ws.RunNet(res->Name());
  }

  Blob* Yblob = ws.GetBlob("Y");
  EXPECT_NE(nullptr, Yblob);
  auto Y = Yblob->GetMutable<TensorCL>();
  EXPECT_EQ(Y->size(), 20);
  cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();
  std::vector<float> data(Y->size());

  context.CopyBytesToCPU(Y->size() * sizeof(float), yBuffer, &data[0]);

  for (int i = 0; i < Y->size(); ++i) {
    EXPECT_EQ(data[i], 1);
    printf("%d - %f\n", i, data[i]);
  }
}

}  // namespace caffe2

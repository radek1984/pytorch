
#include <gtest/gtest.h>
#include "caffe2/operators/leaky_relu_op.h"
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
  }
  arg {
  name: "values"
    floats: -1.0
    floats: -2.5
    floats: 0.0
    floats: 1.0
  }
}
)";


const std::string pred_net = R"DOC(
      name: "lrelu_test"
      external_input: "X"
      op {
        input: "X"
        output: "Y"
        type: "LeakyRelu"
        arg {
          name: "alpha"
          f: 0.5
        }
      }
)DOC";

TEST(OpenCL, leakyReluTest) {

  const int expected_size = 4;
  caffe2::DeviceOption option;
  OpenCLContext context(option);
  Workspace ws;
  std::vector<float> expected_output({-0.5, -1.25, 0.0, 1.0});
  std::vector<long int> expected_shape({expected_size});
  //X:
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

  EXPECT_EQ(Y->size(), expected_size);

  cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();
  std::vector<float> data(Y->size());

  context.CopyBytesToCPU(Y->size() * sizeof(float), yBuffer, &data[0]);

  for (int i = 0; i < Y->size(); ++i) {
    EXPECT_NEAR(data[i], expected_output[i], 0.001);
  }

}

}

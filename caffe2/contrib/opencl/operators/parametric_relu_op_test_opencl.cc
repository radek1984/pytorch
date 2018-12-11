
#include "caffe2/core/blob.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/common.h"
#include "caffe2/core/init.h"
#include "caffe2/core/workspace.h"
#include "caffe2/core/logging.h"
#include "../context.h"
#include "caffe2/operators/prelu_op.h"
#include <google/protobuf/text_format.h>

#include <gtest/gtest.h>


namespace caffe2 {

const std::string pred_net = R"DOC(
      name: "prelu_test"
      external_input: "X"
      external_input: "W"
      op {
        input: "X"
        input: "W"
        output: "Y"
        type: "PRelu"
      }
)DOC";

void runTest(const std::string &init_net,
             std::vector<float>& expected_output,
             std::vector<long int>& expected_shape
              ) {
  DeviceOption option;
  OpenCLContext context(option);
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

  EXPECT_EQ(Y->size(), std::accumulate(expected_shape.begin(), expected_shape.end(),
                                      1, std::multiplies<double>()));

  cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();
  std::vector<float> data(Y->size());
  context.CopyBytes<OpenCLContext, CPUContext>(Y->size() * sizeof(float),
                                              (void*)yBuffer, (void*)&data[0]);

  for (int i = 0; i < Y->size(); ++i) {
    EXPECT_NEAR(data[i], expected_output[i], 0.001);
  }
}

TEST(OpenCL, PRelu_test1) {
  const std::string init_net = R"(
    name: "transpose_test_model_init"
    op {
      output: "X"
      type: "GivenTensorFill"
      arg {
        name: "shape"
        ints: 1
        ints: 1
        ints: 1
        ints: 4
      }
      arg {
      name: "values"
        floats: -1.0
        floats: -2.5
        floats: 0.0
        floats: 1.1
      }
    }
    op {
      output: "W"
      type: "GivenTensorFill"
      arg {
        name: "shape"
        ints: 1
      }
      arg {
      name: "values"
        floats: 2.0
      }
    }
  )";
  std::vector<float> expected_output({-2.0, -5.0, 0.0, 1.1});
  std::vector<long int> expected_shape({1, 1, 1, 4}); //1 image, 1 channel , h 1, w 4

  runTest(init_net, expected_output, expected_shape);
}

TEST(OpenCL, PRelu_test2) {
  const std::string init_net = R"(
    name: "transpose_test_model_init"
    op {
      output: "X"
      type: "GivenTensorFill"
      arg {
        name: "shape"
        ints: 1
        ints: 2
        ints: 1
        ints: 2
      }
      arg {
      name: "values"
        floats: -1.0
        floats: -2.5
        floats: 0.1
        floats: -1.1
      }
    }
    op {
      output: "W"
      type: "GivenTensorFill"
      arg {
        name: "shape"
        ints: 2
      }
      arg {
      name: "values"
        floats: 2.0
        floats: 3.0
      }
    }
  )";
  std::vector<float> expected_output({-2.0, -5.0, 0.1, -3.3});  //1 image, 2 channels , h 1, w 2
  std::vector<long int> expected_shape({1, 2, 1, 2});

  runTest(init_net, expected_output, expected_shape);
}

TEST(OpenCL, PRelu_test3) {//shared weight:
  const std::string init_net = R"(
    name: "transpose_test_model_init"
    op {
      output: "X"
      type: "GivenTensorFill"
      arg {
        name: "shape"
        ints: 2
        ints: 2
        ints: 3
        ints: 1
      }
      arg {
      name: "values"
        floats: -1.0
        floats: -2.5
        floats: 1.5
        floats: 47.0
        floats: 0.0
        floats: -1.1
        floats: -2.0
        floats: -5.0
        floats: 3.0
        floats: 94.0
        floats: 0.0
        floats: -2.2
      }
    }
    op {
      output: "W"
      type: "GivenTensorFill"
      arg {
        name: "shape"
        ints: 1
      }
      arg {
      name: "values"
        floats: 2.0
      }
    }
  )";
  std::vector<float> expected_output({
                                        //im0:
                                        -2.0, -5.0, 1.5,
                                        47.0, 0.0, -2.2,
                                        //im1:
                                        -4.0, -10.0, 3.0,
                                        94.0, 0.0, -4.4,

                                     });
  std::vector<long int> expected_shape({2, 2, 3, 1});

  runTest(init_net, expected_output, expected_shape);
}

TEST(OpenCL, PRelu_test4) {//non shared weights:

  const std::string init_net = R"(
    name: "transpose_test_model_init"
    op {
      output: "X"
      type: "GivenTensorFill"
      arg {
        name: "shape"
        ints: 2
        ints: 2
        ints: 3
        ints: 1
      }
      arg {
      name: "values"
        floats: -1.0
        floats: -2.5
        floats: 1.5
        floats: 47.0
        floats: 0.0
        floats: -1.1
        floats: -2.0
        floats: -5.0
        floats: 3.0
        floats: 94.0
        floats: 0.0
        floats: -2.2
      }
    }
    op {
      output: "W"
      type: "GivenTensorFill"
      arg {
        name: "shape"
        ints: 2
      }
      arg {
      name: "values"
        floats: 2.0
        floats: -1.0
      }
    }
  )";
  std::vector<float> expected_output({
                                        //im0:
                                        -2.0, -5.0, 1.5,
                                        47.0, 0.0, 1.1,
                                        //im1:
                                        -4.0, -10.0, 3.0,
                                        94.0, 0.0, 2.2,

                                     });
  std::vector<long int> expected_shape({2, 2, 3, 1});

  runTest(init_net, expected_output, expected_shape);
}

}

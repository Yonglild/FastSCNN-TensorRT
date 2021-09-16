//
// Created by wyl on 2021/8/19.
//

#ifndef TENSORRTXPRACTICE_FASTSCNN_H
#define TENSORRTXPRACTICE_FASTSCNN_H

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

static const int INPUT_H = 1024;
static const int INPUT_W = 2048;
static const int NUM_CLASSES = 19;
static const int BATCH_SIZE = 1;
static const int OUTPUT_SIZE = BATCH_SIZE * INPUT_H * INPUT_W;

// debug 用于打印层信息
//static const int DEBUG_H = 1024;
//static const int DEBUG_W = 2048;
//static const int OUTPUT_SIZE = 1 * DEBUG_H * DEBUG_W;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;
using namespace std;
static Logger gLogger;

void debug_print(ITensor *input_tensor, std::string head);

std::map<std::string, Weights> loadWeights(const std::string file);
ITensor *MeanStd(INetworkDefinition *network, ITensor *input, float *mean, float *std, bool div255);

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps);
IActivationLayer* ConvBnRelu(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                             int outch, int ksize, int stride, int pad, std::string lname);
IActivationLayer* DWConv(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                         int outch, int stride, int group, const std::string& lname);
IActivationLayer* DSConv(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                         int dwch, int outch, int stride, int pad, int group, const std::string& lname);
ILayer* LinearBottleneck(INetworkDefinition* network, map<string, Weights>& weightMap, ITensor& input,
                         int inch, int outch, int stride, int expansion, const string& lname);
ILayer* addUpsample(INetworkDefinition* network, ITensor& input, Dims outdims);
ILayer* PyramidPooling(INetworkDefinition* network, map<string, Weights>& weightMap, ITensor& input,
                       int outch, const string& lname);
IActivationLayer* FeatureFusionModule(INetworkDefinition* network, map<string, Weights>& weightMap, ITensor& lowerFeature, ITensor& highFeature);
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, map<string, Weights>& weightMap, DataType dt);

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, const string& wtsPath);
void doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, int batchSize);
cv::Mat createLTU(int len);

#endif //TENSORRTXPRACTICE_FASTSCNN_H

//
// Created by wyl on 2021/8/19.
//

#include "fastscnn.h"

#define DEVICE 0 // GPU id

using namespace std;

void debug_print(ITensor *input_tensor, std::string head)
{
    std::cout << head << " : ";

    for (int i = 0; i < input_tensor->getDimensions().nbDims; i++)
    {
        std::cout << input_tensor->getDimensions().d[i] << " ";
    }
    std::cout << std::endl;
}

std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps){
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for(int i = 0; i < len; i++){
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;    // 将计算出的scale和shift等参数放进weightMap中
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

IActivationLayer* ConvBnRelu(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                              int outch, int ksize, int stride, int pad, std::string lname){
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[lname + ".0.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setPaddingNd(DimsHW{pad, pad});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    return relu1;
}

IActivationLayer* DWConv(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
        int outch, int stride, int group, const std::string& lname){
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{3, 3}, weightMap[lname + ".0.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setPaddingNd(DimsHW{1, 1});
    conv1->setNbGroups(group);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    return relu1;
}

/**
 * @param lname  "global_feature_extractor.bottleneck1.0.block", "global_feature_extractor.bottleneck2.1.block"
 * @return
 */
ILayer* LinearBottleneck(INetworkDefinition* network, map<string, Weights>& weightMap, ITensor& input,
        int inch, int outch, int stride, int expansion, const string& lname){
    bool use_short = stride == 1 && inch == outch;
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IActivationLayer* relu1 = ConvBnRelu(network, weightMap, input, inch * expansion, 1, 1, 0, lname + ".0.conv");
    IActivationLayer* relu2 = DWConv(network, weightMap, *relu1->getOutput(0), inch * expansion, stride, inch*expansion, lname + ".1.conv");
    IConvolutionLayer* conv1 = network->addConvolutionNd(*relu2->getOutput(0), outch, DimsHW{1, 1}, weightMap[lname + ".2.weight"], emptywts);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".3", 1e-5);
    IElementWiseLayer* out;
    if(use_short){
        out = network->addElementWise(input, *bn1->getOutput(0), ElementWiseOperation::kSUM);
        return out;
    }
    return bn1;
}


/**
 * @param lname  for example:"learning_to_downsample.dsconv1"
 */
IActivationLayer* DSConv(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
       int dwch, int outch, int stride, int pad, int group, const std::string& lname){
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, dwch, DimsHW{3, 3}, weightMap[lname + ".conv.0.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setPaddingNd(DimsHW{pad, pad});
    conv1->setNbGroups(group);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".conv.1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{1, 1}, weightMap[lname + ".conv.3.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{1, 1});
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".conv.4", 1e-5);
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    return relu2;
}

ILayer* addUpsample(INetworkDefinition* network, ITensor& input, Dims outdims){
    IResizeLayer* upSample = network->addResize(input);
    upSample->setResizeMode(ResizeMode::kLINEAR);
    upSample->setOutputDimensions(outdims);
    upSample->setAlignCorners(true);
    return upSample;
}

ILayer* PyramidPooling(INetworkDefinition* network, map<string, Weights>& weightMap, ITensor& input,
        int outch, const string& lname){
    Dims size = input.getDimensions();
    size.d[0] = 32;

    IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{size.d[1], size.d[2]});
    IActivationLayer* relu1 = ConvBnRelu(network, weightMap, *pool1->getOutput(0), 32, 1, 1, 0, lname + ".conv1.conv");
    ILayer* feat1 = addUpsample(network, *relu1->getOutput(0), size);

    int strideH = floor(size.d[1] / 2);
    int strideW = floor(size.d[2] / 2);
    int kernelH = size.d[1] - (2 - 1) * strideH;
    int kernelW = size.d[2] - (2 - 1) * strideW;
    IPoolingLayer* pool2 = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{kernelH,kernelW});
    pool2->setStrideNd(DimsHW{strideH,strideW});
    debug_print(pool2->getOutput(0), "pool2");
    IActivationLayer* relu2 = ConvBnRelu(network, weightMap, *pool2->getOutput(0), 32, 1, 1, 0, lname + ".conv2.conv");
    ILayer* feat2 = addUpsample(network, *relu2->getOutput(0), size);

    strideH = floor(size.d[1] / 3);
    strideW = floor(size.d[2] / 3);
    kernelH = size.d[1] - (3 - 1) * strideH;
    kernelW = size.d[2] - (3 - 1) * strideW;
    IPoolingLayer* pool3 = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{kernelH,kernelW});
    pool3->setStrideNd(DimsHW{strideH,strideW});
    debug_print(pool3->getOutput(0), "pool3");
    IActivationLayer* relu3 = ConvBnRelu(network, weightMap, *pool3->getOutput(0), 32, 1, 1, 0, lname + ".conv3.conv");
    ILayer* feat3 = addUpsample(network, *relu3->getOutput(0), size);

    strideH = floor(size.d[1] / 6);
    strideW = floor(size.d[2] / 6);
    kernelH = size.d[1] - (6 - 1) * strideH;
    kernelW = size.d[2] - (6 - 1) * strideW;
    IPoolingLayer* pool4 = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{kernelH,kernelW});
    pool4->setStrideNd(DimsHW{strideH,strideW});
    debug_print(pool4->getOutput(0), "pool4");
    IActivationLayer* relu4 = ConvBnRelu(network, weightMap, *pool4->getOutput(0), 32, 1, 1, 0, lname + ".conv4.conv");
    ILayer* feat4 = addUpsample(network, *relu4->getOutput(0), size);

    ITensor* concatTensors[5] = {&input, feat1->getOutput(0), feat2->getOutput(0), feat3->getOutput(0), feat4->getOutput(0)};
    auto concat1 = network->addConcatenation(concatTensors, 5);
    concat1->setAxis(0);
    debug_print(concat1->getOutput(0), "concat");

    IActivationLayer* relu5 = ConvBnRelu(network, weightMap, *concat1->getOutput(0), outch, 1, 1, 0, lname + ".out.conv");
    return relu5;
}

IActivationLayer* FeatureFusionModule(INetworkDefinition* network, map<string, Weights>& weightMap, ITensor& lowerFeature, ITensor& highFeature){
    Dims outdims = lowerFeature.getDimensions();
    outdims.d[1] = outdims.d[1] * 4;
    outdims.d[2] = outdims.d[2] * 4;
    ILayer* up1 = addUpsample(network, lowerFeature, outdims);
    IActivationLayer* relu1 =  DWConv(network, weightMap, *up1->getOutput(0), 128, 1, 128, "feature_fusion.dwconv.conv");
    debug_print(relu1->getOutput(0), "feature_fusion.dwconv.conv");

    // conv_lower_res
    IConvolutionLayer* conv1 = network->addConvolutionNd(*relu1->getOutput(0), 128, DimsHW{1,1}, weightMap["feature_fusion.conv_lower_res.0.weight"], weightMap["feature_fusion.conv_lower_res.0.bias"]);
    conv1->setStrideNd(DimsHW{1, 1});
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "feature_fusion.conv_lower_res.1", 1e-5);
    debug_print(bn1->getOutput(0), "feature_fusion.conv_lower_res");

    // conv_higher_res
    IConvolutionLayer* conv2 = network->addConvolutionNd(highFeature, 128, DimsHW{1, 1}, weightMap["feature_fusion.conv_higher_res.0.weight"], weightMap["feature_fusion.conv_higher_res.0.bias"]);
    conv2->setStrideNd(DimsHW{1, 1});
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), "feature_fusion.conv_higher_res.1", 1e-5);
    debug_print(bn2->getOutput(0), "feature_fusion.conv_higher_res");

    IElementWiseLayer* ew1 = network->addElementWise(*bn1->getOutput(0), *bn2->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* relu2 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    return relu2;
}

ITensor *MeanStd(INetworkDefinition *network, ITensor *input, float *mean, float *std, bool div255)
{
    if (div255)
    {
        Weights Div_225{DataType::kFLOAT, nullptr, 3};
        float *wgt = reinterpret_cast<float *>(malloc(sizeof(float) * 3));
        for (int i = 0; i < 3; ++i)
        {
            wgt[i] = 255.0f;
        }
        Div_225.values = wgt;
        IConstantLayer *d = network->addConstant(Dims3{3, 1, 1}, Div_225);
        input = network->addElementWise(*input, *d->getOutput(0), ElementWiseOperation::kDIV)->getOutput(0);
    }
    Weights Mean{DataType::kFLOAT, nullptr, 3};
    Mean.values = mean;
    IConstantLayer *m = network->addConstant(Dims3{3, 1, 1}, Mean);
    IElementWiseLayer *sub_mean = network->addElementWise(*input, *m->getOutput(0), ElementWiseOperation::kSUB);
    if (std != nullptr)
    {
        Weights Std{DataType::kFLOAT, nullptr, 3};
        Std.values = std;
        IConstantLayer *s = network->addConstant(Dims3{3, 1, 1}, Std);
        IElementWiseLayer *std_mean = network->addElementWise(*sub_mean->getOutput(0), *s->getOutput(0), ElementWiseOperation::kDIV);
        return std_mean->getOutput(0);
    }
    else
    {
        return sub_mean->getOutput(0);
    }
}

ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, map<string, Weights>& weightMap, DataType dt) {
    INetworkDefinition *network = builder->createNetworkV2(0U);

    ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{INPUT_H, INPUT_W, 3});
    assert(data);

    // hwc to chw
    auto ps = network->addShuffle(*data);
    ps->setFirstTranspose(nvinfer1::Permutation{2, 0, 1});
    float mean[3] = {0.485, 0.456, 0.406};
    float std[3] = {0.229, 0.224, 0.225};
    ITensor *preinput = MeanStd(network, ps->getOutput(0), mean, std, true);

    // learningToDownsample.conv
    auto relu1 = ConvBnRelu(network, weightMap, *preinput, 32, 3, 2, 0, "learning_to_downsample.conv.conv");

    assert(relu1);
    debug_print(relu1->getOutput(0), "learning_to_downsample.conv.conv.");
    // learningToDownsample.dsconv
    auto relu2 = DSConv(network, weightMap, *relu1->getOutput(0), 32, 48, 2, 1, 32, "learning_to_downsample.dsconv1");
    auto relu3 = DSConv(network, weightMap, *relu2->getOutput(0), 48, 64, 2, 1, 48, "learning_to_downsample.dsconv2");
    debug_print(relu3->getOutput(0), "learning_to_downsample.dsconv2");

    // globalFeatureExtractor.bottleneck
    auto bottle1 = LinearBottleneck(network, weightMap, *relu3->getOutput(0), 64, 64, 2, 6,
                                    "global_feature_extractor.bottleneck1.0.block");
    auto bottle2 = LinearBottleneck(network, weightMap, *bottle1->getOutput(0), 64, 64, 1, 6,
                                    "global_feature_extractor.bottleneck1.1.block");
    auto bottle3 = LinearBottleneck(network, weightMap, *bottle2->getOutput(0), 64, 64, 1, 6,
                                    "global_feature_extractor.bottleneck1.2.block");
    debug_print(bottle3->getOutput(0), "global_feature_extractor.bottleneck1.2.block");


    auto bottle4 = LinearBottleneck(network, weightMap, *bottle3->getOutput(0), 64, 96, 2, 6,
                                    "global_feature_extractor.bottleneck2.0.block");
    auto bottle5 = LinearBottleneck(network, weightMap, *bottle4->getOutput(0), 96, 96, 1, 6,
                                    "global_feature_extractor.bottleneck2.1.block");
    auto bottle6 = LinearBottleneck(network, weightMap, *bottle5->getOutput(0), 96, 96, 1, 6,
                                    "global_feature_extractor.bottleneck2.2.block");
    debug_print(bottle6->getOutput(0), "global_feature_extractor.bottleneck2.2.block");


    auto bottle7 = LinearBottleneck(network, weightMap, *bottle6->getOutput(0), 96, 128, 1, 6,
                                    "global_feature_extractor.bottleneck3.0.block");
    auto bottle8 = LinearBottleneck(network, weightMap, *bottle7->getOutput(0), 128, 128, 1, 6,
                                    "global_feature_extractor.bottleneck3.1.block");
    auto bottle9 = LinearBottleneck(network, weightMap, *bottle8->getOutput(0), 128, 128, 1, 6,
                                    "global_feature_extractor.bottleneck3.2.block");
    debug_print(bottle9->getOutput(0), "global_feature_extractor.bottleneck3.2.block");

    // globalFeatureExtractor.ppm
    auto relu4 = PyramidPooling(network, weightMap, *bottle9->getOutput(0), 128, "global_feature_extractor.ppm");
    debug_print(relu4->getOutput(0), "global_feature_extractor.ppm");


    // feature_fusion
    auto relu5 = FeatureFusionModule(network, weightMap, *relu4->getOutput(0), *relu3->getOutput(0));
    debug_print(relu5->getOutput(0), "feature_fusion.conv_lower_res");

    // classifier
    auto relu6 = DSConv(network, weightMap, *relu5->getOutput(0), 128, 128, 1, 1, 128, "classifier.dsconv1");
    auto relu7 = DSConv(network, weightMap, *relu6->getOutput(0), 128, 128, 1, 1, 128, "classifier.dsconv2");
    auto conv1 = network->addConvolutionNd(*relu7->getOutput(0), NUM_CLASSES, DimsHW{1, 1}, weightMap["classifier.conv.1.weight"], weightMap["classifier.conv.1.bias"]);
    conv1->setStrideNd(DimsHW{1, 1});

    Dims outdims = conv1->getOutput(0)->getDimensions();
    outdims.d[1] = INPUT_H;
    outdims.d[2] = INPUT_W;
    auto score = addUpsample(network, *conv1->getOutput(0), outdims);

    auto topk = network->addTopK(*score->getOutput(0), TopKOperation::kMAX, 1, 0X01);
    topk->getOutput(1)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*topk->getOutput(1));

    builder->setMaxWorkspaceSize(maxBatchSize);
    config->setMaxWorkspaceSize((1<<30));   // 1G

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build success!" << std::endl;
    network->destroy();
    for (auto &mem : weightMap)
    {
        free((void *)(mem.second.values));
    }
    std::cout << "free mem sucess!" << std::endl;

    return engine;
}


void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, const string& wtsPath)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();    // 配置类指针，可以设置最大空间

    // Create model to populate the network, then set the outputs and create an engine
    std::map<std::string, Weights> weightMap = loadWeights(wtsPath);

    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, weightMap, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, int batchSize){
    context.enqueue(batchSize, buffers, stream, nullptr);
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();
}

cv::Mat createLTU(int len)
{
    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar *p = lookUpTable.data;
    for (int j = 0; j < 256; ++j)
    {
        p[j] = (j * (256 / len) > 255) ? uchar(255) : (uchar)(j * (256 / len));
    }
    return lookUpTable;
}


int main(int argc, char** argv){
    std::string wtsPath = "../weights/fastscnn_citys.wts";
    std::string engine_name = "../weights/fastscnn_citys.engine";
    std::string mode = std::string(argv[1]);        // "s":生成engine文件  or "d":模型推理
    cv::String folder = "/home/wyl/CLionProjects/TensorrtxPractice/FastSCNN-TensorRT/frankfurt_000001_058914_leftImg8bit.png";
    cudaSetDevice(DEVICE);
    std::string img_dir;
    // parse args

    if(argc !=2){
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./fastscnn -s // serialize model to plan file" << std::endl;
        std::cerr << "./fastscnn -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    if (mode == "-s")
    {
        IHostMemory *modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream, wtsPath);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }

    // deserialize the .engine and run inference
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file(engine_name, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    else
    {
        std::cerr << "could not open plan file" << std::endl;
    }

    // prepare input data ---------------------------
    cudaSetDeviceFlags(cudaDeviceMapHost);
    float *data;
    int *prob; // using int. output is index
    CHECK(cudaHostAlloc((void **)&data, BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float), cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void **)&prob, BATCH_SIZE * OUTPUT_SIZE * sizeof(int), cudaHostAllocMapped));

    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    void *buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    vector<cv::String> file_names;
    cv::glob(folder, file_names);
    for (int f = 0; f < (int)file_names.size(); f++)
    {
        std::cout << file_names[f] << std::endl;
        cv::Mat pr_img;
        cv::Mat img_BGR = cv::imread(img_dir + "/" + file_names[f], 1); // BGR
        cv::Mat img;
        cv::cvtColor(img_BGR, img, cv::COLOR_BGR2RGB);
        if (img.empty())
            continue;
        cv::resize(img, pr_img, cv::Size(INPUT_W, INPUT_H));
        img = pr_img.clone(); // for img show
        pr_img.convertTo(pr_img, CV_32FC3);
        if (!pr_img.isContinuous())
        {
            pr_img = pr_img.clone();
        }
        std::memcpy(data, pr_img.data, BATCH_SIZE * 3 * INPUT_W * INPUT_H * sizeof(float));

        cudaHostGetDevicePointer((void **)&buffers[inputIndex], (void *)data, 0);  // buffers[inputIndex]-->data
        cudaHostGetDevicePointer((void **)&buffers[outputIndex], (void *)prob, 0); // buffers[outputIndex] --> prob

        // Run inference
        auto start = std::chrono::high_resolution_clock::now();
        doInference(*context, stream, buffers, BATCH_SIZE);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "infer time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        cv::Mat outimg(INPUT_H, INPUT_W, CV_8UC1);
        for (int row = 0; row < INPUT_H; ++row)
        {
            uchar *uc_pixel = outimg.data + row * outimg.step;
            for (int col = 0; col < INPUT_W; ++col)
            {
                uc_pixel[col] = (uchar)prob[row * INPUT_W + col];
            }
        }
        cv::Mat im_color;
        cv::cvtColor(outimg, im_color, cv::COLOR_GRAY2RGB);
        cv::Mat lut = createLTU(NUM_CLASSES);
        cv::LUT(im_color, lut, im_color);
        // false color
        cv::cvtColor(im_color, im_color, cv::COLOR_RGB2GRAY);
        cv::applyColorMap(im_color, im_color, cv::COLORMAP_HOT);
        // cv::imshow("False Color Map", im_color);

        cv::imwrite(std::to_string(f) + "_false_color_map.png", im_color);
        //fusion
        cv::Mat fusionImg;
        cv::addWeighted(img, 1, im_color, 0.8, 1, fusionImg);
        // cv::imshow("Fusion Img", fusionImg);
        // cv::waitKey(0);
        cv::imwrite(std::to_string(f) + "_fusion_img.png", fusionImg);
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFreeHost(buffers[inputIndex]));
    CHECK(cudaFreeHost(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}





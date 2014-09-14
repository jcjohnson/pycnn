#include <boost/python.hpp>
#include <cudnn.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

using namespace std;
using namespace boost::python;

// Get a human-readable string from a cudnnStatus_t
inline string cudnnGetErrorString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS: return "SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM: return "BAD_PARAM";
    case CUDNN_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR: return "MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
    case CUDNN_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case CUDNN_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR: return "LICENSE_ERROR";
  }
  return "Unknown status";
}


// A macro to wrap calls to cudnn functions, aborting with an error message
// if it doesn't return cleanly.
#define cudnnSafeCall(status) { cudnnAssert((status), __FILE__, __LINE__); }
inline void cudnnAssert(cudnnStatus_t status, string file, int line,
                        bool abort=true) {
  if (status != CUDNN_STATUS_SUCCESS) {
    cerr << "cudnnAssert: " << cudnnGetErrorString(status) << " " << file
         << " " << line << endl;
    if (abort) exit(status);
  }
}

// A macro to wrap calls to cuda functions, aborting with an error message
// if it doesn't return cleanly.
#define cudaSafeCall(code) { cudaAssert((code), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, string file, int line,
                       bool abort=true) {
  if (code != cudaSuccess) {
    cerr << "cudaAssert: " << cudaGetErrorString(code) << " " << file << " "
         << line << endl;
    if (abort) exit(code);
  }
}

// A 4D blob of data. We maintain CPU and GPU versions of the data, and provide
// a numpy array wrapping the CPU data to be accessed from Python.
// The methods toGpu() and fromGpu() sync the GPU memory with the CPU memory.
// Right now these need to be called manually.
class Tensor4D {
 public:
  Tensor4D(int num, int channels, int height, int width) :
      _num(num), _height(height), _width(width), _channels(channels) {
    _size = num * height * width * channels;
    // Set up a tensor descriptor. The NHWC layout seems not to be supported at
    // this time. For the moment we will assume that Tensor4D objects always
    // store floats.
    cudnnSafeCall(cudnnCreateTensor4dDescriptor(&_tensorDesc));
    cudnnSafeCall(cudnnSetTensor4dDescriptor(_tensorDesc, CUDNN_TENSOR_NCHW,
                    CUDNN_DATA_FLOAT, _num, _channels, _height, _width));

    // Also set up a filter descriptor so that any Tensor4D object can be used
    // as a set of convolutional filters. This adds a bit of overhead to all
    // Tensor4D objects, most of which will never be used as a filter bank.
    // I don't think that this will be a problem, but if it is then we can
    // lazily set up the filter descriptor instead.
    cudnnSafeCall(cudnnCreateFilterDescriptor(&_filterDesc));
    cudnnSafeCall(cudnnSetFilterDescriptor(_filterDesc, CUDNN_DATA_FLOAT,
                    _num, _channels, _height, _width));


    // Create GPU memory.
    // TODO: Make it possible to choose which CUDA device to use.
    cudaSafeCall(cudaMalloc(&_gpu_data, _size * sizeof(float)));

    // Set up the Numpy array. The memory that is allocated by the numpy array
    // will be used as the CPU memory for this Tensor4D, so grab a pointer to
    // it. Allowing the numpy array to manage the CPU memory means
    // that references to the numpy array are still valid even after the Tensor
    // object is destroyed. This makes it much easier to avoid segfaults in
    // Python.
    npy_intp np_shape[4] = {_num, _channels, _height, _width};
    PyObject *obj = PyArray_SimpleNew(4, np_shape, NPY_FLOAT);
    _cpu_data = (float *)PyArray_DATA((PyArrayObject *)obj);
    _np_array = object(handle<>(obj));
  }

  ~Tensor4D() {
    cudaSafeCall(cudaFree(_gpu_data));
    cudnnSafeCall(cudnnDestroyTensor4dDescriptor(_tensorDesc));
    cudnnSafeCall(cudnnDestroyFilterDescriptor(_filterDesc));
  }

  // Copy data from main memory to the GPU
  // TODO: Add a mechanism for async copying?
  void toGpu() {
    cudaSafeCall(cudaMemcpy(_gpu_data, _cpu_data, _size * sizeof(float),
                            cudaMemcpyHostToDevice));
  }

  // Copy data from the GPU to main memory
  // TODO: Add a mechanism for async copying?
  void fromGpu() {
    cudaSafeCall(cudaMemcpy(_cpu_data, _gpu_data, _size * sizeof(float),
                            cudaMemcpyDeviceToHost));
  }
  
  // Accessors for the dimensions and size.
  int num() const { return _num; }
  int height() const { return _height; }
  int width() const { return _width; }
  int channels() const { return _channels; }
  int size() const { return _size; }

  // Accessors for the data pointers.
  const float * const_cpu_data() const { return _cpu_data; }
  const float * const_gpu_data() const { return _gpu_data; }
  float * cpu_data() { return _cpu_data; }
  float * gpu_data() { return _gpu_data; }

  // Accessors for the cuddn descriptors.
  const cudnnTensor4dDescriptor_t tensorDesc() const { return _tensorDesc; }
  const cudnnFilterDescriptor_t filterDesc() const { return _filterDesc; }

  // Get a view of the CPU data as a numpy array.
  object numpy_array() { return _np_array; }

 private:
  int _num;
  int _height;
  int _width;
  int _channels;
  int _size;
  cudnnTensor4dDescriptor_t _tensorDesc;
  cudnnFilterDescriptor_t _filterDesc;
  float *_gpu_data;
  float *_cpu_data;
  object _np_array;
};

// Base class for classes that wrap cudnn layer functions.
// Handles creating and destroying the cudnn context handle.
// TODO: Make it possible to configure the GPU / stream
class LayerFn {
 public:
  LayerFn() { cudnnSafeCall(cudnnCreate(&_cudnn_handle)); }
  ~LayerFn() { cudnnSafeCall(cudnnDestroy(_cudnn_handle)); } 
 protected:
  cudnnHandle_t _cudnn_handle;
};


class Softmax : public LayerFn {
 public:
  Softmax() {
    // TODO: Allow these to be set to CUDNN_SOFTMAX_MODE_CHANNEL
    // or CUDNN_SOFTMAX_FAST respectively.
    _mode = CUDNN_SOFTMAX_MODE_INSTANCE;
    _algorithm = CUDNN_SOFTMAX_ACCURATE;
  }

  void forward(const Tensor4D &bottom_vals, Tensor4D *top_vals) {
    cudnnSafeCall(cudnnSoftmaxForward(_cudnn_handle, _algorithm, _mode,
                    bottom_vals.tensorDesc(), bottom_vals.const_gpu_data(),
                    top_vals->tensorDesc(), top_vals->gpu_data()));
  }

  void backward(const Tensor4D &top_vals, const Tensor4D &top_diffs,
                Tensor4D *bottom_diffs) {
    cudnnSafeCall(cudnnSoftmaxBackward(_cudnn_handle, _algorithm, _mode,
                    top_vals.tensorDesc(), top_vals.const_gpu_data(),
                    top_diffs.tensorDesc(), top_diffs.const_gpu_data(),
                    bottom_diffs->tensorDesc(), bottom_diffs->gpu_data()));
  }

 private:
  cudnnSoftmaxAlgorithm_t _algorithm;
  cudnnSoftmaxMode_t _mode;
};


class Convolution : public LayerFn {
 public:
  Convolution(int pad_x, int pad_y, int stride_x, int stride_y) :
      _pad_x(pad_x), _pad_y(pad_y), _stride_x(stride_x), _stride_y(stride_y) {
    cudnnSafeCall(cudnnCreateConvolutionDescriptor(&_convDesc));

    // TODO: Make it possible to do cross-correlation as well
    _mode = CUDNN_CONVOLUTION;

    // TODO: Figure out what these are and make it possible to set them
    _upscale_x = 1;
    _upscale_y = 1;
  }

  void forward(const Tensor4D &bottom_vals, const Tensor4D &filter_vals,
               Tensor4D *top_vals, bool accumulate) {
    // Set up the convolution descriptor
    cudnnSafeCall(cudnnSetConvolutionDescriptor(_convDesc,
          bottom_vals.tensorDesc(), filter_vals.filterDesc(),
          _pad_y, _pad_x, _stride_y, _stride_x, _upscale_x, _upscale_y,
          _mode));

    // Actually perform the convolution
    cudnnSafeCall(cudnnConvolutionForward(_cudnn_handle,
          bottom_vals.tensorDesc(), bottom_vals.const_gpu_data(),
          filter_vals.filterDesc(), filter_vals.const_gpu_data(),
          _convDesc,
          top_vals->tensorDesc(), top_vals->gpu_data(),
          accumulate ? CUDNN_RESULT_ACCUMULATE : CUDNN_RESULT_NO_ACCUMULATE));
  }

  void backward(const Tensor4D &top_vals, const Tensor4D &top_diffs,
                const Tensor4D &filter_vals, Tensor4D *filter_diffs,
                Tensor4D *bottom_diffs, bool accumulate) {
    // Set up convolution descriptor
    cudnnSafeCall(cudnnSetConvolutionDescriptor(_convDesc,
          bottom_diffs->tensorDesc(), filter_vals.filterDesc(),
          _pad_y, _pad_x, _stride_y, _stride_x, _upscale_x, _upscale_y,
          _mode));

    // Compute gradient with respect to bottom
    cudnnSafeCall(cudnnConvolutionBackwardData(_cudnn_handle,
          filter_vals.filterDesc(), filter_vals.const_gpu_data(),
          top_diffs.tensorDesc(), top_diffs.const_gpu_data(),
          _convDesc,
          bottom_diffs->tensorDesc(), bottom_diffs->gpu_data(),
          accumulate ? CUDNN_RESULT_ACCUMULATE : CUDNN_RESULT_NO_ACCUMULATE));

    // Compute gradient with respect to filter
    cudnnSafeCall(cudnnConvolutionBackwardFilter(_cudnn_handle,
          top_vals.tensorDesc(), top_vals.const_gpu_data(),
          top_diffs.tensorDesc(), top_diffs.const_gpu_data(),
          _convDesc,
          filter_diffs->filterDesc(), filter_diffs->gpu_data(),
          accumulate ? CUDNN_RESULT_ACCUMULATE : CUDNN_RESULT_NO_ACCUMULATE));
  }

 private:
  int _pad_x, _pad_y, _stride_x, _stride_y, _upscale_x, _upscale_y;
  cudnnConvolutionDescriptor_t _convDesc;
  cudnnConvolutionMode_t _mode;
};


class Activation : public LayerFn {
 public:
  explicit Activation(cudnnActivationMode_t mode) : _mode(mode) { }

  void forward(const Tensor4D &bottom_vals, Tensor4D *top_vals) {
    cudnnSafeCall(cudnnActivationForward(_cudnn_handle, _mode,
          bottom_vals.tensorDesc(), bottom_vals.const_gpu_data(),
          top_vals->tensorDesc(), top_vals->gpu_data()));
  }

  void backward(const Tensor4D &top_vals, const Tensor4D &top_diffs,
                const Tensor4D &bottom_vals, Tensor4D *bottom_diffs) {
    cudnnSafeCall(cudnnActivationBackward(_cudnn_handle, _mode,
          top_vals.tensorDesc(), top_vals.const_gpu_data(),
          top_diffs.tensorDesc(), top_diffs.const_gpu_data(),
          bottom_vals.tensorDesc(), bottom_vals.const_gpu_data(),
          bottom_diffs->tensorDesc(), bottom_diffs->gpu_data()));
  }
 private:
  cudnnActivationMode_t _mode;
};


class ReLu : public Activation {
 public: ReLu() : Activation(CUDNN_ACTIVATION_RELU) { }
};


class Sigmoid : public Activation {
 public: Sigmoid() : Activation(CUDNN_ACTIVATION_SIGMOID) { }
};


class Tanh : public Activation {
 public: Tanh() : Activation(CUDNN_ACTIVATION_TANH) { }
};


BOOST_PYTHON_MODULE(pycudnn) {
  import_array();
  numeric::array::set_module_and_type("numpy", "ndarray");

  class_<Tensor4D>("Tensor4D", init<int, int, int, int>())
    .add_property("num", &Tensor4D::num)
    .add_property("height", &Tensor4D::height)
    .add_property("width", &Tensor4D::width)
    .add_property("channels", &Tensor4D::channels)
    .add_property("size", &Tensor4D::size)
    .add_property("data", &Tensor4D::numpy_array)
    .def("fromGpu", &Tensor4D::fromGpu)
    .def("toGpu", &Tensor4D::toGpu);

  class_<Softmax>("Softmax")
    .def("forward", &Softmax::forward)
    .def("backward", &Softmax::backward);

  class_<Convolution>("Convolution", init<int, int, int, int>())
    .def("forward", &Convolution::forward)
    .def("backward", &Convolution::backward);

  class_<ReLu>("ReLu")
    .def("forward", &ReLu::forward)
    .def("backward", &ReLu::backward);
  
  class_<Sigmoid>("Sigmoid")
    .def("forward", &Sigmoid::forward)
    .def("backward", &Sigmoid::backward);

  class_<Tanh>("Tanh")
    .def("forward", &Tanh::forward)
    .def("backward", &Tanh::backward);
}


#include <boost/python.hpp>
#include <cudnn.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

using namespace std;
using namespace boost::python;


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


// A nice little macro to wrap functions that return cudnn status codes.
#define cudnnSafeCall(status) { cudnnAssert((status), __FILE__, __LINE__); }
inline void cudnnAssert(cudnnStatus_t status, string file, int line,
                        bool abort=true) {
  if (status != CUDNN_STATUS_SUCCESS) {
    cerr << "cudnnAssert: " << cudnnGetErrorString(status) << " " << file
         << " " << line << endl;
    if (abort) exit(status);
  }
}


#define cudaSafeCall(code) { cudaAssert((code), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, string file, int line,
                       bool abort=true) {
  if (code != cudaSuccess) {
    cerr << "cudaAssert: " << cudaGetErrorString(code) << " " << file << " "
         << line << endl;
    if (abort) exit(code);
  }
}


class Tensor4D {
 public:
  Tensor4D(int num, int height, int width, int channels) :
      _num(num), _height(height), _width(width), _channels(channels) {
    _size = num * height * width * channels;
    // Set up a tensor descriptor. The NHWC layout seems not to be supported at
    // this time. For the moment we will assume that Tensor4D objects always
    // store floats.
    cudnnSafeCall(cudnnCreateTensor4dDescriptor(&_tensorDesc));
    cudnnSafeCall(cudnnSetTensor4dDescriptor(_tensorDesc, CUDNN_TENSOR_NCHW,
                  CUDNN_DATA_FLOAT, _num, _channels, _height, _width));

    // Create GPU memory.
    // TODO: Make it possible to choose which CUDA device to use.
    cudaSafeCall(cudaMalloc(&_gpu_data, _size * sizeof(float)));

    // Set up the Numpy array. The memory that is allocated by the numpy array
    // will be used as the CPU memory for this Tensor4D, so we store a pointer
    // to this memory. Allowing the numpy array to manage the CPU memory means
    // that references to the numpy array are still valid even after the Tensor
    // object is destroyed.
    npy_intp np_shape[4] = {_num, _channels, _height, _width};
    PyObject *obj = PyArray_SimpleNew(4, np_shape, NPY_FLOAT);
    _cpu_data = (float *)PyArray_DATA((PyArrayObject *)obj);
    _np_array = object(handle<>(obj));
  }

  ~Tensor4D() {
    cudaSafeCall(cudaFree(_gpu_data));
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
  
  int num() const { return _num; }
  int height() const { return _height; }
  int width() const { return _width; }
  int channels() const { return _channels; }
  int size() const { return _size; }

  const float * const_cpu_data() const { return _cpu_data; }
  const float * const_gpu_data() const { return _gpu_data; }

  float * cpu_data() { return _cpu_data; }
  float * gpu_data() { return _gpu_data; }
  const cudnnTensor4dDescriptor_t tensorDesc() const { return _tensorDesc; }
  
  object numpy_array() { return _np_array; }

 private:
  int _num;
  int _height;
  int _width;
  int _channels;
  int _size;
  cudnnTensor4dDescriptor_t _tensorDesc;
  float *_gpu_data;
  float *_cpu_data;
  object _np_array;
};


class Softmax {
 public:
  Softmax() {
    // TODO: Allow these to be set to CUDNN_SOFTMAX_MODE_CHANNEL
    // or CUDNN_SOFTMAX_FAST respectively.
    _mode = CUDNN_SOFTMAX_MODE_INSTANCE;
    _algorithm = CUDNN_SOFTMAX_ACCURATE;

    // TODO: Do something better for this
    cudnnSafeCall(cudnnCreate(&_cudnn_handle));
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
  cudnnHandle_t _cudnn_handle;
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
}


PYTHON_INC = -I/usr/include/python2.7
PYTHON_LDFLAGS = -L/usr/lib/python2.7/config
PYTHON_LIBS = -lpython2.7

NUMPY_DIR = -I/usr/local/lib/python2.7/dist-packages/numpy
NUMPY_INC = $(NUMPY_DIR)/core/include

BOOST_INC = -I/usr/include
BOOST_LIBS = -lboost_python-py27
BOOST_LDFLAGS	= -L/usr/lib/x86_64-linux-gnu

CUDA_DIR = /usr/local/cuda
CUDA_INC = -I$(CUDA_DIR)/include
CUDA_LDFLAGS = -L$(CUDA_DIR)/lib64
CUDA_LIBS = -lcudnn -lcuda -lcudart

INCS = $(PYTHON_INC) $(BOOST_INC) $(NUMPY_INC) $(CUDA_INC)
LIBS = $(BOOST_LIBS) $(PYTHON_LIBS) $(CUDA_LIBS)
LDFLAGS = $(BOOST_LDFLAGS) $(PYTHON_LDFLAGS) $(CUDA_LDFLAGS)

pycudnn.so: build/pycudnn.o
	g++ -shared -Wl,--export-dynamic $^ $(LDFLAGS) $(LIBS) -o $@
 
build/pycudnn.o: src/pycudnn.cpp
	g++ $(INCS) -fPIC -c $^ -o $@

clean:
	rm -f *.so build/*

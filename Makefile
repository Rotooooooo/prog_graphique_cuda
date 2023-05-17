CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS1=`pkg-config --libs opencv`
LDLIBS2=-lm -lIL

all: boxb boxb-cu edgec edgec-cu laplg laplg-cu gblur gblur-cu boxb_stream-cu edgec_stream-cu laplg_stream-cu gblur_stream-cu sobel sobel-cu

gblur_stream-cu: gaussian_blur_stream.cu
	nvcc -o $@ $< $(LDLIBS1)

laplg_stream-cu: laplacian_gauss_stream.cu
	nvcc -o $@ $< $(LDLIBS1)

boxb: box_blur1.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS2)

boxb_stream-cu: box_blur1_stream.cu
	     	   nvcc -o $@ $< $(LDLIBS1)

boxb-cu: box_blur1.cu
	nvcc -o $@ $<  $(LDLIBS1)

edgec: edge_detection.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS2)

edgec-cu: edge_detection.cu
	nvcc -o $@ $<  $(LDLIBS1)

edgec_stream-cu: edge_detection_stream.cu
	nvcc -o $@ $<  $(LDLIBS1)

laplg: laplacian_gauss.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS2)

laplg-cu: laplacian_gauss.cu
	nvcc -o $@ $<  $(LDLIBS1)

gblur: gaussian_blur.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS2)

gblur-cu: gaussian_blur.cu
	nvcc -o $@ $<  $(LDLIBS1)

sobel : sobel.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS2)

sobel-cu : sobel.cu
	nvcc -o $@ $<  $(LDLIBS1)


.PHONY: clean

clean:
	rm boxb boxb-cu edgec edgec-cu laplg laplg-cu gblur gblur-cu boxb_stream-cu edgec_stream-cu laplg_stream-cu gblur_stream-cu

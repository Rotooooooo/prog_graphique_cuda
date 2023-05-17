#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>

 __global__ void grayscale_laplacian_shared( unsigned char * rgb, unsigned char * tab_res, std::size_t cols, std::size_t rows ) {
  auto i = blockIdx.x * (blockDim.x-5) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y-5) + threadIdx.y;

  auto li = threadIdx.x;
  auto lj = threadIdx.y;

  auto w = blockDim.x;
  auto h = blockDim.y;

  extern __shared__ unsigned char shared_tab[];

  if( i < cols && j < rows ) {
    shared_tab[ lj * w + li ] = (
			 307 * rgb[ 3 * ( j * cols + i ) ]
			 + 604 * rgb[ 3 * ( j * cols + i ) + 1 ]
			 + 113 * rgb[  3 * ( j * cols + i ) + 2 ]
			 ) >> 10;
  }

  __syncthreads();

  if( i < cols -2 && j < rows-2 && li > 2 && li < (w-2) && lj > 2 && lj < (h-2) )
  {
    auto res = 
           -     shared_tab[ (lj-1)*w + li - 1 ] - 2* shared_tab[ (lj-1)*w + li ] -   shared_tab[ (lj-1)*w + li + 1 ]
           - 2 * shared_tab[ (lj  )*w + li - 1 ] + 16*shared_tab[ (lj  )*w + li ] - 2*shared_tab[ (lj  )*w + li + 1 ]
           -     shared_tab[ (lj+1)*w + li - 1 ] - 2* shared_tab[ (lj+1)*w + li ] -   shared_tab[ (lj+1)*w + li + 1 ]
           -     shared_tab[ (lj-2)*w + li     ] -    shared_tab[ (lj  )*w + li - 2 ] - shared_tab[ (lj+2)*w + li] - shared_tab[ (lj)*w + li + 2 ];

    res = res > 255 ? 255 : res;
    res = res < 0 ? 0 : res;
    tab_res[ j * cols + i ] = res;
  }
}


int main()
{
  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );

  auto rows = m_in.rows;
  auto cols = m_in.cols;
 
  unsigned char * rgb = nullptr;
  cudaMallocHost( &rgb, 3 * rows * cols );

  std::memcpy( rgb, m_in.data, 3 * rows * cols );

  unsigned char * rgb_d;
  unsigned char * tab_d;
  unsigned char * s_d;

  cudaMalloc( &rgb_d, 3 * rows * cols );
  cudaMalloc( &tab_d, rows * cols );
  cudaMalloc( &s_d, rows * cols );

  cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );

  dim3 block( 64, 8 );
  dim3 grid1( ( cols - 1) / (block.x-5) + 1 , ( rows - 1 ) / (block.y-5) + 1 );

  cudaStream_t stream[2];
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);

  cudaEvent_t start, stop;

  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  // Mesure du temps de calcul du kernel
  cudaEventRecord( start );

  grayscale_laplacian_shared<<< grid1, block, block.x * (block.y+2)*sizeof(unsigned char),stream[0] >>>( rgb_d, s_d, cols, rows/2 + 2);
  grayscale_laplacian_shared<<< grid1, block, block.x * (block.y+2)*sizeof(unsigned char),stream[1] >>>(rgb_d+(((rows*cols*3)/2)-cols*3*3),tab_d,cols,rows/2 +3);

  unsigned char* tab = nullptr;
  cudaMallocHost(&tab,rows*cols);
  cudaMemcpyAsync(tab,s_d,(rows*cols)/2,cudaMemcpyDeviceToHost,stream[0]);
  cudaMemcpyAsync(tab+(rows*cols)/2,tab_d+cols*3,(rows*cols)/2,cudaMemcpyDeviceToHost,stream[1]);


  cv :: Mat m_out(rows,cols,CV_8UC1,tab);

  cudaEventRecord( stop );

  cudaEventSynchronize( stop );
  float duration;
  cudaEventElapsedTime( &duration, start, stop );
  std::cout << "time=" << duration << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cv::imwrite( "out.jpg", m_out );

  cudaFree( rgb_d);
  cudaFree( tab_d);
  cudaFree( s_d);

  cudaFreeHost( tab );
  cudaFreeHost( rgb );
  cudaStreamDestroy(stream[0]);
  cudaStreamDestroy(stream[1]);

  return 0;
}


#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>

__global__ void grayscale_edge_shared( unsigned char * rgb, unsigned char * res, std::size_t cols, std::size_t rows ) {
  auto i = blockIdx.x * (blockDim.x-2) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y-2) + threadIdx.y;

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

  if( i < cols -1 && j < rows-1 && li > 0 && li < (w-1) && lj > 0 && lj < (h-1) )
  {
    auto edge_sum =
            - shared_tab[ ((lj-1)*w + li - 1) ] - shared_tab[ ((lj-1)*w + li)] - shared_tab[ ((lj-1)*w + li + 1 )]
            - shared_tab[ ((lj  )*w + li - 1) ] + 8*shared_tab[ ((lj)*w + li)] - shared_tab[ ((lj  )*w + li + 1 )]
            - shared_tab[ ((lj+1)*w + li - 1) ] - shared_tab[ ((lj+1)*w + li)] - shared_tab[ ((lj+1)*w + li + 1 )];

    edge_sum = edge_sum > 255 ? 255 : edge_sum;
    edge_sum = edge_sum < 0 ? 0 : edge_sum;
    res[ j * cols + i ] = edge_sum;
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

  dim3 block( 32, 4 );
  dim3 grid1( ( cols - 1) / (block.x-2) + 1 , ( rows - 1 ) / (block.y-2) + 1 );

  cudaStream_t stream[2];
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);

  cudaEvent_t start, stop;

  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  // Mesure du temps de calcul du kernel
  cudaEventRecord( start );

  grayscale_edge_shared<<< grid1, block, block.x * (block.y+2)*sizeof(unsigned char),stream[0] >>>( rgb_d, s_d, cols, rows/2 + 1);
  grayscale_edge_shared<<< grid1, block, block.x * (block.y+2)*sizeof(unsigned char),stream[1] >>>(rgb_d+(((rows*cols*3)/2)-cols*3),tab_d,cols,rows/2 +1);

  unsigned char* tab = nullptr;
  cudaMallocHost(&tab,rows*cols);
  cudaMemcpyAsync(tab,s_d,(rows*cols)/2,cudaMemcpyDeviceToHost,stream[0]);
  cudaMemcpyAsync(tab+(rows*cols)/2,tab_d+cols,(rows*cols)/2,cudaMemcpyDeviceToHost,stream[1]);


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


#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>

/**
 * Kernel pour transformer l'image RGB en niveaux de gris.
 */
__global__ void grayscale( unsigned char * rgb, unsigned char * tab, std::size_t cols, std::size_t rows ) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
  if( i < cols && j < rows ) {
    tab[ j * cols + i ] = (
			 307 * rgb[ 3 * ( j * cols + i ) ]
			 + 604 * rgb[ 3 * ( j * cols + i ) + 1 ]
			 + 113 * rgb[  3 * ( j * cols + i ) + 2 ]
			 ) >> 10;
  }
}

/**
 * Kernel pour obtenir les contours à partir de l'image en niveaux de gris.
 */
__global__ void edge( unsigned char * tab, unsigned char * res, std::size_t cols, std::size_t rows )
{
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;

  if( i > 1 && i < cols && j > 1 && j < rows )
  {
    auto edge_sum = 
            - tab[ ((j-1)*cols + i - 1) ] - tab[ ((j-1)*cols + i)] - tab[ ((j-1)*cols + i + 1 )]
            - tab[ ((j  )*cols + i - 1) ] + 8*tab[ ((j)*cols + i)] - tab[ ((j  )*cols + i + 1 )]
            - tab[ ((j+1)*cols + i - 1) ] - tab[ ((j+1)*cols + i)] - tab[ ((j+1)*cols + i + 1 )];


    edge_sum = edge_sum > 255 ? 255 : edge_sum;
    edge_sum = edge_sum < 0 ? 0 : edge_sum;
    
    res[ j * cols + i ] = edge_sum;
  }
}


/**
 * Kernel pour obtenir les contours à partir de l'image en niveaux de gris, en utilisant la mémoire shared
 * pour limiter les accès à la mémoire globale.
 */
__global__ void edge_shared( unsigned char * tab, unsigned char * res, std::size_t cols, std::size_t rows )
{
  auto li = threadIdx.x;
  auto lj = threadIdx.y;

  auto w = blockDim.x;
  auto h = blockDim.y;

  auto i = blockIdx.x * (blockDim.x-2) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y-2) + threadIdx.y;

  extern __shared__ unsigned char shared_tab[];

  if( i < cols && j < rows )
  {
    shared_tab[ lj * w + li ] = tab[ j * cols + i ];
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


/**
 * Kernel fusionnant le passage en niveaux de gris et la détection de contours.
 */
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
  cv::Mat m_in = cv::imread("big.jpg", cv::IMREAD_UNCHANGED );


  auto rows = m_in.rows;
  auto cols = m_in.cols;


  unsigned char * tab = nullptr;
  cudaMallocHost( &tab, rows * cols );
  cv::Mat m_out( rows, cols, CV_8UC1, tab );


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

  dim3 grid0( ( cols - 1) / block.x + 1 , ( rows - 1 ) / block.y + 1 );


  dim3 grid1( ( cols - 1) / (block.x-2) + 1 , ( rows - 1 ) / (block.y-2) + 1 );

  cudaEvent_t start, stop;

  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  // Mesure du temps de calcul du kernel uniquement.
  cudaEventRecord( start );

  
  // Version en 2 étapes.
  grayscale<<< grid0, block >>>( rgb_d, tab_d, cols, rows );
  edge<<< grid0, block >>>( tab_d, s_d, cols, rows );
  

  /*
  // Version en 2 étapes, Sobel avec mémoire shared.
  grayscale<<< grid0, block >>>( rgb_d, tab_d, cols, rows );
  edge_shared<<< grid1, block, block.x * block.y >>>( tab_d, s_d, cols, rows );
  */

  // Version fusionnée.
  //grayscale_edge_shared<<< grid1, block, block.x * block.y >>>( rgb_d, s_d, cols, rows );

  cudaEventRecord( stop );

  cudaMemcpy( tab, s_d, rows * cols, cudaMemcpyDeviceToHost );

  cudaEventSynchronize( stop );
  float duration;
  cudaEventElapsedTime( &duration, start, stop );
  std::cout << "time=" << duration << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cv::imwrite( "big_edge_detection.jpg", m_out );

  cudaFree( rgb_d);
  cudaFree( tab_d);
  cudaFree( s_d);

  cudaFreeHost( tab );
  cudaFreeHost( rgb );

  return 0;
}

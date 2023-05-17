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
 * Kernel pour flouter l'image à partir de l'image en niveaux de gris.
 */
__global__ void gauss_blur( unsigned char * tab, unsigned char * tab_res, std::size_t cols, std::size_t rows )
{
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;

  if( i > 3 && i < cols && j > 3 && j < rows )
  {
    auto res =  0.5*tab[ (j-2)*cols + i-2 ] + 1.8*tab[ (j-2)*cols + i-1 ] + 3.2*tab[ (j-2)*cols + i ] + 1.8*tab[ (j-2)*cols + i+1 ] + 0.5*tab[ (j-2)*cols + i+2 ]

    + 1.8*tab[ (j-1)*cols + i-2 ] + 6.4* tab[ (j-1)*cols + i-1 ] + 10* tab[ (j-1)*cols + i ] + 6.4*tab[ (j-1)*cols + i+1 ] + 1.8* tab[ (j-1)*cols + i+2 ]

    + 3.2*tab[ (j)*cols + i-2 ] + 10*tab[ (j)*cols + i-1 ] + 10*tab[ (j)*cols + i ] + 10*tab[ (j)*cols + i+1 ] + 3.2*tab[ (j)*cols + i+2 ]

    + 1.8*tab[ (j+1)*cols + i-2 ] + 6.4* tab[ (j+1)*cols + i-1 ] + 10* tab[ (j+1)*cols + i ] + 6.4*tab[ (j+1)*cols + i+1 ] + 1.8* tab[ (j+1)*cols + i+2 ]

    + 0.5*tab[ (j+2)*cols + i-2 ] + 1.8*tab[ (j+2)*cols + i-1 ] + 3.2*tab[ (j+2)*cols + i ] + 1.8*tab[ (j+2)*cols + i+1 ] + 0.5*tab[ (j+2)*cols + i+2 ]

    + 0.5*tab[ (j+3)*cols + i ] + 0.5*tab[ (j-3)*cols + i ] + 0.5*tab[ (j)*cols + i-3 ] + 0.5*tab[ (j)*cols + i+3 ];

    res = res/(3.5*29);
    res = res > 255 ? 255 : res;
    res = res < 0 ? 0 : res;
    
    tab_res[ j * cols + i ] = res;
  }
}

/**
 * Kernel pour  flouter l'image à partir de l'image en niveaux de gris, en utilisant la mémoire shared
 * pour limiter les accès à la mémoire globale.
 */
 __global__ void gauss_shared( unsigned char * tab, unsigned char * tab_res, std::size_t cols, std::size_t rows )
 {
   auto li = threadIdx.x;
   auto lj = threadIdx.y;
 
   auto w = blockDim.x;
   auto h = blockDim.y;
 
   auto i = blockIdx.x * (blockDim.x-7) + threadIdx.x;
   auto j = blockIdx.y * (blockDim.y-7) + threadIdx.y;
 
   extern __shared__ unsigned char shared_tab[];
 
   if( i < cols && j < rows )
   {
     shared_tab[ lj * w + li ] = tab[ j * cols + i ];
   }
 
   __syncthreads();
 
   if( i < cols -3 && j < rows-3 && li > 3 && li < (w-3) && lj > 3 && lj < (h-3) )
   {
    auto res =  0.5*shared_tab[ (lj-2)*w + li-2 ] + 1.8*shared_tab[ (lj-2)*w + li-1 ] + 3.2*shared_tab[ (lj-2)*w + li ] + 1.8*shared_tab[ (lj-2)*w + li+1 ] + 0.5*shared_tab[ (lj-2)*w + li+2 ]

    + 1.8*shared_tab[ (lj-1)*w + li-2 ] + 6.4* shared_tab[ (lj-1)*w + li-1 ] + 10* shared_tab[ (lj-1)*w + li ] + 6.4*shared_tab[ (lj-1)*w + li+1 ] + 1.8* shared_tab[ (lj-1)*w + li+2 ]

    + 3.2*shared_tab[ (lj)*w + li-2 ] + 10*shared_tab[ (lj)*w + li-1 ] + 10*shared_tab[ (lj)*w + li ] + 10*shared_tab[ (lj)*w + li+1 ] + 3.2*shared_tab[ (lj)*w + li+2 ]

    + 1.8*shared_tab[ (lj+1)*w + li-2 ] + 6.4* shared_tab[ (lj+1)*w + li-1 ] + 10* shared_tab[ (lj+1)*w + li ] + 6.4*shared_tab[ (lj+1)*w + li+1 ] + 1.8* shared_tab[ (lj+1)*w + li+2 ]

    + 0.5*shared_tab[ (lj+2)*w + li-2 ] + 1.8*shared_tab[ (lj+2)*w + li-1 ] + 3.2*shared_tab[ (lj+2)*w + li ] + 1.8*shared_tab[ (lj+2)*w + li+1 ] + 0.5*shared_tab[ (lj+2)*w + li+2 ]

    + 0.5*shared_tab[ (lj+3)*w + li ] + 0.5*shared_tab[ (lj-3)*w + li ] + 0.5*shared_tab[ (lj)*w + li-3 ] + 0.5*shared_tab[ (lj)*w + li+3 ];

    res = res/(3.5*29);
    res = res > 255 ? 255 : res;
    res = res < 0 ? 0 : res;
    
    tab_res[ j * cols + i ] = res;
   }
 }

/**
 * Kernel fusionnant le passage en niveaux de gris et la détection de contours.
 */
 __global__ void grayscale_gauss_shared( unsigned char * rgb, unsigned char * tab_res, std::size_t cols, std::size_t rows ) {
  auto i = blockIdx.x * (blockDim.x-7) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y-7) + threadIdx.y;

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

  if( i < cols -3 && j < rows-3 && li > 3 && li < (w-3) && lj > 3 && lj < (h-3) )
  {
auto res =  0.5*shared_tab[ (lj-2)*w + li-2 ] + 1.8*shared_tab[ (lj-2)*w + li-1 ] + 3.2*shared_tab[ (lj-2)*w + li ] + 1.8*shared_tab[ (lj-2)*w + li+1 ] + 0.5*shared_tab[ (lj-2)*w + li+2 ]

    + 1.8*shared_tab[ (lj-1)*w + li-2 ] + 6.4* shared_tab[ (lj-1)*w + li-1 ] + 10* shared_tab[ (lj-1)*w + li ] + 6.4*shared_tab[ (lj-1)*w + li+1 ] + 1.8* shared_tab[ (lj-1)*w + li+2 ]

    + 3.2*shared_tab[ (lj)*w + li-2 ] + 10*shared_tab[ (lj)*w + li-1 ] + 10*shared_tab[ (lj)*w + li ] + 10*shared_tab[ (lj)*w + li+1 ] + 3.2*shared_tab[ (lj)*w + li+2 ]

    + 1.8*shared_tab[ (lj+1)*w + li-2 ] + 6.4* shared_tab[ (lj+1)*w + li-1 ] + 10* shared_tab[ (lj+1)*w + li ] + 6.4*shared_tab[ (lj+1)*w + li+1 ] + 1.8* shared_tab[ (lj+1)*w + li+2 ]

    + 0.5*shared_tab[ (lj+2)*w + li-2 ] + 1.8*shared_tab[ (lj+2)*w + li-1 ] + 3.2*shared_tab[ (lj+2)*w + li ] + 1.8*shared_tab[ (lj+2)*w + li+1 ] + 0.5*shared_tab[ (lj+2)*w + li+2 ]

    + 0.5*shared_tab[ (lj+3)*w + li ] + 0.5*shared_tab[ (lj-3)*w + li ] + 0.5*shared_tab[ (lj)*w + li-3 ] + 0.5*shared_tab[ (lj)*w + li+3 ];

    res = (res/(3.5*29));
    res = res > 255 ? 255 : res;
    res = res < 0 ? 0 : res;
    
    tab_res[ j * cols + i ] = res;
  }
}


int main()
{
  cv::Mat m_in = cv::imread("casquette.jpg", cv::IMREAD_UNCHANGED );


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

  dim3 block( 64, 8 );
  dim3 grid0( ( cols - 1) / block.x + 1 , ( rows - 1 ) / block.y + 1 );
  dim3 grid1( ( cols - 3) / (block.x-7) + 1 , ( rows - 3 ) / (block.y-7) + 1 );
    
  cudaEvent_t start, stop;

  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  // Mesure du temps de calcul du kernel uniquement.
  cudaEventRecord( start );

  /*  
  // Version en 2 étapes.
  grayscale<<< grid0, block >>>( rgb_d, tab_d, cols, rows );
  gauss_blur<<< grid0, block >>>( tab_d, s_d, cols, rows );
  */

  /*
  // Version en 2 étapes, gauss_blur avec mémoire shared.
  grayscale<<< grid0, block >>>( rgb_d, tab_d, cols, rows );
  gauss_shared<<< grid1, block, block.x * block.y >>>( tab_d, s_d, cols, rows );
  */

  // Version fusionnée.
  grayscale_gauss_shared<<< grid1, block, block.x * block.y >>>( rgb_d, s_d, cols, rows );

  cudaEventRecord( stop );
  
  cudaMemcpy( tab, s_d, rows * cols, cudaMemcpyDeviceToHost );

  cudaEventSynchronize( stop );
  float duration;
  cudaEventElapsedTime( &duration, start, stop );
  std::cout << "time=" << duration << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cv::imwrite( "casquette_gaussian_blur.jpg", m_out );

  cudaFree( rgb_d);
  cudaFree( tab_d);
  cudaFree( s_d);

  cudaFreeHost( tab );
  cudaFreeHost( rgb );
  
  return 0;
}

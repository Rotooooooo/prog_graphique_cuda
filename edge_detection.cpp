#include <iostream>

#include <cmath>

#include <IL/il.h>
#include <chrono>
using namespace std::chrono;

int main() {

  unsigned int image;

  ilInit();

  ilGenImages(1, &image);
  ilBindImage(image);
  ilLoadImage("in.jpg");

  int width, height, bpp, format;

  width = ilGetInteger(IL_IMAGE_WIDTH);
  height = ilGetInteger(IL_IMAGE_HEIGHT); 
  bpp = ilGetInteger(IL_IMAGE_BYTES_PER_PIXEL);
  format = ilGetInteger(IL_IMAGE_FORMAT);

  // Récupération des données de l'image
  unsigned char* data = ilGetData();

  // Traitement de l'image
  unsigned char* out_grey = new unsigned char[ width*height ];
  unsigned char* out_edge_detect = new unsigned char[width*height ];

  auto start = high_resolution_clock::now();

  for( std::size_t i = 0 ; i < width*height ; ++i )
  {

    out_grey[ i ] = ( 307 * data[ 3*i ]
		       + 604 * data[ 3*i+1 ]
		       + 113 * data[ 3*i+2 ]
		       ) >> 10;
  }

  unsigned int i, j;

  int m;


  for(j = 1 ; j < height - 1 ; ++j) {

    for(i = 1 ; i < width - 1 ; ++i) {


	// Calcule de la matrice
	m =  -  out_grey[((j - 1) * width + i - 1)] - out_grey[((j-1) * width + i)] - out_grey[((j - 1) * width + i + 1)]
	  - 	out_grey[(( j  	) * width + i - 1)] + (8* out_grey[((j) * width + i)]) - out_grey[((j    ) * width + i + 1)]
	  -     out_grey[((j + 1) * width + i - 1)] - out_grey[((j+1) * width + i)] - out_grey[((j + 1) * width + i + 1)];

	m = m > 255 ? 255 : m;
	m = m < 0 ? 0 : m;

	out_edge_detect[(height - j - 1) * width + i] = m;

    }

  }

  auto stop =high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop-start);

 std::cout << "Temps d'execution du séquenciel edge detection : " << duration.count() << "ms" << std::endl;

  //Placement des données dans l'image
  ilTexImage( width, height, 1, 1, IL_LUMINANCE, IL_UNSIGNED_BYTE, out_edge_detect );


  // Sauvegarde de l'image


  ilEnable(IL_FILE_OVERWRITE);

  ilSaveImage("out.jpg");

  ilDeleteImages(1, &image); 

  delete [] out_grey;
  delete [] out_edge_detect;

}

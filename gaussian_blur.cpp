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
  ilLoadImage("casquette.jpg");

  int width, height, bpp, format;

  width = ilGetInteger(IL_IMAGE_WIDTH);
  height = ilGetInteger(IL_IMAGE_HEIGHT); 
  bpp = ilGetInteger(IL_IMAGE_BYTES_PER_PIXEL);
  format = ilGetInteger(IL_IMAGE_FORMAT);

  // Récupération des données de l'image
  unsigned char* data = ilGetData();

  // Traitement de l'image
  unsigned char* out_grey = new unsigned char[ width*height ];
  unsigned char* out_gauss_blur = new unsigned char[width*height ];

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


  for(j = 3 ; j < height - 3 ; ++j) {

    for(i = 3 ; i < width - 3 ; ++i) {


	// Calcule de la matrice
	m =
	  + 0.5*out_grey[((j-2) * width + i - 2)] + 1.8*out_grey[((j-2) * width + i - 1)] + 3.2*out_grey[((j-2) * width + i)] + 1.8*out_grey[((j-2) * width + i + 1)] + 0.5*out_grey[((j-2) * width + i + 2)]
	  + 1.8*out_grey[((j-1) * width + i - 2)] + 6.4*out_grey[((j - 1) * width + i - 1)] + 10.0*out_grey[((j-1) * width + i)] + 6.4*out_grey[((j - 1) * width + i + 1)] + 1.8*out_grey[((j-1) * width + i + 2)]
	  + 3.2*out_grey[(j * width + i - 2)]     + 10.0*out_grey[((j) * width + i - 1)]    + (10.0* out_grey[((j) * width + i)]) + 10.0*out_grey[((j) * width + i + 1)]   + 3.2*out_grey[(j * width + i + 2)]
	  + 1.8*out_grey[((j+1) * width + i - 2)] + 6.4*out_grey[((j + 1) * width + i - 1)] + 10.0*out_grey[((j+1) * width + i)] + 6.4*out_grey[((j + 1) * width + i + 1)] + 1.8*out_grey[((j+1) * width + i + 2)]
	  + 0.5*out_grey[((j+2) * width + i - 2)] + 1.8*out_grey[((j + 2) * width + i - 1)] + 3.2*out_grey[((j+2) * width + i)] + 1.8*out_grey[((j+2) * width + i + 1)] + 0.5*out_grey[((j+2) * width + i + 2)]
	  + 0.5*out_grey[((j-3) * width + i)] + 0.5*out_grey[((j+3) * width + i)] + 0.5*out_grey[(j * width + i - 3)] + 0.5*out_grey[(j * width + i + 3)];

	m = m/(3.5*29);
	m = m > 255 ? 255 : m;
	m = m < 0 ? 0 : m;

	out_gauss_blur[(height - j - 1) * width + i] = m;

    }

  }

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast <milliseconds>(stop-start);

  std::cout << "Temps en sequentiel de gaussian blur : " << duration.count() << "ms" << std::endl;
  //Placement des données dans l'image
  ilTexImage( width, height, 1, 1, IL_LUMINANCE, IL_UNSIGNED_BYTE, out_gauss_blur );


  // Sauvegarde de l'image


  ilEnable(IL_FILE_OVERWRITE);

  ilSaveImage("out.jpg");

  ilDeleteImages(1, &image); 

  delete [] out_grey;
  delete [] out_gauss_blur;

}

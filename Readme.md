Bonjour, voici notre projet cuda :)
Mattheo PAILLER RACHALSKI
Ahmet-Kadir ONSEKIZ
Brice JACQUESSON

commande : 
compilation -> nvcc -o image image.cu $(pkg-config --libs --cflags opencv) -std=c++11
afficher l'image -> eog out.png

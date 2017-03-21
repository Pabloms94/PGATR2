#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#define BLOCKSIZE 32

__shared__ float sharedMatM[BLOCKSIZE * BLOCKSIZE];
__shared__ float sharedMatm[BLOCKSIZE * BLOCKSIZE];

void calculate_cdf(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  /* TODO
    1) Encontrar el valor máximo y mínimo de luminancia en min_logLum and max_logLum a partir del canal logLuminance 
	2) Obtener el rango a representar
	3) Generar un histograma de todos los valores del canal logLuminance usando la formula 
	bin = (Lum [i] - lumMin) / lumRange * numBins
	4) Realizar un exclusive scan en el histograma para obtener la distribución acumulada (cdf) 
	de los valores de luminancia. Se debe almacenar en el puntero c_cdf
  */    

	//Conseguimos la posición del píxel en la imagen del que se ocupará el hilo
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
	//Calculamos la posición del hilo en el bloque
	const int posThreadBlock = threadIdx.x * BLOCKSIZE + threadIdx.y;

	//Si estamos fuera de los límites de la imagen, paramos
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	//Almacenamos en memoria compartida el valor correspondiente a cada thread
	sharedMatM[posThreadBlock] = d_logLuminance[thread_1D_pos];
	sharedMatm[posThreadBlock] = d_logLuminance[thread_1D_pos];

	__syncthreads();

	//Ahora iteraremos sobre los elementos de memoria compartida para ir comparando y obtener el elemento menor y el mayor correspondientemente.
	for (int i = BLOCKSIZE / 2; i > 0; i /= 2){
		if (posThreadBlock < i){
			if (sharedMatm[posThreadBlock] > sharedMatm[posThreadBlock + i])
				sharedMatm[posThreadBlock] = sharedMatm[posThreadBlock + i];

			if (sharedMatM[posThreadBlock] < sharedMatM[posThreadBlock + i])
				sharedMatM[posThreadBlock] = sharedMatM[posThreadBlock + i];
		}
		__syncthreads();
	}

	if (posThreadBlock == 0){
		min_logLum = sharedMatm[0];
		max_logLum = sharedMatM[0];
	}

}

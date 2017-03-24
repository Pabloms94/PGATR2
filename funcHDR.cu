#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <math.h>

#define BLOCKSIZE 32

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		system("pause");
		exit(1);
	}
}

__shared__ float sharedMatM[BLOCKSIZE * BLOCKSIZE];
__shared__ float sharedMatm[BLOCKSIZE * BLOCKSIZE];

__global__ void calculateMin(const float* const d_logLuminance,
	float *min_logLum,
	const size_t numRows,
	const size_t numCols){

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
	sharedMatm[posThreadBlock] = d_logLuminance[thread_1D_pos];

	__syncthreads();
	
	//Ahora iteraremos sobre los elementos de memoria compartida para ir comparando y obtener el elemento menor.
	for (int i = BLOCKSIZE * BLOCKSIZE / 2; i > 0; i /= 2){
		if (posThreadBlock < i){
			if (sharedMatm[posThreadBlock] > sharedMatm[posThreadBlock + i])
				sharedMatm[posThreadBlock] = sharedMatm[posThreadBlock + i];
		}
		__syncthreads();
	}


	if (posThreadBlock == 0){
		if (sharedMatm[0] < min_logLum[blockIdx.x])
			min_logLum[blockIdx.x] = sharedMatm[0];
	}
}

__global__ void calculateMax(const float* const d_logLuminance,
	float *max_logLum,
	const size_t numRows,
	const size_t numCols){

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

	__syncthreads();

	//Ahora iteraremos sobre los elementos de memoria compartida para ir comparando y obtener el elemento mayor.
	for (int i = BLOCKSIZE * BLOCKSIZE / 2; i > 0; i /= 2){
		if (posThreadBlock < i){
			if (sharedMatM[posThreadBlock] < sharedMatM[posThreadBlock + i])
				sharedMatM[posThreadBlock] = sharedMatM[posThreadBlock + i];
		}
		__syncthreads();
	}


	if (posThreadBlock == 0){
		if (sharedMatM[0] > max_logLum[blockIdx.x])
			max_logLum[blockIdx.x] = sharedMatM[0];
	}

}

__global__ void histograma(const float* const d_logLuminance, 
	float min_logLum,
	float max_logLum, 
	const size_t numRows,
	const size_t numCols,
	const size_t numBins, 
	unsigned int *histo){

	//Conseguimos la posición del píxel en la imagen del que se ocupará el hilo
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
	//Calculamos la posición del hilo en el bloque
	const int posThreadBlock = threadIdx.x * BLOCKSIZE + threadIdx.y;

	//Si estamos fuera de los límites de la imagen, paramos
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;
	
	/*if (posThreadBlock == 0)
		printf("MIN %f Y MAX %f\n", min_logLum, max_logLum);*/

	int bin = (int) ((d_logLuminance[posThreadBlock] - min_logLum) / (fabs(max_logLum - min_logLum)) * numBins);
	atomicAdd(&histo[bin], 1);
	//printf("Valor %d es %d\n", bin, histo[bin]);
}

__global__ void exclusiveScan(unsigned int *histo, const size_t numBins){

	__shared__ int tempArray[BLOCKSIZE * BLOCKSIZE];

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int threadId = threadIdx.x;
	int offset = 1, temp;
	int ai = threadId;
	int bi = threadId + numBins / 2;
	
	tempArray[ai] = histo[id];
	tempArray[bi] = histo[id + numBins / 2];

	//if (id>0 && id<100)
	//	printf("ID:%d Histograma: %u\n", (id + numBins / 2), histo[id + numBins / 2]);
	//printf("ID Thread2: %d\n", id);
	for (int i = numBins >> 1; i > 0; i >>= 1){
		__syncthreads();
		if (threadId < i){
			ai = offset * (2 * threadId + 1) - 1;
			bi = offset * (2 * threadId + 2) - 1;
			tempArray[bi] += tempArray[ai];
			printf("ID Thread2: %d\n", tempArray[bi]);
		}
		offset <<= 1;
	}
	

	if (threadId == 0){
		tempArray[numBins - 1] = 0;
	}

	for (int i = 1; i < numBins; i <<= 1){
		offset >>= 1;
		__syncthreads();
		if (threadId < i){
			ai = offset * (2 * threadId + 1) - 1;
			bi = offset * (2 * threadId + 2) - 1;
			temp = tempArray[ai];
			tempArray[ai] = tempArray[bi];
			tempArray[bi] += temp;
		}
	}
	
	//if (id>0 && id<100)
		//printf("ID REAL: %u ID:%d Histograma: %u\n",id, (id + numBins / 2), histo[id + numBins / 2]);

	__syncthreads();

	//histo[id] = tempArray[threadId];
	//histo[id + numBins / 2] = tempArray[threadId + numBins / 2];//printf("ID Thread4: %d\n", id);
	

}

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

	//MAXIMO:2.350199

	//TODO: Calcular tamaños de bloque
	const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
	dim3 gridSize((numCols / blockSize.x) + 1, (numRows / blockSize.y) + 1, 1);

	int numBloques = ((numCols * numRows) / (BLOCKSIZE * BLOCKSIZE)) + 2;

	float *myMin, *myMax;

	cudaMalloc((float **)&myMin, sizeof(float) * numBloques);
	cudaMalloc((float **)&myMax, sizeof(float) * numBloques);

	cudaMemset(myMin, min_logLum, sizeof(float) * numBloques);
	cudaMemset(myMax, max_logLum, sizeof(float) * numBloques);

	calculateMin << < gridSize, blockSize >> >(d_logLuminance, myMin, numRows, numCols);
	calculateMax << < gridSize, blockSize >> >(d_logLuminance, myMax, numRows, numCols);
	
	//Lanzamos kernels de manera iterativa hasta que solo quede un valor, el valor final.
	for (int i = numBloques; i > 1; i /= BLOCKSIZE * BLOCKSIZE){
		dim3 newGridSize((sqrt(numBloques) / blockSize.x) + 1, (sqrt(numBloques) / blockSize.y) + 1, 1);
		calculateMin << < newGridSize, blockSize >> >(myMin, myMin, sqrt(numBloques) + 1, sqrt(numBloques) + 1);
		calculateMax << < newGridSize, blockSize >> >(myMax, myMax, sqrt(numBloques) + 1, sqrt(numBloques) + 1);
		numBloques /= (BLOCKSIZE * BLOCKSIZE);
	}

	cudaMemcpy(&min_logLum, myMin, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&max_logLum, myMax, sizeof(float), cudaMemcpyDeviceToHost);

	//printf("MIN %f Y MAX %f\n", min_logLum, max_logLum);

	unsigned int *myHisto;
	unsigned int *hostHisto = new unsigned int[numBins];
	
	cudaMalloc((int **)&myHisto, sizeof(unsigned int) * numBins);
	//cudaMalloc((int **)&hostHisto, sizeof(unsigned int) * numBins);
	cudaMemset(myHisto, 0, sizeof(unsigned int) * numBins);

	//Lanzamos el kernel para la creación de histogramas.
	histograma << < gridSize, blockSize >> >(d_logLuminance, min_logLum, max_logLum, numRows, numCols, numBins, myHisto);

	/*cudaMemcpy(&min_logLum, myMin, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&max_logLum, myMax, sizeof(float), cudaMemcpyDeviceToHost);*/

	//cudaMalloc((unsigned int **)&hostHisto, sizeof(unsigned int) * numBins);
	cudaMemcpy(hostHisto, myHisto, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost);

	//for (unsigned int i = 0; i < numBins; i++)
	//{
	//	printf("id: %d  value: %u\n", i, hostHisto[i]);
	//}

	exclusiveScan << < gridSize, blockSize >> >(myHisto, numBins);

	//cudaMemcpy(d_cdf, myHisto, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize(); 
	checkCudaErrors(cudaGetLastError());
}
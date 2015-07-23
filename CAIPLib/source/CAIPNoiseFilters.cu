/**
\file
\authors Касимов Ильдар
\date 15.07.2015

\brief Файл ядра CAIPNoiseFilters.cu содержит определения ядер для выполнения
операции фильтрации шума.

Файл содердит определения 3-х ядер для графического процессора.

Функция PepperNoiseFilter предназначена для удаления черного шума (pepper noise) на grayscale 
изображениях. Функция SlatNoiseFilter предназначена для удаления белого шума (salt noise) на
том же цветовом пространстве. Предполагается их совместное использование для получения итогового
результата.

Функция MedianNoiseFilter осуществляет медианную фильтрацию grayscale изображения.

*/


#include "..\include\CAIPKernels.h"
#include "..\include\CAIPUtils.h"


__global__ void PepperNoiseFilter(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int index;

	unsigned char currCellValue = GetColorAndIndex(inputImage, x, y, imageWidth, imageHeight, index).mR;

	__shared__ int dx[4];
	__shared__ int dy[4];

	dx[0] = -1; dx[1] = 1; dx[2] = 0; dx[3] = 0; 
	dy[0] =  0; dy[1] = 0; dy[2] = -1; dy[3] = 1; 

	__syncthreads();

	unsigned short averageOfNeighboursValues = 0;

	for (int i = 0; i < 4; i++)
		averageOfNeighboursValues += GetColor(inputImage, x + dx[i], y + dy[i], imageWidth, imageHeight).mR;

	averageOfNeighboursValues >>= 2; //делление на 4

	unsigned char assignedValue = Max<unsigned char>(averageOfNeighboursValues, currCellValue);

	memset(&outputImage[index], assignedValue, sizeof(TColor));
}

__global__ void SaltNoiseFilter(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int index;

	unsigned char currCellValue = GetColorAndIndex(inputImage, x, y, imageWidth, imageHeight, index).mR;

	__shared__ int dx[4];
	__shared__ int dy[4];

	dx[0] = -1; dx[1] = 1; dx[2] = 0; dx[3] = 0;
	dy[0] = 0; dy[1] = 0; dy[2] = -1; dy[3] = 1;

	__syncthreads();

	unsigned short averageOfNeighboursValues = 0;

	for (int i = 0; i < 4; i++)
		averageOfNeighboursValues += GetColor(inputImage, x + dx[i], y + dy[i], imageWidth, imageHeight).mR;

	averageOfNeighboursValues >>= 2; //деление на 4

	unsigned char assignedValue = Min<unsigned char>(averageOfNeighboursValues, currCellValue);

	memset(&outputImage[index], assignedValue, sizeof(TColor));
}

__global__ void MedianNoiseFilter(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int index = x + y * imageWidth;

	unsigned char median = 0;

	unsigned char tmp[4];

	tmp[0] = GetColor(inputImage, x + 1, y, imageWidth, imageHeight).mR;
	tmp[1] = GetColor(inputImage, x - 1, y, imageWidth, imageHeight).mR;
	tmp[2] = GetColor(inputImage, x, y + 1, imageWidth, imageHeight).mR;
	tmp[3] = GetColor(inputImage, x, y - 1, imageWidth, imageHeight).mR;

	unsigned char min = Min(tmp[0], Min(tmp[1], Min(tmp[2], tmp[3])));
	unsigned char max = Max(tmp[0], Max(tmp[1], Max(tmp[2], tmp[3])));

	unsigned short sumOfMiddle = (tmp[0] + tmp[1] + tmp[2] + tmp[3]) - min - max;

	median = sumOfMiddle / 2;

	memset(&outputImage[index], median, sizeof(TColor));
}
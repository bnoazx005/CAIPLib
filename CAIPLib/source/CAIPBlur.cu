/**
\file
\authors Касимов Ильдар
\date 15.07.2015

\brief Файл ядра CAIPBlur.cu содержит определения ядер для выполнения
операции размытия изображения.

Файл содердит определение ядра для равномерного размытия (box blur).

*/


#include "..\include\CAIPKernels.h"
#include "..\include\CAIPUtils.h"


__global__ void BoxBlur(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int index = x + y * imageWidth;

	unsigned short averageOfNeigboursValues = 0;
	
	__shared__ int dx[9];
	__shared__ int dy[9];

	dx[0] = 0; dx[1] = 1; dx[2] = -1; dx[3] = 0; dx[4] =  0; dx[5] = 1; dx[6] = -1; dx[7] =  1; dx[8] = -1;
	dy[0] = 0; dy[1] = 0; dy[2] =  0; dy[3] = 1; dy[4] = -1; dy[5] = 1; dy[6] =  1; dy[7] = -1; dy[8] = -1;

	__syncthreads();

	for (int i = 0; i < 9; i++)
		averageOfNeigboursValues += GetColor(inputImage, x + dx[i], y + dy[i], imageWidth, imageHeight).mR;

	averageOfNeigboursValues /= 9;

	memset(&outputImage[index], averageOfNeigboursValues, sizeof(TColor));
}
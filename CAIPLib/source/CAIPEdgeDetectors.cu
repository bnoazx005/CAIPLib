/**
\file
\authors Касимов Ильдар
\date 15.07.2015

\brief Файл ядра EdgeDetectors.cu содержит определения ядер для выполнения
операции выделения границ на изображении.

Файл содердит определения 3-х ядер для графического процессора, которые 
выполняют вышеуказанную операцию разными способами. Их объявления содержатся
в файле CAIPLib.h.
*/

#include "..\include\CAIPKernels.h"
#include "..\include\CAIPUtils.h"


__global__ void EdgesV1(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight, unsigned char threshold)
{
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int index = x + y * imageWidth;

	unsigned char neighbours[4];

	neighbours[0] = GetColor(inputImage, x - 1, y, imageWidth, imageHeight).mR;
	neighbours[1] = GetColor(inputImage, x + 1, y, imageWidth, imageHeight).mR;
	neighbours[2] = GetColor(inputImage, x, y + 1, imageWidth, imageHeight).mR;
	neighbours[3] = GetColor(inputImage, x, y - 1, imageWidth, imageHeight).mR;

	unsigned char currCellValue = GetColor(inputImage, x, y, imageWidth, imageHeight).mR;

	unsigned int logicCondition = (abs(currCellValue - neighbours[0]) >= threshold) +
                                  (abs(currCellValue - neighbours[1]) >= threshold) +
                                  (abs(currCellValue - neighbours[2]) >= threshold) +
                                  (abs(currCellValue - neighbours[3]) >= threshold);

	unsigned char k = Min<unsigned char>(Max<unsigned char>(logicCondition, 0), 1);

	memset(&outputImage[index], unsigned char(k * 255), sizeof(TColor));
}

__global__ void EdgesV2(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight, unsigned char threshold)
{
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int index = x + y * imageWidth;

	//uchar4 currValue = GetColor(inputImage, x, y);

	unsigned char neighbours[4];
	
	neighbours[0] = GetColor(inputImage, x - 1, y, imageWidth, imageHeight).mR;
	neighbours[1] = GetColor(inputImage, x + 1, y, imageWidth, imageHeight).mR;
	neighbours[2] = GetColor(inputImage, x, y + 1, imageWidth, imageHeight).mR;
	neighbours[3] = GetColor(inputImage, x, y - 1, imageWidth, imageHeight).mR;

	unsigned int logicCondition = (abs(neighbours[0] - neighbours[1]) > threshold) +
		                          (abs(neighbours[2] - neighbours[3]) > threshold) +
								  (abs(neighbours[0] - neighbours[2]) > threshold) +
		                          (abs(neighbours[1] - neighbours[3]) > threshold) +
		                          (abs(neighbours[0] - neighbours[3]) > threshold) +
		                          (abs(neighbours[1] - neighbours[2]) > threshold);

	unsigned char k = Min<unsigned char>(Max<unsigned char>(logicCondition, 0), 1);

	memset(&outputImage[index], unsigned char(k * 255), sizeof(TColor));
}

__global__ void EdgesV3(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int index = x + y * imageWidth;

	unsigned char gx = 0;
	unsigned char gy = 0;

	unsigned char currCellValue = GetColor(inputImage, x, y, imageWidth, imageHeight).mR;
	unsigned char leftNeighbourCellValue = GetColor(inputImage, x - 1, y, imageWidth, imageHeight).mR;
	unsigned char topNeighbourCellValue = GetColor(inputImage, x, y - 1, imageWidth, imageHeight).mR;

	unsigned char gradientVecMag = 0;

	gx = abs(currCellValue - leftNeighbourCellValue);
	gy = abs(currCellValue - topNeighbourCellValue);

	gradientVecMag = sqrt(float(gx * gx + gy * gy));

	memset(&outputImage[index], gradientVecMag, sizeof(TColor));
}
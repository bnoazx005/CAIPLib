/**
\file
\authors Касимов Ильдар
\date 15.07.2015

\brief Файл ядра CAIPImageSpaceConverters.cu содержит определения ядер для выполнения
операции преобразования из одного цветового пространства в другое.

Файл содердит определения 2-х ядер для графического процессора. 

Первое преобразует RGB пространство в grayscale пространство методом усреднения суммы 3-х цветовых 
компонент пикселя. Второе выполняет операцию бинаризации над RGB изображением по заданому порогу.

*/

#include "..\include\CAIPKernels.h"
#include "..\include\CAIPUtils.h"


__global__ void Rgb2Gray(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	unsigned int index;

	TColor color = GetColorAndIndex(inputImage, x, y, imageWidth, imageHeight, index);

	unsigned char value = (color.mR + color.mG + color.mB) / 3;

	memset(&outputImage[index], value, sizeof(TColor));
}

/**
*/
__global__ void Rgb2Bin(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight, unsigned char threshold)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int index;

	TColor color = GetColorAndIndex(inputImage, x, y, imageWidth, imageHeight, index);

	unsigned char value = (color.mR + color.mG + color.mB) / 3;

	value = Min<unsigned char>(Max<short>(Max<short>(value - threshold, 0), 0), 1) * 255;

	memset(&outputImage[index], value, sizeof(TColor));
}
/**
\file
\authors ������� ������
\date 15.07.2015

\brief ���� ���� CAIPImageSpaceConverters.cu �������� ����������� ���� ��� ����������
�������� �������������� �� ������ ��������� ������������ � ������.

���� �������� ����������� 2-� ���� ��� ������������ ����������. 

������ ����������� RGB ������������ � grayscale ������������ ������� ���������� ����� 3-� �������� 
��������� �������. ������ ��������� �������� ����������� ��� RGB ������������ �� �������� ������.

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
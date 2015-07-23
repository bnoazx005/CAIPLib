/**
	\file
	\authors ������� ������
	\date 15.07.2015

	\brief ������������ ���� CAIPKernels.h �������� ����������  �������, ����������� �� �����������
	����������.

	������ ������� ���������� � ��������� ������ �� ���� ��������. 
*/

#ifndef CAIP_KERNELS_H
#define CAIP_KERNELS_H


#include "cuda_runtime.h"
#include "..\include\CAIPTypes.h"


//�������-����

//�������� �������� �� ������ ��������� ������������ � ������
	
DLL __global__ void Rgb2Gray(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight);

DLL __global__ void Rgb2Bin(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight, unsigned char threshold = 128);

//�������� ��������� ������

DLL __global__ void EdgesV1(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight, unsigned char threshold = 22);

DLL __global__ void EdgesV2(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight, unsigned char threshold = 22);

DLL __global__ void EdgesV3(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight);

//�������� ��������

DLL __global__ void BoxBlur(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight);

//�������� ���������� ����

DLL __global__ void PepperNoiseFilter(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight);

DLL __global__ void SaltNoiseFilter(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight);

DLL __global__ void MedianNoiseFilter(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight);

#endif
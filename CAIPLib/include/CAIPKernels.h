/**
	\file
	\authors Касимов Ильдар
	\date 15.07.2015

	\brief Заголовочный файл CAIPKernels.h содержит объявления  функций, выполняемых на графическом
	процессоре.

	Данные функции определены в отдельных файлах по типу операций. 
*/

#ifndef CAIP_KERNELS_H
#define CAIP_KERNELS_H


#include "cuda_runtime.h"
#include "..\include\CAIPTypes.h"


//Функции-ядра

//Операции перевода из одного цветового пространства в другое
	
DLL __global__ void Rgb2Gray(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight);

DLL __global__ void Rgb2Bin(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight, unsigned char threshold = 128);

//Операции выделения границ

DLL __global__ void EdgesV1(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight, unsigned char threshold = 22);

DLL __global__ void EdgesV2(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight, unsigned char threshold = 22);

DLL __global__ void EdgesV3(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight);

//Операции размытия

DLL __global__ void BoxBlur(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight);

//Операции фильтрации шума

DLL __global__ void PepperNoiseFilter(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight);

DLL __global__ void SaltNoiseFilter(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight);

DLL __global__ void MedianNoiseFilter(TColor* inputImage, TColor* outputImage, int imageWidth, int imageHeight);

#endif
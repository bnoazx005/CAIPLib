/**
\file
\authors  асимов »льдар
\date 17.07.2015

\brief ‘айл CAIPLib.cu содержит определени€ функций, определенных в CAIPLib.h.

ƒанный файл содержит определени€ всех пользовательских функций. ѕользователь 
библиотеки взаимодействует только с данным интерфейсом, не прибега€ к использованию
вызовов CUDA вручную.

*/


#include "..\include\CAIPLib.h"
#include "..\include\CAIPKernels.h"
#include "..\include\CAIPUtils.h"


extern "C" caError caLoadImage(const caChar8* filename, TImage& resultImage)
{
	if (!filename)
		return ET_INVALID_ARGUMENT;

	TColor* colorBuffer = NULL;
	TTGAHeader tgaHeader;

	caError error = caLoadTGAFile(filename, &colorBuffer, &tgaHeader);

	if (caCheckError(error))
		return error;

	resultImage.mpColorBuffer = colorBuffer;

	resultImage.mWidth = tgaHeader.mWidth;
	resultImage.mHeight = tgaHeader.mHeight;

	return ET_SUCCESS;
}

extern "C" caError caSaveImage(const caChar8* filename, const TImage& image)
{
	if (!filename || !image.mpColorBuffer || image.mHeight < 0 || image.mWidth < 0)
		return ET_INVALID_ARGUMENT;

	caError error = caSaveTGAFile(filename, image.mpColorBuffer, image.mWidth, image.mHeight, 32); //в бибилиотеке используютс€ только 32-ый формат пиксел€

	if (caCheckError(error))
		return error;

	return ET_SUCCESS;
}

extern "C" caError caFreeImage(TImage& image)
{
	if (!image.mpColorBuffer)
		return ET_FAIL;

	delete[] image.mpColorBuffer;

	memset(&image, 0, sizeof(image));

	return ET_SUCCESS;
}

extern "C" caError caRgb2Gray(const TImage& inputImage, TImage& resultImage)
{
	dim3 gridDim, threadsDim;

	caError error;

	if (caCheckError(error = caConfigGridForImage(inputImage, gridDim, threadsDim)))
		return error;

	TColor* pDeviceInputImage = NULL;
	TColor* pDeviceResultImage = NULL;

	caUInt64 allocatedSize = inputImage.mHeight * inputImage.mWidth * sizeof(TColor);

	if (caCheckCudaError(cudaMalloc(&pDeviceInputImage, allocatedSize + sizeof(TColor)))) //0-ой индекс зарезервирован дл€ внутреннего использовани€
		return ET_FAIL;

	if (caCheckCudaError(cudaMalloc(&pDeviceResultImage, allocatedSize)))
		return ET_FAIL;

	if (caCheckCudaError(cudaMemset(pDeviceInputImage, 0, allocatedSize + sizeof(TColor))))
		return ET_FAIL;

	if (caCheckCudaError(cudaMemcpy(&pDeviceInputImage[1], inputImage.mpColorBuffer, allocatedSize, cudaMemcpyHostToDevice)))
		return ET_FAIL;

	Rgb2Gray << <gridDim, threadsDim >> >(pDeviceInputImage, pDeviceResultImage, inputImage.mWidth, inputImage.mHeight);

	TColor* pTmpColorBuffer = resultImage.mpColorBuffer;

	if (!pTmpColorBuffer)
		pTmpColorBuffer = new TColor[inputImage.mWidth * inputImage.mHeight];

	memset(pTmpColorBuffer, 0, allocatedSize);

	if (caCheckCudaError(cudaMemcpy(pTmpColorBuffer, pDeviceResultImage, allocatedSize, cudaMemcpyDeviceToHost)))
	{
		if (pTmpColorBuffer)
			delete[] pTmpColorBuffer;

		return ET_FAIL;
	}

	resultImage.mpColorBuffer = pTmpColorBuffer;
	resultImage.mWidth = inputImage.mWidth;
	resultImage.mHeight = inputImage.mHeight;

	if (caCheckCudaError(cudaFree(pDeviceInputImage)))
		return ET_MEMORY_LEAK_DETECTED;

	if (caCheckCudaError(cudaFree(pDeviceResultImage)))
		return ET_MEMORY_LEAK_DETECTED;

	return ET_SUCCESS;
}

extern "C" caError caRgb2Bin(const TImage& inputImage, TImage& resultImage, caUInt8 threshold)
{
	dim3 gridDim, threadsDim;

	caError error;

	if (caCheckError(error = caConfigGridForImage(inputImage, gridDim, threadsDim)))
		return error;

	TColor* pDeviceInputImage = NULL;
	TColor* pDeviceResultImage = NULL;

	caUInt64 allocatedSize = inputImage.mHeight * inputImage.mWidth * sizeof(TColor);

	if (caCheckCudaError(cudaMalloc(&pDeviceInputImage, allocatedSize + sizeof(TColor)))) //0-ой индекс зарезервирован дл€ внутреннего использовани€
		return ET_FAIL;

	if (caCheckCudaError(cudaMalloc(&pDeviceResultImage, allocatedSize)))
		return ET_FAIL;

	if (caCheckCudaError(cudaMemset(pDeviceInputImage, 0, allocatedSize + sizeof(TColor))))
		return ET_FAIL;

	if (caCheckCudaError(cudaMemcpy(&pDeviceInputImage[1], inputImage.mpColorBuffer, allocatedSize, cudaMemcpyHostToDevice)))
		return ET_FAIL;

	Rgb2Bin << <gridDim, threadsDim >> >(pDeviceInputImage, pDeviceResultImage, inputImage.mWidth, inputImage.mHeight, threshold);

	TColor* pTmpColorBuffer = resultImage.mpColorBuffer;

	if (!pTmpColorBuffer)
		pTmpColorBuffer = new TColor[inputImage.mWidth * inputImage.mHeight];

	memset(pTmpColorBuffer, 0, allocatedSize);

	if (caCheckCudaError(cudaMemcpy(pTmpColorBuffer, pDeviceResultImage, allocatedSize, cudaMemcpyDeviceToHost)))
	{
		if (pTmpColorBuffer)
			delete[] pTmpColorBuffer;

		return ET_FAIL;
	}

	resultImage.mpColorBuffer = pTmpColorBuffer;
	resultImage.mWidth = inputImage.mWidth;
	resultImage.mHeight = inputImage.mHeight;

	if (caCheckCudaError(cudaFree(pDeviceInputImage)))
		return ET_MEMORY_LEAK_DETECTED;

	if (caCheckCudaError(cudaFree(pDeviceResultImage)))
		return ET_MEMORY_LEAK_DETECTED;

	return ET_SUCCESS;
}

extern "C" caError caEdges(caUInt8 detectorType, const TImage& inputImage, TImage& resultImage, caUInt8 threshold)
{
	dim3 gridDim, threadsDim;

	caError error;

	if (caCheckError(error = caConfigGridForImage(inputImage, gridDim, threadsDim)))
		return error;

	TColor* pDeviceInputImage = NULL;
	TColor* pDeviceResultImage = NULL;

	caUInt64 allocatedSize = inputImage.mHeight * inputImage.mWidth * sizeof(TColor);

	if (caCheckCudaError(cudaMalloc(&pDeviceInputImage, allocatedSize + sizeof(TColor)))) //0-ой индекс зарезервирован дл€ внутреннего использовани€
		return ET_FAIL;

	if (caCheckCudaError(cudaMalloc(&pDeviceResultImage, allocatedSize)))
		return ET_FAIL;

	if (caCheckCudaError(cudaMemset(pDeviceInputImage, 0, allocatedSize + sizeof(TColor))))
		return ET_FAIL;

	if (caCheckCudaError(cudaMemcpy(&pDeviceInputImage[1], inputImage.mpColorBuffer, allocatedSize, cudaMemcpyHostToDevice)))
		return ET_FAIL;

	switch (detectorType)
	{
		case EDT_LUM_DIFF:
			EdgesV1 << <gridDim, threadsDim >> >(pDeviceInputImage, pDeviceResultImage, inputImage.mWidth, inputImage.mHeight, threshold);
			break;
		case EDT_ADV_LUM_DIFF:
			EdgesV2 << <gridDim, threadsDim >> >(pDeviceInputImage, pDeviceResultImage, inputImage.mWidth, inputImage.mHeight, threshold);
			break;
		case EDT_GRADIENT:
			EdgesV3 << <gridDim, threadsDim >> >(pDeviceInputImage, pDeviceResultImage, inputImage.mWidth, inputImage.mHeight);
			break;
		case EDT_DEFAULT:
			return ET_UNKNOWN_EDGE_DETECTOR_TYPE;
	}

	TColor* pTmpColorBuffer = resultImage.mpColorBuffer;

	if (!pTmpColorBuffer)
		pTmpColorBuffer = new TColor[inputImage.mWidth * inputImage.mHeight];

	memset(pTmpColorBuffer, 0, allocatedSize);

	if (caCheckCudaError(cudaMemcpy(pTmpColorBuffer, pDeviceResultImage, allocatedSize, cudaMemcpyDeviceToHost)))
	{
		if (pTmpColorBuffer)
			delete[] pTmpColorBuffer;

		return ET_FAIL;
	}

	resultImage.mpColorBuffer = pTmpColorBuffer;
	resultImage.mWidth = inputImage.mWidth;
	resultImage.mHeight = inputImage.mHeight;

	if (caCheckCudaError(cudaFree(pDeviceInputImage)))
		return ET_MEMORY_LEAK_DETECTED;

	if (caCheckCudaError(cudaFree(pDeviceResultImage)))
		return ET_MEMORY_LEAK_DETECTED;

	return ET_SUCCESS;
}

extern "C" caError caBoxBlur(const TImage& inputImage, TImage& resultImage, caUInt16 numOfIterations)
{
	dim3 gridDim, threadsDim;

	caError error;

	if (caCheckError(error = caConfigGridForImage(inputImage, gridDim, threadsDim)))
		return error;

	TColor* pDeviceInputImage = NULL;
	TColor* pDeviceResultImage = NULL;

	caUInt64 allocatedSize = inputImage.mHeight * inputImage.mWidth * sizeof(TColor);

	if (caCheckCudaError(cudaMalloc(&pDeviceInputImage, allocatedSize + sizeof(TColor)))) //0-ой индекс зарезервирован дл€ внутреннего использовани€
		return ET_FAIL;

	if (caCheckCudaError(cudaMalloc(&pDeviceResultImage, allocatedSize)))
		return ET_FAIL;

	if (caCheckCudaError(cudaMemset(pDeviceInputImage, 0, allocatedSize + sizeof(TColor))))
		return ET_FAIL;

	TColor* pTmpColorBuffer = inputImage.mpColorBuffer;

	for (caUInt16 i = 0; i < numOfIterations; i++)
	{
		if (caCheckCudaError(cudaMemcpy(&pDeviceInputImage[1], pTmpColorBuffer, allocatedSize, cudaMemcpyHostToDevice)))
			return ET_FAIL;

		BoxBlur << <gridDim, threadsDim >> >(pDeviceInputImage, pDeviceResultImage, inputImage.mWidth, inputImage.mHeight);

		if (caCheckCudaError(cudaMemcpy(pTmpColorBuffer, pDeviceResultImage, allocatedSize, cudaMemcpyDeviceToHost)))
		{
			if (pTmpColorBuffer)
				delete[] pTmpColorBuffer;

			return ET_FAIL;
		}
	}

	resultImage.mpColorBuffer = pTmpColorBuffer;
	resultImage.mWidth = inputImage.mWidth;
	resultImage.mHeight = inputImage.mHeight;

	if (caCheckCudaError(cudaFree(pDeviceInputImage)))
		return ET_MEMORY_LEAK_DETECTED;

	if (caCheckCudaError(cudaFree(pDeviceResultImage)))
		return ET_MEMORY_LEAK_DETECTED;

	return ET_SUCCESS;
}

extern "C" caError caNoiseFilter(caUInt8 filterType, const TImage& inputImage, TImage& resultImage, caUInt8 numOfIterations)
{
	dim3 gridDim, threadsDim;

	caError error;

	if (caCheckError(error = caConfigGridForImage(inputImage, gridDim, threadsDim)))
		return error;

	TColor* pDeviceInputImage = NULL;
	TColor* pDeviceResultImage = NULL;

	caUInt64 allocatedSize = inputImage.mHeight * inputImage.mWidth * sizeof(TColor);

	if (caCheckCudaError(cudaMalloc(&pDeviceInputImage, allocatedSize + sizeof(TColor)))) //0-ой индекс зарезервирован дл€ внутреннего использовани€
		return ET_FAIL;

	if (caCheckCudaError(cudaMalloc(&pDeviceResultImage, allocatedSize)))
		return ET_FAIL;

	if (caCheckCudaError(cudaMemset(pDeviceInputImage, 0, allocatedSize + sizeof(TColor))))
		return ET_FAIL;

	TColor* pTmpColorBuffer = inputImage.mpColorBuffer;

	switch (filterType)
	{
	case NFT_PEPPER:

		for (caUInt16 i = 0; i < numOfIterations; i++)
		{
			if (caCheckCudaError(cudaMemcpy(&pDeviceInputImage[1], pTmpColorBuffer, allocatedSize, cudaMemcpyHostToDevice)))
				return ET_FAIL;

			PepperNoiseFilter << <gridDim, threadsDim >> >(pDeviceInputImage, pDeviceResultImage, inputImage.mWidth, inputImage.mHeight);

			if (caCheckCudaError(cudaMemcpy(pTmpColorBuffer, pDeviceResultImage, allocatedSize, cudaMemcpyDeviceToHost)))
			{
				if (pTmpColorBuffer)
					delete[] pTmpColorBuffer;

				return ET_FAIL;
			}
		}

		break;
	case NFT_SALT:

		for (caUInt16 i = 0; i < numOfIterations; i++)
		{
			if (caCheckCudaError(cudaMemcpy(&pDeviceInputImage[1], pTmpColorBuffer, allocatedSize, cudaMemcpyHostToDevice)))
				return ET_FAIL;

			SaltNoiseFilter << <gridDim, threadsDim >> >(pDeviceInputImage, pDeviceResultImage, inputImage.mWidth, inputImage.mHeight);

			if (caCheckCudaError(cudaMemcpy(pTmpColorBuffer, pDeviceResultImage, allocatedSize, cudaMemcpyDeviceToHost)))
			{
				if (pTmpColorBuffer)
					delete[] pTmpColorBuffer;

				return ET_FAIL;
			}
		}

		break;
	case NFT_SALT_AND_PEPPER:

		for (caUInt16 i = 0; i < numOfIterations; i++)
		{
			if (caCheckCudaError(cudaMemcpy(&pDeviceInputImage[1], pTmpColorBuffer, allocatedSize, cudaMemcpyHostToDevice)))
				return ET_FAIL;

			PepperNoiseFilter << <gridDim, threadsDim >> >(pDeviceInputImage, pDeviceResultImage, inputImage.mWidth, inputImage.mHeight);

			if (caCheckCudaError(cudaMemcpy(pTmpColorBuffer, pDeviceResultImage, allocatedSize, cudaMemcpyDeviceToHost)))
			{
				if (pTmpColorBuffer)
					delete[] pTmpColorBuffer;

				return ET_FAIL;
			}
		}

		for (caUInt16 i = 0; i < numOfIterations; i++)
		{
			if (caCheckCudaError(cudaMemcpy(&pDeviceInputImage[1], pTmpColorBuffer, allocatedSize, cudaMemcpyHostToDevice)))
				return ET_FAIL;

			SaltNoiseFilter << <gridDim, threadsDim >> >(pDeviceInputImage, pDeviceResultImage, inputImage.mWidth, inputImage.mHeight);

			if (caCheckCudaError(cudaMemcpy(pTmpColorBuffer, pDeviceResultImage, allocatedSize, cudaMemcpyDeviceToHost)))
			{
				if (pTmpColorBuffer)
					delete[] pTmpColorBuffer;

				return ET_FAIL;
			}
		}

		break;
	case NFT_MEDIAN:

		for (caUInt16 i = 0; i < numOfIterations; i++)
		{
			if (caCheckCudaError(cudaMemcpy(&pDeviceInputImage[1], pTmpColorBuffer, allocatedSize, cudaMemcpyHostToDevice)))
				return ET_FAIL;

			MedianNoiseFilter << <gridDim, threadsDim >> >(pDeviceInputImage, pDeviceResultImage, inputImage.mWidth, inputImage.mHeight);

			if (caCheckCudaError(cudaMemcpy(pTmpColorBuffer, pDeviceResultImage, allocatedSize, cudaMemcpyDeviceToHost)))
			{
				if (pTmpColorBuffer)
					delete[] pTmpColorBuffer;

				return ET_FAIL;
			}
		}

		break;
	case NFT_DEFAULT:
		return ET_UNKNOWN_NOISE_FILTER_TYPE;
	}

	resultImage.mpColorBuffer = pTmpColorBuffer;
	resultImage.mWidth = inputImage.mWidth;
	resultImage.mHeight = inputImage.mHeight;

	if (caCheckCudaError(cudaFree(pDeviceInputImage)))
		return ET_MEMORY_LEAK_DETECTED;

	if (caCheckCudaError(cudaFree(pDeviceResultImage)))
		return ET_MEMORY_LEAK_DETECTED;

	return ET_SUCCESS;
}

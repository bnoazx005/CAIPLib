/**
	\file
	\authors Касимов Ильдар
	\date 14.07.2015

	\brief Файл CAIPUtils.cpp содержит определения дополнительных функций.

	функции, расположенные в данном файле, используются для осуществления 
	различных вспомогательных действий.
*/

#include "..\include\CAIPUtils.h"
#include <fstream>


extern "C" caError caLoadTGAFile(const caChar8* name, TColor** resultBuffer, TTGAHeader* resultHeader)
{
	if (!name)
		return ET_INVALID_ARGUMENT;

	if (*resultBuffer)
		delete[] *resultBuffer;

	std::ifstream inputFile(name, std::ios::binary);

	if (!inputFile.is_open())
		return ET_INVALID_FILE_OPERATION;

	TTGAHeader header;

	inputFile.read((char*)&header, sizeof(header));

	if (header.mImageType != 2)
		return ET_INVALID_TGA_TYPE;

	unsigned int imageSizes = header.mHeight * header.mWidth;

	*resultBuffer = new TColor[imageSizes];

	inputFile.read((char*)(*resultBuffer), imageSizes * (header.mBitsPerPixel >> 3));

	if (!inputFile.good())
		return ET_INVALID_FILE_OPERATION;

	memcpy(resultHeader, &header, sizeof(TTGAHeader));

	inputFile.close();

	return ET_SUCCESS;
}

extern "C" caError caSaveTGAFile(const caChar8* name, const TColor* resultBuffer, caInt16 width, caInt16 height, caInt8 bpp)
{
	if (!name || !resultBuffer || !width || !height || !bpp)
		return ET_INVALID_ARGUMENT;

	std::ofstream outputFile(name, std::ios::binary);

	TTGAHeader header;
	memset(&header, 0, sizeof(header));

	header.mBitsPerPixel = bpp;
	header.mWidth = width;
	header.mHeight = height;
	header.mImageType = 2;

	unsigned int imageSizes = width * height * (bpp >> 3);

	outputFile.write((char*)&header, sizeof(header));

	if (!outputFile.good())
		return ET_INVALID_FILE_OPERATION;

	outputFile.write((char*)resultBuffer, imageSizes);

	if (!outputFile.good())
		return ET_INVALID_FILE_OPERATION;

	return ET_SUCCESS;
}

extern "C" caError caConfigGridForImage(const TImage& currImage, dim3& gridDim, dim3& threadsDim)
{
	if (!currImage.mpColorBuffer || currImage.mHeight < 0 || currImage.mWidth < 0)
		return ET_INVALID_ARGUMENT;

	caInt32 width = currImage.mWidth;
	caInt32 height = currImage.mHeight;

	caInt32 device;
	cudaDeviceProp props;
	
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);

	caInt32 maxThreadsPerAxis = sqrt(props.maxThreadsPerBlock);

	caInt32 longSize = Max(width, height);
	caInt32 currGridX = maxThreadsPerAxis;
	caInt32 currGridY = maxThreadsPerAxis;

	while ((width % currGridX != 0) && (currGridX > 1))
		currGridX--;

	while (height % currGridY != 0 && currGridY > 1)
		currGridY--;

	if (currGridX == 1 || currGridY == 1)
		return ET_INVALID_IMAGE_SIZES;

	gridDim.x = width / currGridX;
	gridDim.y = height / currGridY;

	threadsDim.x = currGridX;
	threadsDim.y = currGridY;

	return ET_SUCCESS;
}
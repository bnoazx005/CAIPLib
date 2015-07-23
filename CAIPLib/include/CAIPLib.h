/**
	\file
	\authors ������� ������
	\date 17.07.2015

	\brief ������������ ���� CAIPLib.h �������� ���������� �������� ��������� � ������� ���������.

	������ ���� �������� �������� ������������ ������ ������ ����������. �������� ��� �����������
	�������� �������, �������� ������, �������� � �.�. ��� ������������� ������ ���������� � �������
	��������� ���������� ������ ���� � ����, ����� ������������ �������� ������� ���������� CUDA.
*/

#ifndef CAIP_LIB_H
#define CAIP_LIB_H


#include <cuda_runtime.h>
#include "..\include\CAIPTypes.h"

#define SHOW_ERROR_DATA

#if defined(SHOW_ERROR_DATA)

	#include <iostream>

	/**
		\brief ������� ���������� ��� ������ �� �� ����

		\param [in] errorType ��� ��������� ������

		\return ���������� ������, ���������� ��� ������
	*/

    extern "C" DLL __forceinline__ const caChar8* caGetErrorName(caError errorType)
	{
		switch(errorType)
		{
			case ET_INVALID_ARGUMENT:
				return "Invalid argument or arguments\0";

			case ET_BAD_ALLOCATION:
				return "Bad allocation\0";

			case ET_INVALID_FILE_OPERATION:
				return "Invalid file operation\0";

			case ET_INVALID_TGA_TYPE:
				return "Invalid TGA type in header\0";

			case ET_FAIL:
				return "Fail\0";

			case ET_INVALID_IMAGE_SIZES:
				return "Invalid image sizes\0";

			case ET_MEMORY_LEAK_DETECTED:
				return "Memory leak was detected\0";

			case ET_UNKNOWN_EDGE_DETECTOR_TYPE:
				return "Unknown an edge detector type\0";

			case ET_UNKNOWN_NOISE_FILTER_TYPE:
				return "Unknown a noise filter type\0";
		}

		return "\0";
	}

	/**
		\brief ������� ��������� ��� ������

		� ������, ���� ������ ��� �� �������� ����� ��������� ����������, ������� �������������
		�� ���� ����� ������������ ��������. ����� ������� ���������� �� ������ � �������.

		\param [in] error ��� ��������� ������

		\return ���������� true � ������ ������������� ������, false � ��������� ������
	*/

	extern "C" DLL __forceinline__ bool caCheckError(caError error)
	{
		if (error == 0x0)
			return false;

		std::cerr << "An error has been occured.\nType : " << std::hex << error << std::dec << ";\n"
			      << "Name: " << caGetErrorName(error) << ";\n";

		return true;
	}

	/**
		\brief ������� ��������� ��� ������ CUDA �������

		� ������, ���� ������ ��� �� �������� ����� ��������� ����������, ������� �������������
		�� ���� ����� ������������ ��������. ����� ������� ���������� �� ������ � �������.

		\param [in] error ��� ��������� ������

		\return ���������� true � ������ ������������� ������, false � ��������� ������
	*/

	extern "C" DLL __forceinline__ bool caCheckCudaError(cudaError_t error)
	{
		if(error == cudaSuccess)
			return false;
		
		std::cerr << "An error has been occured.\nType : " << std::hex << error << std::dec << ";\n"
			      << "Name: " /*<< cudaGetErrorName(error) << */";\n"
			      << "Message: " /*<< cudaGetErrorString(error) << */";\n";

		return true;
	}


#else

	/**
		\brief ������� ��������� ��� ������

		� ������, ���� ������ ��� �� �������� ����� ��������� ����������, ������� �������������
		�� ���� ����� ������������ ��������. 

		\param [in] error ��� ��������� ������

		\return ���������� true � ������ ������������� ������, false � ��������� ������
	*/

    extern "C" DLL  __forceinline__ bool caCheckError(caError error)
	{
		if (error == 0x0)
			return false;

		return true;
	}

	/**
		\brief ������� ��������� ��� ������ CUDA �������

		� ������, ���� ������ ��� �� �������� ����� ��������� ����������, ������� �������������
		�� ���� ����� ������������ ��������.

		\param [in] error ��� ��������� ������

		\return ���������� true � ������ ������������� ������, false � ��������� ������
	*/

	extern "C" DLL __forceinline__ bool caCheckCudaError(cudaError_t error)
	{
		if(error == cudaSuccess)
			return false;

		return true;
	}

#endif


/**
	\brief ������� ��������� TGA ����������� � ������ � �������� �����.

	��������� � ���������������� ����� ����������. ��������� ���������
	�������� TGA ����� � 32-� ������� ���������.

	\param [in] filename ��� �����, ��������� �����������
	\param [out] resultImage ������ �� ��������� TImage, �������� ���������� � ����������� �����������

	\return ���������� ��� ������, ������� ����� �������� � ������� caCheckError ��� �������� ��
	���������� ����������
	*/

	extern "C" DLL caError caLoadImage(const caChar8* filename, TImage& resultImage);

/**
	\brief ������� ��������� TGA ����������� �� �������� �����.

	��������� � ���������������� ����� ����������. ��������� ������������
	���������� �������� TGA ������ � 32-� ������� ���������.

	\param [in] filename ��� �����, ��� ������� ����� ��������� �����������
	\param [in] image ������ �� ��������� TImage, �������� ���������� �� �����������

	\return ���������� ��� ������, ������� ����� �������� � ������� caCheckError ��� �������� ��
	���������� ����������
	*/

	extern "C" DLL caError caSaveImage(const caChar8* filename, const TImage& image);

/**
	\brief ������� ����������� ������ ��-��� �����������.

	��������� � ���������������� ����� ����������.

	\param [in, out] image ������ �� ��������� TImage, �������� ���������� �� �����������

	\return ���������� ��� ������, ������� ����� �������� � ������� caCheckError ��� �������� ��
	���������� ����������
	*/

	extern "C" DLL caError caFreeImage(TImage& image);

///�������, ��������������� ��� ��������� �����������

///�������� ������� �� ������ ��������� ������������ � ������

/**
	\brief �������, �������������� �������� �������������� �� RGB ������������ �
	grayscale.

	������� ���������� ����� �������� ���� �������� ��������� RGB �������.

	\param [in] inputImage ������� �����������
	\param [out] outputImage ��������� ������ �������

	\return ���������� ��� ������, ������� ����� �������� � ������� caCheckError ��� �������� ��
	���������� ����������
	*/

	extern "C" DLL caError caRgb2Gray(const TImage& inputImage, TImage& resultImage);

/**
	\brief �������, �������������� �������� �������������� �� RGB ������������ �
	��������.

	������� ������������� ����������� RGB ���� � grayscale, � ����� �������� �������������
	���� ��� ������ ����������� ������. ��� ���� ���� �������� ������ ����������� � ������,
	���� - �����.

	\param [in] inputImage ������� �����������
	\param [out] outputImage ��������� ������ �������
	\param [in] threshold ����� �����������

	\return ���������� ��� ������, ������� ����� �������� � ������� caCheckError ��� �������� ��
	���������� ����������
	*/

	extern "C" DLL caError caRgb2Bin(const TImage& inputImage, TImage& resultImage, caUInt8 threshold = 128);

///�������� ��������� ������

/**
	\brief �������, ����������� �������� ������ �������� �� �����������.

	����� ����������� ������� ������� ��������� ��������� RGB ����������� � grayscale. � ���������
	������ ��������� ������� ����� ������������.

	\param [in] detectorType ��� ��������� (EDT_LUM_DIFF - �� ������� ��������; EDT_ADV_LUM_DIFF - ������������ ����� ������� ��������;
	EDT_GRADIENT - ��������)

	\param [in] inputImage ������� �����������
	\param [out] outputImage ��������� ������ �������
	\param [in] threshold �����������

	\return ���������� ��� ������, ������� ����� �������� � ������� caCheckError ��� �������� ��
	���������� ����������
	*/

	extern "C" DLL caError caEdges(caUInt8 detectorType, const TImage& inputImage, TImage& resultImage, caUInt8 threshold = 22);

///�������� ��������

/**
	\brief �������, ����������� ������ ������������ ��������.

	����� ����������� ������� ������� ��������� ��������� RGB ����������� � grayscale. � ���������
	������ ��������� ������� ����� ������������.

	\param [in] inputImage ������� �����������
	\param [out] outputImage ��������� ������ �������
	\param [in] numOfIterations ���������� �������� ������� �������, ����������� � �����������

	\return ���������� ��� ������, ������� ����� �������� � ������� caCheckError ��� �������� ��
	���������� ����������
	*/

	extern "C" DLL caError caBoxBlur(const TImage& inputImage, TImage& resultImage, caUInt16 numOfIterations = 5);

///�������� ���������� ����

/**
	\brief �������, ����������� ������ �������� ����.

	����� ����������� ������� ������� ��������� ��������� RGB ����������� � grayscale. � ���������
	������ ��������� ������� ����� ������������.

	\param [in] filterType ��� ������� (NFT_SALT - ������ ������ ����; NFT_PEPPER - ������ ������ ����;
	NFT_SALT_AND_PEPPER - ���������������� ���������� ���� ���������� ��������; NFT_MEDIAN - ��������� ������)

	\param [in] inputImage ������� �����������
	\param [out] outputImage ��������� ������ �������
	\param [in] numOfIterations ���������� �������� ������� �������, ����������� � �����������

	\return ���������� ��� ������, ������� ����� �������� � ������� caCheckError ��� �������� ��
	���������� ����������
	*/

	extern "C" DLL caError caNoiseFilter(caUInt8 filterType, const TImage& inputImage, TImage& resultImage, caUInt8 numOfIterations);

#endif
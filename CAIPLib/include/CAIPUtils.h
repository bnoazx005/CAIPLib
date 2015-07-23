/**
	\file
	\authors ������� ������
	\date 14.07.2015

	\brief ������������ ���� CAIPUtils.h �������� ���������� ��������������� ��������.

	������ ���� �������� �������������� � �������� ���������� ��������� ���������������
	�������, ������� ������������� ������������ ��� ������ � ������ �����������.
*/

#ifndef CAIP_UTILS_H
#define CAIP_UTILS_H


#include <cuda_runtime.h>
#include "..\include\CAIPTypes.h"


/**
	\brief �������, ��������� ��������� TGA ���� � ����� 

	\param [in] name ��� �����
	\param [out] resultBuffer ����� � ������� �����������
	\param [out] resultHeader ������ ��������� TGA �����

	\return ���������� ��� ������, ������� ����� �������� � ������� caCheckError ��� �������� ��
	���������� ����������.
*/

extern "C" DLL caError caLoadTGAFile(const caChar8* name, TColor** resultBuffer, TTGAHeader* resultHeader);

/**
	\brief �������, ��������� ��������� ����� � TGA ����

	\param [in] name ��� �����
	\param [in] resultBuffer ����� � ������� �����������
	\param [in] width ������ �����������
	\param [in] height ������ �����������
	\param [in] bpp ���������� ��� �� �������

	\return ���������� ��� ������, ������� ����� �������� � ������� caCheckError ��� �������� ��
	���������� ����������. 
*/

extern "C" DLL caError caSaveTGAFile(const caChar8* name, const TColor* resultBuffer, caInt16 width, caInt16 height, caInt8 bpp);

/**
	\brief �������, ��������������� ����� ������ ��� CUDA ����������

	\param [in] currImage �����������
	\param [out] gridDim ���������� ����� ������
	\param [out] threadsDim ���������� ������ ����� (���������� ������� �� ���� X � Y ��� ������ �����)

	\return ���������� ��� ������, ������� ����� �������� � ������� caCheckError ��� �������� ��
	���������� ����������. � ������ ���� �� ������� ��������� ������������, ���������� ��� ������
	ET_INVALID_IMAGE_SIZES
*/

extern "C" DLL caError caConfigGridForImage(const TImage& currImage, dim3& gridDim, dim3& threadsDim);

/**
	\brief �������, ������������ ���������� �� ���� ��������

	������� �� ���������� �������� ����������, �������� �� �������������� �����������

	\f[
	max(a, b) = \frac{(a + b + \left|a - b \right|)}{2}
	\f]

	\param [in] a ������ ��������
	\param [in] b ������ ��������

	\return ���������� ���������� �� ���� ����������
*/


template<class T> DLL __device__ __host__  T Max(T a, T b)
{
	return (a + b + abs(a - b)) / 2;
}

/**
	\brief �������, ������������ ���������� �� ���� ��������

	������� �� ���������� �������� ����������, �������� �� �������������� �����������

	\f[
	min(a, b) = \frac{(a + b - \left|a - b \right|)}{2}
	\f]

	\param [in] a ������ ��������
	\param [in] b ������ ��������

	\return ���������� ���������� �� ���� ����������
	*/

template<class T> DLL  __device__ __host__ T Min(T a, T b)
{
	return (a + b - abs(a - b)) / 2;
}


/**
	\brief �������, ���������� ���� ������� � ��������� �����������.

	������� ������������� ��� ���������� �� ����������� ����������.

	\param [in] inputImage �����������
	\param [in] x ���������� �� ��� �������
	\param [in] y ���������� �� ��� �������
	\param [in] imageWidth ������ �����������
	\param [in] imageHeight ������ �����������

	\return ���������� ���� ������� � ��������� �����������. � ������, ���� x ��� y
	������� �� ���������� �������, ������� ������ ������ ����.
	*/

DLL __device__ __host__ TColor GetColor(TColor* inputImage, int x, int y, int imageWidth, int imageHeight);

/**
	\brief �������, ���������� ���� ������� � ��������� �����������.

	������� ������������� ��� ���������� �� ����������� ����������. � ������� �� GetColor,
	����� ���������� ������ � ���������� ������� ������� � ���������� ������������.

	\param [in] inputImage �����������
	\param [in] x ���������� �� ��� �������
	\param [in] y ���������� �� ��� �������
	\param [in] imageWidth ������ �����������
	\param [in] imageHeight ������ �����������
	\param [out] index ������ ������� � ���������� ������������ � ���������� ������� inputImage

	\return ���������� ���� ������� � ��������� �����������. � ������, ���� x ��� y
	������� �� ���������� �������, ������� ������ ������ ����.
	*/

DLL __device__ __host__ TColor GetColorAndIndex(TColor* inputImage, int x, int y, int imageWidth, int imageHeight, unsigned int& index);


#endif
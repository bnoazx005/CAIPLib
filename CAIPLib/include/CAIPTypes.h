/**
\file
\authors ������� ������
\date 17.07.2015

\brief ������������ ���� CAIPTypes.h �������� ���������� ����� ������, ������������ �����������.
*/

#ifndef CAIP_TYPES_H
#define CAIP_TYPES_H


#define DLL __declspec(dllexport)


typedef unsigned char       caUInt8;  ///< 1 ����, ����� ����������� 
typedef unsigned short      caUInt16; ///< 2 �����, ����� �����������
typedef unsigned int        caUInt32; ///< 4 �����, ����� �����������
typedef unsigned long long  caUInt64; ///< 8 ���� (x64), ����� �����������

typedef char                 caInt8; ///< 1 ����, ����� �� ������
typedef short               caInt16; ///< 2 �����, ����� �� ������
typedef int                 caInt32; ///< 4 �����, ����� �� ������
typedef long long           caInt64; ///< 8 ���� (x64), ����� �� ������

typedef char                 caChar8; ///< 1 ����, ������
typedef unsigned char       caUChar8; ///< 1 ����, ����������� ������

typedef float               caFloat32; ///< ����� � ��������� ������, ��������� ��������
typedef double              caFloat64; ///< ����� � ��������� ������, ������� ��������

/// ������������ ����� ������, ������� ����� ��������� � �������� ������ �������

enum E_ERROR_TYPE
{
	ET_SUCCESS = 0x0,                      ///< �������� ���������� ������
	ET_INVALID_ARGUMENT = 0x1,             ///< ������������ �������� ������ ��� ����� ����������
	ET_BAD_ALLOCATION = 0x2,               ///< ������ ��������� ������
	ET_INVALID_FILE_OPERATION = 0x4,       ///< ������, ��������� ��� ���������� �������� �������� (������, ������, ��������, �������� � �.�.)
	ET_INVALID_TGA_TYPE = 0x8,             ///< ��� ������������ TGA ����� �� ������������� ����������� ����������
	ET_FAIL = 0x10,                        ///< ����������� ������, ����������� ������ ����������
	ET_INVALID_IMAGE_SIZES = 0x20,         ///< ������������ ������� ����������� (�� ������� ���������������� ����� ������ ��� CUDA �����������)
	ET_MEMORY_LEAK_DETECTED = 0x40,        ///< ������������� ������ ������. ��������, ���-�� ��������� ������ ������������ ������
	ET_UNKNOWN_EDGE_DETECTOR_TYPE= 0x80,   ///< ����������� ��� ��������� ������
	ET_UNKNOWN_NOISE_FILTER_TYPE = 0x100,  ///< ����������� ��� ������� ����
	ET_UNKNOWN_ERROR                       ///< ���� ������
};

typedef E_ERROR_TYPE        caError; ///< ����������� ���, ������������ � ����������, ��� �������� ���� ������


/// ������������ ����� ���������� ������, ������������� � ����������

enum E_EDGE_DETECTOR_TYPE
{
	EDT_LUM_DIFF = 0x01,		///< ��� ���������, ����������� ������� ������� �������� ������� � ���������
	EDT_ADV_LUM_DIFF = 0x02,    ///< ��� ���������, ����������� ������� ������� ����� ������ �������, �������� �������� ������� �� �����������
	EDT_GRADIENT = 0x04,        ///< ��������, ����������� �������� ������� ��������� ��� ������ �����
	EDT_DEFAULT = 0x00,        
};

/// ������������ ����� �������� ����

enum E_NOISE_FILTER_TYPE
{
	NFT_PEPPER = 0x01,          ///< ��� �������, ��������� ����� � ������ ���
	NFT_SALT = 0x02,            ///< ��� �������, ��������� ����� ���
	NFT_SALT_AND_PEPPER = 0x03, ///< ��� �������, ��������� ��������� ����� � ����� �����
	NFT_MEDIAN = 0x04,          ///< ��������� ������
	NFT_DEFAULT = 0x00
};

#pragma pack(push, 1)

/**
	\brief ���������, ����������� ��������� TGA �����.

	�������������� ������ �������� TGA ����������� � 32-������ �������� ��������.
*/

typedef struct TTGAHeader
{
	caInt8 mIdLength;
	caInt8 mColorMapType;
	caInt8 mImageType;

	caInt16 mFirstEntryIndex;
	caInt16 mColorMapLength;
	caInt8 mColorMapEntrySize;

	caInt16 mXOrigin;
	caInt16 mYOrigin;
	caInt16 mWidth;
	caInt16 mHeight;
	caInt8 mBitsPerPixel;
	caInt8 mImageDescriptor;
} TTGAHeader;

/**
	\brief ���������, ���������� ���������� � ������� �����������. 
	
	� ���������� ��� �����������
	����� ������ RGBA � 8-� ������ �� ����������. ����� ���������� (A) �� ������������.

	��� grayscale � binary ����������� �������� ��������� ������ ����� ����� �����.
	������� ������ ��������� ���������� ������������� �������� ������ � TGA ������.

	��� caUint8 - ��������������� unsigned char.
*/

typedef struct TColor
{
	caUInt8 mB; ///< ����� ����������
	caUInt8 mG; ///< ������� ����������
	caUInt8 mR; ///< ������� ����������
	caUInt8 mA; ///< ����� ����������
} TColor;

/**
	\brief ���������, ���������� ���������� �� �����������

	�������������� ������ �������� TGA ����������� � 32-������ �������� ��������.
*/

typedef struct TImage
{
	TColor* mpColorBuffer; ///< ������ �������� �����������

	int mWidth; ///< ������ �����������
	int mHeight; ///< ������ �����������
} TImage;

#pragma pack(pop)


#endif
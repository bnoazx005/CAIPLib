/**
\file
\authors Касимов Ильдар
\date 17.07.2015

\brief Заголовочный файл CAIPTypes.h содержит объявления типов данных, используемых библиотекой.
*/

#ifndef CAIP_TYPES_H
#define CAIP_TYPES_H


#define DLL __declspec(dllexport)


typedef unsigned char       caUInt8;  ///< 1 байт, целое беззнаковое 
typedef unsigned short      caUInt16; ///< 2 байта, целое беззнаковое
typedef unsigned int        caUInt32; ///< 4 байта, целое беззнаковое
typedef unsigned long long  caUInt64; ///< 8 байт (x64), целое беззнаковое

typedef char                 caInt8; ///< 1 байт, целое со знаком
typedef short               caInt16; ///< 2 байта, целое со знаком
typedef int                 caInt32; ///< 4 байта, целое со знаком
typedef long long           caInt64; ///< 8 байт (x64), целое со знаком

typedef char                 caChar8; ///< 1 байт, символ
typedef unsigned char       caUChar8; ///< 1 байт, беззнаковый символ

typedef float               caFloat32; ///< число с плавающей точкой, одинарной точности
typedef double              caFloat64; ///< число с плавающей точкой, двойной точности

/// Перечисление кодов ошибок, которые могут возникать в процессе работы функций

enum E_ERROR_TYPE
{
	ET_SUCCESS = 0x0,                      ///< Успешное завершение работы
	ET_INVALID_ARGUMENT = 0x1,             ///< Некорректное значение одного или более аргументов
	ET_BAD_ALLOCATION = 0x2,               ///< Ошибка выделения памяти
	ET_INVALID_FILE_OPERATION = 0x4,       ///< Ошибка, возникшая при выполнения файловой операции (чтение, запись, создание, открытие и т.д.)
	ET_INVALID_TGA_TYPE = 0x8,             ///< Тип считываемого TGA файла не соответствует требованиям библиотеки
	ET_FAIL = 0x10,                        ///< Критическая ошибка, продолжение работы невозможно
	ET_INVALID_IMAGE_SIZES = 0x20,         ///< Некорректные размеры изображения (не удалось сконфигурировать сетку блоков для CUDA вычислителя)
	ET_MEMORY_LEAK_DETECTED = 0x40,        ///< Зафиксирована утечка памяти. Возможно, где-то произошла ошибка освобождения памяти
	ET_UNKNOWN_EDGE_DETECTOR_TYPE= 0x80,   ///< Неизвестный тип детектора границ
	ET_UNKNOWN_NOISE_FILTER_TYPE = 0x100,  ///< Неизвестный тип фильтра шума
	ET_UNKNOWN_ERROR                       ///< Иная ошибка
};

typedef E_ERROR_TYPE        caError; ///< Специальный тип, используемый в библиотеке, для указания кода ошибок


/// Перечисление типов детекторов границ, реализованных в библиотеке

enum E_EDGE_DETECTOR_TYPE
{
	EDT_LUM_DIFF = 0x01,		///< Тип детектора, учитывающий разницу яркости текущего пикселя с соседними
	EDT_ADV_LUM_DIFF = 0x02,    ///< Тип детектора, учитывающий разницу яркости между парами соседей, значение текущего пикселя не учитывается
	EDT_GRADIENT = 0x04,        ///< Детектор, вычисляющий величину вектора градиента для каждой точки
	EDT_DEFAULT = 0x00,        
};

/// Перечисление типов фильтров шума

enum E_NOISE_FILTER_TYPE
{
	NFT_PEPPER = 0x01,          ///< Тип фильтра, удаляющий серый и черный шум
	NFT_SALT = 0x02,            ///< Тип фильтра, удаляющий белый шум
	NFT_SALT_AND_PEPPER = 0x03, ///< Тип фильтра, удаляющий сочетание серых и белых шумов
	NFT_MEDIAN = 0x04,          ///< Медианный фильтр
	NFT_DEFAULT = 0x00
};

#pragma pack(push, 1)

/**
	\brief Структура, описывающая заголовок TGA файла.

	Поддежриваются только несжатые TGA изображения с 32-битным форматом пикселей.
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
	\brief Структура, содержащая информацию о пикселе изображения. 
	
	В библиотеке все изображения
	имеют формат RGBA с 8-ю битами на компоненту. Альфа компонента (A) не используется.

	Для grayscale и binary изображений значения компонент цветов равны между собой.
	Порядок членов структуры обусловлен особенностями хранения данных в TGA файлах.

	Тип caUint8 - переопределение unsigned char.
*/

typedef struct TColor
{
	caUInt8 mB; ///< Синяя компонента
	caUInt8 mG; ///< Зеленая компонента
	caUInt8 mR; ///< Красная компонента
	caUInt8 mA; ///< Альфа компонента
} TColor;

/**
	\brief Структура, содержащая информацию об изображении

	Поддежриваются только несжатые TGA изображения с 32-битным форматом пикселей.
*/

typedef struct TImage
{
	TColor* mpColorBuffer; ///< Массив пикселей изображения

	int mWidth; ///< Ширина изображения
	int mHeight; ///< Высота изображения
} TImage;

#pragma pack(pop)


#endif
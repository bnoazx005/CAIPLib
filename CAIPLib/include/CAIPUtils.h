/**
	\file
	\authors Касимов Ильдар
	\date 14.07.2015

	\brief Заголовочный файл CAIPUtils.h содержит объявления вспомогательных стредств.

	Данный файл является дополнительным и содержит объявления различных вспомогательных
	средств, которые рекомендуется использовать при работе с данной библиотекой.
*/

#ifndef CAIP_UTILS_H
#define CAIP_UTILS_H


#include <cuda_runtime.h>
#include "..\include\CAIPTypes.h"


/**
	\brief Функция, загружает указанный TGA файл в буфер 

	\param [in] name Имя файла
	\param [out] resultBuffer Буфер с данными изображения
	\param [out] resultHeader Данные заголовка TGA файла

	\return Возвращает код ошибки, который можно передать в функцию caCheckError для проверки на
	корректное выполнение.
*/

extern "C" DLL caError caLoadTGAFile(const caChar8* name, TColor** resultBuffer, TTGAHeader* resultHeader);

/**
	\brief Функция, сохраняет указанный буфер в TGA файл

	\param [in] name Имя файла
	\param [in] resultBuffer Буфер с данными изображения
	\param [in] width Ширина изображения
	\param [in] height Высота изображения
	\param [in] bpp Количество бит на пиксель

	\return Возвращает код ошибки, который можно передать в функцию caCheckError для проверки на
	корректное выполнение. 
*/

extern "C" DLL caError caSaveTGAFile(const caChar8* name, const TColor* resultBuffer, caInt16 width, caInt16 height, caInt8 bpp);

/**
	\brief Функция, конфигурирующая сетку блоков для CUDA вычислений

	\param [in] currImage Изображение
	\param [out] gridDim Разрешение сетки блоков
	\param [out] threadsDim Разрешение одного блока (количество потоков по осям X и Y для одного блока)

	\return Возвращает код ошибки, который можно передать в функцию caCheckError для проверки на
	корректное выполнение. В случае если не удалось подобрать конфигурацию, возвращает код ошибки
	ET_INVALID_IMAGE_SIZES
*/

extern "C" DLL caError caConfigGridForImage(const TImage& currImage, dim3& gridDim, dim3& threadsDim);

/**
	\brief Функция, возвращающая наибольшее из двух значений

	Функция не использует условных операторов, основана на арифметических вычислениях

	\f[
	max(a, b) = \frac{(a + b + \left|a - b \right|)}{2}
	\f]

	\param [in] a Первый аргумент
	\param [in] b Второй аргумент

	\return Возвращает наибольшее из двух аргументов
*/


template<class T> DLL __device__ __host__  T Max(T a, T b)
{
	return (a + b + abs(a - b)) / 2;
}

/**
	\brief Функция, возвращающая наименьшее из двух значений

	Функция не использует условных операторов, основана на арифметических вычислениях

	\f[
	min(a, b) = \frac{(a + b - \left|a - b \right|)}{2}
	\f]

	\param [in] a Первый аргумент
	\param [in] b Второй аргумент

	\return Возвращает наименьшее из двух аргументов
	*/

template<class T> DLL  __device__ __host__ T Min(T a, T b)
{
	return (a + b - abs(a - b)) / 2;
}


/**
	\brief Функция, возвращает цвет пикселя в указанных координатах.

	Функция предназначена для выполнения на графическом процессоре.

	\param [in] inputImage Изображение
	\param [in] x координата на оси абсцисс
	\param [in] y координата на оси ординат
	\param [in] imageWidth Ширина изображения
	\param [in] imageHeight Высота изображения

	\return Возвращает цвет пикселя в указанных координатах. В случае, если x или y
	выходят за допустимые пределы, функция вернет черный цвет.
	*/

DLL __device__ __host__ TColor GetColor(TColor* inputImage, int x, int y, int imageWidth, int imageHeight);

/**
	\brief Функция, возвращает цвет пикселя в указанных координатах.

	Функция предназначена для выполнения на графическом процессоре. В отличие от GetColor,
	также возвращает индекс в одномерном массиве пикселя с указанными координатами.

	\param [in] inputImage Изображение
	\param [in] x координата на оси абсцисс
	\param [in] y координата на оси ординат
	\param [in] imageWidth Ширина изображения
	\param [in] imageHeight Высота изображения
	\param [out] index Индекс пикселя с указанными координатами в одномерном массиве inputImage

	\return Возвращает цвет пикселя в указанных координатах. В случае, если x или y
	выходят за допустимые пределы, функция вернет черный цвет.
	*/

DLL __device__ __host__ TColor GetColorAndIndex(TColor* inputImage, int x, int y, int imageWidth, int imageHeight, unsigned int& index);


#endif
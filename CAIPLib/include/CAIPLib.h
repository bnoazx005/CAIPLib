/**
	\file
	\authors Касимов Ильдар
	\date 17.07.2015

	\brief Заголовочный файл CAIPLib.h содержит объявления основные структуры и функции обработки.

	Данный файл является основным заголовочным файлом данной библиотеки. Содержит все определения
	основных функций, структур данных, констант и т.п. Для использования данной библиотеки в проекте
	требуется подключить данный файл в него, также обязательным является наличие библиотеки CUDA.
*/

#ifndef CAIP_LIB_H
#define CAIP_LIB_H


#include <cuda_runtime.h>
#include "..\include\CAIPTypes.h"

#define SHOW_ERROR_DATA

#if defined(SHOW_ERROR_DATA)

	#include <iostream>

	/**
		\brief Функция возвращает имя ошибки по ее коду

		\param [in] errorType Тип возникшей ошибки

		\return Возвращает строку, содержащую имя ошибки
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
		\brief Функция проверяет код ошибки

		В случае, если данный код не является кодом успешного завершения, функция сигнализирует
		об этом через возвращаемое значение. Также выводит информацию об ошибке в консоль.

		\param [in] error Тип возникшей ошибки

		\return Возвращает true в случае возникновения ошибки, false в противном случае
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
		\brief Функция проверяет код ошибки CUDA функций

		В случае, если данный код не является кодом успешного завершения, функция сигнализирует
		об этом через возвращаемое значение. Также выводит информацию об ошибке в консоль.

		\param [in] error Тип возникшей ошибки

		\return Возвращает true в случае возникновения ошибки, false в противном случае
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
		\brief Функция проверяет код ошибки

		В случае, если данный код не является кодом успешного завершения, функция сигнализирует
		об этом через возвращаемое значение. 

		\param [in] error Тип возникшей ошибки

		\return Возвращает true в случае возникновения ошибки, false в противном случае
	*/

    extern "C" DLL  __forceinline__ bool caCheckError(caError error)
	{
		if (error == 0x0)
			return false;

		return true;
	}

	/**
		\brief Функция проверяет код ошибки CUDA функций

		В случае, если данный код не является кодом успешного завершения, функция сигнализирует
		об этом через возвращаемое значение.

		\param [in] error Тип возникшей ошибки

		\return Возвращает true в случае возникновения ошибки, false в противном случае
	*/

	extern "C" DLL __forceinline__ bool caCheckCudaError(cudaError_t error)
	{
		if(error == cudaSuccess)
			return false;

		return true;
	}

#endif


/**
	\brief Функция загружает TGA изображение в память с жесткого диска.

	Относится к пользовательской части библиотеки. Позволяет загружать
	несжатые TGA файлы с 32-х битными пикселями.

	\param [in] filename Имя файла, хранящего изображение
	\param [out] resultImage Ссылка на структуру TImage, хранящую информацию о прочитанном изображении

	\return Возвращает код ошибки, который можно передать в функцию caCheckError для проверки на
	корректное выполнение
	*/

	extern "C" DLL caError caLoadImage(const caChar8* filename, TImage& resultImage);

/**
	\brief Функция сохраняет TGA изображение на жесткого диска.

	Относится к пользовательской части библиотеки. Позволяет осуществлять
	сохранение несжатых TGA файлов с 32-х битными пикселями.

	\param [in] filename Имя файла, под которым будет сохранено изображение
	\param [in] image Ссылка на структуру TImage, хранящую информацию об изображении

	\return Возвращает код ошибки, который можно передать в функцию caCheckError для проверки на
	корректное выполнение
	*/

	extern "C" DLL caError caSaveImage(const caChar8* filename, const TImage& image);

/**
	\brief Функция освобождает память из-под изображения.

	Относится к пользовательской части библиотеки.

	\param [in, out] image Ссылка на структуру TImage, хранящую информацию об изображении

	\return Возвращает код ошибки, который можно передать в функцию caCheckError для проверки на
	корректное выполнение
	*/

	extern "C" DLL caError caFreeImage(TImage& image);

///Функции, предназначенные для обработки изображений

///Операции первода из одного цветового пространства в другое

/**
	\brief Функция, осуществляющая операцию преобразования из RGB пространства в
	grayscale.

	Функция использует метод среднего трех цветовых компонент RGB пикселя.

	\param [in] inputImage Входное изображение
	\param [out] outputImage Результат работы функции

	\return Возвращает код ошибки, который можно передать в функцию caCheckError для проверки на
	корректное выполнение
	*/

	extern "C" DLL caError caRgb2Gray(const TImage& inputImage, TImage& resultImage);

/**
	\brief Функция, осуществляющая операцию преобразования из RGB пространства в
	бинарное.

	Функция первоначально преобразует RGB цвет в grayscale, а затем отсекает промежуточные
	тона при помощи задаваемого порога. Все тона ниже значения порога переводятся в черный,
	выше - белый.

	\param [in] inputImage Входное изображение
	\param [out] outputImage Результат работы функции
	\param [in] threshold Порог бинаризации

	\return Возвращает код ошибки, который можно передать в функцию caCheckError для проверки на
	корректное выполнение
	*/

	extern "C" DLL caError caRgb2Bin(const TImage& inputImage, TImage& resultImage, caUInt8 threshold = 128);

///Операция выделения границ

/**
	\brief Функция, реализующая детектор границ объектов на изображении.

	Перед применением данного фильтра требуется перевести RGB изображение в grayscale. В противном
	случае поведение функции будет неопределено.

	\param [in] detectorType тип детектора (EDT_LUM_DIFF - по разнице яркостей; EDT_ADV_LUM_DIFF - доработанный метод разницы яркостей;
	EDT_GRADIENT - градиент)

	\param [in] inputImage Входное изображение
	\param [out] outputImage Результат работы функции
	\param [in] threshold Погрешность

	\return Возвращает код ошибки, который можно передать в функцию caCheckError для проверки на
	корректное выполнение
	*/

	extern "C" DLL caError caEdges(caUInt8 detectorType, const TImage& inputImage, TImage& resultImage, caUInt8 threshold = 22);

///Операция размытия

/**
	\brief Функция, реализующая фильтр равномерного размытия.

	Перед применением данного фильтра требуется перевести RGB изображение в grayscale. В противном
	случае поведение функции будет неопределено.

	\param [in] inputImage Входное изображение
	\param [out] outputImage Результат работы функции
	\param [in] numOfIterations Количество итераций данного фильтра, примененных к изображению

	\return Возвращает код ошибки, который можно передать в функцию caCheckError для проверки на
	корректное выполнение
	*/

	extern "C" DLL caError caBoxBlur(const TImage& inputImage, TImage& resultImage, caUInt16 numOfIterations = 5);

///Операция фильтрации шума

/**
	\brief Функция, реализующая фильтр удаления шума.

	Перед применением данного фильтра требуется перевести RGB изображение в grayscale. В противном
	случае поведение функции будет неопределено.

	\param [in] filterType тип фильтра (NFT_SALT - фильтр белого шума; NFT_PEPPER - фильтр серого шума;
	NFT_SALT_AND_PEPPER - последовательное применение двух предыдущих фильтров; NFT_MEDIAN - медианный фильтр)

	\param [in] inputImage Входное изображение
	\param [out] outputImage Результат работы функции
	\param [in] numOfIterations Количество итераций данного фильтра, примененных к изображению

	\return Возвращает код ошибки, который можно передать в функцию caCheckError для проверки на
	корректное выполнение
	*/

	extern "C" DLL caError caNoiseFilter(caUInt8 filterType, const TImage& inputImage, TImage& resultImage, caUInt8 numOfIterations);

#endif
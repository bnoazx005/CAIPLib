/**
\file
\authors Касимов Ильдар
\date 14.07.2015

\brief Файл CAIPUtils.cu содержит определения дополнительных функций.

функции, расположенные в данном файле, используются для осуществления
различных вспомогательных действий. Все функции расположеные здесь
предназначены для выполнения на устройстве (графический процессор).

*/

#include "..\include\CAIPUtils.h"

__device__ __host__ TColor GetColor(TColor* inputImage, int x, int y, int imageWidth, int imageHeight)
{
	int logicConditionX = Max<int>(0, Min<int>(x + 1, imageWidth - x));
	int borderConditionX = Min<int>(Max<int>(logicConditionX, 0), 1);

	int logicConditionY = Max<int>(0, Min<int>(y + 1, imageHeight - y));
	int borderConditionY = Min<int>(Max<int>(logicConditionY, 0), 1);

	int borderResult = Min(borderConditionX, borderConditionY);

	return inputImage[borderResult * (x + 1 + y * imageWidth)];
}

__device__ __host__ TColor GetColorAndIndex(TColor* inputImage, int x, int y, int imageWidth, int imageHeight, unsigned int& index)
{
	int logicConditionX = Max<int>(0, Min<int>(x + 1, imageWidth - x));
	int borderConditionX = Min<int>(Max<int>(logicConditionX, 0), 1);

	int logicConditionY = Max<int>(0, Min<int>(y + 1, imageHeight - y));
	int borderConditionY = Min<int>(Max<int>(logicConditionY, 0), 1);

	int borderResult = Min<int>(borderConditionX, borderConditionY);

	unsigned int tmpIndex = borderResult * (x + 1 + y * imageWidth);
	index = borderResult * (tmpIndex - 1);

	return inputImage[tmpIndex];
}

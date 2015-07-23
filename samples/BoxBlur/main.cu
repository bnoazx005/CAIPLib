#include <iostream>
#include <cuda_runtime.h>
#include "CAIPLib.h"

#pragma comment(lib, "CAIPLib.lib")


using namespace std;


int main(int argc, char** argv)
{
	caChar8* pFilename = NULL;
	caUInt16 numberOfIterations = 0;

	TImage image;

	if (argc != 5)
	{
		cout << "Incorrect count of arguments\n"
			<< "Please, check out a format of input data.\n"
			<< "It should be -f filename -n numbers_of_iterations\n";

		return -1;
	}
	else if (argc == 5)
	{
		caUInt8 i = 1;

		while (i < argc)
		{
			if (strcmp(argv[i], "-f") == 0)
			{
				i++;

				pFilename = argv[i];
			}
			else if (strcmp(argv[i], "-n") == 0)
			{
				i++;

				numberOfIterations = atoi(argv[i]);
			}

			i++;
		}
	}

	if (caCheckError(caLoadImage(pFilename, image)))
		return -1;

	if (caCheckError(caRgb2Gray(image, image)))
		return -1;

	if (caCheckError(caBoxBlur(image, image, numberOfIterations)))
		return -1;
	
	if (caCheckError(caSaveImage("result.tga\0", image)))
		return -1;

	if (caCheckError(caFreeImage(image)))
		return -1;

	system("pause");
}
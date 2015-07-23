#include <iostream>
#include <cuda_runtime.h>
#include "CAIPLib.h"

#pragma comment(lib, "CAIPLib.lib")


using namespace std;


int main(int argc, char** argv)
{
	caChar8* pFilename = NULL;
	E_NOISE_FILTER_TYPE filterType = NFT_DEFAULT;
	caUInt8 numberOfIterations = 0;

	TImage image;

	if (argc != 7)
	{
		cout << "Incorrect count of arguments\n"
			<< "Please, check out a format of input data.\n"
			<< "It should be -f filename -t type_of_filter -n number_of_iterations\n";

		return -1;
	}
	else if (argc == 7)
	{
		caUInt8 i = 1;

		while (i < argc)
		{
			if (strcmp(argv[i], "-f") == 0)
			{
				i++;

				pFilename = argv[i];
			}
			else if (strcmp(argv[i], "-t") == 0)
			{
				i++;
				
				if (strcmp(argv[i], "median") == 0)
					filterType = NFT_MEDIAN;
				else if (strcmp(argv[i], "salt") == 0)
					filterType = NFT_SALT;
				else if (strcmp(argv[i], "pepper") == 0)
					filterType = NFT_PEPPER;
				else
				{
					cout << "Unknown type of filter\n";
					return -1;
				}
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

	if (caCheckError(caNoiseFilter(filterType, image, image, numberOfIterations)))
		return -1;

	if (caCheckError(caSaveImage("result.tga\0", image)))
		return -1;

	if (caCheckError(caFreeImage(image)))
		return -1;

	system("pause");
}
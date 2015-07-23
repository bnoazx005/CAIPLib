#include <iostream>
#include <cuda_runtime.h>
#include "CAIPLib.h"

#pragma comment(lib, "CAIPLib.lib")


using namespace std;


int main(int argc, char** argv)
{
	caChar8* pFilename = NULL;
	E_EDGE_DETECTOR_TYPE detectorType = EDT_DEFAULT;
	caUInt8 threshold = 22;

	TImage image;

	if (argc != 7)
	{
		cout << "Incorrect count of arguments\n"
			<< "Please, check out a format of input data.\n"
			<< "It should be -f filename -t type_of_detector -q threshold_value\n";

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

				if (strcmp(argv[i], "lumdiff") == 0)
					detectorType = EDT_LUM_DIFF;
				else if (strcmp(argv[i], "adv_lumdiff") == 0)
					detectorType = EDT_ADV_LUM_DIFF;
				else if (strcmp(argv[i], "grad") == 0)
					detectorType = EDT_GRADIENT;
				else
				{
					cout << "Unknown type of detector\n";
					return -1;
				}
			}
			else if (strcmp(argv[i], "-q") == 0)
			{
				i++;

				threshold = atoi(argv[i]);
			}

			i++;
		}
	}

	if (caCheckError(caLoadImage(pFilename, image)))
		return -1;

	if (caCheckError(caRgb2Gray(image, image)))
		return -1;

	if (caCheckError(caEdges(detectorType, image, image, threshold)))
		return -1;

	if (caCheckError(caSaveImage("result.tga\0", image)))
		return -1;

	if (caCheckError(caFreeImage(image)))
		return -1;

	system("pause");
}
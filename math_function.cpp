#include "math_functions.h"
#include <math.h>

void calcAtanForMat(Mat *in) {
	for (int i = 0; i < in->rows; i++) {
		for (int j = 0; j < in->cols; j++) {
			// Retrieve a single value

			float value = in->at<float>(i, j);
			// Calculate the corresponding single direction, done by applying the arctangens function
			float result = atan(value);
			// Store in orientation matrix element
			in->at<float>(i, j) = result;
		}
	}
}

void calcCosForMat(Mat *in) {
	for (int i = 0; i < in->rows; i++) {
		for (int j = 0; j < in->cols; j++) {
			// Retrieve a single value

			float value = in->at<float>(i, j);
			// Calculate the corresponding single direction, done by applying the arctangens function
			float result = cosf(value);
			// Store in orientation matrix element
			in->at<float>(i, j) = result;
		}
	}
}
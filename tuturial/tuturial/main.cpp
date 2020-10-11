#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <math.h>
#include <ctime>

using namespace std;
using namespace cv;

// 1

double method_SSIM(const Mat& image1, const Mat& image2, int flag = 0)
{
	// SSIM код основан на изображении в оттенках серого
	Mat gray_image1;
	Mat gray_image2;

	cvtColor(image1, gray_image1, COLOR_BGR2GRAY);
	if (flag == 0) // нужно для сравнения оригинальной и серой фотографий, чтобы не переводить серую фотографию обратно в серую
		cvtColor(image2, gray_image2, COLOR_BGR2GRAY);
	else
		gray_image2 = image2.clone();

	double avg1 = 0;
	double avg2 = 0;
	double disp1 = 0;
	double disp2 = 0;
	double cov = 0;
	double c1 = 6.5025; // pow(((256 - 1) * 0,01), 2);
	double c2 = 58.5225; // pow(((256 - 1) * 0,03), 2);
	int N = gray_image1.rows * gray_image1.cols;

	for (int i = 0; i < gray_image1.rows; i++)
		for (int j = 0; j < gray_image1.cols; j++)
		{
			avg1 += gray_image1.at<uchar>(i, j); // используем uchar, так как полутоновое (серое) изображение имеет всего 256 значений, или 1 канал
			avg2 += gray_image2.at<uchar>(i, j);
		}

	avg1 /= N;
	avg2 /= N;

	for (int i = 0; i < gray_image1.rows; i++)
		for (int j = 0; j < gray_image1.cols; j++)
		{
			disp1 += pow(gray_image1.at<uchar>(i, j) - avg1, 2); // используем uchar, так как полутоновое (серое) изображение имеет всего 256 значений, или 1 канал
			disp2 += pow(gray_image2.at<uchar>(i, j) - avg2, 2);
			cov += abs((gray_image1.at<uchar>(i, j) - avg1) * (gray_image2.at<uchar>(i, j) - avg2));
		}

	disp1 /= N;
	disp2 /= N;
	cov /= N;

	double l = (2 * avg1 * avg2 + c1) / (pow(avg1, 2) + pow(avg2, 2) + c1); // яркость
	double q = (2 * cov + c2) / (disp1 + disp2 + c2); // объединение констраста и структуры

	double result = l * q;
	return result;
}

// 2

void Gray_Filter(const Mat& photo, Mat& gray_photo) // метод (iv)  Photoshop, GIMP
{
	gray_photo = photo.clone();
	for (int i = 0; i < photo.rows; i++)
	{
		for (int j = 0; j < photo.cols; j++)
		{
			float b = photo.at<Vec3b>(i, j)[0] * 0.11;
			float g = photo.at<Vec3b>(i, j)[1] * 0.59;
			float r = photo.at<Vec3b>(i, j)[2] * 0.3;

			/* это для метода (i) average
			float gray = (photo.at<Vec3b>(i, j)[0] + photo.at<Vec3b>(i, j)[1] + photo.at<Vec3b>(i, j)[2]) / 3;
			*/

			float gray = b + g + r;
			Vec3f vec = { gray, gray, gray };
			gray_photo.at<Vec3b>(i, j) = vec;
		}
	}
}

// 3

void BGR_to_HSV(const Mat& photo, Mat& new_photo)
{
	new_photo = photo.clone();
	for (int i = 0; i < photo.rows; i++)
	{
		for (int j = 0; j < photo.cols; j++)
		{
			double B = photo.at<Vec3b>(i, j)[0];
			double G = photo.at<Vec3b>(i, j)[1];
			double R = photo.at<Vec3b>(i, j)[2];

			double b = B / 255.0;
			double g = G / 255.0;
			double r = R / 255.0;
			double tmp = max(r, g);
			double Cmax = max(tmp, b);
			tmp = min(r, g);
			double Cmin = min(tmp, b);
			double delta = Cmax - Cmin;

			// Находим H, S, V
			double H, S, V;

			if (delta == 0.0)
				H = 0.0;
			else if (Cmax == r)
				H = 60.0 * (g - b) / delta;
			else if (Cmax == g)
				H = 120.0 + 60.0 * (b - r) / delta;
			else if (Cmax == b)
				H = 240.0 + 60.0 * (r - g) / delta;

			if (H < 0.0)
				H += 360.0;

			if (Cmax == 0.0)
				S = 0.0;
			else
				S = delta / Cmax;

			V = Cmax;

			// Преобразуем значения в целевой тип
			V *= 255;
			S *= 255;
			H /= 2; // для установки от 0 до 255

			new_photo.at<Vec3b>(i, j)[0] = H;
			new_photo.at<Vec3b>(i, j)[1] = S;
			new_photo.at<Vec3b>(i, j)[2] = V;
		}
	}
}

void HSV_to_BGR(const Mat& photo, Mat& new_photo)
{
	new_photo = photo.clone();
	for (int i = 0; i < photo.rows; i++)
	{
		for (int j = 0; j < photo.cols; j++)
		{
			double H = photo.at<Vec3b>(i, j)[0];
			double S = photo.at<Vec3b>(i, j)[1];
			double V = photo.at<Vec3b>(i, j)[2];

			// Теперь от значений H, S, V перейдём снова к BGR

			H *= 2;
			S /= 255.0;
			V /= 255.0;

			double C = V * S;
			double div = H / 60.0 / 2.0;
			double X = C * (1 - abs(H / 60.0 - div * 2 - 1));
			double m = V - C;

			double b, g, r;
			if (H < 60)
			{
				b = 0;
				g = X;
				r = C;
			}
			else if (H < 120)
			{
				b = 0;
				g = C;
				r = X;
			}
			else if (H < 180)
			{
				b = X;
				g = C;
				r = 0;
			}
			else if (H < 240)
			{
				b = C;
				g = X;
				r = 0;
			}
			else if (H < 300)
			{
				b = C;
				g = 0;
				r = X;
			}
			else if (H <= 360)
			{
				b = X;
				g = 0;
				r = C;
			}

			double B = (b + m) * 255;
			double G = (g + m) * 255;
			double R = (r + m) * 255;

			// Теперь, используя новые значения BGR, сохраним цвет в пикселе
			new_photo.at<Vec3b>(i, j)[0] = B;
			new_photo.at<Vec3b>(i, j)[1] = G;
			new_photo.at<Vec3b>(i, j)[2] = R;
		}
	}
}

int clamp(int value, int max, int min)
{
	if (value > max)
		return max;
	else if (value < min)
		return min;
	else
		return value;
}

void filter_brightness_rgb(Mat& photo, int k)
{
	for (int i = 0; i < photo.rows; i++)
	{
		for (int j = 0; j < photo.cols; j++)
		{
			int b = photo.at<Vec3b>(i, j)[0];
			int g = photo.at<Vec3b>(i, j)[1];
			int r = photo.at<Vec3b>(i, j)[2];

			photo.at<Vec3b>(i, j)[0] = clamp(b + k, 255, 0);
			photo.at<Vec3b>(i, j)[1] = clamp(g + k, 255, 0);
			photo.at<Vec3b>(i, j)[2] = clamp(r + k, 255, 0);
		}
	}
}

void filter_brightness_hsv(Mat& photo, int k)
{
	for (int i = 0; i < photo.rows; i++)
	{
		for (int j = 0; j < photo.cols; j++)
			photo.at<Vec3b>(i, j)[2] += k;
	}
}

int main()
{
	Mat kat = imread("Kat.jpg");
	Mat car = imread("Car.jpg");
	Mat boat = imread("Boat.jpg");

	// 1
	// метод SSIM

	double value_SSIM1 = method_SSIM(kat, car);
	double value_SSIM2 = method_SSIM(kat, kat);
	double value_SSIM3 = method_SSIM(kat, boat);
	cout << "method_SSIM(image_1, image_2) = " << value_SSIM1 << endl;
	cout << "method_SSIM(image_1, image_1) = " << value_SSIM2 << endl;
	cout << "method_SSIM(image_1, image_3) = " << value_SSIM3 << endl << endl;

	// 2
	// a) метод (iv)  Photoshop, GIMP
	Mat gray_kat1;
	clock_t seconds_1 = clock();
	Gray_Filter(kat, gray_kat1);
	clock_t seconds_2 = clock();
	imwrite("gray_kat1.jpg", gray_kat1);

	// b) opencv метод cvtColor()
	Mat gray_kat2;
	clock_t seconds_3 = clock();
	cvtColor(kat, gray_kat2, COLOR_BGR2GRAY);
	clock_t seconds_4 = clock();
	imwrite("gray_kat2.jpg", gray_kat2);

	// c) сравнение методов
	// по метрике
	double myGray_SSIM = method_SSIM(kat, gray_kat1, 1);
	double opencvGray_SSIM = method_SSIM(kat, gray_kat2, 1);
	cout << "method_SSIM(Original, my_gray_image) = " << myGray_SSIM << endl;
	cout << "method_SSIM(Original, opencv_gray_image) = " << opencvGray_SSIM << endl << endl;
	// по времени
	cout << "Time of my_gray_filter = " << (double)(seconds_2 - seconds_1) / CLOCKS_PER_SEC << " seconds" << endl;
	cout << "Time of opencv_gray_filter = " << (double)(seconds_4 - seconds_3) / CLOCKS_PER_SEC << " seconds" << endl << endl;


	// 3
	// a) по собственному методу
	Mat hsv_kat1;
	seconds_1 = clock();
	BGR_to_HSV(kat, hsv_kat1);
	seconds_2 = clock();
	imwrite("bgr2hsv_kat1.jpg", hsv_kat1);

	Mat new_kat1;
	seconds_3 = clock();
	HSV_to_BGR(hsv_kat1, new_kat1);
	seconds_4 = clock();
	imwrite("hsv2bgr_kat1.jpg", new_kat1);

	// b) по встроенному в opencv методу
	Mat hsv_kat2;
	clock_t seconds_5 = clock();
	cvtColor(kat, hsv_kat2, COLOR_BGR2HSV);
	clock_t seconds_6 = clock();
	imwrite("bgr2hsv_kat2.jpg", hsv_kat2);

	Mat new_kat2;
	clock_t seconds_7 = clock();
	cvtColor(hsv_kat2, new_kat2, COLOR_HSV2BGR);
	clock_t seconds_8 = clock();
	imwrite("hsv2bgr_kat2.jpg", new_kat2);

	// c)
	double my_convert_BGR2HSV_SSIM = method_SSIM(kat, hsv_kat1);
	double opencv_convert_BGR2HSV_SSIM = method_SSIM(kat, hsv_kat2);
	double my_convert_HSV2BGR_SSIM = method_SSIM(hsv_kat1, new_kat1);
	double opencv_convert_HSV2BGR_SSIM = method_SSIM(hsv_kat2, new_kat2);
	double bgr_original_with_my_bgr = method_SSIM(kat, new_kat1);
	double bgr_original_with_opencv_bgr = method_SSIM(kat, new_kat2);
	cout << "method_SSIM(Original, my_hsv_image) = " << my_convert_BGR2HSV_SSIM << endl;
	cout << "method_SSIM(Original, opencv_hsv_image) = " << opencv_convert_BGR2HSV_SSIM << endl << endl;
	cout << "Time of original -> my_hsv = " << (double)(seconds_2 - seconds_1) / CLOCKS_PER_SEC << " seconds" << endl;
	cout << "Time of original -> opencv_hsv = " << (double)(seconds_6 - seconds_5) / CLOCKS_PER_SEC << " seconds" << endl << endl;

	cout << "method_SSIM(HSV_image, my_bgr_image) = " << my_convert_HSV2BGR_SSIM << endl;
	cout << "method_SSIM(HSV_image, opencv_bgr_image) = " << opencv_convert_HSV2BGR_SSIM << endl;
	cout << "\t Comparision BGR-images after convertation hsv->bgr:" << endl;
	cout << "method_SSIM(bgr_original, my_bgr_image) = " << bgr_original_with_my_bgr << endl;
	cout << "method_SSIM(bgr_original, opencv_bgr_image) = " << bgr_original_with_opencv_bgr << endl << endl;

	cout << "Time of hsv -> my_bgr = " << (double)(seconds_4 - seconds_3) / CLOCKS_PER_SEC << " seconds" << endl;
	cout << "Time of hsv -> opencv_bgr = " << (double)(seconds_8 - seconds_7) / CLOCKS_PER_SEC << " seconds" << endl << endl;

	// d)
	int k = 100; // это коэффициент яркости (не имеет смысла делать меньше -255 или больше 255, так как при <= -255 будет чёрное изображение, а при >= 255 белое изображение)
	Mat bgr_bright_kat = kat.clone();
	seconds_1 = clock();
	filter_brightness_rgb(bgr_bright_kat, k);
	seconds_2 = clock();
	imwrite("bgr_bright_kat.jpg", bgr_bright_kat);

	Mat hsv_bright_kat = hsv_kat2.clone();
	seconds_3 = clock();
	filter_brightness_hsv(hsv_bright_kat, k);
	seconds_4 = clock();
	imwrite("hsv_bright_kat.jpg", hsv_bright_kat); // hsv изображение с изменённой яркостью

	// e)
	double bgr_bright_SSIM = method_SSIM(kat, bgr_bright_kat);
	double hsv_bright_SSIM = method_SSIM(hsv_kat2, hsv_bright_kat);
	cout << "method_SSIM(bgr_original, bgr_bright_image) = " << bgr_bright_SSIM << endl;
	cout << "method_SSIM(hsv_original, hsv_bright_image) = " << hsv_bright_SSIM << endl << endl;
	cout << "Time of bgr_original -> bgr_bright = " << (double)(seconds_2 - seconds_1) / CLOCKS_PER_SEC << " seconds" << endl;
	cout << "Time of hsv_original -> hsv_bright = " << (double)(seconds_4 - seconds_3) / CLOCKS_PER_SEC << " seconds" << endl << endl;

	system("pause");
	waitKey(0);
}
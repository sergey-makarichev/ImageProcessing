#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <ctime>
#include <random>

using namespace std;
using namespace cv;

int clamp(int value, int max, int min)
{
	value = value > max ? max : value;
	return value < min ? min : value;
}

void create_noise(Mat& noise, int n)
{
	for (int i = 0; i < noise.rows; i++)
	{
		for (int j = 0; j < noise.cols; j++)
		{
			noise.at<Vec3b>(i, j)[0] = 0;
			noise.at<Vec3b>(i, j)[1] = 0;
			noise.at<Vec3b>(i, j)[2] = 0;
		}
	}

	for (int i = 0; i < n; i++)
	{
		int a = rand() % noise.rows;
		int b = rand() % noise.cols;

		noise.at<Vec3b>(a, b)[0] = 100;
		noise.at<Vec3b>(a, b)[1] = 100;
		noise.at<Vec3b>(a, b)[2] = 100;
	}
}

Vec3b median_filter_one_pixel(const Mat& noise_photo, int x, int y, int radius)
{
	const unsigned int diametr = 2 * radius + 1;
	vector<int> v_blue(diametr * diametr);
	vector<int> v_green(diametr * diametr);
	vector<int> v_red(diametr * diametr);

	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			int idx = (i + radius) * diametr + j + radius; // индексы ячеек (пикселей)

			Vec3b color = noise_photo.at<Vec3b>(clamp(x + j, noise_photo.rows - 1, 0), clamp(y + i, noise_photo.cols - 1, 0)); // Заполняем по горизонтали

			v_blue[idx] = color[0];
			v_green[idx] = color[1];
			v_red[idx] = color[2];
		}
	}

	// Теперь сортируем значения вектора
	sort(v_blue.begin(), v_blue.end());
	sort(v_green.begin(), v_green.end());
	sort(v_red.begin(), v_red.end());

	int center_blue = v_blue[(diametr * diametr) / 2];
	int center_green = v_green[(diametr * diametr) / 2];
	int center_red = v_red[(diametr * diametr) / 2];
	Vec3i result = { center_blue, center_green, center_red };
	return result;
}

void median_filter(const Mat& noise_photo, Mat& new_photo, int radius)
{
	for (int i = 0; i < noise_photo.rows; i++)
	{
		for (int j = 0; j < noise_photo.cols; j++)
		{
			Vec3b vec = median_filter_one_pixel(noise_photo, i, j, radius);
			new_photo.at<Vec3b>(i, j) = vec;
		}
	}
}

vector<float> createGaussianVector(int radius, int sigma)
{
	const unsigned int diametr = 2 * radius + 1;
	float norm = 0; // коэффициент нормировки ядра
	vector<float> vec(diametr * diametr); // создаём ядро фильтра

	// рассчитываем ядро линейного фильтра
	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			int idx = (i + radius) * diametr + j + radius;
			vec[idx] = exp(-(i * i + j * j) / (sigma * sigma));
			norm += vec[idx];
		}
	}

	// нормируем ядро
	for (int i = 0; i < diametr * diametr; i++)
		vec[i] /= norm;
	return vec;
}

void gauss_filter(const Mat& noise_photo, Mat& new_photo, int radius, int sigma)
{
	const unsigned int diametr = 2 * radius + 1;
	vector<float> vec = createGaussianVector(radius, sigma);
	for (int x = 0; x < noise_photo.rows; x++)
	{
		for (int y = 0; y < noise_photo.cols; y++)
		{
			float b = 0;
			float g = 0;
			float r = 0;
			for (int i = -radius; i <= radius; i++)
			{
				for (int j = -radius; j <= radius; j++)
				{
					int idx = (i + radius) * diametr + j + radius;

					Vec3b color = noise_photo.at<Vec3b>(clamp(x + j, noise_photo.rows - 1, 0), clamp(y + i, noise_photo.cols - 1, 0));

					b += color[0] * vec[idx];
					g += color[1] * vec[idx];
					r += color[2] * vec[idx];
				}
			}
			Vec3i v = { clamp(b, 255, 0), clamp(g, 255, 0), clamp(r, 255, 0) };
			new_photo.at<Vec3b>(x, y) = v;
		}
	}
}


int main()
{
	// 1: Gauss-noise
	Mat source_photo = imread("Kat.jpg");

	Mat noise(source_photo.size(), source_photo.type()); // просто выделяем память под хранение всех пикселей нашего изображения(с пом-ю size узнаём кол-во пикселей, а с пом-ю type узнаём, что у нас 3 канала по 8 бит каждый в каждом пикселе)
	int n = noise.rows * noise.cols / 100; // будет шум в каждом 100-ом пикселе
	create_noise(noise, n);
	//randn(noise, 0.0, 30.0); // функция генерации шума Гаусса, куда передаём изображение для записи туда шума, мат. ожидание и дисперсию
	Mat noise_photo = source_photo + noise;

	imwrite("Source_photo.jpg", source_photo);
	imwrite("Noise.jpg", noise);
	imwrite("Noise_photo.jpg", noise_photo);

	// 2: Median filter
	Mat photo_after_median_filter(noise_photo.size(), noise_photo.type());
	clock_t seconds_1 = clock();
	median_filter(noise_photo, photo_after_median_filter, 1);
	clock_t seconds_2 = clock();

	imwrite("Photo_after_median_filter.jpg", photo_after_median_filter);

	// 2: Gauss filter
	Mat photo_after_gauss_filter(noise_photo.size(), noise_photo.type());
	clock_t seconds_3 = clock();
	gauss_filter(noise_photo, photo_after_gauss_filter, 1, 7); // чем больше сигма, тем меньше шума за счёт большего размытия фото
	clock_t seconds_4 = clock();

	imwrite("Photo_after_gauss_filter.jpg", photo_after_gauss_filter);

	// 3: openCV Median filter
	Mat Photo_after_openCVmedian_filter;
	clock_t seconds_5 = clock();
	medianBlur(noise_photo, Photo_after_openCVmedian_filter, 9); // void medianBlur(InputArray src, OutputArray dst, int ksize), ksize — линейный размер апертуры, он должен быть нечетным и больше 1, Будет происходить сглаживание изображения при увеличении размера ядра.
	clock_t seconds_6 = clock();
	imwrite("Photo_after_openCVmedian_filter.jpg", Photo_after_openCVmedian_filter);

	// 4: сравнение по времени 
	cout << "Time of my_median_filter = " << (double)(seconds_2 - seconds_1) / CLOCKS_PER_SEC << " seconds" << endl;
	cout << "Time of my_gauss_filter = " << (double)(seconds_4 - seconds_3) / CLOCKS_PER_SEC << " seconds" << endl;
	cout << "Time of opencv_median_filter = " << (double)(seconds_6 - seconds_5) / CLOCKS_PER_SEC << " seconds" << endl;

	system("pause");
	waitKey();
}
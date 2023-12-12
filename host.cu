#include <stdio.h> 
#include <stdint.h> 
#include <stdlib.h>

#define CHECK(call){\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
}

typedef struct {
     unsigned char red,green,blue;
} PPMPixel;

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;

#define CREATOR "RPFELGUEIRAS"
#define RGB_COMPONENT_COLOR 255

static PPMImage *readPPM(const char *filename)
{
    char buff[16];
    PPMImage *img;
    FILE *fp;
    int c, rgb_comp_color;
    //open PPM file for reading
    fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        exit(1);
    }

    //read image format
    if (!fgets(buff, sizeof(buff), fp)) {
        perror(filename);
        exit(1);
    }

    //check the image format
    if (buff[0] != 'P' || buff[1] != '6') {
         fprintf(stderr, "Invalid image format (must be 'P6')\n");
         exit(1);
    }

    //allocate memory form image
    img = (PPMImage *)malloc(sizeof(PPMImage));
    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //check for comments
    c = getc(fp);
    while (c == '#') {
    while (getc(fp) != '\n') ;
         c = getc(fp);
    }

    ungetc(c, fp);
    //read image size information
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
         fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
         exit(1);
    }

    //read rgb component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
         fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
         exit(1);
    }

    //check rgb component depth
    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }

    while (fgetc(fp) != '\n') ;
    //memory allocation for pixel data
    img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //read pixel data from file
    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
         fprintf(stderr, "Error loading image '%s'\n", filename);
         exit(1);
    }

    fclose(fp);
    return img;
}

void writePPM(const char *filename, PPMImage *img)
{
    FILE *fp;
    // open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
         fprintf(stderr, "Unable to open file '%s'\n", filename);
         exit(1);
    }

    // write the header file
    // image format
    fprintf(fp, "P6\n");

    //comments
    fprintf(fp, "# Created by %s\n",CREATOR);

    //image size
    fprintf(fp, "%d %d\n",img->x,img->y);

    // rgb component depth
    fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

    // pixel data
    fwrite(img->data, 3 * img->x, img->y, fp);
    fclose(fp);
}

uint8_t * ChangeRGBtoGrayScale(PPMImage *img, int numChannels){
    // create grayscale array 
    uint8_t * grayscale_pixels;
    
    if (img){
        // allocate memory 
        grayscale_pixels = (uint8_t *)malloc(img->x*img->y*sizeof(uint8_t));
        for (int i = 0; i < img->x*img->y; i ++){
            grayscale_pixels[i] = 0.299f*img->data[i].red + 0.587f*img->data[i].green + 0.114f*img->data[i].blue;
        }
    }
    return grayscale_pixels;
}

PPMImage * ChangeGrayScaletoRGB(uint8_t *pixels, int width, int height){
    // create PPMImage array 
    PPMImage * img;
    // allocate the memory 
    img = (PPMImage*)malloc(sizeof(PPMImage));
    if (pixels){
      // allocate the memory for data pixels
      img->x = width;
      img->y = height;
      img->data = (PPMPixel*)malloc(width*height*sizeof(PPMPixel));
      for (int i =0; i < width*height; i++){
          int red = pixels[i];
          int green = pixels[i];
          int blue = pixels[i];
          img->data[i].red = red;
          img->data[i].green = green;
          img->data[i].blue = blue;
      }
    }
    return img;
}

void writeGrayScale_Pnm(uint8_t * pixels, int width, int height, int numChannels, char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	if (numChannels == 1)
		fprintf(f, "P2\n");
	else if (numChannels == 3)
		fprintf(f, "P3\n");
	else
	{
		fclose(f);
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n", pixels[i]);

	fclose(f);
}

void printPixels(PPMImage *img){
  if (img){
    for (int i = 0; i < img->x*img->y; i++){
      printf("red: %d, green: %d, blue: %d\n", img->data[i].red, img->data[i].green, img->data[i].blue);
    }
  }
}


float computeError(uchar3 * a1, uchar3 * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
	{
		err += abs((int)a1[i].x - (int)a2[i].x);
		err += abs((int)a1[i].y - (int)a2[i].y);
		err += abs((int)a1[i].z - (int)a2[i].z);
	}
	err /= (n * 3);
	return err;
}

void printError(uchar3 * deviceResult, uchar3 * hostResult, int width, int height)
{
	float err = computeError(deviceResult, hostResult, width * height);
	printf("Error: %f\n", err);
}

void printDeviceInfo()
{
	cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);
    printf("****************************\n");

}

int main(int argc, char **argv){
    // process input arguments
    // Input arguments look like ./out.out (char*)img_name resize_width 

    if (argc != 3)
    {
        printf("The number of argument is invalid\n");
        return EXIT_FAILURE;
    }

    // printDeviceInfo();

    // read input image file
    PPMImage *original_image = readPPM(argv[1]);

    // change RGB to grayscale image 
    uint8_t* grayscale_pixels = ChangeRGBtoGrayScale(original_image, 3);
    
    char out_grayscale[] = "grayscale.pnm";
    // save grayscale image 
    writeGrayScale_Pnm(grayscale_pixels, original_image->x, original_image->y,1, out_grayscale);

    PPMImage * new_img = ChangeGrayScaletoRGB(grayscale_pixels, original_image->x, original_image->y);

    char out_rgb[] = "out_rgb.ppm";
    
    writePPM(out_rgb, new_img);


}

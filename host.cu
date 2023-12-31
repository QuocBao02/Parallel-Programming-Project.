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

// PPMImage * ChangeGrayScaletoRGB(uint8_t *pixels, int width, int height){
//     // create PPMImage array 
//     PPMImage * img;
//     // allocate the memory 
//     img = (PPMImage*)malloc(sizeof(PPMImage));
//     if (pixels){
//       // allocate the memory for data pixels
//       img->x = width;
//       img->y = height;
//       img->data = (PPMPixel*)malloc(width*height*sizeof(PPMPixel));
//       for (int i =0; i < width*height; i++){
//           int red = pixels[i];
//           int green = pixels[i];
//           int blue = pixels[i];
//           img->data[i].red = red;
//           img->data[i].green = green;
//           img->data[i].blue = blue;
//       }
//     }
//     return img;
// }

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



int* ComputeImportanceMap(uint8_t * grayscalepixels, int width, int height){
    int x_sobel[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    int y_sobel[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
    
    int* importancemap = (int*)malloc((width)*(height)*sizeof(int));
    // detect edges in the x direction
  
    // detect edges in the y direction 
    for (int row = 0; row < height; row++){
        for(int col = 0; col < width; col ++){
          int Gx =0, Gy = 0;
          Gx = grayscalepixels[row*width + col]*x_sobel[0][0]+grayscalepixels[row*width + col + 1]*x_sobel[0][1]+grayscalepixels[row*width + col + 2]*x_sobel[0][2]+\
                    grayscalepixels[(row+1)*width + col]*x_sobel[1][0]+grayscalepixels[(row+1)*width + col + 1]*x_sobel[1][1]+grayscalepixels[(row+1)*width + col + 2]*x_sobel[1][2]+\
                    grayscalepixels[(row+2)*width + col]*x_sobel[2][0]+grayscalepixels[(row+2)*width + col + 1]*x_sobel[2][1]+grayscalepixels[(row+2)*width + col + 2]*x_sobel[2][2];

          Gy = grayscalepixels[row*width + col]*y_sobel[0][0]+grayscalepixels[row*width + col + 1]*y_sobel[0][1]+grayscalepixels[row*width + col + 2]*y_sobel[0][2]+\
                    grayscalepixels[(row+1)*width + col]*y_sobel[1][0]+grayscalepixels[(row+1)*width + col + 1]*y_sobel[1][1]+grayscalepixels[(row+1)*width + col + 2]*y_sobel[1][2]+\
                    grayscalepixels[(row+2)*width + col]*y_sobel[2][0]+grayscalepixels[(row+2)*width + col + 1]*y_sobel[2][1]+grayscalepixels[(row+2)*width + col + 2]*y_sobel[2][2];

          importancemap[row*width + col] = abs(Gx) + abs(Gy);
        }
    }
    return importancemap;
}

int findMin(int a, int b){
    if (a <= b)
        return a;
    else 
        return b;
}

int findMinIndex(int *arr, int start, int end){
    int min_index = start;
    int min_value = arr[start];

    for (int i = start + 1; i <= end; i++){
        if(arr[i] < min_value){
            min_value = arr[i];
            min_index = i;
        }
    }
    return min_index;
}

void FindSeam(int* importantmap, int width, int height, int seam[]){
    int **cumulative_energy = new int*[height];
    
    for(int i = 0; i < height; i++){
      cumulative_energy[i] = new int[width];
    }

    // copy the first row of the important map to the cumulative energy map
    for (int j=0; j < height; j++){
        cumulative_energy[0][j] = importantmap[j];
    }

    // Create a cumulative energy map 
    for (int i = 1; i < height; i++){
        for (int j =0; j< width; j++){

            if ( j - 1 < 0){
                cumulative_energy[i][j] = importantmap[i*width+j] +\
                findMin(cumulative_energy[i-1][j], cumulative_energy[i-1][j+1]);
            }
            else if ( j + 1 > width - 1){
                cumulative_energy[i][j] = importantmap[i*width+j] +\
                findMin(cumulative_energy[i-1][j -1], cumulative_energy[i-1][j]);
            }
            else{
                cumulative_energy[i][j] = importantmap[i*width+j] +\
            findMin(findMin(cumulative_energy[i-1][j -1], cumulative_energy[i-1][j]), cumulative_energy[i-1][j+1]);
            }
        }
    }

    // find minumum cumulative energy in the last row 
    int min_energy_index = findMinIndex(cumulative_energy[height - 1], 0, width -1);
    
    seam[height - 1] = min_energy_index;

    // back tracking the seam 
    for (int i = height - 2; i >= 0; i--){
        int min_index = findMinIndex(cumulative_energy[i], max(0, min_energy_index -1), min(width -1, min_energy_index +1));
        min_energy_index = min_index;
        seam[i] = min_energy_index;
    }
}


void SeamCarving(PPMImage *img, int width, int height, int re_width){
    int n = width - re_width;
    for (int i = 1; i <= n; i ++){
      // change rgb image to grayscale image 
      uint8_t* grayscale_pixels = ChangeRGBtoGrayScale(img, 3);

      // char grayscale[] = "grayscale.ppm";
      // writeGrayScale_Pnm(grayscale_pixels, width, height,1, grayscale);

      // find energy map 
      int * importance_map = ComputeImportanceMap(grayscale_pixels, width, height);

      // change energy_map into image
      // uint8_t* energy_map = (uint8_t *)malloc(width*height*sizeof(uint8_t));
      // for (int i =0; i < height; i++ ){
      //   for (int j =0; j < width; j++){
      //       energy_map[i*width + j] = importance_map[i*width + j];
      //   }
      // }
      // char energy[] = "energy.ppm";
      // writeGrayScale_Pnm(energy_map, width, height,1, energy);

      // find seam 
      int seam[1000];
      FindSeam(importance_map, width, height, seam);

      // create new image after removing seam
      PPMImage * new_img = (PPMImage*)malloc(sizeof(PPMImage));
      new_img->data = (PPMPixel*)malloc(height*(width-1)*sizeof(PPMPixel));
      new_img->x = width -1;
      new_img->y = height;

      // removal seam 
      for(int y = 0; y < height; y++){
        if (seam[y] == 0){
          for(int j = 0; j < width-1; j ++){
              new_img->data[y*(width - 1) + j].red = img->data[y*width + j+1].red;
              new_img->data[y*(width - 1) + j].green = img->data[y*width + j+1].green;
              new_img->data[y*(width - 1) + j].blue = img->data[y*width + j+1].blue;
          }
        }
        else if ( seam[y]>0 && seam[y] < width -1){
          for (int j = 0; j < seam[y]; j ++){
            new_img->data[y*(width -1) + j ].red = img->data[y*width + j].red;
            new_img->data[y*(width -1) + j ].green = img->data[y*width + j].green;
            new_img->data[y*(width -1) + j ].blue = img->data[y*width + j].blue;
          }
          for (int j = seam[y]; j < width -1; j ++){
            new_img->data[y*(width -1) + j].red = img->data[y*width + j + 1].red;
            new_img->data[y*(width -1) + j].green = img->data[y*width + j + 1].green;
            new_img->data[y*(width -1) + j].blue = img->data[y*width + j + 1].blue;
          }
        }
        else if ( seam[y] == width -1){
          for (int j= 0 ; j < width -1; j ++){
            new_img->data[y*(width -1) + j ].red = img->data[y*width + j].red;
            new_img->data[y*(width -1) + j ].green = img->data[y*width + j].green;
            new_img->data[y*(width -1) + j ].blue = img->data[y*width + j].blue;
          }
        }
      }

      // update original image size 
      width -=1;
      free(img);
      img = new_img;

    }
    // write image 
    char out_rgb[] = "out_host_rgb.ppm";
    writePPM(out_rgb, img);
}


void test_energy_map(){
    int width = 20;
    int height = 10;

    uint8_t * grayscale = (uint8_t *)malloc(width*height*sizeof(uint8_t));
    

    // asign the value in grayscale
    for(int i = 0; i < height; i ++){
        for(int j = 0; j < width; j++){
            grayscale[i*width+j] = rand()%10;
        }
    }

    // print grayscale 
    printf("grayscale matrix \n");
    for(int i = 0; i<height; i ++){
        for (int j = 0; j < width; j ++){
            printf("%d ", grayscale[i*width + j]);
        }
        printf("\n");
    }


    // compute importan map 

    int* important_map = ComputeImportanceMap(grayscale, width, height);

    // print important map 
    printf("Important Map\n");


    for(int i = 0; i<height; i ++){
        for (int j = 0; j < width; j ++){
            printf("%d ", important_map[i*width + j]);
        }
        printf("\n");
    }


    // Find seam 

    // int seam[100];

    // FindSeam(important_map, width, height, seam);
    int seam[] = {0, 0, 1, 1, 2, 2, 5, 7, 19, 19};
    printf("Print Seam \n");

    for(int tem = 0; tem < height; tem ++){
      printf("%d \n", seam[tem]);
    }

    int* new_img = (int*)malloc(height*(width-1)*sizeof(int));
    // test removal seam 
    for(int y = 0; y < height; y++){
      if (seam[y] == 0){
        for(int j = 0; j < width-1; j ++){
          new_img[y*(width - 1) + j] = grayscale[y*width + j+1];
        }
      }
      else if ( seam[y]>0 && seam[y] < width -1){
        for (int j = 0; j < seam[y]; j ++){
          new_img[y*(width -1) + j ] = grayscale[y*width + j];
        }
        for (int j = seam[y]; j < width -1; j ++){
          new_img[y*(width -1) + j] = grayscale[y*width + j + 1];
        }
      }
      else if ( seam[y] == width -1){
        for (int j= 0 ; j < width -1; j ++){
          new_img[y*(width -1) + j ] = grayscale[y*width + j];
        }
      }
    }

    width = width -1;
    // print grayscale 
    printf("Print grayscale image\n");

    for (int i = 0; i<height; i++ ){
        for(int j = 0; j < width; j++){
            printf("%d ", new_img[i*width + j]);
        }
        printf("\n");
    }
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

    SeamCarving(original_image, original_image->x, original_image->y, atoi(argv[2]));
    
    // test_energy_map();

    return 0;
    
}
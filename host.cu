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
struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);                                                                 
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

typedef struct {
     unsigned char red,green,blue;
} PPMPixel;

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;

#define CREATOR "QUOCBAO"
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

void writeGrayScale_Pnm(int * pixels, int width, int height, int numChannels, char * fileName)
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


float computeError(PPMPixel * a1, PPMPixel * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
	{
		err += abs((int)a1[i].red - (int)a2[i].red);
		err += abs((int)a1[i].green - (int)a2[i].green);
		err += abs((int)a1[i].blue - (int)a2[i].blue);
	}
	err /= (n * 3);
	return err;
}

// void printError(uchar3 * deviceResult, uchar3 * hostResult, int width, int height)
// {
// 	float err = computeError(deviceResult, hostResult, width * height);
// 	printf("Error: %f\n", err);
// }

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


PPMImage* SeamCarvingHost(PPMImage *img, int width, int height, int re_width){
    int n = width - re_width;

    // copy temp image to not change original image 
    PPMImage * temp_img = (PPMImage*)malloc(sizeof(PPMImage));
    temp_img->x = width;
    temp_img->y = height;
    temp_img->data = (PPMPixel*)malloc(width*height*sizeof(PPMPixel));
    for (int i = 0; i < height; i++){
      for(int j =0; j < width; j++){
        temp_img->data[i*width + j].red = img->data[i*width + j].red;
        temp_img->data[i*width + j].green = img->data[i*width + j].green;
        temp_img->data[i*width + j].blue = img->data[i*width + j].blue;
      }
    }

    // seam carving job
    for (int i = 1; i <= n; i ++){
      // change rgb image to grayscale image 
      uint8_t* grayscale_pixels = ChangeRGBtoGrayScale(temp_img, 3);

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
              new_img->data[y*(width - 1) + j].red = temp_img->data[y*width + j+1].red;
              new_img->data[y*(width - 1) + j].green = temp_img->data[y*width + j+1].green;
              new_img->data[y*(width - 1) + j].blue = temp_img->data[y*width + j+1].blue;
          }
        }
        else if ( seam[y]>0 && seam[y] < width -1){
          for (int j = 0; j < seam[y]; j ++){
            new_img->data[y*(width -1) + j ].red = temp_img->data[y*width + j].red;
            new_img->data[y*(width -1) + j ].green = temp_img->data[y*width + j].green;
            new_img->data[y*(width -1) + j ].blue = temp_img->data[y*width + j].blue;
          }
          for (int j = seam[y]; j < width -1; j ++){
            new_img->data[y*(width -1) + j].red = temp_img->data[y*width + j + 1].red;
            new_img->data[y*(width -1) + j].green = temp_img->data[y*width + j + 1].green;
            new_img->data[y*(width -1) + j].blue = temp_img->data[y*width + j + 1].blue;
          }
        }
        else if ( seam[y] == width -1){
          for (int j= 0 ; j < width -1; j ++){
            new_img->data[y*(width -1) + j ].red = temp_img->data[y*width + j].red;
            new_img->data[y*(width -1) + j ].green = temp_img->data[y*width + j].green;
            new_img->data[y*(width -1) + j ].blue = temp_img->data[y*width + j].blue;
          }
        }
      }

      // update original image size 
      width -=1;
      free(temp_img);
      temp_img = new_img;

    }
    // write image 
    char out_rgb[] = "out_host_rgb.ppm";
    writePPM(out_rgb, temp_img);

  return temp_img;
}

__global__ void ConvertRgb2Gray_Kernel(PPMPixel * pixels, int width, int height, uint8_t * grayPic) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < height && c < width) {
        int i = r * width + c;
        grayPic[i] = 0.299f*pixels[i].red + 0.587f*pixels[i].green + 0.114f*pixels[i].blue;
    }
    __syncthreads();
}

__global__ void ComputeImportanceMap_Kernel(uint8_t * grayscalepixels, int *energy, int width, int height){
    int x_sobel[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    int y_sobel[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    int col = threadIdx.x + blockIdx.x*blockDim.x;
    int row = threadIdx.y + blockIdx.y*blockDim.y;
  
    if ( col < width && row < height){    
      int Gx = grayscalepixels[row*width + col]*x_sobel[0][0]+grayscalepixels[row*width + col + 1]*x_sobel[0][1]+grayscalepixels[row*width + col + 2]*x_sobel[0][2]+\
                    grayscalepixels[(row+1)*width + col]*x_sobel[1][0]+grayscalepixels[(row+1)*width + col + 1]*x_sobel[1][1]+grayscalepixels[(row+1)*width + col + 2]*x_sobel[1][2]+\
                    grayscalepixels[(row+2)*width + col]*x_sobel[2][0]+grayscalepixels[(row+2)*width + col + 1]*x_sobel[2][1]+grayscalepixels[(row+2)*width + col + 2]*x_sobel[2][2];
      int Gy = grayscalepixels[row*width + col]*y_sobel[0][0]+grayscalepixels[row*width + col + 1]*y_sobel[0][1]+grayscalepixels[row*width + col + 2]*y_sobel[0][2]+\
                    grayscalepixels[(row+1)*width + col]*y_sobel[1][0]+grayscalepixels[(row+1)*width + col + 1]*y_sobel[1][1]+grayscalepixels[(row+1)*width + col + 2]*y_sobel[1][2]+\
                    grayscalepixels[(row+2)*width + col]*y_sobel[2][0]+grayscalepixels[(row+2)*width + col + 1]*y_sobel[2][1]+grayscalepixels[(row+2)*width + col + 2]*y_sobel[2][2];
      energy[row*width + col] = abs(Gx) + abs(Gy);
    }
    __syncthreads();
}

// __device__ int min3(int a, int b, int c) {
//     return min(a, min(b, c));
// }
__global__ void FindSeam_Kernel(int* importantmap, int width, int height, int* seam, int * cumulative_energy){

    // Create a cumulative energy map 
    // declare a device array to store cumulative values
    // int * cumulative_energy;
    // cudaMalloc(&cumulative_energy, width*height*sizeof(int));
    
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    
    for (int i = 0; i < width; i++){
      cumulative_energy[i] = importantmap[i];
    }
  
    for (int j = 1; j < height; j++){
      // initialize for cumulative map for the first row.
      int left = (col > 0)?cumulative_energy[(j-1)*width + col -1]:INT_MAX;
      int middle = cumulative_energy[(j-1)*width + col];
      int right = (col < width-1) ? cumulative_energy[(j-1)*width +col +1]:INT_MAX;
      cumulative_energy[j*width + col] = importantmap[j*width + col] + min(min(left, middle), right);
      // __syncthreads();
    }
    
  
    // __syncthreads();
    // Find the minimum energy seam in the last row
    if ( row == height - 1){
      int min_value = cumulative_energy[row*width];
      int min_index = 0;
      for(int i = 1; i < width; i++){
        if ( cumulative_energy[row*width+i] < min_value){
          min_value = cumulative_energy[row*width+i];
          min_index = i;
        }
      }
      seam[row] = min_index;

      for (int i = height - 2; i >= 0; --i) {
          int left = (min_index > 0) ? cumulative_energy[i * width + (min_index - 1)] : INT_MAX;
          int middle = cumulative_energy[i * width + min_index];
          int right = (min_index < width - 1) ? cumulative_energy[i * width + (min_index + 1)] : INT_MAX;

          // Determine the minimum energy path
          if (left <= middle && left <= right) {
              min_index = min_index - 1;
          } else if (right <= left && right <= middle) {
              min_index = min_index + 1;
          }
          // Update the seam array
          seam[i] = min_index;
      }

    }
    // __syncthreads();
    // Free allocated memory
    // if (row == height - 1) {
    //     cudaFree(cumulative_energy);
    // }
}

// CUDA kernel to remove a seam from the image
__global__ void removeSeam_Kernel(PPMPixel *inputImage, PPMPixel *outputImage, int *seam, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int seamCol = seam[row];
        // printf("Row: %d, Col: %d\n", row, col);
        if ( col < seamCol){
          outputImage[row*(width-1) + col] = inputImage[row*width+col];
        } 
        else{
          outputImage[row*(width-1)+col-1] = inputImage[row*width+col]; 
        }
    }
    // __syncthreads();
}
void SeamCarving_Kernel(PPMImage *in_host_img, PPMImage* out_host_img,int width, int height, int re_width){
    // create device image 
    PPMImage* d_in_img;
    PPMPixel* d_in_pixels;
    // Allocate d_in_img;
    cudaMalloc(&d_in_img, sizeof(PPMImage));
    cudaMalloc(&d_in_pixels, width*height*sizeof(PPMPixel));

    // copy data from host to device 
    cudaMemcpy(d_in_img, in_host_img, sizeof(PPMImage), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_pixels, in_host_img->data, width*height*sizeof(PPMPixel), cudaMemcpyHostToDevice);
  
    // do seam carving algorithm.
    int n = width - re_width;
    // n = 1;
    dim3 blockDim(32, 32);
    for (int times = 1; times <= n; times ++){
      dim3 gridDim((width + blockDim.x -1)/blockDim.x, (height+blockDim.y -1)/blockDim.y);

      // convert rgb image to grayscale 
      uint8_t * grayscale;
      cudaMalloc(&grayscale, width*height*sizeof(uint8_t));
      ConvertRgb2Gray_Kernel<<<gridDim, blockDim>>>(d_in_pixels, width, height, grayscale);

      // test grayscale 
      // uint8_t * testgrayscale = (uint8_t*)malloc(width*height*sizeof(uint8_t));
      // cudaMemcpy(testgrayscale, grayscale, sizeof(uint8_t)*width*height, cudaMemcpyDeviceToHost);
      // printf("Grayscale image\n");
      // for(int i = 0; i<height; i ++){
      //   for(int j = 0; j < width; j++){
      //     printf("%d ", testgrayscale[i*width + j]);
      //   }
      //   printf("\n");
      // }


      // // Compute energy
      int* energy;
      cudaMalloc(&energy, width*height*sizeof(int));
      ComputeImportanceMap_Kernel<<<gridDim, blockDim>>>(grayscale, energy, width, height);
      
      // // test energy 
      // int * testenergy = (int*)malloc(width*height*sizeof(int));
      // cudaMemcpy(testenergy, energy, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
      // printf("Energy map\n");
      // for(int i = 0; i<height; i ++){
      //   for(int j = 0; j < width; j++){
      //     printf("%d ", testenergy[i*width + j]);
      //   }
      //   printf("\n");
      // }


      // int* temp = (int *)malloc(width*height*sizeof(int));
      // cudaMemcpy(temp, energy, width*height*sizeof(int), cudaMemcpyDeviceToHost);
      // char out_rgb[] = "out_device_energy.ppm";
      // writeGrayScale_Pnm(temp, width, height, 1, out_rgb);

      // // Find min seam 
      int* seam;
      cudaMalloc(&seam, height*sizeof(int));

      
      int * cumulative_energy;
      cudaMalloc(&cumulative_energy, width*height*sizeof(int));

      FindSeam_Kernel<<<gridDim, blockDim>>>(energy, width, height, seam, cumulative_energy);
      
      // test cumulative matrix
      // int * testcumulative = (int*)malloc(width*height*sizeof(int));
      // cudaMemcpy(testcumulative, cumulative_energy, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
      // printf("Cumulative map\n");
      // for(int i = 0; i<height; i ++){
      //   for(int j = 0; j < width; j++){
      //     printf("%d ", testcumulative[i*width + j]);
      //   }
      //   printf("\n");
      // }

      // test seam 
      // int * resultseam = (int*)malloc(height*sizeof(int));
      // cudaMemcpy(resultseam, seam, sizeof(int)*height, cudaMemcpyDeviceToHost);
      // printf("Seam\n");
      // for (int i = 0; i < height; i++){
      //   printf("%d \n", resultseam[i]);
      // }
  
      // allocate device out pixels
      PPMPixel * d_out_pixels;
      cudaMalloc(&d_out_pixels, height*(width-1)*sizeof(PPMPixel));
      removeSeam_Kernel<<<gridDim, blockDim>>>(d_in_pixels, d_out_pixels, seam,width, height);
      
      cudaFree(d_in_pixels);
      d_in_pixels = d_out_pixels;
      // for(int i = 0; i< height; i++){
      //   for (int j = 0; j < width -1; j ++){
      //     printf("[%d %d %d] ",d_in_pixels[i*(width-1) + j].red, d_in_pixels[i*(width-1) + j].green, d_in_pixels[i*(width-1) + j].blue);
      //   }
      //   printf("\n");
      // }

      // // Update size of image
      width-=1;
      // d_in_img->x = width;
      // cudaFree(d_in_pixels);
      // cudaMalloc(&d_in_pixels, width*height*sizeof(PPMPixel));
      // cudaMemcpy(d_in_pixels, d_out_pixels, width*height*sizeof(PPMPixel), cudaMemcpyDeviceToDevice);
      
      // free memory
      // cudaFree(d_out_pixels);
      cudaFree(grayscale);
      cudaFree(energy);
      cudaFree(cumulative_energy);
      cudaFree(seam);
      
    }
    
    // write image 
    out_host_img->data = (PPMPixel*)malloc(width*height*sizeof(PPMPixel));
    cudaMemcpy(out_host_img->data, d_in_pixels,width*height*sizeof(PPMPixel), cudaMemcpyDeviceToHost);
    out_host_img->x = width;
    out_host_img->y = height;
    // return d_out_img;
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
    int width = original_image->x;
    int height = original_image->y;

    
    // temp_img = original_image;
    

    GpuTimer timer;
    timer.Start();
    PPMImage* host_img = SeamCarvingHost(original_image, original_image->x, original_image->y, atoi(argv[2]));
    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time use host: %f ms\n\n", time);
    // write image
    char out_rgb[] = "out_host_rgb.ppm";
    writePPM(out_rgb, host_img);



    // test data for find seam in kernel v1 function 
    PPMImage * temp_img = (PPMImage*)malloc(sizeof(PPMImage));
    temp_img->data = (PPMPixel*)malloc(width*height*sizeof(PPMPixel));

    temp_img->x = 5;
    temp_img->y = 5;
    temp_img->data = (PPMPixel*)malloc(temp_img->x*temp_img->y*sizeof(PPMPixel));

    for (int i = 0; i < temp_img->y; i ++){
      for(int j = 0; j < temp_img->x; j++){
        temp_img->data[i*temp_img->x + j].red = rand()%255 + 1;
        temp_img->data[i*temp_img->x + j].green = rand()%255 + 1;
        temp_img->data[i*temp_img->x + j].blue = rand()%255 + 1;
      }
    }

    // printf("Image\n");
    // for(int i = 0; i< temp_img->y; i++){
    //     for (int j = 0; j < temp_img->x; j ++){
    //       printf(" [%d %d %d] ",temp_img->data[i*(temp_img->x) + j].red,temp_img->data[i*(temp_img->x) + j].green,temp_img->data[i*(temp_img->x) + j].blue);
    //     }
    //     printf("\n");
    //   }

    // PPMImage * out_img = (PPMImage*)malloc(sizeof(PPMImage));
    // SeamCarving_Kernel_v1(temp_img, out_img, temp_img->x, temp_img->y, 300);


    // using kernel v1
    PPMImage* out_device_img = (PPMImage*)malloc(sizeof(PPMImage));
    GpuTimer timer_kernel;
    timer_kernel.Start();
    SeamCarving_Kernel(original_image, out_device_img, width, height, atoi(argv[2]));
    timer_kernel.Stop();
    float newtime_kernel = timer_kernel.Elapsed();
    printf("Processing time use device: %f ms\n\n", newtime_kernel);
    // write image
    char out_device_rgb[] = "out_device_rgb.ppm";
    writePPM(out_device_rgb, out_device_img);

    // compare error
    float err = computeError(host_img->data, out_device_img->data, host_img->x*host_img->y);
    printf("Compare error between host and device:\n");
    printf("Error: %f\n", err);
    

    // set blocksize 
    // dim3 blockSize(32, 32);


    // test_cuda_memory(original_image);
    // PPMImage * d_in_img;
    // CHECK(cudaMalloc(&d_in_img, sizeof(PPMImage)));
    // CHECK(cudaMalloc(&((d_in_img)->data), width*height*sizeof(PPMPixel)));


    // cudaMemcpy(d_in_img, original_image, sizeof(PPMImage), cudaMemcpyHostToDevice);
    // cudaMemcpy((d_in_img)->data, original_image->data, width*height*sizeof(PPMPixel), cudaMemcpyHostToDevice);


    // cudaMemcpy(temp_img, d_in_img, sizeof(PPMImage), cudaMemcpyDeviceToHost);
    // cudaMemcpy(temp_img->data, (d_in_img)->data, sizeof(PPMPixel)*width*height, cudaMemcpyDeviceToHost);




    
    // copy the non-pointer part of the struct to the device 

    // CHECK(cudaMemcpy())

  
    // test_energy_map();

    return 0;
    
}
#include <iostream>
#include <math.h>

#include <curand.h>
#include <curand_kernel.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define RANDSIZL   (4)  /* I recommend 8 for crypto, 4 for simulations */
#define RANDSIZ    (1<<RANDSIZL)

struct pixel {
  char r;
  char g;
  char b;
};

struct random_state
{
  unsigned int randcnt;
  unsigned int randrsl[4];
  unsigned int randmem[4];
  unsigned int randa;
  unsigned int randb;
  unsigned int randc;
};

struct particle {
  int x;
  int y;
  random_state random_state;
};

// Kernel function to add the elements of two arrays
__global__
void add(int n, pixel *pixels)
{
  const int index = threadIdx.x;
  const int stride = blockDim.x;
  for (int i = index; i < n; i+=stride)
    if (i < 512*512) {
      pixels[i].r = 0;
      pixels[i].g = 255;
      pixels[i].b = 0;
    }
}

__device__ int random_int(random_state& r) {
  // https://github.com/scottdraves/flam3/blob/master/isaac.h
  return 1;
}

__global__
void sheep(int n, particle *particles, unsigned int *histogram, int w, int h) {
  const int index = threadIdx.x;
  const int stride = blockDim.x;
  for (int i = index; i < n; i+=stride) {
    particle& p = particles[i];
    // update p, and then update the histogram
    p.x = (p.x + 2) % w;
    p.y = (p.y + 2) % h;

    // TODO: Race condition?
    histogram[p.y * w + p.x] += 1;
    // At each point simulate the next step of
    // the sheep
  }
}

__global__
void histogram_to_image(unsigned int* histogram, pixel* pixels, int w, int h) {
  const int index = threadIdx.x;
  const int stride = blockDim.x;
  for (int i = index; i < w*h; i+=stride) {
    pixels[i].r = histogram[i] > 0 ? 100 : 0;
    pixels[i].g = histogram[i] > 0 ? 100 : 0;
    pixels[i].b = histogram[i] > 0 ? 100 : 0;
  }
}

int main(void)
{
  pixel *pixels;

  const int num_sheep = 100;
  particle *particles;
  unsigned int* histogram;

  const int w = 7860;
  const int h = 4320;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&pixels, w*h*sizeof(pixel));

  cudaMallocManaged(&histogram, w*h*sizeof(unsigned int));

  cudaMallocManaged(&particles, num_sheep*sizeof(particle));

  // Run kernel on 1M elements on the GPU
  sheep<<<100, 1024>>>(num_sheep, particles, histogram, w, h);
  histogram_to_image<<<100, 1024>>>(histogram, pixels, w, h);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  stbi_write_png("test.png", w, h, 3, pixels, 0);

  // Free memory
  cudaFree(pixels);
  cudaFree(particles);
  cudaFree(histogram);

  return 0;
}

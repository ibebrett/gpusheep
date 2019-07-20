#include <cstring>
#include <iostream>

#include <curand.h>
#include <curand_kernel.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct pixel {
  char r;
  char g;
  char b;
};

struct particle {
  int x;
  int y;
};

__global__ void sheep(int n, int iterations, particle *particles,
                      unsigned int *histogram, int w, int h) {
  const int index = threadIdx.x;
  const int stride = blockDim.x;
  for (int i = index; i < n; i += stride) {
    for (int iter = 0; iter < iterations; ++iter) {
      particle &p = particles[i];
      // update p, and then update the histogram
      p.x = (p.x + 2 + i) % w;     //+ p.x * p.x + p.y) % w;
      p.y = (p.y + 2 + i * i) % h; //+ p.x * p.x * p.y) % h;

      // TODO: Does atomicAdd prevent race condition?
      atomicAdd(&histogram[p.y * w + p.x], 1);

      // At each point simulate the next step of
      // the sheep
    }
  }
}

__global__ void histogram_to_image(unsigned int *histogram, pixel *pixels,
                                   int w, int h) {
  const int index = threadIdx.x;
  const int stride = blockDim.x;
  for (int i = index; i < w * h; i += stride) {
    if (histogram[i] > 0) {
      pixels[i].r = 255;
      pixels[i].g = 255;
      pixels[i].b = 255;
    }
  }
}

int main(void) {
  const int num_particles = 100;
  const int w = 1024*2;
  const int h = 768*2;

  pixel *pixels;
  unsigned int *histogram;
  particle *particles;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&pixels, w * h * sizeof(pixel));
  cudaMallocManaged(&histogram, w * h * sizeof(unsigned int));
  cudaMallocManaged(&particles, num_particles * sizeof(particle));

  // clear out our particles
  cudaMemset(particles, 0, num_particles * sizeof(particle));

  // Run kernel on 1M elements on the GPU
  sheep<<<100, 1024>>>(num_particles, 1000, particles, histogram, w, h);
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

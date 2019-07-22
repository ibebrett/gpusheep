#include <cstring>
#include <iostream>

#include <curand.h>
#include <curand_kernel.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NUM_VARIATIONS 4
#define NUM_XFORMS 3

struct affine {
  float a;
  float b;
  float c;
  float d;
  float e;
  float f;
};

struct xform {
  float weights[NUM_VARIATIONS];
  float weight;
  affine pre_affine;
  affine post_affine;
  float r;
  float g;
  float b;
  float a;
};

struct pixel {
  char r;
  char g;
  char b;
};

struct particle {
  float x;
  float y;
};

struct hist {
  unsigned int count;
  float r;
  float g;
  float b;
  float a;
};

__device__ void var0_linear(const particle &in, particle *out, float weight) {
  out->x += in.x * weight;
  out->y += in.y * weight;
}

__device__ void var1_sinusoidal(const particle &in, particle *out,
                                float weight) {
  out->x += weight * sin(in.x);
  out->y += weight * sin(in.y);
}

__device__ void var2_spherical(const particle &in, particle *out,
                               float weight) {
  const float eps = 0.000001f;
  float r2 = weight / (in.x * in.x + in.y * in.y + eps);
  out->x += r2 * in.x;
  out->y += r2 * in.y;
}

__device__ void var3_swirl(const particle &in, particle *out, float weight) {
  float sumsq = (in.x * in.x + in.y * in.y);

  float c1 = sin(sumsq);
  float c2 = cos(sumsq);
  float nx = c1 * in.x - c2 * in.y;
  float ny = c2 * in.x + c1 * in.y;

  out->x += weight * nx;
  out->y += weight * ny;
}

/*
__device__ void var4_horseshoe(flam3_iter_helper *f, double weight) {
double r = weight / (f->precalc_sqrt + EPS);

f->p0 += (f->tx - f->ty) * (f->tx + f->ty) * r;
f->p1 += 2.0 * f->tx * f->ty * r;
}
*/

__device__ particle apply_affine(const particle &in, const affine &a) {
  return {in.x * a.a + in.y * a.b + a.c, in.x * a.d + in.y * a.e + a.f};
}

__global__ void sheep(int num_sheep, int iterations, hist *histogram, int w,
                      int h, xform *xforms) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;

  curandState random_state;
  curandState global_random_state;
  curand_init(index, 0, 0, &random_state);
  curand_init(0, 0, 0, &global_random_state);
  const int wait_time = 200;

  float total_weight = 0.0f;
  float weight_boundaries[NUM_XFORMS] = {0.};
  for (int i = 0; i < NUM_XFORMS; ++i) {
    total_weight += xforms[i].weight;
    weight_boundaries[i] = total_weight;
  }

  // Random sampling.
  particle state = {curand_uniform(&random_state) * 2.0 - 1.0,
                    curand_uniform(&random_state) * 2.0 - 1.0};
  state.x = 0;
  state.y = 0;

  // same starting point
  for (int iter = 0; iter < iterations && index < num_sheep; ++iter) {
    float xform_sample = curand_uniform(&random_state) * total_weight;

    int xform_index = 0;
    for (xform_index = 0; xform_index < NUM_XFORMS - 1; ++xform_index) {
      if (xform_sample < weight_boundaries[xform_index]) {
        break;
      }
    }

    const xform &xform = xforms[xform_index];

    particle input = apply_affine(state, xform.pre_affine);
    particle next = {0., 0.};

    var0_linear(input, &next, xform.weights[0]);
    var1_sinusoidal(input, &next, xform.weights[1]);
    var2_spherical(input, &next, xform.weights[2]);
    var3_swirl(input, &next, xform.weights[3]);

    next = apply_affine(next, xform.post_affine);

    state = next;

    // TODO: Does atomicAdd prevent race condition?
    if (iter > wait_time) {
      int ix = (int)((state.x + 1.0) * (w / 2));
      int iy = (int)((state.y + 1.0) * (h / 2));
      if (ix >= 0 && ix < w && iy >= 0 && iy < h) {
        atomicInc(&(histogram[iy * w + ix].count), 1);

        // race condition for color..., lets just try it
        histogram[iy * w + ix].r = (xform.r + histogram[iy * w + ix].r) / 2.0f;
        histogram[iy * w + ix].g = (xform.g + histogram[iy * w + ix].g) / 2.0f;
        histogram[iy * w + ix].b = (xform.b + histogram[iy * w + ix].b) / 2.0f;
        histogram[iy * w + ix].a = (xform.a + histogram[iy * w + ix].a) / 2.0f;
      }
    }
  }
}

__global__ void histogram_to_image(hist *histogram, pixel *pixels, int w,
                                   int h) {
  const int index = threadIdx.x;
  const int stride = blockDim.x;
  for (int i = index; i < w * h; i += stride) {
    pixels[i].r = 0;
    pixels[i].g = 0;
    pixels[i].b = 0;

    // we have an average
    // lets map to ( val - average) / avarege +
    if (histogram[i].count > 0) {
      const float p = powf(histogram[i].a, 1.0f / 2.2f);
      pixels[i].r = histogram[i].r * p * 255;
      pixels[i].g = histogram[i].g * p * 255;
      pixels[i].b = histogram[i].b * p * 255;
    }
  }
}

void create_xforms(xform *xforms) {
  // 180 degree
  xforms[0].weights[0] = 1.0;
  xforms[0].weights[1] = 0.0;
  xforms[0].weights[2] = 0.0;
  xforms[0].weights[3] = 0.0;
  xforms[0].weight = 33.0f;
  xforms[0].pre_affine = affine{0.5, 0.0, 0.0, 0.0, 0.5, 0.0};
  xforms[0].post_affine = affine{1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  xforms[0].r = 0.0f;
  xforms[0].g = 1.0f;
  xforms[0].b = 0.0f;
  xforms[0].a = 1.0f;

  xforms[1].weights[0] = 1.0;
  xforms[1].weights[1] = 0.0;
  xforms[1].weights[2] = 0.0;
  xforms[1].weights[3] = 0.0;
  xforms[1].weight = 33.0f;
  xforms[1].pre_affine = affine{0.5, -0.0, 0.5, 0.0, 0.5, 0.0};
  xforms[1].post_affine = affine{1.0, 0.0, 0.0, 0.0, 1.0, 0.0};

  xforms[1].r = 0.0f;
  xforms[1].g = 0.0f;
  xforms[1].b = 1.0f;
  xforms[1].a = 1.0f;

  xforms[2].weights[0] = 1.0;
  xforms[2].weights[1] = 0.0;
  xforms[2].weights[2] = 0.0;
  xforms[2].weights[3] = 0.0;
  xforms[2].weight = 33.0f;
  xforms[2].pre_affine = affine{0.5, 0.0, 0.0, 0.0, 0.5, 0.5};
  xforms[2].post_affine = affine{1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  xforms[2].r = 1.0f;
  xforms[2].g = 0.0f;
  xforms[2].b = 0.0f;
  xforms[2].a = 1.0f;
}

int main(void) {
  const int num_particles = 100000;
  const int w = 1024;
  const int h = 1024;
  const int num_threads = 1024;
  const int num_blocks = 100;

  xform *xforms;
  pixel *pixels;
  hist *histogram;

  // xforms
  cudaMallocManaged(&xforms, NUM_XFORMS * sizeof(xform));

  // Setup number of xforms initially.
  create_xforms(xforms);
  cudaDeviceSynchronize();

  // Allocate Unified Memory – accessible from CPU or GPU}
  cudaMallocManaged(&pixels, w * h * sizeof(pixel));
  cudaMallocManaged(&histogram, w * h * sizeof(hist));

  // Run kernel on 1M elements on the GPU
  sheep<<<num_blocks, num_threads>>>(num_particles, 1000000, histogram, w, h,
                                     xforms);

  histogram_to_image<<<num_blocks, num_threads>>>(histogram, pixels, w, h);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  stbi_write_png("test2.png", w, h, 3, pixels, 0);

  // Free memory
  cudaFree(xforms);
  cudaFree(pixels);
  cudaFree(histogram);

  std::cout << "finished..." << std::endl;

  return 0;
}

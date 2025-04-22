#include "cudaHandles.h"

// Define the static members
cudaHandles* cudaHandles::instancePtr = nullptr;
std::mutex cudaHandles::mtx;

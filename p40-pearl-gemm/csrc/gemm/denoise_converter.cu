#include "denoise_converter_host.h"
#include "pearl_api_params.h"

template void run_denoise_converter<128>(PearlAPIParams&, cudaStream_t);
template void run_denoise_converter<64>(PearlAPIParams&, cudaStream_t);
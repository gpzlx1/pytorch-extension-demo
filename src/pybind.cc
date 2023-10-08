#include <torch/script.h>
#include <torch/custom_class.h>

#include "operator.h"

TORCH_LIBRARY(test_ops, m)
{
    m.def("add", &AddCUDA);
}
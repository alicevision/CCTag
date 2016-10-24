#include <iostream>
#include <cuda_runtime.h>
#include "assist.h"
#include "device_prop.h"

using namespace std;

int main( )
{
    device_prop_t dev;
    dev.print();
}


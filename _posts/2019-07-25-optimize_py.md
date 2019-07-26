---
layout: post
title: "Speeding up the python code"
featured-img: bg
categories: [Python]
---

# Speeding up the python code

Python is one of the most popular programming languages. Most of the companies use python in production for rapid delivery. Although it is good when it comes to production. What about its performance? Python fails miserably. It is slower than most of the programming languages such as c++, java, etc. So how can we make it faster?? There are several ways of doing it. 
Let us discuss it one by one.

For demonstration purpose, I have created a script for finding out the Nearest neighbors for each data points in increasing order of euclidean distances.
Before moving to the optimization method we will first do profiling to see where it is taking a lot of time. For that we will use line profiler which is a great tool to provide the execution of each line inside the function line profiler is applied to.

First, let's import NumPy and the line_profiler 


<code id="gist-4672365" data-file="2.java" data-line="2-5,10-14,11,20"></code>

```python3
import numpy as np
%load_ext line_profiler
```

Using this %%writefile cell magic we write our code in a Python script.
```python3
%%writefile simulation.py
import numpy as np
def comp_inner_raw(i, x):
    res = np.zeros(x.shape[0], dtype=np.float64)
    for j in range(x.shape[0]):
        res[j] = np.sqrt(np.sum((i-x[j])**2))
    return res
  
def nearest_ngbr_raw(x):
    dist = {}
    for idx,i in enumerate(x):
        lst = comp_inner_raw(i,x)
        s = np.argsort(lst)
        sorted_array = np.array(x)[s][1:]
        dist[idx] = s[1:]
    return dist
```
```python3
arr = np.random.rand(1000,800)
```

Next, we will import our script so that we can execute and profile our code using %lprun magic command and will save the report in a file named sim_result.

```python3
from simulation import nearest_ngbr_raw
%lprun -T sim_result -f nearest_ngbr_raw nearest_ngbr_raw(arr)
```
To diplay the report execute the below syntax.
```python3
print(open('sim_result', 'r').read())
```

result: <RESULT>

We can see this function comp_inner_raw( ) is consuming a lot of time for computations.  So we need to optimize this function.

**Raw Python Computation time:**
```python3
%%time 
a = nearest_ngbr_raw(arr)
```

```
CPU times: user 10.6 s, sys: 1.77 s, total: 12.3 s
Wall time: 12.3 s
```

## Optimizing the code using numba:

The function comp_inner_raw() is using loop and we know loops in python are slow. This part is computationally slow and uses numpy in calculations so we can use Numba which is a just-in-time complier for Python that works best on the code which uses NumPy arrays, functions, and loops. Using Numba we can generate optimized machine code from our pure python code by using LLVM compiler infrastructure. This will speed up the code.

```python3
@jit('float64[:](float64[:], float64[:, :])')
def comp_inner_numba(i, x):
    res = np.zeros(x.shape[0], dtype=np.float64)
    for j in range(x.shape[0]):
        res[j] = np.sqrt(np.sum((i-x[j])**2))
    return res
```
```python3
def nearest_ngbr_numba(x):
    dist = {}
    for idx,i in enumerate(x):
        lst = comp_inner_numba(i,x)
        s = np.argsort(lst)
        sorted_array = np.array(x)[s][1:]
        dist[idx] = s[1:]
    return dist
```

```python3
%%time 
b = nearest_ngbr_numba(arr)
```

```
CPU times: user 3.46 s, sys: 1.75 s, total: 5.21 s
Wall time: 5.21 s
```

## Optimizing the code using Swig:
The Simplified Wrapper and Interface Generator (SWIG) provides capability to wrap c/c++ libraries with other languages such as Python, Ruby, Java etc.

In order to create wrapper we will first create our cpp file.

```cpp
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <map>
#include <cmath>
#include <math.h>

std::vector<int> argsort(double* input_list, int length) {
    std::vector<int> out(length);
    iota(out.begin(), out.end(), 0);
    sort(out.begin(), out.end(),
       [&input_list](int i1, int i2) {return input_list[i1] < input_list[i2];});
    std::vector <int> v2(out.begin()+1 , out.end());
    return v2;
}

double edist(double* arr1, double* arr2, int n) {
    double sum = 0.0;
    for (int i=0; i<n; i++) {
        sum += pow(arr1[i] - arr2[i], 2);
    }
    return sqrt(sum); 
}

std::map<int, std::vector<int> > distance_knn(const double *array,int N, int M) {
     std::map<int, std::vector<int> > dist;
    double **arr = new double*[N];
    for (int i = 0; i < N; i++) {
        arr[i] = new double[M];
        for(int j=0; j < M; j++) {
            arr[i][j] = array[i*M+j];
        }
    }
     for (int i=0; i<N; i++) {
        double distances[N];
        for(int j=0; j<N; j++) {
            // distances.push_back();
            distances[j] = edist(arr[i], arr[j], M);
        }
        dist[i] = argsort(distances, N);
    }
    return dist;
}

```

We will create interface file which is an input to SWIG that provides wrapper files.

```cpp
%module myknn
#define SWIGPYTHON_BUILTIN

%{
  #include "numpy/arrayobject.h"
  #define SWIG_FILE_WITH_INIT  /* To import_array() below */
  #include "myknn.h"
%}

%include "std_map.i"
%import "std_vector.i" 
%include "numpy.i"

%init %{
import_array();
%}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {
  (const double* array, int m, int n)
}

namespace std {
    %template(IntVector) vector<int>;
}

namespace std {
 %template (mapiv) std::map<int,vector<int> >;
}

%typemap(out) std::map<int,std::vector<int> >{
  $result = PyDict_New();
  std::map<int,std::vector<int> >::iterator iter;
  int* theVal;
  int theKey;
  for (iter = $1.begin(); iter != $1.end(); ++iter) {
    theKey = iter->first;
    int subLength = iter->second.size();
    npy_intp dims[1] = {subLength};
    int *ans = new int[subLength];
    memcpy (ans, iter->second.data(), sizeof(int) * subLength);
    PyObject *values = PyArray_SimpleNewFromData(1, &dims[0], NPY_INT, ans);
    PyDict_SetItem($result, PyInt_FromLong(theKey), values);
  }
};

%include "myknn.h"
```

Next we will create our header file.

```cpp
/* File knn.h */
#ifndef KNN_H
#define KNN_H
#include <stdio.h>
#include <vector>
#include <map>
/* Define function prototype */
std::map<int,std::vector<int> > distance_knn(const double* array,int m, int n);
std::vector<int> argsort(double* input_list, int length);
double edist(double* arr1, double* arr2, int n);
#endif
```

**Creating the wrapper**
We will write a bash file to generate the wrapper for our code.

```bash
rm *.o myknn_wrap.cpp _myknn.so myknn.py
rm -rf __pycache__

g++ -O3 -march=native -fPIC -c myknn.cpp

# Invoke SWIG on the interface file 'knn.i' to produce C/C++ wrapper
# code ('knn_wrap.cpp'):


swig -python -c++ -o myknn_wrap.cpp myknn.i

# Next, compile the wrapper code:

g++ -O3 -march=native -w -fPIC -c $(pkg-config --cflags --libs python3) -I /home/kriti/anaconda3/lib/python3.7/site-packages/numpy/core/include myknn.cpp myknn_wrap.cpp

g++ -std=c++11 -O3 -march=native -shared myknn.o myknn_wrap.o -o _myknn.so -lm
```

We will run ```bash build.sh``` command to generate wrapper.

**testing the function in python**
```python3
from myknn import distance_knn
%%time
c = distance_knn(arr)
```

```
CPU times: user 992 ms, sys: 3.04 ms, total: 995 ms
Wall time: 991 ms
```


## References
https://rushter.com/blog/numba-cython-python-optimization/ 
https://ipython-books.github.io/43-profiling-your-code-line-by-line-with-line_profiler/ 
https://rushter.com/blog/numba-cython-python-optimization/
https://medium.com/coding-with-clarity/speeding-up-python-and-numpy-c-ing-the-way-3b9658ed78f4
https://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/
https://towardsdatascience.com/speed-up-your-algorithms-part-2-numba-293e554c5cc1
https://medium.com/bcggamma/bring-your-python-code-up-to-speed-with-numba-1aa1c0e52885

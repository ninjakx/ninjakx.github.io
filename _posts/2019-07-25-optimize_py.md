---
layout: post
title: "Speeding up the python code"
featured-img: bg
categories: [Python]
---

![img](https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/py_meme.png)

<sub><sup>Image Courtesy: [lloyd-carson](https://www.flickr.com/photos/lloyd-carson/6348275920/?ref=weheartit)</sup></sub
  
# Speeding up the python code



Python is one of the most popular programming languages. Most of the companies use python in production for rapid delivery. Although it is good when it comes to production but what about its performance? In that case Python fails miserably as it is slower than most of the programming languages such as c++, java, etc. So how can we make it faster?? There are several ways of doing it. 
We will discuss it one by one.

<img src="https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/py_meme.jpg" width="500" height="600">

For demonstration purpose, I have created a script for finding out the Nearest neighbors for each data points in increasing order of euclidean distances.
Before moving to the optimization method we will first do profiling to see where it is taking more time. For that we will use line profiler which is a great tool to provide the execution of each line inside the function line profiler is applied to.

First, let's import NumPy and the line_profiler 


<script src="https://gist.github.com/ninjakx/f6c44ea074a016edbf161718e78a239d.js"></script>

Using this ```%%writefile``` cell magic we write our code in a Python script.

<script src="https://gist.github.com/ninjakx/196ebe7ccd5d789c80d85588cd7579fe.js"></script>

Next, we will import our script so that we can execute and profile our code using ```%lprun``` magic command and will save the report in a file named sim_result and display it into the cell.

<script src="https://gist.github.com/ninjakx/9bd151b9113f67e69194345e39212be3.js"></script>

![png](https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/lprof_res.png)

We can see this function comp_inner_raw( ) is consuming a lot of time for computations. So we need to optimize this function.

**Raw Python Computation time:**

<script src="https://gist.github.com/ninjakx/9109e79ef1ac8aaa3a536ed20c7b3f5d.js"></script>

```
CPU times: user 10.6 s, sys: 1.77 s, total: 12.3 s
Wall time: 12.3 s
```

## Optimizing the code using numba:

The function comp_inner_raw() is using loop and we know loops in python are slow. This part is computationally slow and uses numpy in calculations so we can use Numba which is a just-in-time complier for Python that works best on the code which uses NumPy arrays, functions, and loops. Using Numba we can generate optimized machine code from our pure python code by using LLVM compiler infrastructure. This will speed up the code.

<script src="https://gist.github.com/ninjakx/85ed714ba4c3aafac7a5a2b3466b8f25.js"></script>

<script src="https://gist.github.com/ninjakx/9a512b629ef05983b3f48e4e61e78f7f.js"></script>

```
CPU times: user 3.46 s, sys: 1.75 s, total: 5.21 s
Wall time: 5.21 s
```

## Optimizing the code using Swig:
The Simplified Wrapper and Interface Generator (SWIG) provides capability to wrap c/c++ libraries with other languages such as Python, Ruby, Java etc.

In order to create wrapper we will first create our cpp file.

<script src="https://gist.github.com/ninjakx/556f9151a99f7c311809f2cd79328b5c.js"></script>

We will create interface file which is an input to SWIG that provides wrapper files.

<script src="https://gist.github.com/ninjakx/d3e89206dbb9651736c87e7a907fbcfb.js"></script>

Next we will create our header file.

<script src="https://gist.github.com/ninjakx/1840cc6357b87013ca6320900c4c36eb.js"></script>

**Creating the wrapper**
We will write a bash file to generate the wrapper for our code.
<script src="https://gist.github.com/ninjakx/f18242a00e907c808650dfdc8891ffdf.js"></script>

We will run ```bash build.sh``` command to generate wrapper.

**testing the function in python**

<script src="https://gist.github.com/ninjakx/2d7240fc0706037e17a0746db53cd7de.js"></script>

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

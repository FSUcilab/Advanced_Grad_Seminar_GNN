#!/usr/bin/env python
# coding: utf-8

# <script>
#   jQuery(document).ready(function($) {
# 
#   $(window).load(function(){
#     $('#preloader').fadeOut('slow',function(){$(this).remove();});
#   });
# 
#   });
# </script>
# 
# <style type="text/css">
#   div#preloader { position: fixed;
#       left: 0;
#       top: 0;
#       z-index: 999;
#       width: 100%;
#       height: 100%;
#       overflow: visible;
#       background: #fff url('http://preloaders.net/preloaders/720/Moving%20line.gif') no-repeat center center;
#   }
# 
# </style>
# 
# <div id="preloader"></div>

# <script>
#   function code_toggle() {
#     if (code_shown){
#       $('div.input').hide('500');
#       $('#toggleButton').val('Show Code')
#     } else {
#       $('div.input').show('500');
#       $('#toggleButton').val('Hide Code')
#     }
#     code_shown = !code_shown
#   }
# 
#   $( document ).ready(function(){
#     code_shown=false;
#     $('div.input').hide()
#   });
# </script>
# <form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>

# ### Latex Macros
# $\newcommand{\Re}[1]{{\mathbb{R}^{{#1}}}}
# \newcommand{\Rez}{{\mathbb{R}}}$

# In[41]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
#%tableofcontents


# In[42]:


import copy

import ipywidgets as widgets
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sympy
import torch
from ipywidgets import fixed, interact, interact_manual, interactive
from scipy.stats import ortho_group  # compute unitary matrices

import spectral_function_library as spec_lib

get_ipython().run_line_magic('matplotlib', 'inline')


# # Convolution Neural Networks 
# Material is taken from [this Blog](https://www.instapaper.com/read/1477946505)
# 
# Starting from an RGB image: 
# 
# <img src="images/rgb_image_2022-01-24_10-15-38.png" width="800">
# 
# the idea is pass this image through as series of steps in order to extract information. The filter is used for this task. 
# 
# <img src="images/convolution_2022-01-24_10-17-28.png" width="800">
# 
# after image src.
# 
# <img src="images/multihead_2022-01-24_10-19-47.png" width="800">
# 
# <img src="images/step-by-step_2022-01-24_10-18-45.png" width="800">
# 
# 
# ## Two important points about the convolutional layer: 
# 
# 1. The filter is identical for each pixel. This reduces the number of parameters to calculate. 
# The constant filter helps satisfy the inductive bias of *translation invariance*. 
# 
# 2. The convolution is local to the image pixel to which it is applied. Thus, the structure of the image is taken into account during the calculation. 
# 
# A typical CNN architecture: 
# 
# <img src="images/cnn_2022-01-24_10-25-56.png" width="800">

# jupyter nbextension enable --py widgetsnbextensionjupyter nbextension enable --py widgetsnbextensionjupyter nbextension enable --py widgetsnbextensionjupyter nbextension enable --py widgetsnbextension# Alternative view of CNN
# 
# <img src="images/image_graph_2022-01-24_11-00-16.png" width="800">
# 
# * An image can be considered to be a graph
# * The nodes $V$ are the centers of the pixels
# * If a filter has width 3, each nodes is connected to $8 * d$ adjacent nodes, where $d$ is the number of channels

# # Motivation
# Consider a set of nodes $x_i$, and associated attributes $y_i$. This can be graphed. Let us connect these nodes with edges $e_{ij} = (x_i, x_{i+1})$.

# In[43]:


@interact(N=(5, 40))
def plot1d(N):
    x = np.linspace(0, 10, N)
    plt.plot(x, 0 * x, "-o")
    plt.show()


# Add an attribute to each of these nodes. I will add a random noise in $N(0,\sigma)$ and $\sigma=1.5$, which is fairly large. 
# 
# Consider the problem of computing *embeddings* of each node with the requirement that nearby nodes with similar attributes should have similar embeddings. 
# 
# Without further constraints imposed on the problem (also called *inductive biases*, we will apply a local transformation to this function, and specifically an averaging operation. We will replace $y_i$ by the average of its neighbors :
# $$ y_i \longrightarrow \frac12 (y_{i-1} + y_{i+1})$$
# The boundary points need special treatment. There are three main ehoices: 
# 1. Do not move the point
# 2. Move the point in such a way as to satisfy some condition on the slope.
# 3. Develop an algorithm that figures out the proper treatment
# 
# We will consider the first choice for simplicity. For future reference, we call the collection of points $V$, the collection of edges $E$. We denote the boundary nodes by $\partial V$, and the boundary edges (edges attached to $\partial V$ by $\partial E$, which is a common notation in discrete and differential geometry. 

# In[44]:


@interact(seed=(1, 100), eps=(0, 1.5), N=(5, 40))
def plot1d(seed, eps, N):
    np.random.seed(seed)
    x = np.linspace(0, 10, N)
    noise = eps * np.random.randn(N)
    y = np.sin((x / x[-1]) * 2 * np.pi * 2.5) + noise
    plt.plot(x, y, "-o")
    plt.show()


# More generally, each point might have multiple attribute. Thus, the node $x_i$, would have $d$ attributes $y_0, \cdots, y_{d-1}$. These attributes could be categorical or continuous, and the categorical attributes could be nominal (there is nor ordering, such as 'red', 'blue', 'orange') or ordinal (bad, poor, average, good, very good excellent). 

# In[45]:


dSlider = widgets.IntSlider(min=1, max=5, value=3, description="Nb Attributes")
seedSlider = widgets.IntSlider(min=1, max=100, value=50, description="Seed")
epsSlider = widgets.FloatSlider(
    min=0.0, max=1.5, value=0.30, description="Noise $\sigma$"
)


@interact(seed=seedSlider, eps=epsSlider, N=(5, 40), d=dSlider, nb_blur_iter=(0, 5))
def plot1d(seed, eps, N, d, nb_blur_iter):
    np.random.seed(seed)
    eps = eps * np.array([1.0, 2.0, 0.5, 3.0, 4.0])
    x = np.linspace(0, 10, N)
    noise = np.random.randn(d, N)
    y = np.zeros([5, N])
    fcts = {}
    fcts[0] = np.sin((x / x[-1]) * 2 * np.pi * 2.5)
    fcts[1] = 1.5 * np.cos((x / x[-1]) * 2 * np.pi * 2.5) ** 2
    fcts[2] = x ** 2 / 10 * np.exp(3 - 0.5 * x)
    fcts[3] = np.cos((x / x[-1]) * 2 * np.pi * 4.5)
    fcts[4] = 1.5 * np.cos((x / x[-1]) * 2 * np.pi * 2.5)
    for i in range(0, 5):
        y[i] = fcts[i]

    for i in range(0, d):
        y[i] += eps[i] * noise[i]

    yy = copy.copy(y)

    for i in range(0, d):
        for n in range(0, nb_blur_iter):
            yy[i][0] = y[i][0]
            yy[i][N - 1] = y[i][N - 1]
            yy[i][1 : N - 2] = 0.5 * (y[i][0 : N - 3] + y[i][2 : N - 1])
            y = copy.copy(yy)

    for i in range(0, d):
        plt.plot(x, yy[i], "-o")
    plt.grid(True)
    plt.ylim(-2, 5)
    plt.show()


# So far, I am describing vector-valued discrete functions of $x$, which is a 1-D representation of a graph $d$ attributes at each node $x_i$. More generally, nodes are points in *some* space, which can be 1-D, 2-D, higher-D, or more abstract, namely, a space of *points*. 
# 
# Now consider adding attributes $y_{Eij}$ to the edges. What kind of transformation functions should one consider? 
# 
# This averaging function is an example of a local filter defined in physical space. This filter takes attributes at nodes and transforms them into a new set of number defined at these same nodes. More generally, in Graph Neural networks, we will consider operators that take attributes defined at nodes, edges, and the graph, and transform them into a new set of vectors defined on these same nodes, vectors and graphs. 
# 
# Filters can be defined either in physical space or in spectral space. We will illustrate the concept by considering the derivative operator on continuous and discrete grids.

# ## First Derivative operator (also a filter) on 1D grid in physical space
# Consider points $x_i$, $i=0,\cdots, N-1$ connected by edges $e_{i,i+1} = (x_i, x_{i+1})$. The central difference operator of the function $f_i = f(x_i)$ is defined by
# $$
# f'_i = \frac{f_{i+1} - f_{i-1}}{x_{i+1} - x_{i-1}}
# $$ for $i=1,\cdots,N-2$, with one-sided operators defined at the boundaries (which is one of many possibilities): 
# \begin{align}
# f'_0 &= \frac{f_1 - f_0}{x_1-x_0} \\
# f'_{N-1} &= \frac{f_{N-1} - f_{N-2}}{x_{N-1} - x_{N-2}}
# \end{align}
# where $f'_i$ is the approximation of $f'(x)$ evaluated at $x=x_i$. Note that the derivative can be expressed as a vector 
# $f' = (f'_0,\cdots,f'_{N-1})$, and $f'_i$ is linear with respect to the values $f_j$. Therefore one can write the matrix 
# expression: 
# $$ f' = D f $$
# where $D \in \Re{N\times N}$ is an $N \times N$ matrix.  The matrix $D$ is a derivative filter. More specifically, it is a 
# *global* filter since it updates the values at all nodes at once. To the contrary, a *local* filter is defined as the matrix that updates the derivative at a single point. Thus: 
# $$
# f'_i = (\begin{matrix}-\alpha & 0 & \alpha\end{matrix})^T 
#    (\begin{matrix} f_{i+1} & 0 & f_{i-1}) \end{matrix}
# $$
# where a superscript $T$ denotes transpose, and $\alpha = (x_{i+1} - x_{i-1})^{-1}$. Clearly, the local 
# filter is local to the point at which it applies. The new value only depends on the values of its immediate neighbors. 

# ***
# # Spectral Analysis of graphs
# ## Continuous Fourier Transform (CFT)
# When working in the continuous domain $\Rez$, a function $f(x)\in\Rez$ has a Fourier Transform $\hat{f}(k)$ related by 
# $$  \hat{f}(k) = \frac{1}{2\pi} \int_{-\infty}^\infty e^{\iota k x} f(x) \, dx $$
# Conversely, one can apply a similar operation to recover $f(x)$ from its Fourier Transform: 
# 
# $$ f(x) = \frac{1}{2\pi} \int_{-\infty}^\infty e^{-\iota k x} \hat{f}(k) \, dk $$
# 
# Notice the sign in the exponent: positive when transforming from physical to Fourier space, and negative when returning to physical space. The sign is a convention. Different authors might use the opposite sign. So always pay attention to the conventions in any paper you read. 
# 
# (you should all have learned about the Fourier transform previously). 
# 
# Let us compute the first derivative of $f(x)$: 
# $$\frac{d}{dx} f(x) = f'(x)$$
# The conventional approach would be to calculate the derivative manually, or discretize the expression in physical space. However, the alternative is to compute the derivative by first transforming the expression to Fourier (also called spectral) space: 

# \begin{align}
#  \frac{d}{dx} f(x) &= \frac{d}{dx} \frac{1}{2\pi} \int_{-\infty}^\infty e^{-\iota k x} \hat{f}(k)  d k  \\ 
#  &=  \frac{1}{2\pi} \int_{-\infty}^\infty (-\iota k) e^{-\iota k x} \hat{f}(k) dk \\
#  &= \cal{F}^{-1} [-\iota  k \hat{f}(k)]
# \end{align}
# where 
# \begin{align}
# \cal{F}f(x) &= \hat{f}(k) \\
# \cal{F}^{-1} \hat{f}(k) &= f(x) \\
# \end{align}

# So to given a function $f(x)$, one can compute the derivative with the following three steps: 
# 1.  $f(x) \longrightarrow \hat{f}(k)$
# 2.  $\hat{f}(k) \longrightarrow (-\iota k) \hat{f}(k)$
# 3.  $(-\iota k)\hat{f}(k) \longrightarrow \cal{F}^{-1} \left[(-\iota k)\hat{f}(k)\right] = \frac{d}{dx} f(x)$

# Thus, the derivative operation is applied in Fourier space. A complex operation in physical space becomes a simple multiplication in Fourier space, *at the cost* of two Fourier Transforms. 

# ### Fourier Spectrum
# $\hat{f}(k)$ is called the Fourier Spectrum and is generally a complex variable. 
# $P(k) = |\hat{f}(k)|^2$ is the power spectrum, and satisfies the property: 
# $$
# \int_{-\infty}^\infty P(k) dk = \int_{-\infty}^\infty |\hat{f}(k)|^2 dx = \int_{-\infty}^\infty |f(x)|^2 dx
# $$
# a rule that generalizes to and holds in $\Re{n}$. 

# ### Filter
# The coefficient $(-\iota k)$ above is an example of a complex operator in Fourier space. This operator tranforms a function $\hat{f}(k)$ into a "filtered" function $\hat{g}(k)$: 
# $$
# \hat{g}(k) = (-\iota k) \hat{f}(k)
# $$
# and in this particular case, results in the Fourier transform of the $x$-derivative of $f(x)$. More generally, one can define an operator $\hat{H}(k)$ acting on $\hat{f}(k)$, which "shapes" the power spectrum, leading to filters with different characteristics: low-pass, band-pass, high-pass, custom. 
# 
# Given a function $f(x)$, the resulting filtered function $f_H(x)$ can be defined similarly to the derivative: 
# 
# \begin{align}
#  f(x) & \longrightarrow \cal{F}(f(x)) = \hat{f}(k) \\
#   \hat{f}(k) & \longrightarrow \hat{H}(k) \hat{f}(k) \\
#  \hat{H}(k)\hat{f}(k) & \longrightarrow \cal{F}^{-1} (\hat{H}(k)\hat{f}(k)) = f_H(x)
# \end{align}
# 
# We will often omit the argument $x$ or $k$, letting the "hat" notation indicate whether or not we are in Fourier space. Thus, we can write
# $$
# f_H = \cal{F}^{-1} [\hat{H} \; \cal{F}(f) ]
# $$ or the equivalent form (requiring the definition of product of operators): 
# 
# \begin{align}
# f_H &= (\cal{F}^{-1} \, \hat{H} \, \cal{F}) \; f \\
#    &= H f
# \end{align}
# which defines the filter $H(x)$ in physical space, acting on $f(x)$ to produce $f_H(x)$: 
# $$
# f_H(x)   = H(x) * f(x)
# $$
# where $*$ denotes the convolution operator: 
# $$
# H(x) * f(x) = \int_{-\infty}^\infty H(x-s) f(s) \, ds
# $$

# ## Formal proof of convolution theorem in continuous space
# We start with the relation: 
# $$ H = \cal{F}^{-1} \hat{H} \cal{F} $$
# and express both sides of the equation in integral form: 
# \begin{align}
# \int e^{-\iota k x} \left( \hat{H}(k)\hat{f}(k)\right) \, dk &=
#    \int e^{-\iota k x}\, dk \left( \int e^{\iota k x''} H(x'')\,dx'' \int e^{\iota k x'} f(x') \, dx' \right) \\
#      &= \int dk \int e^{\iota k (x'' + x' - x)} H(x'') f(x) \, dx' \, dx''
# \end{align}
# Now make use of the following integral definition of the Dirac function: 
# $$
# \int e^{\iota k x} \, dk = 2\pi \delta(x)
# $$
# which leads to
# \begin{align}
# \int e^{-\iota k x} \left( \hat{H}(k)\hat{f}(k)\right) \, dk &=
#   \int dk \int e^{\iota k (x'' + x' - x)} H(x'') f(x') \, dx' \, dx'' \\
# &= 2\pi \int \delta(x'' + x' - x) H(x'') f(x') \, dx' \, dx'' \\
# &= 2\pi \int H(x-x') f(x') \, dx' \\
# &= C \; H(x) * f(x) = L(x)
# \end{align}
# where $C$ is a constant of proportionality. 
# I was not careful with constants in front of the integrals when taking Fourier transforms and their 
# inverses. 
# 
# We thus find that 
# $$
# \cal{F}^{-1} \left(\hat{H}(k)\hat{f}(k)\right) =  H * f
# $$
# Careful calculations show that the constant $C=1$. 
# 
# Integrating $A(x)$ over $x$ leads to: 
# $$
# \int \hat{H}(k) \hat{f}(k) \, dk = \int H(x) f(x) \, dx
# $$
# often referred to as [Plancherel's identity](https://en.wikipedia.org/wiki/Parseval%27s_identity).
# 
# All integrals are taken over the domain $[-\infty, \infty]$. 

# ---
# # Ideal Low-, Mid-, High-pass filters
# ## Low-pass filter
# 
# \begin{align}
# H(k) &= 1, \hspace{1in} k < k_0 \\
# &= 0, \hspace{1in} k \ge k_0 
# \end{align}
# ## Band-pass filter
# 
# \begin{align}
# H(k) &= 1, \hspace{1in} k_0 < k < k_1, \; k_0 < k_1 \\
# &= 0  \hspace{1in} \rm{otherwise}
# \end{align}
# ## High-pass filter
# 
# \begin{align}
# H(k) &= 1, \hspace{1in} k > k_0 \\
# &= 0, \hspace{1in} k \le k_0 
# \end{align}
# 

# #### Notes: 
# * np.fft uses the discrete Fourier Tranform since the grid is discrete (we skip over these details)
# * The $x$-domain is $[0,0.5]$. 
# * $\sin(2\pi f_1 x)= 0$ at $x=0$ and $x=0.5$. The $x-derivative is $2\pi f_1\cos(f_1 2\pi x)$, equal 
# to $2\pi f_1$ at $x=0$ and $2\pi f_1 \cos(\pi f_1)$ at $x=0.5$, equal to 2\pi f_1$ if $f_1$ is even. 
# Therefore the function is  periodic over the domain, since the $f_1$ slider ranges from -40 to 40 by increments of 10.  
# On the other hand, $\cos(2\pi f_3 x + 0.7)$ is not periodic over the $x$ domain (the phase is 0.7, which is not a multiple of $2\pi$. The frequencies are obtained by 
# decomposing this function into a series of $\sin$ and $\cos$ at different frequencies with zero phase. 

# In[46]:


grid = widgets.GridspecLayout(3, 3)


# In[47]:


freq1Slider = widgets.IntSlider(min=0, max=60, value=30)
freq2Slider = widgets.IntSlider(min=30, max=120, value=70)
freq3Slider = widgets.IntSlider(min=90, max=200, value=110)
ampl1Slider = widgets.FloatSlider(min=-15, max=15, value=5)
ampl2Slider = widgets.FloatSlider(min=-15, max=15, value=10)
ampl3Slider = widgets.FloatSlider(min=-15, max=15, value=10)
k0Slider = widgets.IntSlider(min=0, max=50, value=15)
k1Slider = widgets.IntSlider(min=5, max=150, value=100, Description="k1")


# In[48]:


@interact_manual(
    freq1=freq1Slider,  # (-20, 60, 10),
    freq2=freq2Slider,  # (-90, 90, 10),
    freq3=freq3Slider,  # (-300, 300, 15),
    ampl1=ampl1Slider,  # 1,
    ampl2=ampl2Slider,  # 0.5,
    ampl3=ampl3Slider,  # 1,
    k0=k0Slider,  # (0, 50, 5),
    k1=k1Slider,  # (5, 150, 10),
)
def plotSin2(freq1, freq2, freq3, ampl1, ampl2, ampl3, k0, k1):
    fig = plt.figure(figsize=(16, 7))
    x = np.linspace(0, 0.5, 500)
    k = np.linspace(0, 499, 500)

    # NOTE: These functions are NOT periodic over the domain.
    # Therefore, the spectrum is not exactly a collection of delta functions
    # I could be more precise, but that is not the point of this demonstration.

    s = (
        ampl1 * np.sin(freq1 * 2 * np.pi * x)
        + ampl2 * np.sin(freq2 * 2 * np.pi * x)
        + ampl3 * np.cos(freq3 * 2 * np.pi * x + 0.7)
    )
    nrows, ncols = 3, 2

    # ax1.clear()  # to avoid flicker, does not work
    ax = fig.add_subplot(nrows, ncols, 1)
    # fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5))
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time [s]")
    ax.plot(x, s)

    fft = np.fft.fft(s)
    ifft = np.fft.ifft(s)
    # print("s: ", s[0:10])
    # print("ifft: ", ifft[0:11])
    # print("fft[0-10]: ", fft[0:11])
    # print("fft[:-10,:]: ", fft[-10:])
    power_spec = np.abs(fft) ** 2
    # power_spec[0] = 0 # REMOVE MEAN COMPONENT  (simply equal to the mean of the function)
    ax2 = fig.add_subplot(nrows, ncols, 2)
    ax = ax2
    ax.plot(power_spec[0:250])
    ax.set_ylabel("Power Spectrum")
    ax.set_xlabel("k")

    heaviside = np.where((k > k0) & (k < k1), 1, 0)
    # Symmetrize this function with respect to $k=500/2$
    for i in range(1, 250):  # 250 = 500/2
        heaviside[500 - i] = heaviside[i]  # in Fourier space
    # print(heaviside)

    filtered_power_spectrum = power_spec * heaviside
    # print(list(zip(power_spec, heaviside, filtered_power_spectrum)))
    # print("power spec: ", power_spec[0:50])
    # print("filtered_spec: ", filtered_power_spectrum[0:50])
    filtered_function = np.fft.ifft(filtered_power_spectrum)

    ax = fig.add_subplot(nrows, ncols, 3)
    ax.plot(filtered_function)
    ax.set_ylabel("Filtered $f_H(x) = H(x) f(x)$")
    ax.set_xlabel("x")

    ax = fig.add_subplot(nrows, ncols, 4)
    ax.plot(filtered_power_spectrum[0:250])
    ax.set_xlabel("k")
    ax.set_ylabel("Filtered Power Spectrum")

    filter_phys = np.fft.ifft(heaviside)
    ax = fig.add_subplot(nrows, ncols, 5)
    ax.plot(filter_phys)
    ax.set_ylabel("Filter $H(x)$")
    ax.set_xlabel("k")

    ax = fig.add_subplot(nrows, ncols, 6)
    ax.plot(heaviside[0:250])
    ax.set_ylabel("Filter $\hat{H}(k)$")
    ax.set_xlabel("k")

    plt.tight_layout()
    plt.show()
    sumf2 = np.sum(s ** 2)
    sump2 = np.sum(power_spec[0:250])
    sump3 = np.sum(power_spec)
    # print(sum2, sump2, sump2 / sumf2, sump3 / sumf2)
    # print(np.sum(power_spec[0:250]), np.sum(power_spec[0:500]), power_spec.shape)

    # The ratio sump2 / sumf2 = 250 (when there is no mean component)
    # The k=0 component has no complex conjugate. All other components have a complex conjugate.
    # These details are beyond the scope of this lecture.
    #     = Number of points N / 2
    # sum f[i]^2 dx = sum f[i]^2 (0.5/N) = sum power_spectrum  * normalizing constant
    #  (one must be careful with this constant)


# Alternative to @interact
# interact(plotSin2, freq1=(-40,40,10), freq2=(-90,90,10), freq3=(-300,300,15), ampl1=1, ampl2=.5, ampl3=1)


# The strong oscilations in the Filter $H(x)$ are due to the discontinuity of the filter in Fourier space. 
# A property of these 1-D filters is that localization in Fourier space (the filter is nonzero for very few $k$) leads 
# to non-local filters $H(x)$ in physical space, and vice-versa. 
# 
# The challenge is to construct filters local in both physical and Fourier space, which is the strength of wavelets (beyond the scope of these lectures). Note that the Fourier transform of a Gaussian is a Gaussian, and it is local in both spaces. (Demonstrate it for yourself as a homework exercise). 

# 
# ### Discrete 1D domain
# * A set of nodes $x_i$, $i=0,1,\cdots,N-1$, such that $x_i$ is connected to $x_{i+1}$. This graph is acyclic (there are no cycles. 
# * If the first and last node are connected, we add the edge $(x_{N-1}, x_{0})$ and create a cyclic graph.
# * The adjacency matrix of the cyclic graph is as follows: 
# $$
# A = \left(\begin{matrix}
# 0 & 0 & 0 & \cdots & 0 & 1 \\
# 1 & 0 & 0 & \cdots & 0 & 0 \\
# 0 & 1 & 0 & \cdots & 0 & 0 \\
# 0 & 0 & 1 & \cdots & 0 & 0 \\
# \cdots
# \end{matrix}\right)
# $$
# * A signal $s$ on a graph is defined as the sequence of $N$ elements
# $$ x = (x_0, x_1, \cdots, x_{N-1}) $$
# where each $x_i\in\Rez$.

# ### 1-D Periodic Domain
# #### Fourier Filter
# ### 1-D Non-periodic Domain
# ## Fourier Transform, Discrete (DFT)
# ### 1-D Periodic Domain 
# ### 1-D Non-periodic Domain
# ## Graph Signal Processing, Discrete
# ### 1-D cyclic graph
# ### 2=D Discrete periodic 
# ### Adjoint $A$
# ### Degree Matrix $D$
# ### Laplacian $L$
# ###

# In[49]:


# layout = ['circular','planar','random']
seed_slider = widgets.IntSlider(min=100, max=120, step=2, value=110)
N_slider = widgets.IntSlider(min=5, max=40, step=1, value=10)
# matrix = ['Adjacency Matrix', 'Laplacian', 'D^-1 A', 'D^-1 L', 'D^-1/2 L D^-1/2']


@interact(N=N_slider, seed=seed_slider)
def generate_graph_from_adjacency_matrix(N, seed):
    """
    Arguments
    N: number of nodes
    """
    np.random.seed(seed)
    ints = np.random.randint(0, 2, N * N).reshape(N, N)
    for i in range(N):
        ints[i,i] = 0

    # Symmetric array
    ints = ints + ints.transpose()
    ints = np.clip(ints, 0, 1)  # the elements should be zero or 1

    # Different matrices
    A = ints
    D = np.sum(A, axis=0)
    D = np.diag(D)
    L = D - A
    invD = np.linalg.inv(D)
    invDA = A * invD
    invDL = invD * L
    invDLinvD = np.sqrt(invD) * L * np.sqrt(invD)

    matrix = ["A", "D", "L", "invD", "invDA", "invDL", "invDinvD"]
    matrices = [A, D, L, invD, invDA, invDL, invDLinvD]

    # Eigenvalues
    fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    axes = axes.reshape(-1)
    fig.suptitle("Sorted Eigenvalues of various matrices")
    for i, m in enumerate(matrices):
        ax = axes[i]
        eigs = np.linalg.eigvals(m)
        eigs = np.sort(eigs)[::-1]
        ax.set_title(matrix[i])
        ax.grid(True)
        ax.plot(eigs, "-o")
    for i in range(i + 1, axes.shape[-1]):
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


# ### Notes
# * The eigenvalues (spectrum )of A and L are approximatley related (the plots look very similar) but not equal.
# * The spectra shape depend very little on the seed (A is filled with random numbers (0,1) and is symmetrized to make sure that the eigenvalues $\lambda_i \in \Rez$.

# ***
# ## Same plot as above but allowing for different types of graph types. 
# * Generate the graph, compute the adjacent matrix, and call the previous function
# 
# 

# In[50]:


def generate_graph_from_adjacency_matrix_1(G, N, seed):
    """
    Arguments
    N: number of nodes
    """
    np.random.seed(seed)

    # Convert to np.ndArray
    A = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
    nx.linalg
    # print("Adj: ", A, "\n", A.shape, "\n", type(A))

    # Different matrices
    D = np.sum(A, axis=0)
    D = np.diag(D)
    L = D - A
    invD = np.linalg.inv(D)
    invDA = A * invD
    invDL = invD * L
    invDLinvD = np.sqrt(invD) * L * np.sqrt(invD)

    Ln = nx.normalized_laplacian_matrix(G)
    Ln = Ln.toarray()  # from sparse array to ndarray

    matrix = ["A", "D", "L", "invD", "invDA", "invDL", "invDinvD", "Ln"]
    matrices = [A, D, L, invD, invDA, invDL, invDLinvD, Ln]

    # Eigenvalues
    fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    axes = axes.reshape(-1)
    fig.suptitle("Eigenvalues of various matrices")
    for i, m in enumerate(matrices):
        ax = axes[i]
        eigs = np.linalg.eigvals(m)
        eigs = np.sort(eigs)[::-1]
        ax.set_title(matrix[i])
        ax.grid(True)
        ax.plot(eigs, "-o")
    for i in range(i + 2, axes.shape[-1]):
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


# In[51]:


prob_slider = widgets.FloatSlider(min=0, max=1, step=0.1, value=0.5)
node_slider = widgets.IntSlider(min=3, max=30, step=1, value=10)
nb_neigh_slider = widgets.IntSlider(min=1, max=10, step=1, value=4)
nb_edges_per_node_slider = widgets.IntSlider(min=1, max=20, step=2, value=5)
seed_slider = widgets.IntSlider(int=1, max=50, step=1, value=25)
graph_type = ["connected_watts_strogatz", "powerlaw_cluster_graph"]


@interact(
    nb_nodes=node_slider,
    prob=prob_slider,
    nb_neigh=nb_neigh_slider,
    nb_edges_per_node=nb_edges_per_node_slider,
    seed=seed_slider,
    graph_type=graph_type,
    # directed=True,
)
def drawGraph(nb_nodes, nb_neigh, prob, seed, nb_edges_per_node, graph_type):
    if graph_type == "connected_watts_strogatz":
        nb_edges_per_node_slider.style.handle_color = 'red'
        nb_neigh_slider.style.handle_color = 'black'
        nb_tries = 20
        edge_prob = prob
        G = nx.connected_watts_strogatz_graph(
            nb_nodes, nb_neigh, edge_prob, nb_tries, seed
        )
    elif graph_type == "powerlaw_cluster_graph":
        nb_neigh_slider.style.handle_color = 'red'
        nb_edges_per_node_slider.style.handle_color = 'black'
        add_tri_prob = prob
        if nb_edges_per_node >= nb_nodes:
            nb_edges_per_node = nb_nodes - 1
        G = nx.powerlaw_cluster_graph(nb_nodes, nb_edges_per_node, add_tri_prob, seed)

    generate_graph_from_adjacency_matrix_1(G, nb_nodes, seed)


# <script>
#   $(document).ready(function(){
#     $('div.prompt').hide();
#     $('div.back-to-top').hide();
#     $('nav#menubar').hide();
#     $('.breadcrumb').hide();
#     $('.hidden-print').hide();
#   });
# </script>
# 
# <footer id="attribution" style="float:right; color:#999; background:#fff;">
# Created with Jupyter, delivered by Fastly, rendered by Rackspace.
# </footer>

# # prob_slider = widgets.FloatSlider(
#     min=0, max=1, step=0.1, value=0.5, description="Probability"
# )
# node_slider = widgets.IntSlider(min=3, max=20, step=1, value=7)
# nb_neigh_slider = widgets.IntSlider(min=1, max=10, step=1, value=4)
# nb_edges_per_node_slider = widgets.IntSlider(min=1, max=20, step=2, value=5)
# seed_slider = widgets.IntSlider(int=1, max=50, step=1, value=25)
# graph_type = ["connected_watts_strogatz", "powerlaw_cluster_graph", "circular_graph"]
# 
# # Also draw the eigenfunctions for the cyclic case where the nodes are arranged in a circular layout,
# # with labels in the nodes
# 
# 
# @interact_manual(
#     nb_nodes=node_slider,
#     prob=prob_slider,
#     nb_neigh=nb_neigh_slider,
#     nb_edges_per_node=nb_edges_per_node_slider,
#     seed=seed_slider,
#     graph_type=graph_type,
# )
# def drawGraphEigenvalues(nb_nodes, nb_neigh, prob, seed, nb_edges_per_node, graph_type):
#     if graph_type == "connected_watts_strogatz":
#         nb_edges_per_node_slider.style.handle_color = "red"
#         nb_neigh_slider.style.handle_color = "black"
#         nb_tries = 20
#         edge_prob = prob
#         G = nx.connected_watts_strogatz_graph(
#             nb_nodes, nb_neigh, edge_prob, nb_tries, seed
#         )
#     elif graph_type == "powerlaw_cluster_graph":
#         nb_neigh_slider.style.handle_color = "red"
#         nb_edges_per_node_slider.style.handle_color = "black"
#         add_tri_prob = prob
#         if nb_edges_per_node >= nb_nodes:
#             nb_edges_per_node = nb_nodes - 1
#         G = nx.powerlaw_cluster_graph(nb_nodes, nb_edges_per_node, add_tri_prob, seed)
#     elif graph_type == "circular_graph":
#         nb_neigh_slider.style.handle_color = "red"
#         nb_edges_per_node_slider.style.handle_color = "red"
#         nb_neigh_slider.style.handle_color = "red"
#         prob_slider.style.handle_color = "red"
#         seed_slider.style.handle_color = "red"
# 
#         G = nx.Graph()
#         for n in range(nb_nodes):
#             G.add_node(n)
#         for n in range(nb_nodes):
#             G.add_edge(n, n + 1)
#         G.add_edge(nb_nodes - 1, 0)
# 
#     spec_lib.generate_eigenvectors_from_adjacency_matrix_1(G, nb_nodes, seed)

# In[52]:


# Test Eigenfunction, sorting, etc. by creating a matrix whose eigenvalues I know
N_slider = widgets.IntSlider(min=3, max=10, step=1, value=5)
seed_slider = widgets.IntSlider(min=100, max=200, step=1)


@interact(N=N_slider, seed=seed_slider)
def test_eigen(N, seed):
    # generate eigenvalues
    np.random.seed(seed)
    # large variance for wider spread of spectrum
    eigens = (20.0 + 100.0 * np.random.randn(N)) / 20
    eigens = np.where(eigens < 0, -eigens, eigens)
    print("eigens= ", eigens)
    print("eigens[0]= ", eigens[0])
    print("eigens[1]= \n", eigens[1])
    # print("eigens= \n", eigens)
    eigens = np.diag(eigens)
    ee = np.linalg.eig(eigens)
    print("ee= \n", ee)
    print("ee[0]= ", ee[0], type(ee[0]))
    print("ee[1]= \n", ee[1])

    args = np.argsort(ee[0])
    print("args:", args, type(args))
    ee0 = ee[0][args]
    ee1 = ee[1][:, args]
    print("sorted ee")
    print("ee[0]= ", ee0)
    print("ee[1]= \n", ee1)
    recursivelyrecursively

    # create eigenvectors
    x = ortho_group.rvs(N)
    # Similarity transform (eigenvalues of A are invariant)
    A = x.T @ eigens @ x
    # A = x @ np.linalg.inv(x)
    # print("A= \n", A)
    # print("x.T= \n", x.T)
    # print("inv(x)= \n", np.linalg.inv(x))
    eigens = np.linalg.eig(A)
    args = np.argsort(eigens[0])
    print("===============================")
    print("args: \n", args)
    eigs = eigens[0][args]
    print("unsorted eigs: \n", eigens[0])
    print("sorted eigs: \n", eigs)
    eigv = eigens[1][:, args]
    print("unsorted x:\n ", x.T)
    print("unsorted eigv: \n", eigens[1])
    print("sorted x: \n", x.T[:, args])
    print("sorted eigv= \n", eigv)

    pass


# # Exploration of eigenvalue and eigenfunctions for the 1-D cyclic and non-cyclic cases
# As we have seen,  a signal $s^1=(s_0, s_1, \cdots, s_{N-1})\in\Re{N}$, is transformed into a signal $s^2\in\Re{N}$ by a filter $H$ according to 
# $$ s^2 = H s^1$$  where $H$ is a matrix in $\Re{N\times N}$. Applying this filter recursively, one finds that 
# \begin{align}
# s^3 &= H s^2 \\
# s^4 &= H s^3 \\
# s^l &= H s^{l-1}
# \end{align}
# If this is done a large number of times, and if one assumes convergence of $s^l$ to a vector of finite norm, one finds in the limit: 
# $$
# s^\infty = H s^\infty
# $$
# which states that $s^\infty$ is an eigenvector of the filter $H$ with a unit eigenvalue $\lambda=1$. 

# ## Cyclic case, directed graph
# The adjoint matrix is
# $$
# A = \left(\begin{matrix}
# 0 & 0 & 0 & \cdots & 0 & 1 \\
# 1 & 0 & 0 & \cdots & 0 & 0 \\
# 0 & 1 & 0 & \cdots & 0 & 0 \\
# 0 & 0 & 1 & \cdots & 0 & 0 \\
# \cdots
# \end{matrix}\right)
# $$
# Recall: $A_{i,j} = 1$ means an edge goes from node $j$ to node $i$. In this case, there is an edge from node $i+1$ to node $i$
# for all nodes. There is also an edge from node $N-1$ to node $0$. This matrix is periodic. 
# 
# Given a signal 
# $$
# s =  (s_0, s_1, \cdots, s_{N-1})
# $$
# the action of $A$ on $s$ simply shifts the value $s_i$ on node $i$ to node $i-1$: 
# $$
# s^1 = A s = (s_{N-1}, s_0, s_1, \cdots, s_{N-2})
# $$
# 
# In the next animation, we define a graph over a set of nodes, and a signal on this graph, and we apply the operator
# $A$ multiple times. 

# In[53]:


j = -1


@interact_manual(seed=(1, 100), eps=(0, 1.5), N=(5, 40))
def plot1d(seed, eps, N=15):
    global j
    np.random.seed(seed)
    # Define a NxN matrix
    A = np.zeros([N, N])
    for i in range(1, N):
        A[i, i - 1] = 1
    A[0, N - 1] = 1
    x = np.linspace(0, 10, N)

    # Signal s
    noise = eps * np.random.randn(N)
    s = np.sin((x / x[-1]) * 2 * np.pi * 2.5) + noise

    j += 1
    Aj = np.linalg.matrix_power(A, j)
    new_s = Aj @ s

    print(Aj)
    plt.plot(x, s, "-o", color="red")
    plt.plot(x, new_s, "-o")
    plt.title("Press button to apply $A$")
    plt.show()


# A is called the shift operator in  1-D signal processing. Application of $A$ to a time signal translates the signal by $\Delta t$. The same is true with our graph. Of course, we are working with a special kind of graph. Let us now repeat this process with an undirected cyclic graph. Since node $i$ has a bidirectional connection to node $j$, each row of $A$ has two columns with a unit value. Thus, the adjacency matrix (now symmetric) becomes: 
# $$
# A = \left(\begin{matrix}
# 0 & 1 & 0 & \cdots & 0 & 1 \\
# 1 & 0 & 1 & \cdots & 0 & 0 \\
# 0 & 1 & 0 & \cdots & 0 & 0 \\
# 0 & 0 & 1 & \cdots & 0 & 0 \\
# \cdots \\
# 0 & 0 & 0 & \cdots & 0 & 1 \\
# 1 & 0 & 0 & \cdots & 1 & 0 \\
# \end{matrix}\right)
# $$
# 

# In[54]:


j = -1


@interact_manual(seed=(1, 100), eps=(0, 1.5), N=(5, 40))
def plot1d(seed, eps, N=15):
    global j
    np.random.seed(seed)
    # Define a NxN matrix
    A = np.zeros([N, N])
    for i in range(1, N):
        A[i, i - 1] = 1
    A[0, N - 1] = 1
    A = A + A.T

    x = np.linspace(0, 10, N)

    # Signal s
    noise = eps * np.random.randn(N)
    s = np.sin((x / x[-1]) * 2 * np.pi * 2.5) + noise

    j += 1
    Aj = np.linalg.matrix_power(A, j)
    new_s = Aj @ s

    print(Aj)
    plt.plot(x, s, "-", color="red")
    plt.plot(x, new_s, "-o")
    plt.title("Press button to apply $A$")
    plt.show()


# The result: instability. The signal $A^n s$ goes to infinity as the number of iterations grows without bound (i.e., $n\rightarrow\infty$). Later, when working with neural networks, we want to avoid weights that converge towards infinity or zero. 
# 
# This justifies the use of normalized adjacency matrices. The most common normalization is to premultiply $A$ by $D^{-1}$, where $D$ is the degree matrix. For our graph, all nodes have degree 2. Let us try again. We define a left normalization:
# $$
# A^* = D^{-1} A 
# $$
# Another popular normalization technique is the symmetric version of the preceding one: 
# $$
# A^* = D^{-1/2} A D^{-1/2}
# $$

# In[55]:


j = -1


@interact_manual(
    seed=(1, 100),
    eps=(0, 1.5),
    N=(5, 40),
    jincr=(1, 10),
    normalization=["left", "symmetric"],
)
def plot1d(seed, eps=0.1, N=15, normalization="left", jincr=1):
    global j
    np.random.seed(seed)
    # Define a NxN matrix
    A = np.zeros([N, N])
    for i in range(1, N):
        A[i, i - 1] = 1
    A[0, N - 1] = 1
    A = A + A.T
    D = np.sum(A, axis=1)  # works for all A
    Dinv = np.diag(1.0 / D)

    if normalization == "left":
        Dinv = np.diag(1.0 / D)
        A = Dinv @ A
        print("DinvSq @ A @ DinvSq= ", A)
    else:
        DinvSq = np.sqrt(Dinv)
        A = DinvSq @ A @ DinvSq

    x = np.linspace(0, 10, N)

    # Signal s
    noise = eps * np.random.randn(N)
    s = np.sin((x / x[-1]) * 2 * np.pi * 2.5) + noise
    print("mean(s) = ", np.mean(s))

    j += jincr
    Aj = np.linalg.matrix_power(A, j)
    new_s = Aj @ s
    print("mean(new_s) = ", np.mean(new_s))
    print("new_s= ", new_s)

    plt.plot(x, s, "-", color="red")
    plt.plot(x, new_s, "-o")
    plt.title("Press button to apply $A$")
    plt.show()


# One observes that after many repetitions of normalized (left or symmetric), $A$, the signal converges to a constant equal to the mean of the original signal: 
# $$
# \lim_{n\rightarrow\infty} s_{new} = \text{mean}(s) = \frac1N\sum_0^{n-1} s_i
# $$
# 
# From a theoretical point of view, if $s_{new}$ converges to a constant, it means that in the limit of $n\rightarrow\infty$, 
# $$
# (A^*)^n  s_{new} = (A^*)^{n-1} s_{new}
# $$
# which implies that 
# $$ A^* s_{new} = s_{new} $$
# In other words, $\lambda=1$ is an eigenvalue of the normalized adjacency matrix (corresonding to a bidirectional cyclic graph), either 
# $A^* = D^{-1} A$ or $A^* = D^{-1/2} A D^{-1/2}$. 
# 
# One can easily show that if a single eigenvalue is greater than 1, $s_{new} \rightarrow \infty$. Since that does not happen, the maximum eigenvalue must be unity.
# 
# We check this out by computing the eigenvalues of the normalized matrix (which must be real since the matrix is symmetric). One also notices that since $A$ is symmetric, both normalizations produce the same results. 
# 
# Exercise: Can you prove this? 

# In[56]:


@interact_manual(N=(5, 40), normalization=["left", "symmetric"])
def plot1d(N=15, normalization="left"):
    # Define a NxN matrix
    A = np.zeros([N, N])
    # cyclic linear chain with two connections per node
    for i in range(1, N):
        A[i, i - 1] = 1
    A[0, N - 1] = 1
    A = A + A.T
    
    D = np.sum(A, axis=1)  # works for all A
    Dinv = np.diag(1.0 / D)

    if normalization == "left":
        Dinv = np.diag(1.0 / D)
        A = Dinv @ A
    else:
        DinvSq = np.sqrt(Dinv)
        A = DinvSq @ A @ DinvSq

    print("A^*= ", A)
    evalue, evector = np.linalg.eig(A)
    print("\nSorted eigenvalues: ", np.sort(evalue))
    print(f"NOTE: the maximum eigenvalue = 1")


# ---
# ## Cyclic case, non-directed graph
# We now repeat the last few experiments with a linear graph (i.e., a chain), but non-periodic: the boundary points are not considered as a single point. 
# 
# ### Directed Graph
#  $A_{i+1,i}=1$, for $i=0,\cdots,N-2$. 
# 
# ### Undirected Graph
# $A_{i+1,i}$ and $A_{i,i+1}=1$ for $i=0,\cdots,N-2$. 
# 
# Let us apply the previous code to this case and see the effect of successive applications of $A$ on the signal. 
# 
# Undirected graphs lead to NaNs in the normalized matrices. 

# In[57]:


@interact_manual(
    N=(5, 20),
    normalization=["none", "left", "symmetric"],
    graph=["undirected", "directed"],
)
def plot1d(N=15, normalization="left", graph=["undirected"]):
    # Define a NxN matrix
    A = np.zeros([N, N])
    for i in range(1, N):
        A[i, i - 1] = 1

    if graph == "undirected":
        A = A + A.T

    D = np.sum(A, axis=1)  # works for all A
    print("D= ", D)

    if normalization == "left":
        Dinv = np.diag(1.0 / D)
        An = Dinv @ A
    elif normalization == "none":
        An = A
    else:
        Dinv = np.diag(1.0 / D)
        DinvSq = np.sqrt(Dinv)
        An = DinvSq @ A @ DinvSq

    print("A = ", A)
    print("An= ", An)
    evalue, evector = np.linalg.eig(An)
    print(np.sort(evalue))


# When the graph is directed, the first row of $A$ is zero, which leads to a zero eigenvalue, and the matrix is not invertible. 
# 
# With no normalization, the maximum eigenvalue magnitude is greater than unity, which is not desirable for an iterative process. However, with both left and symmetric normalization, the eigenvalues are still greater than unity. 
# 
# This leads to the idea of iterating with a matrix whose eigenvalues have better properties. This matrix is the Laplacian: 
# $$
# L = D - A
# $$
# whose rows sum to zero. One easily sees that this represents a first or second order approximation to the second derivative is the nodes are equally spaced. The Laplacian measure curvature. 
# 
# Let us compute the eigenvalues of $L$, and its normalized version: 
# \begin{align}
# L^* &= D^{-1} L \\
# L^* &= D^{-1/2} L D^{-1/2}
# \end{align}
# where $D$ is still defined as the degree matrix of $A$. 
# 

# In[58]:


np.diag([1, 2, 3])


# In[59]:


@interact_manual(
    N=(5, 20),
    normalization=["none", "left", "symmetric"],
    graph=["undirected", "directed"],
)
def plot1d(N=15, normalization="none", graph=["undirected"]):
    # Define a NxN matrix
    A = np.zeros([N, N])
    for i in range(1, N):
        A[i, i - 1] = 1

    if graph == "undirected":
        A = A + A.T

    diagD = np.sum(A, axis=1)  # works for all A
    Dinv = np.diag(1 / diagD)
    D = np.diag(diagD)
    # print("D= ", D)
    # print("Dinv= ", Dinv)
    # print("diag(D) ", np.diag(D))

    # print("D= ", D)
    # print("A= ", A)
    L = D - A
    # print("L= ", L)

    # We will call L (normalized or not, the filter)
    H = L

    if normalization == "left":
        Hn = Dinv @ H  # normalized
    elif normalization == "none":
        Hn = L
    else:
        DinvSq = np.sqrt(Dinv)
        Hn = DinvSq @ H @ DinvSq

    print("A= ", A)
    print("Dinv= ", Dinv)
    print("(Dinv@D)= ", (Dinv @ np.diag(D)))
    print("norm(Dinv@D-np.eye(N))= ", np.linalg.norm(Dinv @ np.diag(D) - np.eye(N)))
    print("L=H = ", L)
    print("Hn= ", Hn)
    evalue, evector = np.linalg.eig(Hn)
    print("Sorted eigenvalues: ", np.sort(evalue))


# Everything works as expected for undirected graphs. The two normalizations (left and symmetric) produce real eigenvalues in the range $[0,2]$. The unormalized Laplacian has unbounded eigenvalues. $\lambda=1$ is another eigenvalue, independent of the number of nodes, $N$. 
# 
# Clearly, the iteration 
# $$
# s^{n+1} = A^n s^n
# $$
# diverges as $n\rightarrow\infty$. 

# From linear algebra, any symmetric matrix $L$ can be expressed as
# $$
# L = U^{-1} \Lambda U
# $$
# where the *columns* of $U^{-1}$ are the eigenvectors of $A$ and $\Lambda$ is a diagonal matrix with the eigenvalues of $L$. This is easily seen by multiplying both sides by $U^{-1}$: 
# $$
# L \, U^{-1}  = U^{-1} \Lambda 
# $$
# In component notation: 
# \begin{align}
# \sum_j L_{ij} U^{-1}_{jk} &= \sum_j U^{-1}_{ij} \Lambda_{jk} \\
#     &= \sum_j U^{-1}_{ij} \delta_{jk} \Lambda_{jk} \\
#     &= U^{-1}_{ik} \lambda_k
# \end{align}
# where $\Lambda\in\Re{N\times N}$ is a diagonal matrix. If the eigenvectors are normalized, $U^{-1} = U^T$, which is a normal matrix (i.e., the eigenvectors have unit length, and are orthogonal). We made the implicit asumptions that all eigenvalues are different. Otherwise, one has to resort to the Jordan normal form, which is out of scope.
# The LHS (left-hand side) of the last equation represents $L U^{-1}_k$, where $U^{-1}_k$ is the $k^{th}$ column of $U^{-1}$. Therefore, $U^{-1}_k$ is an eigenfunction of $L$ with eigenvalue $\lambda_k$. Again, we assume all eigenvalues are different. If an eigenvalue $\lambda_k$ has multiplicity $m_k$, the corresponding eigenvectors form a subspace $U^{-1}_k \in \Re{m_k\times m_k}$. 

# ## Eigenvectors of various operators on the cyclic and non-cyclic chain
# We write a program to plot the eigenvectors of $A$, normalized $A$ (left and symmetric), and $L$ (normalized or not). 
# 
# Consider a matrix that has a unit eigenvalue $\lambda = 1$ with associated eigenvector $v$. Assume that $\lambda=1$ is the largest eigenvector. Starting from a random signal $s$, we know that it can be expressed as a linear combination of the eigenvectors of $A$. Since the eigenvectors form a basis of $A$, this expansion is unique. Thus: 
# $$
# s = \sum_k a_k v_k
# $$
# where $v_k$, $k=0,1,\cdots,N-1$ is the $k^{th}$ eigenvectors and $\lambda_k$ is the $k^{th}$ eigenvalue. Apply $A$ to both sides: 
# $$
# A s = \sum_k a_k A_k v_k = \sum_k a_k \lambda_k v_k
# $$
# Therefore, applying $A$ multiple times to both sides: 
# $$
# A^n s = \sum_k a_k \lambda^n_k v_k
# $$
# If we assume that $\lambda_{max}$ is the eigenvalue of maximum magnitude, we reexpress the equation above as
# \begin{align}
# A^n s &= \lambda_{max}^n \sum_k a_k \left(\frac{\lambda_k}{\lambda_{max}}\right)^n v_k 
# \end{align}
# As $n\rightarrow\infty$, the term with the largest eigenvalue in magnitude will dominate the expression. Therefore, 
# $$
# A^n s \rightarrow a_{k^*} \lambda_{k^*}^n v_{k^*}
# $$ 
# for very large $n$. Setting $\lambda_{max}=1$, we find that
# $$
# A^n s \rightarrow a_{k^*} v_{k^*}
# $$
# which is finite. This result holds for any matrix $A$. 
# 
# We demonstrated this earlier in the case when $A$ is the shift operator of a linear undirected graph, that $v_{k^*} \rightarrow \text{mean}(s}$. In this case, the constant function is an eigenvector that corresponds to the unity eigenvalue. 
# 
# <font color='red'>
# (NEED MORE DEVELOPMENT). I AM CONFUSED. WHAT INFORMATION AM I TRYING TO IMPART? 
# </font>

# In[60]:


# I would like which_eig to plot to have a maximum of N
which_eig_slider = widgets.IntSlider(min=0, max=100, value=0)


@interact(N=(5, 100), which_eig=(0, 100), graph=["undirected", "directed"])
def plot1d(N=10, which_eig=which_eig_slider, graph=["undirected"]):
    # poor programming but allows me to change the slider position
    # global which_eig_slider

    # Define a NxN matrix
    # if which_eig > N:
    #     which_eig = N - 1 # count from 0
    # which_eig_slider.value = N - 1
    # print(which_eig_slider)
    # print("which_eig: ", which_eig)

    A = np.zeros([N, N])
    for i in range(1, N):
        A[i, i - 1] = 1

    if graph == "undirected":
        A = A + A.T

    diagD = np.sum(A, axis=1)  # works for all A
    # The undirected version has an Inf in Dinv
    Dinv = np.diag(1 / diagD)
    D = np.diag(diagD)
    L = D - A
    # We will call L (normalized or not, the filter)
    H = L

    H_dict = {}
    eigval_dict = {}
    eigvec_dict = {}
    H_dict["none"] = L

    # Next two matrices have NaNs in the undirected graph case
    H_dict["left"] = Dinv @ H  # normalized
    DinvSq = np.sqrt(Dinv)
    H_dict["symmetric"] = DinvSq @ H @ DinvSq

    if graph == "directed":
        # Remove keys (works even when key is not in dict)
        H_dict.pop("left", None)
        H_dict.pop("symmetric", None)

    # Draw three columns: no normalization, left, and symmetric
    # Draw 5 eigenvectors for first five eigenvalues, sorted by magnitude
    # Below the eigenvectors, plot the first 10 eigenvalues, sorted by magnitude

    nrows = 3
    ncols = 3
    # rows and cols are used to access axes array elements
    row_eigf, row_eigv = 0, 1
    cols_dict = {"none": 0, "left": 1, "symmetric": 2}
    pos_none_eig = 2, 1
    pos_none_tot_var = 2, 0

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6))

    for k, v in H_dict.items():
        eigval_dict[k], eigvec_dict[k] = np.linalg.eig(v)
        arg = np.argsort(eigval_dict[k])
        eigval_dict[k] = eigval_dict[k][arg]
        eigvec_dict[k] = eigvec_dict[k][:, arg]

    for k in H_dict.keys():
        ax = axes[row_eigf, cols_dict[k]]
        for i in range(0, 5):
            ax.plot(eigvec_dict[k][:, i], "-o", label=f"$\lambda_{i}$")
        ax.set_xlabel("k")
        ax.set_ylabel("v_k")
        ax.legend(framealpha=0.5)

        ax = axes[row_eigv, cols_dict[k]]
        ax.plot(eigval_dict[k], "-o", color="black")
        ax.set_ylim(0, 5)
        ax.set_xlabel("k")
        ax.set_ylabel("$\lambda_k$")
        ax.grid(True)

    ax = axes[pos_none_eig]  # [0], pos_none_eig[1]]
    ax.set_ylim(-0.2, 0.2)
    ax.grid(True)
    ax.set_title("Single Eigenvector, no normalization")

    try:
        eigvec = eigvec_dict["none"][:, which_eig]
    except:
        print(f"which_eig must be < N! Reset value to ${N-1}$")
        which_eig = N - 1
        eigvec = eigvec_dict["none"][:, which_eig]

    # print("norm(eigvec): ", np.linalg.norm(eigvec, 2))
    # eig_string = "$\lambda_%s$" % which_eig
    # print("eig_string: ", eig_string)
    ax.plot(eigvec, "-o", color="black", label=f"$\lambda_{which_eig}$")
    ax = axes[row_eigv, cols_dict["none"]]
    ax.plot(which_eig, eigval_dict["none"][which_eig], "o", ms=10, color="red")
    ax.set_title(f"Eigenvalues $\lambda_k$")

    ax = axes[pos_none_tot_var]

    def tot_var(L, v):
        """
        Calculate the total variation: \sum_i (s[i]-s[j])^2
        where s is a signal, which could be an eigenvector of $A$.
        The function is inefficient but will work on general graphs
        """
        total_variat = 0
        for i in range(N):
            for j in range(N):
                if abs(A[i, j]) > 0.01:
                    total_variat += (v[i] - v[j]) ** 2
        return total_variat

    # Calculate total variation for all eigenvalues, and for 'none' and 'symmetric' normaliz
    totvar = []
    for i in range(N):
        v = eigvec_dict["none"][:, i]
        totvar.append(tot_var(L, v))

    ax.plot(totvar, "-o", color="black")
    ax.plot(which_eig, totvar[which_eig], "o", ms=10, color="red")
    ax.grid(True)
    ax.set_title("Total Variation, $L$, no normalization")

    # Plot curve

    for k in H_dict.keys():
        ax = axes[0, cols_dict[k]]
        ax.set_title(k + " normalization")
        ax = axes[1, cols_dict[k]]
        ax.set_title(k + " normalization")

    plt.suptitle(
        "Eigenvectors and eigenvalues for $L$ (left), $D^{-1}L$ (middle), $D^{-1/2}LD^{-1/2}$ (right)",
        fontsize=16,
    )

    plt.tight_layout()
    # plt.show()


# ## Findings
# * The spectrum (eigenvalues) range is independent of $N$. 
# * The eigenvector of the unnormalized Laplacian has a fixed range for most $N$. It always has unit $l_2$ norm. 
# * The total variation $\sum_{i,j} A_{i,j} (v_i-v_j)^2$ increases with the eigenvalue. Here, $v_j$ is the eigenvector $j$ that corresponds to eigenvalue $\lambda_i$. 

# ## Code complexity
# The plotting code above is getting complicated. It is therefore time to simplify the code by refactoring common operations. Different plots have different number of subplots, and each subplot draws one or more curves. They require an axis (`ax`) and dependent and independent variables, either one or a group. Therefore, smaller routines dedicated to drawing a single subplot would be useful. 
# Furthermore, there is a need to create routines to create different kinds of matrices, alogn with their eigenvalues, and eigenvectors. Of course, the `Networkx` graph already does this, but doing it ourselves is good coding practice. 

# # Code refactoring

# ## Refactored version of previous function
# * The new functions are located in the file `spectral_function_library.py` in the main folder.
# * Pay attention to the first two lines of this notebook: 
#     * %load_ext autoreload
#     * %autoreload 2
#     
# These two lines ensure that modules are automatically reloaded when changed on disk. 

# In[61]:


# I would like which_eig to plot to have a maximum of N
which_eig_slider = widgets.IntSlider(min=0, max=100, value=0)


@interact(N=(5, 100), which_eig=(0, 100), graph=["undirected", "directed"])
def plot1d(N=10, which_eig=which_eig_slider, graph=["undirected"]):
    A = spec_lib.linear_acyclic_chain(N, graph)
    D = spec_lib.degree_matrix(A)
    H = L = D - A  # H stands for filter

    norms = ["none", "left", "symmetric"]
    H_dict = {k: spec_lib.normalized_matrix(L, D, k) for k in norms}

    eigval_dict = {}
    eigvec_dict = {}

    if graph == "directed":
        # Remove keys (works even when key is not in dict)
        H_dict.pop("left", None)
        H_dict.pop("symmetric", None)

    # Draw three columns: no normalization, left, and symmetric
    # Draw 5 eigenvectors for first five eigenvalues, sorted by magnitude
    # Below the eigenvectors, plot the first 10 eigenvalues, sorted by magnitude

    for k, v in H_dict.items():
        eigval_dict[k], eigvec_dict[k] = np.linalg.eig(v)
        arg = np.argsort(eigval_dict[k])
        eigval_dict[k] = eigval_dict[k][arg]
        eigvec_dict[k] = eigvec_dict[k][:, arg]
        
    # Total variation (based on Laplaci
    totvar = [spec_lib.tot_var(A, eigvec_dict["none"][:, i]) for i in range(N)]
    
    """
    Six plots of eigenvalues and eigenvectors of L and two normalized versions, 
    left and symmetric normalization by Dinv and sqrt(Dinv). 
    Also plotted: 
       1) total variation of the signal a a function of eigenvalue
       2) the k^{th} eigenvector of the Laplacian. The chosen eigenvector is controlled
        with a slider bar (which_eigen)
    """
    spec_lib.plot_data1(H_dict, eigval_dict, eigvec_dict, totvar, which_eig)


# # Example of a simple embedding calculation using a spectral approach. 
# * We will not be concerned with efficiency
# * We will linearize any nonlinearities. 

# ---
# ## Time to think about node embeddings and Neural networks
# The simplest algorithm would be to iterate the following: 
# $$
# H^{n+1}_{i,l} = \sum_{j\in\cal{N}(v_j)\cup v_j} (I_{i,j} +A_{i,j}) H^{n+1}_{j,k} W_{k,l} 
# $H^{i,l}$ to feature $l$ on graph node $i$. Feature $l$ is also called the value of element $l$ of the node's embedding. The number of features on a node need not be equal to the number of embeddings. 
# 
# The subscript refers to the iteration number. In practice, a nonlinear funciton is applied between iterations. Thus, 
# $$
# H^{n+1} = \sigma((I+A) H^{n} W) 
# $$
# where $W$ is a weight matrix that will be determined by optimizing an appropriate cost function. 
# 
# Let us link together multiple iterations: 
# \begin{align}
# H^{1} &= \sigma((I+A) H^0 W^0)  \\
# H^{2} &= \sigma((I+A) H^1 W^1) \\
# \cdots &=  \cdots
# \end{align}
# Note that $w^n$ could be independent of $n$, which reminds us of recursion, or have different values for each iteration, which reminds us of a multi-stage convolution network. The weight matrix $W^n \in \Re{d^{n}\times d^{n+1}}$ where $d^n$ is the size of the embedding vector at iteration $n$. $H^0$ is usually chosen to be the existing feature matrix of the graph. 
# 
# Now let us remove the nonlinearity. This gives the linear algorithm :
# \begin{align}
# H^{1} &= (I+A) H^0 W^0 \\
# H^{2} &= (I+A) H^1 W^1 \\
#       &= (I+A)^2 W^0 W^1 \\
# \cdots &=  \cdots
# \end{align}
# Since $W^0$ and $W^1$ were computed by the algorithm being developed, their product can be replaced by a single matrix $W$. After $n$ iterations,  we have: 
# $$
# H^n = (I+A)^n H^0 W^0
# $$
# We will actually replace $I+A$ by its symmetrized normalized form
# $$
# \tilde{A} = (I+A)^{-1/2} (I+A) (I+A)^{-1/2}
# $$
# We will use PyTorch to implement one layer of a GNN, namely
# $$
# H^{n+1} = \sigma(\tilde{A} H^{n} W) 
# $$
# where we will assume an embedding in $\Re{d}$, $W\in\Re{d^n\times d^{n+1}}$. 
# 

# # def generate_graph_from_adjacency_matrix_1(G, N, seed):
#     """
#     Arguments
#     N: number of nodes
#     """
#     np.random.seed(seed)
# 
#     # Convert to np.ndArray
#     A = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
#     nx.linalg
#     # print("Adj: ", A, "\n", A.shape, "\n", type(A))
# 
#     # Different matrices
#     D = np.sum(A, axis=0)
#     D = np.diag(D)
#     I = np.eye(N)
#     Sqinv = np.sqrt(np.inv(I+A))
#     An = Sqinv @ (A+I) @ Sqinv
# 
#     print(I)
#     L = D - A
#     Ln = nx.normalized_laplacian_matrix(G)   # Is it symmetric?
#     Ln = Ln.toarray()  # from sparse array to ndarray
# 
#     # Create a signal on the graph. We will choose a sine wave. 
#     def sig(ivec, freq):
#         return np.sin(2.*np.pi*freq*ivec / ivec[-1])
#                    
#     ivec = np.asarray(list(range(D.shape[0])))
#     s = sig(ivec, freq=2)
#     
#     ### WHAT NEXT?
#                    
#                    
#     matrix = ["A", "D", "L", "invD", "invDA", "invDL", "invDinvD", "Ln"]
#     matrices = [A, D, L, invD, invDA, invDL, invDLinvD, Ln]
# 
#     # Eigenvalues
#     fig, axes = plt.subplots(3, 3, figsize=(10, 8))
#     axes = axes.reshape(-1)
#     fig.suptitle("Eigenvalues of various matrices")
#     for i, m in enumerate(matrices):
#         ax = axes[i]
#         eigs = np.linalg.eigvals(m)
#         eigs = np.sort(eigs)[::-1]
#         ax.set_title(matrix[i])
#         ax.grid(True)
#         ax.plot(eigs, "-o")
#     for i in range(i + 2, axes.shape[-1]):
#         axes[i].axis("off")
#     plt.tight_layout()
#     plt.show()

# In[62]:


prob_slider = widgets.FloatSlider(min=0, max=1, step=0.1, value=0.5)
node_slider = widgets.IntSlider(min=3, max=30, step=1, value=10)
nb_neigh_slider = widgets.IntSlider(min=1, max=10, step=1, value=4)
nb_edges_per_node_slider = widgets.IntSlider(min=1, max=20, step=2, value=5)
seed_slider = widgets.IntSlider(int=1, max=50, step=1, value=25)
graph_type = ["connected_watts_strogatz", "powerlaw_cluster_graph"]

@interact(
    nb_nodes=node_slider,
    prob=prob_slider,
    nb_neigh=nb_neigh_slider,
    nb_edges_per_node=nb_edges_per_node_slider,
    seed=seed_slider,
    graph_type=graph_type,
    # directed=True,
)
def drawGraph(nb_nodes, nb_neigh, prob, seed, nb_edges_per_node, graph_type):
    if graph_type == "connected_watts_strogatz":
        nb_edges_per_node_slider.style.handle_color = 'red'
        nb_neigh_slider.style.handle_color = 'black'
        nb_tries = 20
        edge_prob = prob
        G = nx.connected_watts_strogatz_graph(
            nb_nodes, nb_neigh, edge_prob, nb_tries, seed
        )
    elif graph_type == "powerlaw_cluster_graph":
        nb_neigh_slider.style.handle_color = 'red'
        nb_edges_per_node_slider.style.handle_color = 'black'
        add_tri_prob = prob
        if nb_edges_per_node >= nb_nodes:
            nb_edges_per_node = nb_nodes - 1
        G = nx.powerlaw_cluster_graph(nb_nodes, nb_edges_per_node, add_tri_prob, seed)

    generate_graph_from_adjacency_matrix_1(G, nb_nodes, seed)


# In[ ]:





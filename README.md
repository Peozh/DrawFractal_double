# DrawFractal_double
double precision calculation using CUDA library
![image](https://github.com/Peozh/DrawFractal_double/assets/35257391/a469bc3c-1439-45ef-8413-113065ef513a)
- max widget size setted to same as FHD (1080p, 1920 x 1080)
- device memory & pinned host memory allocated with above fixed max size to avoid reallocations
- much slower than float openGL fragment shader version when high iteration pixels are visible
- more cuda core -> faster texture generation
- openGL's PBO(PixelBuffer Object, data async transfer between host-device) may be appliable for optimization

## key bindings
 - mouse left button 2-dimensional drag : change complex constant
 - alt + mouse left button vertical drag : change real exponent
 - mouse right click : toggle log_expression

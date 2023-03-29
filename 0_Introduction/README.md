vectorAdd:
``` 
一个非常基本的示例，实现了逐元素的矢量添加
API:
<<<grid, block>>>: 调用内核函数，分配线程，这里涉及到grid和block维度的计算
cudaError_t: 发生错误时返回的错误类型，可通过cudaGetErrorString(err)获取对应的字符串描述
cudaMalloc/cudaFree: 分配/释放device内存空间
cudaMemcpy: 内存拷贝，host->device / device->host
cudaGetLastError: 返回运行时调用中的最后一个错误
```
vectorAddUniMem:
``` 
使用共享内存来实现逐元素的矢量添加
API:
cudaMallocManaged: 用于分配统一内存的函数，统一内存是一种可以被CPU或GPU访问的内存。这个函数可以简化编程，
因为它可以自动地在CPU和GPU之间迁移内存，而不需要显式地使用cudaMemcpy函数
cudaDeviceSynchronize:
用于同步CPU和GPU的函数，它会阻塞CPU端的线程，直到GPU端完成所有之前请求的任务，包括kernel函数和内存拷贝等
可以用与检测GPU端的错误，或者测量GPU端的执行时间
```
matrixMul:
``` 
此示例实现矩阵乘法，它的编写是为了阐明各种CUDA编程原理，而不是为了为矩阵乘法提供最具性能的通用内核。
API：

```
asyyncAPI
```
此示例说明了CUDA事件在GPU计时以及CPU和GPU执行重叠时的使用情况。事件被插入到CUDA调用流中。
由于CUDA流调用是异步的，CPU可以在GPU执行时执行计算（包括主机和设备之间的DMA内存复制）。
CPU可以查询CUDA事件以确定GPU是否已完成任务

API:
cudaDeviceProp:
cudaDeviceProp数据类型是针对函式cudaGetDeviceProperties定义的
cudaGetDeviceProperties函数的功能是取得支持GPU计算装置的相关属性，
比如支持CUDA版本号装置的名称、内存的大小、最大的thread数目、执行单元的频率等


```
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
使用统一内存来实现逐元素的矢量添加
API:
cudaMallocManaged: 用于分配统一内存的函数，统一内存是一种可以被CPU或GPU访问的内存。这个函数可以简化编程，
因为它可以自动地在CPU和GPU之间迁移内存，而不需要显式地使用cudaMemcpy函数
cudaDeviceSynchronize:
用于同步CPU和GPU的函数，它会阻塞CPU端的线程，直到GPU端完成所有之前请求的任务，包括kernel函数和内存拷贝等
可以用与检测GPU端的错误，或者测量GPU端的执行时间
```
matrixMul:
``` 
此示例实现基于全局内存的矩阵乘法的简单实现
每个线程读取矩阵 A 的一行和矩阵 B 的一列，并计算输出矩阵 C 的相应元素
dimsA(A.x, A.y, 1) 则矩阵A为 (A.y, A.x)
dimsB(B.x, B.y, 1) 则矩阵B为 (B.y, B.x)
因此： A.x == B.y
API：
cudaProfilerStart / cudaProfilerStop:
CUDA运行时API函数，用于启用/关闭当前上下文的活动分析工具的分析数据收集。
如果分析已经启用，那么cudaProfilerStart没有任何效果。可以用于通过只对选择的代码片段进行分析来控制分析的粒度。
cudaStreamCreateWithFlags：
用于创建一个异步流。第二个参数flags可以指定流的属性。有两种选项：
cudaStreamDefault表示创建一个阻塞流，即该流会被默认流阻塞。
cudaStreamNonBlocking表示创建一个非阻塞流，即该流不会被默认流阻塞。
cudaStreamSynchronize：
用于使主机等待指定流中的所有操作完成。该
函数会阻塞主机线程，直到流中的所有操作都执行完毕。如果流中有任何错误，该函数会返回相应的错误码
cudaMemcpyAsync：
用于异步地将数据从主机内存复制到设备内存，或从设备内存复制到主机内存，
该函数需要指定复制的方向、大小和流。cudaMemcpyAsync是相对于主机而言异步的，即该函数可能在复制完成之前就返回。
因此，该函数只能在锁页主机内存上工作，如果传入可分页的主机内存指针，会返回错误。如果指定一个非零的流参数，该函数可以与其他流中的操作重叠
```
MatrixMulSharedMem:  
```  
使用共享内存的方式来计算矩阵乘法。目的是为了减少全局内存的访问次数，同时由于共享内存基于片内，这样做会大大提升内存访问速度
API:
cudaEventCreate / cudaEventDestroy:
用于创建/销毁一个事件对象。事件对象可以用来标记流中的某个特定点，或者测量两个事件之间的时间差。
cudaEventRecord / cudaEventSynchronize
用于在指定的流中记录一个事件。如果流是非零的，事件会在流中所有前面的操作完成后被记录；否则，事件会在CUDA上下文中所有前面的操作完成后被记录。
由于操作是异步的，必须使用cudaEventQuery或cudaEventSynchronize来确定事件是否已经被记录。
如果cudaEventRecord之前已经在事件上调用过，那么这次调用会覆盖事件中的任何现有状态。
任何后续检查事件状态的调用都只会检查这次cudaEventRecord的完成情况
cudaEventElapsedTime：
用于计算两个事件之间的经过时间（以毫秒为单位，分辨率约为0.5微秒）
如果两个事件中的任何一个是在非空流中最后记录的，那么得到的时间可能会比预期的大（即使两个事件使用了相同的流句柄）。
这是因为cudaEventRecord操作是异步的，不能保证测量的延迟实际上只是在两个事件之间。
在两个测量事件之间，可能有其他不同的流操作执行，从而显著地改变了时间
__shared__ 
CUDA C/C++中的关键字，用于声明一个共享内存变量。共享内存是一种位于芯片上的快速内存，访问速度比全局内存快得多。
共享内存是按线程块分配的，所以同一个线程块中的所有线程都可以访问同一块共享内存。
线程可以访问由同一线程块中的其他线程从全局内存加载到共享内存中的数据。需要结合__syncthreads()使用。避免发生错误
#pragma unroll: 放在一个循环之前，并且只作用于这个循环
```
simpleAssert
```  
在device code中使用assert
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


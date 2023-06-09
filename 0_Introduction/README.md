helloworld:
```  
打印helloWorld
API:
__host__: host端调用，decice端执行
<<<grid, block>>>: 调用内核函数，分配线程，这里涉及到grid和block维度的计算
device可以打印信息，但只能使用printf而不能使用cout
计算thread编号：
    blockId = blockIdx.z *  gridDim.y *  gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
    blockSize = blockDim.z * blockDim.y * blockDim.x;
    threadId = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * gridDim.x + threadIdx.x;
    thread = blockId * blockSize + threadId;
```
vectorAdd:
``` 
逐元素的矢量添加
API:
cudaError_t: 发生错误时返回的错误类型，可通过cudaGetErrorString(err)获取对应的字符串描述
cudaMalloc/cudaFree: 分配/释放device内存空间
cudaMemcpy: 内存拷贝，host->device / device->host
cudaGetLastError: 返回运行时调用中的最后一个错误
```
vectorAddUniMem:
``` 
使用统一内存来实现逐元素的矢量添加
API:
cudaMallocManaged: 
用于分配统一内存的函数，统一内存是一种可以被CPU或GPU访问的内存。这个函数可以简化编程，
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
    用于使主机等待指定流中的所有操作完成。该函数会阻塞主机线程，直到流中的所有操作都执行完毕。如果流中有任何错误，该函数会返回相应的错误码
cudaMemcpyAsync：
    用于异步地将数据从主机内存复制到设备内存，或从设备内存复制到主机内存
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
    用于在指定的流中记录一个事件。如果流是非零的，事件会在流中所有前面的操作完成后被记录；
    否则，事件会在CUDA上下文中所有前面的操作完成后被记录。
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
#pragma unroll: 
    放在一个循环之前，并且只作用于这个循环，展开循环
```
simpleAssert
```  
在device code中使用assert
```
simpleAttributes:
```  
通过设置流属性来影响L2缓存
API:
cudaAccessPolicyWindow:
    一个结构体，指定了一个窗口的访问策略，一个窗口是一个连续的内存区域，从base_ptr开始，到base_ptr + num_bytes结束
    窗口被划分为多个段，每个段被分配一个访问策略，要么是hitProp，要么是missProp。
    hitProp和missProp是两种不同的缓存属性，分别用于命中和未命中的访问。hitRatio指定了被分配hitProp的段占窗口的百分比
StopWatchInterface:
    一个用于计时的类，它定义了一些虚函数，如start(), stop() 等用于控制和获取计时器的状态和时间
cudaCtxResetPersistingL2Cache：
    CUDA驱动API函数，它用于将所有持久的L2缓存行重置为正常状态
atmoicExch、atomicAdd:
    CUDA原子操作函数，它用于无条件地替换 / Add一个全局或共享内存中的32位或64位字，并返回原来的值
```
mallocPitch:
```  
使用cudaMallocPitch()分配2D数组，确保分配被适当地填充以满足设备内存访问中描述的对齐要求，
从而确保在访问行地址或在 2D 数组和其他区域设备内存之间执行复制时获得最佳性能
API：
cudaError_t cudaMallocPitch( void** devPtr，size_t* pitch，size_t widthInBytes，size_t height ):
    向设备分配至少widthInBytes * height字节的线性存储器，并以*devPtr的形式返回指向所分配存储器的指针。
    返回的间距（或步幅）必须用于访问数组元素,cudaMallocPitch在分配内存时，每行会多分配一些字节，
    以保证widthofx*sizeof(元素)+多分配的字节是256的倍数(对齐)
```
malloc3D:
```  
使用cudaMalloc3D()分配2D数组，确保分配被适当地填充以满足设备内存访问中描述的对齐要求，
从而确保在访问行地址或在 2D 数组和其他区域设备内存之间执行复制时获得最佳性能
API:
cudaExtent:
    结构体，用于描述3D Array和3D线性内存在三个维度上的尺寸,cudaExtent可以用make_cudaExtent函数创建
cudaPitchedPtr:
    结构体，用于描述分配给GPU的线性内存的指针、间距、宽度和高度, cudaPitchedPtr可以用make_cudaPitchedPtr函数创建
cudaMalloc3D:
    用于在GPU中分配三维数组的内存。它可以保证分配的内存是对齐的，以提高内存访问的效率
```
cudaMemory:
```  
Runtime API访问全局变量的各种方法
API:
__constant__ 
    声明内存为常量内存
__device__:
    在设备（device）中声明一个全局变量用__device__关键字修饰
cudaGetSymbolAddress:
    用于检索指向为全局内存空间中声明的变量分配的内存的地址
cudaGetSymbolSize:
    获得分配内存的大小
cudaMemcpyToSymbol:
    用于将数据从主机内存复制到设备内存中的全局或常量变量
```
asyyncAPI:
```
说明CUDA事件在GPU计时以及CPU和GPU执行重叠时的使用情况。事件被插入到CUDA调用流中。
由于CUDA流调用是异步的，CPU可以在GPU执行时执行计算（包括主机和设备之间的DMA内存复制）。
CPU可以查询CUDA事件以确定GPU是否已完成任务
API:
cudaDeviceProp:
    cudaDeviceProp数据类型是针对函式cudaGetDeviceProperties定义的
    cudaGetDeviceProperties函数的功能是取得支持GPU计算装置的相关属性，
    比如支持CUDA版本号装置的名称、内存的大小、最大的thread数目、执行单元的频率等
cudaMallocHost:
    用于分配大小为 size 字节的主机内存，该内存是页锁定的，可以被设备访问
```
C++11_cuda:
``` 
演示了CUDA中对C++11特性的支持。它扫描输入文本文件并打印x、y、z、w个字符的出现次数。
Thrust Lib:
C++并行算法库，将并行算法引入C++标准库。Thrust的高级接口极大地提高了程序员的生产力，同时实现了GPU和多核CPU之间的性能可移植性。
它建立在已建立的并行编程框架（如CUDA、TBB和OpenMP）之上。它还提供了一些与C++标准库中类似的通用工具。
```
clock:
``` 
这个例子展示了如何使用时钟函数来精确地测量内核线程块的性能。
API:
这个代码主要是展示随着设定的block的增加，并不是block越多时间越快，反而超过一定的限制，会出现明显的传输拥堵，latency越来越难以隐藏。
```
conCurKernels:
```
演示了使用CUDA流在GPU设备上并发执行几个内核。
说明了如何使用新的cudaStreamWaitEvent函数在CUDA流之间引入依赖关系。
API:
cudaEventCreateWithFlags:
   用于创建一个 CUDA 事件，并指定一些标志
cudaStreamWaitEvent:
   用于将一个CUDA流等待一个CUDA事件的触发
   使用 cudaStreamWaitEvent 函数等待一个 CUDA 事件可以使得当前流在该事件触发之后才继续执行后续操作，
   从而可以实现异步执行和流并发。使用 cudaStreamWaitEvent 函数等待的 CUDA 事件必须使用 cudaEventRecord 函数在某个 CUDA 流中记录，
   否则会出现未定义行为。
```
UnifiedMemoryStreams
```

API:
cudaDeviceSynchronize():
    
```

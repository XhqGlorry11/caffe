#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
// CPU模式下，通过调用C语言的malloc函数分配内存
inline void CaffeMallocHost(void** ptr, size_t size) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    return;
  }
#endif
  *ptr = malloc(size);
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

// CPU模式下，通过调用C语言的free函数释放内存
inline void CaffeFreeHost(void* ptr) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
  free(ptr);
}

/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
// 在主机(Host/CPU)和设备(Device/GPU)之间管理内存分配和数据同步，封装CPU和GPU之间数据交互操作
class SyncedMemory {
 public:
// 默认构造函数，简单初始化，数据状态置为UNINITIALIZED
  SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false), gpu_device_(-1) {}
// 带size参数的显示构造函数，并未分配内存，数据状态置为UNINITIALIZED
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false), gpu_device_(-1) {}
// 析构函数，CPU模式下，当cpu_ptr_非空并且own_cpu_data_为true时，仅会调用CaffeFreeHost函数释放内存
  ~SyncedMemory();
// 获取CPU数据指针,数据不可更改，内部会调用to_cpu函数，在CPU模式下，数据状态为HEAD_AT_CPU，在GPU模式下，数据状态置为SYNCED
  const void* cpu_data();
// 调用CaffeFreeHost释放内存，如果own_cpu_data_为非空，则调用CaffeFreeHost释放内存，并修改CPU数据指针使其指向data，并置own_cpu_data_为false，数据状态置为HEAD_AT_CPU
  void set_cpu_data(void* data);
// 获取GPU数据指针，数据不可更改，在GPU模式下，数据状态为HEAD_AT_GPU，在CPU模式下，数据状态置为SYNCED
  const void* gpu_data();
// 在GPU模式下，内部会调用to_gpu函数，如果own_gpu_data_为非空，调用cudaFree释放显存，并修改GPU数据指针使其指向data,并置own_gpu_data_为false，在GPU模式下，数据状态置为HEAD_AT_GPU
  void set_gpu_data(void* data);
// 获取CPU数据指针，数据可更改，内部会调用to_cpu函数，数据状态置为HEAD_AT_CPU
  void* mutable_cpu_data();
// 获取GPU数据指针，数据可更改，在GPU模式下，内部会调用to_gpu函数，数据状态置为HEAD_AT_GPU
  void* mutable_gpu_data();
// SyncedHead为枚举类型，数据存放的位置，包括四种数据状态，依次为未初始化、在CPU、在GPU、已同步
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
// 返回数据状态，即数据存放的位置
  SyncedHead head() { return head_; }
// 返回数据大小(字节)
  size_t size() { return size_; }

#ifndef CPU_ONLY
// 异步推送数据从CPU到GPU，并置数据状态为SYNCED
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
// 把数据存放到CPU上，
// 如果数据状态为UNINITIALIZED，则调用CaffeMallocHost分配内存，并初始化数据内容为0，置own_cpu_data_为true，置数据状态为HEAD_AT_CPU，
// 如果数据状态为HEAD_AT_GPU，如果在GPU模式下，如果cpu_ptr_为空，则调用CaffeMallocHost分配内存，并置own_cpu_data_为true，然后则将显存数据拷贝到内存(数据同步)，并将数据状态置为SYNCED
 // 其它数据状态不作任何操作
  void to_cpu();
// 把数据存放到GPU上，仅在GPU模式作操作，在CPU模式下不作任何操作，
// 如果数据状态为UNINITIALIZED，则调用cudaMalloc分配显存，并初始化数据内容为0，置数据状态为HEAD_AT_GPU，并置own_gpu_data_为true
// 如果数据状态为HEAD_AT_CPU，如果gpu_ptr_为空，则调用cudaMalloc分配显存，并置own_gpu_data_为true，然后将内存数据拷贝到显存(数据同步)，并将数据状态置为SYNCED
// 其它数据状态不作任何操作
  void to_gpu();
// 指向CPU的数据指针
  void* cpu_ptr_;
// 指向GPU的数据指针
  void* gpu_ptr_;
// 数据大小(字节)
  size_t size_;
// 数据状态，当前数据存放的位置
  SyncedHead head_;
// 是否通过SyncedMemory类分配了CPU内存
  bool own_cpu_data_;
// 是否通过cuda分配了CPU内存
  bool cpu_malloc_use_cuda_;
// 是否通过SyncedMemory类分配了GPU显存
  bool own_gpu_data_;
// 设备编号
  int gpu_device_;

// 禁止使用SyncedMemory类的拷贝和赋值操作
  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_

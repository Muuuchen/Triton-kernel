import torch
import triton 
import triton.language as tl
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)
# jit装饰符 jit编译接口
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elem, BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(axis=0)
    # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    #一个kernel就是一个pid，然后相加？
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0,BLOCK_SIZE) # start到能够处理的范围,其实还是类似一个数组
    mask = offsets < n_elem # 保证不越界
    # load from DRAM
    x = tl.load(x_ptr + offsets,mask=mask)
    y = tl.load(y_ptr + offsets,mask=mask)
    # x,y 相当于load了两个数组，然后output就是x+y
    # mask再处理边界上还是很有用的
    output = x + y
    tl.store(output_ptr+offsets,output,mask=mask)

def add(x:torch.Tensor,y:torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elem = x.numel() #张量（tensor）中元素的总数
    grid = lambda meta : (triton.cdiv(n_elem, meta['BLOCK_SIZE']),)
    add_kernel[grid](x,y,output,n_elem,BLOCK_SIZE=1024)
    return output    
    
    
if __name__ == "__main__":
    torch.manual_seed(0)
size = 98432222
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
benchmark.run(print_data=True, show_plots=False)
    
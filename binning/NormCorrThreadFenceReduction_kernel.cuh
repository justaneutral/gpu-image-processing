#ifndef _NORMCORRELATE_AND_REDUCE_KERNEL_H_
#define _NORMCORRELATE_AND_REDUCE_KERNEL_H_


#include "prototypes.h"


#include <device_functions.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define maxstreams (2)
cudaStream_t stream[maxstreams];
int dsrcp = 0;
bool dbuff = true;

texture<uchar,2,cudaReadModeElementType> tref, tsrc;    // need to use cudaReadModeElementType for tex2Dgather
cudaArray *dref=0,*dsrc=0,*dsrc1=0;

extern "C" void initTexture(uchar *href, int rw, int rh, uchar *hsrc, int sw, int sh)
{
    for(int i=0;i<maxstreams;i++)
		checkCudaErrors(cudaStreamCreate(&stream[i]));
	// allocate array and copy image data
    cudaChannelFormatDesc cdref = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaChannelFormatDesc cdsrc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    checkCudaErrors(cudaMallocArray(&dref, &cdref, rw, rh));
	checkCudaErrors(cudaMallocArray(&dsrc, &cdsrc, sw, sh));
	checkCudaErrors(cudaMallocArray(&dsrc1, &cdsrc, sw, sh));
    uint rl = rw * rh * sizeof(uchar);
	uint sl = sw * sh * sizeof(uchar);
    checkCudaErrors(cudaMemcpyToArray(dref, 0, 0, href, rl, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToArray(dsrc, 0, 0, hsrc, sl, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToArray(dsrc1, 0, 0, hsrc, sl, cudaMemcpyHostToDevice));
    tref.addressMode[0] = cudaAddressModeClamp;
    tref.addressMode[1] = cudaAddressModeClamp;
    tref.filterMode = cudaFilterModePoint;
    tref.normalized = false;    // access with integer texture coordinates
    tsrc.addressMode[0] = cudaAddressModeClamp;
    tsrc.addressMode[1] = cudaAddressModeClamp;
    tsrc.filterMode = cudaFilterModePoint;
    tsrc.normalized = false;    // access with integer texture coordinates
    checkCudaErrors(cudaBindTextureToArray(tref,dref));
	checkCudaErrors(cudaBindTextureToArray(tsrc,dsrc1));
	dsrcp = 0;
	dbuff = false;
}

extern "C" void reloadTexture(uchar *hsrc, int sw, int sh)
{
	uint sl = sw * sh * sizeof(uchar);
	if(dbuff)
	{
		checkCudaErrors(cudaBindTextureToArray(tsrc,dsrc));
		checkCudaErrors(cudaMemcpyToArrayAsync(dsrc1,0,0,hsrc,sl,cudaMemcpyHostToDevice,stream[dsrcp]));
		dbuff=false;
	}
	else
	{
		checkCudaErrors(cudaBindTextureToArray(tsrc,dsrc1));
		checkCudaErrors(cudaMemcpyToArrayAsync(dsrc,0,0,hsrc,sl,cudaMemcpyHostToDevice,stream[dsrcp]));
		dbuff=true;	
	}
	dsrcp = dsrcp>=(maxstreams-1)?0:dsrcp+1;
}

extern "C" void freeTexture()
{
    checkCudaErrors(cudaFreeArray(dref));
    checkCudaErrors(cudaFreeArray(dsrc));
	checkCudaErrors(cudaFreeArray(dsrc1));
	for(int i=0;i<maxstreams;i++)
			checkCudaErrors(cudaStreamDestroy(stream[i]));
}



template <unsigned int blockSize> __device__ void reduceBlock(volatile float *sdata, float *mySum, const unsigned int tid, cg::thread_block cta)
{
    unsigned int sn;

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    
	for(sn=0;sn<NumSumsTotal;sn++)
		sdata[blockSize*sn+tid] = mySum[sn];

    cg::sync(tile32);

	const int VEC = 32;
    const int vid = tid & (VEC-1);

	float beta[NumSumsTotal];
	memcpy(beta,mySum,NumSumsTotal*sizeof(float));

    float temp;

    for (int i = VEC/2; i > 0; i>>=1)
    {
        if (vid < i)
			for(sn=0;sn<NumSumsTotal;sn++)
			{
				temp = sdata[blockSize*sn+tid+i];
				beta[sn] += temp;
				sdata[blockSize*sn+tid] = beta[sn];
			}
        cg::sync(tile32);
    }

    cg::sync(cta);

    if (cta.thread_rank() == 0) 
    {
        memset(beta,0.0,NumSumsTotal*sizeof(float));
		for(sn=0;sn<NumSumsTotal;sn++)
		{
			for (int i = 0; i < blockDim.x; i += VEC) 
				beta[sn]  += sdata[blockSize*sn+i];
			sdata[blockSize*sn] = beta[sn];
		}
	}

    cg::sync(cta);
}




template <unsigned int blockSize, bool nIsPow2> __device__ void reduceBlocks(const unsigned char *g_idata, const unsigned char *g_idata1, float *g_odata, unsigned int sizex, unsigned int sizey, unsigned int step, bool binning, unsigned int cw, unsigned int ch, cg::thread_block cta)
{
    extern __shared__ float sdata[];
	//__shared__ float sdata[NumSums*blockSize];
    unsigned int sn,ix,iy,sizexy = sizex*sizey,tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
	float mySum[NumSumsTotal];
	for(sn=0;sn<NumSumsTotal;sn++)
		mySum[sn]=0;
    float refimgval,srcimgval;

	while (i < sizexy)
    {
		//refimgval = g_idata1[i];
		ix = step*(i%sizex);
		iy = step*(i/sizey);
		refimgval  = tex2D(tref,(float)ix,(float)iy);
		unsigned int iyoffset = ch, ixoffset = cw;
		for(unsigned int tn=0;tn<NumSumsTotal;tn+=NumSums)
		{
			//srcimgval = g_idata[(iy + ch)*idata1pitch+ix+cw];
			srcimgval = tex2D(tsrc,(float)ix+ixoffset,(float)iy+iyoffset);
			mySum[tn] += (float)srcimgval;
			mySum[tn+1] += (float)srcimgval*srcimgval;
			mySum[tn+2] +=(float)srcimgval*refimgval;
			if (nIsPow2 || i + blockSize < sizexy)
			{
				unsigned int ib = i+blockSize;
				//refimgval = g_idata1[ib];
				ix = step*(ib%sizex);
				iy = step*(ib/sizey);
				refimgval  = tex2D(tref,(float)ix,(float)iy);
				//srcimgval = g_idata[(iy + ch)*idata1pitch+ix+cw];
				srcimgval = tex2D(tsrc,(float)ix+ixoffset,(float)iy+iyoffset);
				mySum[tn] += (float)srcimgval;
				mySum[tn+1] += (float)srcimgval*srcimgval;
				mySum[tn+2] +=(float)srcimgval*refimgval;
			}

			ixoffset++;
			if(ixoffset>=NumCorrsW)
			{
				ixoffset = 0;
				iyoffset++;
			}

		}

        i += gridSize;
    }
    reduceBlock<blockSize>(sdata, mySum, tid, cta);
	if (tid == 0)
		for(sn=0;sn<NumSumsTotal;sn++)
			g_odata[gridDim.x*sn+blockIdx.x] = sdata[blockSize*sn];
}
__device__ unsigned int retirementCount = 0;
cudaError_t setRetirementCount(int retCnt)
{
    return cudaMemcpyToSymbol(retirementCount, &retCnt, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
}


template <unsigned int blockSize, bool nIsPow2> __global__ void reduceSinglePass(float T, float TT, const unsigned char *g_idata, const unsigned char *g_idata1, float *g_odata, unsigned int sizex, unsigned int sizey, unsigned int step, bool binning, unsigned int cw, unsigned int ch)
{
    cg::thread_block cta = cg::this_thread_block();
    reduceBlocks<blockSize,nIsPow2>(g_idata, g_idata1, g_odata, sizex, sizey, step, binning, cw, ch, cta);
    if (gridDim.x > 1)
    {
        const unsigned int tid = threadIdx.x;
        __shared__ bool amLast;
		//__shared__ float smem[blockSize*NumSums];
		extern __shared__ float smem[];

        __threadfence();
        if (tid==0)
        {
            unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
            amLast = (ticket == gridDim.x-1);
        }
        cg::sync(cta);
        if(amLast)
        {
            unsigned int sn, i = tid;
			float mySum[NumSumsTotal];
			for(sn=0; sn<NumSumsTotal; sn++)
				mySum[sn] = 0;

            while(i<gridDim.x)
            {
                for(sn=0;sn<NumSumsTotal;sn++)
					mySum[sn] += g_odata[gridDim.x*sn+i];
                i += blockSize;
            }
            reduceBlock<blockSize>(smem, mySum, tid, cta);
            if(tid==0)
            { //correlation terms are here
				float I,II,IT,tmp,fDenom,Numerator,Denominator,corrval;
				double cpixels=sizex*sizey;
				for(unsigned int tn=0; tn<NumCorrs; tn++)
				{
					for(sn=0; sn<NumSums; sn++)
					{
						unsigned int un=NumSums*tn+sn;
						tmp = smem[blockSize*un];
						g_odata[gridDim.x*un] = tmp;
						switch(sn)
						{
						case 0:
							I = tmp;
							break;
						case 1:
							II = tmp;
							break;
						case 2:
							IT = tmp;
						}
					}
					retirementCount = 0;
					
					if(TT<=0.0)
					{
						fDenom = float((double)cpixels*TT-T*T);
						Numerator = cpixels*IT-I*T;
						Denominator = rsqrt((cpixels*II-I*I)*fDenom);
						corrval = Numerator*Denominator;
						//g_odata[gridDim.x*NumSums*(0+0)+1] = corrval; //corr val is here
						g_odata[0] = I;
						g_odata[1] = II;
						g_odata[2] = IT;
					}
					else
					{
						fDenom = float((double)cpixels*TT-T*T);
						Numerator = cpixels*IT-I*T;
						Denominator = rsqrt((cpixels*II-I*I)*fDenom);
						corrval = Numerator*Denominator;
						//g_odata[gridDim.x*NumSums*(0+0)+1] = corrval; //corr val is here
						g_odata[tn] = corrval; //0.6555; //I;
						//g_odata[0] = corrval; //corr val is here
					}
				}
            }
        }
    }
}



bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}


extern "C" void reduceSinglePass(float T, float TT, unsigned int sizex0,unsigned int sizey0, int threads, int blocks, unsigned char *d_idata, unsigned char *d_idata1, float *d_odata, unsigned int step, bool binning, unsigned int cw, unsigned int ch)
{
	
	unsigned int sizex = sizex0/step;
	unsigned int sizey = sizey0/step;

	const unsigned int sizexy = sizex*sizey;
	//const unsigned int NumSums = 3;

	//float *d_idata = NULL;
	
	dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smemSize = NumSumsTotal * threads * sizeof(float);

    // choose which of the optimized versions of reduction to launch
    if (isPow2(sizexy))
    {
        switch (threads)
        {
            case 512:
                reduceSinglePass<512,true><<< dimGrid, dimBlock, smemSize, stream[dsrcp]>>>(T,TT,d_idata, d_idata1, d_odata, sizex, sizey, step, binning, cw, ch);
                break;

            case 256:
                reduceSinglePass<256,true><<< dimGrid, dimBlock, smemSize, stream[dsrcp]>>>(T,TT,d_idata, d_idata1, d_odata, sizex, sizey, step, binning, cw, ch);
                break;

            case 128:
                reduceSinglePass<128,true><<< dimGrid, dimBlock, smemSize, stream[dsrcp]>>>(T,TT,d_idata, d_idata1, d_odata, sizex, sizey, step, binning, cw, ch);
                break;

            case 64:
                reduceSinglePass<64,true><<< dimGrid, dimBlock, smemSize, stream[dsrcp]>>>(T,TT,d_idata, d_idata1, d_odata, sizex, sizey, step, binning, cw, ch);
                break;

            case 32:
                reduceSinglePass<32,true><<< dimGrid, dimBlock, smemSize, stream[dsrcp]>>>(T,TT,d_idata, d_idata1, d_odata, sizex, sizey, step, binning, cw, ch);
                break;

            case 16:
                reduceSinglePass<16,true><<< dimGrid, dimBlock, smemSize, stream[dsrcp]>>>(T,TT,d_idata, d_idata1, d_odata, sizex, sizey, step, binning, cw, ch);
                break;

            case  8:
                reduceSinglePass<8,true><<< dimGrid, dimBlock, smemSize, stream[dsrcp]>>>(T,TT,d_idata, d_idata1, d_odata, sizex, sizey, step, binning, cw, ch);
                break;

            case  4:
                reduceSinglePass<4,true><<< dimGrid, dimBlock, smemSize, stream[dsrcp]>>>(T,TT,d_idata, d_idata1, d_odata, sizex, sizey, step, binning, cw, ch);
                break;

            case  2:
                reduceSinglePass<2,true><<< dimGrid, dimBlock, smemSize, stream[dsrcp]>>>(T,TT,d_idata, d_idata1, d_odata, sizex, sizey, step, binning, cw, ch);
                break;

            case  1:
                reduceSinglePass<1,true><<< dimGrid, dimBlock, smemSize, stream[dsrcp]>>>(T,TT,d_idata, d_idata1, d_odata, sizex, sizey, step, binning, cw, ch);
                break;
        }
    }
    else
    {
        switch (threads)
        {
            case 512:
                reduceSinglePass<512,false><<< dimGrid, dimBlock, smemSize, stream[dsrcp]>>>(T,TT,d_idata, d_idata1, d_odata, sizex, sizey, step, binning, cw, ch);
                break;

            case 256:
                reduceSinglePass<256,false><<< dimGrid, dimBlock, smemSize, stream[dsrcp]>>>(T,TT,d_idata, d_idata1, d_odata, sizex, sizey, step, binning, cw, ch);
                break;

            case 128:
                reduceSinglePass<128,false><<< dimGrid, dimBlock, smemSize, stream[dsrcp]>>>(T,TT,d_idata, d_idata1, d_odata, sizex, sizey, step, binning, cw, ch);
                break;

            case 64:
                reduceSinglePass<64,false><<< dimGrid, dimBlock, smemSize, stream[dsrcp]>>>(T,TT,d_idata, d_idata1, d_odata, sizex, sizey, step, binning, cw, ch);
                break;

            case 32:
                reduceSinglePass<32,false><<< dimGrid, dimBlock, smemSize, stream[dsrcp]>>>(T,TT,d_idata, d_idata1, d_odata, sizex, sizey, step, binning, cw, ch);
                break;

            case 16:
                reduceSinglePass<16,false><<< dimGrid, dimBlock, smemSize, stream[dsrcp]>>>(T,TT,d_idata, d_idata1, d_odata, sizex, sizey, step, binning, cw, ch);
                break;

            case  8:
                reduceSinglePass<8,false><<< dimGrid, dimBlock, smemSize, stream[dsrcp]>>>(T,TT,d_idata, d_idata1, d_odata, sizex, sizey, step, binning, cw, ch);
                break;

            case  4:
                reduceSinglePass<4,false><<< dimGrid, dimBlock, smemSize, stream[dsrcp]>>>(T,TT,d_idata, d_idata1, d_odata, sizex, sizey, step, binning, cw, ch);
                break;

            case  2:
                reduceSinglePass<2,false><<< dimGrid, dimBlock, smemSize, stream[dsrcp]>>>(T,TT,d_idata, d_idata1, d_odata, sizex, sizey, step, binning, cw, ch);
                break;

            case  1:
                reduceSinglePass<1,false><<< dimGrid, dimBlock, smemSize, stream[dsrcp]>>>(T,TT,d_idata, d_idata1, d_odata, sizex, sizey, step, binning, cw, ch);
                break;
        }
    }

	dsrcp = dsrcp>=(maxstreams-1)?0:dsrcp+1;
}




///////////////////binning using integral image approach ///////////////////

__global__ void scan(float *input, float *output, int n)
{//0
	extern __shared__ float temp[];
	int tdx = threadIdx.x; int offset = 1;
	temp[2*tdx] = input[2*tdx];
	temp[2*tdx+1] = input[2*tdx+1];
	for(int d = n>>1; d > 0; d >>= 1)
	{//1
		__syncthreads();
		if(tdx < d)
		{//2
			int ai = offset*(2*tdx+1)-1;
			int bi = offset*(2*tdx+2)-1;
			temp[bi] += temp[ai];
		}//2
		offset *= 2;
	}//1
	if(tdx == 0)
		temp[n - 1] = 0;
	for(int d = 1; d < n; d *= 2)
	{//1
		offset >>= 1;
		__syncthreads();
		if(tdx < d)
		{//2
			int ai = offset*(2*tdx+1)-1;
			int bi = offset*(2*tdx+2)-1;
			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}//2
	}//1
	__syncthreads();
	output[2*tdx] = temp[2*tdx];
	output[2*tdx+1] = temp[2*tdx+1];
}//0


template <unsigned int BLOCK_DIM> __global__ void transpose(float *input, float *output, int width, int height)
{//0
	__shared__ float temp[BLOCK_DIM][BLOCK_DIM+1];
	int xIndex = blockIdx.x*BLOCK_DIM + threadIdx.x;
	int yIndex = blockIdx.y*BLOCK_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{//1
		int id_in = yIndex * width + xIndex;
		temp[threadIdx.y][threadIdx.x] = input[id_in];
	}//1
	__syncthreads();
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{//1
		int id_out = yIndex * height + xIndex;
		output[id_out] = temp[threadIdx.x][threadIdx.y];
	}//1
}//0



///////binnig with box filter ///////
//Box Filter Kernel For Gray scale image with 8bit depth
__global__ void binning_ref_kernel(unsigned char* binned, const unsigned int imagewidth, const unsigned int imageheight, const int binsize)
{
    int imagex = blockIdx.x * blockDim.x + threadIdx.x;
    int imagey = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int pitch = (imagewidth+binsize-1)/binsize;

	__shared__ float sum;

    if(imagex>=imagewidth || imagey>=imageheight)
		return;

	if(threadIdx.x == 0 && threadIdx.y == 0)
		sum = tex2D(tsrc,(float)imagex,(float)imagey);
	__syncthreads();
	atomicAdd(&sum,tex2D(tref,(float)imagex,(float)imagey));
	__syncthreads();
	if(threadIdx.x == 0 && threadIdx.y == 0)
		binned[blockIdx.y*pitch+blockIdx.x] = static_cast<unsigned char>(sum/(blockDim.y*blockDim.x));
}


__global__ void binning_src_kernel(uchar* binned, const uint imagewidth, const uint imageheight, const uint binsize)
{
	const uint ix = (blockIdx.x * blockDim.x + threadIdx.x)*binsize;
    const uint iy = (blockIdx.y * blockDim.y + threadIdx.y)*binsize;

	const uint pitch = imagewidth*binsize;
	const uint scale = binsize*binsize;

	if(iy>=(imageheight*binsize)||(ix>=(pitch)))
		return;

	float t = 0.0f;
	
	for(uint i=0;i<binsize;i++)
		for(uint j=0;j<binsize;j++)
			t += tex2D(tsrc,(float)(ix+j),(float)(iy+i));
	
	t/=scale;

	/*for(uint i=0;i<binsize;i++)
		for(uint j=0;j<binsize;j++)
			binned[((iy+i)*pitch+ix+j)] = static_cast<unsigned char>(t);*/
	binned[((iy)*pitch+ix)] = static_cast<unsigned char>(t);
}


__global__ void binning_src_kernel1(unsigned char* binned, const unsigned int imagewidth, const unsigned int imageheight, const int binsize, const int locy, const int locx)
{
    unsigned int imagex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int imagey = blockIdx.y * blockDim.y + threadIdx.y;

	//unsigned int pitch = (imagewidth+blockDim.x-1)/blockDim.x;
	unsigned int pitch = imagewidth;

    if(imagex>=imagewidth || imagey>=imageheight)
		return;

	unsigned int binx = imagex/binsize;
	unsigned int biny = imagey/binsize;

	unsigned int numsubblocksx = (blockDim.x+binsize-1)/binsize;
	unsigned int numsubblocksy = (blockDim.y+binsize-1)/binsize;

	unsigned int box = binx%numsubblocksx;
	unsigned int boy = biny%numsubblocksy;

	

	unsigned int binnedwidth = ((imagewidth+binsize-1)/binsize);

	uchar v = tex2D(tsrc,(float)imagex,(float)imagey);

	__shared__ float sum[32][32];

	if(!(binx || biny))
		sum[boy][box] = v;
	
	__syncthreads();

	if(binx || biny)
		atomicAdd(&(sum[boy][box]),v);

	__syncthreads();
	
	
	//if(!(binx || biny))
		binned[imagey*pitch+imagex] = static_cast<unsigned char>(sum[boy][box]/((float)binsize*binsize));
	//else
		//binned[imagey*pitch+imagex] = 0;
	

	/*if(imagex==locx && imagey==locy)
	{
		binned[0] = biny;
		binned[1] = binx;
		binned[2] = boy;
		binned[3] = box;
		binned[4] = numsubblocksx;
		binned[5] = binnedwidth;
		binned[6] = blockIdx.x;
		binned[7] = blockDim.x;
		binned[8] = threadIdx.x;
		binned[9] = blockIdx.y;
		binned[10] = blockDim.y;
		binned[11] = threadIdx.y;
		binned[12] = numsubblocksy;
		binned[13] = ;
		binned[14] = ;
		binned[15] = ;
		binned[16] = ;
	}*/
}

#if 0
void box_filter_8u_c1(unsigned char* CPUinput, unsigned char* CPUoutput, const int width, const int height, const int widthStep, const int filterWidth, const int filterHeight)
{

    /*
     * 2D memory is allocated as strided linear memory on GPU.
     * The terminologies "Pitch", "WidthStep", and "Stride" are exactly the same thing.
     * It is the size of a row in bytes.
     * It is not necessary that width = widthStep.
     * Total bytes occupied by the image = widthStep x height.
     */

    //Declare GPU pointer
    unsigned char *GPU_input, *GPU_output;

    //Allocate 2D memory on GPU. Also known as Pitch Linear Memory
    size_t gpu_image_pitch = 0;
    cudaMallocPitch<unsigned char>(&GPU_input,&gpu_image_pitch,width,height);
    cudaMallocPitch<unsigned char>(&GPU_output,&gpu_image_pitch,width,height);

    //Copy data from host to device.
    cudaMemcpy2D(GPU_input,gpu_image_pitch,CPUinput,widthStep,width,height,cudaMemcpyHostToDevice);

    //Bind the image to the texture. Now the kernel will read the input image through the texture cache.
    //Use tex2D function to read the image
    cudaBindTexture2D(NULL,tex8u,GPU_input,width,height,gpu_image_pitch);

    /*
     * Set the behavior of tex2D for out-of-range image reads.
     * cudaAddressModeBorder = Read Zero
     * cudaAddressModeClamp  = Read the nearest border pixel
     * We can skip this step. The default mode is Clamp.
     */
    tex8u.addressMode[0] = tex8u.addressMode[1] = cudaAddressModeBorder;

    /*
     * Specify a block size. 256 threads per block are sufficient.
     * It can be increased, but keep in mind the limitations of the GPU.
     * Older GPUs allow maximum 512 threads per block.
     * Current GPUs allow maximum 1024 threads per block
     */

    dim3 block_size(16,16);

    /*
     * Specify the grid size for the GPU.
     * Make it generalized, so that the size of grid changes according to the input image size
     */

    dim3 grid_size;
    grid_size.x = (width + block_size.x - 1)/block_size.x;  /*< Greater than or equal to image width */
    grid_size.y = (height + block_size.y - 1)/block_size.y; /*< Greater than or equal to image height */

    //Launch the kernel
    box_filter_kernel_8u_c1<<<grid_size,block_size>>>(GPU_output,width,height,gpu_image_pitch,filterWidth,filterHeight);

    //Copy the results back to CPU
    cudaMemcpy2D(CPUoutput,widthStep,GPU_output,gpu_image_pitch,width,height,cudaMemcpyDeviceToHost);

    //Release the texture
    cudaUnbindTexture(tex8u);

    //Free GPU memory
    cudaFree(GPU_input);
    cudaFree(GPU_output);
}
#endif


#endif // #ifndef _NORMCORRELATE_AND_REDUCE_KERNEL_H_

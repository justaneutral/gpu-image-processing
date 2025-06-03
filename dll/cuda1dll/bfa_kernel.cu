#include <time.h>
#include <stdio.h>
typedef unsigned char uchar;
typedef unsigned int uint;
#include "Cimg.h"
using namespace cimg_library;


void display_image(uchar *p,int w,int h,uchar *p1,int w1,int h1,float step,float *pc,int wc,int hc,float cgranularity,uint delayms,float fpxoff, float fpyoff)
{
	

	if(delayms>0)
	{
		int xmax = 0;
		int ymax = 0;
		float minc = 1.0;
		float maxc = 0.0;
		float averc = 0.0;

		int s = (uint)(step<0?-step:step);

		int pxoff = (int)fpxoff;
		int pyoff = (int)fpyoff;
	
		int orgx = pxoff/s; orgx=orgx<0?orgx:0;
		int orgy = pyoff/s; orgy=orgy<0?orgy:0;
		int trmx = (pxoff+w-s)/s; trmx=trmx>(w1-s)/s?trmx:(w1-s)/s; trmx=trmx<wc-1?wc-1:trmx;
		int trmy = (pyoff+h-s)/s; trmy=trmy>(h1-s)/s?trmy:(h1-s)/s; trmy=trmy<hc-1?hc-1:trmy;

		int wimg = wc+trmx+orgx+1;
		int himg = hc+trmy+orgy+1;

		static CImg<uchar>im(wimg+(wc-s)/s,himg+(hc-s)/s,1,3);

        if(im.width()<wimg || im.height()<himg)
		{
		    wimg+=(wc-s)/s;
            himg+=(hc-s)/s;
			im.assign(wimg,himg,1,3);
		}

		cimg_forXY(im,x,y)
		{
			im(x,y,0,0) = 0;
			im(x,y,0,1) = 0;
			im(x,y,0,2) = 0;
		}

		cimg_forXY(im,x,y)
		{

			if(p && (int)(s*x)<w && (int)(s*y)<h && (int)(x+pxoff/s-orgx)+wc>=0 && (int)(y+pyoff/s-orgy)+hc>=0 && (int)(x+pxoff/s-orgx)<wimg && (int)(y+pyoff/s-orgy)<himg)
			{
				im((int)(x+pxoff/s-orgx+wc),(int)(y+pyoff/s-orgy+hc),0,2)=p[s*(x+y*w)];
			}

			if(p1 && (int)(s*x)<w1 && (int)(s*y)<h1 && (int)(x-orgx)+wc>=0 && (int)(y-orgy)+hc>=0 && (int)(x-orgx)<wimg && (int)(y-orgy)<himg)
			{
				im((int)(x-orgx+wc),(int)(y-orgy+hc),0,1)=p1[s*(x+y*w1)];
			}

			if(pc && x<wc && y<hc /*&& -1.0 <=pc[x+y*wc] && pc[x+y*wc]<=1.0*/)
			{
				float tc = (float)(pc[x+y*wc]);
				averc += tc;
				if(minc>tc)
					minc = tc;
				if(maxc<tc)
				{
					maxc = tc;
					xmax = x;
					ymax = y;
				}
			}
		}
		averc/=(wc*hc);

		if(pc && maxc>minc)
		{
			ymax=((hc-1)/2)-ymax;
			xmax=((wc-1)/2)-xmax;

			ymax=fpyoff-ymax;
			xmax=fpxoff-xmax;

			cimg_forXY(im,x,y)
			{
				if(x<wc && y<hc /*&& -1.0<=pc[x+y*wc] && pc[x+y*wc]<=1.0*/)
				{
					//im(x,y,0,1)=(pc[x+y*wc]==maxc)?255:(uint)(pow(250.0,(pc[x+y*wc]-minc)/(maxc-minc))-1.0);
                    //im(x,y,0,2)=(pc[x+y*wc]==maxc)?255:(125-(uint)(125.0*(pc[x+y*wc]-minc)/(maxc-minc)));
                    //im(x,y,0,0)=(pc[x+y*wc]==maxc)?255:(125+(uint)(125.0*(pc[x+y*wc]-minc)/(maxc-minc)));
					float t = pc[x+y*wc];
					if(t<=minc)
						im(x,y,0,1) = im(x,y,0,2) = 255;												//maroon
					else
					if(t>=maxc)
					{
						im(x,y,0,0) = im(x,y,0,1) = im(x,y,0,2) = 255;									//white
					}
					else
					{
						if(t>=0.77)
							im(x,y,0,0) = (uchar)(255.0*t);												//red
						else
						{
							if(t>=0.5)
								im(x,y,0,0) = im(x,y,0,1) = (uchar)(255.0*t/0.77);						//yellow
							else
							{
								if(t>=0)
									im(x,y,0,1) = (uchar)(255.0*t/0.5);									//green
								else
								{
									im(x,y,0,2) = (uchar)(255.0*(1.0+t)/(-2.0*minc));					//blue
								}
							}
						}
					}
					{//osi
						for(int i=0;i<2;i++)
							im(x,y,0,i)=(abs(2*x-wc)>1)?((abs(2*y-hc)>1)?im(x,y,0,i):(abs(2*x-wc)>wc/2?255:im(x,y,0,i))):(abs(2*y-hc)>hc/2?255:im(x,y,0,i));
					}

				}

				if(p && (int)(s*x)<w && (int)(s*y)<h && (int)(x+(xmax)/s-orgx)+wc>=0 && (int)(y+(ymax)/s-orgy)+hc>=0 && (int)(x+(xmax)/s-orgx)<wimg && (int)(y+(ymax)/s-orgy)<himg)
				{
					im((int)(x+(xmax)/s-orgx+wc),(int)(y+(ymax)/s-orgy+hc),0,0)=p[s*(x+y*w)];
				}
			}
			unsigned char textcolor[] = { 255,255,255 };
			char stxt[256];
			sprintf(stxt,"norm. corr: min = %f, max = %f, average = %f",minc,maxc,averc);
			im.draw_text((int)(1.1*(float)wc),(int)(0.1*(float)hc),stxt,textcolor);
		}
		
		char *condition="In range";
		
		if(xmax<=pxoff-(wc-1)/2)
			condition = "On left border";
		else if(xmax>=pxoff+(wc-1)/2)
			condition = "On rignt border";

		if(ymax<=pyoff-(hc-1)/2)
			condition = "On upper border";
		else if(ymax>=pyoff+(hc-1)/2)
			condition = "On lover border";

		if(xmax<=pxoff-(wc-1)/2 && ymax<=pyoff-(hc-1)/2)
			condition = "On upper left corner";
		else
		{
			if(xmax<=pxoff-(wc-1)/2 && ymax>=pyoff+(hc-1)/2)
				condition = "On lower left corner";
			else
			{
				if(xmax>=pxoff+(wc-1)/2 && ymax<=pyoff-(hc-1)/2)
					condition = "On upper right corner";
				else
				{
					if(xmax>=pxoff+(wc-1)/2 && ymax>=pyoff+(hc-1)/2)
						condition = "On lower rignt corner";
				}
			}
		}

		char ttl[256];
		sprintf(ttl,"%s, C(%d,%d)=%f, #1:red,#2:green,size:x=%u,y=%u,step=%u,corr:blue,size:w=%u,h=%u,granularity=%4.2f",condition,xmax,ymax,maxc,w,h,s,wc,hc,cgranularity); 
		static int imhprev = 0;
		static int imwprev = 0;
		static CImgDisplay d;
		if(imhprev < im.height() || imwprev < im.width())
		{
			imhprev = im.height();
			imwprev = im.width();
			d.assign(imwprev,imhprev);
		}
		d.display(im).set_title(ttl);
		Sleep(delayms);
	}
}


void display_image1(uchar *p,int w,int h,uchar *p1,int w1,int h1,float step,float *pc,int wc,int hc,float cgranularity,uint delayms,float fpxoff, float fpyoff)
{
	int xmax = 0;
	int ymax = 0;
	if(delayms>0)
	{
		float minc = 1.0,maxc = 0.0;
		int s = (uint)(step<0?-step:step);

		int pxoff = (int)fpxoff;
		int pyoff = (int)fpyoff;
	
		int orgx = (pxoff)/s; orgx=orgx<0?orgx:0;
		int orgy = (pyoff)/s; orgy=orgy<0?orgy:0;
		int trmx = (pxoff+w-s)/s; trmx=trmx>(w1-s)/s?trmx:(w1-s)/s; trmx=trmx<wc-1?wc-1:trmx;
		int trmy = (pyoff+h-s)/s; trmy=trmy>(h1-s)/s?trmy:(h1-s)/s; trmy=trmy<hc-1?hc-1:trmy;

		int uletydown = (h+hc)/2+pyoff-w1; uletydown=uletydown>0?uletydown/s:0;
		int uletyup = (h-hc)/2+pyoff; uletyup=uletyup<0?-1*uletyup/s:0;
		int uletxdown = (w+wc)/2+pxoff-h1; uletxdown=uletxdown>0?uletxdown/s:0;
		int uletxup = (w-wc)/2+pxoff; uletxup=uletxup<0?-1*uletxup/s:0;

		static int wimg = 0; int wmg = trmx+orgx+1+uletxdown+uletxup;
		static int himg = 0; int hmg = trmy+orgy+1+uletydown+uletyup;

		static int oxo = 0;
		static int oyo = 0;
		static CImg<uchar>im;
		if(wimg == 0 || wimg < wmg || himg == 0 || himg < hmg)
		{
			wimg = (int)(1.1*wmg);
			himg = (int)(1.1*hmg);
			im.assign(wimg,himg,1,3);
			oxo = (wimg-wmg)/2;
			oyo = (himg-hmg)/2;
		}

		cimg_forXY(im,x,y)
		{
			im(x,y,0,0) = 0;
			im(x,y,0,1) = 0;
			im(x,y,0,2) = 0;
		}

		cimg_forXY(im,x,y)
		{
			int tx1 = x-orgx+uletxup+oxo;
			int tx = (int)(tx1+pxoff/s);
			int ty1 = y-orgy+uletyup+oyo;
			int ty = (int)(ty1+pyoff/s);

			if(p && (int)(s*x)<w && (int)(s*y)<h && tx>=0 && ty>=0 && tx<wimg && ty<himg)
			{
				im((int)(tx),(int)(y+pyoff/s-orgy+uletyup+oyo),0,2)=p[s*(x+y*w)];
			}

			if(p1 && (int)(s*x)<w1 && (int)(s*y)<h1 && tx1>=0 && ty1>=0 && tx1<wimg && ty1<himg)
			{
				im((int)(tx1),(int)(ty1),0,1)=p1[s*(x+y*w1)];
			}

			if(pc && x<wc && y<hc)
			{
				if(minc>pc[x+y*wc])
					minc = pc[x+y*wc];
				if(maxc<pc[x+y*wc])
				{
					maxc = pc[x+y*wc];
					xmax = x;
					ymax = y;
				}
			}
		}

		if(pc && maxc>minc)
		{
			ymax=((hc-1)/2)-ymax;
			xmax=((wc-1)/2)-xmax;

			ymax=pyoff-ymax;
			xmax=pxoff-xmax;

			cimg_forXY(im,x,y)
			{
				int tx1 = x-orgx+uletxup+oxo;
				int tx = (int)((tx1+xmax)/s);
				int ty1 = y-orgy+uletyup+oyo;
				int ty = (int)((ty1+ymax)/s);
				if(x<wc && y<hc && tx>=0 && ty>=0 && tx<wimg && ty<himg)
				{
					im(tx,ty,0,2)=(pc[x+y*wc]==maxc)?255:(uint)(pow(125.0,(pc[x+y*wc]-minc)/(maxc-minc))-1.0);
				}
				tx-=(int)((wc-1)/(2*s));
				ty-=(int)((hc-1)/(2*s));
				if(p && (int)(s*x)<w && (int)(s*y)<h && tx>=0 && ty>=0 && tx<wimg && ty<himg)
				{
					im(tx,ty,0,0)=p[s*(x+y*w)];
				}
			}
		}
		
		char *condition="In range";
		
		if(xmax<=pxoff-(wc-1)/2)
			condition = "On left border";
		else if(xmax>=pxoff+(wc-1)/2)
			condition = "On rignt border";

		if(ymax<=pyoff-(hc-1)/2)
			condition = "On upper border";
		else if(ymax>=pyoff+(hc-1)/2)
			condition = "On lover border";

		if(xmax<=pxoff-(wc-1)/2 && ymax<=pyoff-(hc-1)/2)
			condition = "On upper left corner";
		else
		{
			if(xmax<=pxoff-(wc-1)/2 && ymax>=pyoff+(hc-1)/2)
				condition = "On lower left corner";
			else
			{
				if(xmax>=pxoff+(wc-1)/2 && ymax<=pyoff-(hc-1)/2)
					condition = "On upper right corner";
				else
				{
					if(xmax>=pxoff+(wc-1)/2 && ymax>=pyoff+(hc-1)/2)
						condition = "On lower rignt corner";
				}
			}
		}

		char ttl[256];
		sprintf(ttl,"%s, C(%d,%d)=%f, #1:red,#2:green,size:x=%u,y=%u,step=%u,corr:blue,size:w=%u,h=%u,granularity=%4.2f",condition,xmax,ymax,maxc,w,h,s,wc,hc,cgranularity); 
		static int imhprev = 0;
		static int imwprev = 0;
		static CImgDisplay d;
		if(imhprev < im.height() || imwprev < im.width())
		{
			imhprev = im.height();
			imwprev = im.width();
			d.assign(imwprev,imhprev);
		}
		d.display(im).set_title(ttl);
		Sleep(delayms);
	}
}



#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define INTCEIL(a,b) (((a)+(b)-1)/(b))

texture<uchar,2> texImageTT;
texture<uchar,2> texTemplateTT;

__device__ __host__ inline float CorrelationValue( float SumI, float SumISq, float SumIT, float SumT, float cPixels, float fDenomExp )
{
    if(fDenomExp==0.0)
		return 0.0;
	 float Denominator = rsqrtf( (cPixels*SumISq - SumI*SumI)*fDenomExp );
	 if(Denominator==0.0)
		 return 0.0;
	float Numerator = cPixels*SumIT - SumI*SumT;
	if(Numerator==0.0)
		return 0.0;
    return Numerator*Denominator;
}

__global__ void corr_pp_kernel(float *pCorr,size_t CorrPitch,size_t wTemplate,size_t hTemplate,int wCorr,int hCorr,float step,unsigned long long* mutex,float offx,float offy)
{
	float row = ((float)(blockIdx.y)-(wCorr-1)/2)/2; ///*-(float)blockDim.y*/)/2.0;
    float col = ((float)(blockIdx.x)-(hCorr-1)/2)/2; ///*-(float)blockDim.x)*//2.0;
	
	//*blockDim.y + threadIdx.y;
	
	//*blockDim.x + threadIdx.x;
	
    //if(blockIdx.x>=wCorr || blockIdx.y>=hCorr)
    //    return;
	
    unsigned char Ic;
    unsigned char Tc;
	float I,T;

	float SumI = 0.0;
    float SumT = 0.0;
    float SumISq = 0.0;
    float SumTSq = 0.0;
    float SumIT = 0.0;
	float cPixels = 0;

	__shared__ float tSumI;
	__shared__ float tSumT;
	__shared__ float tSumISq;
	__shared__ float tSumTSq;
	__shared__ float tSumIT;
	__shared__ float tcPixels;

	if(!(threadIdx.y||threadIdx.x))
	{
		tSumI = 0.0;
		tSumT = 0.0;
		tSumISq = 0.0;
		tSumTSq = 0.0;
		tSumIT = 0.0;
		tcPixels = 0.0;
	}

	__syncthreads(); //1-must be there

    for(float y=threadIdx.y+fabs(row);y<hTemplate-blockDim.y+threadIdx.y-fabs(row);y+=blockDim.y*step)
	{	
        for (float x=threadIdx.x+fabs(col);x<wTemplate-blockDim.x+threadIdx.x-fabs(col);x+=blockDim.x*step)
		{
            Ic = tex2D(texImageTT,x+col+offx,y+row+offy);
            Tc = tex2D(texTemplateTT,x-col,y-row);
			I = ((float)Ic);
			T = ((float)Tc);
			SumI += (float)I;
			SumT += (float)T;
			SumISq += (float)I*I;
			SumTSq += (float)T*T;
			SumIT += (float)I*T;
			cPixels++;
		}
	}

	atomicAdd(&tSumI,SumI);
	atomicAdd(&tSumT,SumT);
	atomicAdd(&tSumISq,SumISq);
	atomicAdd(&tSumTSq,SumTSq);
	atomicAdd(&tSumIT,SumIT);
	atomicAdd(&tcPixels,cPixels);

	__syncthreads(); //0-must be there

	if(!(threadIdx.y||threadIdx.x))
	{
		SumI = tSumI;
		SumT = tSumT;
		SumISq = tSumISq;
		SumTSq = tSumTSq;
		SumIT = tSumIT;
		cPixels = tcPixels;

		float fDenom = float( (double) cPixels*SumTSq - (double) SumT*SumT);
		float cvl = CorrelationValue( SumI, SumISq, SumIT, SumT, cPixels, fDenom );
		
		float *pC = &(((float *)((char *)pCorr+(blockIdx.y)*CorrPitch))[blockIdx.x]);
		*pC = cvl;

		//this is critical section for finding image offset as max nc. position
		//bool isSet = false; 
		//do 
		//{
		//	if (isSet = atomicCAS(mutex, 0, 1) == 0) 
		//	{
				// critical section goes here
		//	}
		//	if (isSet) 
		//	{
		//		mutex = 0;
		//	}
		//} 
		//while (!isSet);
	}
}


float bfa_kernel_main(int& x,int& y,uchar *imgptr, uint imgwidth, uint imgheight, uint corrh, uint corrw,float step,float refoffx, float refoffy)
{
	float milliseconds = -1.0;

	//cudaError_t status;

	//define grid parameters
	dim3 threads, blocks;
#ifdef _DEBUG
	threads.x = 8;
	threads.y = 8;
#else
	threads.x = 32;
	threads.y = 32;
#endif
	threads.z = 1;
	blocks.x = corrw; //INTCEIL(corrw,threads.x); 
	blocks.y = corrh; //INTCEIL(corrh,threads.y); 
	blocks.z = 1;
	
	//display images for debugging
	static uint delayimagesms = 0;
	static float *correlation = NULL;

	static bool initialized = false;

	static cudaEvent_t start;
	static cudaEvent_t stop;
	
	//creare correlation function array in GPU
	static size_t correlationpitch = 0; 
	static float *dcorrelation = NULL;
	static unsigned long long *mutex;
	static cudaArray *pArrayTemplate;
	static cudaArray *pArrayImage;


	static uchar *refimgptr = NULL;
	static uint refimgwidth = 0;
	static uint refimgheight = 0;
	static uchar *srcimgptr = NULL;
	static uint srcimgwidth = 0;
	static uint srcimgheight = 0;

	static uint allocatedsrcimgwidth = 0;
	static uint allocatedsrcimgheight = 0;


	if(imgptr==NULL || imgwidth==0 || imgheight==0 || corrh==0 || corrw==0)
		return 0.0f;
	
	if(x==0)
	{
		if(imgwidth<=allocatedsrcimgwidth && imgheight<=allocatedsrcimgheight)
		{
			srcimgwidth = imgwidth;
			srcimgheight = imgheight;
			srcimgptr = imgptr;
		}
		else
		{
			x=2; //forse adding more buffer memory
		}
	}
			



	//upon return x and y are image displacements
	//at the function start x and y are control parameters:
	// x - a command, y - parameter
	
	//command parser:
	switch(x)
	{
	case -1: //reset and leave
			if(initialized)
			{
				if(refimgptr)
					free(refimgptr);
				refimgptr=NULL;
				if(correlation)
					free(correlation);
				correlation=NULL;
				srcimgwidth = 0;
				allocatedsrcimgwidth = 0;
				srcimgheight = 0;
				allocatedsrcimgheight = 0;
				srcimgptr = NULL;
				cudaEventDestroy(stop);
				cudaEventDestroy(start);
				cudaFree(dcorrelation);
				cudaFree(mutex);
				cudaFreeArray(pArrayTemplate);
				cudaFreeArray(pArrayImage);
				x = 0;
				y = 0;
				cudaDeviceReset();
				delayimagesms = 0;
				initialized = false;
			}
			return 0.0;
	case 1: //device initialization and memory (re)allocation
			if(initialized)
			{
				free(correlation);
				cudaEventDestroy(stop);
				cudaEventDestroy(start);
				cudaFree(dcorrelation);
				cudaFree(mutex);
				cudaFreeArray(pArrayTemplate);
				cudaFreeArray(pArrayImage);
				if(refimgptr)
					free(refimgptr);
				refimgptr = NULL;
				refimgwidth = 0;
				refimgheight = 0;
			}
			srcimgwidth = imgwidth;
			allocatedsrcimgwidth = imgwidth;
			srcimgheight = imgheight;
			allocatedsrcimgheight = imgheight;
			srcimgptr = imgptr;
			refimgwidth = imgwidth;
			refimgheight = imgheight;
			refimgptr = (uchar*)malloc(imgwidth*imgheight*sizeof(uchar));
			memcpy(refimgptr,imgptr,imgwidth*imgheight*sizeof(uchar));
			correlation = (float*)malloc(corrw*corrh*sizeof(float));
			cudaDeviceReset();
			cudaEventCreate(&start); 
			cudaEventCreate(&stop);
			cudaDeviceSynchronize();		
			cudaMallocPitch((void**)&dcorrelation,&correlationpitch,corrw*sizeof(float),corrh);
			//create mutex for inter-block synchronization
			cudaMalloc((void **)&mutex, sizeof(unsigned long long));
			cudaMemset(mutex, 0, sizeof(unsigned long long));
			//create template in GPU
			{
				cudaChannelFormatDesc desct = cudaCreateChannelDesc<unsigned char>();
				cudaMallocArray(&pArrayTemplate,&desct,refimgwidth,refimgheight);
			}
			cudaMemcpyToArray(pArrayTemplate,0,0,refimgptr,refimgwidth*refimgheight,cudaMemcpyHostToDevice);
			cudaBindTextureToArray(texTemplateTT,pArrayTemplate);
			//create image buffer in GPU
			{
				cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
				cudaMallocArray(&pArrayImage,&desc,srcimgwidth,srcimgheight);
			}
			cudaBindTextureToArray(texImageTT,pArrayImage);
			//show images for debigging
			delayimagesms = y;
			initialized = true;				
			break;

	case 2: //src image memory reallocation
			if(initialized)
			{
				cudaFreeArray(pArrayImage);
				srcimgwidth = imgwidth;
				allocatedsrcimgwidth = imgwidth;
				srcimgheight = imgheight;
				allocatedsrcimgheight = imgheight;
				srcimgptr = imgptr;
				//recreate image buffer in GPU
				{
					cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
					cudaMallocArray(&pArrayImage,&desc,srcimgwidth,srcimgheight);
				}
				cudaBindTextureToArray(texImageTT,pArrayImage);
				//show images for debigging
				delayimagesms = y;
			}
			break;

	}

	if(!initialized)
	{
		x = 0;
		y = 0;
		return 0.0;
	}

	//copy new image to gpu
	cudaMemcpyToArray(pArrayImage,0,0,srcimgptr,srcimgwidth*srcimgheight,cudaMemcpyHostToDevice);
	
	//call correlation using texture memory
	//cudaMemset2D(dcorrelation,correlationpitch,0,corrw*sizeof(float),corrh);
	cudaEventRecord(start);
	corr_pp_kernel<<<blocks,threads>>>(dcorrelation,correlationpitch,refimgwidth,refimgheight,corrw,corrh,step,mutex,refoffx,refoffy);
	cudaEventRecord(stop);
	cudaMemcpy2D(correlation,corrw*sizeof(float),dcorrelation,correlationpitch,corrw*sizeof(float),corrh,cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds,start,stop);

	//int posx=0,posy=0;
	float maxcor=0.0;
	for(int posx = 0; posx<corrw; posx++)
	{
		for(int posy = 0;posy<corrh; posy++)
		{
			float t = correlation[posy*corrw+posx];
			if(maxcor<t)
			{
				maxcor=t;
				x = posx;
				y = posy;
			}
		}
	}
	y=((corrh-1)/2)-y;
	x=((corrw-1)/2)-x;

	y=refoffy-y;
	x=refoffx-x;

	float corrgranilarity = 1.0;
	display_image(refimgptr,refimgwidth,refimgheight,srcimgptr,srcimgwidth,srcimgheight,step,correlation,corrw,corrh,corrgranilarity,delayimagesms,refoffx,refoffy);	

	return milliseconds;
}




#define P3000COEF (1229000)
#define TitanXpCOEF (1895000)

__global__ void msdelay_kernel(uint delaymilliseconds,clock_t *global_now)
{
    size_t row = blockIdx.y*blockDim.y + threadIdx.y;
    size_t col = blockIdx.x*blockDim.x + threadIdx.x;

	volatile long x = 0;

    if(col>0 || row>0)
        return;

	clock_t start = clock();
	clock_t now;
	
	for (;;) 
	{
	  now = clock();
	  clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
	  if (cycles >= TitanXpCOEF*delaymilliseconds)
	  {
		break;
	  }
	}
	*global_now = now;
	

	//for(long i = 0;i<1000;i++)
	//	for(long j=0;j<1000;j++)
	//		(*global_now)++;

}

uint bfa_kernel_msdelay(uint delaymilliseconds)
{
	uint numiter = delaymilliseconds/1000;
	uint lasdelay = delaymilliseconds%1000;

	clock_t *global_now = NULL;
	clock_t cpu_now;
	float milliseconds;
	dim3 threads, blocks; 
	threads.z = 1;
	threads.x = 1;
	threads.y = 1;
	blocks.x = 1; 
	blocks.y = 1; 
	blocks.z = 1;

	cudaEvent_t start, stop;
	cudaDeviceReset();
	cudaDeviceSynchronize();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	cudaMalloc((void**)&global_now,sizeof(clock_t));
	cudaMemset((void*)&global_now,0,1);

	if(numiter)
	{
		for(uint i=0;i<numiter;i++)
		{
			msdelay_kernel<<<blocks,threads>>>(1000,global_now);
			cudaDeviceSynchronize();
		}
	}

	if(lasdelay)
	{
		msdelay_kernel<<<blocks,threads>>>(lasdelay,global_now);
		cudaDeviceSynchronize();
	}

	cudaMemcpy((void*)&cpu_now,(const void*)&global_now,sizeof(clock_t),cudaMemcpyDeviceToHost);
	cudaFree(global_now);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds,start,stop);
	return (uint)milliseconds;
}





__global__ void add_and_msdelay_kernel(uint delaymilliseconds,uint *gpubuffer)
{
    size_t row = blockIdx.y*blockDim.y + threadIdx.y;
    size_t col = blockIdx.x*blockDim.x + threadIdx.x;

    if(col>0 || row>0)
        return;

	clock_t start = clock();
	clock_t now;
	
	for (;;) 
	{
	  now = clock();
	  clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
	  if (cycles >= TitanXpCOEF*delaymilliseconds)
	  {
		break;
	  }
	}
	//printf("\n\n\n ============>>>>>gpubuffer = %u,%u\n",gpubuffer[0],gpubuffer[1]);
	gpubuffer[0] += gpubuffer[1];
	gpubuffer[1] = (uint)now;
}


uint bfa_kernel_add_and_msdelay(uint sumval,uint inval,uint *delaymilliseconds)
{
	uint numiter = (*delaymilliseconds)/1000;
	uint lasdelay = (*delaymilliseconds)%1000;

	uint *gpubuffer = NULL;
	uint cpubuffer[2] = {sumval,inval};
	float milliseconds;
	dim3 threads, blocks; 
	threads.z = 1;
	threads.x = 1;
	threads.y = 1;
	blocks.x = 1; 
	blocks.y = 1; 
	blocks.z = 1;

	cudaEvent_t start, stop;
	//cudaDeviceReset();
	//cudaDeviceSynchronize();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	cudaMalloc((void**)&gpubuffer,2*sizeof(uint));
	//cudaMemset((void*)&gpubuffer,0,2);
	//cpubuffer[0] = 5;cpubuffer[1]=6;
	//printf("===|===|===>>> %u,%u\n",cpubuffer[0],cpubuffer[1]);
	cudaMemcpy((void*)gpubuffer,(const void*)cpubuffer,2*sizeof(uint),cudaMemcpyHostToDevice);
	//cudaDeviceSynchronize();
	//cpubuffer[0] = 7;cpubuffer[1]=8;
	//cudaMemcpy((void*)cpubuffer,(const void*)gpubuffer,2*sizeof(uint),cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	//printf("===X===X===>>> %u,%u\n",cpubuffer[0],cpubuffer[1]);
	if(numiter)
	{
		for(uint i=0;i<numiter;i++)
		{
			add_and_msdelay_kernel<<<blocks,threads>>>(1000,gpubuffer);
			cudaDeviceSynchronize();
		}
	}

	if(lasdelay)
	{
		add_and_msdelay_kernel<<<blocks,threads>>>(lasdelay,gpubuffer);
		cudaDeviceSynchronize();
	}

	cudaMemcpy((void*)cpubuffer,(const void*)gpubuffer,2*sizeof(clock_t),cudaMemcpyDeviceToHost);
	cudaFree(gpubuffer);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds,start,stop);
	*delaymilliseconds = (uint)milliseconds;

	if(cpubuffer[0] > sumval)
		return (cpubuffer[0]);
	else
		return sumval;
}

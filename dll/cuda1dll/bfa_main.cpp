/*****************************************************************************************************************
turn exe into DLL:
#ifdef TESTLIB_EXPORTS
	__declspec(dllexport)
#endif
 -> in front of exported functions,
project properties > 
(output directory : $(SolutionDir)$(Configuration)\ -> ..\top\bin\Release\ for Release, ..\top\bin\Debug\ for Debug configuration),
(general > configuration type : application -> dll), 
(c/c++ > preprocessor : define NDEBUG;WINDOWS;USRDLL;BFA_LIB_EXPORTS), 
(linker > system : not set -> ubsystem/windows)
in c++ file place extern "C"{dll entry point} calling C++ function which calls cu kernel
in c# project ensure that project properties > biuld > platform : x64 for all configurations, remove all win32 and other.
******************************************************************************************************************/

#include <stdio.h>
#include <math.h>

typedef unsigned char uchar;
typedef unsigned int uint;

uint bfa_kernel_msdelay(uint delaymilliseconds);
uint bfa_msdelay_cuda(uint delaymilliseconds)
{
	return bfa_kernel_msdelay(delaymilliseconds);
}

uint bfa_kernel_add_and_msdelay(uint sumval,uint inval,uint *delaymilliseconds);
uint bfa_add_and_msdelay_cuda(uint sumval, uint inval, uint *delaymilliseconds)
{
	return bfa_kernel_add_and_msdelay(sumval,inval,delaymilliseconds);
}







float bfa_kernel_main(int& x,int& y,uchar *imgptr, uint imgwidth, uint imgheight,uint corrh, uint corrw,float step,float refoffx, float refoffy);
float bfa_main_cuda(int& x,int& y,uchar *imgptr, uint imgwidth, uint imgheight,uint corrh, uint corrw,float step,float refoffx, float refoffy)
{
	float rv = x+y;
	if(step>0.0)
	{
		rv = bfa_kernel_main(x,y,imgptr,imgwidth,imgheight,corrh,corrw,step,refoffx,refoffy);
	}
	else
	{
		x = 0;
		y = 0;
	}
	return rv;
}

extern "C"
{

	#ifdef BFA_LIB_EXPORTS
		__declspec(dllexport)
	#endif
	uint bfa_msdelay(uint delaymilliseconds)
	{
		return bfa_msdelay_cuda(delaymilliseconds);
	}

	#ifdef BFA_LIB_EXPORTS
		__declspec(dllexport)
	#endif
	uint bfa_add_and_msdelay(uint sumval, uint inval, uint *delaymilliseconds)
	{
		return bfa_add_and_msdelay_cuda(sumval, inval, delaymilliseconds);
	}



	#ifdef BFA_LIB_EXPORTS
		__declspec(dllexport)
	#endif
	float bfa_main(int& x,int& y,uchar *imgptr, uint imgwidth, uint imgheight,uint corrh, uint corrw,float step,float refoffx, float refoffy)
	{
		return bfa_main_cuda(x,y,imgptr,imgwidth,imgheight,corrh,corrw,step,refoffx,refoffy);
	}

	#ifdef BFA_LIB_EXPORTS
		void main()
		{	
			float refoffx=0.0f,refoffy=0.0f;
			int x,y;
			x=0;
			y=0;
			float step = 1.0;
			uint corrh = 20, corrw=20;
			uchar refimgptr[10000];
			uint refimgwidth=100,refimgheight=100;
			uchar srcimgptr[14400];
			uint srcimgwidth=refimgwidth+corrw, srcimgheight=refimgheight+corrh;
			x=1; y=0;
			float t = bfa_main(x,y,refimgptr,refimgwidth,refimgheight,corrh,corrw,step,refoffx,refoffy);
			x=0; y=0;
			t = bfa_main(x,y,srcimgptr,srcimgwidth,srcimgheight,corrh,corrw,step,refoffx,refoffy);
			printf("called from c++, time = %d ms., x=%u, y=%u\n",t,x,y);
		}
	#endif
}
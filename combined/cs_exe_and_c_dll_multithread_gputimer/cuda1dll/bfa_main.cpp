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
#include <windows.h>
#include <stdio.h>

#define MAXCRITICALSECTION (2)

typedef unsigned char uchar;
typedef unsigned int uint;

//float bfa_kernel_main(uint& x,uint& y,uchar *refimgptr, uint refimgwidth, uint refimgheight, uchar *srcimgptr, uint srcimgwidth, uint srcimgheight,uint corrh, uint corrw,uint step);
uint bfa_kernel_msdelay(uint delaymilliseconds);
uint bfa_kernel_add_and_msdelay(uint sumval,uint inval,uint delaymilliseconds);

//float bfa_main_cuda(uint& x,uint& y,uchar *refimgptr, uint refimgwidth, uint refimgheight, uchar *srcimgptr, uint srcimgwidth, uint srcimgheight,uint corrh, uint corrw,uint step)
//{
//	return bfa_kernel_main(x,y,refimgptr,refimgwidth,refimgheight,srcimgptr,srcimgwidth,srcimgheight,corrh,corrw,step);
//}

uint bfa_msdelay_cuda(uint delaymilliseconds)
{
	return bfa_kernel_msdelay(delaymilliseconds);
}

uint bfa_add_and_msdelay_cuda(uint sumval, uint inval, uint delaymilliseconds)
{
	return bfa_kernel_add_and_msdelay(sumval,inval,delaymilliseconds);
}

extern "C"
{
//	#ifdef BFA_LIB_EXPORTS
//		__declspec(dllexport)
//	#endif
//	float bfa_main(uint& x,uint& y,uchar *refimgptr, uint refimgwidth, uint refimgheight, uchar *srcimgptr, uint srcimgwidth, uint srcimgheight,uint corrh, uint corrw,uint step)
//	{
//		return bfa_main_cuda(x,y,refimgptr,refimgwidth,refimgheight,srcimgptr,srcimgwidth,srcimgheight,corrh,corrw,step);
//	}

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
	uint bfa_add_and_msdelay(uint sumval, uint inval, uint delaymilliseconds)
	{
		return bfa_add_and_msdelay_cuda(sumval, inval, delaymilliseconds);
	}


	#ifndef BFA_LIB_EXPORTS
		void main()
		{		
			uint x,y;
			uint step = 1;
			uint corrh = 20, corrw=20;
			uchar refimgptr[10000];
			uint refimgwidth=100,refimgheight=100;
			uchar srcimgptr[14400];
			uint srcimgwidth=refimgwidth+corrw, srcimgheight=refimgheight+corrh;
			float t = bfa_main(x,y,refimgptr,refimgwidth,refimgheight,srcimgptr,srcimgwidth,srcimgheight,corrh,corrw,step);
			printf("called from c++, time = %d ms., x=%u, y=%u\n",t,x,y);
		}
	#endif
}
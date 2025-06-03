// c_exe.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdio.h>
#include <conio.h>
typedef unsigned char uchar;
typedef unsigned int uint;
#include "Cimg.h"
using namespace cimg_library;

extern "C" __declspec(dllimport) float 
	bfa_main(int& x,int& y,uchar *imgptr, uint imgwidth, uint imgheight,uint corrh, uint corrw,float step,float refoffx, float refoffy);

int _tmain(int argc, _TCHAR* argv[])
{
	const char pathprefix[] = "C:\\paul\\prj\\0_active\\NexGenPlatform\\GPU\\Set_";
	//const char pathprefix[] = "G:\\paul\\GPU\\Set_";

	int erroraccum = 0;

	uint corrh = 201;
	uint corrw = 201;
	float step = 3.0f;
	CImgDisplay disp;
	char ptt[256],pth[256];

	int setn = 1;
	int noisen = 0;
	char *noisetag[] = {"clean","no_noise","2_percent_noise"};

	const int oxa[] = {470,  715, 715,1072,1072,1112,1112,1151,1178,1382,1382,1623,1623,1624,1624};
	const int lxa[] = { 79,  982,1553, 165, 166,  62, 291, 546, 519, 315,1536,  74, 823,  73,1636};
	const int oya[] = {124, 1014,1014, 356, 356,1087, 904, 951,1216, 163, 163,1024,1024, 202, 202};
	const int lya[] = {1055, 267,1228, 700, 701, 194,  42, 330,  45,1118,1169, 183, 182,1079,1242};

	while(!_kbhit()) { puts("."); Sleep(1000);}
	
	for(int refn = 0/*5*/; refn < 15/*6*/; refn++) //reference file number from 0 to 3
	{
		int ox = oxa[refn];
		int lx = lxa[refn];
		int oy = oya[refn];
		int ly = lya[refn];
		sprintf(ptt,"%s%d\\original_%d_%d_%d_%d.bmp",pathprefix,setn,ox,lx,oy,ly);
		//sprintf(ptt,"G:\\paul\\GPU\\Set_%d\\original_%d_%d_%d_%d.bmp",setn,ox,lx,oy,ly);
		//sprintf(pth,"G:\\paul\\GPU\\Set_%d\\original.bmp",setn); ox = 100; oy=200;
		CImg<uchar>refim(ptt); //refim.crop(ox+10,oy+10,refim.width()-1,refim.height()-1);
		//disp.display(refim).set_title("reference");

		ox = oxa[refn]-1;
		lx = refim.width();
		oy = oya[refn]-1;
		ly = refim.height();

		if(noisen==0)
			sprintf(pth,"%s%d\\Original.bmp",pathprefix,setn,noisetag[noisen],0,0);
		else
			sprintf(pth,"%s%d\\%s\\DieAddress_(%d,%d)_InspectionSiteAddress_(0,0).bmp",pathprefix,setn,noisetag[noisen],0,0);

		char *fn = strcat(ptt,".txt");
		FILE *fh = fopen(fn,"w");
		fprintf(fh,"Reference file: %s\nimage file: %s\n",ptt,pth);
		fflush(fh);
		fclose(fh);

		CImg<uchar>im(pth);
		//disp.display(im).set_title("first image");

		//setup reference image
		uchar *imgptr = &(refim.at(0));
		uint imgwidth = refim.width();
		uint imgheight = refim.height();
		int x = 1;
		int y = 0;
		bfa_main(x,y,imgptr,imgwidth,imgheight,corrh,corrw,step,.0f,.0f);

		//setup image size
		imgptr = &(im.at(0));
		imgwidth = im.width();
		imgheight = im.height();
		x = 2;
		y = 1;
		bfa_main(x,y,imgptr,imgwidth,imgheight,corrh,corrw,step,ox,oy);


		//processing
		for(double relative_roi_extent = 1.0/4.0; relative_roi_extent <= 1./2.; relative_roi_extent+=1./8)
		{
			int dxmax=(int)((corrw-1.)*relative_roi_extent); dxmax=(int)((lx-1.)*relative_roi_extent)<dxmax?(int)((lx-1)*relative_roi_extent):dxmax;
			int dymax=(int)((corrh-1.)*relative_roi_extent); dymax=(int)((ly-1.)*relative_roi_extent)<dymax?(int)((ly-1)*relative_roi_extent):dymax;

			//for(int nx=0;nx<21;nx++)
			int nx=0;
			{
				//for(int ny=0;ny<21;ny++)
				int ny = nx;
				{
					if(noisen==0)
						sprintf(pth,"%s%d\\Original.bmp",pathprefix,setn,noisetag[noisen],nx,ny);
					else
						sprintf(pth,"%s%d\\%s\\DieAddress_(%d,%d)_InspectionSiteAddress_(0,0).bmp",pathprefix,setn,noisetag[noisen],nx,ny);

					im.assign(pth);
					//disp.display(im).set_title(pth);
					imgptr = &(im.at(0));
					imgwidth = im.width();
					imgheight = im.height();

					//offsetting ROI for debugging
					//for(int dx = 0; dx<(corrw-1)/2 && dx<(rand()%lx-lx/2)/2; dx++)
					//for(int dx = 0; dx<(corrw-1)/2 && dx<(rand()%lx-lx/2)/2; dx++)
					for(int dx = -dxmax; dx<=dxmax; dx++/*=dxmax/3*/)
					{
						for(int dy = -dymax; dy<=dymax; dy++/*=dymax/3*/)
						{
							x = 0; y = 0; //must ensure that x is not -1,1,or 2!!!!
							bfa_main(x,y,imgptr,imgwidth,imgheight,corrh,corrw,step,ox+dx,oy+dy);
							int err = abs(nx+ox-x)+abs(ny+oy-y);
							char diagstr[256];
							erroraccum+=err;
							sprintf(diagstr,"rel.extent = %0.3f, %s, img#: %d,%d, dx=%d, dy=%d, x=%d, y=%d, err = %d, acc.err = %d\n",relative_roi_extent,noisetag[noisen],nx,ny,dx,dy,x,y,err,erroraccum);
							printf(diagstr);
							if(err>0)
							{
								sprintf(diagstr,"rel.extent = %0.3f, %s, img#: %d,%d, dx=%d, dy=%d, x=%d, y=%d, err = %d\n",relative_roi_extent,noisetag[noisen],nx,ny,dx,dy,x,y,err);
								fh = fopen(fn,"a");
								fprintf(fh,diagstr);
								fflush(fh);
								fclose(fh);
							}
						}
					}
				}
			}
		}
	} //end loop for refn
	return erroraccum;
}


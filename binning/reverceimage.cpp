#include "prototypes.h"

CImg<ivt> reverceimage(CImg<ivt> *im)
{
	CImg<ivt> mi(*im);
	int x,y,X=mi.width()-1,Y=mi.height()-1;
	//ivt t = motim.mean();
	cimg_forXY(*im,x,y)
	{
		mi.atXY(x,y) = im->atXY(X-x,Y-y); // - t;
	}
	return mi;
}
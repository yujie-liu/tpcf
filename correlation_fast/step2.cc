#include <TH1D.h>
#include <TH2D.h>
#include <TH3D.h>
#include <TTree.h>
#include <TFile.h>
#include <TMath.h>
#include <TVector3.h>
#include <TRandom3.h>

#include <iostream>
#include <vector>

#include "hist_fast.h"

using namespace std;
using namespace TMath;

int main(int argc, char** argv)
{
	string filein(argv[1]);
	string fileout(argv[2]);

	Hist1D* corDD = new Hist1D(filein, "DD_cor");	

	Hist1D* corRR = new Hist1D(corDD->getNumBins(), corDD->getXMin(), corDD->getXMax());
	Hist1D* corRD = new Hist1D(corDD->getNumBins(), corDD->getXMin(), corDD->getXMax());

	Hist1D* RR_alpha = new Hist1D(filein, "RR_alpha");	
	Hist1D* RR_r = new Hist1D(filein, "RR_r");	
	Hist2D* DR_alpha_r = new Hist2D(filein, "DR_alpha_r");

	double rsum = RR_r->getEntries();
	rsum = RR_r->scale(1./rsum);
	cout << rsum << endl;
	
	//RR
	cout << "start RR" << endl;
	for(int ang = 0 ; ang < RR_alpha->getNumBins() ; ++ang)
	{
		cout << ang <<"/"<< RR_alpha->getNumBins() << endl;
		if(RR_alpha->getBinValue(ang) == 0) continue;
		double cab = Cos(RR_alpha->getBinMeanX(ang));
		for(int ar = 0 ; ar < RR_r->getNumBins() ; ++ar)
		{
			if(RR_r->getBinValue(ar) == 0.) continue;
			double Ar = RR_r->getBinMeanX(ar);
			for(int br = 0 ; br <= ar ; ++br)
			{
				if(RR_r->getBinValue(br) == 0.) continue;
				double Br = RR_r->getBinMeanX(br);
				double f = 2.;
				if(ar == br) {f = 1.;}
				corRR->fill(Sqrt(Ar*Ar + Br*Br - 2.*Ar*Br*cab), f*RR_alpha->getBinValue(ang)*RR_r->getBinValue(br)*RR_r->getBinValue(ar));
			}
		}
	}

	//RD
	cout << "start RD" << endl;
	for(int b = 0 ; b < DR_alpha_r->getNumBins() ; ++b)
	{
		if(DR_alpha_r->getBinValue(b) == 0) continue;
		double Ar = DR_alpha_r->getBinMeanX(b);
		double cab = Cos(DR_alpha_r->getBinMeanY(b));
		for(int br = 0 ; br < RR_r->getNumBins() ; ++br)
		{
			if(RR_r->getBinValue(br) == 0.) continue;
			double Br = RR_r->getBinMeanX(br);
			corRD->fill(Sqrt(Ar*Ar + Br*Br - 2.*Ar*Br*cab), DR_alpha_r->getBinValue(b)*RR_r->getBinValue(br));
		}
	}

	TFile* fin = TFile::Open(filein.c_str());
	TH1D* htime = dynamic_cast<TH1D*>(fin->Get("htime"));
	TFile* fout = TFile::Open(fileout.c_str(), "recreate");
	corDD->writeTH1D("DD");
	corRD->writeTH1D("RD");
	corRR->writeTH1D("RR");
	htime->Write("htime");
	fout->Write();
	fout->Close();
}

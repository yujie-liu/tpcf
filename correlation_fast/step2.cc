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
#include "ConfigParser.h"

using namespace std;
using namespace TMath;

double omegaM = 0.274;
double norm = 3000.;

struct Galaxy
{
        double phi;
        double theta;
        double z;
        double w;
};

inline double z2r(const double& z)
{
        double r = norm * z * (1 - .75*omegaM*z);
	return r;
}

inline double dist(const Galaxy& A, const Galaxy& B)
{
        double A_r = z2r(A.z)
        double B_r = z2r(B.z)
        double C = Cos(A.phi)*Sin(A.theta) * Cos(B.phi)*Sin(B.theta) +
             Sin(A.phi)*Sin(A.theta) * Sin(B.phi)*Sin(B.theta) +
             Cos(A.theta) * Cos(B.theta);
        return Sqrt(A_r*A_r + B_r*B_r - 2.*B_r*A_r*C);
}

int main(int argc, char** argv)
{
	string configfile(argv[1]);
	ConfigParser cfg(configfile);

	string filein = cfg.Get<string>("step2_file_in");
	string fileout = cfg.Get<string>("step2_file_out");
	int sbins = cfg.Get<int>("sbins");
	double smin = cfg.Get<double>("smin");
	double smax = cfg.Get<double>("smax");

	Hist1D* corDD = new Hist1D(sbins, smin, smax);
	Hist1D* corRR = new Hist1D(sbins, smin, smax);
	Hist1D* corRD = new Hist1D(sbins, smin, smax);

	Hist1D* RR_alpha = new Hist1D(filein, "RR_alpha");	
	Hist1D* RR_z = new Hist1D(filein, "RR_z");	
	Hist2D* DR_alpha_z = new Hist2D(filein, "DR_alpha_z");
	Hist3D* DD_alpha_z_z = new Hist3D(filein, "DD_alpha_z_z");

	double zsum = RR_z->getEntries();
	zsum = RR_z->scale(1./zsum);
	cout << zsum << endl;
	
	//RR
	cout << "start RR" << endl;
  // Loop over angular bins
	for(int ang = 0 ; ang < RR_alpha->getNumBins() ; ++ang)
	{
		if(RR_alpha->getBinValue(ang) == 0) continue;
		double cab = Cos(RR_alpha->getBinMeanX(ang));
    // Integrate over radial distribution along one axis
		for(int az = 0 ; az < RR_z->getNumBins() ; ++az)
		{
			if(RR_z->getBinValue(az) == 0.) continue;
			double Az = RR_z->getBinMeanX(az);
			double Ar = z2r(Az);
      // Integrate over radial distribution along other axis
			for(int bz = 0 ; bz <= az ; ++bz)
			{
				if(RR_z->getBinValue(bz) == 0.) continue;
				double Bz = RR_z->getBinMeanX(bz);
				double Br = z2r(Bz);
				double f = 2.;
				if(az == bz) {f = 1.;}
        // Note: s calculation below ignores curvature, assumes isotropy
        // To do:
        //  1) Compute Az, Bz in terms of redshifts, not precomputed radial dist
        //  2) Convert Az, Bz to Ar, Br assuming a particular cosmology
        //  3) Calculate distances using formulas 6-8 in the MNRAS paper
				corRR->fill(Sqrt(Ar*Ar + Br*Br - 2.*Ar*Br*cab),
                    f*RR_alpha->getBinValue(ang)*RR_z->getBinValue(bz)*RR_z->getBinValue(az));
			}
		}
	}

	//RD
	cout << "start RD" << endl;
	for(int b = 0 ; b < DR_alpha_z->getNumBins() ; ++b)
	{
		if(DR_alpha_z->getBinValue(b) == 0) continue;
		double Az = DR_alpha_z->getBinMeanX(b);
		double Ar = z2r(Az);
		double cab = Cos(DR_alpha_z->getBinMeanY(b));
		for(int bz = 0 ; bz < RR_z->getNumBins() ; ++bz)
		{
			if(RR_z->getBinValue(bz) == 0.) continue;
			double Bz = RR_z->getBinMeanX(bz);
			double Br = z2r(Bz);
      // Note: s calculation ignores curvature and assumes isotropy
			corRD->fill(Sqrt(Ar*Ar + Br*Br - 2.*Ar*Br*cab),
                  DR_alpha_z->getBinValue(b)*RR_z->getBinValue(bz));
		}
	}

	//DD
	cout << "start DD" << endl;
	for(int a = 0 ; a < DD_alpha_z_z->getNumBins() ; ++a)
	{
		if(DD_alpha_z_z->getBinValue(a) == 0) continue;
		double Az = DD_alpha_z_z->getBinMeanX(a);
		double Ar = z2r(Az);
		double Bz = DD_alpha_z_z->getBinMeanY(a);
		double Br = z2r(Bz);
		double cab = Cos(DD_alpha_z_z->getBinMeanZ(a));
		corDD->fill(Sqrt(Ar*Ar + Br*Br - 2.*Ar*Br*cab),
		    DD_alpha_z_z->getBinValue(a));
	}

	TFile* fin = TFile::Open(filein.c_str());
	TH1D* htime = dynamic_cast<TH1D*>(fin->Get("htime"));
	TH1D* hnorm = dynamic_cast<TH1D*>(fin->Get("hnorm"));
	TFile* fout = TFile::Open(fileout.c_str(), "recreate");
	TH1D* htpcf = new TH1D("tpcf", "tpcf", corDD->getNumBins(), corDD->getXMin(), corDD->getXMax());
	for(int b = 0 ; b < htpcf->GetNbinsX() ; ++b)
	{
		if(corRR->getBinValue(b) > 0)
		{
			double rr = corRR->getBinValue(b)/hnorm->GetBinContent(1);
			double rd = corRD->getBinValue(b)/hnorm->GetBinContent(2);
			double dd = corDD->getBinValue(b)/hnorm->GetBinContent(3);
			htpcf->SetBinContent(b+1, (dd-2*rd+rr)/rr);
		}
	}
	corDD->writeTH1D("DD");
	corRD->writeTH1D("RD");
	corRR->writeTH1D("RR");
	htime->Write("htime");
	fout->Write();
	fout->Close();
}

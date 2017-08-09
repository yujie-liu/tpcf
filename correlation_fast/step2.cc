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

inline double z2r(const Hist1D* inttbl, const double& z)
{
        return inttbl->getBinValue(z);
}

int main(int argc, char** argv)
{
	string configfile(argv[1]);
	ConfigParser cfg(configfile);

	string filein = cfg.Get<string>("step2_file_in");
	string fileout = cfg.Get<string>("step2_file_out");
	string intfile = cfg.Get<string>("integral_file");
	int sbins = cfg.Get<int>("s_bins");
	double smin = cfg.Get<double>("s_min");
	double smax = cfg.Get<double>("s_max");

	Hist1D* corDD = new Hist1D(sbins, smin, smax);
	Hist1D* corRR = new Hist1D(sbins, smin, smax);
	Hist1D* corRD = new Hist1D(sbins, smin, smax);

	Hist1D* RR_alpha = new Hist1D(filein, "RR_alpha");	
	Hist1D* RR_z = new Hist1D(filein, "RR_z");	
	Hist2D* DR_alpha_z = new Hist2D(filein, "DR_alpha_z");
	Hist3D* DD_alpha_z_dz = new Hist3D(filein, "DD_alpha_z_dz");

	Hist1D* int_table = new Hist1D(intfile, "int_table");
	double omegaK = cfg.Get<double>("omegaK");
	double D_H = 300000/cfg.Get<double>("H0");

	double zsum = RR_z->getEntries();
	zsum = RR_z->scale(1./zsum);
	cout << zsum << endl;
	
	//RR
	cout << "start RR" << endl;
  // Loop over angular bins
	for(int ang = 0 ; ang < RR_alpha->getNumBins() ; ++ang)
	{
		if(RR_alpha->getBinValue(ang) == 0) continue;
		double cab2 = Cos((RR_alpha->getBinMeanX(ang))/2);
		double sab2 = Sqrt(1-(cab2*cab2));
    // Integrate over radial distribution along one axis
		for(int az = 0 ; az < RR_z->getNumBins() ; ++az)
		{
			if(RR_z->getBinValue(az) == 0.) continue;
			double Az = RR_z->getBinMeanX(az);
			double Ar = z2r(int_table, Az);
      // Integrate over radial distribution along other axis
			for(int bz = 0 ; bz <= az ; ++bz)
			{
				if(RR_z->getBinValue(bz) == 0.) continue;
				double Bz = RR_z->getBinMeanX(bz);
				double Br = z2r(int_table, Bz);
				double Dc = (Ar+Br)/2;
				double K = omegaK*Dc*Dc/(6*D_H*D_H);
				double s12 = (1+K) * (Ar+Br) * sab2;
				double p12 = Abs(Ar-Br) * cab2;
				double f = 2.;
				if(az == bz) {f = 1.;}
				corRR->fill(Sqrt(s12*s12 + p12*p12),
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
		double Ar = z2r(int_table, Az);
		double cab2 = Cos((DR_alpha_z->getBinMeanY(b))/2);
		double sab2 = Sqrt(1-(cab2*cab2));
		for(int bz = 0 ; bz < RR_z->getNumBins() ; ++bz)
		{
			if(RR_z->getBinValue(bz) == 0.) continue;
			double Bz = RR_z->getBinMeanX(bz);
			double Br = z2r(int_table, Bz);
			double Dc = (Ar+Br)/2;
			double K = omegaK*Dc*Dc/(6*D_H*D_H);
			double s12 = (1+K) * (Ar+Br) * sab2;
			double p12 = Abs(Ar-Br) * cab2;
			corRD->fill(Sqrt(s12*s12 + p12*p12),
                  DR_alpha_z->getBinValue(b)*RR_z->getBinValue(bz));
		}
	}

	//DD
	cout << "start DD" << endl;
	for(int a = 0 ; a < DD_alpha_z_dz->getNumBins() ; ++a)
	{
		if(DD_alpha_z_dz->getBinValue(a) == 0) continue;
		double Az = DD_alpha_z_dz->getBinMeanX(a);
		double Ar = z2r(int_table, Az);
		double Bz = Az + DD_alpha_z_dz->getBinMeanY(a);
		double Br = z2r(int_table, Bz);
		double cab = Cos(DD_alpha_z_dz->getBinMeanZ(a));
		corDD->fill(Sqrt(Ar*Ar + Br*Br - 2.*Ar*Br*cab),
		    DD_alpha_z_dz->getBinValue(a));
	}

	TFile* fin = TFile::Open(filein.c_str());
	TH1D* htime = dynamic_cast<TH1D*>(fin->Get("htime"));
	TH1D* hnorm = dynamic_cast<TH1D*>(fin->Get("hnorm"));
	TFile* fout = TFile::Open(fileout.c_str(), "recreate");
	TH1D* htpcf = new TH1D("tpcf", "tpcf", sbins, smin, smax);
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

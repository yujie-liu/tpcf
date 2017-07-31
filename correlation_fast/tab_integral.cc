#include <TH1D.h>
#include <TTree.h>
#include <TFile.h>
#include <TMath.h>

#include <iostream>
#include <vector>

#include "hist_fast.h"
#include "ConfigParser.h"

using namespace std;
using namespace TMath;

inline double Efunc(const double& omegaM, const double& z)
{
	return Sqrt(1 + omegaM * (3*(z+z*z) + z*z*z));
}

int main(int argc, char** argv)
{
	string configfile(argv[1]);
	ConfigParser cfg(configfile);

	string outfile = cfg.Get<string>("integral_file");
	size_t zbins = cfg.Get<size_t>("zbins");
	double zmax = cfg.Get<double>("zmax");
	double DH = 300000/cfg.Get<double>("H_0");
	double omegaM = cfg.Get<double>("omegaM");
	double bwidth = zmax/zbins;

	Hist1D* int_table = new Hist1D(zbins, 0., zmax);
	double intval = 0.;
	double s1 = 1/Efunc(omegaM, 0.);
	double s2 = 1/Efunc(omegaM, bwidth);
	double bc = 0.5 * bwidth;
	for(int i = 0 ; i < zbins; ++i)
	{
		intval += 0.5 * bwidth * (s1+s2);
		int_table->fill(bc, DH*intval);
		s1 = s2;
		s2 = 1/Efunc(omegaM, bwidth*(i+2));
		bc += bwidth;
	}

	TFile* fout = TFile::Open(outfile.c_str(), "recreate");
	int_table->writeTH1D("int_table");
	fout->Write();
	fout->Close();
}

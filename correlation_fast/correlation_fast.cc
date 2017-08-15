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

struct Galaxy
{
	double phi;
	double theta;
	double z;
	double w;
};

inline double alpha(const Galaxy& A, const Galaxy& B)
{
	double cdist = Cos(A.phi)*Sin(A.theta) * Cos(B.phi)*Sin(B.theta) +
             Sin(A.phi)*Sin(A.theta) * Sin(B.phi)*Sin(B.theta) +
             Cos(A.theta) * Cos(B.theta);
	return ACos(cdist);
}

// Galaxy 2D angular position and weight, using
// direction cosines of theta and phi
struct Galaxy_ang
{
        double w;
        double cphi;
        double sphi;
        double ctheta;
        double stheta;
};

inline double Calpha(const Galaxy_ang& A, const Galaxy_ang& B)
{
	double cdist = A.cphi*A.stheta * B.cphi*B.stheta +
	     A.sphi*A.stheta * B.sphi*B.stheta +
	     A.ctheta * B.ctheta;
	return cdist;
}

inline double Calpha(const Galaxy& A, const Galaxy_ang& B)
{
	double cdist = Cos(A.phi)*Sin(A.theta) * B.cphi*B.stheta +
	     Sin(A.phi)*Sin(A.theta) * B.sphi*B.stheta +
	     Cos(A.theta) * B.ctheta;
	return cdist;
}

inline double z2r(const Hist1D* inttbl, const double& z)
{
        return inttbl->getBinValue(z);
}

// Angular map for galaxy positions. Hard-codes rectangular bins
// in theta and phi.
template< typename T > class Map2D
{
	private:
		size_t nxbins_, nybins_;
		double xmin_, xmax_;
		double ymin_, ymax_;
		vector< vector< T > > data_;
	public:
		Map2D(size_t nxbins, double xmin, double xmax, size_t nybins, double ymin, double ymax) : nxbins_(nxbins), nybins_(nybins), xmin_(xmin), xmax_(xmax), ymin_(ymin), ymax_(ymax), data_(nxbins*nybins)
	{
	}
		int getNumBinsX() const {return nxbins_;}
		int getNumBinsY() const {return nybins_;}
		int getNumBins() const {return nxbins_*nybins_;}
		int getXBin(double x) const {return (x-xmin_)*nxbins_/(xmax_ - xmin_);}
		int getYBin(double y) const {return (y-ymin_)*nybins_/(ymax_ - ymin_);}
		int getBin(double x, double y) const {return getXBin(x) + nxbins_ * getYBin(y);}

		void fill(double x, double y, const T& obj)
		{
			int bin = getBin(x, y);
			if(bin < 0 || bin >= int(data_.size())) return;

			data_[bin].push_back(obj);
		}

		int getBinByBins(int xbin, int ybin) const {return xbin + nxbins_ * ybin;}
		int getXBinByBin(int bin) const {return bin % nxbins_;}
		int getYBinByBin(int bin) const {return bin / nxbins_;}

		size_t getBinEntries(int bin) const
		{
			return data_[bin].size();
		}
		size_t getBinEntries(int xbin, int ybin) const
		{
			return getBinEntries(getBinByBins(xbin, ybin));
		}
		vector<T>& getBinValue(int bin)
		{
			return data_[bin];
		}
		vector<T>& getBinValue(int xbin, int ybin)
		{
			return getBinValue(getBinByBins(xbin, ybin));
		}
		const vector<T>& getBinValue(int bin) const
		{
			return data_[bin];
		}
		const vector<T>& getBinValue(int xbin, int ybin) const
		{
			return getBinValue(getBinByBins(xbin, ybin));
		}
};

class Correlations
{
	private:
		double thetamin_;
		double thetamax_;
		double phimin_;
		double phimax_;
		double zmin_;
		double zmax_;
		double amax_;
		int thetaregions_;
		int phiregions_;
		int zbins_;
		int abins_;
		int thetabins_;
		int phibins_;
		double smin_;
		double smax_;
		int sbins_;
		double omegaK;
		double D_H;
		string outfile_;
		string intfile_;

		Map2D<Galaxy>* D;
		Map2D<Galaxy_ang>* R;
		double dw = 0.;
		double dww = 0.;
		double rw = 0.;
		double rww = 0.;

		Hist1D* RR_z = nullptr;	
		Hist1D* RR_alpha = nullptr;
		Hist2D* DR_alpha_z = nullptr;
		Hist1D* corDD = nullptr;
		Hist1D* int_table = nullptr;
		TH1D* htime = nullptr;
		TH1D* hnorm = nullptr;

		void LoadMCTree(string filename)
		{
			Galaxy pos;
			Galaxy_ang pos_ang;

			TFile* tf = TFile::Open(filename.c_str());
			TTree* tr = dynamic_cast<TTree*>(tf->Get("data_pol"));
			tr->SetBranchAddress("position_pol", &pos);
			Hist2D MC_phi_theta(phibins_, phimin_, phimax_, thetabins_, thetamin_, thetamax_);
			rw = 0.;
			rww = 0.;

      // Loop over entries in TTree (theta, phi, r, w in each record)
			for(int n = 0 ; n < tr->GetEntries() ; ++n)
			{
				tr->GetEntry(n);
        // Fill 2D distribution.
				MC_phi_theta.fill(pos.phi, pos.theta);

        // Fill radial distribution with weights.
				RR_z->fill(pos.z, pos.w);

        // Track weight sums and sum^2
				rw += pos.w;
				rww += pos.w*pos.w;
			}

      // Create 2D map with the mean positions and weights (random)
			for(int b = 0 ; b < MC_phi_theta.getNumBins() ; ++b)
			{
				if(MC_phi_theta.getBinValue(b) != 0.)
				{
					pos_ang.w = MC_phi_theta.getBinValue(b);
					pos_ang.cphi = Cos(MC_phi_theta.getBinMeanX(b));
					pos_ang.sphi = Sin(MC_phi_theta.getBinMeanX(b));
					pos_ang.ctheta = Cos(MC_phi_theta.getBinMeanY(b));
					pos_ang.stheta = Sin(MC_phi_theta.getBinMeanY(b));
					R->fill(MC_phi_theta.getBinMeanX(b), MC_phi_theta.getBinMeanY(b), pos_ang);
				}
			}
			tf->Close();
		}

		void LoadDATree(string filename)
		{
			Galaxy pos;

			TFile* tf = TFile::Open(filename.c_str());
			TTree* tr = dynamic_cast<TTree*>(tf->Get("data_pol"));
			tr->SetBranchAddress("position_pol", &pos);
			dw = 0.;
			dww = 0;

      // Store data galaxy positions, weights, and weight sums
			for(int n = 0 ; n < tr->GetEntries() ; ++n)
			{
				tr->GetEntry(n);
				D->fill(pos.phi, pos.theta, pos);
				dw += pos.w;
				dww += pos.w*pos.w;
			}
			tf->Close();
		}

		void calculateRR(int bina, int binb)
		{
			cout << "Start RR " << bina << " " << binb << endl;

      // Correlations in two different random bins
			if(bina != binb)
			{
				const vector<Galaxy_ang>& va = R->getBinValue(bina);
				const vector<Galaxy_ang>& vb = R->getBinValue(binb);

        // Compute histogram of galaxy-galaxy angular distance,
        // weighted by galaxy weights.
				for(const Galaxy_ang& ga : va)
				{
					for(const Galaxy_ang& gb : vb)
					{
						double dist = Calpha(ga, gb);
						RR_alpha->fill(ACos(dist), ga.w*gb.w);	
					}
				}
			}
      // Correlations within the same bin
			else
			{
				const vector<Galaxy_ang>& va = R->getBinValue(bina);
				for(size_t x = 0 ; x < va.size() ; ++x)
				{   
					const Galaxy_ang& ga = va[x];
					for(size_t y = 0 ; y <= x ; ++y)
					{   
						const Galaxy_ang& gb = va[y];
						double dist = Calpha(ga, gb);
						double f = 1.;
						if(x == y) {f = 0.5;}
						RR_alpha->fill(ACos(dist), f*ga.w*gb.w);	
					}
				}
			}
		}

		void calculateRD(size_t bina, size_t binb)
		{
			cout << "Start RD " << bina << " " << binb << endl;
			const vector<Galaxy>& va = D->getBinValue(bina);
			const vector<Galaxy_ang>& vb = R->getBinValue(binb);
			for(const Galaxy& ga : va)
			{
				for(const Galaxy_ang& gb : vb)
				{
					double dist = Calpha(ga, gb);
					DR_alpha_z->fill(ga.z, ACos(dist), ga.w*gb.w);	
				}
			}
		}


		void calculateDD(int bina, int binb)
		{   
			cout << "Start DD " << bina << " " << binb << endl;
			if(bina != binb)
			{
				const vector<Galaxy>& va = D->getBinValue(bina);
				const vector<Galaxy>& vb = D->getBinValue(binb);
				for(const Galaxy& ga : va)
				{
					for(const Galaxy& gb : vb)
					{
                                                if(zmin_ >= ga.z) continue;
                                                if(zmin_ >= gb.z) continue;
                                                if(zmax_ <= ga.z) continue;
                                                if(zmax_ <= gb.z) continue;
                                                double Ar = z2r(int_table, ga.z);
						if(Ar == 0) continue;
                                                double Br = z2r(int_table, gb.z);
						if(Br == 0) continue;
						double cab2 = Cos(alpha(ga,gb)/2);
						double sab2 = Sqrt(1 - cab2*cab2);
                                		double K1 = omegaK*Ar*Ar/(6*D_H*D_H);
                                		double K2 = omegaK*Br*Br/(6*D_H*D_H);
                		                double s12 = ((1+K1)*Ar + (1+K2)*Br) * sab2;
		                                double p12 = Abs(Ar-Br) * cab2;
						corDD->fill(Sqrt(s12*s12 + p12*p12), ga.w*gb.w);
					}
				}
			}
			else
			{
				const vector<Galaxy>& va = D->getBinValue(bina);
				for(size_t x = 0 ; x < va.size() ; ++x)
				{   
					for(size_t y = 0 ; y < x ; ++y)
					{   
						if(zmin_ >= va[x].z) continue;
						if(zmin_ >= va[y].z) continue;
						if(zmax_ <= va[x].z) continue;
						if(zmax_ <= va[y].z) continue;
                                                double Ar = z2r(int_table, va[x].z);
						if(Ar == 0) continue;
                                                double Br = z2r(int_table, va[y].z);
						if(Br == 0) continue;
                                                double cab2 = Sqrt(alpha(va[x],va[y])/2);
                                                double sab2 = Sqrt(1 - cab2*cab2);
                                                double K1 = omegaK*Ar*Ar/(6*D_H*D_H);
                                                double K2 = omegaK*Br*Br/(6*D_H*D_H);
                                                double s12 = ((1+K1)*Ar + (1+K2)*Br) * sab2;
                                                double p12 = Abs(Ar-Br) * cab2;
                                                corDD->fill(Sqrt(s12*s12 + p12*p12), va[x].w*va[y].w);
					}
				}
			}
		}

    // Parallel job management
		pair<int, int > getjobrange(int job_n, int job_tot, int totev)
		{   
			int nev = 0;
			int startev = 0;
			int perjob = totev/job_tot;
			if(job_n < (totev % job_tot))
			{   
				nev = perjob + 1; 
				startev = (perjob + 1)*job_n;
			}
			else
			{   
				nev = perjob;
				startev = job_n*perjob + (totev % job_tot);
			}
			return pair<int, int>(startev, startev+nev);
		}

	public:
		Correlations(string configfile)
	{
		ConfigParser cfg(configfile);
		outfile_ = cfg.Get<string>("file_out");
	        intfile_ = cfg.Get<string>("integral_file");
		thetamin_ = cfg.Get<double>("theta_min");
		thetamax_ = cfg.Get<double>("theta_max");
		phimin_ = cfg.Get<double>("phi_min");
		phimax_ = cfg.Get<double>("phi_max");
		zmin_ = cfg.Get<double>("z_min");
		zmax_ = cfg.Get<double>("z_max");
		amax_ = cfg.Get<double>("alpha_max");
		thetaregions_ = cfg.Get<int>("theta_regions");
		phiregions_ = cfg.Get<int>("phi_regions");
		thetabins_ = cfg.Get<int>("theta_bins");
		phibins_ = cfg.Get<int>("phi_bins");
		zbins_ = cfg.Get<int>("z_bins");
		abins_ = cfg.Get<int>("alpha_bins");
		smin_ = cfg.Get<double>("s_min");
		smax_ = cfg.Get<double>("s_max");
		sbins_ = cfg.Get<int>("s_bins");

        	int_table = new Hist1D(intfile_, "int_table");
        	omegaK = cfg.Get<double>("omegaK");
	        D_H = 300000/cfg.Get<double>("H_0");

		R = new Map2D<Galaxy_ang>(phiregions_, phimin_, phimax_, thetaregions_, thetamin_, thetamax_);
		D = new Map2D<Galaxy>(phiregions_, phimin_, phimax_, thetaregions_, thetamin_, thetamax_);
		RR_z = new Hist1D(zbins_, zmin_, zmax_);
		RR_alpha = new Hist1D(abins_, 0, amax_);
		DR_alpha_z = new Hist2D(zbins_, zmin_, zmax_, abins_, 0, amax_);
		corDD = new Hist1D(sbins_, smin_, smax_);

		htime = new TH1D("htime", "htime", 10, 0, 10);
		hnorm = new TH1D("hnorm", "hnorm", 3, 0, 3);

		LoadMCTree(cfg.Get<string>("file_random"));
		LoadDATree(cfg.Get<string>("file_data"));
	}

		~Correlations()
		{
			TFile* fout = TFile::Open(outfile_.c_str(), "recreate");
			RR_z->writeTH1D("RR_z");
			RR_alpha->writeTH1D("RR_alpha");
			DR_alpha_z->writeTH2D("DR_alpha_z");
			corDD->writeTH1D("corDD");
			htime->Write("htime");
			hnorm->Write("hnorm");
			fout->Write();
			fout->Close();
		}

		void Calculate(int job_n, int job_tot)
		{
			stringstream ss;
			ss << outfile_ << "_" << job_n << "_" << job_tot << ".root";
			outfile_ = ss.str();
			if(job_n == 0)
			{
				hnorm->SetBinContent(1, (rw*rw-rww)*0.5);
				hnorm->SetBinContent(2, rw*dw);
				hnorm->SetBinContent(3, (dw*dw-dww)*0.5);
				cout << "Normalization RR = " <<  hnorm->GetBinContent(1) << ", RD = " << hnorm->GetBinContent(2) << ", DD = " << hnorm->GetBinContent(3) << endl;
			}
			int av_jobs = R->getNumBins()*5;
			cout << "Job number: " << job_n << ", total number of jobs: " << job_tot << ", available jobs: " << av_jobs << endl;
			pair<int, int > range(getjobrange(job_n, job_tot, av_jobs));
			cout << range.first << " " << range.second << endl;
			for(int i = range.first ; i < range.second ; ++i)
			{
				int bina = i / 5;
				int nb = i % 5;
				int xb = R->getXBinByBin(bina);
				int yb = R->getYBinByBin(bina);
				if(nb == 1){ xb++;}	
				else if(nb == 2){ xb--; yb++;}	
				else if(nb == 3){ yb++;}	
				else if(nb == 4){ xb++; yb++;}
				int binb = R->getBinByBins(xb, yb);
				int start;
				if(bina > 0 && binb > 0 && binb < R->getNumBins() && bina < R->getNumBins())
				{
					start = time(nullptr);	
					calculateDD(bina, binb);
					htime->Fill(0.5, time(nullptr) - start);
					start = time(nullptr);	
					calculateRD(bina, binb);
					if(bina != binb) {calculateRD(binb, bina);};
					htime->Fill(1.5, time(nullptr) - start);
					start = time(nullptr);	
					calculateRR(bina, binb);
					htime->Fill(2.5, time(nullptr) - start);
				}
			}
		}

};


int main(int argc, char** argv)
{
    int job_n = atoi(argv[1]);
    int job_tot = atoi(argv[2]);
    string configfile(argv[3]);

    Correlations cor(configfile);
    cor.Calculate(job_n, job_tot);
}

#include "PotentialSolver.h"
#include "Field.h"
#include <math.h>
#include <iostream>
#include <iomanip>
#include "World.h"
#include <numeric>

#include <cmath>
#include <fftw3.h>

#include <vector>
#include<algorithm>

using namespace std;
using namespace Const;

using dvector = vector<double>;

//matrix-vector multiplication
dvector Matrix::operator*(dvector &v) {
	dvector r(nu);
	for (int u=0;u<nu;u++) {
		auto &row = rows[u];
		r[u] = 0;
		for (int i=0;i<nvals;i++){
			if (row.col[i]>=0) r[u]+=row.a[i]*v[row.col[i]];
			else break;	//end at the first -1
		}
	}
	return r;
}

//returns reference to A[r,c] element in the full matrix
double& Matrix::operator()(int r, int c){
	//find this entry
	auto &row = rows[r]; int v;
	for (v=0;v<nvals;v++)
	{
		if (row.col[v]==c) break;	//if found
		if (row.col[v]<0) {row.col[v]=c;   //set
						   break;}
	}
	assert(v!=nvals);	//check for overflow
	return row.a[v];
}

/*returns inverse of a diagonal preconditioner*/
Matrix Matrix::invDiagonal()
{
	Matrix M(nu);
	for (int r=0;r<nu;r++)	M(r,r) = 1.0/(*this)(r,r);
   return M;
}

/*subtracts diagonal matrix diag from A*/
Matrix Matrix::diagSubtract(dvector &P) {
	Matrix M(*this);	//make a copy
	for (int u=0;u<nu;u++) M(u,u)=(*this)(u,u)-P[u];
	return M;
}

//multiplies row r with vector x
double Matrix::multRow(int r, dvector &x){
	auto &row = rows[r];
	double sum=0;
	for (int i=0;i<nvals;i++)
	{
		if (row.col[i]>=0) sum+=row.a[i]*x[row.col[i]];
		else break;
	}
	return sum;
}


dvector operator-(const dvector &a, const dvector &b) {
	size_t nu = a.size();
	dvector r(nu);
	for (size_t u=0;u<nu;u++) r[u] = a[u]-b[u];
	return r;
}

dvector operator+(const dvector &a, const dvector &b) {
	size_t nu = a.size();
	dvector r(nu);
	for (size_t u=0;u<nu;u++) r[u] = a[u]+b[u];
	return r;
}

dvector operator*(const double s, const dvector &a) {
	size_t nu = a.size();
	dvector r(nu);
	for (size_t u=0;u<nu;u++) r[u] = s*a[u];
	return r;
}

/*vector math helper functions*/
namespace vec
{
	/*returns sum of v1[i]*v2[i]*/
	double dot(dvector v1, dvector v2)
	{
	    double dot = 0;
	    size_t nu = v1.size();
        for (size_t j=0;j<nu;j++)
            dot+=v1[j]*v2[j];
        return dot;
	}

	/*returns l2 norm*/
	double norm(dvector v)
	{
		double sum = 0;
		int nu = v.size();
        for (int j=0;j<nu;j++)
            sum+=v[j]*v[j];
		return sqrt(sum/nu);
	}

	/** converts 3D field to a 1D vector*/
	dvector deflate(Field &f3)
	{
		dvector r(f3.ni*f3.nj*f3.nk);
		for (int i=0;i<f3.ni;i++)
			  for (int j=0;j<f3.nj;j++)
					for (int k=0;k<f3.nk;k++)
						 r[f3.U(i,j,k)] = f3[i][j][k];
		return r;
	}

	/** converts 1D vector to 3D field*/
	void inflate(dvector &d1, Field& f3)
	{
		for (int i=0;i<f3.ni;i++)
			for (int j=0;j<f3.nj;j++)
				for (int k=0;k<f3.nk;k++)
					f3[i][j][k] = d1[f3.U(i,j,k)];
	}

};

//constructs the coefficient matrix
void ConjugateGradientSolver::buildMatrix()
{
	cout << "\nConjugateGradientSolver::buildMatrix() \n";
	double3 dh = world.getDh();
	double idx = 1.0/dh[0];
	double idy = 1.0/dh[1];
	double idz = 1.0/dh[2];
    double idx2 = idx*idx;	/*1/(dx*dx)*/
	double idy2 = idy*idy;
	double idz2 = idz*idz;
	int ni = world.ni;
	int nj = world.nj;
	int nk = world.nk;
	int nu = ni*nj*nk;

	/*reserve space for node types*/
	node_type.reserve(nu);

	/*solve potential*/
	for (int k=0;k<nk;k++)
        for (int j=0;j<nj;j++)
        	for (int i=0;i<ni;i++)
            {
                int u = k*ni*nj+j*ni+i;
                A.clearRow(u);
                //dirichlet node?
				if (i==0 || i == ni-1 || j == 0 || j == nj-1 || k == 0 || k == nk-1)
                {
                    A(u,u)=1;	//set 1 on the diagonal
                    node_type[u] = DIRICHLET;
					//world.phi[i][j][k] = 0.0;
                    continue;
                } else {

                	//standard internal stencil
                	A(u,u-ni*nj) = idz2;
                	A(u,u-ni) = idy2;
                	A(u,u-1) = idx2;
                	A(u,u) = -2.0*(idx2+idy2+idz2);
                	A(u,u+1) = idx2;
                	A(u,u+ni) = idy2;
                	A(u,u+ni*nj) = idz2;
                	node_type[u] = REG;	//regular internal node
                }
            }

	//solve potential
	/*
	for (int k=0;k<nk;k++)
        for (int j=0;j<nj;j++)
        	for (int i=0;i<ni;i++)
            {
                int u = world.U(i,j,k);
                A.clearRow(u);
                //dirichlet node?
				if (world.object_id[i][j][k]>0)
                {
                    A(u,u)=1;	//set 1 on the diagonal
                    node_type[u] = DIRICHLET;
                    continue;
                }
	*/
				/*
				//Neumann boundaries
				node_type[u] = NEUMANN;		//set default
                if (i==0) {A(u,u)=idx;A(u,u+1)=-idx;}
                else if (i==ni-1) {A(u,u)=idx;A(u,u-1)=-idx;}
                else if (j==0) {A(u,u)=idy;A(u,u+ni)=-idy;}
                else if (j==nj-1) {A(u,u)=idy;A(u,u-ni)=-idy;}
                else if (k==0) {A(u,u)=idz;A(u,u+ni*nj)=-idz;}
				else if (k==nk-1) {
					A(u,u)=idz;
					A(u,u-ni*nj)=-idz;}
				*/
				/*
                else {
                	//standard internal stencil
                	A(u,u-ni*nj) = idz2;
                	A(u,u-ni) = idy2;
                	A(u,u-1) = idx2;
                	A(u,u) = -2.0*(idx2+idy2+idz2);
                	A(u,u+1) = idx2;
                	A(u,u+ni) = idy2;
                	A(u,u+ni*nj) = idz2;
                	node_type[u] = REG;	//regular internal node
                }
            }
		*/
}


/*solves Poisson equation using Gauss-Seidel*/
bool GaussSeidelSolver::solve()
{
    //references to avoid having to write world.phi
	Field &phi = world.phi;
    Field &rho = world.rho;

	//precompute 1/(dx^2)
    double3 dh = world.getDh();
    double idx2 = 1.0/(dh[0]*dh[0]);
    double idy2 = 1.0/(dh[1]*dh[1]);
    double idz2 = 1.0/(dh[2]*dh[2]);

    double L2=0;			//norm
    bool converged= false;

    /*solve potential*/
    for (unsigned it=0;it<max_solver_it;it++)
    {
		 for (int i=1;i<world.ni-1;i++)
            for (int j=1;j<world.nj-1;j++)
                for (int k=1;k<world.nk-1;k++)
                {
					//standard internal open node
					double phi_new = (rho[i][j][k]/Const::EPS_0 +
									idx2*(phi[i-1][j][k] + phi[i+1][j][k]) +
									idy2*(phi[i][j-1][k]+phi[i][j+1][k]) +
									idz2*(phi[i][j][k-1]+phi[i][j][k+1]))/(2*idx2+2*idy2+2*idz2);

					/*SOR*/
					phi[i][j][k] = phi[i][j][k] + 1.4*(phi_new-phi[i][j][k]);
				}

		 /*check for convergence*/
		 if (it%25==0)
		 {
			double sum = 0;
			for (int i=1;i<world.ni-1;i++)
				for (int j=1;j<world.nj-1;j++)
					for (int k=1;k<world.nk-1;k++)
					{
						double R = -phi[i][j][k]*(2*idx2+2*idy2+2*idz2) +
									rho[i][j][k]/Const::EPS_0 +
									idx2*(phi[i-1][j][k] + phi[i+1][j][k]) +
									idy2*(phi[i][j-1][k]+phi[i][j+1][k]) +
									idz2*(phi[i][j][k-1]+phi[i][j][k+1]);

						sum += R*R;
					}

			L2 = sqrt(sum/(world.ni*world.nj*world.nk));
			if (L2<tolerance) {converged=true;break;}
		}
    }

    if (!converged) cerr<<"GS failed to converge, L2="<<L2<<endl;
    return converged;
}


FourierSolver::FourierSolver(World &world): PotentialSolver(world), phiF(world.ni,world.nj,world.nk) {
	Nx = world.ni-2;     // Don't include the boundary nodes for Fourier Solver
	Ny = world.nj-2;
	Nz = world.nk-2;
	//Nzh = (Nz/2+1);
    in1.resize(Nx*Ny*Nz,0.0);
    in2.resize(Nx*Ny*Nz,0.0);
    out1.resize(Nx*Ny*Nz,0.0);
    out2.resize(Nx*Ny*Nz,0.0);
	//mem = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nx * Ny * Nzh);
	//out = mem;
	//in = mem[0];

	//fwrd = fftw_plan_dft_r2c_3d(Nx,Ny,Nz,in,out,FFTW_MEASURE);
	//bwrd = fftw_plan_dft_c2r_3d(Nx,Ny,Nz,out,in,FFTW_MEASURE);
    fwrd = fftw_plan_r2r_3d(Nx,Ny,Nz, in1.data(), out1.data(), FFTW_RODFT00, FFTW_RODFT00, FFTW_RODFT00, FFTW_EXHAUSTIVE);
    bwrd = fftw_plan_r2r_3d(Nx,Ny,Nz, in2.data(), out2.data(), FFTW_RODFT00, FFTW_RODFT00, FFTW_RODFT00, FFTW_EXHAUSTIVE);

}

FourierSolver::~FourierSolver() {
	//fftw_free(mem);
	fftw_destroy_plan(fwrd);
	fftw_destroy_plan(bwrd);
	fftw_cleanup();
}
	

/*solves Poisson equation using FFT*/
bool FourierSolver::solve()
{
    double Pi = M_PI;
	Field &phi = world.phi;
    Field &rho = world.rho;
	//Field tmp(world.ni,world.nj,world.nk);
   
	//cout << "Fourier Solver \n";
	
    //double L0=0.0;
    //double L1=2*Pi;
    //double xlen = (L1-L0);
	double3 X0 = world.getX0();
	double3 Xm = world.getXm();
    double xlen = Xm[0]-X0[0];
	double ylen = Xm[1]-X0[1];
	double zlen = Xm[2]-X0[2];
	
    double dx = xlen/(double)(Nx+1);
    double dy=ylen/(double)(Ny+1);
    double dz=zlen/(double)(Nz+1);

    int l=0;
	// set some values
	for (int i=0; i<Nx; i++)
		for (int j=0;j<Ny;j++)
			for(int k=0;k<Nz;k++) {				
				// consecutive ordering
				//size_t u = k*Nx*Ny + j*Nx + i;
				//in[u] = u;
				in1[l] = rho[i+1][j+1][k+1]/Const::EPS_0;
				l=l+1;
			}

    fftw_execute(fwrd);
	
    l=-1;
    for (int i = 0; i < Nx; i++){  
        for(int j = 0; j < Ny; j++){
			for(int k = 0; k < Nz; k++){

				l=l+1;
				double fact=0;

				fact=(2-2*cos((i+1)*Pi/(Nx+1)))/(dx*dx);

				fact+= (2-2*cos((j+1)*Pi/(Ny+1)))/(dy*dy);

				fact+= (2-2*cos((k+1)*Pi/(Nz+1)))/(dz*dz);

				in2[l] = out1[l]/fact;
			}
        }
    }


    fftw_execute(bwrd);
	
	//cout << "\n\n";
    //cout<<"Executed bwrd transform " << endl;
	
    l=-1;
    //double erl1 = 0.;
    //double erl2 = 0.;
    //double erl3 = 0.;
	for (int i=0; i<Nx; i++)
		for (int j=0;j<Ny;j++)
			for(int k=0;k<Nz;k++) {				
				// consecutive ordering
				l=l+1;
				//tmp[i+1][j+1][k+1] = phi[i+1][j+1][k+1];
				//phiF[i+1][j+1][k+1] = 0.125*out2[l]/((double)(Nx+1))/((double)(Ny+1))/((double)(Nz+1));
				//erl1 +=pow(fabs(phi[i+1][j+1][k+1]-phiF[i+1][j+1][k+1]),2);
				//erl2 +=pow(fabs(phi[i+1][j+1][k+1]),2);
				//erl3 +=pow(fabs(phiF[i+1][j+1][k+1]),2);
				
				/*
				if ((l > Nx*Ny*Nz/2) && (l < Nx*Ny*Nz/2 + Nz*2))
				{
					cout<< setprecision(7) << "phi[" << i << "][" << j << "]["<<k<<"] = " << tmp[i+1][j+1][k+1] << " , phiF[" << i << "][" << j << "]["<<k<<"] = " << phiF[i+1][j+1][k+1] << "\n";
					//cout<< "phi[" << i << "][" << j << "]["<<k<<"] = " << phi[i][j][k] << " , phiF[" << i << "][" << j << "]["<<k<<"] = " << phiF[i][j][k] << "\n";
				}
				*/
				
				phi[i+1][j+1][k+1] = 0.125*out2[l]/((double)(Nx+1))/((double)(Ny+1))/((double)(Nz+1));
								
			}
    //cout<< setprecision(7) << "\n phiF error=" <<sqrt(erl1) << " , " << sqrt(erl2) << " , " << sqrt(erl3) << endl ;  
      

    return true;
	
}	

/*computes electric field = -gradient(phi) using 2nd order differencing*/
void PotentialSolver::computeEF()
{
	//reference to phi to avoid needing to write world.phi
	Field &phi = world.phi;

	double3 dh = world.getDh();
	double dx = dh[0];
	double dy = dh[1];
	double dz = dh[2];

	for (int i=0;i<world.ni;i++)
		for (int j=0;j<world.nj;j++)
			for (int k=0;k<world.nk;k++)
			{
				double3 &ef = world.ef[i][j][k]; //reference to (i,j,k) ef vec3

				/*x component*/
				if (i==0)	/*forward*/
					ef[0] = -(-3*phi[i][j][k]+4*phi[i+1][j][k]-phi[i+2][j][k])/(2*dx);	
				else if (i==world.ni-1)  /*backward*/
					ef[0] = -(phi[i-2][j][k]-4*phi[i-1][j][k]+3*phi[i][j][k])/(2*dx);	
				else  /*central*/
					ef[0] = -(phi[i+1][j][k] - phi[i-1][j][k])/(2*dx);	

				/*y component*/
				if (j==0)
					ef[1] = -(-3*phi[i][j][k] + 4*phi[i][j+1][k]-phi[i][j+2][k])/(2*dy);
				else if (j==world.nj-1)
					ef[1] = -(phi[i][j-2][k] - 4*phi[i][j-1][k] + 3*phi[i][j][k])/(2*dy);
				else
					ef[1] = -(phi[i][j+1][k] - phi[i][j-1][k])/(2*dy);

				/*z component*/
				if (k==0)
					ef[2] = -(-3*phi[i][j][k] + 4*phi[i][j][k+1]-phi[i][j][k+2])/(2*dz);
				else if (k==world.nk-1)
					ef[2] = -(phi[i][j][k-2] - 4*phi[i][j][k-1]+3*phi[i][j][k])/(2*dz);
				else
					ef[2] = -(phi[i][j][k+1] - phi[i][j][k-1])/(2*dz);
			}
}

/*solves Poisson equation using ConjugateGradient*/
bool ConjugateGradientSolver::solve()
{
	int nu = A.nu;
	dvector x = vec::deflate(world.phi);
	dvector b = vec::deflate(world.rho);

	/*set RHS to zero on boundary nodes (zero electric field)
      and to existing potential on fixed nodes */
	//cout << "b[u] = ";
    for (int u=0;u<nu;u++)
    {
		if (node_type[u]==NEUMANN) b[u] = 0;			/*neumann boundary*/
        else if (node_type[u]==DIRICHLET) b[u] = x[u];	/*dirichlet boundary*/
        else b[u] = -b[u]/Const::EPS_0;            /*regular node*/
    }
	
	
	bool converged= false;

	double l2 = 0;

	//initialization
	dvector r = b-A*x;
	dvector d = r;
	dvector rk = r;

	for (unsigned it=0;it<max_solver_it;it++)
	{
		dvector z = A*d;
		//double beta = vec::dot(d,z);
		double dz = vec::dot(d,z);
		
		// Step 1
		double alpha = vec::dot(r,r)/dz;
		
		// Step 2
		x = x+alpha*d;
		
		// Step 3
		rk = r - alpha*z;
		
		// Step 4
		double beta = vec::dot(rk,rk)/vec::dot(r,r);
		
		// Step 5
		d = rk + beta*d;
		
		l2 = vec::norm(rk);
		if (l2<tolerance) {converged=true;break;}
		
		r = rk;
	}

	if (!converged) {	
		cerr<<"CG failed to converge, norm(g) = "<<l2<<endl;
	} else {
		//convert to 3d data
		vec::inflate(x,world.phi);
	}
    return converged;

}

/*solves Poisson equation using PreconditionedConjugateGradient*/
bool PreconditionedConjugateGradientSolver::solve()
{
	int nu = A.nu;
	dvector x = vec::deflate(world.phi);
	dvector b = vec::deflate(world.rho);

	
	/*set RHS to zero on boundary nodes (zero electric field)
      and to existing potential on fixed nodes */
	//cout << "b[u] = ";
    for (int u=0;u<nu;u++)
    {
		if (node_type[u]==NEUMANN) b[u] = 0;			/*neumann boundary*/
        else if (node_type[u]==DIRICHLET) b[u] = x[u];	/*dirichlet boundary*/
        else b[u] = -b[u]/Const::EPS_0;            /*regular node*/
    }
	
	bool converged= false;

	double l2 = 0;
	Matrix M = A.invDiagonal(); //inverse of Jacobi preconditioner

	/*initialization*/
	dvector g = A*x-b;
	dvector s = M*g;
	dvector d = -1*s;

	for (unsigned it=0;it<max_solver_it;it++)
	{
		dvector z = A*d;
		double alpha = vec::dot(g,s);
		double beta = vec::dot(d,z);

		x = x+(alpha/beta)*d;
		g = g+(alpha/beta)*z;
		s = M*g;

		beta = alpha;
		alpha = vec::dot(g,s);

		d = (alpha/beta)*d-s;
		l2 = vec::norm(g);
		if (l2<tolerance) {converged=true;break;}
	}

	if (!converged) {	
		cerr<<"PCG failed to converge, norm(g) = "<<l2<<endl;
	} else {
		//convert to 3d data
		vec::inflate(x,world.phi);
	}
	
    return converged;

	
}

/*
//PCG solver for a linear system Ax=b
bool ConjugateGradientSolver::solvePCGLinear(Matrix &A, dvector &x, dvector &b)
{
	bool converged= false;

	double l2 = 0;
	Matrix M = A.invDiagonal(); //inverse of Jacobi preconditioner

	//initialization
	dvector g = A*x-b;
	dvector s = M*g;
	dvector d = -1*s;

	for (unsigned it=0;it<max_solver_it;it++)
	{
		dvector z = A*d;
		double alpha = vec::dot(g,s);
		double beta = vec::dot(d,z);

		x = x+(alpha/beta)*d;
		g = g+(alpha/beta)*z;
		s = M*g;

		beta = alpha;
		alpha = vec::dot(g,s);

		d = (alpha/beta)*d-s;
		l2 = vec::norm(g);
		if (l2<tolerance) {converged=true;break;}
	}

	if (!converged)	cerr<<"PCG failed to converge, norm(g) = "<<l2<<endl;
    return converged;
}
*/

/*solves Poisson equation using MultiGrid*/
bool MultiGridSolverB::solve()
{
    //references to avoid having to write world.phi
	Field &phi = world.phi;
    Field &rho = world.rho;

	/*
	for (int i=0;i<world.ni;i++)
		for (int j=0;j<world.nj;j++)
			for (int k=0;k<world.nk;k++)
			{
				phi_test[i][j][k] = phi[i][j][k];
			}
	*/

	//precompute 1/(dx^2)
    double3 dh = world.getDh();
    double idx2 = 1.0/(dh[0]*dh[0]);
    double idy2 = 1.0/(dh[1]*dh[1]);
    double idz2 = 1.0/(dh[2]*dh[2]);

    double idx2_2h = 1.0/(4*dh[0]*dh[0]);
    double idy2_2h = 1.0/(4*dh[1]*dh[1]);
    double idz2_2h = 1.0/(4*dh[2]*dh[2]);

    double L2=0;			//norm
    bool converged= false;

    /*solve potential*/
    for (unsigned it=0;it<max_solver_it;it++)
    {
        //number of steps to iterate at the finest level
        unsigned inner_its = 3;

        //number of steps to iterate at the coarse level        
        unsigned inner2h_its = 50;

        // 1) perform one or more iterations on fine mesh
		for (unsigned its=0;its<inner_its;its++)
		{
			 for (int i=1;i<world.ni-1;i++)
				for (int j=1;j<world.nj-1;j++)
					for (int k=1;k<world.nk-1;k++)
					{
						//standard internal open node
						double phi_new = (rho[i][j][k]/Const::EPS_0 +
										idx2*(phi[i-1][j][k] + phi[i+1][j][k]) +
										idy2*(phi[i][j-1][k]+phi[i][j+1][k]) +
										idz2*(phi[i][j][k-1]+phi[i][j][k+1]))/(2*idx2+2*idy2+2*idz2);

						/*SOR*/
						phi[i][j][k] = phi[i][j][k] + 1.4*(phi_new-phi[i][j][k]);					
					}
		}
				
        // 2) compute residue on the fine mesh, R = A*phi - b
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					if ((i == 0) || (i == world.ni-1) || (j == 0) || (j == world.nj-1) || (k == 0) || (k == world.nk-1)) {
						R_h[i][j][k] = phi[i][j][k] - 0;  // dirichlet boundary
					} else {
						R_h[i][j][k] = phi[i][j][k]*(2*idx2+2*idy2+2*idz2) -
								rho[i][j][k]/Const::EPS_0 -
								idx2*(phi[i-1][j][k] + phi[i+1][j][k]) -
								idy2*(phi[i][j-1][k]+phi[i][j+1][k]) -
								idz2*(phi[i][j][k-1]+phi[i][j][k+1]);
					}
				}
                    
        // 2b) check for termination
		double sum = 0;
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					sum += R_h[i][j][k]*R_h[i][j][k];
				}

		L2 = sqrt(sum/(world.ni*world.nj*world.nk));
		if (L2<tolerance) {converged=true;break;}
		
        // 3) restrict residue to the 2h mesh
		for (int i=0;i<world.ni-1;i+=2)
			for (int j=0;j<world.nj-1;j+=2)
				for (int k=0;k<world.nk-1;k+=2)
				{
					if ((i == 0) || (j == 0)  || (k == 0) ) {
						R_2h[i/2][j/2][k/2] = R_h[i][j][k];  //R_2h[0] = R_h[0]
					} else {
						R_2h[i/2][j/2][k/2] = (R_h[i-1][j][k] + 2*R_h[i][j][k] + R_h[i+1][j][k] +
											   R_h[i][j-1][k] + 2*R_h[i][j][k] + R_h[i][j+1][k] +
											   R_h[i][j][k-1] + 2*R_h[i][j][k] + R_h[i][j][k+1] )/12.0;
					}
				}
        //R_2h[0] = R_h[0]
        
		// 4) perform few iteration of the correction vector on the 2h mesh
		for (unsigned its=0;its<inner2h_its;its++)
		{
			for (int i=1;i<world.ni/2-1;i++)
				for (int j=1;j<world.nj/2-1;j++)
					for (int k=1;k<world.nk/2-1;k++)
					{
						//standard internal open node
						double g = (R_2h[i][j][k] +
										idx2_2h*(eps_2h[i-1][j][k] + eps_2h[i+1][j][k]) +
										idy2_2h*(eps_2h[i][j-1][k]+eps_2h[i][j+1][k]) +
										idz2_2h*(eps_2h[i][j][k-1]+eps_2h[i][j][k+1]))/(2*idx2_2h+2*idy2_2h+2*idz2_2h);

						// SOR
						eps_2h[i][j][k] = eps_2h[i][j][k] + 1.4*(g-eps_2h[i][j][k]);
					}
		}

        // 5) interpolate eps to h mesh
		for (int i=0;i<world.ni-1;i++)
			for (int j=0;j<world.nj-1;j++)
				for (int k=0;k<world.nk-1;k++)
				{
					if ((i == 0) || (i == world.ni-1) || (j == 0) || (j == world.nj-1) || (k == 0) || (k == world.nk-1)) {
						eps_h[i][j][k] = 0;   // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_h[i][j][k] = eps_2h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2 + 1][j/2][k/2] +
						                  eps_2h[i/2][j/2+1][k/2] + eps_2h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2+1][j/2][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2+1][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2][j/2+1][k/2+1]);
					} else {
						eps_h[i][j][k] = 0.125*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2 + 1][j/2][k/2] +
						                  eps_2h[i/2][j/2+1][k/2] + eps_2h[i/2 + 1][j/2+1][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2 + 1][j/2][k/2+1] +
						                  eps_2h[i/2][j/2+1][k/2+1] + eps_2h[i/2 + 1][j/2+1][k/2+1]);
					}
				}

        // 6) update solution on the fine mesh
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					phi[i][j][k] = phi[i][j][k] - eps_h[i][j][k];
				}

    }

	/*
	int l = 0;
    double erl1 = 0.;
    double erl2 = 0.;
    double erl3 = 0.;
	for (int i=1;i<world.ni-1;i++)
		for (int j=1;j<world.nj-1;j++)
			for (int k=1;k<world.nk-1;k++)
			{
				// consecutive ordering
				//tmp[i+1][j+1][k+1] = phi[i+1][j+1][k+1];
				//phiF[i+1][j+1][k+1] = 0.125*out2[l]/((double)(Nx+1))/((double)(Ny+1))/((double)(Nz+1));
				erl1 +=pow(fabs(phi[i][j][k] - phi_test[i][j][k]),2);
				erl2 +=pow(fabs(phi[i][j][k]),2);
				erl3 +=pow(fabs(phi_test[i][j][k]),2);
				
				
				if ((l > world.ni*world.nj*world.nk/2) && (l < world.ni*world.nj*world.nk/2 + world.nk*2))
				{
					//cout<< setprecision(7) << "phi[" << i << "][" << j << "]["<<k<<"] = " << tmp[i+1][j+1][k+1] << " , phiF[" << i << "][" << j << "]["<<k<<"] = " << phiF[i+1][j+1][k+1] << "\n";
					cout<< setprecision(7) << "phiMG[" << i << "][" << j << "]["<<k<<"] = " << phi[i][j][k] << " , phi[" << i << "][" << j << "]["<<k<<"] = " << phi_test[i][j][k] << "\n";
				}
				l++;
												
			}
    cout<< setprecision(7) << "\n phi - phi_test error=" <<sqrt(erl1) << " , " << sqrt(erl2) << " , " << sqrt(erl3) << endl ;  
	*/

    if (!converged) cerr<<"GS failed to converge, L2="<<L2<<endl;
    return converged;
}

/*solves Poisson equation using MultiGrid*/
bool MultiGridSolverV2B::solve()
{
    //references to avoid having to write world.phi
	Field &phi = world.phi;
    Field &rho = world.rho;

	/*
	for (int i=0;i<world.ni;i++)
		for (int j=0;j<world.nj;j++)
			for (int k=0;k<world.nk;k++)
			{
				phi_test[i][j][k] = phi[i][j][k];
			}
	*/

	//precompute 1/(dx^2)
    double3 dh = world.getDh();
    double idx2 = 1.0/(dh[0]*dh[0]);
    double idy2 = 1.0/(dh[1]*dh[1]);
    double idz2 = 1.0/(dh[2]*dh[2]);

    double idx2_2h = 1.0/(4*dh[0]*dh[0]);
    double idy2_2h = 1.0/(4*dh[1]*dh[1]);
    double idz2_2h = 1.0/(4*dh[2]*dh[2]);

    double idx2_4h = 1.0/(16*dh[0]*dh[0]);
    double idy2_4h = 1.0/(16*dh[1]*dh[1]);
    double idz2_4h = 1.0/(16*dh[2]*dh[2]);
    double L2=0;			//norm
    bool converged= false;

    /*solve potential*/
    for (unsigned it=0;it<max_solver_it;it++)
    {
        //number of steps to iterate at the finest level
        unsigned inner_its = 3;

        //number of steps to iterate at the coarse level        
        unsigned inner2h_its = 5;

        unsigned inner4h_its = 50;

        // 1) perform one or more iterations on fine mesh
		for (unsigned its=0;its<inner_its;its++)
		{
			 for (int i=1;i<world.ni-1;i++)
				for (int j=1;j<world.nj-1;j++)
					for (int k=1;k<world.nk-1;k++)
					{
						//standard internal open node
						double phi_new = (rho[i][j][k]/Const::EPS_0 +
										idx2*(phi[i-1][j][k] + phi[i+1][j][k]) +
										idy2*(phi[i][j-1][k]+phi[i][j+1][k]) +
										idz2*(phi[i][j][k-1]+phi[i][j][k+1]))/(2*idx2+2*idy2+2*idz2);

						/*SOR*/
						phi[i][j][k] = phi[i][j][k] + 1.4*(phi_new-phi[i][j][k]);
						
					}
		}
				
        // 2) compute residue on the fine mesh, R = A*phi - b
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					if ((i == 0) || (i == world.ni-1) || (j == 0) || (j == world.nj-1) || (k == 0) || (k == world.nk-1)) {
						R_h[i][j][k] = phi[i][j][k] - 0;  // dirichlet boundary
					} else {
						R_h[i][j][k] = phi[i][j][k]*(2*idx2+2*idy2+2*idz2) -
								rho[i][j][k]/Const::EPS_0 -
								idx2*(phi[i-1][j][k] + phi[i+1][j][k]) -
								idy2*(phi[i][j-1][k]+phi[i][j+1][k]) -
								idz2*(phi[i][j][k-1]+phi[i][j][k+1]);
					}
				}
                    
        // 2b) check for termination

		 /*check for convergence*/
		double sum = 0;
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					sum += R_h[i][j][k]*R_h[i][j][k];
				}

		L2 = sqrt(sum/(world.ni*world.nj*world.nk));
		if (L2<tolerance) {converged=true;break;}
		
        // 3) restrict residue to the 2h mesh
		for (int i=0;i<world.ni-1;i+=2)
			for (int j=0;j<world.nj-1;j+=2)
				for (int k=0;k<world.nk-1;k+=2)
				{
					if ((i == 0) || (j == 0)  || (k == 0) ) {
						R_2h[i/2][j/2][k/2] = R_h[i][j][k];  //R_2h[0] = R_h[0]
					} else {
						R_2h[i/2][j/2][k/2] = (R_h[i-1][j][k] + 2*R_h[i][j][k] + R_h[i+1][j][k] +
											   R_h[i][j-1][k] + 2*R_h[i][j][k] + R_h[i][j+1][k] +
											   R_h[i][j][k-1] + 2*R_h[i][j][k] + R_h[i][j][k+1] )/12.0;
					}
				}
		
         // 3) restrict residue to the 4h mesh
		for (int i=0;i<world.ni/2-1;i+=2)
			for (int j=0;j<world.nj/2-1;j+=2)
				for (int k=0;k<world.nk/2-1;k+=2)
				{
					if ((i == 0) || (j == 0) || (k == 0) ) {
						R_4h[i/2][j/2][k/2] = R_2h[i][j][k];  // dirichlet boundary
					} else {
						R_4h[i/2][j/2][k/2] = (R_2h[i-1][j][k] + 2*R_2h[i][j][k] + R_2h[i+1][j][k] +
											   R_2h[i][j-1][k] + 2*R_2h[i][j][k] + R_2h[i][j+1][k] +
											   R_2h[i][j][k-1] + 2*R_2h[i][j][k] + R_2h[i][j][k+1] )/12.0;
					}
				}
           
		// 4) perform few iteration of the correction vector on the 4h mesh
		for (unsigned its=0;its<inner4h_its;its++)
		{
			 for (int i=1;i<world.ni/4-1;i++)
				for (int j=1;j<world.nj/4-1;j++)
					for (int k=1;k<world.nk/4-1;k++)
					{
						//standard internal open node
						double g = (R_4h[i][j][k] +
										idx2_4h*(eps_4h[i-1][j][k] + eps_4h[i+1][j][k]) +
										idy2_4h*(eps_4h[i][j-1][k]+eps_4h[i][j+1][k]) +
										idz2_4h*(eps_4h[i][j][k-1]+eps_4h[i][j][k+1]))/(2*idx2_4h+2*idy2_4h+2*idz2_4h);

						/*SOR*/
						eps_4h[i][j][k] = eps_4h[i][j][k] + 1.4*(g-eps_4h[i][j][k]);
					}
		}

        // 5) interpolate eps to 2h mesh
		for (int i=0;i<world.ni/2-1;i++)
			for (int j=0;j<world.nj/2-1;j++)
				for (int k=0;k<world.nk/2-1;k++)
				{
					if ((i == 0) || (i == world.ni/2-1) || (j == 0) || (j == world.nj/2-1) || (k == 0) || (k == world.nk/2-1)) {
						eps_2h[i][j][k] = 0;    // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_2h[i][j][k] = eps_4h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2 + 1][j/2][k/2] +
						                  eps_4h[i/2][j/2+1][k/2] + eps_4h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2+1][j/2][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2+1][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2][j/2+1][k/2+1]);
					} else {
						eps_2h[i][j][k] = 0.125*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2 + 1][j/2][k/2] +
						                  eps_4h[i/2][j/2+1][k/2] + eps_4h[i/2 + 1][j/2+1][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2 + 1][j/2][k/2+1] +
						                  eps_4h[i/2][j/2+1][k/2+1] + eps_4h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
		// 4) perform few iteration of the correction vector on the 2h mesh
		for (unsigned its=0;its<inner2h_its;its++)
		{
			for (int i=1;i<world.ni/2-1;i++)
				for (int j=1;j<world.nj/2-1;j++)
					for (int k=1;k<world.nk/2-1;k++)
					{
						//standard internal open node
						double g = (R_2h[i][j][k] +
										idx2_2h*(eps_2h[i-1][j][k] + eps_2h[i+1][j][k]) +
										idy2_2h*(eps_2h[i][j-1][k]+eps_2h[i][j+1][k]) +
										idz2_2h*(eps_2h[i][j][k-1]+eps_2h[i][j][k+1]))/(2*idx2_2h+2*idy2_2h+2*idz2_2h);

						// SOR
						eps_2h[i][j][k] = eps_2h[i][j][k] + 1.4*(g-eps_2h[i][j][k]);
					}
		}

        // 5) interpolate eps to h mesh
		for (int i=0;i<world.ni-1;i++)
			for (int j=0;j<world.nj-1;j++)
				for (int k=0;k<world.nk-1;k++)
				{
					if ((i == 0) || (i == world.ni-1) || (j == 0) || (j == world.nj-1) || (k == 0) || (k == world.nk-1)) {
						eps_h[i][j][k] = 0;   // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_h[i][j][k] = eps_2h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2 + 1][j/2][k/2] +
						                  eps_2h[i/2][j/2+1][k/2] + eps_2h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2+1][j/2][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2+1][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2][j/2+1][k/2+1]);
					} else {
						eps_h[i][j][k] = 0.125*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2 + 1][j/2][k/2] +
						                  eps_2h[i/2][j/2+1][k/2] + eps_2h[i/2 + 1][j/2+1][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2 + 1][j/2][k/2+1] +
						                  eps_2h[i/2][j/2+1][k/2+1] + eps_2h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
        // 6) update solution on the fine mesh
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					phi[i][j][k] = phi[i][j][k] - eps_h[i][j][k];
				}

    }

	/*
	int l = 0;
    double erl1 = 0.;
    double erl2 = 0.;
    double erl3 = 0.;
	for (int i=1;i<world.ni-1;i++)
		for (int j=1;j<world.nj-1;j++)
			for (int k=1;k<world.nk-1;k++)
			{
				// consecutive ordering
				//tmp[i+1][j+1][k+1] = phi[i+1][j+1][k+1];
				//phiF[i+1][j+1][k+1] = 0.125*out2[l]/((double)(Nx+1))/((double)(Ny+1))/((double)(Nz+1));
				erl1 +=pow(fabs(phi[i][j][k] - phi_test[i][j][k]),2);
				erl2 +=pow(fabs(phi[i][j][k]),2);
				erl3 +=pow(fabs(phi_test[i][j][k]),2);
				
				
				if ((l > world.ni*world.nj*world.nk/2) && (l < world.ni*world.nj*world.nk/2 + world.nk*2))
				{
					//cout<< setprecision(7) << "phi[" << i << "][" << j << "]["<<k<<"] = " << tmp[i+1][j+1][k+1] << " , phiF[" << i << "][" << j << "]["<<k<<"] = " << phiF[i+1][j+1][k+1] << "\n";
					cout<< setprecision(7) << "phiMG[" << i << "][" << j << "]["<<k<<"] = " << phi[i][j][k] << " , phi[" << i << "][" << j << "]["<<k<<"] = " << phi_test[i][j][k] << "\n";
				}
				l++;
												
			}
    cout<< setprecision(7) << "\n phi - phi_test error=" <<sqrt(erl1) << " , " << sqrt(erl2) << " , " << sqrt(erl3) << endl ;  
	*/

    if (!converged) cerr<<"GS failed to converge, L2="<<L2<<endl;
    return converged;
}

/*solves Poisson equation using MultiGrid*/
bool MultiGridSolverV3B::solve()
{
    //references to avoid having to write world.phi
	Field &phi = world.phi;
    Field &rho = world.rho;

	/*
	for (int i=0;i<world.ni;i++)
		for (int j=0;j<world.nj;j++)
			for (int k=0;k<world.nk;k++)
			{
				phi_test[i][j][k] = phi[i][j][k];
			}
	*/

	//precompute 1/(dx^2)
    double3 dh = world.getDh();
    double idx2 = 1.0/(dh[0]*dh[0]);
    double idy2 = 1.0/(dh[1]*dh[1]);
    double idz2 = 1.0/(dh[2]*dh[2]);

    double idx2_2h = 1.0/(4*dh[0]*dh[0]);
    double idy2_2h = 1.0/(4*dh[1]*dh[1]);
    double idz2_2h = 1.0/(4*dh[2]*dh[2]);

    double idx2_4h = 1.0/(16*dh[0]*dh[0]);
    double idy2_4h = 1.0/(16*dh[1]*dh[1]);
    double idz2_4h = 1.0/(16*dh[2]*dh[2]);

    double idx2_8h = 1.0/(64*dh[0]*dh[0]);
    double idy2_8h = 1.0/(64*dh[1]*dh[1]);
    double idz2_8h = 1.0/(64*dh[2]*dh[2]);
	
    double L2=0;			//norm
    bool converged= false;

    /*solve potential*/
    for (unsigned it=0;it<max_solver_it;it++)
    {
        //number of steps to iterate at the finest level
        unsigned inner_its = 3;

        //number of steps to iterate at the coarse level        
        unsigned inner2h_its = 5;

        unsigned inner4h_its = 5;

        unsigned inner8h_its = 50;

        // 1) perform one or more iterations on fine mesh
		for (unsigned its=0;its<inner_its;its++)
		{
			 for (int i=1;i<world.ni-1;i++)
				for (int j=1;j<world.nj-1;j++)
					for (int k=1;k<world.nk-1;k++)
					{
						//standard internal open node
						double phi_new = (rho[i][j][k]/Const::EPS_0 +
										idx2*(phi[i-1][j][k] + phi[i+1][j][k]) +
										idy2*(phi[i][j-1][k]+phi[i][j+1][k]) +
										idz2*(phi[i][j][k-1]+phi[i][j][k+1]))/(2*idx2+2*idy2+2*idz2);

						/*SOR*/
						phi[i][j][k] = phi[i][j][k] + 1.4*(phi_new-phi[i][j][k]);
						
					}
		}
				
        // 2) compute residue on the fine mesh, R = A*phi - b
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					if ((i == 0) || (i == world.ni-1) || (j == 0) || (j == world.nj-1) || (k == 0) || (k == world.nk-1)) {
						R_h[i][j][k] = phi[i][j][k] - 0;  // dirichlet boundary
					} else {
						R_h[i][j][k] = phi[i][j][k]*(2*idx2+2*idy2+2*idz2) -
								rho[i][j][k]/Const::EPS_0 -
								idx2*(phi[i-1][j][k] + phi[i+1][j][k]) -
								idy2*(phi[i][j-1][k]+phi[i][j+1][k]) -
								idz2*(phi[i][j][k-1]+phi[i][j][k+1]);
					}
				}
                    
        // 2b) check for termination

		 /*check for convergence*/
		double sum = 0;
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					sum += R_h[i][j][k]*R_h[i][j][k];
				}

		L2 = sqrt(sum/(world.ni*world.nj*world.nk));
		if (L2<tolerance) {converged=true;break;}
		
        // 3) restrict residue to the 2h mesh
		for (int i=0;i<world.ni-1;i+=2)
			for (int j=0;j<world.nj-1;j+=2)
				for (int k=0;k<world.nk-1;k+=2)
				{
					if ((i == 0) || (j == 0)  || (k == 0) ) {
						R_2h[i/2][j/2][k/2] = R_h[i][j][k];  //R_2h[0] = R_h[0]
					} else {
						R_2h[i/2][j/2][k/2] = (R_h[i-1][j][k] + 2*R_h[i][j][k] + R_h[i+1][j][k] +
											   R_h[i][j-1][k] + 2*R_h[i][j][k] + R_h[i][j+1][k] +
											   R_h[i][j][k-1] + 2*R_h[i][j][k] + R_h[i][j][k+1] )/12.0;
					}
				}
		
         // 3) restrict residue to the 4h mesh
		for (int i=0;i<world.ni/2-1;i+=2)
			for (int j=0;j<world.nj/2-1;j+=2)
				for (int k=0;k<world.nk/2-1;k+=2)
				{
					if ((i == 0) || (j == 0) || (k == 0) ) {
						R_4h[i/2][j/2][k/2] = R_2h[i][j][k];  // dirichlet boundary
					} else {
						R_4h[i/2][j/2][k/2] = (R_2h[i-1][j][k] + 2*R_2h[i][j][k] + R_2h[i+1][j][k] +
											   R_2h[i][j-1][k] + 2*R_2h[i][j][k] + R_2h[i][j+1][k] +
											   R_2h[i][j][k-1] + 2*R_2h[i][j][k] + R_2h[i][j][k+1] )/12.0;
					}
				}
           
         // 3) restrict residue to the 8h mesh
		for (int i=0;i<world.ni/4-1;i+=2)
			for (int j=0;j<world.nj/4-1;j+=2)
				for (int k=0;k<world.nk/4-1;k+=2)
				{
					if ((i == 0) || (j == 0) || (k == 0) ) {
						R_8h[i/2][j/2][k/2] = R_4h[i][j][k];  // dirichlet boundary
					} else {
						R_8h[i/2][j/2][k/2] = (R_4h[i-1][j][k] + 2*R_4h[i][j][k] + R_4h[i+1][j][k] +
											   R_4h[i][j-1][k] + 2*R_4h[i][j][k] + R_4h[i][j+1][k] +
											   R_4h[i][j][k-1] + 2*R_4h[i][j][k] + R_4h[i][j][k+1] )/12.0;
					}
				}
           
		// 4) perform few iteration of the correction vector on the 8h mesh
		for (unsigned its=0;its<inner8h_its;its++)
		{
			 for (int i=1;i<world.ni/8-1;i++)
				for (int j=1;j<world.nj/8-1;j++)
					for (int k=1;k<world.nk/8-1;k++)
					{
						//standard internal open node
						double g = (R_8h[i][j][k] +
										idx2_8h*(eps_8h[i-1][j][k] + eps_8h[i+1][j][k]) +
										idy2_8h*(eps_8h[i][j-1][k]+eps_8h[i][j+1][k]) +
										idz2_8h*(eps_8h[i][j][k-1]+eps_8h[i][j][k+1]))/(2*idx2_8h+2*idy2_8h+2*idz2_8h);

						// SOR
						eps_8h[i][j][k] = eps_8h[i][j][k] + 1.4*(g-eps_8h[i][j][k]);
					}
		}

        // 5) interpolate eps to 4h mesh
		for (int i=0;i<world.ni/4-1;i++)
			for (int j=0;j<world.nj/4-1;j++)
				for (int k=0;k<world.nk/4-1;k++)
				{
					if ((i == 0) || (i == world.ni/4-1) || (j == 0) || (j == world.nj/4-1) || (k == 0) || (k == world.nk/4-1)) {
						eps_4h[i][j][k] = 0;    // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_4h[i][j][k] = eps_8h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_4h[i][j][k] = 0.5*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_4h[i][j][k] = 0.5*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_4h[i][j][k] = 0.5*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_4h[i][j][k] = 0.25*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2 + 1][j/2][k/2] +
						                  eps_8h[i/2][j/2+1][k/2] + eps_8h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_4h[i][j][k] = 0.25*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2+1][j/2][k/2] +
						                  eps_8h[i/2][j/2][k/2+1] + eps_8h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_4h[i][j][k] = 0.25*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2][j/2+1][k/2] +
						                  eps_8h[i/2][j/2][k/2+1] + eps_8h[i/2][j/2+1][k/2+1]);
					} else {
						eps_4h[i][j][k] = 0.125*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2 + 1][j/2][k/2] +
						                  eps_8h[i/2][j/2+1][k/2] + eps_8h[i/2 + 1][j/2+1][k/2] +
						                  eps_8h[i/2][j/2][k/2+1] + eps_8h[i/2 + 1][j/2][k/2+1] +
						                  eps_8h[i/2][j/2+1][k/2+1] + eps_8h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
		// 4) perform few iteration of the correction vector on the 4h mesh
		for (unsigned its=0;its<inner4h_its;its++)
		{
			 for (int i=1;i<world.ni/4-1;i++)
				for (int j=1;j<world.nj/4-1;j++)
					for (int k=1;k<world.nk/4-1;k++)
					{
						//standard internal open node
						double g = (R_4h[i][j][k] +
										idx2_4h*(eps_4h[i-1][j][k] + eps_4h[i+1][j][k]) +
										idy2_4h*(eps_4h[i][j-1][k]+eps_4h[i][j+1][k]) +
										idz2_4h*(eps_4h[i][j][k-1]+eps_4h[i][j][k+1]))/(2*idx2_4h+2*idy2_4h+2*idz2_4h);

						/*SOR*/
						eps_4h[i][j][k] = eps_4h[i][j][k] + 1.4*(g-eps_4h[i][j][k]);
					}
		}

        // 5) interpolate eps to 2h mesh
		for (int i=0;i<world.ni/2-1;i++)
			for (int j=0;j<world.nj/2-1;j++)
				for (int k=0;k<world.nk/2-1;k++)
				{
					if ((i == 0) || (i == world.ni/2-1) || (j == 0) || (j == world.nj/2-1) || (k == 0) || (k == world.nk/2-1)) {
						eps_2h[i][j][k] = 0;    // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_2h[i][j][k] = eps_4h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2 + 1][j/2][k/2] +
						                  eps_4h[i/2][j/2+1][k/2] + eps_4h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2+1][j/2][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2+1][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2][j/2+1][k/2+1]);
					} else {
						eps_2h[i][j][k] = 0.125*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2 + 1][j/2][k/2] +
						                  eps_4h[i/2][j/2+1][k/2] + eps_4h[i/2 + 1][j/2+1][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2 + 1][j/2][k/2+1] +
						                  eps_4h[i/2][j/2+1][k/2+1] + eps_4h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
		// 4) perform few iteration of the correction vector on the 2h mesh
		for (unsigned its=0;its<inner2h_its;its++)
		{
			for (int i=1;i<world.ni/2-1;i++)
				for (int j=1;j<world.nj/2-1;j++)
					for (int k=1;k<world.nk/2-1;k++)
					{
						//standard internal open node
						double g = (R_2h[i][j][k] +
										idx2_2h*(eps_2h[i-1][j][k] + eps_2h[i+1][j][k]) +
										idy2_2h*(eps_2h[i][j-1][k]+eps_2h[i][j+1][k]) +
										idz2_2h*(eps_2h[i][j][k-1]+eps_2h[i][j][k+1]))/(2*idx2_2h+2*idy2_2h+2*idz2_2h);

						// SOR
						eps_2h[i][j][k] = eps_2h[i][j][k] + 1.4*(g-eps_2h[i][j][k]);
					}
		}

        // 5) interpolate eps to h mesh
		for (int i=0;i<world.ni-1;i++)
			for (int j=0;j<world.nj-1;j++)
				for (int k=0;k<world.nk-1;k++)
				{
					if ((i == 0) || (i == world.ni-1) || (j == 0) || (j == world.nj-1) || (k == 0) || (k == world.nk-1)) {
						eps_h[i][j][k] = 0;   // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_h[i][j][k] = eps_2h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2 + 1][j/2][k/2] +
						                  eps_2h[i/2][j/2+1][k/2] + eps_2h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2+1][j/2][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2+1][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2][j/2+1][k/2+1]);
					} else {
						eps_h[i][j][k] = 0.125*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2 + 1][j/2][k/2] +
						                  eps_2h[i/2][j/2+1][k/2] + eps_2h[i/2 + 1][j/2+1][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2 + 1][j/2][k/2+1] +
						                  eps_2h[i/2][j/2+1][k/2+1] + eps_2h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
        // 6) update solution on the fine mesh
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					phi[i][j][k] = phi[i][j][k] - eps_h[i][j][k];
				}

    }

	/*
	int l = 0;
    double erl1 = 0.;
    double erl2 = 0.;
    double erl3 = 0.;
	for (int i=1;i<world.ni-1;i++)
		for (int j=1;j<world.nj-1;j++)
			for (int k=1;k<world.nk-1;k++)
			{
				// consecutive ordering
				//tmp[i+1][j+1][k+1] = phi[i+1][j+1][k+1];
				//phiF[i+1][j+1][k+1] = 0.125*out2[l]/((double)(Nx+1))/((double)(Ny+1))/((double)(Nz+1));
				erl1 +=pow(fabs(phi[i][j][k] - phi_test[i][j][k]),2);
				erl2 +=pow(fabs(phi[i][j][k]),2);
				erl3 +=pow(fabs(phi_test[i][j][k]),2);
				
				
				if ((l > world.ni*world.nj*world.nk/2) && (l < world.ni*world.nj*world.nk/2 + world.nk*2))
				{
					//cout<< setprecision(7) << "phi[" << i << "][" << j << "]["<<k<<"] = " << tmp[i+1][j+1][k+1] << " , phiF[" << i << "][" << j << "]["<<k<<"] = " << phiF[i+1][j+1][k+1] << "\n";
					cout<< setprecision(7) << "phiMG[" << i << "][" << j << "]["<<k<<"] = " << phi[i][j][k] << " , phi[" << i << "][" << j << "]["<<k<<"] = " << phi_test[i][j][k] << "\n";
				}
				l++;
												
			}
    cout<< setprecision(7) << "\n phi - phi_test error=" <<sqrt(erl1) << " , " << sqrt(erl2) << " , " << sqrt(erl3) << endl ;  
	*/

    if (!converged) cerr<<"GS failed to converge, L2="<<L2<<endl;
    return converged;
}

/*solves Poisson equation using MultiGrid*/
bool MultiGridSolverV4B::solve()
{
    //references to avoid having to write world.phi
	Field &phi = world.phi;
    Field &rho = world.rho;

	/*
	for (int i=0;i<world.ni;i++)
		for (int j=0;j<world.nj;j++)
			for (int k=0;k<world.nk;k++)
			{
				phi_test[i][j][k] = phi[i][j][k];
			}
	*/

	//precompute 1/(dx^2)
    double3 dh = world.getDh();
    double idx2 = 1.0/(dh[0]*dh[0]);
    double idy2 = 1.0/(dh[1]*dh[1]);
    double idz2 = 1.0/(dh[2]*dh[2]);

    double idx2_2h = 1.0/(4*dh[0]*dh[0]);
    double idy2_2h = 1.0/(4*dh[1]*dh[1]);
    double idz2_2h = 1.0/(4*dh[2]*dh[2]);

    double idx2_4h = 1.0/(16*dh[0]*dh[0]);
    double idy2_4h = 1.0/(16*dh[1]*dh[1]);
    double idz2_4h = 1.0/(16*dh[2]*dh[2]);

    double idx2_8h = 1.0/(64*dh[0]*dh[0]);
    double idy2_8h = 1.0/(64*dh[1]*dh[1]);
    double idz2_8h = 1.0/(64*dh[2]*dh[2]);
	
    double idx2_16h = 1.0/(256*dh[0]*dh[0]);
    double idy2_16h = 1.0/(256*dh[1]*dh[1]);
    double idz2_16h = 1.0/(256*dh[2]*dh[2]);
	
    double L2=0;			//norm
    bool converged= false;

    /*solve potential*/
    for (unsigned it=0;it<max_solver_it;it++)
    {
        //number of steps to iterate at the finest level
        unsigned inner_its = 3;

        //number of steps to iterate at the coarse level        
        unsigned inner2h_its = 5;

        unsigned inner4h_its = 5;

        unsigned inner8h_its = 5;

        unsigned inner16h_its = 50;

        // 1) perform one or more iterations on fine mesh
		for (unsigned its=0;its<inner_its;its++)
		{
			 for (int i=1;i<world.ni-1;i++)
				for (int j=1;j<world.nj-1;j++)
					for (int k=1;k<world.nk-1;k++)
					{
						//standard internal open node
						double phi_new = (rho[i][j][k]/Const::EPS_0 +
										idx2*(phi[i-1][j][k] + phi[i+1][j][k]) +
										idy2*(phi[i][j-1][k]+phi[i][j+1][k]) +
										idz2*(phi[i][j][k-1]+phi[i][j][k+1]))/(2*idx2+2*idy2+2*idz2);

						/*SOR*/
						phi[i][j][k] = phi[i][j][k] + 1.4*(phi_new-phi[i][j][k]);
						
					}
		}
				
        // 2) compute residue on the fine mesh, R = A*phi - b
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					if ((i == 0) || (i == world.ni-1) || (j == 0) || (j == world.nj-1) || (k == 0) || (k == world.nk-1)) {
						R_h[i][j][k] = phi[i][j][k] - 0;  // dirichlet boundary
					} else {
						R_h[i][j][k] = phi[i][j][k]*(2*idx2+2*idy2+2*idz2) -
								rho[i][j][k]/Const::EPS_0 -
								idx2*(phi[i-1][j][k] + phi[i+1][j][k]) -
								idy2*(phi[i][j-1][k]+phi[i][j+1][k]) -
								idz2*(phi[i][j][k-1]+phi[i][j][k+1]);
					}
				}
                    
        // 2b) check for termination

		 /*check for convergence*/
		double sum = 0;
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					sum += R_h[i][j][k]*R_h[i][j][k];
				}

		L2 = sqrt(sum/(world.ni*world.nj*world.nk));
		if (L2<tolerance) {converged=true;break;}
		
        // 3) restrict residue to the 2h mesh
		for (int i=0;i<world.ni-1;i+=2)
			for (int j=0;j<world.nj-1;j+=2)
				for (int k=0;k<world.nk-1;k+=2)
				{
					if ((i == 0) || (j == 0)  || (k == 0) ) {
						R_2h[i/2][j/2][k/2] = R_h[i][j][k];  //R_2h[0] = R_h[0]
					} else {
						R_2h[i/2][j/2][k/2] = (R_h[i-1][j][k] + 2*R_h[i][j][k] + R_h[i+1][j][k] +
											   R_h[i][j-1][k] + 2*R_h[i][j][k] + R_h[i][j+1][k] +
											   R_h[i][j][k-1] + 2*R_h[i][j][k] + R_h[i][j][k+1] )/12.0;
					}
				}
		
         // 3) restrict residue to the 4h mesh
		for (int i=0;i<world.ni/2-1;i+=2)
			for (int j=0;j<world.nj/2-1;j+=2)
				for (int k=0;k<world.nk/2-1;k+=2)
				{
					if ((i == 0) || (j == 0) || (k == 0) ) {
						R_4h[i/2][j/2][k/2] = R_2h[i][j][k];  // dirichlet boundary
					} else {
						R_4h[i/2][j/2][k/2] = (R_2h[i-1][j][k] + 2*R_2h[i][j][k] + R_2h[i+1][j][k] +
											   R_2h[i][j-1][k] + 2*R_2h[i][j][k] + R_2h[i][j+1][k] +
											   R_2h[i][j][k-1] + 2*R_2h[i][j][k] + R_2h[i][j][k+1] )/12.0;
					}
				}
           
         // 3) restrict residue to the 8h mesh
		for (int i=0;i<world.ni/4-1;i+=2)
			for (int j=0;j<world.nj/4-1;j+=2)
				for (int k=0;k<world.nk/4-1;k+=2)
				{
					if ((i == 0) || (j == 0) || (k == 0) ) {
						R_8h[i/2][j/2][k/2] = R_4h[i][j][k];  // dirichlet boundary
					} else {
						R_8h[i/2][j/2][k/2] = (R_4h[i-1][j][k] + 2*R_4h[i][j][k] + R_4h[i+1][j][k] +
											   R_4h[i][j-1][k] + 2*R_4h[i][j][k] + R_4h[i][j+1][k] +
											   R_4h[i][j][k-1] + 2*R_4h[i][j][k] + R_4h[i][j][k+1] )/12.0;
					}
				}
           
         // 3) restrict residue to the 16h mesh
		for (int i=0;i<world.ni/8-1;i+=2)
			for (int j=0;j<world.nj/8-1;j+=2)
				for (int k=0;k<world.nk/8-1;k+=2)
				{
					if ((i == 0) || (j == 0) || (k == 0) ) {
						R_16h[i/2][j/2][k/2] = R_8h[i][j][k];  // dirichlet boundary
					} else {
						R_16h[i/2][j/2][k/2] = (R_8h[i-1][j][k] + 2*R_8h[i][j][k] + R_8h[i+1][j][k] +
											   R_8h[i][j-1][k] + 2*R_8h[i][j][k] + R_8h[i][j+1][k] +
											   R_8h[i][j][k-1] + 2*R_8h[i][j][k] + R_8h[i][j][k+1] )/12.0;
					}
				}

		// 4) perform few iteration of the correction vector on the 16h mesh
		for (unsigned its=0;its<inner16h_its;its++)
		{
			 for (int i=1;i<world.ni/16-1;i++)
				for (int j=1;j<world.nj/16-1;j++)
					for (int k=1;k<world.nk/16-1;k++)
					{
						//standard internal open node
						double g = (R_16h[i][j][k] +
										idx2_16h*(eps_16h[i-1][j][k] + eps_16h[i+1][j][k]) +
										idy2_16h*(eps_16h[i][j-1][k]+eps_16h[i][j+1][k]) +
										idz2_16h*(eps_16h[i][j][k-1]+eps_16h[i][j][k+1]))/(2*idx2_16h+2*idy2_16h+2*idz2_16h);

						// SOR
						eps_16h[i][j][k] = eps_16h[i][j][k] + 1.4*(g-eps_16h[i][j][k]);
					}
		}

        // 5) interpolate eps to 8h mesh
		for (int i=0;i<world.ni/8-1;i++)
			for (int j=0;j<world.nj/8-1;j++)
				for (int k=0;k<world.nk/8-1;k++)
				{
					if ((i == 0) || (i == world.ni/4-1) || (j == 0) || (j == world.nj/4-1) || (k == 0) || (k == world.nk/4-1)) {
						eps_8h[i][j][k] = 0;    // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_8h[i][j][k] = eps_16h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_8h[i][j][k] = 0.5*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_8h[i][j][k] = 0.5*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_8h[i][j][k] = 0.5*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_8h[i][j][k] = 0.25*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2 + 1][j/2][k/2] +
						                  eps_16h[i/2][j/2+1][k/2] + eps_16h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_8h[i][j][k] = 0.25*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2+1][j/2][k/2] +
						                  eps_16h[i/2][j/2][k/2+1] + eps_16h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_8h[i][j][k] = 0.25*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2][j/2+1][k/2] +
						                  eps_16h[i/2][j/2][k/2+1] + eps_16h[i/2][j/2+1][k/2+1]);
					} else {
						eps_8h[i][j][k] = 0.125*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2 + 1][j/2][k/2] +
						                  eps_16h[i/2][j/2+1][k/2] + eps_16h[i/2 + 1][j/2+1][k/2] +
						                  eps_16h[i/2][j/2][k/2+1] + eps_16h[i/2 + 1][j/2][k/2+1] +
						                  eps_16h[i/2][j/2+1][k/2+1] + eps_16h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
           
		// 4) perform few iteration of the correction vector on the 8h mesh
		for (unsigned its=0;its<inner8h_its;its++)
		{
			 for (int i=1;i<world.ni/8-1;i++)
				for (int j=1;j<world.nj/8-1;j++)
					for (int k=1;k<world.nk/8-1;k++)
					{
						//standard internal open node
						double g = (R_8h[i][j][k] +
										idx2_8h*(eps_8h[i-1][j][k] + eps_8h[i+1][j][k]) +
										idy2_8h*(eps_8h[i][j-1][k]+eps_8h[i][j+1][k]) +
										idz2_8h*(eps_8h[i][j][k-1]+eps_8h[i][j][k+1]))/(2*idx2_8h+2*idy2_8h+2*idz2_8h);

						// SOR
						eps_8h[i][j][k] = eps_8h[i][j][k] + 1.4*(g-eps_8h[i][j][k]);
					}
		}

        // 5) interpolate eps to 4h mesh
		for (int i=0;i<world.ni/4-1;i++)
			for (int j=0;j<world.nj/4-1;j++)
				for (int k=0;k<world.nk/4-1;k++)
				{
					if ((i == 0) || (i == world.ni/4-1) || (j == 0) || (j == world.nj/4-1) || (k == 0) || (k == world.nk/4-1)) {
						eps_4h[i][j][k] = 0;    // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_4h[i][j][k] = eps_8h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_4h[i][j][k] = 0.5*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_4h[i][j][k] = 0.5*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_4h[i][j][k] = 0.5*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_4h[i][j][k] = 0.25*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2 + 1][j/2][k/2] +
						                  eps_8h[i/2][j/2+1][k/2] + eps_8h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_4h[i][j][k] = 0.25*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2+1][j/2][k/2] +
						                  eps_8h[i/2][j/2][k/2+1] + eps_8h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_4h[i][j][k] = 0.25*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2][j/2+1][k/2] +
						                  eps_8h[i/2][j/2][k/2+1] + eps_8h[i/2][j/2+1][k/2+1]);
					} else {
						eps_4h[i][j][k] = 0.125*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2 + 1][j/2][k/2] +
						                  eps_8h[i/2][j/2+1][k/2] + eps_8h[i/2 + 1][j/2+1][k/2] +
						                  eps_8h[i/2][j/2][k/2+1] + eps_8h[i/2 + 1][j/2][k/2+1] +
						                  eps_8h[i/2][j/2+1][k/2+1] + eps_8h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
		// 4) perform few iteration of the correction vector on the 4h mesh
		for (unsigned its=0;its<inner4h_its;its++)
		{
			 for (int i=1;i<world.ni/4-1;i++)
				for (int j=1;j<world.nj/4-1;j++)
					for (int k=1;k<world.nk/4-1;k++)
					{
						//standard internal open node
						double g = (R_4h[i][j][k] +
										idx2_4h*(eps_4h[i-1][j][k] + eps_4h[i+1][j][k]) +
										idy2_4h*(eps_4h[i][j-1][k]+eps_4h[i][j+1][k]) +
										idz2_4h*(eps_4h[i][j][k-1]+eps_4h[i][j][k+1]))/(2*idx2_4h+2*idy2_4h+2*idz2_4h);

						/*SOR*/
						eps_4h[i][j][k] = eps_4h[i][j][k] + 1.4*(g-eps_4h[i][j][k]);
					}
		}

        // 5) interpolate eps to 2h mesh
		for (int i=0;i<world.ni/2-1;i++)
			for (int j=0;j<world.nj/2-1;j++)
				for (int k=0;k<world.nk/2-1;k++)
				{
					if ((i == 0) || (i == world.ni/2-1) || (j == 0) || (j == world.nj/2-1) || (k == 0) || (k == world.nk/2-1)) {
						eps_2h[i][j][k] = 0;    // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_2h[i][j][k] = eps_4h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2 + 1][j/2][k/2] +
						                  eps_4h[i/2][j/2+1][k/2] + eps_4h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2+1][j/2][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2+1][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2][j/2+1][k/2+1]);
					} else {
						eps_2h[i][j][k] = 0.125*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2 + 1][j/2][k/2] +
						                  eps_4h[i/2][j/2+1][k/2] + eps_4h[i/2 + 1][j/2+1][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2 + 1][j/2][k/2+1] +
						                  eps_4h[i/2][j/2+1][k/2+1] + eps_4h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
		// 4) perform few iteration of the correction vector on the 2h mesh
		for (unsigned its=0;its<inner2h_its;its++)
		{
			for (int i=1;i<world.ni/2-1;i++)
				for (int j=1;j<world.nj/2-1;j++)
					for (int k=1;k<world.nk/2-1;k++)
					{
						//standard internal open node
						double g = (R_2h[i][j][k] +
										idx2_2h*(eps_2h[i-1][j][k] + eps_2h[i+1][j][k]) +
										idy2_2h*(eps_2h[i][j-1][k]+eps_2h[i][j+1][k]) +
										idz2_2h*(eps_2h[i][j][k-1]+eps_2h[i][j][k+1]))/(2*idx2_2h+2*idy2_2h+2*idz2_2h);

						// SOR
						eps_2h[i][j][k] = eps_2h[i][j][k] + 1.4*(g-eps_2h[i][j][k]);
					}
		}

        // 5) interpolate eps to h mesh
		for (int i=0;i<world.ni-1;i++)
			for (int j=0;j<world.nj-1;j++)
				for (int k=0;k<world.nk-1;k++)
				{
					if ((i == 0) || (i == world.ni-1) || (j == 0) || (j == world.nj-1) || (k == 0) || (k == world.nk-1)) {
						eps_h[i][j][k] = 0;   // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_h[i][j][k] = eps_2h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2 + 1][j/2][k/2] +
						                  eps_2h[i/2][j/2+1][k/2] + eps_2h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2+1][j/2][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2+1][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2][j/2+1][k/2+1]);
					} else {
						eps_h[i][j][k] = 0.125*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2 + 1][j/2][k/2] +
						                  eps_2h[i/2][j/2+1][k/2] + eps_2h[i/2 + 1][j/2+1][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2 + 1][j/2][k/2+1] +
						                  eps_2h[i/2][j/2+1][k/2+1] + eps_2h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
        // 6) update solution on the fine mesh
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					phi[i][j][k] = phi[i][j][k] - eps_h[i][j][k];
				}

    }

	/*
	int l = 0;
    double erl1 = 0.;
    double erl2 = 0.;
    double erl3 = 0.;
	for (int i=1;i<world.ni-1;i++)
		for (int j=1;j<world.nj-1;j++)
			for (int k=1;k<world.nk-1;k++)
			{
				// consecutive ordering
				//tmp[i+1][j+1][k+1] = phi[i+1][j+1][k+1];
				//phiF[i+1][j+1][k+1] = 0.125*out2[l]/((double)(Nx+1))/((double)(Ny+1))/((double)(Nz+1));
				erl1 +=pow(fabs(phi[i][j][k] - phi_test[i][j][k]),2);
				erl2 +=pow(fabs(phi[i][j][k]),2);
				erl3 +=pow(fabs(phi_test[i][j][k]),2);
				
				
				if ((l > world.ni*world.nj*world.nk/2) && (l < world.ni*world.nj*world.nk/2 + world.nk*2))
				{
					//cout<< setprecision(7) << "phi[" << i << "][" << j << "]["<<k<<"] = " << tmp[i+1][j+1][k+1] << " , phiF[" << i << "][" << j << "]["<<k<<"] = " << phiF[i+1][j+1][k+1] << "\n";
					cout<< setprecision(7) << "phiMG[" << i << "][" << j << "]["<<k<<"] = " << phi[i][j][k] << " , phi[" << i << "][" << j << "]["<<k<<"] = " << phi_test[i][j][k] << "\n";
				}
				l++;
												
			}
    cout<< setprecision(7) << "\n phi - phi_test error=" <<sqrt(erl1) << " , " << sqrt(erl2) << " , " << sqrt(erl3) << endl ;  
	*/

    if (!converged) cerr<<"GS failed to converge, L2="<<L2<<endl;
    return converged;
}

/*solves Poisson equation using MultiGrid*/
bool MultiGridSolverV5B::solve()
{
    //references to avoid having to write world.phi
	Field &phi = world.phi;
    Field &rho = world.rho;

	/*
	for (int i=0;i<world.ni;i++)
		for (int j=0;j<world.nj;j++)
			for (int k=0;k<world.nk;k++)
			{
				phi_test[i][j][k] = phi[i][j][k];
			}
	*/

	//precompute 1/(dx^2)
    double3 dh = world.getDh();
    double idx2 = 1.0/(dh[0]*dh[0]);
    double idy2 = 1.0/(dh[1]*dh[1]);
    double idz2 = 1.0/(dh[2]*dh[2]);

    double idx2_2h = 1.0/(4*dh[0]*dh[0]);
    double idy2_2h = 1.0/(4*dh[1]*dh[1]);
    double idz2_2h = 1.0/(4*dh[2]*dh[2]);

    double idx2_4h = 1.0/(16*dh[0]*dh[0]);
    double idy2_4h = 1.0/(16*dh[1]*dh[1]);
    double idz2_4h = 1.0/(16*dh[2]*dh[2]);

    double idx2_8h = 1.0/(64*dh[0]*dh[0]);
    double idy2_8h = 1.0/(64*dh[1]*dh[1]);
    double idz2_8h = 1.0/(64*dh[2]*dh[2]);
	
    double idx2_16h = 1.0/(256*dh[0]*dh[0]);
    double idy2_16h = 1.0/(256*dh[1]*dh[1]);
    double idz2_16h = 1.0/(256*dh[2]*dh[2]);
	
    double idx2_32h = 1.0/(1024*dh[0]*dh[0]);
    double idy2_32h = 1.0/(1024*dh[1]*dh[1]);
    double idz2_32h = 1.0/(1024*dh[2]*dh[2]);
	
    double L2=0;			//norm
    bool converged= false;

    /*solve potential*/
    for (unsigned it=0;it<max_solver_it;it++)
    {
        //number of steps to iterate at the finest level
        unsigned inner_its = 1;

        //number of steps to iterate at the coarse level        
        unsigned inner2h_its = 2;

        unsigned inner4h_its = 3;

        unsigned inner8h_its = 4;

        unsigned inner16h_its = 5;

        unsigned inner32h_its = 50;

        // 1) perform one or more iterations on fine mesh
		for (unsigned its=0;its<inner_its;its++)
		{
			 for (int i=1;i<world.ni-1;i++)
				for (int j=1;j<world.nj-1;j++)
					for (int k=1;k<world.nk-1;k++)
					{
						//standard internal open node
						double phi_new = (rho[i][j][k]/Const::EPS_0 +
										idx2*(phi[i-1][j][k] + phi[i+1][j][k]) +
										idy2*(phi[i][j-1][k]+phi[i][j+1][k]) +
										idz2*(phi[i][j][k-1]+phi[i][j][k+1]))/(2*idx2+2*idy2+2*idz2);

						/*SOR*/
						phi[i][j][k] = phi[i][j][k] + 1.4*(phi_new-phi[i][j][k]);
						
					}
		}
				
        // 2) compute residue on the fine mesh, R = A*phi - b
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					if ((i == 0) || (i == world.ni-1) || (j == 0) || (j == world.nj-1) || (k == 0) || (k == world.nk-1)) {
						R_h[i][j][k] = phi[i][j][k] - 0;  // dirichlet boundary
					} else {
						R_h[i][j][k] = phi[i][j][k]*(2*idx2+2*idy2+2*idz2) -
								rho[i][j][k]/Const::EPS_0 -
								idx2*(phi[i-1][j][k] + phi[i+1][j][k]) -
								idy2*(phi[i][j-1][k]+phi[i][j+1][k]) -
								idz2*(phi[i][j][k-1]+phi[i][j][k+1]);
					}
				}
                    
        // 2b) check for termination

		 /*check for convergence*/
		double sum = 0;
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					sum += R_h[i][j][k]*R_h[i][j][k];
				}

		L2 = sqrt(sum/(world.ni*world.nj*world.nk));
		if (L2<tolerance) {converged=true;break;}
		
        // 3) restrict residue to the 2h mesh
		for (int i=0;i<world.ni-1;i+=2)
			for (int j=0;j<world.nj-1;j+=2)
				for (int k=0;k<world.nk-1;k+=2)
				{
					if ((i == 0) || (j == 0)  || (k == 0) ) {
						R_2h[i/2][j/2][k/2] = R_h[i][j][k];  //R_2h[0] = R_h[0]
					} else {
						R_2h[i/2][j/2][k/2] = (R_h[i-1][j][k] + 2*R_h[i][j][k] + R_h[i+1][j][k] +
											   R_h[i][j-1][k] + 2*R_h[i][j][k] + R_h[i][j+1][k] +
											   R_h[i][j][k-1] + 2*R_h[i][j][k] + R_h[i][j][k+1] )/12.0;
					}
				}
		
         // 3) restrict residue to the 4h mesh
		for (int i=0;i<world.ni/2-1;i+=2)
			for (int j=0;j<world.nj/2-1;j+=2)
				for (int k=0;k<world.nk/2-1;k+=2)
				{
					if ((i == 0) || (j == 0) || (k == 0) ) {
						R_4h[i/2][j/2][k/2] = R_2h[i][j][k];  // dirichlet boundary
					} else {
						R_4h[i/2][j/2][k/2] = (R_2h[i-1][j][k] + 2*R_2h[i][j][k] + R_2h[i+1][j][k] +
											   R_2h[i][j-1][k] + 2*R_2h[i][j][k] + R_2h[i][j+1][k] +
											   R_2h[i][j][k-1] + 2*R_2h[i][j][k] + R_2h[i][j][k+1] )/12.0;
					}
				}
           
         // 3) restrict residue to the 8h mesh
		for (int i=0;i<world.ni/4-1;i+=2)
			for (int j=0;j<world.nj/4-1;j+=2)
				for (int k=0;k<world.nk/4-1;k+=2)
				{
					if ((i == 0) || (j == 0) || (k == 0) ) {
						R_8h[i/2][j/2][k/2] = R_4h[i][j][k];  // dirichlet boundary
					} else {
						R_8h[i/2][j/2][k/2] = (R_4h[i-1][j][k] + 2*R_4h[i][j][k] + R_4h[i+1][j][k] +
											   R_4h[i][j-1][k] + 2*R_4h[i][j][k] + R_4h[i][j+1][k] +
											   R_4h[i][j][k-1] + 2*R_4h[i][j][k] + R_4h[i][j][k+1] )/12.0;
					}
				}
           
         // 3) restrict residue to the 16h mesh
		for (int i=0;i<world.ni/8-1;i+=2)
			for (int j=0;j<world.nj/8-1;j+=2)
				for (int k=0;k<world.nk/8-1;k+=2)
				{
					if ((i == 0) || (j == 0) || (k == 0) ) {
						R_16h[i/2][j/2][k/2] = R_8h[i][j][k];  // dirichlet boundary
					} else {
						R_16h[i/2][j/2][k/2] = (R_8h[i-1][j][k] + 2*R_8h[i][j][k] + R_8h[i+1][j][k] +
											   R_8h[i][j-1][k] + 2*R_8h[i][j][k] + R_8h[i][j+1][k] +
											   R_8h[i][j][k-1] + 2*R_8h[i][j][k] + R_8h[i][j][k+1] )/12.0;
					}
				}

         // 3) restrict residue to the 32h mesh
		for (int i=0;i<world.ni/16-1;i+=2)
			for (int j=0;j<world.nj/16-1;j+=2)
				for (int k=0;k<world.nk/16-1;k+=2)
				{
					if ((i == 0) || (j == 0) || (k == 0) ) {
						R_32h[i/2][j/2][k/2] = R_16h[i][j][k];  // dirichlet boundary
					} else {
						R_32h[i/2][j/2][k/2] = (R_16h[i-1][j][k] + 2*R_16h[i][j][k] + R_16h[i+1][j][k] +
											   R_16h[i][j-1][k] + 2*R_16h[i][j][k] + R_16h[i][j+1][k] +
											   R_16h[i][j][k-1] + 2*R_16h[i][j][k] + R_16h[i][j][k+1] )/12.0;
					}
				}

		// 4) perform few iteration of the correction vector on the 32h mesh
		for (unsigned its=0;its<inner32h_its;its++)
		{
			 for (int i=1;i<world.ni/32-1;i++)
				for (int j=1;j<world.nj/32-1;j++)
					for (int k=1;k<world.nk/32-1;k++)
					{
						//standard internal open node
						double g = (R_32h[i][j][k] +
										idx2_32h*(eps_32h[i-1][j][k] + eps_32h[i+1][j][k]) +
										idy2_32h*(eps_32h[i][j-1][k]+eps_32h[i][j+1][k]) +
										idz2_32h*(eps_32h[i][j][k-1]+eps_32h[i][j][k+1]))/(2*idx2_32h+2*idy2_32h+2*idz2_32h);

						// SOR
						eps_32h[i][j][k] = eps_32h[i][j][k] + 1.4*(g-eps_32h[i][j][k]);
					}
		}

        // 5) interpolate eps to 16h mesh
		for (int i=0;i<world.ni/16-1;i++)
			for (int j=0;j<world.nj/16-1;j++)
				for (int k=0;k<world.nk/16-1;k++)
				{
					if ((i == 0) || (i == world.ni/4-1) || (j == 0) || (j == world.nj/4-1) || (k == 0) || (k == world.nk/4-1)) {
						eps_16h[i][j][k] = 0;    // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_16h[i][j][k] = eps_32h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_16h[i][j][k] = 0.5*(eps_32h[i/2][j/2][k/2] + eps_32h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_16h[i][j][k] = 0.5*(eps_32h[i/2][j/2][k/2] + eps_32h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_16h[i][j][k] = 0.5*(eps_32h[i/2][j/2][k/2] + eps_32h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_16h[i][j][k] = 0.25*(eps_32h[i/2][j/2][k/2] + eps_32h[i/2 + 1][j/2][k/2] +
						                  eps_32h[i/2][j/2+1][k/2] + eps_32h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_16h[i][j][k] = 0.25*(eps_32h[i/2][j/2][k/2] + eps_32h[i/2+1][j/2][k/2] +
						                  eps_32h[i/2][j/2][k/2+1] + eps_32h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_16h[i][j][k] = 0.25*(eps_32h[i/2][j/2][k/2] + eps_32h[i/2][j/2+1][k/2] +
						                  eps_32h[i/2][j/2][k/2+1] + eps_32h[i/2][j/2+1][k/2+1]);
					} else {
						eps_16h[i][j][k] = 0.125*(eps_32h[i/2][j/2][k/2] + eps_32h[i/2 + 1][j/2][k/2] +
						                  eps_32h[i/2][j/2+1][k/2] + eps_32h[i/2 + 1][j/2+1][k/2] +
						                  eps_32h[i/2][j/2][k/2+1] + eps_32h[i/2 + 1][j/2][k/2+1] +
						                  eps_32h[i/2][j/2+1][k/2+1] + eps_32h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
           
		// 4) perform few iteration of the correction vector on the 16h mesh
		for (unsigned its=0;its<inner16h_its;its++)
		{
			 for (int i=1;i<world.ni/16-1;i++)
				for (int j=1;j<world.nj/16-1;j++)
					for (int k=1;k<world.nk/16-1;k++)
					{
						//standard internal open node
						double g = (R_16h[i][j][k] +
										idx2_16h*(eps_16h[i-1][j][k] + eps_16h[i+1][j][k]) +
										idy2_16h*(eps_16h[i][j-1][k]+eps_16h[i][j+1][k]) +
										idz2_16h*(eps_16h[i][j][k-1]+eps_16h[i][j][k+1]))/(2*idx2_16h+2*idy2_16h+2*idz2_16h);

						// SOR
						eps_16h[i][j][k] = eps_16h[i][j][k] + 1.4*(g-eps_16h[i][j][k]);
					}
		}

        // 5) interpolate eps to 8h mesh
		for (int i=0;i<world.ni/8-1;i++)
			for (int j=0;j<world.nj/8-1;j++)
				for (int k=0;k<world.nk/8-1;k++)
				{
					if ((i == 0) || (i == world.ni/4-1) || (j == 0) || (j == world.nj/4-1) || (k == 0) || (k == world.nk/4-1)) {
						eps_8h[i][j][k] = 0;    // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_8h[i][j][k] = eps_16h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_8h[i][j][k] = 0.5*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_8h[i][j][k] = 0.5*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_8h[i][j][k] = 0.5*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_8h[i][j][k] = 0.25*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2 + 1][j/2][k/2] +
						                  eps_16h[i/2][j/2+1][k/2] + eps_16h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_8h[i][j][k] = 0.25*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2+1][j/2][k/2] +
						                  eps_16h[i/2][j/2][k/2+1] + eps_16h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_8h[i][j][k] = 0.25*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2][j/2+1][k/2] +
						                  eps_16h[i/2][j/2][k/2+1] + eps_16h[i/2][j/2+1][k/2+1]);
					} else {
						eps_8h[i][j][k] = 0.125*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2 + 1][j/2][k/2] +
						                  eps_16h[i/2][j/2+1][k/2] + eps_16h[i/2 + 1][j/2+1][k/2] +
						                  eps_16h[i/2][j/2][k/2+1] + eps_16h[i/2 + 1][j/2][k/2+1] +
						                  eps_16h[i/2][j/2+1][k/2+1] + eps_16h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
           
		// 4) perform few iteration of the correction vector on the 8h mesh
		for (unsigned its=0;its<inner8h_its;its++)
		{
			 for (int i=1;i<world.ni/8-1;i++)
				for (int j=1;j<world.nj/8-1;j++)
					for (int k=1;k<world.nk/8-1;k++)
					{
						//standard internal open node
						double g = (R_8h[i][j][k] +
										idx2_8h*(eps_8h[i-1][j][k] + eps_8h[i+1][j][k]) +
										idy2_8h*(eps_8h[i][j-1][k]+eps_8h[i][j+1][k]) +
										idz2_8h*(eps_8h[i][j][k-1]+eps_8h[i][j][k+1]))/(2*idx2_8h+2*idy2_8h+2*idz2_8h);

						// SOR
						eps_8h[i][j][k] = eps_8h[i][j][k] + 1.4*(g-eps_8h[i][j][k]);
					}
		}

        // 5) interpolate eps to 4h mesh
		for (int i=0;i<world.ni/4-1;i++)
			for (int j=0;j<world.nj/4-1;j++)
				for (int k=0;k<world.nk/4-1;k++)
				{
					if ((i == 0) || (i == world.ni/4-1) || (j == 0) || (j == world.nj/4-1) || (k == 0) || (k == world.nk/4-1)) {
						eps_4h[i][j][k] = 0;    // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_4h[i][j][k] = eps_8h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_4h[i][j][k] = 0.5*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_4h[i][j][k] = 0.5*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_4h[i][j][k] = 0.5*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_4h[i][j][k] = 0.25*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2 + 1][j/2][k/2] +
						                  eps_8h[i/2][j/2+1][k/2] + eps_8h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_4h[i][j][k] = 0.25*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2+1][j/2][k/2] +
						                  eps_8h[i/2][j/2][k/2+1] + eps_8h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_4h[i][j][k] = 0.25*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2][j/2+1][k/2] +
						                  eps_8h[i/2][j/2][k/2+1] + eps_8h[i/2][j/2+1][k/2+1]);
					} else {
						eps_4h[i][j][k] = 0.125*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2 + 1][j/2][k/2] +
						                  eps_8h[i/2][j/2+1][k/2] + eps_8h[i/2 + 1][j/2+1][k/2] +
						                  eps_8h[i/2][j/2][k/2+1] + eps_8h[i/2 + 1][j/2][k/2+1] +
						                  eps_8h[i/2][j/2+1][k/2+1] + eps_8h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
		// 4) perform few iteration of the correction vector on the 4h mesh
		for (unsigned its=0;its<inner4h_its;its++)
		{
			 for (int i=1;i<world.ni/4-1;i++)
				for (int j=1;j<world.nj/4-1;j++)
					for (int k=1;k<world.nk/4-1;k++)
					{
						//standard internal open node
						double g = (R_4h[i][j][k] +
										idx2_4h*(eps_4h[i-1][j][k] + eps_4h[i+1][j][k]) +
										idy2_4h*(eps_4h[i][j-1][k]+eps_4h[i][j+1][k]) +
										idz2_4h*(eps_4h[i][j][k-1]+eps_4h[i][j][k+1]))/(2*idx2_4h+2*idy2_4h+2*idz2_4h);

						/*SOR*/
						eps_4h[i][j][k] = eps_4h[i][j][k] + 1.4*(g-eps_4h[i][j][k]);
					}
		}

        // 5) interpolate eps to 2h mesh
		for (int i=0;i<world.ni/2-1;i++)
			for (int j=0;j<world.nj/2-1;j++)
				for (int k=0;k<world.nk/2-1;k++)
				{
					if ((i == 0) || (i == world.ni/2-1) || (j == 0) || (j == world.nj/2-1) || (k == 0) || (k == world.nk/2-1)) {
						eps_2h[i][j][k] = 0;    // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_2h[i][j][k] = eps_4h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2 + 1][j/2][k/2] +
						                  eps_4h[i/2][j/2+1][k/2] + eps_4h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2+1][j/2][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2+1][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2][j/2+1][k/2+1]);
					} else {
						eps_2h[i][j][k] = 0.125*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2 + 1][j/2][k/2] +
						                  eps_4h[i/2][j/2+1][k/2] + eps_4h[i/2 + 1][j/2+1][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2 + 1][j/2][k/2+1] +
						                  eps_4h[i/2][j/2+1][k/2+1] + eps_4h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
		// 4) perform few iteration of the correction vector on the 2h mesh
		for (unsigned its=0;its<inner2h_its;its++)
		{
			for (int i=1;i<world.ni/2-1;i++)
				for (int j=1;j<world.nj/2-1;j++)
					for (int k=1;k<world.nk/2-1;k++)
					{
						//standard internal open node
						double g = (R_2h[i][j][k] +
										idx2_2h*(eps_2h[i-1][j][k] + eps_2h[i+1][j][k]) +
										idy2_2h*(eps_2h[i][j-1][k]+eps_2h[i][j+1][k]) +
										idz2_2h*(eps_2h[i][j][k-1]+eps_2h[i][j][k+1]))/(2*idx2_2h+2*idy2_2h+2*idz2_2h);

						// SOR
						eps_2h[i][j][k] = eps_2h[i][j][k] + 1.4*(g-eps_2h[i][j][k]);
					}
		}

        // 5) interpolate eps to h mesh
		for (int i=0;i<world.ni-1;i++)
			for (int j=0;j<world.nj-1;j++)
				for (int k=0;k<world.nk-1;k++)
				{
					if ((i == 0) || (i == world.ni-1) || (j == 0) || (j == world.nj-1) || (k == 0) || (k == world.nk-1)) {
						eps_h[i][j][k] = 0;   // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_h[i][j][k] = eps_2h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2 + 1][j/2][k/2] +
						                  eps_2h[i/2][j/2+1][k/2] + eps_2h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2+1][j/2][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2+1][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2][j/2+1][k/2+1]);
					} else {
						eps_h[i][j][k] = 0.125*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2 + 1][j/2][k/2] +
						                  eps_2h[i/2][j/2+1][k/2] + eps_2h[i/2 + 1][j/2+1][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2 + 1][j/2][k/2+1] +
						                  eps_2h[i/2][j/2+1][k/2+1] + eps_2h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
        // 6) update solution on the fine mesh
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					phi[i][j][k] = phi[i][j][k] - eps_h[i][j][k];
				}

    }

	/*
	int l = 0;
    double erl1 = 0.;
    double erl2 = 0.;
    double erl3 = 0.;
	for (int i=1;i<world.ni-1;i++)
		for (int j=1;j<world.nj-1;j++)
			for (int k=1;k<world.nk-1;k++)
			{
				// consecutive ordering
				//tmp[i+1][j+1][k+1] = phi[i+1][j+1][k+1];
				//phiF[i+1][j+1][k+1] = 0.125*out2[l]/((double)(Nx+1))/((double)(Ny+1))/((double)(Nz+1));
				erl1 +=pow(fabs(phi[i][j][k] - phi_test[i][j][k]),2);
				erl2 +=pow(fabs(phi[i][j][k]),2);
				erl3 +=pow(fabs(phi_test[i][j][k]),2);
				
				
				if ((l > world.ni*world.nj*world.nk/2) && (l < world.ni*world.nj*world.nk/2 + world.nk*2))
				{
					//cout<< setprecision(7) << "phi[" << i << "][" << j << "]["<<k<<"] = " << tmp[i+1][j+1][k+1] << " , phiF[" << i << "][" << j << "]["<<k<<"] = " << phiF[i+1][j+1][k+1] << "\n";
					cout<< setprecision(7) << "phiMG[" << i << "][" << j << "]["<<k<<"] = " << phi[i][j][k] << " , phi[" << i << "][" << j << "]["<<k<<"] = " << phi_test[i][j][k] << "\n";
				}
				l++;
												
			}
    cout<< setprecision(7) << "\n phi - phi_test error=" <<sqrt(erl1) << " , " << sqrt(erl2) << " , " << sqrt(erl3) << endl ;  
	*/

    if (!converged) cerr<<"GS failed to converge, L2="<<L2<<endl;
    return converged;
}

/*solves Poisson equation using MultiGrid*/
bool MultiGridSolver::solve()
{
    //references to avoid having to write world.phi
	Field &phi = world.phi;
    Field &rho = world.rho;

	/*
	for (int i=0;i<world.ni;i++)
		for (int j=0;j<world.nj;j++)
			for (int k=0;k<world.nk;k++)
			{
				phi_test[i][j][k] = phi[i][j][k];
			}
	*/

	//precompute 1/(dx^2)
    double3 dh = world.getDh();
    double idx2 = 1.0/(dh[0]*dh[0]);
    double idy2 = 1.0/(dh[1]*dh[1]);
    double idz2 = 1.0/(dh[2]*dh[2]);

    double idx2_2h = 1.0/(4*dh[0]*dh[0]);
    double idy2_2h = 1.0/(4*dh[1]*dh[1]);
    double idz2_2h = 1.0/(4*dh[2]*dh[2]);

    double L2=0;			//norm
    bool converged= false;

    /*solve potential*/
    for (unsigned it=0;it<max_solver_it;it++)
    {
        //number of steps to iterate at the finest level
        unsigned inner_its = 3;

        //number of steps to iterate at the coarse level        
        unsigned inner2h_its = 50;

        // 1) perform one or more iterations on fine mesh
		for (unsigned its=0;its<inner_its;its++)
		{
			 for (int i=1;i<world.ni-1;i++)
				for (int j=1;j<world.nj-1;j++)
					for (int k=1;k<world.nk-1;k++)
					{
						//standard internal open node
						double phi_new = (rho[i][j][k]/Const::EPS_0 +
										idx2*(phi[i-1][j][k] + phi[i+1][j][k]) +
										idy2*(phi[i][j-1][k]+phi[i][j+1][k]) +
										idz2*(phi[i][j][k-1]+phi[i][j][k+1]))/(2*idx2+2*idy2+2*idz2);

						/*SOR*/
						phi[i][j][k] = phi[i][j][k] + 1.4*(phi_new-phi[i][j][k]);					
					}
		}
				
        // 2) compute residue on the fine mesh, R = A*phi - b
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					if ((i == 0) || (i == world.ni-1) || (j == 0) || (j == world.nj-1) || (k == 0) || (k == world.nk-1)) {
						R_h[i][j][k] = phi[i][j][k] - 0;  // dirichlet boundary
					} else {
						R_h[i][j][k] = phi[i][j][k]*(2*idx2+2*idy2+2*idz2) -
								rho[i][j][k]/Const::EPS_0 -
								idx2*(phi[i-1][j][k] + phi[i+1][j][k]) -
								idy2*(phi[i][j-1][k]+phi[i][j+1][k]) -
								idz2*(phi[i][j][k-1]+phi[i][j][k+1]);
					}
				}
                    
        // 2b) check for termination
		double sum = 0;
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					sum += R_h[i][j][k]*R_h[i][j][k];
				}

		L2 = sqrt(sum/(world.ni*world.nj*world.nk));
		if (L2<tolerance) {converged=true;break;}
		
        // 3) restrict residue to the 2h mesh
		for (int i=0;i<world.ni-1;i+=2)
			for (int j=0;j<world.nj-1;j+=2)
				for (int k=0;k<world.nk-1;k+=2)
				{
					if ((i == 0) || (j == 0)  || (k == 0) ) {
						R_2h[i/2][j/2][k/2] = R_h[i][j][k];  //R_2h[0] = R_h[0]
					} else {
						R_2h[i/2][j/2][k/2] = (R_h[i-1][j-1][k-1] + 2*R_h[i][j-1][k-1] + R_h[i+1][j-1][k-1] +
											   2*R_h[i-1][j][k-1] + 4*R_h[i][j][k-1] + 2*R_h[i+1][j][k-1] +
											   R_h[i-1][j+1][k-1] + 2*R_h[i][j+1][k-1] + R_h[i+1][j+1][k-1] +
											   2*R_h[i-1][j-1][k] + 4*R_h[i][j-1][k] + 2*R_h[i+1][j-1][k] +
											   4*R_h[i-1][j][k] + 8*R_h[i][j][k] + 4*R_h[i+1][j][k] +
											   2*R_h[i-1][j+1][k] + 4*R_h[i][j+1][k] + 2*R_h[i+1][j+1][k] +
											   R_h[i-1][j-1][k+1] + 2*R_h[i][j-1][k+1] + R_h[i+1][j-1][k+1] +
											   2*R_h[i-1][j][k+1] + 4*R_h[i][j][k+1] + 2*R_h[i+1][j][k+1] +
											   R_h[i-1][j+1][k+1] + 2*R_h[i][j+1][k+1] + R_h[i+1][j+1][k+1])/64.0;
					}
				}
        //R_2h[0] = R_h[0]
        
		// 4) perform few iteration of the correction vector on the 2h mesh
		for (unsigned its=0;its<inner2h_its;its++)
		{
			for (int i=1;i<world.ni/2-1;i++)
				for (int j=1;j<world.nj/2-1;j++)
					for (int k=1;k<world.nk/2-1;k++)
					{
						//standard internal open node
						double g = (R_2h[i][j][k] +
										idx2_2h*(eps_2h[i-1][j][k] + eps_2h[i+1][j][k]) +
										idy2_2h*(eps_2h[i][j-1][k]+eps_2h[i][j+1][k]) +
										idz2_2h*(eps_2h[i][j][k-1]+eps_2h[i][j][k+1]))/(2*idx2_2h+2*idy2_2h+2*idz2_2h);

						// SOR
						eps_2h[i][j][k] = eps_2h[i][j][k] + 1.4*(g-eps_2h[i][j][k]);
					}
		}

        // 5) interpolate eps to h mesh
		for (int i=0;i<world.ni-1;i++)
			for (int j=0;j<world.nj-1;j++)
				for (int k=0;k<world.nk-1;k++)
				{
					if ((i == 0) || (i == world.ni-1) || (j == 0) || (j == world.nj-1) || (k == 0) || (k == world.nk-1)) {
						eps_h[i][j][k] = 0;   // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_h[i][j][k] = eps_2h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2 + 1][j/2][k/2] +
						                  eps_2h[i/2][j/2+1][k/2] + eps_2h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2+1][j/2][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2+1][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2][j/2+1][k/2+1]);
					} else {
						eps_h[i][j][k] = 0.125*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2 + 1][j/2][k/2] +
						                  eps_2h[i/2][j/2+1][k/2] + eps_2h[i/2 + 1][j/2+1][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2 + 1][j/2][k/2+1] +
						                  eps_2h[i/2][j/2+1][k/2+1] + eps_2h[i/2 + 1][j/2+1][k/2+1]);
					}
				}

        // 6) update solution on the fine mesh
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					phi[i][j][k] = phi[i][j][k] - eps_h[i][j][k];
				}

    }

	/*
	int l = 0;
    double erl1 = 0.;
    double erl2 = 0.;
    double erl3 = 0.;
	for (int i=1;i<world.ni-1;i++)
		for (int j=1;j<world.nj-1;j++)
			for (int k=1;k<world.nk-1;k++)
			{
				// consecutive ordering
				//tmp[i+1][j+1][k+1] = phi[i+1][j+1][k+1];
				//phiF[i+1][j+1][k+1] = 0.125*out2[l]/((double)(Nx+1))/((double)(Ny+1))/((double)(Nz+1));
				erl1 +=pow(fabs(phi[i][j][k] - phi_test[i][j][k]),2);
				erl2 +=pow(fabs(phi[i][j][k]),2);
				erl3 +=pow(fabs(phi_test[i][j][k]),2);
				
				
				if ((l > world.ni*world.nj*world.nk/2) && (l < world.ni*world.nj*world.nk/2 + world.nk*2))
				{
					//cout<< setprecision(7) << "phi[" << i << "][" << j << "]["<<k<<"] = " << tmp[i+1][j+1][k+1] << " , phiF[" << i << "][" << j << "]["<<k<<"] = " << phiF[i+1][j+1][k+1] << "\n";
					cout<< setprecision(7) << "phiMG[" << i << "][" << j << "]["<<k<<"] = " << phi[i][j][k] << " , phi[" << i << "][" << j << "]["<<k<<"] = " << phi_test[i][j][k] << "\n";
				}
				l++;
												
			}
    cout<< setprecision(7) << "\n phi - phi_test error=" <<sqrt(erl1) << " , " << sqrt(erl2) << " , " << sqrt(erl3) << endl ;  
	*/

    if (!converged) cerr<<"GS failed to converge, L2="<<L2<<endl;
    return converged;
}

/*solves Poisson equation using MultiGrid*/
bool MultiGridSolverV2::solve()
{
    //references to avoid having to write world.phi
	Field &phi = world.phi;
    Field &rho = world.rho;

	/*
	for (int i=0;i<world.ni;i++)
		for (int j=0;j<world.nj;j++)
			for (int k=0;k<world.nk;k++)
			{
				phi_test[i][j][k] = phi[i][j][k];
			}
	*/

	//precompute 1/(dx^2)
    double3 dh = world.getDh();
    double idx2 = 1.0/(dh[0]*dh[0]);
    double idy2 = 1.0/(dh[1]*dh[1]);
    double idz2 = 1.0/(dh[2]*dh[2]);

    double idx2_2h = 1.0/(4*dh[0]*dh[0]);
    double idy2_2h = 1.0/(4*dh[1]*dh[1]);
    double idz2_2h = 1.0/(4*dh[2]*dh[2]);

    double idx2_4h = 1.0/(16*dh[0]*dh[0]);
    double idy2_4h = 1.0/(16*dh[1]*dh[1]);
    double idz2_4h = 1.0/(16*dh[2]*dh[2]);
    double L2=0;			//norm
    bool converged= false;

    /*solve potential*/
    for (unsigned it=0;it<max_solver_it;it++)
    {
        //number of steps to iterate at the finest level
        unsigned inner_its = 3;

        //number of steps to iterate at the coarse level        
        unsigned inner2h_its = 5;

        unsigned inner4h_its = 50;

        // 1) perform one or more iterations on fine mesh
		for (unsigned its=0;its<inner_its;its++)
		{
			 for (int i=1;i<world.ni-1;i++)
				for (int j=1;j<world.nj-1;j++)
					for (int k=1;k<world.nk-1;k++)
					{
						//standard internal open node
						double phi_new = (rho[i][j][k]/Const::EPS_0 +
										idx2*(phi[i-1][j][k] + phi[i+1][j][k]) +
										idy2*(phi[i][j-1][k]+phi[i][j+1][k]) +
										idz2*(phi[i][j][k-1]+phi[i][j][k+1]))/(2*idx2+2*idy2+2*idz2);

						/*SOR*/
						phi[i][j][k] = phi[i][j][k] + 1.4*(phi_new-phi[i][j][k]);
						
					}
		}
				
        // 2) compute residue on the fine mesh, R = A*phi - b
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					if ((i == 0) || (i == world.ni-1) || (j == 0) || (j == world.nj-1) || (k == 0) || (k == world.nk-1)) {
						R_h[i][j][k] = phi[i][j][k] - 0;  // dirichlet boundary
					} else {
						R_h[i][j][k] = phi[i][j][k]*(2*idx2+2*idy2+2*idz2) -
								rho[i][j][k]/Const::EPS_0 -
								idx2*(phi[i-1][j][k] + phi[i+1][j][k]) -
								idy2*(phi[i][j-1][k]+phi[i][j+1][k]) -
								idz2*(phi[i][j][k-1]+phi[i][j][k+1]);
					}
				}
                    
        // 2b) check for termination

		 /*check for convergence*/
		double sum = 0;
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					sum += R_h[i][j][k]*R_h[i][j][k];
				}

		L2 = sqrt(sum/(world.ni*world.nj*world.nk));
		if (L2<tolerance) {converged=true;break;}
		
        // 3) restrict residue to the 2h mesh
		for (int i=0;i<world.ni-1;i+=2)
			for (int j=0;j<world.nj-1;j+=2)
				for (int k=0;k<world.nk-1;k+=2)
				{
					if ((i == 0) || (j == 0)  || (k == 0) ) {
						R_2h[i/2][j/2][k/2] = R_h[i][j][k];  //R_2h[0] = R_h[0]
					} else {
						R_2h[i/2][j/2][k/2] = (R_h[i-1][j-1][k-1] + 2*R_h[i][j-1][k-1] + R_h[i+1][j-1][k-1] +
											   2*R_h[i-1][j][k-1] + 4*R_h[i][j][k-1] + 2*R_h[i+1][j][k-1] +
											   R_h[i-1][j+1][k-1] + 2*R_h[i][j+1][k-1] + R_h[i+1][j+1][k-1] +
											   2*R_h[i-1][j-1][k] + 4*R_h[i][j-1][k] + 2*R_h[i+1][j-1][k] +
											   4*R_h[i-1][j][k] + 8*R_h[i][j][k] + 4*R_h[i+1][j][k] +
											   2*R_h[i-1][j+1][k] + 4*R_h[i][j+1][k] + 2*R_h[i+1][j+1][k] +
											   R_h[i-1][j-1][k+1] + 2*R_h[i][j-1][k+1] + R_h[i+1][j-1][k+1] +
											   2*R_h[i-1][j][k+1] + 4*R_h[i][j][k+1] + 2*R_h[i+1][j][k+1] +
											   R_h[i-1][j+1][k+1] + 2*R_h[i][j+1][k+1] + R_h[i+1][j+1][k+1])/64.0;
					}
				}
		
         // 3) restrict residue to the 4h mesh
		for (int i=0;i<world.ni/2-1;i+=2)
			for (int j=0;j<world.nj/2-1;j+=2)
				for (int k=0;k<world.nk/2-1;k+=2)
				{
					if ((i == 0) || (j == 0) || (k == 0) ) {
						R_4h[i/2][j/2][k/2] = R_2h[i][j][k];  // dirichlet boundary
					} else {
						R_4h[i/2][j/2][k/2] = (R_2h[i-1][j-1][k-1] + 2*R_2h[i][j-1][k-1] + R_2h[i+1][j-1][k-1] +
											   2*R_2h[i-1][j][k-1] + 4*R_2h[i][j][k-1] + 2*R_2h[i+1][j][k-1] +
											   R_2h[i-1][j+1][k-1] + 2*R_2h[i][j+1][k-1] + R_2h[i+1][j+1][k-1] +
											   2*R_2h[i-1][j-1][k] + 4*R_2h[i][j-1][k] + 2*R_2h[i+1][j-1][k] +
											   4*R_2h[i-1][j][k] + 8*R_2h[i][j][k] + 4*R_2h[i+1][j][k] +
											   2*R_2h[i-1][j+1][k] + 4*R_2h[i][j+1][k] + 2*R_2h[i+1][j+1][k] +
											   R_2h[i-1][j-1][k+1] + 2*R_2h[i][j-1][k+1] + R_2h[i+1][j-1][k+1] +
											   2*R_2h[i-1][j][k+1] + 4*R_2h[i][j][k+1] + 2*R_2h[i+1][j][k+1] +
											   R_2h[i-1][j+1][k+1] + 2*R_2h[i][j+1][k+1] + R_2h[i+1][j+1][k+1])/64.0;
					}
				}
           
		// 4) perform few iteration of the correction vector on the 4h mesh
		for (unsigned its=0;its<inner4h_its;its++)
		{
			 for (int i=1;i<world.ni/4-1;i++)
				for (int j=1;j<world.nj/4-1;j++)
					for (int k=1;k<world.nk/4-1;k++)
					{
						//standard internal open node
						double g = (R_4h[i][j][k] +
										idx2_4h*(eps_4h[i-1][j][k] + eps_4h[i+1][j][k]) +
										idy2_4h*(eps_4h[i][j-1][k]+eps_4h[i][j+1][k]) +
										idz2_4h*(eps_4h[i][j][k-1]+eps_4h[i][j][k+1]))/(2*idx2_4h+2*idy2_4h+2*idz2_4h);

						/*SOR*/
						eps_4h[i][j][k] = eps_4h[i][j][k] + 1.4*(g-eps_4h[i][j][k]);
					}
		}

        // 5) interpolate eps to 2h mesh
		for (int i=0;i<world.ni/2-1;i++)
			for (int j=0;j<world.nj/2-1;j++)
				for (int k=0;k<world.nk/2-1;k++)
				{
					if ((i == 0) || (i == world.ni/2-1) || (j == 0) || (j == world.nj/2-1) || (k == 0) || (k == world.nk/2-1)) {
						eps_2h[i][j][k] = 0;    // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_2h[i][j][k] = eps_4h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2 + 1][j/2][k/2] +
						                  eps_4h[i/2][j/2+1][k/2] + eps_4h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2+1][j/2][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2+1][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2][j/2+1][k/2+1]);
					} else {
						eps_2h[i][j][k] = 0.125*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2 + 1][j/2][k/2] +
						                  eps_4h[i/2][j/2+1][k/2] + eps_4h[i/2 + 1][j/2+1][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2 + 1][j/2][k/2+1] +
						                  eps_4h[i/2][j/2+1][k/2+1] + eps_4h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
		// 4) perform few iteration of the correction vector on the 2h mesh
		for (unsigned its=0;its<inner2h_its;its++)
		{
			for (int i=1;i<world.ni/2-1;i++)
				for (int j=1;j<world.nj/2-1;j++)
					for (int k=1;k<world.nk/2-1;k++)
					{
						//standard internal open node
						double g = (R_2h[i][j][k] +
										idx2_2h*(eps_2h[i-1][j][k] + eps_2h[i+1][j][k]) +
										idy2_2h*(eps_2h[i][j-1][k]+eps_2h[i][j+1][k]) +
										idz2_2h*(eps_2h[i][j][k-1]+eps_2h[i][j][k+1]))/(2*idx2_2h+2*idy2_2h+2*idz2_2h);

						// SOR
						eps_2h[i][j][k] = eps_2h[i][j][k] + 1.4*(g-eps_2h[i][j][k]);
					}
		}

        // 5) interpolate eps to h mesh
		for (int i=0;i<world.ni-1;i++)
			for (int j=0;j<world.nj-1;j++)
				for (int k=0;k<world.nk-1;k++)
				{
					if ((i == 0) || (i == world.ni-1) || (j == 0) || (j == world.nj-1) || (k == 0) || (k == world.nk-1)) {
						eps_h[i][j][k] = 0;   // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_h[i][j][k] = eps_2h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2 + 1][j/2][k/2] +
						                  eps_2h[i/2][j/2+1][k/2] + eps_2h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2+1][j/2][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2+1][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2][j/2+1][k/2+1]);
					} else {
						eps_h[i][j][k] = 0.125*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2 + 1][j/2][k/2] +
						                  eps_2h[i/2][j/2+1][k/2] + eps_2h[i/2 + 1][j/2+1][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2 + 1][j/2][k/2+1] +
						                  eps_2h[i/2][j/2+1][k/2+1] + eps_2h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
        // 6) update solution on the fine mesh
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					phi[i][j][k] = phi[i][j][k] - eps_h[i][j][k];
				}

    }

	/*
	int l = 0;
    double erl1 = 0.;
    double erl2 = 0.;
    double erl3 = 0.;
	for (int i=1;i<world.ni-1;i++)
		for (int j=1;j<world.nj-1;j++)
			for (int k=1;k<world.nk-1;k++)
			{
				// consecutive ordering
				//tmp[i+1][j+1][k+1] = phi[i+1][j+1][k+1];
				//phiF[i+1][j+1][k+1] = 0.125*out2[l]/((double)(Nx+1))/((double)(Ny+1))/((double)(Nz+1));
				erl1 +=pow(fabs(phi[i][j][k] - phi_test[i][j][k]),2);
				erl2 +=pow(fabs(phi[i][j][k]),2);
				erl3 +=pow(fabs(phi_test[i][j][k]),2);
				
				
				if ((l > world.ni*world.nj*world.nk/2) && (l < world.ni*world.nj*world.nk/2 + world.nk*2))
				{
					//cout<< setprecision(7) << "phi[" << i << "][" << j << "]["<<k<<"] = " << tmp[i+1][j+1][k+1] << " , phiF[" << i << "][" << j << "]["<<k<<"] = " << phiF[i+1][j+1][k+1] << "\n";
					cout<< setprecision(7) << "phiMG[" << i << "][" << j << "]["<<k<<"] = " << phi[i][j][k] << " , phi[" << i << "][" << j << "]["<<k<<"] = " << phi_test[i][j][k] << "\n";
				}
				l++;
												
			}
    cout<< setprecision(7) << "\n phi - phi_test error=" <<sqrt(erl1) << " , " << sqrt(erl2) << " , " << sqrt(erl3) << endl ;  
	*/

    if (!converged) cerr<<"GS failed to converge, L2="<<L2<<endl;
    return converged;
}

/*solves Poisson equation using MultiGrid*/
bool MultiGridSolverV3::solve()
{
    //references to avoid having to write world.phi
	Field &phi = world.phi;
    Field &rho = world.rho;

	/*
	for (int i=0;i<world.ni;i++)
		for (int j=0;j<world.nj;j++)
			for (int k=0;k<world.nk;k++)
			{
				phi_test[i][j][k] = phi[i][j][k];
			}
	*/

	//precompute 1/(dx^2)
    double3 dh = world.getDh();
    double idx2 = 1.0/(dh[0]*dh[0]);
    double idy2 = 1.0/(dh[1]*dh[1]);
    double idz2 = 1.0/(dh[2]*dh[2]);

    double idx2_2h = 1.0/(4*dh[0]*dh[0]);
    double idy2_2h = 1.0/(4*dh[1]*dh[1]);
    double idz2_2h = 1.0/(4*dh[2]*dh[2]);

    double idx2_4h = 1.0/(16*dh[0]*dh[0]);
    double idy2_4h = 1.0/(16*dh[1]*dh[1]);
    double idz2_4h = 1.0/(16*dh[2]*dh[2]);

    double idx2_8h = 1.0/(64*dh[0]*dh[0]);
    double idy2_8h = 1.0/(64*dh[1]*dh[1]);
    double idz2_8h = 1.0/(64*dh[2]*dh[2]);
	
    double L2=0;			//norm
    bool converged= false;

    /*solve potential*/
    for (unsigned it=0;it<max_solver_it;it++)
    {
        //number of steps to iterate at the finest level
        unsigned inner_its = 3;

        //number of steps to iterate at the coarse level        
        unsigned inner2h_its = 5;

        unsigned inner4h_its = 5;

        unsigned inner8h_its = 50;

        // 1) perform one or more iterations on fine mesh
		for (unsigned its=0;its<inner_its;its++)
		{
			 for (int i=1;i<world.ni-1;i++)
				for (int j=1;j<world.nj-1;j++)
					for (int k=1;k<world.nk-1;k++)
					{
						//standard internal open node
						double phi_new = (rho[i][j][k]/Const::EPS_0 +
										idx2*(phi[i-1][j][k] + phi[i+1][j][k]) +
										idy2*(phi[i][j-1][k]+phi[i][j+1][k]) +
										idz2*(phi[i][j][k-1]+phi[i][j][k+1]))/(2*idx2+2*idy2+2*idz2);

						/*SOR*/
						phi[i][j][k] = phi[i][j][k] + 1.4*(phi_new-phi[i][j][k]);
						
					}
		}
				
        // 2) compute residue on the fine mesh, R = A*phi - b
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					if ((i == 0) || (i == world.ni-1) || (j == 0) || (j == world.nj-1) || (k == 0) || (k == world.nk-1)) {
						R_h[i][j][k] = phi[i][j][k] - 0;  // dirichlet boundary
					} else {
						R_h[i][j][k] = phi[i][j][k]*(2*idx2+2*idy2+2*idz2) -
								rho[i][j][k]/Const::EPS_0 -
								idx2*(phi[i-1][j][k] + phi[i+1][j][k]) -
								idy2*(phi[i][j-1][k]+phi[i][j+1][k]) -
								idz2*(phi[i][j][k-1]+phi[i][j][k+1]);
					}
				}
                    
        // 2b) check for termination

		 /*check for convergence*/
		double sum = 0;
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					sum += R_h[i][j][k]*R_h[i][j][k];
				}

		L2 = sqrt(sum/(world.ni*world.nj*world.nk));
		if (L2<tolerance) {converged=true;break;}
		
        // 3) restrict residue to the 2h mesh
		for (int i=0;i<world.ni-1;i+=2)
			for (int j=0;j<world.nj-1;j+=2)
				for (int k=0;k<world.nk-1;k+=2)
				{
					if ((i == 0) || (j == 0)  || (k == 0) ) {
						R_2h[i/2][j/2][k/2] = R_h[i][j][k];  //R_2h[0] = R_h[0]
					} else {
						R_2h[i/2][j/2][k/2] = (R_h[i-1][j-1][k-1] + 2*R_h[i][j-1][k-1] + R_h[i+1][j-1][k-1] +
											   2*R_h[i-1][j][k-1] + 4*R_h[i][j][k-1] + 2*R_h[i+1][j][k-1] +
											   R_h[i-1][j+1][k-1] + 2*R_h[i][j+1][k-1] + R_h[i+1][j+1][k-1] +
											   2*R_h[i-1][j-1][k] + 4*R_h[i][j-1][k] + 2*R_h[i+1][j-1][k] +
											   4*R_h[i-1][j][k] + 8*R_h[i][j][k] + 4*R_h[i+1][j][k] +
											   2*R_h[i-1][j+1][k] + 4*R_h[i][j+1][k] + 2*R_h[i+1][j+1][k] +
											   R_h[i-1][j-1][k+1] + 2*R_h[i][j-1][k+1] + R_h[i+1][j-1][k+1] +
											   2*R_h[i-1][j][k+1] + 4*R_h[i][j][k+1] + 2*R_h[i+1][j][k+1] +
											   R_h[i-1][j+1][k+1] + 2*R_h[i][j+1][k+1] + R_h[i+1][j+1][k+1])/64.0;
					}
				}
		
         // 3) restrict residue to the 4h mesh
		for (int i=0;i<world.ni/2-1;i+=2)
			for (int j=0;j<world.nj/2-1;j+=2)
				for (int k=0;k<world.nk/2-1;k+=2)
				{
					if ((i == 0) || (j == 0) || (k == 0) ) {
						R_4h[i/2][j/2][k/2] = R_2h[i][j][k];  // dirichlet boundary
					} else {
						R_4h[i/2][j/2][k/2] = (R_2h[i-1][j-1][k-1] + 2*R_2h[i][j-1][k-1] + R_2h[i+1][j-1][k-1] +
											   2*R_2h[i-1][j][k-1] + 4*R_2h[i][j][k-1] + 2*R_2h[i+1][j][k-1] +
											   R_2h[i-1][j+1][k-1] + 2*R_2h[i][j+1][k-1] + R_2h[i+1][j+1][k-1] +
											   2*R_2h[i-1][j-1][k] + 4*R_2h[i][j-1][k] + 2*R_2h[i+1][j-1][k] +
											   4*R_2h[i-1][j][k] + 8*R_2h[i][j][k] + 4*R_2h[i+1][j][k] +
											   2*R_2h[i-1][j+1][k] + 4*R_2h[i][j+1][k] + 2*R_2h[i+1][j+1][k] +
											   R_2h[i-1][j-1][k+1] + 2*R_2h[i][j-1][k+1] + R_2h[i+1][j-1][k+1] +
											   2*R_2h[i-1][j][k+1] + 4*R_2h[i][j][k+1] + 2*R_2h[i+1][j][k+1] +
											   R_2h[i-1][j+1][k+1] + 2*R_2h[i][j+1][k+1] + R_2h[i+1][j+1][k+1])/64.0;
					}
				}
           
         // 3) restrict residue to the 8h mesh
		for (int i=0;i<world.ni/4-1;i+=2)
			for (int j=0;j<world.nj/4-1;j+=2)
				for (int k=0;k<world.nk/4-1;k+=2)
				{
					if ((i == 0) || (j == 0) || (k == 0) ) {
						R_8h[i/2][j/2][k/2] = R_4h[i][j][k];  // dirichlet boundary
					} else {
						R_8h[i/2][j/2][k/2] = (R_4h[i-1][j-1][k-1] + 2*R_4h[i][j-1][k-1] + R_4h[i+1][j-1][k-1] +
											   2*R_4h[i-1][j][k-1] + 4*R_4h[i][j][k-1] + 2*R_4h[i+1][j][k-1] +
											   R_4h[i-1][j+1][k-1] + 2*R_4h[i][j+1][k-1] + R_4h[i+1][j+1][k-1] +
											   2*R_4h[i-1][j-1][k] + 4*R_4h[i][j-1][k] + 2*R_4h[i+1][j-1][k] +
											   4*R_4h[i-1][j][k] + 8*R_4h[i][j][k] + 4*R_4h[i+1][j][k] +
											   2*R_4h[i-1][j+1][k] + 4*R_4h[i][j+1][k] + 2*R_4h[i+1][j+1][k] +
											   R_4h[i-1][j-1][k+1] + 2*R_4h[i][j-1][k+1] + R_4h[i+1][j-1][k+1] +
											   2*R_4h[i-1][j][k+1] + 4*R_4h[i][j][k+1] + 2*R_4h[i+1][j][k+1] +
											   R_4h[i-1][j+1][k+1] + 2*R_4h[i][j+1][k+1] + R_4h[i+1][j+1][k+1])/64.0;
					}
				}
           
		// 4) perform few iteration of the correction vector on the 8h mesh
		for (unsigned its=0;its<inner8h_its;its++)
		{
			 for (int i=1;i<world.ni/8-1;i++)
				for (int j=1;j<world.nj/8-1;j++)
					for (int k=1;k<world.nk/8-1;k++)
					{
						//standard internal open node
						double g = (R_8h[i][j][k] +
										idx2_8h*(eps_8h[i-1][j][k] + eps_8h[i+1][j][k]) +
										idy2_8h*(eps_8h[i][j-1][k]+eps_8h[i][j+1][k]) +
										idz2_8h*(eps_8h[i][j][k-1]+eps_8h[i][j][k+1]))/(2*idx2_8h+2*idy2_8h+2*idz2_8h);

						// SOR
						eps_8h[i][j][k] = eps_8h[i][j][k] + 1.4*(g-eps_8h[i][j][k]);
					}
		}

        // 5) interpolate eps to 4h mesh
		for (int i=0;i<world.ni/4-1;i++)
			for (int j=0;j<world.nj/4-1;j++)
				for (int k=0;k<world.nk/4-1;k++)
				{
					if ((i == 0) || (i == world.ni/4-1) || (j == 0) || (j == world.nj/4-1) || (k == 0) || (k == world.nk/4-1)) {
						eps_4h[i][j][k] = 0;    // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_4h[i][j][k] = eps_8h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_4h[i][j][k] = 0.5*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_4h[i][j][k] = 0.5*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_4h[i][j][k] = 0.5*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_4h[i][j][k] = 0.25*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2 + 1][j/2][k/2] +
						                  eps_8h[i/2][j/2+1][k/2] + eps_8h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_4h[i][j][k] = 0.25*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2+1][j/2][k/2] +
						                  eps_8h[i/2][j/2][k/2+1] + eps_8h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_4h[i][j][k] = 0.25*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2][j/2+1][k/2] +
						                  eps_8h[i/2][j/2][k/2+1] + eps_8h[i/2][j/2+1][k/2+1]);
					} else {
						eps_4h[i][j][k] = 0.125*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2 + 1][j/2][k/2] +
						                  eps_8h[i/2][j/2+1][k/2] + eps_8h[i/2 + 1][j/2+1][k/2] +
						                  eps_8h[i/2][j/2][k/2+1] + eps_8h[i/2 + 1][j/2][k/2+1] +
						                  eps_8h[i/2][j/2+1][k/2+1] + eps_8h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
		// 4) perform few iteration of the correction vector on the 4h mesh
		for (unsigned its=0;its<inner4h_its;its++)
		{
			 for (int i=1;i<world.ni/4-1;i++)
				for (int j=1;j<world.nj/4-1;j++)
					for (int k=1;k<world.nk/4-1;k++)
					{
						//standard internal open node
						double g = (R_4h[i][j][k] +
										idx2_4h*(eps_4h[i-1][j][k] + eps_4h[i+1][j][k]) +
										idy2_4h*(eps_4h[i][j-1][k]+eps_4h[i][j+1][k]) +
										idz2_4h*(eps_4h[i][j][k-1]+eps_4h[i][j][k+1]))/(2*idx2_4h+2*idy2_4h+2*idz2_4h);

						/*SOR*/
						eps_4h[i][j][k] = eps_4h[i][j][k] + 1.4*(g-eps_4h[i][j][k]);
					}
		}

        // 5) interpolate eps to 2h mesh
		for (int i=0;i<world.ni/2-1;i++)
			for (int j=0;j<world.nj/2-1;j++)
				for (int k=0;k<world.nk/2-1;k++)
				{
					if ((i == 0) || (i == world.ni/2-1) || (j == 0) || (j == world.nj/2-1) || (k == 0) || (k == world.nk/2-1)) {
						eps_2h[i][j][k] = 0;    // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_2h[i][j][k] = eps_4h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2 + 1][j/2][k/2] +
						                  eps_4h[i/2][j/2+1][k/2] + eps_4h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2+1][j/2][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2+1][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2][j/2+1][k/2+1]);
					} else {
						eps_2h[i][j][k] = 0.125*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2 + 1][j/2][k/2] +
						                  eps_4h[i/2][j/2+1][k/2] + eps_4h[i/2 + 1][j/2+1][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2 + 1][j/2][k/2+1] +
						                  eps_4h[i/2][j/2+1][k/2+1] + eps_4h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
		// 4) perform few iteration of the correction vector on the 2h mesh
		for (unsigned its=0;its<inner2h_its;its++)
		{
			for (int i=1;i<world.ni/2-1;i++)
				for (int j=1;j<world.nj/2-1;j++)
					for (int k=1;k<world.nk/2-1;k++)
					{
						//standard internal open node
						double g = (R_2h[i][j][k] +
										idx2_2h*(eps_2h[i-1][j][k] + eps_2h[i+1][j][k]) +
										idy2_2h*(eps_2h[i][j-1][k]+eps_2h[i][j+1][k]) +
										idz2_2h*(eps_2h[i][j][k-1]+eps_2h[i][j][k+1]))/(2*idx2_2h+2*idy2_2h+2*idz2_2h);

						// SOR
						eps_2h[i][j][k] = eps_2h[i][j][k] + 1.4*(g-eps_2h[i][j][k]);
					}
		}

        // 5) interpolate eps to h mesh
		for (int i=0;i<world.ni-1;i++)
			for (int j=0;j<world.nj-1;j++)
				for (int k=0;k<world.nk-1;k++)
				{
					if ((i == 0) || (i == world.ni-1) || (j == 0) || (j == world.nj-1) || (k == 0) || (k == world.nk-1)) {
						eps_h[i][j][k] = 0;   // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_h[i][j][k] = eps_2h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2 + 1][j/2][k/2] +
						                  eps_2h[i/2][j/2+1][k/2] + eps_2h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2+1][j/2][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2+1][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2][j/2+1][k/2+1]);
					} else {
						eps_h[i][j][k] = 0.125*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2 + 1][j/2][k/2] +
						                  eps_2h[i/2][j/2+1][k/2] + eps_2h[i/2 + 1][j/2+1][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2 + 1][j/2][k/2+1] +
						                  eps_2h[i/2][j/2+1][k/2+1] + eps_2h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
        // 6) update solution on the fine mesh
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					phi[i][j][k] = phi[i][j][k] - eps_h[i][j][k];
				}

    }

	/*
	int l = 0;
    double erl1 = 0.;
    double erl2 = 0.;
    double erl3 = 0.;
	for (int i=1;i<world.ni-1;i++)
		for (int j=1;j<world.nj-1;j++)
			for (int k=1;k<world.nk-1;k++)
			{
				// consecutive ordering
				//tmp[i+1][j+1][k+1] = phi[i+1][j+1][k+1];
				//phiF[i+1][j+1][k+1] = 0.125*out2[l]/((double)(Nx+1))/((double)(Ny+1))/((double)(Nz+1));
				erl1 +=pow(fabs(phi[i][j][k] - phi_test[i][j][k]),2);
				erl2 +=pow(fabs(phi[i][j][k]),2);
				erl3 +=pow(fabs(phi_test[i][j][k]),2);
				
				
				if ((l > world.ni*world.nj*world.nk/2) && (l < world.ni*world.nj*world.nk/2 + world.nk*2))
				{
					//cout<< setprecision(7) << "phi[" << i << "][" << j << "]["<<k<<"] = " << tmp[i+1][j+1][k+1] << " , phiF[" << i << "][" << j << "]["<<k<<"] = " << phiF[i+1][j+1][k+1] << "\n";
					cout<< setprecision(7) << "phiMG[" << i << "][" << j << "]["<<k<<"] = " << phi[i][j][k] << " , phi[" << i << "][" << j << "]["<<k<<"] = " << phi_test[i][j][k] << "\n";
				}
				l++;
												
			}
    cout<< setprecision(7) << "\n phi - phi_test error=" <<sqrt(erl1) << " , " << sqrt(erl2) << " , " << sqrt(erl3) << endl ;  
	*/

    if (!converged) cerr<<"GS failed to converge, L2="<<L2<<endl;
    return converged;
}

/*solves Poisson equation using MultiGrid*/
bool MultiGridSolverV4::solve()
{
    //references to avoid having to write world.phi
	Field &phi = world.phi;
    Field &rho = world.rho;

	/*
	for (int i=0;i<world.ni;i++)
		for (int j=0;j<world.nj;j++)
			for (int k=0;k<world.nk;k++)
			{
				phi_test[i][j][k] = phi[i][j][k];
			}
	*/

	//precompute 1/(dx^2)
    double3 dh = world.getDh();
    double idx2 = 1.0/(dh[0]*dh[0]);
    double idy2 = 1.0/(dh[1]*dh[1]);
    double idz2 = 1.0/(dh[2]*dh[2]);

    double idx2_2h = 1.0/(4*dh[0]*dh[0]);
    double idy2_2h = 1.0/(4*dh[1]*dh[1]);
    double idz2_2h = 1.0/(4*dh[2]*dh[2]);

    double idx2_4h = 1.0/(16*dh[0]*dh[0]);
    double idy2_4h = 1.0/(16*dh[1]*dh[1]);
    double idz2_4h = 1.0/(16*dh[2]*dh[2]);

    double idx2_8h = 1.0/(64*dh[0]*dh[0]);
    double idy2_8h = 1.0/(64*dh[1]*dh[1]);
    double idz2_8h = 1.0/(64*dh[2]*dh[2]);
	
    double idx2_16h = 1.0/(256*dh[0]*dh[0]);
    double idy2_16h = 1.0/(256*dh[1]*dh[1]);
    double idz2_16h = 1.0/(256*dh[2]*dh[2]);
	
    double L2=0;			//norm
    bool converged= false;

    /*solve potential*/
    for (unsigned it=0;it<max_solver_it;it++)
    {
        //number of steps to iterate at the finest level
        unsigned inner_its = 3;

        //number of steps to iterate at the coarse level        
        unsigned inner2h_its = 5;

        unsigned inner4h_its = 5;

        unsigned inner8h_its = 5;

        unsigned inner16h_its = 50;

        // 1) perform one or more iterations on fine mesh
		for (unsigned its=0;its<inner_its;its++)
		{
			 for (int i=1;i<world.ni-1;i++)
				for (int j=1;j<world.nj-1;j++)
					for (int k=1;k<world.nk-1;k++)
					{
						//standard internal open node
						double phi_new = (rho[i][j][k]/Const::EPS_0 +
										idx2*(phi[i-1][j][k] + phi[i+1][j][k]) +
										idy2*(phi[i][j-1][k]+phi[i][j+1][k]) +
										idz2*(phi[i][j][k-1]+phi[i][j][k+1]))/(2*idx2+2*idy2+2*idz2);

						/*SOR*/
						phi[i][j][k] = phi[i][j][k] + 1.4*(phi_new-phi[i][j][k]);
						
					}
		}
				
        // 2) compute residue on the fine mesh, R = A*phi - b
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					if ((i == 0) || (i == world.ni-1) || (j == 0) || (j == world.nj-1) || (k == 0) || (k == world.nk-1)) {
						R_h[i][j][k] = phi[i][j][k] - 0;  // dirichlet boundary
					} else {
						R_h[i][j][k] = phi[i][j][k]*(2*idx2+2*idy2+2*idz2) -
								rho[i][j][k]/Const::EPS_0 -
								idx2*(phi[i-1][j][k] + phi[i+1][j][k]) -
								idy2*(phi[i][j-1][k]+phi[i][j+1][k]) -
								idz2*(phi[i][j][k-1]+phi[i][j][k+1]);
					}
				}
                    
        // 2b) check for termination

		 /*check for convergence*/
		double sum = 0;
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					sum += R_h[i][j][k]*R_h[i][j][k];
				}

		L2 = sqrt(sum/(world.ni*world.nj*world.nk));
		if (L2<tolerance) {converged=true;break;}
		
          
        // 3) restrict residue to the 2h mesh
		for (int i=0;i<world.ni-1;i+=2)
			for (int j=0;j<world.nj-1;j+=2)
				for (int k=0;k<world.nk-1;k+=2)
				{
					if ((i == 0) || (j == 0)  || (k == 0) ) {
						R_2h[i/2][j/2][k/2] = R_h[i][j][k];  //R_2h[0] = R_h[0]
					} else {
						R_2h[i/2][j/2][k/2] = (R_h[i-1][j-1][k-1] + 2*R_h[i][j-1][k-1] + R_h[i+1][j-1][k-1] +
											   2*R_h[i-1][j][k-1] + 4*R_h[i][j][k-1] + 2*R_h[i+1][j][k-1] +
											   R_h[i-1][j+1][k-1] + 2*R_h[i][j+1][k-1] + R_h[i+1][j+1][k-1] +
											   2*R_h[i-1][j-1][k] + 4*R_h[i][j-1][k] + 2*R_h[i+1][j-1][k] +
											   4*R_h[i-1][j][k] + 8*R_h[i][j][k] + 4*R_h[i+1][j][k] +
											   2*R_h[i-1][j+1][k] + 4*R_h[i][j+1][k] + 2*R_h[i+1][j+1][k] +
											   R_h[i-1][j-1][k+1] + 2*R_h[i][j-1][k+1] + R_h[i+1][j-1][k+1] +
											   2*R_h[i-1][j][k+1] + 4*R_h[i][j][k+1] + 2*R_h[i+1][j][k+1] +
											   R_h[i-1][j+1][k+1] + 2*R_h[i][j+1][k+1] + R_h[i+1][j+1][k+1])/64.0;
					}
				}
		
         // 3) restrict residue to the 4h mesh
		for (int i=0;i<world.ni/2-1;i+=2)
			for (int j=0;j<world.nj/2-1;j+=2)
				for (int k=0;k<world.nk/2-1;k+=2)
				{
					if ((i == 0) || (j == 0) || (k == 0) ) {
						R_4h[i/2][j/2][k/2] = R_2h[i][j][k];  // dirichlet boundary
					} else {
						R_4h[i/2][j/2][k/2] = (R_2h[i-1][j-1][k-1] + 2*R_2h[i][j-1][k-1] + R_2h[i+1][j-1][k-1] +
											   2*R_2h[i-1][j][k-1] + 4*R_2h[i][j][k-1] + 2*R_2h[i+1][j][k-1] +
											   R_2h[i-1][j+1][k-1] + 2*R_2h[i][j+1][k-1] + R_2h[i+1][j+1][k-1] +
											   2*R_2h[i-1][j-1][k] + 4*R_2h[i][j-1][k] + 2*R_2h[i+1][j-1][k] +
											   4*R_2h[i-1][j][k] + 8*R_2h[i][j][k] + 4*R_2h[i+1][j][k] +
											   2*R_2h[i-1][j+1][k] + 4*R_2h[i][j+1][k] + 2*R_2h[i+1][j+1][k] +
											   R_2h[i-1][j-1][k+1] + 2*R_2h[i][j-1][k+1] + R_2h[i+1][j-1][k+1] +
											   2*R_2h[i-1][j][k+1] + 4*R_2h[i][j][k+1] + 2*R_2h[i+1][j][k+1] +
											   R_2h[i-1][j+1][k+1] + 2*R_2h[i][j+1][k+1] + R_2h[i+1][j+1][k+1])/64.0;
					}
				}
           
         // 3) restrict residue to the 8h mesh
		for (int i=0;i<world.ni/4-1;i+=2)
			for (int j=0;j<world.nj/4-1;j+=2)
				for (int k=0;k<world.nk/4-1;k+=2)
				{
					if ((i == 0) || (j == 0) || (k == 0) ) {
						R_8h[i/2][j/2][k/2] = R_4h[i][j][k];  // dirichlet boundary
					} else {
						R_8h[i/2][j/2][k/2] = (R_4h[i-1][j-1][k-1] + 2*R_4h[i][j-1][k-1] + R_4h[i+1][j-1][k-1] +
											   2*R_4h[i-1][j][k-1] + 4*R_4h[i][j][k-1] + 2*R_4h[i+1][j][k-1] +
											   R_4h[i-1][j+1][k-1] + 2*R_4h[i][j+1][k-1] + R_4h[i+1][j+1][k-1] +
											   2*R_4h[i-1][j-1][k] + 4*R_4h[i][j-1][k] + 2*R_4h[i+1][j-1][k] +
											   4*R_4h[i-1][j][k] + 8*R_4h[i][j][k] + 4*R_4h[i+1][j][k] +
											   2*R_4h[i-1][j+1][k] + 4*R_4h[i][j+1][k] + 2*R_4h[i+1][j+1][k] +
											   R_4h[i-1][j-1][k+1] + 2*R_4h[i][j-1][k+1] + R_4h[i+1][j-1][k+1] +
											   2*R_4h[i-1][j][k+1] + 4*R_4h[i][j][k+1] + 2*R_4h[i+1][j][k+1] +
											   R_4h[i-1][j+1][k+1] + 2*R_4h[i][j+1][k+1] + R_4h[i+1][j+1][k+1])/64.0;
					}
				}
           
         // 3) restrict residue to the 16h mesh
		for (int i=0;i<world.ni/8-1;i+=2)
			for (int j=0;j<world.nj/8-1;j+=2)
				for (int k=0;k<world.nk/8-1;k+=2)
				{
					if ((i == 0) || (j == 0) || (k == 0) ) {
						R_16h[i/2][j/2][k/2] = R_8h[i][j][k];  // dirichlet boundary
					} else {
						R_16h[i/2][j/2][k/2] = (R_8h[i-1][j-1][k-1] + 2*R_8h[i][j-1][k-1] + R_8h[i+1][j-1][k-1] +
											   2*R_8h[i-1][j][k-1] + 4*R_8h[i][j][k-1] + 2*R_8h[i+1][j][k-1] +
											   R_8h[i-1][j+1][k-1] + 2*R_8h[i][j+1][k-1] + R_8h[i+1][j+1][k-1] +
											   2*R_8h[i-1][j-1][k] + 4*R_8h[i][j-1][k] + 2*R_8h[i+1][j-1][k] +
											   4*R_8h[i-1][j][k] + 8*R_8h[i][j][k] + 4*R_8h[i+1][j][k] +
											   2*R_8h[i-1][j+1][k] + 4*R_8h[i][j+1][k] + 2*R_8h[i+1][j+1][k] +
											   R_8h[i-1][j-1][k+1] + 2*R_8h[i][j-1][k+1] + R_8h[i+1][j-1][k+1] +
											   2*R_8h[i-1][j][k+1] + 4*R_8h[i][j][k+1] + 2*R_8h[i+1][j][k+1] +
											   R_8h[i-1][j+1][k+1] + 2*R_8h[i][j+1][k+1] + R_8h[i+1][j+1][k+1])/64.0;
					}
				}

		// 4) perform few iteration of the correction vector on the 16h mesh
		for (unsigned its=0;its<inner16h_its;its++)
		{
			 for (int i=1;i<world.ni/16-1;i++)
				for (int j=1;j<world.nj/16-1;j++)
					for (int k=1;k<world.nk/16-1;k++)
					{
						//standard internal open node
						double g = (R_16h[i][j][k] +
										idx2_16h*(eps_16h[i-1][j][k] + eps_16h[i+1][j][k]) +
										idy2_16h*(eps_16h[i][j-1][k]+eps_16h[i][j+1][k]) +
										idz2_16h*(eps_16h[i][j][k-1]+eps_16h[i][j][k+1]))/(2*idx2_16h+2*idy2_16h+2*idz2_16h);

						// SOR
						eps_16h[i][j][k] = eps_16h[i][j][k] + 1.4*(g-eps_16h[i][j][k]);
					}
		}

        // 5) interpolate eps to 8h mesh
		for (int i=0;i<world.ni/8-1;i++)
			for (int j=0;j<world.nj/8-1;j++)
				for (int k=0;k<world.nk/8-1;k++)
				{
					if ((i == 0) || (i == world.ni/4-1) || (j == 0) || (j == world.nj/4-1) || (k == 0) || (k == world.nk/4-1)) {
						eps_8h[i][j][k] = 0;    // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_8h[i][j][k] = eps_16h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_8h[i][j][k] = 0.5*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_8h[i][j][k] = 0.5*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_8h[i][j][k] = 0.5*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_8h[i][j][k] = 0.25*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2 + 1][j/2][k/2] +
						                  eps_16h[i/2][j/2+1][k/2] + eps_16h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_8h[i][j][k] = 0.25*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2+1][j/2][k/2] +
						                  eps_16h[i/2][j/2][k/2+1] + eps_16h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_8h[i][j][k] = 0.25*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2][j/2+1][k/2] +
						                  eps_16h[i/2][j/2][k/2+1] + eps_16h[i/2][j/2+1][k/2+1]);
					} else {
						eps_8h[i][j][k] = 0.125*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2 + 1][j/2][k/2] +
						                  eps_16h[i/2][j/2+1][k/2] + eps_16h[i/2 + 1][j/2+1][k/2] +
						                  eps_16h[i/2][j/2][k/2+1] + eps_16h[i/2 + 1][j/2][k/2+1] +
						                  eps_16h[i/2][j/2+1][k/2+1] + eps_16h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
           
		// 4) perform few iteration of the correction vector on the 8h mesh
		for (unsigned its=0;its<inner8h_its;its++)
		{
			 for (int i=1;i<world.ni/8-1;i++)
				for (int j=1;j<world.nj/8-1;j++)
					for (int k=1;k<world.nk/8-1;k++)
					{
						//standard internal open node
						double g = (R_8h[i][j][k] +
										idx2_8h*(eps_8h[i-1][j][k] + eps_8h[i+1][j][k]) +
										idy2_8h*(eps_8h[i][j-1][k]+eps_8h[i][j+1][k]) +
										idz2_8h*(eps_8h[i][j][k-1]+eps_8h[i][j][k+1]))/(2*idx2_8h+2*idy2_8h+2*idz2_8h);

						// SOR
						eps_8h[i][j][k] = eps_8h[i][j][k] + 1.4*(g-eps_8h[i][j][k]);
					}
		}

        // 5) interpolate eps to 4h mesh
		for (int i=0;i<world.ni/4-1;i++)
			for (int j=0;j<world.nj/4-1;j++)
				for (int k=0;k<world.nk/4-1;k++)
				{
					if ((i == 0) || (i == world.ni/4-1) || (j == 0) || (j == world.nj/4-1) || (k == 0) || (k == world.nk/4-1)) {
						eps_4h[i][j][k] = 0;    // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_4h[i][j][k] = eps_8h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_4h[i][j][k] = 0.5*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_4h[i][j][k] = 0.5*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_4h[i][j][k] = 0.5*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_4h[i][j][k] = 0.25*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2 + 1][j/2][k/2] +
						                  eps_8h[i/2][j/2+1][k/2] + eps_8h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_4h[i][j][k] = 0.25*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2+1][j/2][k/2] +
						                  eps_8h[i/2][j/2][k/2+1] + eps_8h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_4h[i][j][k] = 0.25*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2][j/2+1][k/2] +
						                  eps_8h[i/2][j/2][k/2+1] + eps_8h[i/2][j/2+1][k/2+1]);
					} else {
						eps_4h[i][j][k] = 0.125*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2 + 1][j/2][k/2] +
						                  eps_8h[i/2][j/2+1][k/2] + eps_8h[i/2 + 1][j/2+1][k/2] +
						                  eps_8h[i/2][j/2][k/2+1] + eps_8h[i/2 + 1][j/2][k/2+1] +
						                  eps_8h[i/2][j/2+1][k/2+1] + eps_8h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
		// 4) perform few iteration of the correction vector on the 4h mesh
		for (unsigned its=0;its<inner4h_its;its++)
		{
			 for (int i=1;i<world.ni/4-1;i++)
				for (int j=1;j<world.nj/4-1;j++)
					for (int k=1;k<world.nk/4-1;k++)
					{
						//standard internal open node
						double g = (R_4h[i][j][k] +
										idx2_4h*(eps_4h[i-1][j][k] + eps_4h[i+1][j][k]) +
										idy2_4h*(eps_4h[i][j-1][k]+eps_4h[i][j+1][k]) +
										idz2_4h*(eps_4h[i][j][k-1]+eps_4h[i][j][k+1]))/(2*idx2_4h+2*idy2_4h+2*idz2_4h);

						/*SOR*/
						eps_4h[i][j][k] = eps_4h[i][j][k] + 1.4*(g-eps_4h[i][j][k]);
					}
		}

        // 5) interpolate eps to 2h mesh
		for (int i=0;i<world.ni/2-1;i++)
			for (int j=0;j<world.nj/2-1;j++)
				for (int k=0;k<world.nk/2-1;k++)
				{
					if ((i == 0) || (i == world.ni/2-1) || (j == 0) || (j == world.nj/2-1) || (k == 0) || (k == world.nk/2-1)) {
						eps_2h[i][j][k] = 0;    // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_2h[i][j][k] = eps_4h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2 + 1][j/2][k/2] +
						                  eps_4h[i/2][j/2+1][k/2] + eps_4h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2+1][j/2][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2+1][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2][j/2+1][k/2+1]);
					} else {
						eps_2h[i][j][k] = 0.125*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2 + 1][j/2][k/2] +
						                  eps_4h[i/2][j/2+1][k/2] + eps_4h[i/2 + 1][j/2+1][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2 + 1][j/2][k/2+1] +
						                  eps_4h[i/2][j/2+1][k/2+1] + eps_4h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
		// 4) perform few iteration of the correction vector on the 2h mesh
		for (unsigned its=0;its<inner2h_its;its++)
		{
			for (int i=1;i<world.ni/2-1;i++)
				for (int j=1;j<world.nj/2-1;j++)
					for (int k=1;k<world.nk/2-1;k++)
					{
						//standard internal open node
						double g = (R_2h[i][j][k] +
										idx2_2h*(eps_2h[i-1][j][k] + eps_2h[i+1][j][k]) +
										idy2_2h*(eps_2h[i][j-1][k]+eps_2h[i][j+1][k]) +
										idz2_2h*(eps_2h[i][j][k-1]+eps_2h[i][j][k+1]))/(2*idx2_2h+2*idy2_2h+2*idz2_2h);

						// SOR
						eps_2h[i][j][k] = eps_2h[i][j][k] + 1.4*(g-eps_2h[i][j][k]);
					}
		}

        // 5) interpolate eps to h mesh
		for (int i=0;i<world.ni-1;i++)
			for (int j=0;j<world.nj-1;j++)
				for (int k=0;k<world.nk-1;k++)
				{
					if ((i == 0) || (i == world.ni-1) || (j == 0) || (j == world.nj-1) || (k == 0) || (k == world.nk-1)) {
						eps_h[i][j][k] = 0;   // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_h[i][j][k] = eps_2h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2 + 1][j/2][k/2] +
						                  eps_2h[i/2][j/2+1][k/2] + eps_2h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2+1][j/2][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2+1][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2][j/2+1][k/2+1]);
					} else {
						eps_h[i][j][k] = 0.125*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2 + 1][j/2][k/2] +
						                  eps_2h[i/2][j/2+1][k/2] + eps_2h[i/2 + 1][j/2+1][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2 + 1][j/2][k/2+1] +
						                  eps_2h[i/2][j/2+1][k/2+1] + eps_2h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
        // 6) update solution on the fine mesh
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					phi[i][j][k] = phi[i][j][k] - eps_h[i][j][k];
				}

    }

	/*
	int l = 0;
    double erl1 = 0.;
    double erl2 = 0.;
    double erl3 = 0.;
	for (int i=1;i<world.ni-1;i++)
		for (int j=1;j<world.nj-1;j++)
			for (int k=1;k<world.nk-1;k++)
			{
				// consecutive ordering
				//tmp[i+1][j+1][k+1] = phi[i+1][j+1][k+1];
				//phiF[i+1][j+1][k+1] = 0.125*out2[l]/((double)(Nx+1))/((double)(Ny+1))/((double)(Nz+1));
				erl1 +=pow(fabs(phi[i][j][k] - phi_test[i][j][k]),2);
				erl2 +=pow(fabs(phi[i][j][k]),2);
				erl3 +=pow(fabs(phi_test[i][j][k]),2);
				
				
				if ((l > world.ni*world.nj*world.nk/2) && (l < world.ni*world.nj*world.nk/2 + world.nk*2))
				{
					//cout<< setprecision(7) << "phi[" << i << "][" << j << "]["<<k<<"] = " << tmp[i+1][j+1][k+1] << " , phiF[" << i << "][" << j << "]["<<k<<"] = " << phiF[i+1][j+1][k+1] << "\n";
					cout<< setprecision(7) << "phiMG[" << i << "][" << j << "]["<<k<<"] = " << phi[i][j][k] << " , phi[" << i << "][" << j << "]["<<k<<"] = " << phi_test[i][j][k] << "\n";
				}
				l++;
												
			}
    cout<< setprecision(7) << "\n phi - phi_test error=" <<sqrt(erl1) << " , " << sqrt(erl2) << " , " << sqrt(erl3) << endl ;  
	*/

    if (!converged) cerr<<"GS failed to converge, L2="<<L2<<endl;
    return converged;
}

/*solves Poisson equation using MultiGrid*/
bool MultiGridSolverV5::solve()
{
    //references to avoid having to write world.phi
	Field &phi = world.phi;
    Field &rho = world.rho;

	/*
	for (int i=0;i<world.ni;i++)
		for (int j=0;j<world.nj;j++)
			for (int k=0;k<world.nk;k++)
			{
				phi_test[i][j][k] = phi[i][j][k];
			}
	*/

	//precompute 1/(dx^2)
    double3 dh = world.getDh();
    double idx2 = 1.0/(dh[0]*dh[0]);
    double idy2 = 1.0/(dh[1]*dh[1]);
    double idz2 = 1.0/(dh[2]*dh[2]);

    double idx2_2h = 1.0/(4*dh[0]*dh[0]);
    double idy2_2h = 1.0/(4*dh[1]*dh[1]);
    double idz2_2h = 1.0/(4*dh[2]*dh[2]);

    double idx2_4h = 1.0/(16*dh[0]*dh[0]);
    double idy2_4h = 1.0/(16*dh[1]*dh[1]);
    double idz2_4h = 1.0/(16*dh[2]*dh[2]);

    double idx2_8h = 1.0/(64*dh[0]*dh[0]);
    double idy2_8h = 1.0/(64*dh[1]*dh[1]);
    double idz2_8h = 1.0/(64*dh[2]*dh[2]);
	
    double idx2_16h = 1.0/(256*dh[0]*dh[0]);
    double idy2_16h = 1.0/(256*dh[1]*dh[1]);
    double idz2_16h = 1.0/(256*dh[2]*dh[2]);
	
    double idx2_32h = 1.0/(1024*dh[0]*dh[0]);
    double idy2_32h = 1.0/(1024*dh[1]*dh[1]);
    double idz2_32h = 1.0/(1024*dh[2]*dh[2]);
	
    double L2=0;			//norm
    bool converged= false;

    /*solve potential*/
    for (unsigned it=0;it<max_solver_it;it++)
    {
        //number of steps to iterate at the finest level
        unsigned inner_its = 1;

        //number of steps to iterate at the coarse level        
        unsigned inner2h_its = 5;

        unsigned inner4h_its = 5;

        unsigned inner8h_its = 5;

        unsigned inner16h_its = 5;

        unsigned inner32h_its = 50;

        // 1) perform one or more iterations on fine mesh
		for (unsigned its=0;its<inner_its;its++)
		{
			 for (int i=1;i<world.ni-1;i++)
				for (int j=1;j<world.nj-1;j++)
					for (int k=1;k<world.nk-1;k++)
					{
						//standard internal open node
						double phi_new = (rho[i][j][k]/Const::EPS_0 +
										idx2*(phi[i-1][j][k] + phi[i+1][j][k]) +
										idy2*(phi[i][j-1][k]+phi[i][j+1][k]) +
										idz2*(phi[i][j][k-1]+phi[i][j][k+1]))/(2*idx2+2*idy2+2*idz2);

						/*SOR*/
						phi[i][j][k] = phi[i][j][k] + 1.4*(phi_new-phi[i][j][k]);
						
					}
		}
				
        // 2) compute residue on the fine mesh, R = A*phi - b
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					if ((i == 0) || (i == world.ni-1) || (j == 0) || (j == world.nj-1) || (k == 0) || (k == world.nk-1)) {
						R_h[i][j][k] = phi[i][j][k] - 0;  // dirichlet boundary
					} else {
						R_h[i][j][k] = phi[i][j][k]*(2*idx2+2*idy2+2*idz2) -
								rho[i][j][k]/Const::EPS_0 -
								idx2*(phi[i-1][j][k] + phi[i+1][j][k]) -
								idy2*(phi[i][j-1][k]+phi[i][j+1][k]) -
								idz2*(phi[i][j][k-1]+phi[i][j][k+1]);
					}
				}
                    
        // 2b) check for termination

		 /*check for convergence*/
		double sum = 0;
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					sum += R_h[i][j][k]*R_h[i][j][k];
				}

		L2 = sqrt(sum/(world.ni*world.nj*world.nk));
		if (L2<tolerance) {converged=true;break;}
		
        // 3) restrict residue to the 2h mesh
		for (int i=0;i<world.ni-1;i+=2)
			for (int j=0;j<world.nj-1;j+=2)
				for (int k=0;k<world.nk-1;k+=2)
				{
					if ((i == 0) || (j == 0)  || (k == 0) ) {
						R_2h[i/2][j/2][k/2] = R_h[i][j][k];  //R_2h[0] = R_h[0]
					} else {
						R_2h[i/2][j/2][k/2] = (R_h[i-1][j-1][k-1] + 2*R_h[i][j-1][k-1] + R_h[i+1][j-1][k-1] +
											   2*R_h[i-1][j][k-1] + 4*R_h[i][j][k-1] + 2*R_h[i+1][j][k-1] +
											   R_h[i-1][j+1][k-1] + 2*R_h[i][j+1][k-1] + R_h[i+1][j+1][k-1] +
											   2*R_h[i-1][j-1][k] + 4*R_h[i][j-1][k] + 2*R_h[i+1][j-1][k] +
											   4*R_h[i-1][j][k] + 8*R_h[i][j][k] + 4*R_h[i+1][j][k] +
											   2*R_h[i-1][j+1][k] + 4*R_h[i][j+1][k] + 2*R_h[i+1][j+1][k] +
											   R_h[i-1][j-1][k+1] + 2*R_h[i][j-1][k+1] + R_h[i+1][j-1][k+1] +
											   2*R_h[i-1][j][k+1] + 4*R_h[i][j][k+1] + 2*R_h[i+1][j][k+1] +
											   R_h[i-1][j+1][k+1] + 2*R_h[i][j+1][k+1] + R_h[i+1][j+1][k+1])/64.0;
					}
				}
		
         // 3) restrict residue to the 4h mesh
		for (int i=0;i<world.ni/2-1;i+=2)
			for (int j=0;j<world.nj/2-1;j+=2)
				for (int k=0;k<world.nk/2-1;k+=2)
				{
					if ((i == 0) || (j == 0) || (k == 0) ) {
						R_4h[i/2][j/2][k/2] = R_2h[i][j][k];  // dirichlet boundary
					} else {
						R_4h[i/2][j/2][k/2] = (R_2h[i-1][j-1][k-1] + 2*R_2h[i][j-1][k-1] + R_2h[i+1][j-1][k-1] +
											   2*R_2h[i-1][j][k-1] + 4*R_2h[i][j][k-1] + 2*R_2h[i+1][j][k-1] +
											   R_2h[i-1][j+1][k-1] + 2*R_2h[i][j+1][k-1] + R_2h[i+1][j+1][k-1] +
											   2*R_2h[i-1][j-1][k] + 4*R_2h[i][j-1][k] + 2*R_2h[i+1][j-1][k] +
											   4*R_2h[i-1][j][k] + 8*R_2h[i][j][k] + 4*R_2h[i+1][j][k] +
											   2*R_2h[i-1][j+1][k] + 4*R_2h[i][j+1][k] + 2*R_2h[i+1][j+1][k] +
											   R_2h[i-1][j-1][k+1] + 2*R_2h[i][j-1][k+1] + R_2h[i+1][j-1][k+1] +
											   2*R_2h[i-1][j][k+1] + 4*R_2h[i][j][k+1] + 2*R_2h[i+1][j][k+1] +
											   R_2h[i-1][j+1][k+1] + 2*R_2h[i][j+1][k+1] + R_2h[i+1][j+1][k+1])/64.0;
					}
				}
           
         // 3) restrict residue to the 8h mesh
		for (int i=0;i<world.ni/4-1;i+=2)
			for (int j=0;j<world.nj/4-1;j+=2)
				for (int k=0;k<world.nk/4-1;k+=2)
				{
					if ((i == 0) || (j == 0) || (k == 0) ) {
						R_8h[i/2][j/2][k/2] = R_4h[i][j][k];  // dirichlet boundary
					} else {
						R_8h[i/2][j/2][k/2] = (R_4h[i-1][j-1][k-1] + 2*R_4h[i][j-1][k-1] + R_4h[i+1][j-1][k-1] +
											   2*R_4h[i-1][j][k-1] + 4*R_4h[i][j][k-1] + 2*R_4h[i+1][j][k-1] +
											   R_4h[i-1][j+1][k-1] + 2*R_4h[i][j+1][k-1] + R_4h[i+1][j+1][k-1] +
											   2*R_4h[i-1][j-1][k] + 4*R_4h[i][j-1][k] + 2*R_4h[i+1][j-1][k] +
											   4*R_4h[i-1][j][k] + 8*R_4h[i][j][k] + 4*R_4h[i+1][j][k] +
											   2*R_4h[i-1][j+1][k] + 4*R_4h[i][j+1][k] + 2*R_4h[i+1][j+1][k] +
											   R_4h[i-1][j-1][k+1] + 2*R_4h[i][j-1][k+1] + R_4h[i+1][j-1][k+1] +
											   2*R_4h[i-1][j][k+1] + 4*R_4h[i][j][k+1] + 2*R_4h[i+1][j][k+1] +
											   R_4h[i-1][j+1][k+1] + 2*R_4h[i][j+1][k+1] + R_4h[i+1][j+1][k+1])/64.0;
					}
				}
           
         // 3) restrict residue to the 16h mesh
		for (int i=0;i<world.ni/8-1;i+=2)
			for (int j=0;j<world.nj/8-1;j+=2)
				for (int k=0;k<world.nk/8-1;k+=2)
				{
					if ((i == 0) || (j == 0) || (k == 0) ) {
						R_16h[i/2][j/2][k/2] = R_8h[i][j][k];  // dirichlet boundary
					} else {
						R_16h[i/2][j/2][k/2] = (R_8h[i-1][j-1][k-1] + 2*R_8h[i][j-1][k-1] + R_8h[i+1][j-1][k-1] +
											   2*R_8h[i-1][j][k-1] + 4*R_8h[i][j][k-1] + 2*R_8h[i+1][j][k-1] +
											   R_8h[i-1][j+1][k-1] + 2*R_8h[i][j+1][k-1] + R_8h[i+1][j+1][k-1] +
											   2*R_8h[i-1][j-1][k] + 4*R_8h[i][j-1][k] + 2*R_8h[i+1][j-1][k] +
											   4*R_8h[i-1][j][k] + 8*R_8h[i][j][k] + 4*R_8h[i+1][j][k] +
											   2*R_8h[i-1][j+1][k] + 4*R_8h[i][j+1][k] + 2*R_8h[i+1][j+1][k] +
											   R_8h[i-1][j-1][k+1] + 2*R_8h[i][j-1][k+1] + R_8h[i+1][j-1][k+1] +
											   2*R_8h[i-1][j][k+1] + 4*R_8h[i][j][k+1] + 2*R_8h[i+1][j][k+1] +
											   R_8h[i-1][j+1][k+1] + 2*R_8h[i][j+1][k+1] + R_8h[i+1][j+1][k+1])/64.0;
					}
				}

         // 3) restrict residue to the 32h mesh
		for (int i=0;i<world.ni/16-1;i+=2)
			for (int j=0;j<world.nj/16-1;j+=2)
				for (int k=0;k<world.nk/16-1;k+=2)
				{
					if ((i == 0) || (j == 0) || (k == 0) ) {
						R_32h[i/2][j/2][k/2] = R_16h[i][j][k];  // dirichlet boundary
					} else {
						R_32h[i/2][j/2][k/2] = (R_16h[i-1][j-1][k-1] + 2*R_16h[i][j-1][k-1] + R_16h[i+1][j-1][k-1] +
											   2*R_16h[i-1][j][k-1] + 4*R_16h[i][j][k-1] + 2*R_16h[i+1][j][k-1] +
											   R_16h[i-1][j+1][k-1] + 2*R_16h[i][j+1][k-1] + R_16h[i+1][j+1][k-1] +
											   2*R_16h[i-1][j-1][k] + 4*R_16h[i][j-1][k] + 2*R_16h[i+1][j-1][k] +
											   4*R_16h[i-1][j][k] + 8*R_16h[i][j][k] + 4*R_16h[i+1][j][k] +
											   2*R_16h[i-1][j+1][k] + 4*R_16h[i][j+1][k] + 2*R_16h[i+1][j+1][k] +
											   R_16h[i-1][j-1][k+1] + 2*R_16h[i][j-1][k+1] + R_16h[i+1][j-1][k+1] +
											   2*R_16h[i-1][j][k+1] + 4*R_16h[i][j][k+1] + 2*R_16h[i+1][j][k+1] +
											   R_16h[i-1][j+1][k+1] + 2*R_16h[i][j+1][k+1] + R_16h[i+1][j+1][k+1])/64.0;
					}
				}

		// 4) perform few iteration of the correction vector on the 32h mesh
		for (unsigned its=0;its<inner32h_its;its++)
		{
			 for (int i=1;i<world.ni/32-1;i++)
				for (int j=1;j<world.nj/32-1;j++)
					for (int k=1;k<world.nk/32-1;k++)
					{
						//standard internal open node
						double g = (R_32h[i][j][k] +
										idx2_32h*(eps_32h[i-1][j][k] + eps_32h[i+1][j][k]) +
										idy2_32h*(eps_32h[i][j-1][k]+eps_32h[i][j+1][k]) +
										idz2_32h*(eps_32h[i][j][k-1]+eps_32h[i][j][k+1]))/(2*idx2_32h+2*idy2_32h+2*idz2_32h);

						// SOR
						eps_32h[i][j][k] = eps_32h[i][j][k] + 1.4*(g-eps_32h[i][j][k]);
					}
		}

        // 5) interpolate eps to 16h mesh
		for (int i=0;i<world.ni/16-1;i++)
			for (int j=0;j<world.nj/16-1;j++)
				for (int k=0;k<world.nk/16-1;k++)
				{
					if ((i == 0) || (i == world.ni/4-1) || (j == 0) || (j == world.nj/4-1) || (k == 0) || (k == world.nk/4-1)) {
						eps_16h[i][j][k] = 0;    // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_16h[i][j][k] = eps_32h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_16h[i][j][k] = 0.5*(eps_32h[i/2][j/2][k/2] + eps_32h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_16h[i][j][k] = 0.5*(eps_32h[i/2][j/2][k/2] + eps_32h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_16h[i][j][k] = 0.5*(eps_32h[i/2][j/2][k/2] + eps_32h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_16h[i][j][k] = 0.25*(eps_32h[i/2][j/2][k/2] + eps_32h[i/2 + 1][j/2][k/2] +
						                  eps_32h[i/2][j/2+1][k/2] + eps_32h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_16h[i][j][k] = 0.25*(eps_32h[i/2][j/2][k/2] + eps_32h[i/2+1][j/2][k/2] +
						                  eps_32h[i/2][j/2][k/2+1] + eps_32h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_16h[i][j][k] = 0.25*(eps_32h[i/2][j/2][k/2] + eps_32h[i/2][j/2+1][k/2] +
						                  eps_32h[i/2][j/2][k/2+1] + eps_32h[i/2][j/2+1][k/2+1]);
					} else {
						eps_16h[i][j][k] = 0.125*(eps_32h[i/2][j/2][k/2] + eps_32h[i/2 + 1][j/2][k/2] +
						                  eps_32h[i/2][j/2+1][k/2] + eps_32h[i/2 + 1][j/2+1][k/2] +
						                  eps_32h[i/2][j/2][k/2+1] + eps_32h[i/2 + 1][j/2][k/2+1] +
						                  eps_32h[i/2][j/2+1][k/2+1] + eps_32h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
           
		// 4) perform few iteration of the correction vector on the 16h mesh
		for (unsigned its=0;its<inner16h_its;its++)
		{
			 for (int i=1;i<world.ni/16-1;i++)
				for (int j=1;j<world.nj/16-1;j++)
					for (int k=1;k<world.nk/16-1;k++)
					{
						//standard internal open node
						double g = (R_16h[i][j][k] +
										idx2_16h*(eps_16h[i-1][j][k] + eps_16h[i+1][j][k]) +
										idy2_16h*(eps_16h[i][j-1][k]+eps_16h[i][j+1][k]) +
										idz2_16h*(eps_16h[i][j][k-1]+eps_16h[i][j][k+1]))/(2*idx2_16h+2*idy2_16h+2*idz2_16h);

						// SOR
						eps_16h[i][j][k] = eps_16h[i][j][k] + 1.4*(g-eps_16h[i][j][k]);
					}
		}

        // 5) interpolate eps to 8h mesh
		for (int i=0;i<world.ni/8-1;i++)
			for (int j=0;j<world.nj/8-1;j++)
				for (int k=0;k<world.nk/8-1;k++)
				{
					if ((i == 0) || (i == world.ni/4-1) || (j == 0) || (j == world.nj/4-1) || (k == 0) || (k == world.nk/4-1)) {
						eps_8h[i][j][k] = 0;    // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_8h[i][j][k] = eps_16h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_8h[i][j][k] = 0.5*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_8h[i][j][k] = 0.5*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_8h[i][j][k] = 0.5*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_8h[i][j][k] = 0.25*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2 + 1][j/2][k/2] +
						                  eps_16h[i/2][j/2+1][k/2] + eps_16h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_8h[i][j][k] = 0.25*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2+1][j/2][k/2] +
						                  eps_16h[i/2][j/2][k/2+1] + eps_16h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_8h[i][j][k] = 0.25*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2][j/2+1][k/2] +
						                  eps_16h[i/2][j/2][k/2+1] + eps_16h[i/2][j/2+1][k/2+1]);
					} else {
						eps_8h[i][j][k] = 0.125*(eps_16h[i/2][j/2][k/2] + eps_16h[i/2 + 1][j/2][k/2] +
						                  eps_16h[i/2][j/2+1][k/2] + eps_16h[i/2 + 1][j/2+1][k/2] +
						                  eps_16h[i/2][j/2][k/2+1] + eps_16h[i/2 + 1][j/2][k/2+1] +
						                  eps_16h[i/2][j/2+1][k/2+1] + eps_16h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
           
		// 4) perform few iteration of the correction vector on the 8h mesh
		for (unsigned its=0;its<inner8h_its;its++)
		{
			 for (int i=1;i<world.ni/8-1;i++)
				for (int j=1;j<world.nj/8-1;j++)
					for (int k=1;k<world.nk/8-1;k++)
					{
						//standard internal open node
						double g = (R_8h[i][j][k] +
										idx2_8h*(eps_8h[i-1][j][k] + eps_8h[i+1][j][k]) +
										idy2_8h*(eps_8h[i][j-1][k]+eps_8h[i][j+1][k]) +
										idz2_8h*(eps_8h[i][j][k-1]+eps_8h[i][j][k+1]))/(2*idx2_8h+2*idy2_8h+2*idz2_8h);

						// SOR
						eps_8h[i][j][k] = eps_8h[i][j][k] + 1.4*(g-eps_8h[i][j][k]);
					}
		}

        // 5) interpolate eps to 4h mesh
		for (int i=0;i<world.ni/4-1;i++)
			for (int j=0;j<world.nj/4-1;j++)
				for (int k=0;k<world.nk/4-1;k++)
				{
					if ((i == 0) || (i == world.ni/4-1) || (j == 0) || (j == world.nj/4-1) || (k == 0) || (k == world.nk/4-1)) {
						eps_4h[i][j][k] = 0;    // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_4h[i][j][k] = eps_8h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_4h[i][j][k] = 0.5*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_4h[i][j][k] = 0.5*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_4h[i][j][k] = 0.5*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_4h[i][j][k] = 0.25*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2 + 1][j/2][k/2] +
						                  eps_8h[i/2][j/2+1][k/2] + eps_8h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_4h[i][j][k] = 0.25*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2+1][j/2][k/2] +
						                  eps_8h[i/2][j/2][k/2+1] + eps_8h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_4h[i][j][k] = 0.25*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2][j/2+1][k/2] +
						                  eps_8h[i/2][j/2][k/2+1] + eps_8h[i/2][j/2+1][k/2+1]);
					} else {
						eps_4h[i][j][k] = 0.125*(eps_8h[i/2][j/2][k/2] + eps_8h[i/2 + 1][j/2][k/2] +
						                  eps_8h[i/2][j/2+1][k/2] + eps_8h[i/2 + 1][j/2+1][k/2] +
						                  eps_8h[i/2][j/2][k/2+1] + eps_8h[i/2 + 1][j/2][k/2+1] +
						                  eps_8h[i/2][j/2+1][k/2+1] + eps_8h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
		// 4) perform few iteration of the correction vector on the 4h mesh
		for (unsigned its=0;its<inner4h_its;its++)
		{
			 for (int i=1;i<world.ni/4-1;i++)
				for (int j=1;j<world.nj/4-1;j++)
					for (int k=1;k<world.nk/4-1;k++)
					{
						//standard internal open node
						double g = (R_4h[i][j][k] +
										idx2_4h*(eps_4h[i-1][j][k] + eps_4h[i+1][j][k]) +
										idy2_4h*(eps_4h[i][j-1][k]+eps_4h[i][j+1][k]) +
										idz2_4h*(eps_4h[i][j][k-1]+eps_4h[i][j][k+1]))/(2*idx2_4h+2*idy2_4h+2*idz2_4h);

						/*SOR*/
						eps_4h[i][j][k] = eps_4h[i][j][k] + 1.4*(g-eps_4h[i][j][k]);
					}
		}

        // 5) interpolate eps to 2h mesh
		for (int i=0;i<world.ni/2-1;i++)
			for (int j=0;j<world.nj/2-1;j++)
				for (int k=0;k<world.nk/2-1;k++)
				{
					if ((i == 0) || (i == world.ni/2-1) || (j == 0) || (j == world.nj/2-1) || (k == 0) || (k == world.nk/2-1)) {
						eps_2h[i][j][k] = 0;    // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_2h[i][j][k] = eps_4h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.5*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2 + 1][j/2][k/2] +
						                  eps_4h[i/2][j/2+1][k/2] + eps_4h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2+1][j/2][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_2h[i][j][k] = 0.25*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2][j/2+1][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2][j/2+1][k/2+1]);
					} else {
						eps_2h[i][j][k] = 0.125*(eps_4h[i/2][j/2][k/2] + eps_4h[i/2 + 1][j/2][k/2] +
						                  eps_4h[i/2][j/2+1][k/2] + eps_4h[i/2 + 1][j/2+1][k/2] +
						                  eps_4h[i/2][j/2][k/2+1] + eps_4h[i/2 + 1][j/2][k/2+1] +
						                  eps_4h[i/2][j/2+1][k/2+1] + eps_4h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
		// 4) perform few iteration of the correction vector on the 2h mesh
		for (unsigned its=0;its<inner2h_its;its++)
		{
			for (int i=1;i<world.ni/2-1;i++)
				for (int j=1;j<world.nj/2-1;j++)
					for (int k=1;k<world.nk/2-1;k++)
					{
						//standard internal open node
						double g = (R_2h[i][j][k] +
										idx2_2h*(eps_2h[i-1][j][k] + eps_2h[i+1][j][k]) +
										idy2_2h*(eps_2h[i][j-1][k]+eps_2h[i][j+1][k]) +
										idz2_2h*(eps_2h[i][j][k-1]+eps_2h[i][j][k+1]))/(2*idx2_2h+2*idy2_2h+2*idz2_2h);

						// SOR
						eps_2h[i][j][k] = eps_2h[i][j][k] + 1.4*(g-eps_2h[i][j][k]);
					}
		}

        // 5) interpolate eps to h mesh
		for (int i=0;i<world.ni-1;i++)
			for (int j=0;j<world.nj-1;j++)
				for (int k=0;k<world.nk-1;k++)
				{
					if ((i == 0) || (i == world.ni-1) || (j == 0) || (j == world.nj-1) || (k == 0) || (k == world.nk-1)) {
						eps_h[i][j][k] = 0;   // Dirichlet boundary
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 0))  // even nodes, overlapping coarse mesh
					{
						eps_h[i][j][k] = eps_2h[i/2][j/2][k/2];
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2+1][j/2][k/2]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2+1][k/2]);
					} else if ((i%2 == 0) && (j%2 == 0) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.5*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2][k/2+1]);
					} else if ((i%2 == 1) && (j%2 == 1) && (k%2 == 0))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2 + 1][j/2][k/2] +
						                  eps_2h[i/2][j/2+1][k/2] + eps_2h[i/2 + 1][j/2+1][k/2]);
					} else if ((i%2 == 1) && (j%2 == 0) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2+1][j/2][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2+1][j/2][k/2+1]);
					} else if ((i%2 == 0) && (j%2 == 1) && (k%2 == 1))
					{
						eps_h[i][j][k] = 0.25*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2][j/2+1][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2][j/2+1][k/2+1]);
					} else {
						eps_h[i][j][k] = 0.125*(eps_2h[i/2][j/2][k/2] + eps_2h[i/2 + 1][j/2][k/2] +
						                  eps_2h[i/2][j/2+1][k/2] + eps_2h[i/2 + 1][j/2+1][k/2] +
						                  eps_2h[i/2][j/2][k/2+1] + eps_2h[i/2 + 1][j/2][k/2+1] +
						                  eps_2h[i/2][j/2+1][k/2+1] + eps_2h[i/2 + 1][j/2+1][k/2+1]);
					}
				}
        
        // 6) update solution on the fine mesh
		for (int i=0;i<world.ni;i++)
			for (int j=0;j<world.nj;j++)
				for (int k=0;k<world.nk;k++)
				{
					phi[i][j][k] = phi[i][j][k] - eps_h[i][j][k];
				}

    }

	/*
	int l = 0;
    double erl1 = 0.;
    double erl2 = 0.;
    double erl3 = 0.;
	for (int i=1;i<world.ni-1;i++)
		for (int j=1;j<world.nj-1;j++)
			for (int k=1;k<world.nk-1;k++)
			{
				// consecutive ordering
				//tmp[i+1][j+1][k+1] = phi[i+1][j+1][k+1];
				//phiF[i+1][j+1][k+1] = 0.125*out2[l]/((double)(Nx+1))/((double)(Ny+1))/((double)(Nz+1));
				erl1 +=pow(fabs(phi[i][j][k] - phi_test[i][j][k]),2);
				erl2 +=pow(fabs(phi[i][j][k]),2);
				erl3 +=pow(fabs(phi_test[i][j][k]),2);
				
				
				if ((l > world.ni*world.nj*world.nk/2) && (l < world.ni*world.nj*world.nk/2 + world.nk*2))
				{
					//cout<< setprecision(7) << "phi[" << i << "][" << j << "]["<<k<<"] = " << tmp[i+1][j+1][k+1] << " , phiF[" << i << "][" << j << "]["<<k<<"] = " << phiF[i+1][j+1][k+1] << "\n";
					cout<< setprecision(7) << "phiMG[" << i << "][" << j << "]["<<k<<"] = " << phi[i][j][k] << " , phi[" << i << "][" << j << "]["<<k<<"] = " << phi_test[i][j][k] << "\n";
				}
				l++;
												
			}
    cout<< setprecision(7) << "\n phi - phi_test error=" <<sqrt(erl1) << " , " << sqrt(erl2) << " , " << sqrt(erl3) << endl ;  
	*/

    if (!converged) cerr<<"GS failed to converge, L2="<<L2<<endl;
    return converged;
}





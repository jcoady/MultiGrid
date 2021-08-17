#include "PotentialSolver.h"
#include "Field.h"
#include <math.h>
#include <iostream>
#include <iomanip>
#include "World.h"
#include "solvers.h"

#include <cmath>
#include <fftw3.h>

#define SIZE 8
#define NXX SIZE
#define NYY SIZE
#define NZZ SIZE
#include <vector>
#include<algorithm>

using namespace alglib;



using namespace std;
using namespace Const;

using vec    = vector<double>;         // vector
using matrix = vector<vec>;            // matrix (=collection of (row) vectors)

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
	/*
    /// Grid size
    int Nx=NXX,Ny=NYY,Nz=NZZ,Nzh=(Nz/2+1);

    /// Declare FFTW components.
    fftw_complex *mem;
    fftw_complex *out;
    double *in;
    mem = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nx * Ny * Nzh);
    out = mem;
    in = mem[0];

    fftw_plan fwrd = fftw_plan_dft_r2c_3d(Nx,Ny,Nz,in,out,FFTW_MEASURE);
    fftw_plan bwrd = fftw_plan_dft_c2r_3d(Nx,Ny,Nz,out,in,FFTW_MEASURE);
    */
	
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

ConjugateGradientSolver::ConjugateGradientSolver(World &world): PotentialSolver(world) {	


	Nx = world.ni-2;     // Don't include the boundary nodes for Fourier Solver
	Ny = world.nj-2;
	Nz = world.nk-2;
	
	int n = Nx*Ny*Nz;
	int m = Nx*Ny*Nz;
	
	//K.resize(Nx, vec (Nx, 0));
	//I.resize(Ny, vec (Ny, 0));
	//J.resize(Nz, vec (Nz, 0));

	double3 X0 = world.getX0();
	double3 Xm = world.getXm();
    double xlen = Xm[0]-X0[0];
	double ylen = Xm[1]-X0[1];
	double zlen = Xm[2]-X0[2];
	
    double dx = xlen/(double)(Nx+1);
    double dy=ylen/(double)(Ny+1);
    double dz=zlen/(double)(Nz+1);

	/*
	for (int i=0; i<Nx; i++)
		for (int j=0;j<Nx;j++) {
			if(i == j) {
				K[i][j] = 2;
			}
			if (j == i+1) {
				K[i][j] = -1;
			}
			if (j == i-1) {
				K[i][j] = -1;
			}
		}

	for (int i=0; i<Ny; i++)
		for (int j=0;j<Ny;j++) {
			if(i == j) {
				I[i][j] = 1;
			}
		}

	for (int i=0; i<Nz; i++)
		for (int j=0;j<Nz;j++) {
			if(i == j) {
				J[i][j] = 1;
			}
		}

	int n = I.size() * K.size();
	int m = I[0].size() * K[0].size();
	matrix IK(n,vec (m,0.0));
	matrix KI(n,vec (m,0.0));
	//matrix K2D(n,vec (m,0.0));
		
	Kroneckerproduct(I, K, IK);
	Kroneckerproduct(K, I, KI);

	n = J.size() * IK.size();
	m = J[0].size() * IK[0].size();
	//n = rowa * rowb * rowc;
	//m = cola * colb * colc;
	matrix IIK(n,vec (m,0.0));
	matrix IKI(n,vec (m,0.0));
	matrix KII(n,vec (m,0.0));
	
	//matrix K3D(n,vec (m,0.0));
	//A.resize(n, vec (m, 0));
	//B.resize(Nx*Ny*Nz,0.0);
		
	Kroneckerproduct(J, IK, IIK);
	Kroneckerproduct(IK, J, IKI);
	Kroneckerproduct(KI, J, KII);
	*/
	cout << endl << endl << endl;
	
    //sparsematrix a;
    sparsecreate(n, m, a);
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < m; j++)
		{
			if (i == j) {
				sparseset(a, i, j, 2/(dx*dx) + 2/(dy*dy) + 2/(dz*dz));
			}
			else if ((j == i+1) && (j != ((i+1)/Nz)*Nz)) {
				sparseset(a, i, j, -1/(dz*dz));
			}
			else if ((i == j+1) && (i != ((j+1)/Nz)*Nz)) {
				sparseset(a, i, j, -1/(dz*dz));
			}
			else if ((j == (i+Nz)) && ((i%Nz) != (j%(Nz*Ny)))) {
				sparseset(a, i, j, -1/(dy*dy));
			}
			else if ((i == (j+Nz)) && ((j%Nz) != (i%(Nz*Ny)))) {
				sparseset(a, i, j, -1/(dy*dy));
			}
			else if (j == (i+Nz*Ny)) {
				sparseset(a, i, j, -1/(dx*dx));
			}
			else if (i == (j+Nz*Ny)) {
				sparseset(a, i, j, -1/(dx*dx));
			}
		}
	}

	
    //
    // Now our matrix is fully initialized, but we have to do one more
    // step - convert it from Hash-Table format to CRS format (see
    // documentation on sparse matrices for more information about these
    // formats).
    //
    // If you omit this call, ALGLIB will generate exception on the first
    // attempt to use A in linear operations. 
    //
    sparseconverttocrs(a);

    //real_1d_array b;
	b.setlength(m);

    lincgcreate(Nx*Ny*Nz, s);


	/*
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < m; j++)
		{
			A[i][j] = IIK[i][j]/(dz*dz) + IKI[i][j]/(dx*dx) + KII[i][j]/(dy*dy);
			//K3D[i][j] = IIK[i][j]/(dz*dz) + IKI[i][j]/(dx*dx) + KII[i][j]/(dy*dy);
			//K3D[i][j] = IIK[i][j] + IKI[i][j] + KII[i][j];
		}
	}
	*/

}

ConjugateGradientSolver::~ConjugateGradientSolver() {
	
}
	

/*solves Poisson equation using FFT*/
bool ConjugateGradientSolver::solve()
{
	Field &phi = world.phi;
    Field &rho = world.rho;
	
	/*
	double3 X0 = world.getX0();
	double3 Xm = world.getXm();
    double xlen = Xm[0]-X0[0];
	double ylen = Xm[1]-X0[1];
	double zlen = Xm[2]-X0[2];
	
    double dx = xlen/(double)(Nx+1);
    double dy=ylen/(double)(Ny+1);
    double dz=zlen/(double)(Nz+1);
	*/
	
    int l=0;
	// set some values
	for (int i=0; i<Nx; i++)
		for (int j=0;j<Ny;j++)
			for(int k=0;k<Nz;k++) {				
				// consecutive ordering
				//size_t u = k*Nx*Ny + j*Nx + i;
				//in[u] = u;
				//B[l] = rho[i+1][j+1][k+1]/Const::EPS_0;
				b[l] = rho[i+1][j+1][k+1]/Const::EPS_0;
				l=l+1;
			}

    //
    // Now we have to create linear solver object and to use it for the
    // solution of the linear system.
    //
    // NOTE: lincgsolvesparse() accepts additional parameter which tells
    //       what triangle of the symmetric matrix should be used - upper
    //       or lower. Because we've filled both parts of the matrix, we
    //       can use any part - upper or lower.
    //
	/*
    lincgstate s;
    lincgreport rep;
    real_1d_array x;
    lincgcreate(Nx*Ny*Nz, s);
	*/
	
    lincgsolvesparse(s, a, true, b);

    //vec U = conjugateGradientSolver( A, B );
	
    lincgresults(s, x, rep);
	
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
				
				//phi[i+1][j+1][k+1] = U[l];
				phi[i+1][j+1][k+1] = x[l];
								
			}
    //cout<< setprecision(7) << "\n phiF error=" <<sqrt(erl1) << " , " << sqrt(erl2) << " , " << sqrt(erl3) << endl ;  
     


	return true;
}

vec ConjugateGradientSolver::matrixTimesVector( const matrix &A, const vec &V )     // Matrix times vector
{
   int n = A.size();
   vec C( n );
   for ( int i = 0; i < n; i++ ) C[i] = innerProduct( A[i], V );
   return C;
}


//======================================================================


vec ConjugateGradientSolver::vectorCombination( double a, const vec &U, double b, const vec &V )        // Linear combination of vectors
{
   int n = U.size();
   vec W( n );
   for ( int j = 0; j < n; j++ ) W[j] = a * U[j] + b * V[j];
   return W;
}


//======================================================================


double ConjugateGradientSolver::innerProduct( const vec &U, const vec &V )          // Inner product of U and V
{
   return inner_product( U.begin(), U.end(), V.begin(), 0.0 );
}


//======================================================================


double ConjugateGradientSolver::vectorNorm( const vec &V )                          // Vector norm
{
   return sqrt( innerProduct( V, V ) );
}


//======================================================================


vec ConjugateGradientSolver::conjugateGradientSolver( const matrix &A, const vec &B )
{
   double TOLERANCE = 1.0e-10;

   int n = A.size();
   vec X( n, 0.0 );

   vec R = B;
   vec P = R;
   int k = 0;

   while ( k < n )
   {
      vec Rold = R;                                         // Store previous residual
      vec AP = matrixTimesVector( A, P );

      double alpha = innerProduct( R, R ) / max( innerProduct( P, AP ), NEARZERO );
      X = vectorCombination( 1.0, X, alpha, P );            // Next estimate of solution
      R = vectorCombination( 1.0, R, -alpha, AP );          // Residual 

      if ( vectorNorm( R ) < TOLERANCE ) break;             // Convergence test

      double beta = innerProduct( R, R ) / max( innerProduct( Rold, Rold ), NEARZERO );
      P = vectorCombination( 1.0, R, beta, P );             // Next gradient
      k++;
   }

   return X;
}


void ConjugateGradientSolver::Kroneckerproduct(matrix &A, matrix &B, matrix &C)
{

   int p = A.size(), q = A[0].size();                      // A is an p x q matrix
   int r = B.size(), s = B[0].size();                      // B is an r x s matrix
   
	// i loops till rowa
	for (int i = 0; i < p; i++) {

		// k loops till rowb
		for (int k = 0; k < r; k++) {

			// j loops till cola
			for (int j = 0; j < q; j++) {

				// l loops till colb
				for (int l = 0; l < s; l++) {

					// Each element of matrix A is
					// multiplied by whole Matrix B
					// resp and stored as Matrix C
					C[i*r + k][j*s + l] = A[i][j] * B[k][l];
					//cout << C[i*r + k][j*s + l] << " ";
				}
			}
		}
	}
}


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





#ifndef _SOLVER_H
#define _SOLVER_H

#include "World.h"
#include <cmath>
#include <fftw3.h>
#include "Field.h"
#include <vector>
#include <math.h>
#include "solvers.h"

using namespace alglib;

using namespace std;

using vec    = vector<double>;         // vector
using matrix = vector<vec>;            // matrix (=collection of (row) vectors)


class PotentialSolver
{
public:
	/*constructor, sets world*/
	PotentialSolver(World &world): 
		world(world) {}
	
	/*solves potential using Gauss-Seidel*/
	virtual bool solve() = 0;
	
	/*computes electric field = -gradient(phi)*/
	void computeEF();

protected:
	World &world;
};

class GaussSeidelSolver: public PotentialSolver
{
public:
	/*constructor, sets world*/
	GaussSeidelSolver(World &world, int max_it, double tol): 
		PotentialSolver(world), max_solver_it(max_it), tolerance(tol) {}
	
	/*solves potential using Gauss-Seidel*/
	bool solve();
	
protected:
	unsigned max_solver_it;	//maximum number of solver iterations
	double tolerance;		//solver tolerance
};

class FourierSolver: public PotentialSolver
{
public:
	/*constructor, sets world*/
	FourierSolver(World &world);

	~FourierSolver();
	
	/* solves potential using 3D FFT */
	bool solve();

private:
    /// Grid size
	int Nx, Ny, Nz;

    /// Declare FFTW components.
    //fftw_complex *mem;
    //fftw_complex *out;
    //double *in;
	
	std::vector<double> in1;
	std::vector<double> in2;
	std::vector<double> out1;
	std::vector<double> out2;
	
	fftw_plan fwrd;
    fftw_plan bwrd;
	Field phiF;			//potential

};

class ConjugateGradientSolver: public PotentialSolver
{
public:
	/*constructor, sets world*/
	ConjugateGradientSolver(World &world);

	~ConjugateGradientSolver();
	
	/*solves potential using ConjugateGradient*/
	bool solve();
	void Kroneckerproduct(matrix &A, matrix &B, matrix &C);
	vec matrixTimesVector( const matrix &A, const vec &V );
	vec vectorCombination( double a, const vec &U, double b, const vec &V );
	double innerProduct( const vec &U, const vec &V );
	double vectorNorm( const vec &V );
	vec conjugateGradientSolver( const matrix &A, const vec &B );
	
protected:
    /// Grid size
	int Nx, Ny, Nz;
	
	//std::vector<double> B;
    real_1d_array b;
	
	//matrix A;
    sparsematrix a;

    lincgstate s;
    lincgreport rep;
    real_1d_array x;
	
	//matrix I;
	//matrix K;
	//matrix J;
	
	const double NEARZERO = 1.0e-10;       // interpretation of "zero"

};

class MultiGridSolver: public GaussSeidelSolver
{
public:
	/*constructor, sets world*/
	MultiGridSolver(World &world, int max_it, double tol): 
		GaussSeidelSolver(world, max_it, tol), phi_test(world.ni,world.nj,world.nk),
		R_h(world.ni,world.nj,world.nk), R_2h(world.ni/2,world.nj/2,world.nk/2),
		eps_h(world.ni,world.nj,world.nk), eps_2h(world.ni/2+1,world.nj/2+1,world.nk/2+1) {}
	
	/*solves potential using Gauss-phi*/
	bool solve();
	
protected:
	Field R_h;		// Residual R_h
	Field R_2h;		// Residual R_2h
	Field eps_2h;	// error eps_2h (Correction term)
	Field eps_h;	// error eps_h (Correction term)

	Field phi_test;		// Field for test purposes

};

class MultiGridSolverV2: public MultiGridSolver
{
public:
	/*constructor, sets world*/
	MultiGridSolverV2(World &world, int max_it, double tol): 
		MultiGridSolver(world, max_it, tol), 
		R_4h(world.ni/4,world.nj/4,world.nk/4), eps_4h(world.ni/4,world.nj/4,world.nk/4) {}
	
	/*solves potential using Gauss-phi*/
	bool solve();
	
protected:
	Field R_4h;		// Residual R_2h
	Field eps_4h;	// error eps_2h (Correction term)

};

class MultiGridSolverV3: public MultiGridSolverV2
{
public:
	/*constructor, sets world*/
	MultiGridSolverV3(World &world, int max_it, double tol): 
		MultiGridSolverV2(world, max_it, tol), 
		R_8h(world.ni/8,world.nj/8,world.nk/8), eps_8h(world.ni/8+1,world.nj/8+1,world.nk/8+1) {}
	
	/*solves potential using Gauss-phi*/
	bool solve();
	
protected:
	Field R_8h;		// Residual R_2h
	Field eps_8h;	// error eps_2h (Correction term)

};


class MultiGridSolverV4: public MultiGridSolverV3
{
public:
	//constructor, sets world
	MultiGridSolverV4(World &world, int max_it, double tol): 
		MultiGridSolverV3(world, max_it, tol), 
		R_16h(world.ni/16,world.nj/16,world.nk/16), eps_16h(world.ni/16,world.nj/16,world.nk/16) {}
	
	//solves potential using Gauss-phi
	bool solve();
	
protected:
	Field R_16h;		// Residual R_2h
	Field eps_16h;	// error eps_2h (Correction term)

};

class MultiGridSolverB: public MultiGridSolver
{
public:
	//constructor, sets world
	MultiGridSolverB(World &world, int max_it, double tol): 
		MultiGridSolver(world, max_it, tol) {}
	
	/*solves potential using Gauss-phi*/
	bool solve();
	
protected:

};

class MultiGridSolverV2B: public MultiGridSolverV2
{
public:
	//constructor, sets world
	MultiGridSolverV2B(World &world, int max_it, double tol): 
		MultiGridSolverV2(world, max_it, tol) {}
	
	/*solves potential using Gauss-phi*/
	bool solve();
	
protected:

};


#endif

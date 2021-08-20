#ifndef _SOLVER_H
#define _SOLVER_H

#include "World.h"
#include <cmath>
#include <fftw3.h>
#include "Field.h"
#include <vector>
#include <math.h>
#include <assert.h>
#include <iostream>

using namespace std;

//structure to hold data for a single row
template <int S>
struct Row {
	Row() {for (int i=0;i<S;i++) {a[i]=0;col[i]=-1;}}
	void operator= (const Row &o) {for (int i=0;i<S;i++) {a[i] = o.a[i];col[i]=o.col[i];}}
	double a[S];		//coefficients
	int col[S];
};

/*matrix with up to seven non zero diagonals*/
class Matrix
{
public:
    Matrix(int nr):nu{nr} {rows=new Row<nvals>[nr];}
    Matrix(const Matrix &o):Matrix(o.nu) {
    	for (int r=0;r<nu;r++) rows[r] = o.rows[r];
    };	//copy constructor
    ~Matrix() {if (rows) delete[] rows;}
	dvector operator*(dvector &v);	//matrix-vector multiplication

	double& operator() (int r, int c); //reference to A[r,c] value in a full matrix
	void clearRow(int r) {rows[r]=Row<nvals>();} //reinitializes a row
	Matrix diagSubtract(dvector &P);	//subtracts a vector from the diagonal
	Matrix invDiagonal();		//returns a matrix containing inverse of our diagonal
	double multRow(int r, dvector &x);	//multiplies row r with vector x

	static constexpr int nvals = 7;	//maximum 7 non-zero values
	const int nu;			//number of rows (unknowns)

protected:
	Row<nvals> *rows;	//row data
};


class PotentialSolver
{
public:
	/*constructor, sets world*/
	PotentialSolver(World &world): 
		world(world) {
			cout << "PotentialSolver Constructor" << endl;
		}
	
	virtual ~PotentialSolver() {};

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
		PotentialSolver(world), max_solver_it(max_it), tolerance(tol) {			
			cout << "GaussSeidelSolver Constructor" << endl;
		}
		
	virtual ~GaussSeidelSolver() {};
	
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

	virtual ~FourierSolver();
	
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
	ConjugateGradientSolver(World &world, int max_it, double tol): 
		PotentialSolver(world), A(world.ni*world.nj*world.nk), max_solver_it(max_it), tolerance(tol) {
			cout << "ConjugateGradientSolver Constructor" << endl;
			buildMatrix();
		}

	virtual ~ConjugateGradientSolver() {};
	
	/*solves potential using ConjugateGradient*/
	bool solve();

	/*builds the "A" matrix for linear potential solver*/
    void buildMatrix();

	/*linear PCG solver for Ax=b system*/
	//bool solvePCGLinear(Matrix &A, dvector &x, dvector &b);

	/*CG solver for Ax=b system*/
	bool solveCG(Matrix &A, dvector &x, dvector &b);
	
protected:
    Matrix A;				//system matrix for the linear equation

    enum NodeType {REG,NEUMANN,DIRICHLET};
    std::vector<NodeType> node_type;	//flag for different node types

	unsigned max_solver_it;	//maximum number of solver iterations
	double tolerance;		//solver tolerance

};

class PreconditionedConjugateGradientSolver: public ConjugateGradientSolver
{
public:
	/*constructor, sets world*/
	PreconditionedConjugateGradientSolver(World &world, int max_it, double tol): 
		ConjugateGradientSolver(world, max_it, tol) {
		}

	virtual ~PreconditionedConjugateGradientSolver() {};
	
	/*solves potential using ConjugateGradient*/
	bool solve();
	

};

class MultiGridSolver: public GaussSeidelSolver
{
public:
	/*constructor, sets world*/
	MultiGridSolver(World &world, int max_it, double tol): 
		GaussSeidelSolver(world, max_it, tol), phi_test(world.ni,world.nj,world.nk),
		R_h(world.ni,world.nj,world.nk), R_2h(world.ni/2,world.nj/2,world.nk/2),
		eps_h(world.ni,world.nj,world.nk), eps_2h(world.ni/2+1,world.nj/2+1,world.nk/2+1) {			
			cout << "MultiGridSolver Constructor" << endl;
		}

	virtual ~MultiGridSolver() {};
	
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
		R_4h(world.ni/4,world.nj/4,world.nk/4), eps_4h(world.ni/4+1,world.nj/4+1,world.nk/4+1) {}

	virtual ~MultiGridSolverV2() {};
	
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
	
	virtual ~MultiGridSolverV3() {};
	
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
		R_16h(world.ni/16,world.nj/16,world.nk/16), eps_16h(world.ni/16+1,world.nj/16+1,world.nk/16+1) {}
	
	virtual ~MultiGridSolverV4() {};
	
	//solves potential using Gauss-phi
	bool solve();
	
protected:
	Field R_16h;		// Residual R_2h
	Field eps_16h;	// error eps_2h (Correction term)

};

class MultiGridSolverV5: public MultiGridSolverV4
{
public:
	//constructor, sets world
	MultiGridSolverV5(World &world, int max_it, double tol): 
		MultiGridSolverV4(world, max_it, tol), 
		R_32h(world.ni/32,world.nj/32,world.nk/32), eps_32h(world.ni/32+1,world.nj/32+1,world.nk/32+1) {}
	
	virtual ~MultiGridSolverV5() {};
	
	//solves potential using Gauss-phi
	bool solve();
	
protected:
	Field R_32h;		// Residual R_2h
	Field eps_32h;	// error eps_2h (Correction term)

};

class MultiGridSolverB: public MultiGridSolver
{
public:
	//constructor, sets world
	MultiGridSolverB(World &world, int max_it, double tol): 
		MultiGridSolver(world, max_it, tol) {}
	
	virtual ~MultiGridSolverB() {};
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
	
	virtual ~MultiGridSolverV2B() {};
	
	/*solves potential using Gauss-phi*/
	bool solve();
	
protected:

};

class MultiGridSolverV3B: public MultiGridSolverV3
{
public:
	//constructor, sets world
	MultiGridSolverV3B(World &world, int max_it, double tol): 
		MultiGridSolverV3(world, max_it, tol) {}
	
	virtual ~MultiGridSolverV3B() {};
	
	/*solves potential using Gauss-phi*/
	bool solve();
	
protected:

};

class MultiGridSolverV4B: public MultiGridSolverV4
{
public:
	//constructor, sets world
	MultiGridSolverV4B(World &world, int max_it, double tol): 
		MultiGridSolverV4(world, max_it, tol) {}
	
	virtual ~MultiGridSolverV4B() {};
	
	/*solves potential using Gauss-phi*/
	bool solve();
	
protected:

};

class MultiGridSolverV5B: public MultiGridSolverV5
{
public:
	//constructor, sets world
	MultiGridSolverV5B(World &world, int max_it, double tol): 
		MultiGridSolverV5(world, max_it, tol) {}
	
	virtual ~MultiGridSolverV5B() {};
	
	/*solves potential using Gauss-phi*/
	bool solve();
	
protected:

};


#endif



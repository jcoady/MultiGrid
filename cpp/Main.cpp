#include <math.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "World.h"
#include "PotentialSolver.h"
#include "Species.h"
#include "Output.h"
#include <chrono>

using namespace std;		//to avoid having to write std::cout
using namespace Const;		//to avoid having to write Const::ME

/*program execution starts here*/
int main(int argc, char *argv[])
{
	//grab starting time clock
	using namespace std::chrono;
	high_resolution_clock::time_point t_start = high_resolution_clock::now();
	high_resolution_clock::time_point t_end = high_resolution_clock::now();
	high_resolution_clock::time_point t_start2 = high_resolution_clock::now();
	high_resolution_clock::time_point t_end2 = high_resolution_clock::now();
	high_resolution_clock::time_point t_start3 = high_resolution_clock::now();
	high_resolution_clock::time_point t_end3 = high_resolution_clock::now();
	std::chrono::duration<double> duration = t_end-t_start;
	std::chrono::duration<double> duration2 = t_end2-t_start2;
	std::chrono::duration<double> duration3 = t_end3-t_start3;
	   
	int mesh_size = 21;
	if (argc == 2) {
		mesh_size = atoi(argv[1]);
	}
	cout << "Using mesh size : " << mesh_size << "x" << mesh_size << "x" << mesh_size << endl;
	/*initialize domain*/
    World world(mesh_size,mesh_size,mesh_size);
    //world.setExtents({-0.1,-0.1,0},{0.1,0.1,0.2});
    world.setExtents({-0.2,-0.2,-0.1},{0.2,0.2,0.3});
    world.setTime(2e-10,1000);

	/*set up particle species*/
	vector<Species> species;
	species.reserve(2);	//pre-allocate space for two species
	species.push_back(Species("O+", 16*AMU, QE, world));
	species.push_back(Species("e-", ME, -1*QE, world));

	cout<<"Size of species "<<species.size()<<endl;

	/*create particles*/
	int3 np_ions_grid = {41,41,41};
	int3 np_eles_grid = {21,21,21};
	//species[0].loadParticlesBoxQS(world.getX0(),world.getXm(),1e11,np_ions_grid);	//ions
	//species[1].loadParticlesBoxQS(world.getX0(),world.getXc(),1e11,np_eles_grid);	//electrons
	species[0].loadParticlesBoxQS({-0.1,-0.1,0},{0.1,0.1,0.2},1e11,np_ions_grid);	//ions
	species[1].loadParticlesBoxQS({-0.1,-0.1,0},world.getXc(),1e11,np_eles_grid);	//electrons

	/*initialize potential solver and solve initial potential*/
	
	PotentialSolver * solver;
	int choice, choice2;
	
	cout << "Choose solver (1)Gauss-Seidel (2)FFT (3)Conjugate Gradient (4)MultiGrid: ";
	cin >> choice;
	
	switch (choice)
	{
		case 1: solver = new GaussSeidelSolver(world,10000,1e-4);
		break;
		case 2: solver = new FourierSolver(world);
		break;
		case 3: solver = new ConjugateGradientSolver(world);
		break;
		case 4:
			cout << "MultiGrid V-Cycle size (1)V1 (2)V2 (3)V3 (4)V4 (5)V5: ";
			cin >> choice2;
			switch(choice2)
			{
				case 1: solver = new MultiGridSolver(world,10000,1e-4);
				break;
				case 2: solver = new MultiGridSolverV2(world,10000,1e-4);
				break;
				case 3: solver = new MultiGridSolverV3(world,10000,1e-4);
				break;
				case 4: solver = new MultiGridSolverV4(world,10000,1e-4);
				break;
				case 5: solver = new MultiGridSolverV5(world,10000,1e-4);
				break;
				default: 
					cout << "Invalid choice using V1 as default \n";
					solver = new MultiGridSolver(world,10000,1e-4);
				break;
			}
		default: 
			cout << "Invalid choice using Gauss-Seidel as default \n";
			new GaussSeidelSolver(world,10000,1e-4);
		break;
	}
			

	
    //GaussSeidelSolver solver(world,10000,1e-4);
    //MultiGridSolver solver(world,10000,1e-4);
    //MultiGridSolverV2 solver2(world,10000,1e-4);
    //FourierSolver solver(world);
    //ConjugateGradientSolver solver(world);
    //FourierSolver solver(world);
    solver->solve();
    //GaussSeidelSolver solver2(world,10000,1e-4);
    //MultiGridSolver solver2(world,10000,1e-4);
    //MultiGridSolverB solver2(world,10000,1e-4);
    //MultiGridSolverV2 solver2(world,10000,1e-4);
    //MultiGridSolverV3 solver2(world,10000,1e-4);
    //MultiGridSolverV4 solver2(world,10000,1e-4);
    //MultiGridSolverV5 solver2(world,10000,1e-4);
    //MultiGridSolverV2B solver2(world,10000,1e-4);
    //MultiGridSolverV3B solver2(world,10000,1e-4);
    //MultiGridSolverV4B solver2(world,10000,1e-4);
    //MultiGridSolverV5B solver2(world,10000,1e-4);
    //FourierSolver solver2(world);
    //FourierSolver solver(world);
    //solver2.solve();
    //solver2.solve();

    /*obtain initial electric field*/
    solver->computeEF();

    /* main loop*/
	while(world.advanceTime())
    {
		//grab starting time clock
		t_start = high_resolution_clock::now();
		
        /*move particles*/
		for (Species &sp:species)
		{
			try {
				sp.advance();
			}
			catch(...)
			{
				cout << "Something went wrong in sp.advance\n";
			}
			sp.computeNumberDensity();
		}
		//grab ending time
		t_end = high_resolution_clock::now();
		duration = t_end-t_start;

		/*compute charge density*/
		world.computeChargeDensity(species);
		

		//grab starting time clock
		t_start2 = high_resolution_clock::now();
		
		solver->solve();

		//grab ending time
		t_end2 = high_resolution_clock::now();
		duration2 = t_end2-t_start2;
		
		/*
		if (world.getTs()%100==0 && world.getTs() >0) {  // try other solver for comparison purposes
			t_start3 = high_resolution_clock::now();
			solver2.solve();
			//grab ending time
			t_end3 = high_resolution_clock::now();
			duration3 = t_end3-t_start3;
		}
		*/
		
		
        /*obtain electric field*/
        solver->computeEF();

		/*screen and file output*/
        Output::screenOutput(world,species);
        Output::diagOutput(world,species);

		/*periodically write out results*/
        if (world.getTs()%100==0 || world.isLastTimeStep()) {
			Output::fields(world, species);
			
			cout<<"Simulation took "<<duration.count()	<<" secs to advanceSpecies"<<endl;			
			cout<<"Simulation took "<<duration2.count()	<<" secs for solver1 to solve potential"<<endl;		
			//cout<<"Simulation took "<<duration3.count()	<<" secs for solver2 to solve potential"<<endl;		
			
			/*
			//grab starting time clock
			t_start3 = high_resolution_clock::now();
			
			//update potential
			solver2.solve();
			
			//grab ending time
			t_end3 = high_resolution_clock::now();
			duration3 = t_end3-t_start3;
			cout<<"Simulation took "<<duration3.count()	<<" secs to solve Fourier potential"<<endl;		
			*/
		}
    }
	
	/* grab starting time*/
	cout<<"Simulation took "<<world.getWallTime()<<" seconds";
	delete solver;
	solver = 0;
	return 0;		//indicate normal exit
}

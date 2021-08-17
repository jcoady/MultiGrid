# MultiGrid

This file contains 4 jupyter notebooks which contain 1D multigrid solvers for 4 different V-Cycle levels. The programs compare the MultiGrid solver
with the Gauss-Seidel solver to see how long they take to solve the problem. The variable ni = 128 is the default size of the 1D grid. 
You can try changing the value of this variable to something larger like 256 or 512 and compare the results.

The Cpp directory contains a program to test different 3D solvers including MultiGrid as well as Gauss-Seidel, FFT and Conjugate Gradient. 
These C++ files make use of the FFTW (Fastest Fourier Transform in the West) library from MIT.

    https://www.fftw.org/
    
As well as the alglib library.

   https://www.alglib.net/download.php
  
You will need to install these libraries if you want to compile the code in the Cpp directory.

The code can be compiled with the command.

    g++ -O3 *.cpp -o box -lfftw3 -Wall
    
and you will need to create a directory called "results" in this directory and then run the code with the command

   ./box 41
   
Which will create a 41x41x41 3D mesh on which the solver will use. You can set the second parameter to the mesh size you would like to use.
You will be prompted on which solver to use and you can try running the program with different solvers to compare them.

On my laptop computer I get the following execution time results running 1000 iterations of the program.

       Gauss-Seidel                639 seconds
       FFT                          41 seconds
       Conjugate Gradient          989 seconds
       MultiGrid 1 level v-cycle   141 seconds
       MultiGrid 2 level V-cycles  122 seconds
       MultiGrid 3 level V-cycles  140 seconds

       
       

#include "CCubicDomain.cuh"           //ds domain structure
#include <iostream>                   //ds cout
#include <cuda_runtime.h>             //ds needed for eclipse indexer
#include <device_launch_parameters.h> //ds needed for eclipse indexer

#define N 20


//ds GPU kernels
__global__ void updateAcceleration( float p_arrAccelerations[100][3] )
{
    p_arrAccelerations[0][0] += 1.0;
    p_arrAccelerations[0][1] += 1.0;
    p_arrAccelerations[0][2] += 1.0;
}

int main()
{
    //ds domain configuration
    const std::pair< double, double > pairBoundaries( -1.0, 1.0 );
    const unsigned int uNumberOfParticles( 100 );

    //ds host side information (float used since CUDA does not like double)
    float h_arrPositions[100][3];     //= new float*[uNumberOfParticles];
    float h_arrVelocities[100][3];    //= new float*[uNumberOfParticles];
    float h_arrAccelerations[100][3]; // = new float*[uNumberOfParticles];

    //ds allocate and set values correctly
    for( unsigned int u = 0; u < 100; ++u )
    {
        //ds initialize positions
        h_arrPositions[u][0] = 0.0;
        h_arrPositions[u][1] = 0.0;
        h_arrPositions[u][2] = 0.0;

        //ds initialize velocities
        h_arrPositions[u][0] = 0.0;
        h_arrVelocities[u][1] = 0.0;
        h_arrVelocities[u][2] = 0.0;

        //ds initialize
        h_arrAccelerations[u][0] = 0.0;
        h_arrAccelerations[u][1] = 0.0;
        h_arrAccelerations[u][2] = 0.0;
    }

    //ds accelerations on device
    float d_arrAccelerations[uNumberOfParticles][3];

    //ds allocate memory
    cudaMalloc( (void**)d_arrAccelerations, 100*3*sizeof( float ) );

    //ds copy memory to host
    cudaMemcpy( d_arrAccelerations, h_arrAccelerations, uNumberOfParticles*3*sizeof( float ), cudaMemcpyHostToDevice );

    //ds current simulation configuration
    //const double dTimeStepSize( 0.0001 );
    const unsigned int uNumberOfTimeSteps( 5000 );
    //const double dMinimumDistance( 0.05 );
    //const double dPotentialDepth( 0.01 );

    //ds start simulation
    for( unsigned int uCurrentTimeStep = 0; uCurrentTimeStep < uNumberOfTimeSteps; ++uCurrentTimeStep )
    {
        //ds call the kernel
        updateAcceleration<<< 1, uNumberOfParticles >>>( d_arrAccelerations );

        //ds copy the information back from the GPU to the CPU
        cudaMemcpy( h_arrAccelerations, d_arrAccelerations, uNumberOfParticles*3*sizeof( float ), cudaMemcpyDeviceToHost );

        //ds print
        std::cout << h_arrAccelerations[0][0] << std::endl;
        std::cout << h_arrAccelerations[0][1] << std::endl;
        std::cout << h_arrAccelerations[0][2] << std::endl;
    }



    /*ds host information
    float h_fTest[3];

    h_fTest[0] = 1;
    h_fTest[1] = 2;
    h_fTest[2] = 3;

    //ds device information
    float* d_fTest;

    //ds allocate memory
    cudaMalloc( (void**)&d_fTest, 3*sizeof( float ) );

    //ds copy memory to host
    cudaMemcpy( d_fTest, h_fTest, 3*sizeof( float ), cudaMemcpyHostToDevice );

    //ds call the kernel
    updatePosition<<< 1, 3 >>>( d_fTest );

    //ds copy the information back from the GPU to the CPU
    cudaMemcpy( h_fTest, d_fTest, 3*sizeof( float ), cudaMemcpyDeviceToHost );

    //ds free the memory
    cudaFree( d_fTest );

    std::cout << h_fTest[0] << std::endl;
    std::cout << h_fTest[1] << std::endl;
    std::cout << h_fTest[2] << std::endl;*/


    /*ds current positions
    int h_iPositionX;
    int h_iPositionY;
    int h_iPositionZ;

    //ds device positions
    int* d_iPositionX;
    int* d_iPositionY;
    int* d_iPositionZ;

    //ds allocate the memory on the host
    d_iPositionX = new int;
    d_iPositionY = new int;
    d_iPositionZ = new int;

    //ds allocate the memory on the GPU
    cudaMalloc( (void**)&d_iPositionX, N*sizeof( int ) );
    cudaMalloc( (void**)&d_iPositionY, N*sizeof( int ) );
    cudaMalloc( (void**)&d_iPositionZ, N*sizeof( int ) );

    //ds set the memory layout
    cudaMemset( d_iPositionX, 0, N );
    cudaMemset( d_iPositionY, 0, N );
    cudaMemset( d_iPositionZ, 0, N );

    //ds call the GPU kernels and compute
    kernel1<<< 1, N >>>( d_iPositionX );
    kernel2<<< 1, N >>>( d_iPositionY );
    kernel3<<< 1, N >>>( d_iPositionZ );

    //ds copy the information back from the GPU to the CPU
    cudaMemcpy( &h_iPositionX, d_iPositionX, N*sizeof(int), cudaMemcpyDeviceToHost );
    cudaMemcpy( &h_iPositionY, d_iPositionY, N*sizeof(int), cudaMemcpyDeviceToHost );
    cudaMemcpy( &h_iPositionZ, d_iPositionZ, N*sizeof(int), cudaMemcpyDeviceToHost );

    // display the results
    std::cout << " Results from kernel1:" << std::endl;
    for (int i = 0; i<N; i++)
        std::cout<< h_iPositionX[i] << " ";
    std::cout<< std::endl;

    std::cout << " Results from kernel2:" << std::endl;
    for (int i = 0; i<N; i++)
        std::cout<< h_iPositionY[i] << " ";
    std::cout<< std::endl;

    std::cout << " Results from kernel3:" << std::endl;
    for (int i = 0; i<N; i++)
        std::cout<< h_iPositionZ[i] << " ";
    std::cout << std::endl;


    //free the memory allocated on the GPU
    cudaFree(d_pa);
    cudaFree(d_pb);
    cudaFree(d_pc);*/

    return 0;

}

/*
 *     //ds domain configuration
    const std::pair< double, double > pairBoundaries( -1.0, 1.0 );
    const unsigned int uNumberOfParticles( 100 );

    //ds allocate a domain to work with specifying number of particles and timing
    NBody::CCubicDomain cDomain( pairBoundaries, uNumberOfParticles );

    //ds target kinetic energy
    const double dTargetKineticEnergy( 1000.0 );

    //ds create particles uniformly from a normal distribution
    cDomain.createParticlesUniformFromNormalDistribution( dTargetKineticEnergy );

    //ds current simulation configuration
    const double dTimeStepSize( 0.0001 );
    const unsigned int uNumberOfTimeSteps( 5000 );
    const double dMinimumDistance( 0.05 );
    const double dPotentialDepth( 0.01 );

    //ds start simulation
    for( unsigned int uCurrentTimeStep = 0; uCurrentTimeStep < uNumberOfTimeSteps; ++uCurrentTimeStep )
    {
        //ds update particles
        cDomain.updateParticlesVelocityVerlet( dTimeStepSize, dMinimumDistance, dPotentialDepth );

        //ds record situation (we will write the stream to the file in one operation)
        cDomain.saveParticlesToStream( );
        cDomain.saveIntegralsToStream( dMinimumDistance, dPotentialDepth );

        //ds dump progress information and first integrals
        std::cout << "------------------------------------------------------------" << std::endl;
        std::cout << "step: " << uCurrentTimeStep << std::endl;
        std::cout << "total energy: " << std::endl;
        std::cout << cDomain.getTotalEnergy( dMinimumDistance, dPotentialDepth ) << std::endl;
        std::cout << "center of mass: " << std::endl;
        std::cout << cDomain.getCenterOfMass( ) << std::endl;
        std::cout << "angular momentum: " << std::endl;
        std::cout << cDomain.getAngularMomentum( ) << std::endl;
        std::cout << "linear momentum: " << std::endl;
        std::cout << cDomain.getLinearMomentum( ) << std::endl;
    }

    //ds save the streams to a file
    cDomain.writeParticlesToFile( "bin/simulation.txt", uNumberOfTimeSteps );
    cDomain.writeIntegralsToFile( "bin/integrals.txt", uNumberOfTimeSteps, dTimeStepSize );

    std::cout << "------------------------------------------------------------" << std::endl;

    return 0;
 */

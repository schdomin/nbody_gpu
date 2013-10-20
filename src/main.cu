#include "CCubicDomain.cuh"           //ds domain structure
#include <iostream>                   //ds cout
#include <time.h>                     //ds timing
#include <cuda_runtime.h>             //ds needed for eclipse indexer
#include <device_launch_parameters.h> //ds needed for eclipse indexer



//ds kernels
__global__ void updateParticlesVelocityVerlet( float* arrPositions, float* arrVelocities, float* arrAccelerations )
{
    //ds particle index equals thread index
    unsigned int uIndex = threadIdx.x;

    //ds get the current structures and save them in vector elements (only for readability)
    float* vecPosition     = &arrPositions[3*uIndex];
    float* vecVelocity     = &arrVelocities[3*uIndex];
    float* vecAcceleration = &arrAccelerations[3*uIndex];

    //ds do calculations
    vecPosition[0] += 1;
    vecPosition[1] += 2;
    vecPosition[2] += 3;

    vecVelocity[0] += 1;
    vecVelocity[1] += 1;
    vecVelocity[2] += 1;

    vecAcceleration[0] += 1;
    vecAcceleration[1] += 1;
    vecAcceleration[2] += 1;
}

int main( int argc, char** argv )
{
    //ds domain configuration
    const std::pair< double, double > pairBoundaries( -1.0, 1.0 );
    const unsigned int uNumberOfParticles( 100 );

    //ds host information
    float h_arrPositions[uNumberOfParticles][3];
    float h_arrVelocities[uNumberOfParticles][3];
    float h_arrAccelerations[uNumberOfParticles][3];

    //ds set values
    for( unsigned int u = 0; u < uNumberOfParticles; ++u )
    {
        h_arrPositions[u][0] = 0;
        h_arrPositions[u][1] = 0;
        h_arrPositions[u][2] = 0;

        h_arrVelocities[u][0] = 0;
        h_arrVelocities[u][1] = 0;
        h_arrVelocities[u][2] = 0;

        h_arrAccelerations[u][0] = 0;
        h_arrAccelerations[u][1] = 0;
        h_arrAccelerations[u][2] = 0;
    }

    //ds device handles
    float *d_arrPositions     ( 0 );
    float *d_arrVelocities    ( 0 );
    float *d_arrAccelerations ( 0 );

    //ds allocate memory (here we see the advantage of using single pointers instead doubles)
    cudaMalloc( (void **)&d_arrPositions    , uNumberOfParticles * 3 * sizeof( float ) );
    cudaMalloc( (void **)&d_arrVelocities   , uNumberOfParticles * 3 * sizeof( float ) );
    cudaMalloc( (void **)&d_arrAccelerations, uNumberOfParticles * 3 * sizeof( float ) ) ;

    //ds copy memory to gpu
    cudaMemcpy( d_arrPositions    , h_arrPositions    , uNumberOfParticles * 3 * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_arrVelocities   , h_arrVelocities   , uNumberOfParticles * 3 * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_arrAccelerations, h_arrAccelerations, uNumberOfParticles * 3 * sizeof( float ), cudaMemcpyHostToDevice );

    std::cout << "position[0][0]: " << h_arrPositions[0][0] << std::endl;
    std::cout << "position[0][1]: " << h_arrPositions[0][1] << std::endl;
    std::cout << "position[0][2]: " << h_arrPositions[0][2] << std::endl;

    //ds current simulation configuration
    //const double dTimeStepSize( 0.0001 );
    const unsigned int uNumberOfTimeSteps( 5000 );
    //const double dMinimumDistance( 0.05 );
    //const double dPotentialDepth( 0.01 );

    //ds start simulation
    for( unsigned int uCurrentTimeStep = 0; uCurrentTimeStep < uNumberOfTimeSteps; ++uCurrentTimeStep )
    {
        //ds execute timestep -> 1 block: one thread for each particle
        updateParticlesVelocityVerlet<<< 1, uNumberOfParticles >>>( d_arrPositions, d_arrVelocities, d_arrAccelerations );

        //ds get the memory from gpu to cpu
        cudaMemcpy( h_arrPositions    , d_arrPositions    , uNumberOfParticles * 3 * sizeof( float ), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_arrVelocities   , d_arrVelocities   , uNumberOfParticles * 3 * sizeof( float ), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_arrAccelerations, d_arrAccelerations, uNumberOfParticles * 3 * sizeof( float ), cudaMemcpyDeviceToHost );

        std::cout << "position[0][0]: " << h_arrPositions[0][0] << std::endl;
        std::cout << "position[0][1]: " << h_arrPositions[0][1] << std::endl;
        std::cout << "position[0][2]: " << h_arrPositions[0][2] << std::endl;
        std::cout << "position[1][0]: " << h_arrPositions[1][0] << std::endl;
        std::cout << "position[1][1]: " << h_arrPositions[1][1] << std::endl;
        std::cout << "position[1][2]: " << h_arrPositions[1][2] << std::endl;
        std::cout << "position[2][0]: " << h_arrPositions[2][0] << std::endl;
        std::cout << "position[2][1]: " << h_arrPositions[2][1] << std::endl;
        std::cout << "position[2][2]: " << h_arrPositions[2][2] << std::endl;
    }

    cudaFree( d_arrPositions );
    cudaFree( d_arrVelocities );
    cudaFree( d_arrAccelerations );

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

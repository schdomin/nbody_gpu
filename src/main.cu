#include "CCubicDomain.cuh"           //ds domain structure
#include <iostream>                   //ds cout
#include <time.h>                     //ds timing
#include <cuda.h>                     //ds needed for eclipse indexer only (not for compilation)
#include <cuda_runtime.h>             //ds needed for eclipse indexer only (not for compilation)
#include <device_launch_parameters.h> //ds needed for eclipse indexer only (not for compilation)



//ds kernels
__global__ void updateParticlesVelocityVerlet( float* p_arrPositions,
                                               float* p_arrVelocities,
                                               float* p_arrAccelerations,
                                               float* p_arrMasses,
                                               const float p_fLowerBoundary,
                                               const float p_fUpperBoundary,
                                               const float p_fTimeStep,
                                               const float p_fMinimumDistance,
                                               const float p_fPotentialDepth )
{
    //ds dynamic shared memory
    extern __shared__ float s_arrPositions[];
    extern __shared__ float s_arrVelocities[];
    extern __shared__ float s_arrAccelerations[];
    extern __shared__ float s_arrMasses[];

    //ds particle index equals thread index
    unsigned int uIndex = threadIdx.x;

    //ds fill the shared memory
    s_arrPositions[uIndex]     = p_arrPositions[uIndex];
    s_arrVelocities[uIndex]    = p_arrVelocities[uIndex];
    s_arrAccelerations[uIndex] = p_arrAccelerations[uIndex];
    s_arrMasses[uIndex]        = p_arrMasses[uIndex];

    //ds wait until all threads are done and the shared memory is set
    __syncthreads( );

    //ds get our particle
    float* vecPosition     = &s_arrPositions[3*uIndex];
    float* vecVelocity     = &s_arrVelocities[3*uIndex];
    float* vecAcceleration = &s_arrAccelerations[3*uIndex];
    float fMass            = s_arrMasses[uIndex];

    //ds force instance to accumulate
    float vecForce[3];

    //ds get the domain size
    const float fDomainSize = abs( p_fLowerBoundary ) + abs( p_fUpperBoundary );

    //ds loop over all other particles
    for( unsigned int u = 0; u < 100; ++u )
    {
        //ds get the position from the other particle
        float* vecPositionOther = &s_arrPositions[3*u];

        //ds cutoff distance
        const float fDistanceCutoff = 2.5*p_fMinimumDistance;

        //ds we have to loop over the cubic boundary conditions
        for( float dX = p_fLowerBoundary; dX < p_fUpperBoundary+1; ++dX )
        {
            for( float dY = p_fLowerBoundary; dY < p_fUpperBoundary+1; ++dY )
            {
                for( float dZ = p_fLowerBoundary; dZ < p_fUpperBoundary+1; ++dZ )
                {
                    //ds get the radial vector between the particles
                    float vecRadius[3];

                    //ds calculate the distance
                    vecRadius[0] = dX*fDomainSize + vecPositionOther[0] - vecPosition[0];
                    vecRadius[1] = dY*fDomainSize + vecPositionOther[1] - vecPosition[1];
                    vecRadius[2] = dZ*fDomainSize + vecPositionOther[2] - vecPosition[2];

                    //ds get the absolute distance
                    const float fDistanceAbsolute = sqrt( pow( vecRadius[0], 2 ) + pow( vecRadius[1], 2 ) + pow( vecRadius[2], 2 ) );

                    //ds if we are within the cutoff range (only smaller here to avoid double overhead for >=)
                    if( fDistanceCutoff > fDistanceAbsolute )
                    {
                        //ds calculate the lennard jones force prefix
                        const float fLJPrefix = -24*p_fPotentialDepth*( 2*pow( p_fMinimumDistance/fDistanceAbsolute, 12 ) - pow( p_fMinimumDistance/fDistanceAbsolute, 6  ) )
                                                                     *1/pow( fDistanceAbsolute, 2 );

                        //ds add the information to the force including the radial component
                        vecForce[0] += fLJPrefix*vecRadius[0];
                        vecForce[1] += fLJPrefix*vecRadius[1];
                        vecForce[2] += fLJPrefix*vecRadius[2];
                    }
                }
            }
        }
    }

    //ds new acceleration
    float vecNewAcceleration[3];

    //ds calculate it from the force
    vecNewAcceleration[0] = vecForce[0]/fMass;
    vecNewAcceleration[1] = vecForce[1]/fMass;
    vecNewAcceleration[2] = vecForce[2]/fMass;

    //ds wait until all threads are done so we can proceed setting the new position and velocities without interfering the calculations
    __syncthreads( );

    //ds velocity-verlet for position
    vecPosition[0] = vecPosition[0] + p_fTimeStep*vecVelocity[0] + 1/2*pow( p_fTimeStep, 2 )*vecAcceleration[0];
    vecPosition[1] = vecPosition[1] + p_fTimeStep*vecVelocity[1] + 1/2*pow( p_fTimeStep, 2 )*vecAcceleration[1];
    vecPosition[2] = vecPosition[2] + p_fTimeStep*vecVelocity[2] + 1/2*pow( p_fTimeStep, 2 )*vecAcceleration[2];

    //ds produce periodic boundary shifting - check each element: x,y,z
    for( unsigned int v = 0; v < 3; ++v )
    {
        //ds check if we are below the boundary
        while( p_fLowerBoundary > vecPosition[v] )
        {
            //ds map the particle to the other boundary by shifting it up to the boundary
            vecPosition[v] += fDomainSize;
        }

        //ds check if we are above the boundary
        while( p_fUpperBoundary < vecPosition[v] )
        {
            //ds map the particle to the other boundary by shifting it back to the boundary
            vecPosition[v] -= fDomainSize;
        }
    }

    //ds velocity-verlet for velocity
    vecVelocity[0] = vecVelocity[0] + p_fTimeStep/2*( vecNewAcceleration[0] + vecAcceleration[0] );
    vecVelocity[1] = vecVelocity[1] + p_fTimeStep/2*( vecNewAcceleration[1] + vecAcceleration[1] );
    vecVelocity[2] = vecVelocity[2] + p_fTimeStep/2*( vecNewAcceleration[2] + vecAcceleration[2] );

    //ds update the old accelerations
    vecAcceleration[0] = vecNewAcceleration[0];
    vecAcceleration[1] = vecNewAcceleration[1];
    vecAcceleration[2] = vecNewAcceleration[2];

    //ds make sure every thread is done
    //__syncthreads( );


    //ds update the original arrays from the shared memory
    //p_arrPositions[uIndex]     = s_arrPositions[uIndex];
    //p_arrVelocities[uIndex]    = s_arrVelocities[uIndex];
    //p_arrAccelerations[uIndex] = s_arrAccelerations[uIndex];
    //p_arrMasses[uIndex]        = s_arrMasses[uIndex];
}

int main( int argc, char** argv )
{
    //ds domain configuration
    const std::pair< float, float > pairBoundaries( -1, 1 );
    const unsigned int uNumberOfParticles( 100 );

    //ds allocate a domain to work with specifying number of particles and timing
    NBody::CCubicDomain cDomain( pairBoundaries, uNumberOfParticles );

    //ds target kinetic energy
    const double dTargetKineticEnergy( 1000.0 );

    //ds create particles uniformly from a normal distribution - no CUDA call here
    cDomain.createParticlesUniformFromNormalDistribution( dTargetKineticEnergy );

    //ds host information
    float** h_arrPositions     = cDomain.getPositions( );
    float** h_arrVelocities    = cDomain.getVelocities( );
    float** h_arrAccelerations = cDomain.getAccelerations( );
    float* h_arrMasses         = cDomain.getMasses( );

    //ds device handles
    float* d_arrPositions     ( 0 );
    float* d_arrVelocities    ( 0 );
    float* d_arrAccelerations ( 0 );
    float* d_arrMasses        ( 0 );

    //ds allocate memory (here we see the advantage of using single pointers instead doubles)
    cudaMalloc( (void **)&d_arrPositions    , uNumberOfParticles * 3 * sizeof( float ) );
    cudaMalloc( (void **)&d_arrVelocities   , uNumberOfParticles * 3 * sizeof( float ) );
    cudaMalloc( (void **)&d_arrAccelerations, uNumberOfParticles * 3 * sizeof( float ) ) ;
    cudaMalloc( (void **)&d_arrMasses       , uNumberOfParticles * sizeof( float ) ) ;

    //ds copy memory to gpu
    cudaMemcpy( d_arrPositions    , h_arrPositions    , uNumberOfParticles * 3 * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_arrVelocities   , h_arrVelocities   , uNumberOfParticles * 3 * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_arrAccelerations, h_arrAccelerations, uNumberOfParticles * 3 * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_arrMasses       , h_arrMasses       , uNumberOfParticles * sizeof( float ), cudaMemcpyHostToDevice );

    //ds current simulation configuration
    const float fTimeStepSize( 0.0001 );
    const unsigned int uNumberOfTimeSteps( 5000 );
    const float fMinimumDistance( 0.05 );
    const float fPotentialDepth( 0.01 );

    //ds start simulation
    for( unsigned int uCurrentTimeStep = 0; uCurrentTimeStep < uNumberOfTimeSteps; ++uCurrentTimeStep )
    {
        std::cout << "BEFORE" << std::endl;
        std::cout << "h_arrPositions[0][0]: " << h_arrPositions[0][0] << std::endl;
        std::cout << "h_arrPositions[0][1]: " << h_arrPositions[0][1] << std::endl;
        std::cout << "h_arrPositions[0][2]: " << h_arrPositions[0][2] << std::endl;

        //ds execute timestep -> 1 block: one thread for each particle
        updateParticlesVelocityVerlet<<< 1, uNumberOfParticles, uNumberOfParticles * 3 * sizeof( float ) >>>( d_arrPositions,
                                                                                                              d_arrVelocities,
                                                                                                              d_arrAccelerations,
                                                                                                              d_arrMasses,
                                                                                                              pairBoundaries.first,
                                                                                                              pairBoundaries.second,
                                                                                                              fTimeStepSize,
                                                                                                              fMinimumDistance,
                                                                                                              fPotentialDepth );

        //ds get the memory from gpu to cpu
        cudaMemcpy( h_arrPositions    , d_arrPositions    , uNumberOfParticles * 3 * sizeof( float ), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_arrVelocities   , d_arrVelocities   , uNumberOfParticles * 3 * sizeof( float ), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_arrAccelerations, d_arrAccelerations, uNumberOfParticles * 3 * sizeof( float ), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_arrMasses       , d_arrMasses       , uNumberOfParticles * sizeof( float ), cudaMemcpyDeviceToHost );

        std::cout << "h_arrPositions[0][0]: " << h_arrPositions[0][0] << std::endl;
        std::cout << "h_arrPositions[0][1]: " << h_arrPositions[0][1] << std::endl;
        std::cout << "h_arrPositions[0][2]: " << h_arrPositions[0][2] << std::endl;

        /*ds update domain
        cDomain.setPositions( h_arrPositions );
        cDomain.setVelocities( h_arrVelocities );
        cDomain.setAccelerations( h_arrAccelerations );
        cDomain.setMasses( h_arrMasses );*/

        //cDomain.saveParticlesToStream( );
        //cDomain.saveIntegralsToStream( dMinimumDistance, dPotentialDepth );
    }

    cudaFree( d_arrPositions );
    cudaFree( d_arrVelocities );
    cudaFree( d_arrAccelerations );
    cudaFree( d_arrMasses );

    //ds save the streams to a file
    cDomain.writeParticlesToFile( "bin/simulation.txt", uNumberOfTimeSteps );
    //cDomain.writeIntegralsToFile( "bin/integrals.txt", uNumberOfTimeSteps, fTimeStepSize );

    return 0;
}

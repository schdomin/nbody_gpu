#include "CCubicDomain.cuh"           //ds domain structure
#include "Timer.h"                    //ds time measurement
#include <iostream>                   //ds cout
#include <cuda.h>                     //ds needed for eclipse indexer only (not for compilation)
#include <cuda_runtime.h>             //ds needed for eclipse indexer only (not for compilation)
#include <device_launch_parameters.h> //ds needed for eclipse indexer only (not for compilation)



//ds ugly but necessary since dynamic shared arrays seem to mess up things
#define NumberOfParticles 100



//ds kernels - only total energy emulated for the integrals, since the other kernels are too simple
__global__ void updateParticlesVelocityVerletShared( const unsigned int p_uNumberOfParticles,
                                                     float* p_arrPositions,
                                                     float* p_arrVelocities,
                                                     float* p_arrAccelerations,
                                                     float* p_arrMasses,
                                                     const float p_fLowerBoundary,
                                                     const float p_fUpperBoundary,
                                                     const float p_fTimeStepSize,
                                                     const float p_fMinimumDistance,
                                                     const float p_fPotentialDepth );

__global__ void updateParticlesVelocityVerlet( const unsigned int p_uNumberOfParticles,
                                               float* p_arrPositions,
                                               float* p_arrVelocities,
                                               float* p_arrAccelerations,
                                               float* p_arrMasses,
                                               const float p_fLowerBoundary,
                                               const float p_fUpperBoundary,
                                               const float p_fTimeStepSize,
                                               const float p_fMinimumDistance,
                                               const float p_fPotentialDepth );

__global__ void getTotalEnergy( const unsigned int p_uNumberOfParticles,
                                float* p_arrPositions,
                                float* p_arrVelocities,
                                float* p_arrMasses,
                                const float p_fMinimumDistance,
                                const float p_fPotentialDepth,
                                float* p_fTotalEnergy );

int main( int argc, char** argv )
{
    //ds start timing
    Timer tmTimer; tmTimer.start( );

    //ds domain configuration
    const std::pair< float, float > pairBoundaries( -1, 1 );
    const unsigned int uNumberOfParticles( 100 );

    //ds allocate a domain to work with specifying number of particles and timing
    NBody::CCubicDomain cDomain( uNumberOfParticles );

    //ds target kinetic energy
    const double dTargetKineticEnergy( 1000.0 );

    //ds create particles uniformly from a normal distribution - no CUDA call here
    cDomain.createParticlesUniformFromNormalDistribution( dTargetKineticEnergy );

    //ds host information: particles
    float* h_arrPositions    ( cDomain.getPositions( ) );
    float* h_arrVelocities   ( cDomain.getVelocities( ) );
    float* h_arrAccelerations( cDomain.getAccelerations( ) );
    float* h_arrMasses       ( cDomain.getMasses( ) );

    //ds host information: integrals
    float h_fTotalEnergy( 0.0 );

    //ds device handles: particles
    float* d_arrPositions     ( 0 ); //Nx3
    float* d_arrVelocities    ( 0 ); //Nx3
    float* d_arrAccelerations ( 0 ); //Nx3
    float* d_arrMasses        ( 0 ); //Nx3

    //ds device handles: integrals
    float* d_fTotalEnergy           ( 0 ); //1x1

    //ds allocate memory: particles (here we see the advantage of using single pointers instead doubles)
    cudaMalloc( (void **)&d_arrPositions    , uNumberOfParticles*3*sizeof( float ) );
    cudaMalloc( (void **)&d_arrVelocities   , uNumberOfParticles*3*sizeof( float ) );
    cudaMalloc( (void **)&d_arrAccelerations, uNumberOfParticles*3*sizeof( float ) ) ;
    cudaMalloc( (void **)&d_arrMasses       , uNumberOfParticles*sizeof( float ) ) ;

    //ds allocate memory: integrals
    cudaMalloc( (void **)&d_fTotalEnergy           , sizeof( float ) ) ;

    //ds copy memory to gpu to initialize the situation
    cudaMemcpy( d_arrPositions    , h_arrPositions    , uNumberOfParticles*3*sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_arrVelocities   , h_arrVelocities   , uNumberOfParticles*3*sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_arrAccelerations, h_arrAccelerations, uNumberOfParticles*3*sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_arrMasses       , h_arrMasses       , uNumberOfParticles*sizeof( float )  , cudaMemcpyHostToDevice );

    //ds current simulation configuration
    const float fTimeStepSize            ( 0.0001 );
    const unsigned int uNumberOfTimeSteps( 5000 );
    const float fMinimumDistance         ( 0.05 );
    const float fPotentialDepth          ( 0.01 );

    //ds start simulation
    for( unsigned int uCurrentTimeStep = 0; uCurrentTimeStep < uNumberOfTimeSteps; ++uCurrentTimeStep )
    {
        //ds execute timestep -> 1 block: one thread for each particle
        updateParticlesVelocityVerletShared<<< 1, uNumberOfParticles >>>( uNumberOfParticles,
                                                                          d_arrPositions,
                                                                          d_arrVelocities,
                                                                          d_arrAccelerations,
                                                                          d_arrMasses,
                                                                          pairBoundaries.first,
                                                                          pairBoundaries.second,
                                                                          fTimeStepSize,
                                                                          fMinimumDistance,
                                                                          fPotentialDepth );

        //ds compute total energy
        getTotalEnergy<<< 1, uNumberOfParticles >>>( uNumberOfParticles,
                                                     d_arrPositions,
                                                     d_arrVelocities,
                                                     d_arrMasses,
                                                     fMinimumDistance,
                                                     fPotentialDepth,
                                                     d_fTotalEnergy );

        //ds get the particle information from gpu to cpu
        cudaMemcpy( h_arrPositions    , d_arrPositions    , uNumberOfParticles*3*sizeof( float ), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_arrVelocities   , d_arrVelocities   , uNumberOfParticles*3*sizeof( float ), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_arrAccelerations, d_arrAccelerations, uNumberOfParticles*3*sizeof( float ), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_arrMasses       , d_arrMasses       , uNumberOfParticles*sizeof( float )  , cudaMemcpyDeviceToHost );

        //ds get the integrals information from gpu to cpu
        cudaMemcpy( &h_fTotalEnergy, d_fTotalEnergy, sizeof( float ), cudaMemcpyDeviceToHost );

        //ds save particle and integral information
        cDomain.saveParticlesToStream( );
        cDomain.saveIntegralsToStream( h_fTotalEnergy );

        std::cout << "total energy: " << h_fTotalEnergy << std::endl;
        getchar( );
    }

    cudaFree( d_arrPositions );
    cudaFree( d_arrVelocities );
    cudaFree( d_arrAccelerations );
    cudaFree( d_arrMasses );

    //ds save the streams to a file
    cDomain.writeParticlesToFile( "bin/simulation.txt", uNumberOfTimeSteps );
    cDomain.writeIntegralsToFile( "bin/integrals.txt", uNumberOfTimeSteps, fTimeStepSize );

    //ds stop timing
    const double dDurationSeconds( tmTimer.stop( ) );

    std::cout << "-------GPU SETUP------------------------------------------------------------" << std::endl;
    std::cout << "  Number of particles: " << uNumberOfParticles << std::endl;
    std::cout << "Target kinetic energy: " << dTargetKineticEnergy << std::endl;
    std::cout << "  Number of timesteps: " << uNumberOfTimeSteps << std::endl;
    std::cout << "     Computation time: " << dDurationSeconds << std::endl;
    std::cout << "----------------------------------------------------------------------------" << std::endl;

    return 0;
}

__global__ void updateParticlesVelocityVerletShared( const unsigned int p_uNumberOfParticles,
                                                     float* p_arrPositions,
                                                     float* p_arrVelocities,
                                                     float* p_arrAccelerations,
                                                     float* p_arrMasses,
                                                     const float p_fLowerBoundary,
                                                     const float p_fUpperBoundary,
                                                     const float p_fTimeStepSize,
                                                     const float p_fMinimumDistance,
                                                     const float p_fPotentialDepth )
{
    //ds dynamic shared memory - also linear data model - the +0 index is kept for readability consistency
    __shared__ float s_arrPositions[NumberOfParticles*3];
    __shared__ float s_arrVelocities[NumberOfParticles*3];
    __shared__ float s_arrAccelerations[NumberOfParticles*3];

    //ds regular index and "real" particle index equals three times thread index, since were working with a linear 2d array
    const unsigned int uIndex    ( threadIdx.x );
    const unsigned int uIndexReal( 3*uIndex );

    //ds for each array fill the shared memory counterparts s = p
    for( unsigned int u = 0; u < 3; ++u )
    {
        s_arrPositions[uIndexReal+u]     = p_arrPositions[uIndexReal+u];
        s_arrVelocities[uIndexReal+u]    = p_arrVelocities[uIndexReal+u];
        s_arrAccelerations[uIndexReal+u] = p_arrAccelerations[uIndexReal+u];
    }

    //ds get current mass
    const float fCurrentMass( p_arrMasses[uIndex] );

    //ds wait until all threads are done and the shared memory is set - below this point we only work with s_arrays for maximum speed
    __syncthreads( );

    //ds force instance to calculate for the current particle
    float vecTotalForce[3];

    //ds make sure it is zero
    vecTotalForce[0] = 0.0;
    vecTotalForce[1] = 0.0;
    vecTotalForce[2] = 0.0;

    //ds get the domain size
    const float fDomainSize( abs( p_fLowerBoundary ) + abs( p_fUpperBoundary ) );

    //ds loop over all other particles
    for( unsigned int u = 0; u < p_uNumberOfParticles; ++u )
    {
        //ds do not treat itself (else nan results because division by zero)
        if( u != uIndex )
        {
            //ds cutoff distance
            const float fDistanceCutoff = 2.5*p_fMinimumDistance;

            //ds we have to loop over the cubic boundary conditions
            for( float dX = p_fLowerBoundary; dX < p_fUpperBoundary+1.0; ++dX )
            {
                for( float dY = p_fLowerBoundary; dY < p_fUpperBoundary+1.0; ++dY )
                {
                    for( float dZ = p_fLowerBoundary; dZ < p_fUpperBoundary+1.0; ++dZ )
                    {
                        //ds get the radial vector between the particles
                        float vecRadius[3];

                        //ds calculate the distance: domain + particle2 - particle1
                        vecRadius[0] = dX*fDomainSize + s_arrPositions[3*u+0] - s_arrPositions[uIndexReal+0];
                        vecRadius[1] = dY*fDomainSize + s_arrPositions[3*u+1] - s_arrPositions[uIndexReal+1];
                        vecRadius[2] = dZ*fDomainSize + s_arrPositions[3*u+2] - s_arrPositions[uIndexReal+2];

                        //ds get the absolute distance
                        const float fDistanceAbsolute( sqrt( pow( vecRadius[0], 2 ) + pow( vecRadius[1], 2 ) + pow( vecRadius[2], 2 ) ) );

                        //ds if we are within the cutoff range (only smaller here to avoid double overhead for >=)
                        if( fDistanceCutoff > fDistanceAbsolute )
                        {
                            //ds calculate the lennard jones force prefix
                            const float fLJFPrefix( -24*p_fPotentialDepth*( 2*pow( p_fMinimumDistance/fDistanceAbsolute, 12 ) - pow( p_fMinimumDistance/fDistanceAbsolute, 6  ) )
                                                                         *1/pow( fDistanceAbsolute, 2 ) );

                            //ds add the information to the force including the radial component
                            vecTotalForce[0] += fLJFPrefix*vecRadius[0];
                            vecTotalForce[1] += fLJFPrefix*vecRadius[1];
                            vecTotalForce[2] += fLJFPrefix*vecRadius[2];
                        }
                    }
                }
            }
        }
    }

    //ds new acceleration
    float vecNewAcceleration[3];

    //ds calculate it from the force
    vecNewAcceleration[0] = vecTotalForce[0]/fCurrentMass;
    vecNewAcceleration[1] = vecTotalForce[1]/fCurrentMass;
    vecNewAcceleration[2] = vecTotalForce[2]/fCurrentMass;

    //ds wait until all threads are done so we can proceed setting the new position and velocities without interfering the calculations
    __syncthreads( );

    //ds velocity-verlet for position
    s_arrPositions[uIndexReal+0] = s_arrPositions[uIndexReal+0] + p_fTimeStepSize*s_arrVelocities[uIndexReal+0] + ( 1/2 )*pow( p_fTimeStepSize, 2 )*s_arrAccelerations[uIndexReal+0];
    s_arrPositions[uIndexReal+1] = s_arrPositions[uIndexReal+1] + p_fTimeStepSize*s_arrVelocities[uIndexReal+1] + ( 1/2 )*pow( p_fTimeStepSize, 2 )*s_arrAccelerations[uIndexReal+1];
    s_arrPositions[uIndexReal+2] = s_arrPositions[uIndexReal+2] + p_fTimeStepSize*s_arrVelocities[uIndexReal+2] + ( 1/2 )*pow( p_fTimeStepSize, 2 )*s_arrAccelerations[uIndexReal+2];

    //ds produce periodic boundary shifting - check each element: x,y,z
    for( unsigned int v = 0; v < 3; ++v )
    {
        //ds check if we are below the boundary
        while( p_fLowerBoundary > s_arrPositions[uIndexReal+v] )
        {
            //ds map the particle to the other boundary by shifting it up to the boundary
            s_arrPositions[uIndexReal+v] += fDomainSize;
        }

        //ds check if we are above the boundary
        while( p_fUpperBoundary < s_arrPositions[uIndexReal+v] )
        {
            //ds map the particle to the other boundary by shifting it back to the boundary
            s_arrPositions[uIndexReal+v] -= fDomainSize;
        }
    }

    //ds velocity-verlet for velocity
    s_arrVelocities[uIndexReal+0] = s_arrVelocities[uIndexReal+0] + ( p_fTimeStepSize/2 )*( vecNewAcceleration[0] + s_arrAccelerations[uIndexReal+0] );
    s_arrVelocities[uIndexReal+1] = s_arrVelocities[uIndexReal+1] + ( p_fTimeStepSize/2 )*( vecNewAcceleration[1] + s_arrAccelerations[uIndexReal+1] );
    s_arrVelocities[uIndexReal+2] = s_arrVelocities[uIndexReal+2] + ( p_fTimeStepSize/2 )*( vecNewAcceleration[2] + s_arrAccelerations[uIndexReal+2] );

    //ds update the old accelerations
    s_arrAccelerations[uIndexReal+0] = vecNewAcceleration[0];
    s_arrAccelerations[uIndexReal+1] = vecNewAcceleration[1];
    s_arrAccelerations[uIndexReal+2] = vecNewAcceleration[2];

    //ds make sure every thread is done
    __syncthreads( );

    //ds update the original arrays from the shared memory p = s
    for( unsigned int u = 0; u < 3; ++u )
    {
        p_arrPositions[uIndexReal+u]     = s_arrPositions[uIndexReal+u];
        p_arrVelocities[uIndexReal+u]    = s_arrVelocities[uIndexReal+u];
        p_arrAccelerations[uIndexReal+u] = s_arrAccelerations[uIndexReal+u];
    }
}

__global__ void updateParticlesVelocityVerlet( const unsigned int p_uNumberOfParticles,
                                               float* p_arrPositions,
                                               float* p_arrVelocities,
                                               float* p_arrAccelerations,
                                               float* p_arrMasses,
                                               const float p_fLowerBoundary,
                                               const float p_fUpperBoundary,
                                               const float p_fTimeStepSize,
                                               const float p_fMinimumDistance,
                                               const float p_fPotentialDepth )
{
    //ds regular index and "real" particle index equals three times thread index, since were working with a linear 2d array
    const unsigned int uIndex    ( threadIdx.x );
    const unsigned int uIndexReal( 3*uIndex );

    //ds get current mass
    const float fCurrentMass( p_arrMasses[uIndex] );

    //ds force instance to calculate for the current particle
    float vecTotalForce[3];

    //ds make sure it is zero
    vecTotalForce[0] = 0.0;
    vecTotalForce[1] = 0.0;
    vecTotalForce[2] = 0.0;

    //ds get the domain size
    const float fDomainSize( abs( p_fLowerBoundary ) + abs( p_fUpperBoundary ) );

    //ds loop over all other particles
    for( unsigned int u = 0; u < p_uNumberOfParticles; ++u )
    {
        //ds do not treat itself (else nan results because division by zero)
        if( u != uIndex )
        {
            //ds cutoff distance
            const float fDistanceCutoff = 2.5*p_fMinimumDistance;

            //ds we have to loop over the cubic boundary conditions
            for( float dX = p_fLowerBoundary; dX < p_fUpperBoundary+1.0; ++dX )
            {
                for( float dY = p_fLowerBoundary; dY < p_fUpperBoundary+1.0; ++dY )
                {
                    for( float dZ = p_fLowerBoundary; dZ < p_fUpperBoundary+1.0; ++dZ )
                    {
                        //ds get the radial vector between the particles
                        float vecRadius[3];

                        //ds calculate the distance: domain + particle2 - particle1
                        vecRadius[0] = dX*fDomainSize + p_arrPositions[3*u+0] - p_arrPositions[uIndexReal+0];
                        vecRadius[1] = dY*fDomainSize + p_arrPositions[3*u+1] - p_arrPositions[uIndexReal+1];
                        vecRadius[2] = dZ*fDomainSize + p_arrPositions[3*u+2] - p_arrPositions[uIndexReal+2];

                        //ds get the absolute distance
                        const float fDistanceAbsolute( sqrt( pow( vecRadius[0], 2 ) + pow( vecRadius[1], 2 ) + pow( vecRadius[2], 2 ) ) );

                        //ds if we are within the cutoff range (only smaller here to avoid double overhead for >=)
                        if( fDistanceCutoff > fDistanceAbsolute )
                        {

                            //ds calculate the lennard jones force prefix
                            const float fLJFPrefix( -24*p_fPotentialDepth*( 2*pow( p_fMinimumDistance/fDistanceAbsolute, 12 ) - pow( p_fMinimumDistance/fDistanceAbsolute, 6  ) )
                                                                         *1/pow( fDistanceAbsolute, 2 ) );

                            //ds add the information to the force including the radial component
                            vecTotalForce[0] += fLJFPrefix*vecRadius[0];
                            vecTotalForce[1] += fLJFPrefix*vecRadius[1];
                            vecTotalForce[2] += fLJFPrefix*vecRadius[2];
                        }
                    }
                }
            }
        }
    }

    //ds new acceleration
    float vecNewAcceleration[3];

    //ds calculate it from the force
    vecNewAcceleration[0] = vecTotalForce[0]/fCurrentMass;
    vecNewAcceleration[1] = vecTotalForce[1]/fCurrentMass;
    vecNewAcceleration[2] = vecTotalForce[2]/fCurrentMass;

    //ds wait until all threads are done so we can proceed setting the new position and velocities without interfering the calculations
    __syncthreads( );

    //ds velocity-verlet for position
    p_arrPositions[uIndexReal+0] = p_arrPositions[uIndexReal+0] + p_fTimeStepSize*p_arrVelocities[uIndexReal+0] + ( 1/2 )*pow( p_fTimeStepSize, 2 )*p_arrAccelerations[uIndexReal+0];
    p_arrPositions[uIndexReal+1] = p_arrPositions[uIndexReal+1] + p_fTimeStepSize*p_arrVelocities[uIndexReal+1] + ( 1/2 )*pow( p_fTimeStepSize, 2 )*p_arrAccelerations[uIndexReal+1];
    p_arrPositions[uIndexReal+2] = p_arrPositions[uIndexReal+2] + p_fTimeStepSize*p_arrVelocities[uIndexReal+2] + ( 1/2 )*pow( p_fTimeStepSize, 2 )*p_arrAccelerations[uIndexReal+2];

    //ds produce periodic boundary shifting - check each element: x,y,z
    for( unsigned int v = 0; v < 3; ++v )
    {
        //ds check if we are below the boundary
        while( p_fLowerBoundary > p_arrPositions[uIndexReal+v] )
        {
            //ds map the particle to the other boundary by shifting it up to the boundary
            p_arrPositions[uIndexReal+v] += fDomainSize;
        }

        //ds check if we are above the boundary
        while( p_fUpperBoundary < p_arrPositions[uIndexReal+v] )
        {
            //ds map the particle to the other boundary by shifting it back to the boundary
            p_arrPositions[uIndexReal+v] -= fDomainSize;
        }
    }

    //ds velocity-verlet for velocity
    p_arrVelocities[uIndexReal+0] = p_arrVelocities[uIndexReal+0] + ( p_fTimeStepSize/2 )*( vecNewAcceleration[0] + p_arrAccelerations[uIndexReal+0] );
    p_arrVelocities[uIndexReal+1] = p_arrVelocities[uIndexReal+1] + ( p_fTimeStepSize/2 )*( vecNewAcceleration[1] + p_arrAccelerations[uIndexReal+1] );
    p_arrVelocities[uIndexReal+2] = p_arrVelocities[uIndexReal+2] + ( p_fTimeStepSize/2 )*( vecNewAcceleration[2] + p_arrAccelerations[uIndexReal+2] );

    //ds update the old accelerations
    p_arrAccelerations[uIndexReal+0] = vecNewAcceleration[0];
    p_arrAccelerations[uIndexReal+1] = vecNewAcceleration[1];
    p_arrAccelerations[uIndexReal+2] = vecNewAcceleration[2];
}

__global__ void getTotalEnergy( const unsigned int p_uNumberOfParticles,
                                float* p_arrPositions,
                                float* p_arrVelocities,
                                float* p_arrMasses,
                                const float p_fMinimumDistance,
                                const float p_fPotentialDepth,
                                float* p_fTotalEnergy )
{
    //ds shared total energy to sum up by first thread
    __shared__ float s_arrTotalEnergy[NumberOfParticles];

    //ds regular index and "real" particle index equals three times thread index, since were working with a linear 2d array
    const unsigned int uIndex    ( threadIdx.x );
    const unsigned int uIndexReal( 3*uIndex );

    //ds calculate the total energy of the new configuration - loop over all other particles (dont do the same particles twice)
    for( unsigned int u = uIndex+1; u < p_uNumberOfParticles; ++u )
    {
        //ds add the kinetic component from the other particle
        s_arrTotalEnergy[uIndex] += p_arrMasses[u]/2*pow( sqrt( pow( p_arrVelocities[3*u+0], 2 ) + pow( p_arrVelocities[3*u+1], 2 ) + pow( p_arrVelocities[3*u+2], 2 ) ), 2 );

        //ds get the radial vector between the particles
        float vecRadius[3];

        //ds calculate the distance: particle2 - particle1
        vecRadius[0] = p_arrPositions[3*u+0] - p_arrPositions[uIndexReal+0];
        vecRadius[1] = p_arrPositions[3*u+1] - p_arrPositions[uIndexReal+1];
        vecRadius[2] = p_arrPositions[3*u+2] - p_arrPositions[uIndexReal+2];

        //ds get the absolute distance
        const float fDistanceAbsolute( sqrt( pow( vecRadius[0], 2 ) + pow( vecRadius[1], 2 ) + pow( vecRadius[2], 2 ) ) );

        //ds add the potential component
        s_arrTotalEnergy[uIndex] += 4*p_fPotentialDepth*( pow( p_fMinimumDistance/fDistanceAbsolute, 12 ) - pow( p_fMinimumDistance/fDistanceAbsolute, 6 ) );
    }

    //ds wait until all threads are done
    __syncthreads( );

    //ds thread 0 calculates the total energy
    if( 0 == uIndex )
    {
        //ds total energy to sum up
        float fTotalEnergy( 0.0 );

        for( unsigned int u = 0; u < p_uNumberOfParticles; ++u )
        {
            fTotalEnergy += s_arrTotalEnergy[u];
        }

        //ds set the return value
        *p_fTotalEnergy = fTotalEnergy;
    }
}

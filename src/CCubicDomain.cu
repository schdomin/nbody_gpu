#include "CCubicDomain.cuh"



namespace NBody
{

//ds ctor/dtor
//ds default constructor requires environmental parameters: N number of bodies, dT time step, T number of time steps
CCubicDomain::CCubicDomain( const std::pair< double, double >& p_pairBoundaries,
                            const unsigned int& p_uNumberOfParticles ): m_arrParticles( 0 ),
                                                                        m_pairBoundaries( p_pairBoundaries ),
                                                                        m_dDomainSize( labs( m_pairBoundaries.first ) + labs( m_pairBoundaries.second ) ),
                                                                        m_uNumberOfParticles( p_uNumberOfParticles ),
                                                                        m_strParticleInformation( "" ),
                                                                        m_strIntegralsInformation( "" )
{
    //ds nothing to do
}

//ds default destructor
CCubicDomain::~CCubicDomain( )
{
    //ds deallocate memory
    delete[] m_arrParticles;
}

//ds accessors
void CCubicDomain::createParticlesUniformFromNormalDistribution( const double& p_dTargetKineticEnergy, const double& p_dParticleMass )
{
    //ds allocate an array for the new particles
    m_arrParticles = new CParticle[m_uNumberOfParticles];

    //ds kinetic energy to derive from initial situation
    double dKineticEnergy( 0.0 );

    //ds set particle information for each
    for( unsigned int u = 0; u < m_uNumberOfParticles; ++u )
    {
        //ds set the particle mass (same for all particles in this case)
        m_arrParticles[u].m_dMass = p_dParticleMass;

        //ds set the position: uniformly distributed between boundaries in this case
        m_arrParticles[u].m_cPosition = NBody::CVector( _getUniformlyDistributedNumber( ),
                                                        _getUniformlyDistributedNumber( ),
                                                        _getUniformlyDistributedNumber( ) );

        //ds set velocity values: from normal distribution
        m_arrParticles[u].m_cVelocity = NBody::CVector( _getNormallyDistributedNumber( ) ,
                                                        _getNormallyDistributedNumber( ) ,
                                                        _getNormallyDistributedNumber( ) );

        //ds add the resulting kinetic component (needed below)
        dKineticEnergy += m_arrParticles[u].m_dMass/2*pow( NBody::CVector::absoluteValue( m_arrParticles[u].m_cVelocity ), 2 );
    }

    //ds calculate the rescaling factor
    const double dRescalingFactor( sqrt( p_dTargetKineticEnergy/dKineticEnergy ) );

    //ds for each particle
    for( unsigned int u = 0; u < m_uNumberOfParticles; ++u )
    {
        //ds rescale the velocity component
        m_arrParticles[u].m_cVelocity *= dRescalingFactor;
    }
}

void CCubicDomain::updateParticlesVelocityVerlet( const double& p_dTimeStep, const double& p_dMinimumDistance, const double& p_dPotentialDepth )
{
    //ds allocate a temporary array to hold the accelerations
    CVector *arrNewAccelerations = new CVector[m_uNumberOfParticles];

    //ds for each particle
    for( unsigned int u = 0; u < m_uNumberOfParticles; ++u )
    {
        //ds sum of forces acting on the particle
        CVector cTotalForce;

        //ds loop over all other particles
        for( unsigned int v = 0; v < m_uNumberOfParticles; ++v )
        {
            //ds if its not the same particle
            if( u != v )
            {
                //ds collect the force from the current particle and add it (the function takes care of periodic boundary condition)
                cTotalForce += _getLennardJonesForce( m_arrParticles[u], m_arrParticles[v], p_dMinimumDistance, p_dPotentialDepth );
            }
        }

        //ds if we got the total force calculate the resulting acceleration and save it to our array
        arrNewAccelerations[u] = cTotalForce/m_arrParticles[u].m_dMass;
    }

    //ds for each particle we have to calculate the effect of the acceleration now
    for( unsigned int u = 0; u < m_uNumberOfParticles; ++u )
    {
        //ds velocity-verlet for position
        m_arrParticles[u].m_cPosition = m_arrParticles[u].m_cPosition + p_dTimeStep*m_arrParticles[u].m_cVelocity + 1/2*pow( p_dTimeStep, 2 )*m_arrParticles[u].m_cAcceleration;

        //ds produce periodic boundary shifting - check each element: x,y,z
        for( unsigned int v = 0; v < 3; ++v )
        {
            //ds check if we are below the boundary
            while( m_pairBoundaries.first > m_arrParticles[u].m_cPosition( v ) )
            {
                //ds map the particle to the other boundary by shifting it up to the boundary
                m_arrParticles[u].m_cPosition( v ) += m_dDomainSize;
            }

            //ds check if we are above the boundary
            while( m_pairBoundaries.second < m_arrParticles[u].m_cPosition ( v ) )
            {
                //ds map the particle to the other boundary by shifting it back to the boundary
                m_arrParticles[u].m_cPosition( v ) -= m_dDomainSize;
            }
        }

        //ds velocity-verlet for velocity
        m_arrParticles[u].m_cVelocity = m_arrParticles[u].m_cVelocity + p_dTimeStep/2*( arrNewAccelerations[u] + m_arrParticles[u].m_cAcceleration );

        //ds update the acceleration
        m_arrParticles[u].m_cAcceleration = arrNewAccelerations[u];
    }

    //ds deallocate the temporary accelerations array
    delete[] arrNewAccelerations;
}

void CCubicDomain::saveParticlesToStream( )
{
    //ds format: X Y Z U V W

    //ds for each particle
    for( unsigned int u = 0; u < m_uNumberOfParticles; ++u )
    {
        //ds get a buffer for snprintf (max possible buffer size checked)
        char chBuffer[256];

        //ds get the particle stream
        std::snprintf( chBuffer, 100, "%f %f %f %f %f %f", m_arrParticles[u].m_cPosition( 0 ), m_arrParticles[u].m_cPosition( 1 ), m_arrParticles[u].m_cPosition( 2 ),
                                            m_arrParticles[u].m_cVelocity( 0 ), m_arrParticles[u].m_cVelocity( 1 ), m_arrParticles[u].m_cVelocity( 2 ) );

        //ds append the buffer to our string
        m_strParticleInformation += chBuffer;
        m_strParticleInformation += "\n";
    }
}

void CCubicDomain::saveIntegralsToStream( const double& p_dMinimumDistance, const double& p_dPotentialDepth )
{
    //ds format: E X Y Z X Y Z X Y Z

    //ds buffer for snprintf
    char chBuffer[256];

    //ds get the integrals stream
    std::snprintf( chBuffer, 100, "%f %f %f %f %f %f %f %f %f %f", getTotalEnergy( p_dMinimumDistance, p_dPotentialDepth ),
                                                                   getCenterOfMass( )( 0 ), getCenterOfMass( )( 1 ), getCenterOfMass( )( 2 ),
                                                                   getAngularMomentum( )( 0 ), getAngularMomentum( )( 1 ), getAngularMomentum( )( 2 ),
                                                                   getLinearMomentum( )( 0 ), getLinearMomentum( )( 1 ), getLinearMomentum( )( 2 ) );

    //ds append the buffer to our string
    m_strIntegralsInformation += chBuffer;
    m_strIntegralsInformation += "\n";
}

void CCubicDomain::writeParticlesToFile( const std::string& p_strFilename, const unsigned int& p_uNumberOfTimeSteps )
{
    //ds ofstream object
    std::ofstream ofsFile;

    //ds open the file for writing
    ofsFile.open( p_strFilename.c_str( ), std::ofstream::out );

    //ds if it worked
    if( ofsFile.is_open( ) )
    {
        //ds first dump setup information number of particles and timesteps
        ofsFile << m_uNumberOfParticles << " " << p_uNumberOfTimeSteps << "\n" << m_strParticleInformation;
    }

    //ds close the file
    ofsFile.close( );
}

void CCubicDomain::writeIntegralsToFile( const std::string& p_strFilename, const unsigned int& p_uNumberOfTimeSteps, const double& p_dTimeStepSize )
{
    //ds ofstream object
    std::ofstream ofsFile;

    //ds open the file for writing
    ofsFile.open( p_strFilename.c_str( ), std::ofstream::out );

    //ds if it worked
    if( ofsFile.is_open( ) )
    {
        //ds dump first integrals information
        ofsFile << p_uNumberOfTimeSteps << " " << p_dTimeStepSize << "\n" << m_strIntegralsInformation;
    }

    //ds close the file
    ofsFile.close( );
}

//ds accessors/helpers
double CCubicDomain::getTotalEnergy( const double& p_dMinimumDistance, const double& p_dPotentialDepth ) const
{
    //ds total energy to accumulate
    double dTotalEnergy( 0.0 );

    //ds for each particle
    for( unsigned int u = 0; u < m_uNumberOfParticles; ++u )
    {
        //ds loop over all other particles (dont do the same particles twice)
        for( unsigned int v = u+1; v < m_uNumberOfParticles; ++v )
        {
            //ds add the kinetic component
            dTotalEnergy += m_arrParticles[u].m_dMass/2*pow( NBody::CVector::absoluteValue( m_arrParticles[u].m_cVelocity ), 2 );

            //ds add the potential component
            dTotalEnergy += _getLennardJonesPotential( m_arrParticles[u], m_arrParticles[v], p_dMinimumDistance, p_dPotentialDepth );
        }
    }

    return dTotalEnergy;
}

CVector CCubicDomain::getCenterOfMass( ) const
{
    //ds center to find
    CVector cCenter;

    //ds total mass
    double dMassTotal( 0.0 );

    //ds for each particle
    for( unsigned int u = 0; u < m_uNumberOfParticles; ++u )
    {
        //ds add the current relative mass
        cCenter += m_arrParticles[u].m_dMass*m_arrParticles[u].m_cPosition;

        //ds add the current mass
        dMassTotal += m_arrParticles[u].m_dMass;
    }

    //ds divide by total mass
    cCenter /= dMassTotal;

    return cCenter;
}

CVector CCubicDomain::getAngularMomentum( ) const
{
    //ds momentum
    CVector cMomentum;

    //ds for each particle
    for( unsigned int u = 0; u < m_uNumberOfParticles; ++u )
    {
        //ds add the current momentum
        cMomentum += NBody::CVector::crossProduct( m_arrParticles[u].m_cPosition, m_arrParticles[u].m_dMass*m_arrParticles[u].m_cVelocity );
    }

    return cMomentum;
}

CVector CCubicDomain::getLinearMomentum( ) const
{
    //ds momentum
    CVector cMomentum;

    //ds for each particle
    for( unsigned int u = 0; u < m_uNumberOfParticles; ++u )
    {
        //ds add the current momentum
        cMomentum += m_arrParticles[u].m_dMass*m_arrParticles[u].m_cVelocity;
    }

    return cMomentum;
}

//ds helpers
double CCubicDomain::_getLennardJonesPotential( const CParticle& p_CParticle1,  const CParticle& p_CParticle2, const double& p_dMinimumDistance, const double& p_dPotentialDepth ) const
{
    //ds formula
    return 4*p_dPotentialDepth*( pow( p_dMinimumDistance/NBody::CVector::absoluteValue( p_CParticle1.m_cPosition-p_CParticle2.m_cPosition ), 12 )
                               - pow( p_dMinimumDistance/NBody::CVector::absoluteValue( p_CParticle1.m_cPosition-p_CParticle2.m_cPosition ), 6 ) );
}

CVector CCubicDomain::_getLennardJonesForce( const CParticle& p_CParticle1,  const CParticle& p_CParticle2, const double& p_dMinimumDistance, const double& p_dPotentialDepth ) const
{
    //ds cutoff distance
    const double dDistanceCutoff( 2.5*p_dMinimumDistance );

    //ds force to calculate
    CVector cForce;

    //ds we have to loop over the cubic boundary conditions
    for( double dX = m_pairBoundaries.first; dX < m_pairBoundaries.second+1; ++dX )
    {
        for( double dY = m_pairBoundaries.first; dY < m_pairBoundaries.second+1; ++dY )
        {
            for( double dZ = m_pairBoundaries.first; dZ < m_pairBoundaries.second+1; ++dZ )
            {
                CVector cRadius( dX*m_dDomainSize + p_CParticle2.m_cPosition( 0 ) - p_CParticle1.m_cPosition( 0 ),
                                 dY*m_dDomainSize + p_CParticle2.m_cPosition( 1 ) - p_CParticle1.m_cPosition( 1 ),
                                 dZ*m_dDomainSize + p_CParticle2.m_cPosition( 2 ) - p_CParticle1.m_cPosition( 2 ) );

                //ds get the current distance between 2 and 1
                const double dDistanceAbsolute( NBody::CVector::absoluteValue( cRadius ) );

                //ds if we are within the cutoff range (only smaller here to avoid double overhead for >=)
                if( dDistanceCutoff > dDistanceAbsolute )
                {
                    //ds add the force
                    cForce += -24*p_dPotentialDepth*( 2*pow( p_dMinimumDistance/dDistanceAbsolute, 12 ) - pow( p_dMinimumDistance/dDistanceAbsolute, 6  ) )
                                                   *1/pow( dDistanceAbsolute, 2 )*cRadius;
                }
            }
        }
    }

    return cForce;
}

double CCubicDomain::_getUniformlyDistributedNumber( ) const
{
    //ds drand48 returns [0,1], we need [-1,1] -> therefore 2x[0,1] -> [0,2] -> -1 ->[0-1,2-1] = [-1,1]
    return 2*drand48( )-1;
}

double CCubicDomain::_getNormallyDistributedNumber( ) const
{
    //ds calculate the uniform number first [0,1]
    const double dUniformNumber( drand48( ) );

    //ds return the normal one
    return sqrt( -2*log( dUniformNumber ) )*cos( 2*M_PI*dUniformNumber );
}

} //ds namespace NBody

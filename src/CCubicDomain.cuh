#ifndef CCUBICDOMAIN_CUH_
#define CCUBICDOMAIN_CUH_

#include <fstream>  //ds streaming, utilities etc



namespace NBody
{

//ds domain to solve n-body problems
class CCubicDomain
{

//ds ctor/dtor
public:

    //ds default constructor requires environmental parameters: N number of bodies, dT time step, T number of time steps
    CCubicDomain( const unsigned int& p_uNumberOfParticles );

    //ds default destructor
    ~CCubicDomain( );

//ds attributes
private:

    //ds float references of particles for cuda
    float* m_arrPositions;
    float* m_arrVelocities;
    float* m_arrAccelerations;
    float* m_arrMasses;

    //ds domain properties
    const unsigned int m_uNumberOfParticles;

    //ds streams for offline data - needed for the movie and graphs
    std::string m_strParticleInformation;
    std::string m_strIntegralsInformation;

//ds accessors
public:

    void createParticlesUniformFromNormalDistribution( const float& p_dTargetKineticEnergy, const float& p_fParticleMass = 1.0 );
    void saveParticlesToStream( );
    void saveIntegralsToStream( const float& p_fTotalEnergy );
    void saveIntegralsToStream( const float& p_fTotalEnergy, const float p_vecCenterOfMass[3] );
    void writeParticlesToFile( const std::string& p_strFilename, const unsigned int& p_uNumberOfTimeSteps );
    void writeIntegralsToFile( const std::string& p_strFilename, const unsigned int& p_uNumberOfTimeSteps, const double& p_dTimeStepSize );

    //ds cuda access
    float* getPositions( );
    float* getVelocities( );
    float* getAccelerations( );
    float* getMasses( );

//ds accessors/helpers
public:

    const float* getCenterOfMass( ) const;
    const float* getAngularMomentum( ) const;
    const float* getLinearMomentum( ) const;

//ds helpers
private:

    float _getUniformlyDistributedNumber( ) const;
    float _getNormallyDistributedNumber( ) const;

};

} //ds namespace NBody



#endif //ds CCUBICDOMAIN_CUH_

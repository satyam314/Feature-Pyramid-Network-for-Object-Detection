import { useCallback } from 'react';
import Particles from 'react-tsparticles';
import { loadBasic } from 'tsparticles-basic'; // <- use this now

const ParticlesBackground = () => {
  const particlesInit = useCallback(async (engine) => {
    // Only load basic features (faster, avoids engine errors)
    await loadBasic(engine);
  }, []);

  return (
    <Particles
      id="tsparticles"
      init={particlesInit}
      options={{
        fullScreen: {
          enable: true,
          zIndex: -1,
        },
        background: {
          color: '#242424',
        },
        particles: {
          number: {
            value: 50,
            density: {
              enable: true,
              area: 800,
            },
          },
          color: {
            value: '#888',
          },
          links: {
            enable: true,
            color: '#888',
            distance: 150,
            opacity: 0.5,
          },
          move: {
            enable: true,
            speed: 1,
          },
          size: {
            value: 3,
          },
        },
      }}
    />
  );
};

export default ParticlesBackground;

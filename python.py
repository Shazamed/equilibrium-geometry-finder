import numpy as np
import matplotlib.pyplot as plt

RELATIVE_TOLERANCE = 1e-12 # for comparing if equilibrium energyhas been reached
DELTA = 1e-7
MIN_SEPARATION = 1 # minimum separation the particles in the initial random distribution

n = 7 # no of particles
TRIALS = 50 # no of runs

class SystemBase:
    def __init__(self, n):
        self.generate_random_particles(n)
        self.dE = np.zeros([self.n,3])

    def generate_random_particles(self, n):
        '''Generate n particles randomly such that
        the particles are placed with a distance of more than the MIN_SEPARATION'''
        self.n = n
        particle_list = []
        for i in range(n):
            while True:
                fail = False
                new_particle = np.random.random(3)*5
                for particle in particle_list:
                    if np.sqrt(np.sum((particle - new_particle)**2)) < MIN_SEPARATION:
                        fail = True
                if fail == True:
                    continue
                else:
                    break
            particle_list.append(new_particle)
        self.particles = np.array(particle_list)



    def step(self):
        self.calc_energy()
        self.calc_energy_derivative()
        self.calc_pos()
    
    def calc_energy(self):
        self.r = self.calc_r(self.particles)
        self.E = self.potential(self.r)
        self.total_E = np.nansum(self.E)/2 # divide by 2 as the sum of the energies of all particles is twice the total energy
    
    def potential(self): # to be overridden
        raise NotImplementedError("Must override potential method")

    def calc_energy_derivative(self):
        for direction_idx in range(3):
            for particle_idx in range(self.n):
                particle_plus = self.particles.copy()
                particle_minus = self.particles.copy()

                particle_plus[particle_idx,direction_idx] += DELTA
                particle_minus[particle_idx,direction_idx] -= DELTA

                r_pos = self.calc_r(particle_plus)
                r_neg = self.calc_r(particle_minus)

                self.dE[particle_idx,direction_idx] = np.nansum((self.potential(r_pos)-self.potential(r_neg))/(2*DELTA))/2


    def calc_r(self, particles):
        r_carte = particles.reshape(self.n,1,3) - self.particles
        r = np.sqrt((r_carte**2).sum(2))
        return r
    
    def calc_pos(self):
        self.particles -= self.dE * self.learning_rate
    
    def xyz_output(self):
        output_string = f"{self.n}\nCoordinates from potential E = {self.total_E}\n"
        for particle in range(self.n):
            output_string+=f"H   {self.particles[particle,0]}    {self.particles[particle,1]}    {self.particles[particle,2]}\n"
        with open("output.xyz", "w") as f:
            f.write(output_string)
        return output_string
    
    
class LennardJones(SystemBase):    
    def __init__(self, n):
        super().__init__(n)
        self.learning_rate = 1e-2

    def potential(self, r):
        return 4*((1/r)**12-(1/r)**6) 

class Morse(SystemBase): 
    def __init__(self, n, r_e):
        super().__init__(n)
        self.r_e = r_e
        self.learning_rate = 1e-2

    def potential(self, r):
        return (1-np.exp(-(r-self.r_e)))**2



step_count = 0 # step count
best_state = None
for i in range(TRIALS):
    state = LennardJones(n)
    # state = Morse(n, 1)
    # state = Morse(n, 3)
    while True:
        state.step()
        if step_count%1000 == 0:
            if step_count >= 2000:
                if np.allclose(-prev_E, -state.total_E, atol=0, rtol=RELATIVE_TOLERANCE):
                    print("convergence reached")
                    state.xyz_output()
                    print(f'Run {i+1} Energy: {state.total_E}')
                    # print(state.r)
                    if best_state == None:
                        best_state = state
                    elif best_state.total_E > state.total_E:
                        best_state = state
                    break

            # print(state.total_E)
            prev_E = state.total_E.copy()
        step_count += 1
        if np.sum(state.r) > 1000000000:
            print("values diverged")
            break

print(f'Best Energy = {best_state.total_E}')
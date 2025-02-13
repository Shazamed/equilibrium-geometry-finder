import numpy as np
import matplotlib.pyplot as plt

RELATIVE_TOLERANCE = 1e-12 # for comparing if equilibrium energy has been reached
DELTA = 1e-7
MIN_SEPARATION = 1 # minimum separation the particles in the initial random distribution

np.seterr(divide='ignore')
np.seterr(invalid='ignore')

class SystemBase:
    def __init__(self, n):
        self.generate_random_particles(n)
        self.dE = np.zeros([self.n,3]) # nx3 array where each row has the cartesian coordinates of a particle

    def generate_random_particles(self, n):
        '''Generate n particles randomly such that
        the particles are placed with a distance of more than the MIN_SEPARATION.
        Will become stuck if MIN_SEPARATION is too large or if there are too many particles
        '''
        self.n = n
        particle_list = []
        for i in range(n):
            while True:
                fail = False
                new_particle = np.random.random(3)*5
                for particle in particle_list:
                    if np.sqrt(np.sum((particle - new_particle)**2)) < MIN_SEPARATION: # compare if newly placed particle is within MIN_SEPARATION of another
                        fail = True
                if fail == True:
                    continue
                else:
                    break # break and append new particle coordinates to list if it is further than MIN_SEPARATION of all other particles otherwise try again with another random coordinate
            particle_list.append(new_particle)
        self.particles = np.array(particle_list)
        print("Random Particle Placement Complete")


    def step(self):
        self.calc_energy()
        self.calc_energy_derivative()
        self.calc_pos()
    
    def calc_energy(self):
        '''
        Calculate energy of the system
        self.E has the following form for a 3 particle system:
        [[        nan -0.00345295 -0.00018482]
        [-0.00345295         nan -0.00163058]
        [-0.00018482 -0.00163058         nan]]
        '''
        self.r = self.calc_r(self.particles)
        self.E = self.potential(self.r)
        print(self.E)
        self.total_E = np.nansum(self.E)/2 # divide by 2 as the sum of the energies of all particles is twice the total energy
    
    def potential(self): # to be overridden
        raise NotImplementedError("Must override potential method")

    def calc_energy_derivative(self):
        '''
        Calculate the change in energy if the particle was shifted by DELTA in both positive and negative directions
        eg. 
        self.dE has the following form where rows are the atoms and columns are the x,y,z axes
        [[ 0.00996625 -0.01268623  0.00440692]
        [-0.012583    0.01660699 -0.00834977]
        [ 0.01250682  0.01176487  0.00646947]
        [ 0.00373357 -0.00595344  0.00350425]
        [-0.01402332 -0.00501582  0.00831021]
        [ 0.00342937 -0.0005282  -0.01086932]
        [-0.00302968 -0.00418817 -0.00347175]]
        '''
        for direction_idx in range(3): # iterate over the x,y,z axes to find the gradient in the specific axis
            for particle_idx in range(self.n):
                particle_plus = self.particles.copy()
                particle_minus = self.particles.copy()

                particle_plus[particle_idx,direction_idx] += DELTA
                particle_minus[particle_idx,direction_idx] -= DELTA

                r_pos = self.calc_r(particle_plus)
                r_neg = self.calc_r(particle_minus)

                self.dE[particle_idx,direction_idx] = np.nansum((self.potential(r_pos)-self.potential(r_neg))/(2*DELTA))/2

    def calc_r(self, particles):
        '''
        Calculate the distance, r, between particles,
        matrix element r_ij is the distance between particle i and particle j
        eg for a 3 particle system:
        [[0.00000000e+00 3.39945506e+00 1.12244861e+00]
        [3.39945506e+00 0.00000000e+00 3.22263196e+00]
        [1.12244860e+00 3.22263186e+00 0.00000000e+00]]
        '''
        r_carte = particles.reshape(self.n,1,3) - self.particles
        r = np.sqrt((r_carte**2).sum(2)) # converting from cartesian coordinates to absolute distances
        return r
    
    def calc_pos(self):
        # move particles based on the gradient and scaling
        self.particles -= self.dE * self.scaling_rate
    
    def xyz_output(self): # save output.xyz
        output_string = f"{self.n}\nCoordinates from potential E = {self.total_E}ϵ\n"
        for particle in range(self.n):
            output_string+=f"H   {self.particles[particle,0]}    {self.particles[particle,1]}    {self.particles[particle,2]}\n"
        with open("output.xyz", "w", encoding="utf-8") as f:
            f.write(output_string)
        return output_string
    
    def r_table(self): # print table of interparticle distances and saves it to table.out
        output_string = f"Interparticle distances/σ"
        np.savetxt('table.out', self.r, delimiter=',',header=output_string, encoding='utf-8')
        print(output_string)
        print(self.r)
    
    
class LennardJones(SystemBase):    
    def __init__(self, n):
        super().__init__(n)
        self.scaling_rate = 1e-2

    def potential(self, r):
        return 4*((1/r)**12 - (1/r)**6) 

class Morse(SystemBase): 
    def __init__(self, n, r_e):
        super().__init__(n)
        self.r_e = r_e
        self.scaling_rate = 3e-2

    def potential(self, r):
        return (1-np.exp(-(r-self.r_e)))**2

# user interface
print("Select the potential to be used for the determination of the equilibrium geometry")
print('''1. Lennard-Jones
2. Morse r_e/σ = 1
3. Morse r_e/σ = 3

Type the selection number of the potential:''')
input_success = False
while not input_success:
    num_selection = input()
    if not num_selection.isdigit():
        print("Please enter a valid number!")
    elif 0 < int(num_selection) <= 3:
        num_selection = int(num_selection)
        input_success = True
    else:
        print("Please enter a valid number!")

print("Type the number of runs to complete")
input_success = False
while not input_success:
    trials = input()
    if not trials.isdigit():
        print("Please enter a valid number!")
    else:
        trials = int(trials)
        input_success = True

print("Type the number of atoms in the system")
input_success = False
while not input_success:
    n = input()
    if not n.isdigit():
        print("Please enter a valid number!")
    else:
        n = int(n)
        input_success = True


step_count = 0 # step count
best_state = None
for i in range(trials):
    match num_selection: # determine the potential used from user selection
        case 1:
            state = LennardJones(n)
        case 2:
            state = Morse(n, 1)
        case 3:
            state = Morse(n, 3)

    while True: # iterate continuously until the system energy converges
        state.step()
        if step_count % 1000 == 0: # compare energies every 1000 steps
            if step_count >= 2000: 
                if np.allclose(prev_E, state.total_E, atol=0, rtol=RELATIVE_TOLERANCE):
                    print("Convergence Reached")
                    print(f'Run {i+1} Energy: {state.total_E}ϵ\n')
                    if best_state == None:
                        best_state = state
                    elif best_state.total_E > state.total_E:
                        best_state = state
                    break

            prev_E = state.total_E.copy() # save state for comparison
        step_count += 1

        if np.sum(state.r) > 1000000000: # stops the calculation if distances become too large and diverges
            print("values diverged")
            break

print(f'Best Energy = {best_state.total_E}ϵ')
best_state.xyz_output() # saves best energy into .xyz file
best_state.r_table() 
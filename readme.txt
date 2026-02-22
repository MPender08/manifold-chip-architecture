Installation & Usage
=========================

To run the simulation and generate the phase transition graphs or the decision boundary comparison yourself, you will need Python and a few dependencies.

	1. Install the required libraries: This project relies on torch and geoopt (for Riemannian optimization and hyperbolic manifolds).

   		pip install torch matplotlib geoopt numpy

    
	2. Run the simulation:

		python run_brain_sim.py

	
	These will reproduce the visuals I shared.


If you'd like to try it with your own seeds, or random seeds, use these settings at the top of run_brain_sim.py:

	# --- SEED CONTROL ---
	# Try your own seeds, or turn off for random ones.
	# The seed for the visual used in the README.me is 137
	    USE_LOCKED_SEED = True
	    LOCKED_SEED = 137


# uvlm2d
2D UVLM solver in python


# Prerequisites: 
- python 3.6 or above (tested on Python 3.6.0|Anaconda 4.3.0 with IPython 5.1.0)


# Installation:
- No installation required;
- Optional: 
    - In your bash file, create the environmental variable:
    export DIRuvlm2d="/home/sm6110/git/uvlm2d"
    - At the beginning of each input file, add:
    import os, sys
    sys.path.append(os.environ["DIRuvlm2d"])
    

# Run:
- see ./test/valid*.ipynb or ./test/test_dyn.py


# Versions summary
- v0.0: static/dynamic solutions with prescribed wake working. Version not available in git.
- v1.0: nondimensional static/dynamic solutions with prescribed wake working.
- v2.0: (in progress) solutions with vectorised formulation/linearisation.

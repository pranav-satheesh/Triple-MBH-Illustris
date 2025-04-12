# Triple-Outcomes

### Folder structure

```md
Triples
├── Illustris_Data
        ├── mbhb_evolution_no-ecc_lc-full-0.6.hdf5
        ├── ...
        ├── ill-1_blackhole_mergers_fixed.npz
├── Triple-Outcome (this repo)  
        ├── Figures
        ├── Old notebooks
        ├── obj_data
                ├── iso_bin.pkl
        ├── triple_mbhb
                ├── find_triple_mbhs.py
                ├── read_mbhb.py
                ├── ms_script_pranav.py
                ├── create_files_from_triple_mbhb.py
       
        
        ├── add_dynamics.ipynb
        ├── Triple_dynamics.py
```

The binary merger files and triple finding files are stored in the parent directory (**~/Triples/**) since they are too large to be stored in Github. The files include

1. An hdf5 file containing that stores the parameters of the binary with the code. eg: **mbhb_evolution_no-ecc_lc-full-0.6.hdf5** referes to an MBHB evolution model with no eccentricity assumed but the loss cone refelling parameter set to 0.6. In the caser of eccentricity the file would be **mbhb-evolution_fiducial_ecc-evo-0.6_lc-shm06.hdf5**

2. A .npz file containing the ids of binary mergers in Illustris-1. **ill-1_blackhole_mergers_fixed.npz**. 

3. Folder containing the triple finding .py files. **~/Triples/triple_mbhb** 

### Current pipeline 

1. ```ms_script_pranav.py``` in  **~/Triples/triple_mbhb**  calls the triple finding algorithm to find the triples in Illustris and the output is stored in ```~\illustris_Data\``` as a `.csv` file. The output consists of `.csv` file for strong triples, all triples and isolated binaries.

2. ```Triple_dynamics.py``` has classes defined for strong triples, isolated binaries and weak triples. In the `strongtriple` class, dynamics or outcomes of their evolution is added based on interpolation with Bonetti simulation data.

3. ```add_MBH_dynamics.ipynb``` is the main notebook that calls all the other functions and make plots which are stored in **~/Triples/Triple-Outcomes/Figures**


Slurm script *create_triple_files_N100.slurm* runs 100 instances and storkes the pkl file inside obj_data.
# Triple-Outcomes

### Folder structure

```md
Triples
├── Github
│   ├── Triple-Outcome
│   ├── Illustris_Data
│       ├── mbhb_evolution_no-ecc_lc-full-0.6.hdf5
        ├── ...
│       ├── ill-1_blackhole_mergers_fixed.npz
│   ├── Sayeb
│       ├── find_triple_mbhs.py
```

The binary merger files and triple finding files are stored in the parent directory (**~/Triples/Github/**) since they are too large to be stored in Github. The files include

1. An hdf5 file containing that stores the parameters of the binary with the code. eg: **mbhb_evolution_no-ecc_lc-full-0.6.hdf5** referes to an MBHB evolution model with no eccentricity assumed but the loss cone refelling parameter set to 0.6. In the caser of eccentricity the file would be **mbhb-evolution_fiducial_ecc-evo-0.6_lc-shm06.hdf5**

2. A .npz file containing the ids of binary mergers in Illustris-1. **ill-1_blackhole_mergers_fixed.npz**. 

3. Sayeb's triple finding algorithm files. **~/Triples/Github/sayeb** 

### Current pipeline 

1. ```/Triple-Outcome/Findig_triples.ipynb``` calls the triple finding algorithm from Sayeb to find the triples in the binary file and the output ```triple_data_ill.csv```is stored in ```~\illustris_Data\``` as a .csv file.

2. 
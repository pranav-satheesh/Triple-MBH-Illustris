import numpy as np
fname = 'Data/Paper2_tables.csv'

c1,c2,c3,c4,c5 = np.loadtxt(fname,delimiter=",", dtype=str,unpack=True)
data_range = []
q_start = []
q_stop = []

for i in range(len(c1)):
    if (c1[i]==''):
        data_range.append(i)
    if (c1[i]=='q in / q out'):
        q_start.append(i+1)
    if (c1[i]=='e in / e out'):
        q_stop.append(i-1)
        
M1=[]
qin = []
qout = []
a = []
b = []
c = []
d = []

for j in range(6):
    M1_string = "10e" + str(10-j)
    
    for i in range(q_start[j],q_stop[j]):
        
        M1.append(float(M1_string))
        
        qs = c1[i].split("/")
        qin.append(float(qs[0]))
        qout.append(float(qs[1]))
        
        a.append(float(c2[i]))
        b.append(float(c3[i]))
        c.append(float(c4[i]))
    
        ds = c5[i].split("(")
        ds1 = ds[1].split(")")
        d.append(float(ds1[0]))
        
        
np.savetxt('Data/paper2_sorted_data.txt', np.column_stack([M1,qin,qout,a]))
print("File saved in Data/..")
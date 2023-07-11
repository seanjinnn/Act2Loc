from evaluations import *
from models.utils import *

def distance(gps1,gps2):
    x1,y1 = gps1
    x2,y2 = gps2
    return np.sqrt((x1-x2)**2+(y1-y2)**2 )

def gen_matrix(data='shenzhen'):
    train_data = read_data_from_file('../data/%s/real.data'%data)
    gps = get_gps('../data/%s/location.txt'%data)
    max_locs = 2236


    reg1 = np.zeros([max_locs+1, max_locs+1])
    num_data = len(train_data)
    for i in range(num_data):
        line = train_data[i]
        l = len(line)-1
        for j in range(l):
            reg1[line[j],line[j+1]] +=1
    # reg1 = reg1[1:,1:]
    reg2 = np.zeros([max_locs+1, max_locs+1])
    for i in range(max_locs):
        for j in range(max_locs):
            if i!=j:
                    reg2[i+1,j+1] = distance((gps[0][i],gps[1][i]),(gps[0][j],gps[1][j]))
    # reg2 = reg2[1:,1:]

    np.save('../data/%s/M1.npy'%data,reg1)
    np.save('../data/%s/M2.npy'%data,reg2)

    print('Matrix Generation Finished')


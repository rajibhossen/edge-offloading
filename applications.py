import scipy.stats as stats
import pdb

# Applications
SPEECH_RECOGNITION = 1
NLP = 2
FACE_RECOGNITION = 3
SEARCH_REQ = 4
LANGUAGE_TRANSLATION = 5
PROC_3D_GAME = 6
VR = 7
AR = 8

# Data size scales
BYTE = 8
KB = 1024*BYTE
MB = 1024*KB
GB = 1024*MB
TB = 1024*GB
PB = 1024*TB

# CPU clock frequency scales
KHZ = 1e3
MHZ = KHZ*1e3
GHZ = MHZ*1e3

app_info={
    SPEECH_RECOGNITION : {'workload':10435,
        'popularity': 5,
        'min_bits':40000*BYTE,
        'max_bits':300000*BYTE
    },NLP : {'workload':25346,
        'popularity': 8,
        'min_bits':4000*BYTE,
        'max_bits':100000*BYTE
    },FACE_RECOGNITION : {'workload':45043,
        'popularity': 4,
        'min_bits':10000*BYTE,
        'max_bits':100000*BYTE
    },SEARCH_REQ : {'workload':8405,
        'popularity': 0.125,
        'min_bits':800*BYTE,
        'max_bits':8000*BYTE
    },LANGUAGE_TRANSLATION : {'workload':34252,
        'popularity': 0.125,
        'min_bits':800*BYTE,
        'max_bits':8000*BYTE
    },PROC_3D_GAME : {'workload':54633,
        'popularity': 0.125,
        'min_bits':800*BYTE,
        'max_bits':8000*BYTE
    },VR : {'workload':40305,
        'popularity': 0.125,
        'min_bits':800*BYTE,
        'max_bits':8000*BYTE
    },AR : {'workload':34532,
        'popularity': 0.125,
        'min_bits':800*BYTE,
        'max_bits':8000*BYTE
    }
}


def app_type_list():
    return app_info.keys()


def app_type_pop():
    # result =[]
    # for i in list(app_info.keys()):

    #     result.append([i, app_info[i]['popularity']])
    # return result
    return [(i, app_info[i]['popularity']) for i in list(app_info.keys())]


def arrival_bits(app_type, dist = 'deterministic'):
    min_bits = app_info[app_type]['min_bits']
    max_bits = app_info[app_type]['max_bits']
    mu = (min_bits+max_bits)/2
    sigma = (max_bits-min_bits)/4
    if dist=='normal':
        return int(stats.truncnorm.rvs((min_bits-mu)/sigma, (max_bits-mu)/sigma, loc=mu, scale=sigma))
    elif dist=='deterministic':
        return mu
    else:
        return 1


def main():
    for j in range(5):
        result =[]
        for i in range(1,9):
            result.append(arrival_bits(i, dist='normal'))
        print(result)
    # for i in range(1,9):
    #     result.append(app_info[i]['workload']*app_info[i]['popularity']*arrival_bits(i, dist='normal'))
   # result = np.array(result)/GHZ

    #pdb.set_trace()


if __name__=='__main__':
    main()
parameter = {
    "esc": 1e-27,  # effective switch capacitance, 0.5e-28
    "w1": 1.5,  # weight factor for time 0.5
    "w2": 1,  # weight factor for energy
    "w3": 50,  # weight factor for money, only two will have impact, 10
    "edge_cps": 0.01 * 2,  # lambda@edge in cents, each request $0.0001 per GB-s, total 2GB
    "edge_request": 6e-5,  # edge request price in cents, $0.6 per 1M
    "cloud_cps": 0.002 * 3,  # aws lambda price in cents, $0.0000166667 per gb-sec, for 3 GB
    "cloud_request": 2e-5,  # cloud lambda request price in cents, $0.2 per 1M request
    "tr_power": 0.5,  # transmission power of mobile device in W, 500mW
    "tail_energy": 1.125,  # average lte tail energy is 1125 mW
    "tail_duration": 11.5,  # 11576.0±26.1 in seconds
    "mobile_com_cap": 1.5e9,  # 1.6 GHz processor speed, 1.6e9 cpu cycle per second
    "edge_com_cap": 3.2e9,  # 3.2 GHz processor speed
    "cloud_com_cap": 3.4e9,  # cloud computing capability 3.6 GHZ processor speed
    "cloud_cap": 0.6,  # allocated resource in cloud
    "max_penalty": -100,  # maximum penalty if violates constraint
    "total_energy": 120  # in J, energy = watt * voltage = wh, 3110 mah * 3.7v, 342 J
}

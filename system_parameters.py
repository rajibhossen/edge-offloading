parameter = {
    "esc": 1e-28,  # effective switch capacitance
    "w1": 0.25,  # weight factor for time 0.28
    "w2": 1,  # weight factor for energy 120
    "w3": 10,  # weight factor for money, only two will have impact
    "edge_cps": 0.01 * 2,  # lambda@edge in cents, each request $0.0001 per GB-s, total 2GB
    "edge_request": 6e-5,  # edge request price in cents, $0.6 per 1M
    "cloud_cps": 0.002 * 3,  # aws lambda price in cents, $0.0000166667 per gb-sec, for 3 GB
    "cloud_request": 2e-5,  # cloud lambda request price in cents, $0.2 per 1M request
    "tr_power": 0.5,  # transmission power of mobile device in W, 1865mW
    "tail_energy": 1.125,  # average lte tail energy is 1125 mW
    "tail_duration": 11.5,  # 11576.0±26.1 in seconds
    "mobile_com_cap": 2.6e9,  # 2.6 GHz processor speed, 1e9 cpu cycle per second
    "edge_com_cap": 3.4e9,  # 3.4 GHz processor speed
    "cloud_com_cap": 3.4e9,  # cloud computing capability 3.6 GHZ processor speed
    "cloud_cap": 1,  # allocated resource in cloud
    "max_penalty": -50,  # maximum penalty if violates constraint
    "total_energy": 340  # in J, energy = watt * voltage = wh, 3110 mah * 3.7v
}

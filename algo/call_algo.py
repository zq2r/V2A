
def call_algo(algo_name, config, mode, device):
    if mode == 0:
       pass

    elif mode == 1:
        pass

    elif mode == 2:
        pass

    elif mode == 3:
        algo_name = algo_name.lower()
        assert algo_name in ["v2a_igdf"]
        # offline offline setting
        from algo.offline_offline.v2a_igdf import V2A_IGDF

        algo_to_call = {
            "v2a_igdf": V2A_IGDF,
        }

        algo = algo_to_call[algo_name]
        policy = algo(config, device)
    
    return policy
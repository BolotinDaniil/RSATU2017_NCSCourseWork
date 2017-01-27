import json

config = None

def load_config(file_name):
    global config
    json_data = open(file_name).read()
    config = json.loads(json_data)
    g_s = 0
    for i in range(1, len(config['MLP']['layers'])):
        g_s += config['MLP']['layers'][i] * (1 + config['MLP']['layers'][i - 1])
    config['GA']['genotype_size'] = g_s

load_config('config1.json')






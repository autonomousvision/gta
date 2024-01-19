import os

mode = 'satori'

if mode == 'satori':
    logging_root = '/mnt/takeru/cross_attention_renderer/logs'
    results_root = '/mnt/takeru/cross_attention_renderer/logs'
    os.environ["TORCH_HOME"] = '/mnt/takeru/cross_attention_renderer'
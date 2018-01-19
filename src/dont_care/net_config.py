import yaml


class NetConfig(object):
    def __init__(self, config):
        with open(config, 'r') as stream:
            docs = yaml.load_all(stream)
            for doc in docs:
                for k, v in doc.items():
                    if k == "train":
                        for k1, v1 in v.items():
                            cmd = "self." + k1 + "=" + repr(v1)
                            print(cmd)
                            exec(cmd)
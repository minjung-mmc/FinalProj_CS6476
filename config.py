class EasyConfig(dict):
    def __getattr__(self, item):
        v = self.get(item)
        if isinstance(v, dict):
            v = EasyConfig(v) 
            self[item] = v
        return v
        
        
    def __setattr__(self, key, value):
        self[key] = value

        
cfg = EasyConfig({
        "DATA": {
            "BATCH_SIZE": 256,
            "VAL_SPLIT": 0.1,
            "GRAY": True,
        } ,
        "TRAIN": {
            "NUM_EPOCHS": 20,
            "LR":1e-3,
            "LOG_INTERVAL": 100,
        },
        "DEMO": {
            "THRESH": 0.7,
        },
        "MODEL":{
            "FREEZE_FEATURES": False,
        }
    })
class Params:
    def __init__(self):
        self.name = 'Lora'
        self.seed = 4123
        self.custom = True
        # path
        self.basemodel = 'pretrained_models/chilloutmixNiPruned_Tw1O'
        self.revision = None
        self.pretrain_path = None

        # data 
        self.resolution = 512
        self.center_crop = True
        self.random_flip = True
        self.dataset_name = './dataset/pokemon-blip-captions/data'
        self.img_column = 'image'
        self.caption_column = 'text'

        self.base = 'dataset/custom'

        # model 
        self.mixed_precision = True
        self.max_grad_norm = 1.0


        # optim
        self.learning_rate = 1e-05
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_weight_decay = 1e-2
        self.adam_epsilon = 1e-08
        self.lr_scheduler = 'constant'
        self.lr_warmup_steps = 0
        self.gradient_accumulation_steps = 4
        self.max_train_steps = 2000

        # loss
        self.snr_gamma = None
        self.noise_offset = 0
        self.prediction_type = None

        # eval 
        self.num_validation_images = 4


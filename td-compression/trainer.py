class Trainer:
    def __init__(
        self,
        train_loader,
        valid_loader,
        test_loader,
        model,
        optimizer,
        criterion,
        writer,
        device,
        scheduler=None,
        save=None,
        results={},
        tuning=False,
        **kwargs
    ):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        
        self.optimizer = optimizer
        self.scheduler = scheduler if scheduler is not None else None

        self.model = model
        self.criterion = criterion
        self.device = device
        self.model.train()

        self.iteration = 0
        self.writer = writer
        self.results = results

        self.tuning = tuning

        if save is None:
            self.save_every_epoch = None
            self.save_location = './'
            self.save_best = True
            self.save_final = True
            self.save_model_name = "model"
        else:
            self.save_every_epoch = save["save_every_epoch"]
            self.save_location = save["save_location"]
            self.save_best = save["save_best"]
            self.save_final = save["save_final"]
            self.save_model_name = save["save_model_name"]
    
    def test(self, loader=None):
        self.model = self.model.to(self.device)
        self.model.eval()
        correct = 0
        steps = 0
        total_time = 0
        val_loss = 0.0

        if loader == "valid":
            loader = self.valid_loader
        elif loader == "train":
            loader = self.train_loader
        elif loader == "test":
            loader = self.test_loader
        
        t = tqdm(loader, total=int(len(loader)))
        for i, (batch, label) in enumerate(t):
            with torch.no_grad():
                batch = batch.to(self.device)
                # TODO: finish
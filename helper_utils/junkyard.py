os.environ['TORCH_HOME'] = 'models/torch'

# load or create new model
if config['MODE'] == 'train':
    # a new segmentation model:
    CatSegModel = smp_model.CatSegModel(config)
    # train:
    train_logs, valid_logs, best_model = smp_model.train_model(
        CatSegModel, config, train_dl, valid_dl, 
        num_epochs=config['NUM_EPOCHS'])
    
if config['MODE'] == 'load':
    try:
        model, train_logs, valid_logs = smp_model.load_model(config['MODEL_PATH'])
        model.eval()  # Ensure the model is in evaluation mode
        print(f"Model loaded from {config['MODEL_PATH']}")
        print(f"Training Logs: {train_logs}")
        print(f"Validation Logs: {valid_logs}")
    except FileNotFoundError:
        raise ValueError('Cannot load Model. Check the path and file name.')
    
    
    
    def train_model(seg_model:CatSegModel, config:dict, 
                train_loader:torch.utils.data.DataLoader, valid_loader:torch.utils.data.DataLoader, 
                num_epochs):
    
    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #seg_model.to(device)
    
    # build optimizer
    lr = config.get('LR', 0.0001)
    optimizer = torch.optim.Adam(params=seg_model.parameters(), lr=lr)
    
    
    # build metrics
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]
    
    # create runners
    train_epoch = smputils.train.TrainEpoch(
        seg_model,
        loss=loss,
        optimizer=optimizer,
        metrics=metrics, 
        device=device,
        verbose=True
    )
    
    valid_epoch = smputils.train.ValidEpoch(
        seg_model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True
    )
    
    # train model for n epochs
    max_score = 0
    best_model = None

    # indices to plot
    sample_indices = np.random.randint(0, len(valid_loader.dataset), size = 1)
    
    all_train_logs = []
    all_valid_logs = []
    
    for i in range(0, num_epochs):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        all_train_logs.append(train_logs)
        all_valid_logs.append(valid_logs)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            best_model_path = save_model(seg_model, config, all_train_logs, all_valid_logs)
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
        
        # plot a few samples
        plot_for_epoch(seg_model, valid_loader.dataset, sample_indices, device)
    
    # save best model
    print(f"Best model is saved at: {best_model_path}")
    
    return train_logs, valid_logs, best_model
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



# Debug only

subset_size = len(train_dataset) // 10
indices = np.random.choice(len(train_dataset), subset_size, replace=False)
train_subset = torch.utils.data.Subset(train_dataset, indices)

subset_size = len(valid_dataset) // 10
indices = np.random.choice(len(valid_dataset), subset_size, replace=False)
valid_subset = torch.utils.data.Subset(valid_dataset, indices)

train_dl = DataLoader(train_subset, batch_size=6, shuffle=True)
valid_dl = DataLoader(valid_subset, batch_size=6, shuffle=True)

def load_model(model_path):
    checkpoint = torch.load(model_path)
    model = CatSegModel(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    train_logs = checkpoint['train_logs']
    valid_logs = checkpoint['valid_logs']
    
    return model, train_logs, valid_logs

def save_model(model, train_logs, valid_logs):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_name = f"{config['NAME']}_{timestamp}.pth"
    model_path = os.path.join('models', 'models_cache', model_name)

    save_data = {
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'train_logs': model.train_logs,
        'valid_logs': model.valid_logs
    }

    torch.save(save_data, model_path)
    return model_path

# Example transformations
# @TODO: what is the corect image size
data_transforms = {}
data_transforms['train'] = T.Compose([
    T.RandomResizedCrop(size=640, scale=(0.5, 1.1)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToImage(),
    T.RandomPerspective
    #T.ToDtype(torch.float32, scale=True)
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_transforms['valid'] = T.Compose([
    #T.Resize(670),
    T.CenterCrop(640),
    T.ToImage(),
    #T.ToDtype(torch.float32, scale=True)
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_transforms['test'] = data_transforms['valid']


# build datasets
total_size = len(full_dataset)
train_size = int(train_ratio * total_size)
valid_size = int(valid_ratio * total_size)
test_size = total_size - train_size - valid_size
train_dataset, valid_dataset, test_dataset = random_split(full_dataset, [train_size, valid_size, test_size])
train_dataset.dataset = copy(full_dataset) # disgusting solution for pytorch



    # transforms build
    transforms_config = config.get("TRANSFORMS", {})
    data_transforms = {}
    # build of each phase
    for phase, phase_transforms in transforms_config.items():
        transform_list = []
        for transform_name, params in phase_transforms.items():
            transform_class = getattr(T, transform_name)
            if params:
                transform_list.append(transform_class(**params))
            else:
                transform_list.append(transform_class())
        # display
        
        data_transforms[phase] = T.Compose(transform_list)
        
            # print the transforms
    for phase, transforms in config['TRANSFORMS'].items():
        df_transforms = pd.DataFrame(transforms.items())
        ipy_display(df_transforms)
    
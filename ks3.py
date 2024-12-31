def test_model(weights):
    # 1. Construct model
    tsm = ResNetTSM(pretrained=None,
                    layers=layers,
                    num_seg=num_seg)
    head = TSMHead(num_classes=num_classes,
                   in_channels=in_channels,
                   drop_ratio=drop_ratio)
    model = Recognizer2D(backbone=tsm, head=head)

    # 2. Construct dataset and dataloader.
    test_pipeline = Compose(train_mode=False)
    test_dataset = VideoDataset(file_path=valid_file_path,
                                pipeline=test_pipeline,
                                suffix=suffix)
    test_sampler = paddle.io.DistributedBatchSampler(test_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=valid_shuffle,
                                                     drop_last=True)
    test_loader = paddle.io.DataLoader(test_dataset,
                                       batch_sampler=test_sampler,
                                       places=paddle.set_device('gpu'),
                                       return_list=return_list)

    model.eval()

    state_dicts = paddle.load(weights)
    model.set_state_dict(state_dicts)

    # add params to metrics
    data_size = len(test_dataset)

    metric = CenterCropMetric(data_size=data_size, batch_size=batch_size)
    for batch_id, data in enumerate(test_loader):
        outputs = model.test_step(data)
        metric.update(batch_id, data, outputs)
    metric.accumulate()
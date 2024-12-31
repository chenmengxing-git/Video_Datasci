def inference(model_file):
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
                                                     batch_size=1,
                                                     shuffle=True,
                                                     drop_last=True)
    test_loader = paddle.io.DataLoader(test_dataset,
                                       batch_sampler=test_sampler,
                                       places=paddle.set_device('gpu'),
                                       return_list=return_list)

    model.eval()
    state_dicts = paddle.load(model_file)
    model.set_state_dict(state_dicts)

    for batch_id, data in enumerate(test_loader):
        _, labels = data
        outputs = model.test_step(data)
        scores = F.softmax(outputs)
        class_id = paddle.argmax(scores, axis=-1)
        pred = class_id.numpy()[0]
        label = labels.numpy()[0][0]

        print('真实类别：{}, 模型预测类别：{}'.format(pred, label))
        if batch_id > 5:
            break

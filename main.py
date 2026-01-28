import sys
sys.path.insert(0,'./')
import torch
import utils
import numpy as np
from args import Args
import matplotlib.pyplot as plt


module_dict = {
    'LSTM': {
        'module': 'model.LSTM',
        'class': 'LSTMPredictor'
    },
}
data_dict = {
    'RBRTEd.csv': {
        'module': 'dataProcessor.EuropeBrentSpotPriceFOB',
        'class': 'EuropeBrentSpotPriceFOB'
    },
}


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    args = Args().parse_args()  # 解析命令行参数
    print("<======= 参数如下 =======>")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    
        
    # 导入指定的模型和数据处理模块
    try:
        from importlib import import_module
        import os
        from dataProcessor.utils import get_dataloader
        
        model_info = module_dict[args.model_choice]
        data_info = data_dict[args.data_choice]

        model_module = import_module(model_info['module'])
        data_module = import_module(data_info['module'])

        ModelClass = getattr(model_module, model_info['class'])
        DataClass = getattr(data_module, data_info['class'])
        
        # 若传入的是纯文件名，则拼接到本项目的 data 目录
        data_path = args.data_choice
        if not os.path.isabs(data_path) and os.path.dirname(data_path) == '':
            data_path = os.path.join('oil-price-predictor\\data', data_path)
        
        # 创建数据集
        print("\n<======= 加载数据集 =======>")
        train_dataset = DataClass(
            path=data_path, 
            mode='train',
            look_back=args.look_back,
            test_size=args.test_size,
            val_size=args.val_size,  
            use_multifeature=False
        )
        val_dataset = DataClass(
            path=data_path, 
            mode='val',
            look_back=args.look_back,
            test_size=args.test_size,
            val_size=args.val_size,  
            use_multifeature=False
        )
        test_dataset = DataClass(
            path=data_path, 
            mode='test',
            look_back=args.look_back,
            test_size=args.test_size,
            val_size=args.val_size,  
            use_multifeature=False
        )

        print(f"训练集样本数: {len(train_dataset)}")
        print(f"验证集样本数: {len(val_dataset)}")
        print(f"测试集样本数: {len(test_dataset)}")
        print(f"输入维度: {train_dataset.X.shape[2]}")
        
        # 创建数据加载器
        train_loader = get_dataloader(train_dataset, args.batch_size, True, 0)
        val_loader = get_dataloader(val_dataset, args.batch_size, False, 0)
        test_loader = get_dataloader(test_dataset, args.batch_size, False, 0)
        
        # 初始化模型
        print("\n<======= 初始化模型 =======>")
        model = ModelClass(
            input_dim=train_dataset.X.shape[2],
            hidden_dim=64,
            num_layers=2,
            output_size=1,
            dropout=0.2
        ).to(device)
        print(model)
        args.mode = 'train'
        from model.LSTM import train_model
        print("\n<======= 开始训练 =======>")
        train_losses, val_losses, outputs, val_outputs = train_model(
            model, 
            train_loader, 
            test_loader, 
            num_epochs=100, 
            learning_rate=0.001
        )
        # 绘制损失曲线
        utils.plot_losses(train_losses, val_losses)
        # 保存模型
        utils.save_model(model, 'best_lstm_model.pth')
        

        args.mode = 'test'
        # 加载预训练模型
        if os.path.exists('best_lstm_model.pth'):
            print("加载预训练模型...")
            model = utils.load_model(ModelClass, 'best_lstm_model.pth', device)
        else:
            print("警告: 未找到预训练模型，使用随机初始化模型")
        

        # 预测未来多步
        print("\n<======= 未来预测 =======>")
        from model.LSTM import predict
        # 使用最近的数据进行预测
        recent_data = test_dataset.X[-1].reshape(-1, 1)  # 获取最后一个序列
        future_predictions = predict(
            model, 
            recent_data, 
            test_dataset.target_scaler,
            sequence_length=60
        )
        print(f"预测: {future_predictions}")

        # 反归一化并绘图
        def to_numpy_2d(preds):
            import numpy as np
            import torch
            if isinstance(preds, list):
                if len(preds) == 0:
                    return np.empty((0, 1), dtype=np.float32)
                # list 里可能是 Tensor/ndarray/标量，统一拉平后拼接
                flat = []
                for p in preds:
                    if torch.is_tensor(p):
                        flat.append(p.detach().cpu().view(-1).numpy())
                    else:
                        flat.append(np.array(p, dtype=np.float32).reshape(-1))
                return np.concatenate(flat).reshape(-1, 1)
            elif torch.is_tensor(preds):
                return preds.detach().cpu().view(-1, 1).numpy()
            else:
                return np.array(preds, dtype=np.float32).reshape(-1, 1)

        train_targets = train_dataset.target_scaler.inverse_transform(train_dataset.y.reshape(-1, 1)).flatten()
        train_preds = train_dataset.target_scaler.inverse_transform(to_numpy_2d(outputs)).flatten()
        val_targets = val_dataset.target_scaler.inverse_transform(val_dataset.y.reshape(-1, 1)).flatten()
        val_preds = val_dataset.target_scaler.inverse_transform(to_numpy_2d(val_outputs)).flatten()
        
        utils.plot_predictions(train_targets, train_preds, val_targets, val_preds,
                              future_predictions=future_predictions,
                       title="未来价格预测对比")

        utils.evaluate_model(
            model, 
            test_loader, 
            test_dataset.target_scaler,
            device
        )

        

        
    except Exception as e:
        import traceback
        print(f"\n错误: {e}")
        traceback.print_exc()
        raise

    
    
if __name__ == "__main__":
    main()
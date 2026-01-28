from argparse import Namespace
from tap import Tap


class Args(Tap):
    """ <=== 该部分书写不可变参数 ===> """
    
    
    
    
    """ <=== 该部分书写不可变参数 End ===> """
    
    
    
    """ <=== 该部分书写可变参数 ===> """
    # 数据参数
    data_choice: str = 'RBRTEd.csv'
    look_back: int = 60
    test_size: float = 0.05
    val_size: float = 0.05
    use_multifeature: bool = False
    
    # 模型参数
    model_choice: str = 'LSTM'
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    
    # 训练参数
    mode: str = 'train'
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 0.001
    
    
    
    """ <=== 该部分书写可变参数 End ===> """
    
    
    
    def configure(self):
        # 添加命令行参数，允许用户显式指定模型和数据
        # 添加所有参数
        self.add_argument(
            "--look_back", 
            type=int, 
            default=60
        )
        self.add_argument(
            "--test_size", 
            type=float, 
            default=0.05
        )
        self.add_argument(
            "--val_size", 
            type=float, 
            default=0.05
        )
        self.add_argument(
            "--hidden_dim", 
            type=int, 
            default=64
        )
        self.add_argument(
            "--num_epochs", 
            type=int, 
            default=100
        )
        self.add_argument(
            "--model_choice",
            type=str,
            default='LSTM',
            help="选择使用的模型: LSTM (默认: LSTM)"
        )
        self.add_argument(
            "--data_choice",
            type=str,
            default='RBRTEd.csv',
            help="选择使用的数据文件: RBRTEd.csv (默认: RBRTEd.csv)"
        )
        self.add_argument(
            "--batch_size",
            type=int,
            default=16,
            help="批处理大小 (默认: 16)"
        )
        self.add_argument(
            "--mode",
            type=str,
            choices=['train', 'test'],
            default='train',
            help="运行模式: train 或 test (默认: train)"
        )
        
        
if __name__ == "__main__":
    args = Args().parse_args()
    
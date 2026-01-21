from argparse import Namespace
from tap import TAP

class Args(TAP):
    """ <=== 该部分书写不可变参数 ===> """
    
    
    
    
    """ <=== 该部分书写不可变参数 End ===> """
    
    
    
    """ <=== 该部分书写可变参数 ===> """
    
    batch_size:int = 16
    data_choice = 'RBRTEd.csv'
    model_choice = 'LSTM'
    mode = 'train'
    
    
    
    """ <=== 该部分书写可变参数 End ===> """
    
    
    
    def normal_init(self,):
        self.add_argument("mode",type='str',choices=['train',])
        
        
if __name__ == "__main__":
    args = Args()
    args.normal_init()
    
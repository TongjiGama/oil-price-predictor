import sys
sys.path.insert(0,'./')

from args import Args




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
    args = Args()
    print("<======= 参数如下 =======>")
    # for key, value in vars(args).items():
    #     print(f"{key}: {value}")
        
    # 导入指定的模型和数据处理模块
    try:
        from importlib import import_module
        # model_module = import_module(module_dict[args.model])
        # data_module = import_module(data_dict[args.data])
    except:
        raise ImportError("无法导入指定的模型或数据处理模块，请检查参数是否正确。")

    
    
if __name__ == "__main__":
    main()
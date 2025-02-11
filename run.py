import argparse
import time

import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='iTransformer')

    # ablation control flags
    parser.add_argument('--revin', action='store_false', help='non-stationary for short-term', default=True)
    parser.add_argument('--alpha', type=float, default=0.2, help='factor of frequency loss')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--attn_dropout', type=float, default=0.15, help='dropout')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--wo_time', action='store_true', help='dont use timestamp', default=False)
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--t_layers', type=int, default=1, help='num of temporal layers')
    parser.add_argument('--stable_len', type=int, default=6, help='num of temporal layers')
    parser.add_argument('--use_group', action='store_true', default=False)
    parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument('--router', type=int, default=None, help='num of router')
    parser.add_argument('--zero', type=str, default=None, help='zero of ac/dc')
    parser.add_argument('--attn_type', type=int, default=None, help='type of attention mode. 0 channel 1 segment')
    parser.add_argument('--temporal', action='store_true', default=False)
    # parser.add_argument('--alpha', type=float, default=0.1, help='frequency save of original ac')
    parser.add_argument('--Asym', action='store_false', help='use Asymmetric self-attention', default=True)
    parser.add_argument('--kernel', type=int, default=25, help='size of window, 7 12 24 36 ...')
    parser.add_argument('--wavelet', type=str, default='coif3',  help='the wavelet use')
    parser.add_argument('--layer_norm', action='store_false', default=True)
    parser.add_argument('--group', type=int, default=None, help='num of group')
    parser.add_argument('--num_p', type=int, default=None, help='num of kernel')
    parser.add_argument('--period', type=int, default=24, help='num of kernel')
    parser.add_argument('--ratio', type=int, default=1, help='times of num_p')
    parser.add_argument('--periods', type=int, nargs='+', default=None, help='num of kernel')

    parser.add_argument('--enc_in', type=int, default=7, help='channel_decoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='channel_decoder input size')
    parser.add_argument('--c_in', type=int, default=None, help='input size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--sz_row', type=int, default=None, help='input size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=0, help='num of fc1 layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of channel_decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--k_segments', type=int, default=1, help='number of segments')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in fc1, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--embedding_epochs', type=int, default=5, help='train epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--pct_start', type=float, default=0.2, help='optimizer learning rate')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--embedding_lr', type=float, default=0.0005, help='optimizer learning rate of embedding')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='None',
                        help='experiment name, options:[station_train, partial_train, zero_shot]')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.exp_name == 'partial_train':
        Exp = Exp_Long_Term_Forecast_Partial
    elif args.exp_name == 'station_train':
        Exp = Exp_Long_Term_Forecast_Station
    else:
        Exp = Exp_Long_Term_Forecast


    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        start_time = time.time()
        exp.test(setting, test=1)
        end_time = time.time()
        print(f"运行时间: {end_time - start_time:.4f} 秒")
        torch.cuda.empty_cache()

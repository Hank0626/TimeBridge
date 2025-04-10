if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/TimeBridge" ]; then
    mkdir ./logs/LongForecasting/TimeBridge
fi

model_name=TimeBridge
seq_len=720
GPU=0,1
root=./dataset

alpha=0.35
data_name=ETTh1
for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/ETT-small/ \
    --data_path $data_name.csv \
    --model_id $data_name_$seq_len_$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_in 7 \
    --c_out 7 \
    --e_layers 0 \
    --t_layer 3 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --batch_size 64 \
    --alpha $alpha \
    --learning_rate 0.0002 \
    --train_epochs 100 \
    --patience 10 \
    --itr 1 >logs/LongForecasting/TimeBridge/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done


alpha=0.35
data_name=ETTm1
for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/ETT-small/ \
    --data_path $data_name.csv \
    --model_id $data_name_$seq_len_$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_in 7 \
    --c_out 7 \
    --t_layer 3 \
    --e_layers 0 \
    --des 'Exp' \
    --n_heads 4 \
    --d_model 64 \
    --d_ff 128 \
    --period 48 \
    --num_p 6 \
    --lradj 'TST' \
    --learning_rate 0.0002 \
    --train_epochs 100 \
    --pct_start 0.2 \
    --patience 15 \
    --batch_size 64 \
    --alpha $alpha \
    --itr 1 >logs/LongForecasting/TimeBridge/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done

alpha=0.35
data_name=ETTm2
for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/ETT-small/ \
    --data_path $data_name.csv \
    --model_id $data_name_$seq_len_$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_in 7 \
    --c_out 7 \
    --e_layers 0 \
    --t_layers 3 \
    --des 'Exp' \
    --n_heads 4 \
    --d_model 64  \
    --d_ff 128 \
    --lradj 'TST' \
    --period 48 \
    --train_epochs 100 \
    --learning_rate 0.0002 \
    --pct_start 0.2 \
    --patience 10 \
    --batch_size 64 \
    --alpha $alpha \
    --itr 1 >logs/LongForecasting/TimeBridge/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done

alpha=0.35
data_name=ETTh2
for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/ETT-small/ \
    --data_path $data_name.csv \
    --model_id $data_name_$seq_len_$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_in 7 \
    --c_out 7 \
    --period 48 \
    --t_layer 3 \
    --e_layers 0 \
    --des 'Exp' \
    --n_heads 4 \
    --period 48 \
    --d_model 128 \
    --d_ff 128 \
    --train_epochs 100 \
    --learning_rate 0.0001 \
    --patience 15 \
    --alpha $alpha \
    --batch_size 16 \
    --itr 1 >logs/LongForecasting/TimeBridge/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done

alpha=0.1
data_name=weather
for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/weather/ \
    --data_path weather.csv \
    --model_id weather_$seq_len_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 21 \
    --dec_in 21 \
    --c_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --period 48 \
    --num_p 12 \
    --d_model 128 \
    --d_ff 128 \
    --alpha $alpha \
    --itr 1 >logs/LongForecasting/TimeBridge/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done

alpha=0.05
data_name=Solar
for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/Solar/ \
    --data_path solar_AL.txt \
    --model_id solar_96_96 \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 137 \
    --dec_in 137 \
    --c_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --period 48 \
    --d_model 128 \
    --d_ff 128 \
    --alpha $alpha \
    --learning_rate 0.0005 \
    --train_epochs 100 \
    --patience 15 \
    --itr 1 >logs/LongForecasting/TimeBridge/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done

alpha=0.2
data_name=electricity
for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/electricity/ \
    --data_path electricity.csv \
    --model_id electricity_$seq_len_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 321 \
    --dec_in 321 \
    --c_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --n_heads 32 \
    --d_ff 512 \
    --d_model 512 \
    --e_layers 2 \
    --attn_dropout 0.1 \
    --num_p 4 \
    --stable_len 4 \
    --alpha $alpha \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --itr 1 >logs/LongForecasting/TimeBridge/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done

alpha=0.35
data_name=traffic
for pred_len in 336 192 720 96; do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_$seq_len_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 862 \
    --dec_in 862 \
    --c_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --num_p 8 \
    --n_heads 64 \
    --stable_len 2 \
    --d_ff 512 \
    --d_model 512 \
    --e_layers 3 \
    --batch_size 16 \
    --attn_dropout 0.15 \
    --patience 5 \
    --train_epochs 100 \
    --devices 0,1 \
    --use_multi_gpu \
    --alpha $alpha \
    --learning_rate 0.001 \
    --itr 1 >logs/LongForecasting/TimeBridge/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done

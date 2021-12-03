for WINDOW_SIZE in 8 16 32 64 128 256 512 1024 2048 4096
do
    python train.py --model RNN --dataset CAESAR --window_size $WINDOW_SIZE --num_epochs 20 
done
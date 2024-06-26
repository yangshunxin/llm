python main.py \
                            e_GRU   \
                            d_GRU   \
                            -num_epoch 100  \
                            -batch_size 256  \
                            -lr 0.0001  \
                            -train 0    \
                            -teacher_forcing_ratio  0.5     \
                            -hidden_size 768
python main.py \
                            e_GRU   \
                            d_GRU   \
                            -num_epoch 100  \
                            -batch_size 256  \
                            -lr 0.0001  \
                            -train 1    \
                            -teacher_forcing_ratio  0.5     \
                            -hidden_size 768
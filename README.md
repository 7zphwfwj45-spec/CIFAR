“This project was developed as an educational exercise to understand CNNs from first principles.
Discussions with ChatGPT were instrumental in debugging, architectural reasoning, and experimental design.”

The training rate was 88% with a 75% test rate. Performed on an M4 Mac-Mini with 16GB

configuration:

No dropout, LR = 0.001 and 0.0003 after 75 epochs;Weight Decay 5e-4, Conv Layers 32,32, pool, 64,64, pool, 128,128  pool.  Batch Normalization on each conv layer,  Augmentation crop and horizontal Flip. Global Averaging Pooling. dense layer 10, Batch Size 64, 100 Epochs.

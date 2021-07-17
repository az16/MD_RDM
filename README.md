# MD_RDM
This is a pytorch implementation of 'Monocular Depth Estimation Using Relative Depth Maps'

* (tested) - element has been tested in conjunction with other network parts
* (tested on its own) - element has been tested by manually feeding input directly into it but not in conjuction with the rest of the network

# Network
- [x] Encoder structure (tested)
- [x] Decoder DenseBlocks (tested)
- [x] Decoder WSM-Blocks (tested)
- [x] Decoder DORN-Blocks (tested)
- [x] Relative depth pair estimation (tested)
- [x] Lloyd Quantization of depth pairs (tested)
- [x] ALS depth map reconstruction (tested)
- [x] Network propagation up to batch size 16 (tested) 
# Finalizing Output
- [x] Depth Map Decomposition (tested)
- [x] Weight optimization for fine detail maps (tested)
- [x] Decomposition of ground truth while propagating input through network (tested)
- [x] optimization of dorn decoder components (tested)
- [x] Optimal Map Reconstruction (tested)
# Training and Data
- [x] Training cycle
- [x] Dataloaders + Augmentation
- [x] Training (in progress)
# Debugging and current Issues
## Current Issues
--
## Current Training Progress
Configurations:
- [x] Encoder + D3                                        (lr = 1-e4, b = 4, epochs = 15)              Result: d1 = 0.6 
- [ ] Encoder (frozen weights) + D3 + D6                  (lr = 1-e4, b = 4, epochs = 15)
- [ ] Encoder (frozen weights) + D3 + D6 + D7             (lr = 1-e4, b = 4, epochs = 15)
- [ ] Encoder (frozen weights) + D3 + D6 + D7 + D8        (lr = 1-e4, b = 4, epochs = 15)
- [ ] Encoder (frozen weights) + D3 + D6 + D7 + D8 + D9   (lr = 1-e4, b = 4, epochs = 15)

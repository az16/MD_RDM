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
- [x] Weight optimization for fine detail maps
- [ ] Decomposition of ground truth while propagating input through network
- [ ] optimization of dorn decoder components 
- [x] Optimal Map Reconstruction (tested on its own)
# Training and Data
- [x] Training cycle
- [x] Dataloaders + Augmentation
- [ ] Training

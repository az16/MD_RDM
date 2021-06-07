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
- [x] Decomposition of ground truth while propagating input through network
- [x] optimization of dorn decoder components 
- [x] Optimal Map Reconstruction (tested on its own)
# Training and Data
- [x] Training cycle
- [x] Dataloaders + Augmentation
- [ ] Training
# Debugging and current Issues
## Debugging
- [x] Embedded geometric mean into resize method (before only used regular pytorch resize)
- [x] Eigenvector calculation: torch.lobgc method takes symmetric matrices as input --> our input is a reciprocal matrix so torch.eig is now used
- [x] DORN loss adjusted to fit our depth maps
## Current Issues
- [ ] Encoder uses conv layers --> cause negative values in tensor due to negative kernel weights --> influences the ordinal layers since they expect positive inputs   
- [ ] torch.eig returns complex eigenvalues --> backward method does not support complex tensor types yet

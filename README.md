# G3FA: Geometry-Guided GAN for Face Animation (BMVC 2024)


This is the official repository for **G3FA: Geometry-Guided GAN for Face Animation**, presented at BMVC 2024. 
## Overview
Animating a human face image from a single source frame involves generating a natural-looking, identity-consistent representation that mimics the movements of a driving video. While Generative Adversarial Networks (GANs) have demonstrated promising results in real-time face reenactment, they often lack the geometric consistency provided by graphics-based methods.

**G3FA** bridges this gap by integrating 3D geometric information derived from 2D images into a GAN-based face animation framework. Our method utilizes inverse rendering techniques to extract 3D facial geometry properties, enhancing the generator’s output through a weighted ensemble of discriminators. By combining 2D motion warping with volumetric rendering, G3FA captures intricate motion dynamics, producing high-quality, geometrically consistent animations.

### Key Features
- Incorporates 3D facial geometry using inverse rendering.
- Leverages a weighted ensemble of discriminators for enhanced feedback.
- Combines 2D motion warping and volumetric rendering for detailed motion dynamics.
- Achieves state-of-the-art performance on VoxCeleb2 and TalkingHead benchmarks.


[paper](https://www.dfki.de/fileadmin/user_upload/import/15124_0657.pdf)

![Pipeline](https://github.com/user-attachments/assets/9d9bf31e-7582-4c2c-8fe4-dcf56c9049eb)

## Getting Started

### Prerequisites
You will need Python 3.8 or later.

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/dfki-av/G3FA.git
   cd G3FA
   ```

2. **Set Up the Conda Environment**
   ```bash
   conda create -n g3fa_env python=3.10 -y
   conda activate g3fa_env
   ```

3. **Install Required Packages**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   ```

4. **Download Pre-Trained Checkpoints**
[Checkpoint](https://drive.google.com/uc?id=1hBy6r00HGgRtyGaeC0OsNXZkmyT82tFT)
   

### Running the Demo

To run the live demo, run the following command:

```bash
python live_demo.py --source_image path/to/source.png \
               --checkpoint checkpoints/g3fa.pt \
```

## To-Do
- Including training scripts
- Adding evaluation scripts 


## Acknowledgements
This repository builds upon the following works:
- [First Order Motion Model for Image Animation](https://github.com/AliaksandrSiarohin/first-order-model)
- [FNeVR](https://github.com/zengbohan0217/FNeVR)




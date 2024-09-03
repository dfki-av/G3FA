# G3FA
This is the official code for BMVC 2024 paper, G3FA: Geometry-guided GAN for Face Animation.

Abstract:
Animating human face images aims to synthesize a desired source identity in a
natural-looking way mimicking a driving video’s facial movements. In this context,
Generative Adversarial Networks have demonstrated remarkable potential in real-time
face reenactment using a single source image, yet are constrained by limited geometry
consistency compared to graphic-based approaches. In this paper, we introduce
Geometry-guided GAN for Face Animation (G3FA) to tackle this limitation. Our
novel approach empowers the face animation model to incorporate 3D information
using only 2D images, improving the image generation capabilities of the talking
head synthesis model. We integrate inverse rendering techniques to extract 3D facial
geometry properties, improving the feedback loop to the generator through a weighted
average ensemble of discriminators. In our face reenactment model, we leverage 2D
motion warping to capture motion dynamics along with orthogonal ray sampling and
volume rendering techniques to produce the ultimate visual output. To evaluate the
performance of our G3FA, we conducted comprehensive experiments using various
evaluation protocols on VoxCeleb2 and TalkingHead benchmarks to demonstrate the
effectiveness of our proposed framework compared to the state-of-the-art real-time face
animation methods
![pipe3_new-1](https://github.com/user-attachments/assets/9d9bf31e-7582-4c2c-8fe4-dcf56c9049eb)

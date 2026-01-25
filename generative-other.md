# General Generative Models - Other
**update at 2026-01-25 10:36:50**

Sorted by classifier confidence (high to low).

## **1. Safeguarding Facial Identity against Diffusion-based Face Swapping via Cascading Pathway Disruption**

cs.CV

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.14738v1) [paper-pdf](https://arxiv.org/pdf/2601.14738v1)

**Confidence**: 0.95

**Authors**: Liqin Wang, Qianyue Hu, Wei Lu, Xiangyang Luo

**Abstract**: The rapid evolution of diffusion models has democratized face swapping but also raises concerns about privacy and identity security. Existing proactive defenses, often adapted from image editing attacks, prove ineffective in this context. We attribute this failure to an oversight of the structural resilience and the unique static conditional guidance mechanism inherent in face swapping systems. To address this, we propose VoidFace, a systemic defense method that views face swapping as a coupled identity pathway. By injecting perturbations at critical bottlenecks, VoidFace induces cascading disruption throughout the pipeline. Specifically, we first introduce localization disruption and identity erasure to degrade physical regression and semantic embeddings, thereby impairing the accurate modeling of the source face. We then intervene in the generative domain by decoupling attention mechanisms to sever identity injection, and corrupting intermediate diffusion features to prevent the reconstruction of source identity. To ensure visual imperceptibility, we perform adversarial search in the latent manifold, guided by a perceptual adaptive strategy to balance attack potency with image quality. Extensive experiments show that VoidFace outperforms existing defenses across various diffusion-based swapping models, while producing adversarial faces with superior visual quality.



## **2. PhaseMark: A Post-hoc, Optimization-Free Watermarking of AI-generated Images in the Latent Frequency Domain**

cs.CV

Accepted to the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2026

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2601.13128v1) [paper-pdf](https://arxiv.org/pdf/2601.13128v1)

**Confidence**: 0.95

**Authors**: Sung Ju Lee, Nam Ik Cho

**Abstract**: The proliferation of hyper-realistic images from Latent Diffusion Models (LDMs) demands robust watermarking, yet existing post-hoc methods are prohibitively slow due to iterative optimization or inversion processes. We introduce PhaseMark, a single-shot, optimization-free framework that directly modulates the phase in the VAE latent frequency domain. This approach makes PhaseMark thousands of times faster than optimization-based techniques while achieving state-of-the-art resilience against severe attacks, including regeneration, without degrading image quality. We analyze four modulation variants, revealing a clear performance-quality trade-off. PhaseMark demonstrates a new paradigm where efficient, resilient watermarking is achieved by exploiting intrinsic latent properties.



## **3. Semantic Mismatch and Perceptual Degradation: A New Perspective on Image Editing Immunity**

cs.CV

11 pages, 4 figures

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14320v1) [paper-pdf](https://arxiv.org/pdf/2512.14320v1)

**Confidence**: 0.95

**Authors**: Shuai Dong, Jie Zhang, Guoying Zhao, Shiguang Shan, Xilin Chen

**Abstract**: Text-guided image editing via diffusion models, while powerful, raises significant concerns about misuse, motivating efforts to immunize images against unauthorized edits using imperceptible perturbations. Prevailing metrics for evaluating immunization success typically rely on measuring the visual dissimilarity between the output generated from a protected image and a reference output generated from the unprotected original. This approach fundamentally overlooks the core requirement of image immunization, which is to disrupt semantic alignment with attacker intent, regardless of deviation from any specific output. We argue that immunization success should instead be defined by the edited output either semantically mismatching the prompt or suffering substantial perceptual degradations, both of which thwart malicious intent. To operationalize this principle, we propose Synergistic Intermediate Feature Manipulation (SIFM), a method that strategically perturbs intermediate diffusion features through dual synergistic objectives: (1) maximizing feature divergence from the original edit trajectory to disrupt semantic alignment with the expected edit, and (2) minimizing feature norms to induce perceptual degradations. Furthermore, we introduce the Immunization Success Rate (ISR), a novel metric designed to rigorously quantify true immunization efficacy for the first time. ISR quantifies the proportion of edits where immunization induces either semantic failure relative to the prompt or significant perceptual degradations, assessed via Multimodal Large Language Models (MLLMs). Extensive experiments show our SIFM achieves the state-of-the-art performance for safeguarding visual content against malicious diffusion-based manipulation.



## **4. GeMM-GAN: A Multimodal Generative Model Conditioned on Histopathology Images and Clinical Descriptions for Gene Expression Profile Generation**

cs.AI

12 pages, 2 figures. Published at Image Analysis and Processing - ICIAP 2025 Workshops

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.15392v1) [paper-pdf](https://arxiv.org/pdf/2601.15392v1)

**Confidence**: 0.95

**Authors**: Francesca Pia Panaccione, Carlo Sgaravatti, Pietro Pinoli

**Abstract**: Biomedical research increasingly relies on integrating diverse data modalities, including gene expression profiles, medical images, and clinical metadata. While medical images and clinical metadata are routinely collected in clinical practice, gene expression data presents unique challenges for widespread research use, mainly due to stringent privacy regulations and costly laboratory experiments. To address these limitations, we present GeMM-GAN, a novel Generative Adversarial Network conditioned on histopathology tissue slides and clinical metadata, designed to synthesize realistic gene expression profiles. GeMM-GAN combines a Transformer Encoder for image patches with a final Cross Attention mechanism between patches and text tokens, producing a conditioning vector to guide a generative model in generating biologically coherent gene expression profiles. We evaluate our approach on the TCGA dataset and demonstrate that our framework outperforms standard generative models and generates more realistic and functionally meaningful gene expression profiles, improving by more than 11\% the accuracy on downstream disease type prediction compared to current state-of-the-art generative models. Code will be available at: https://github.com/francescapia/GeMM-GAN



## **5. Universal Adversarial Purification with DDIM Metric Loss for Stable Diffusion**

cs.CV

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07253v1) [paper-pdf](https://arxiv.org/pdf/2601.07253v1)

**Confidence**: 0.95

**Authors**: Li Zheng, Liangbin Xie, Jiantao Zhou, He YiMin

**Abstract**: Stable Diffusion (SD) often produces degraded outputs when the training dataset contains adversarial noise. Adversarial purification offers a promising solution by removing adversarial noise from contaminated data. However, existing purification methods are primarily designed for classification tasks and fail to address SD-specific adversarial strategies, such as attacks targeting the VAE encoder, UNet denoiser, or both. To address the gap in SD security, we propose Universal Diffusion Adversarial Purification (UDAP), a novel framework tailored for defending adversarial attacks targeting SD models. UDAP leverages the distinct reconstruction behaviors of clean and adversarial images during Denoising Diffusion Implicit Models (DDIM) inversion to optimize the purification process. By minimizing the DDIM metric loss, UDAP can effectively remove adversarial noise. Additionally, we introduce a dynamic epoch adjustment strategy that adapts optimization iterations based on reconstruction errors, significantly improving efficiency without sacrificing purification quality. Experiments demonstrate UDAP's robustness against diverse adversarial methods, including PID (VAE-targeted), Anti-DreamBooth (UNet-targeted), MIST (hybrid), and robustness-enhanced variants like Anti-Diffusion (Anti-DF) and MetaCloak. UDAP also generalizes well across SD versions and text prompts, showcasing its practical applicability in real-world scenarios.



## **6. Attack-Resistant Watermarking for AIGC Image Forensics via Diffusion-based Semantic Deflection**

cs.CR

**SubmitDate**: 2026-01-10    [abs](http://arxiv.org/abs/2601.06639v1) [paper-pdf](https://arxiv.org/pdf/2601.06639v1)

**Confidence**: 0.95

**Authors**: Qingyu Liu, Yitao Zhang, Zhongjie Ba, Chao Shuai, Peng Cheng, Tianhang Zheng, Zhibo Wang

**Abstract**: Protecting the copyright of user-generated AI images is an emerging challenge as AIGC becomes pervasive in creative workflows. Existing watermarking methods (1) remain vulnerable to real-world adversarial threats, often forced to trade off between defenses against spoofing and removal attacks; and (2) cannot support semantic-level tamper localization. We introduce PAI, a training-free inherent watermarking framework for AIGC copyright protection, plug-and-play with diffusion-based AIGC services. PAI simultaneously provides three key functionalities: robust ownership verification, attack detection, and semantic-level tampering localization. Unlike existing inherent watermark methods that only embed watermarks at noise initialization of diffusion models, we design a novel key-conditioned deflection mechanism that subtly steers the denoising trajectory according to the user key. Such trajectory-level coupling further strengthens the semantic entanglement of identity and content, thereby further enhancing robustness against real-world threats. Moreover, we also provide a theoretical analysis proving that only the valid key can pass verification. Experiments across 12 attack methods show that PAI achieves 98.43\% verification accuracy, improving over SOTA methods by 37.25\% on average, and retains strong tampering localization performance even against advanced AIGC edits. Our code is available at https://github.com/QingyuLiu/PAI.



## **7. Training-Free Color-Aware Adversarial Diffusion Sanitization for Diffusion Stegomalware Defense at Security Gateways**

cs.CR

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.24499v1) [paper-pdf](https://arxiv.org/pdf/2512.24499v1)

**Confidence**: 0.95

**Authors**: Vladimir Frants, Sos Agaian

**Abstract**: The rapid expansion of generative AI has normalized large-scale synthetic media creation, enabling new forms of covert communication. Recent generative steganography methods, particularly those based on diffusion models, can embed high-capacity payloads without fine-tuning or auxiliary decoders, creating significant challenges for detection and remediation. Coverless diffusion-based techniques are difficult to counter because they generate image carriers directly from secret data, enabling attackers to deliver stegomalware for command-and-control, payload staging, and data exfiltration while bypassing detectors that rely on cover-stego discrepancies. This work introduces Adversarial Diffusion Sanitization (ADS), a training-free defense for security gateways that neutralizes hidden payloads rather than detecting them. ADS employs an off-the-shelf pretrained denoiser as a differentiable proxy for diffusion-based decoders and incorporates a color-aware, quaternion-coupled update rule to reduce artifacts under strict distortion limits. Under a practical threat model and in evaluation against the state-of-the-art diffusion steganography method Pulsar, ADS drives decoder success rates to near zero with minimal perceptual impact. Results demonstrate that ADS provides a favorable security-utility trade-off compared to standard content transformations, offering an effective mitigation strategy against diffusion-driven steganography.



## **8. Smark: A Watermark for Text-to-Speech Diffusion Models via Discrete Wavelet Transform**

cs.SD

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18791v1) [paper-pdf](https://arxiv.org/pdf/2512.18791v1)

**Confidence**: 0.95

**Authors**: Yichuan Zhang, Chengxin Li, Yujie Gu

**Abstract**: Text-to-Speech (TTS) diffusion models generate high-quality speech, which raises challenges for the model intellectual property protection and speech tracing for legal use. Audio watermarking is a promising solution. However, due to the structural differences among various TTS diffusion models, existing watermarking methods are often designed for a specific model and degrade audio quality, which limits their practical applicability. To address this dilemma, this paper proposes a universal watermarking scheme for TTS diffusion models, termed Smark. This is achieved by designing a lightweight watermark embedding framework that operates in the common reverse diffusion paradigm shared by all TTS diffusion models. To mitigate the impact on audio quality, Smark utilizes the discrete wavelet transform (DWT) to embed watermarks into the relatively stable low-frequency regions of the audio, which ensures seamless watermark-audio integration and is resistant to removal during the reverse diffusion process. Extensive experiments are conducted to evaluate the audio quality and watermark performance in various simulated real-world attack scenarios. The experimental results show that Smark achieves superior performance in both audio quality and watermark extraction accuracy.



## **9. Towards Transferable Defense Against Malicious Image Edits**

cs.CV

14 pages, 5 figures

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14341v1) [paper-pdf](https://arxiv.org/pdf/2512.14341v1)

**Confidence**: 0.95

**Authors**: Jie Zhang, Shuai Dong, Shiguang Shan, Xilin Chen

**Abstract**: Recent approaches employing imperceptible perturbations in input images have demonstrated promising potential to counter malicious manipulations in diffusion-based image editing systems. However, existing methods suffer from limited transferability in cross-model evaluations. To address this, we propose Transferable Defense Against Malicious Image Edits (TDAE), a novel bimodal framework that enhances image immunity against malicious edits through coordinated image-text optimization. Specifically, at the visual defense level, we introduce FlatGrad Defense Mechanism (FDM), which incorporates gradient regularization into the adversarial objective. By explicitly steering the perturbations toward flat minima, FDM amplifies immune robustness against unseen editing models. For textual enhancement protection, we propose an adversarial optimization paradigm named Dynamic Prompt Defense (DPD), which periodically refines text embeddings to align the editing outcomes of immunized images with those of the original images, then updates the images under optimized embeddings. Through iterative adversarial updates to diverse embeddings, DPD enforces the generation of immunized images that seek a broader set of immunity-enhancing features, thereby achieving cross-model transferability. Extensive experimental results demonstrate that our TDAE achieves state-of-the-art performance in mitigating malicious edits under both intra- and cross-model evaluations.



## **10. MFE-GAN: Efficient GAN-based Framework for Document Image Enhancement and Binarization with Multi-scale Feature Extraction**

cs.CV

Extended Journal Version of APSIPA ASC 2025

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14114v1) [paper-pdf](https://arxiv.org/pdf/2512.14114v1)

**Confidence**: 0.95

**Authors**: Rui-Yang Ju, KokSheik Wong, Yanlin Jin, Jen-Shiun Chiang

**Abstract**: Document image enhancement and binarization are commonly performed prior to document analysis and recognition tasks for improving the efficiency and accuracy of optical character recognition (OCR) systems. This is because directly recognizing text in degraded documents, particularly in color images, often results in unsatisfactory recognition performance. To address these issues, existing methods train independent generative adversarial networks (GANs) for different color channels to remove shadows and noise, which, in turn, facilitates efficient text information extraction. However, deploying multiple GANs results in long training and inference times. To reduce both training and inference times of document image enhancement and binarization models, we propose MFE-GAN, an efficient GAN-based framework with multi-scale feature extraction (MFE), which incorporates Haar wavelet transformation (HWT) and normalization to process document images before feeding them into GANs for training. In addition, we present novel generators, discriminators, and loss functions to improve the model's performance, and we conduct ablation studies to demonstrate their effectiveness. Experimental results on the Benchmark, Nabuco, and CMATERdb datasets demonstrate that the proposed MFE-GAN significantly reduces the total training and inference times while maintaining comparable performance with respect to state-of-the-art (SOTA) methods. The implementation of this work is available at https://ruiyangju.github.io/MFE-GAN.



## **11. Beyond Memorization: Gradient Projection Enables Selective Learning in Diffusion Models**

cs.LG

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.11194v1) [paper-pdf](https://arxiv.org/pdf/2512.11194v1)

**Confidence**: 0.95

**Authors**: Divya Kothandaraman, Jaclyn Pytlarz

**Abstract**: Memorization in large-scale text-to-image diffusion models poses significant security and intellectual property risks, enabling adversarial attribute extraction and the unauthorized reproduction of sensitive or proprietary features. While conventional dememorization techniques, such as regularization and data filtering, limit overfitting to specific training examples, they fail to systematically prevent the internalization of prohibited concept-level features. Simply discarding all images containing a sensitive feature wastes invaluable training data, necessitating a method for selective unlearning at the concept level.   To address this, we introduce a Gradient Projection Framework designed to enforce a stringent requirement of concept-level feature exclusion. Our defense operates during backpropagation by systematically identifying and excising training signals aligned with embeddings of prohibited attributes. Specifically, we project each gradient update onto the orthogonal complement of the sensitive feature's embedding space, thereby zeroing out its influence on the model's weights. Our method integrates seamlessly into standard diffusion model training pipelines and complements existing defenses. We analyze our method against an adversary aiming for feature extraction. In extensive experiments, we demonstrate that our framework drastically reduces memorization while rigorously preserving generation quality and semantic fidelity. By reframing memorization control as selective learning, our approach establishes a new paradigm for IP-safe and privacy-preserving generative AI.



## **12. Interpreting Structured Perturbations in Image Protection Methods for Diffusion Models**

cs.CV

32 pages, 17 figures, 1 table, 5 algorithms, preprint

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08329v1) [paper-pdf](https://arxiv.org/pdf/2512.08329v1)

**Confidence**: 0.95

**Authors**: Michael R. Martin, Garrick Chan, Kwan-Liu Ma

**Abstract**: Recent image protection mechanisms such as Glaze and Nightshade introduce imperceptible, adversarially designed perturbations intended to disrupt downstream text-to-image generative models. While their empirical effectiveness is known, the internal structure, detectability, and representational behavior of these perturbations remain poorly understood. This study provides a systematic, explainable AI analysis using a unified framework that integrates white-box feature-space inspection and black-box signal-level probing. Through latent-space clustering, feature-channel activation analysis, occlusion-based spatial sensitivity mapping, and frequency-domain characterization, we show that protection mechanisms operate as structured, low-entropy perturbations tightly coupled to underlying image content across representational, spatial, and spectral domains. Protected images preserve content-driven feature organization with protection-specific substructure rather than inducing global representational drift. Detectability is governed by interacting effects of perturbation entropy, spatial deployment, and frequency alignment, with sequential protection amplifying detectable structure rather than suppressing it. Frequency-domain analysis shows that Glaze and Nightshade redistribute energy along dominant image-aligned frequency axes rather than introducing diffuse noise. These findings indicate that contemporary image protection operates through structured feature-level deformation rather than semantic dislocation, explaining why protection signals remain visually subtle yet consistently detectable. This work advances the interpretability of adversarial image protection and informs the design of future defenses and detection strategies for generative AI systems.



## **13. TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows**

cs.CV

arxiv v0

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.05150v1) [paper-pdf](https://arxiv.org/pdf/2512.05150v1)

**Confidence**: 0.95

**Authors**: Zhenglin Cheng, Peng Sun, Jianguo Li, Tao Lin

**Abstract**: Recent advances in large multi-modal generative models have demonstrated impressive capabilities in multi-modal generation, including image and video generation. These models are typically built upon multi-step frameworks like diffusion and flow matching, which inherently limits their inference efficiency (requiring 40-100 Number of Function Evaluations (NFEs)). While various few-step methods aim to accelerate the inference, existing solutions have clear limitations. Prominent distillation-based methods, such as progressive and consistency distillation, either require an iterative distillation procedure or show significant degradation at very few steps (< 4-NFE). Meanwhile, integrating adversarial training into distillation (e.g., DMD/DMD2 and SANA-Sprint) to enhance performance introduces training instability, added complexity, and high GPU memory overhead due to the auxiliary trained models. To this end, we propose TwinFlow, a simple yet effective framework for training 1-step generative models that bypasses the need of fixed pretrained teacher models and avoids standard adversarial networks during training, making it ideal for building large-scale, efficient models. On text-to-image tasks, our method achieves a GenEval score of 0.83 in 1-NFE, outperforming strong baselines like SANA-Sprint (a GAN loss-based framework) and RCGM (a consistency-based framework). Notably, we demonstrate the scalability of TwinFlow by full-parameter training on Qwen-Image-20B and transform it into an efficient few-step generator. With just 1-NFE, our approach matches the performance of the original 100-NFE model on both the GenEval and DPG-Bench benchmarks, reducing computational cost by $100\times$ with minor quality degradation. Project page is available at https://zhenglin-cheng.com/twinflow.



## **14. Seedream 4.0: Toward Next-generation Multimodal Image Generation**

cs.CV

Seedream 4.0/4.5 Technical Report

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2509.20427v3) [paper-pdf](https://arxiv.org/pdf/2509.20427v3)

**Confidence**: 0.95

**Authors**: Team Seedream, :, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, Xiaowen Jian, Huafeng Kuang, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, Wei Liu, Yanzuo Lu, Zhengxiong Luo, Tongtong Ou, Guang Shi, Yichun Shi, Shiqi Sun, Yu Tian, Zhi Tian, Peng Wang, Rui Wang, Xun Wang, Ye Wang, Guofeng Wu, Jie Wu, Wenxu Wu, Yonghui Wu, Xin Xia, Xuefeng Xiao, Shuang Xu, Xin Yan, Ceyuan Yang, Jianchao Yang, Zhonghua Zhai, Chenlin Zhang, Heng Zhang, Qi Zhang, Xinyu Zhang, Yuwei Zhang, Shijia Zhao, Wenliang Zhao, Wenjia Zhu

**Abstract**: We introduce Seedream 4.0, an efficient and high-performance multimodal image generation system that unifies text-to-image (T2I) synthesis, image editing, and multi-image composition within a single framework. We develop a highly efficient diffusion transformer with a powerful VAE which also can reduce the number of image tokens considerably. This allows for efficient training of our model, and enables it to fast generate native high-resolution images (e.g., 1K-4K). Seedream 4.0 is pretrained on billions of text-image pairs spanning diverse taxonomies and knowledge-centric concepts. Comprehensive data collection across hundreds of vertical scenarios, coupled with optimized strategies, ensures stable and large-scale training, with strong generalization. By incorporating a carefully fine-tuned VLM model, we perform multi-modal post-training for training both T2I and image editing tasks jointly. For inference acceleration, we integrate adversarial distillation, distribution matching, and quantization, as well as speculative decoding. It achieves an inference time of up to 1.8 seconds for generating a 2K image (without a LLM/VLM as PE model). Comprehensive evaluations reveal that Seedream 4.0 can achieve state-of-the-art results on both T2I and multimodal image editing. In particular, it demonstrates exceptional multimodal capabilities in complex tasks, including precise image editing and in-context reasoning, and also allows for multi-image reference, and can generate multiple output images. This extends traditional T2I systems into an more interactive and multidimensional creative tool, pushing the boundary of generative AI for both creativity and professional applications. We further scale our model and data as Seedream 4.5. Seedream 4.0 and Seedream 4.5 are accessible on Volcano Engine https://www.volcengine.com/experience/ark?launch=seedream.



## **15. If generative AI is the answer, what is the question?**

cs.LG

To appear as a book chapter in a Springer book titled "Statistical Foundations and Applications of Artificial Intelligence, Machine Learning and Deep Learning" and edited by S. Ejaz Ahmed, Pierre Alquier, Yi Li, Shuangge Ma

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2509.06120v2) [paper-pdf](https://arxiv.org/pdf/2509.06120v2)

**Confidence**: 0.95

**Authors**: Ambuj Tewari

**Abstract**: Beginning with text and images, generative AI has expanded to audio, video, computer code, and molecules. Yet, if generative AI is the answer, what is the question? We explore the foundations of generation as a distinct machine learning task with connections to prediction, compression, and decision-making. We survey five major generative model families: autoregressive models, variational autoencoders, normalizing flows, generative adversarial networks, and diffusion models. We then introduce a probabilistic framework that emphasizes the distinction between density estimation and generation. We review a game-theoretic framework with a two-player adversary-learner setup to study generation. We discuss post-training modifications that prepare generative models for deployment. We end by highlighting some important topics in socially responsible generation such as privacy, detection of AI-generated content, and copyright and IP. We adopt a task-first framing of generation, focusing on what generation is as a machine learning problem, rather than only on how models implement it.



## **16. Vocoder-Projected Feature Discriminator**

cs.SD

Accepted to Interspeech 2025. Project page: https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/vpfd/

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.17874v2) [paper-pdf](https://arxiv.org/pdf/2508.17874v2)

**Confidence**: 0.95

**Authors**: Takuhiro Kaneko, Hirokazu Kameoka, Kou Tanaka, Yuto Kondo

**Abstract**: In text-to-speech (TTS) and voice conversion (VC), acoustic features, such as mel spectrograms, are typically used as synthesis or conversion targets owing to their compactness and ease of learning. However, because the ultimate goal is to generate high-quality waveforms, employing a vocoder to convert these features into waveforms and applying adversarial training in the time domain is reasonable. Nevertheless, upsampling the waveform introduces significant time and memory overheads. To address this issue, we propose a vocoder-projected feature discriminator (VPFD), which uses vocoder features for adversarial training. Experiments on diffusion-based VC distillation demonstrated that a pretrained and frozen vocoder feature extractor with a single upsampling step is necessary and sufficient to achieve a VC performance comparable to that of waveform discriminators while reducing the training time and memory consumption by 9.6 and 11.4 times, respectively.



## **17. On the detection of synthetic images generated by diffusion models**

cs.CV

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2211.00680v1) [paper-pdf](https://arxiv.org/pdf/2211.00680v1)

**Confidence**: 0.95

**Authors**: Riccardo Corvi, Davide Cozzolino, Giada Zingarini, Giovanni Poggi, Koki Nagano, Luisa Verdoliva

**Abstract**: Over the past decade, there has been tremendous progress in creating synthetic media, mainly thanks to the development of powerful methods based on generative adversarial networks (GAN). Very recently, methods based on diffusion models (DM) have been gaining the spotlight. In addition to providing an impressive level of photorealism, they enable the creation of text-based visual content, opening up new and exciting opportunities in many different application fields, from arts to video games. On the other hand, this property is an additional asset in the hands of malicious users, who can generate and distribute fake media perfectly adapted to their attacks, posing new challenges to the media forensic community. With this work, we seek to understand how difficult it is to distinguish synthetic images generated by diffusion models from pristine ones and whether current state-of-the-art detectors are suitable for the task. To this end, first we expose the forensics traces left by diffusion models, then study how current detectors, developed for GAN-generated images, perform on these new synthetic images, especially in challenging social-networks scenarios involving image compression and resizing. Datasets and code are available at github.com/grip-unina/DMimageDetection.



## **18. CloneShield: A Framework for Universal Perturbation Against Zero-Shot Voice Cloning**

cs.SD

10pages, 4figures

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2505.19119v1) [paper-pdf](https://arxiv.org/pdf/2505.19119v1)

**Confidence**: 0.95

**Authors**: Renyuan Li, Zhibo Liang, Haichuan Zhang, Tianyu Shi, Zhiyuan Cheng, Jia Shi, Carl Yang, Mingjie Tang

**Abstract**: Recent breakthroughs in text-to-speech (TTS) voice cloning have raised serious privacy concerns, allowing highly accurate vocal identity replication from just a few seconds of reference audio, while retaining the speaker's vocal authenticity. In this paper, we introduce CloneShield, a universal time-domain adversarial perturbation framework specifically designed to defend against zero-shot voice cloning. Our method provides protection that is robust across speakers and utterances, without requiring any prior knowledge of the synthesized text. We formulate perturbation generation as a multi-objective optimization problem, and propose Multi-Gradient Descent Algorithm (MGDA) to ensure the robust protection across diverse utterances. To preserve natural auditory perception for users, we decompose the adversarial perturbation via Mel-spectrogram representations and fine-tune it for each sample. This design ensures imperceptibility while maintaining strong degradation effects on zero-shot cloned outputs. Experiments on three state-of-the-art zero-shot TTS systems, five benchmark datasets and evaluations from 60 human listeners demonstrate that our method preserves near-original audio quality in protected inputs (PESQ = 3.90, SRS = 0.93) while substantially degrading both speaker similarity and speech quality in cloned samples (PESQ = 1.07, SRS = 0.08).



## **19. AvatarSync: Rethinking Talking-Head Animation through Phoneme-Guided Autoregressive Perspective**

cs.CV

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2509.12052v2) [paper-pdf](https://arxiv.org/pdf/2509.12052v2)

**Confidence**: 0.90

**Authors**: Yuchen Deng, Xiuyang Wu, Hai-Tao Zheng, Suiyang Zhang, Yi He, Yuxing Han

**Abstract**: Talking-head animation focuses on generating realistic facial videos from audio input. Following Generative Adversarial Networks (GANs), diffusion models have become the mainstream, owing to their robust generative capacities. However, inherent limitations of the diffusion process often lead to inter-frame flicker and slow inference, restricting their practical deployment. To address this, we introduce AvatarSync, an autoregressive framework on phoneme representations that generates realistic and controllable talking-head animations from a single reference image, driven directly by text or audio input. To mitigate flicker and ensure continuity, AvatarSync leverages an autoregressive pipeline that enhances temporal modeling. In addition, to ensure controllability, we introduce phonemes, which are the basic units of speech sounds, and construct a many-to-one mapping from text/audio to phonemes, enabling precise phoneme-to-visual alignment. Additionally, to further accelerate inference, we adopt a two-stage generation strategy that decouples semantic modeling from visual dynamics, and incorporate a customized Phoneme-Frame Causal Attention Mask to support multi-step parallel acceleration. Extensive experiments conducted on both Chinese (CMLR) and English (HDTF) datasets demonstrate that AvatarSync outperforms existing talking-head animation methods in visual fidelity, temporal consistency, and computational efficiency, providing a scalable and controllable solution.



## **20. HyperNet-Adaptation for Diffusion-Based Test Case Generation**

cs.LG

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.15041v1) [paper-pdf](https://arxiv.org/pdf/2601.15041v1)

**Confidence**: 0.85

**Authors**: Oliver Weißl, Vincenzo Riccio, Severin Kacianka, Andrea Stocco

**Abstract**: The increasing deployment of deep learning systems requires systematic evaluation of their reliability in real-world scenarios. Traditional gradient-based adversarial attacks introduce small perturbations that rarely correspond to realistic failures and mainly assess robustness rather than functional behavior. Generative test generation methods offer an alternative but are often limited to simple datasets or constrained input domains. Although diffusion models enable high-fidelity image synthesis, their computational cost and limited controllability restrict their applicability to large-scale testing. We present HyNeA, a generative testing method that enables direct and efficient control over diffusion-based generation. HyNeA provides dataset-free controllability through hypernetworks, allowing targeted manipulation of the generative process without relying on architecture-specific conditioning mechanisms or dataset-driven adaptations such as fine-tuning. HyNeA employs a distinct training strategy that supports instance-level tuning to identify failure-inducing test cases without requiring datasets that explicitly contain examples of similar failures. This approach enables the targeted generation of realistic failure cases at substantially lower computational cost than search-based methods. Experimental results show that HyNeA improves controllability and test diversity compared to existing generative test generators and generalizes to domains where failure-labeled training data is unavailable.



## **21. Diffusion-Driven Synthetic Tabular Data Generation for Enhanced DoS/DDoS Attack Classification**

cs.CR

7 pages, 8 figures, 2025 International Conference on Signal Processing, Computation, Electronics, Power and Telecommunication (IConSCEPT), National Institute of Technology, Puducherry, India

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2601.13197v1) [paper-pdf](https://arxiv.org/pdf/2601.13197v1)

**Confidence**: 0.85

**Authors**: Aravind B, Anirud R. S., Sai Surya Teja N, Bala Subrahmanya Sriranga Navaneeth A, Karthika R, Mohankumar N

**Abstract**: Class imbalance refers to a situation where certain classes in a dataset have significantly fewer samples than oth- ers, leading to biased model performance. Class imbalance in network intrusion detection using Tabular Denoising Diffusion Probability Models (TabDDPM) for data augmentation is ad- dressed in this paper. Our approach synthesizes high-fidelity minority-class samples from the CIC-IDS2017 dataset through iterative denoising processes. For the minority classes that have smaller samples, synthetic samples were generated and merged with the original dataset. The augmented training data enables an ANN classifier to achieve near-perfect recall on previously underrepresented attack classes. These results establish diffusion models as an effective solution for tabular data imbalance in security domains, with potential applications in fraud detection and medical diagnostics.



## **22. EmoLat: Text-driven Image Sentiment Transfer via Emotion Latent Space**

cs.CV

10 pages, 5 figures

**SubmitDate**: 2026-01-17    [abs](http://arxiv.org/abs/2601.12079v1) [paper-pdf](https://arxiv.org/pdf/2601.12079v1)

**Confidence**: 0.85

**Authors**: Jing Zhang, Bingjie Fan, Jixiang Zhu, Zhe Wang

**Abstract**: We propose EmoLat, a novel emotion latent space that enables fine-grained, text-driven image sentiment transfer by modeling cross-modal correlations between textual semantics and visual emotion features. Within EmoLat, an emotion semantic graph is constructed to capture the relational structure among emotions, objects, and visual attributes. To enhance the discriminability and transferability of emotion representations, we employ adversarial regularization, aligning the latent emotion distributions across modalities. Building upon EmoLat, a cross-modal sentiment transfer framework is proposed to manipulate image sentiment via joint embedding of text and EmoLat features. The network is optimized using a multi-objective loss incorporating semantic consistency, emotion alignment, and adversarial regularization. To support effective modeling, we construct EmoSpace Set, a large-scale benchmark dataset comprising images with dense annotations on emotions, object semantics, and visual attributes. Extensive experiments on EmoSpace Set demonstrate that our approach significantly outperforms existing state-of-the-art methods in both quantitative metrics and qualitative transfer fidelity, establishing a new paradigm for controllable image sentiment editing guided by textual input. The EmoSpace Set and all the code are available at http://github.com/JingVIPLab/EmoLat.



## **23. Speak the Art: A Direct Speech to Image Generation Framework**

eess.AS

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.00827v2) [paper-pdf](https://arxiv.org/pdf/2601.00827v2)

**Confidence**: 0.85

**Authors**: Mariam Saeed, Manar Amr, Farida Adel, Nada Hassan, Nour Walid, Eman Mohamed, Mohamed Hussein, Marwan Torki

**Abstract**: Direct speech-to-image generation has recently shown promising results. However, compared to text-to-image generation, there is still a large gap to enclose. Current approaches use two stages to tackle this task: speech encoding network and image generative adversarial network (GAN). The speech encoding networks in these approaches produce embeddings that do not capture sufficient linguistic information to semantically represent the input speech. GANs suffer from issues such as non-convergence, mode collapse, and diminished gradient, which result in unstable model parameters, limited sample diversity, and ineffective generator learning, respectively. To address these weaknesses, we introduce a framework called Speak the Art (STA) which consists of a speech encoding network and a VQ-Diffusion network conditioned on speech embeddings. To improve speech embeddings, the speech encoding network is supervised by a large pre-trained image-text model during training. Replacing GANs with diffusion leads to more stable training and the generation of diverse images. Additionally, we investigate the feasibility of extending our framework to be multilingual. As a proof of concept, we trained our framework with two languages: English and Arabic. Finally, we show that our results surpass state-of-the-art models by a large margin.



## **24. CLOAK: Contrastive Guidance for Latent Diffusion-Based Data Obfuscation**

cs.LG

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.12086v1) [paper-pdf](https://arxiv.org/pdf/2512.12086v1)

**Confidence**: 0.85

**Authors**: Xin Yang, Omid Ardakanian

**Abstract**: Data obfuscation is a promising technique for mitigating attribute inference attacks by semi-trusted parties with access to time-series data emitted by sensors. Recent advances leverage conditional generative models together with adversarial training or mutual information-based regularization to balance data privacy and utility. However, these methods often require modifying the downstream task, struggle to achieve a satisfactory privacy-utility trade-off, or are computationally intensive, making them impractical for deployment on resource-constrained mobile IoT devices. We propose Cloak, a novel data obfuscation framework based on latent diffusion models. In contrast to prior work, we employ contrastive learning to extract disentangled representations, which guide the latent diffusion process to retain useful information while concealing private information. This approach enables users with diverse privacy needs to navigate the privacy-utility trade-off with minimal retraining. Extensive experiments on four public time-series datasets, spanning multiple sensing modalities, and a dataset of facial images demonstrate that Cloak consistently outperforms state-of-the-art obfuscation techniques and is well-suited for deployment in resource-constrained settings.



## **25. MAGE-ID: A Multimodal Generative Framework for Intrusion Detection Systems**

cs.LG

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03375v1) [paper-pdf](https://arxiv.org/pdf/2512.03375v1)

**Confidence**: 0.85

**Authors**: Mahdi Arab Loodaricheh, Mohammad Hossein Manshaei, Anita Raja

**Abstract**: Modern Intrusion Detection Systems (IDS) face severe challenges due to heterogeneous network traffic, evolving cyber threats, and pronounced data imbalance between benign and attack flows. While generative models have shown promise in data augmentation, existing approaches are limited to single modalities and fail to capture cross-domain dependencies. This paper introduces MAGE-ID (Multimodal Attack Generator for Intrusion Detection), a diffusion-based generative framework that couples tabular flow features with their transformed images through a unified latent prior. By jointly training Transformer and CNN-based variational encoders with an EDM style denoiser, MAGE-ID achieves balanced and coherent multimodal synthesis. Evaluations on CIC-IDS-2017 and NSL-KDD demonstrate significant improvements in fidelity, diversity, and downstream detection performance over TabSyn and TabDDPM, highlighting the effectiveness of MAGE-ID for multimodal IDS augmentation.



## **26. DepFlow: Disentangled Speech Generation to Mitigate Semantic Bias in Depression Detection**

cs.CL

**SubmitDate**: 2026-01-01    [abs](http://arxiv.org/abs/2601.00303v1) [paper-pdf](https://arxiv.org/pdf/2601.00303v1)

**Confidence**: 0.85

**Authors**: Yuxin Li, Xiangyu Zhang, Yifei Li, Zhiwei Guo, Haoyang Zhang, Eng Siong Chng, Cuntai Guan

**Abstract**: Speech is a scalable and non-invasive biomarker for early mental health screening. However, widely used depression datasets like DAIC-WOZ exhibit strong coupling between linguistic sentiment and diagnostic labels, encouraging models to learn semantic shortcuts. As a result, model robustness may be compromised in real-world scenarios, such as Camouflaged Depression, where individuals maintain socially positive or neutral language despite underlying depressive states. To mitigate this semantic bias, we propose DepFlow, a three-stage depression-conditioned text-to-speech framework. First, a Depression Acoustic Encoder learns speaker- and content-invariant depression embeddings through adversarial training, achieving effective disentanglement while preserving depression discriminability (ROC-AUC: 0.693). Second, a flow-matching TTS model with FiLM modulation injects these embeddings into synthesis, enabling control over depressive severity while preserving content and speaker identity. Third, a prototype-based severity mapping mechanism provides smooth and interpretable manipulation across the depression continuum. Using DepFlow, we construct a Camouflage Depression-oriented Augmentation (CDoA) dataset that pairs depressed acoustic patterns with positive/neutral content from a sentiment-stratified text bank, creating acoustic-semantic mismatches underrepresented in natural data. Evaluated across three depression detection architectures, CDoA improves macro-F1 by 9%, 12%, and 5%, respectively, consistently outperforming conventional augmentation strategies in depression Detection. Beyond enhancing robustness, DepFlow provides a controllable synthesis platform for conversational systems and simulation-based evaluation, where real clinical data remains limited by ethical and coverage constraints.



## **27. Arabic TTS with FastPitch: Reproducible Baselines, Adversarial Training, and Oversmoothing Analysis**

eess.AS

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2512.00937v1) [paper-pdf](https://arxiv.org/pdf/2512.00937v1)

**Confidence**: 0.85

**Authors**: Lars Nippert

**Abstract**: Arabic text-to-speech (TTS) remains challenging due to limited resources and complex phonological patterns. We present reproducible baselines for Arabic TTS built on the FastPitch architecture and introduce cepstral-domain metrics for analyzing oversmoothing in mel-spectrogram prediction. While traditional Lp reconstruction losses yield smooth but over-averaged outputs, the proposed metrics reveal their temporal and spectral effects throughout training. To address this, we incorporate a lightweight adversarial spectrogram loss, which trains stably and substantially reduces oversmoothing. We further explore multi-speaker Arabic TTS by augmenting FastPitch with synthetic voices generated using XTTSv2, resulting in improved prosodic diversity without loss of stability. The code, pretrained models, and training recipes are publicly available at: https://github.com/nipponjo/tts-arabic-pytorch.



## **28. RealGen: Photorealistic Text-to-Image Generation via Detector-Guided Rewards**

cs.CV

**SubmitDate**: 2025-11-29    [abs](http://arxiv.org/abs/2512.00473v1) [paper-pdf](https://arxiv.org/pdf/2512.00473v1)

**Confidence**: 0.85

**Authors**: Junyan Ye, Leiqi Zhu, Yuncheng Guo, Dongzhi Jiang, Zilong Huang, Yifan Zhang, Zhiyuan Yan, Haohuan Fu, Conghui He, Weijia Li

**Abstract**: With the continuous advancement of image generation technology, advanced models such as GPT-Image-1 and Qwen-Image have achieved remarkable text-to-image consistency and world knowledge However, these models still fall short in photorealistic image generation. Even on simple T2I tasks, they tend to produce " fake" images with distinct AI artifacts, often characterized by "overly smooth skin" and "oily facial sheens". To recapture the original goal of "indistinguishable-from-reality" generation, we propose RealGen, a photorealistic text-to-image framework. RealGen integrates an LLM component for prompt optimization and a diffusion model for realistic image generation. Inspired by adversarial generation, RealGen introduces a "Detector Reward" mechanism, which quantifies artifacts and assesses realism using both semantic-level and feature-level synthetic image detectors. We leverage this reward signal with the GRPO algorithm to optimize the entire generation pipeline, significantly enhancing image realism and detail. Furthermore, we propose RealBench, an automated evaluation benchmark employing Detector-Scoring and Arena-Scoring. It enables human-free photorealism assessment, yielding results that are more accurate and aligned with real user experience. Experiments demonstrate that RealGen significantly outperforms general models like GPT-Image-1 and Qwen-Image, as well as specialized photorealistic models like FLUX-Krea, in terms of realism, detail, and aesthetics. The code is available at https://github.com/yejy53/RealGen.



## **29. One-Step Diffusion Transformer for Controllable Real-World Image Super-Resolution**

cs.CV

**SubmitDate**: 2025-11-27    [abs](http://arxiv.org/abs/2511.17138v3) [paper-pdf](https://arxiv.org/pdf/2511.17138v3)

**Confidence**: 0.85

**Authors**: Yushun Fang, Yuxiang Chen, Shibo Yin, Qiang Hu, Jiangchao Yao, Ya Zhang, Xiaoyun Zhang, Yanfeng Wang

**Abstract**: Recent advances in diffusion-based real-world image super-resolution (Real-ISR) have demonstrated remarkable perceptual quality, yet the balance between fidelity and controllability remains a problem: multi-step diffusion-based methods suffer from generative diversity and randomness, resulting in low fidelity, while one-step methods lose control flexibility due to fidelity-specific finetuning. In this paper, we present ODTSR, a one-step diffusion transformer based on Qwen-Image that performs Real-ISR considering fidelity and controllability simultaneously: a newly introduced visual stream receives low-quality images (LQ) with adjustable noise (Control Noise), and the original visual stream receives LQs with consistent noise (Prior Noise), forming the Noise-hybrid Visual Stream (NVS) design. ODTSR further employs Fidelity-aware Adversarial Training (FAA) to enhance controllability and achieve one-step inference. Extensive experiments demonstrate that ODTSR not only achieves state-of-the-art (SOTA) performance on generic Real-ISR, but also enables prompt controllability on challenging scenarios such as real-world scene text image super-resolution (STISR) of Chinese characters without training on specific datasets. Codes are available at https://github.com/RedMediaTech/ODTSR.



## **30. SceneGuard: Training-Time Voice Protection with Scene-Consistent Audible Background Noise**

cs.SD

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16114v1) [paper-pdf](https://arxiv.org/pdf/2511.16114v1)

**Confidence**: 0.85

**Authors**: Rui Sang, Yuxuan Liu

**Abstract**: Voice cloning technology poses significant privacy threats by enabling unauthorized speech synthesis from limited audio samples. Existing defenses based on imperceptible adversarial perturbations are vulnerable to common audio preprocessing such as denoising and compression. We propose SceneGuard, a training-time voice protection method that applies scene-consistent audible background noise to speech recordings. Unlike imperceptible perturbations, SceneGuard leverages naturally occurring acoustic scenes (e.g., airport, street, park) to create protective noise that is contextually appropriate and robust to countermeasures. We evaluate SceneGuard on text-to-speech training attacks, demonstrating 5.5% speaker similarity degradation with extremely high statistical significance (p < 10^{-15}, Cohen's d = 2.18) while preserving 98.6% speech intelligibility (STOI = 0.986). Robustness evaluation shows that SceneGuard maintains or enhances protection under five common countermeasures including MP3 compression, spectral subtraction, lowpass filtering, and downsampling. Our results suggest that audible, scene-consistent noise provides a more robust alternative to imperceptible perturbations for training-time voice protection. The source code are available at: https://github.com/richael-sang/SceneGuard.



## **31. Approximate Gaussian Mapping for Generative Image Steganography**

cs.CR

13 pages

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2510.07219v2) [paper-pdf](https://arxiv.org/pdf/2510.07219v2)

**Confidence**: 0.85

**Authors**: Yuhua Xu, Wei Sun, Chengpei Tang, Jiaxing Lu, Jingying Zhou, Chen Gu

**Abstract**: Ordinary differential equation (ODE)-based diffusion models enable deterministic image synthesis, establishing a reversible mapping suitable for generative steganography. While prevailing methods strictly adhere to a standard normal prior, empirical evidence indicates that controlled deviations from this distribution reduce numerical inversion errors without compromising perceptual quality. Leveraging this observation, the Approximate Gaussian Mapping (AGM) is proposed as a linear transformation strategy that embeds secrets by modulating noise scale and variance. To balance retrieval numerical consistence and security, a two-stage decoupled optimization strategy is introduced to minimize the Kullback-Leibler divergence subject to target bit accuracy constraints. Beyond the proposed method, we conduct a mechanistic analysis of the divergent behaviors between pixel-space and latent-space architectures. The experimental results reveal that the VAE encoder enhances robustness by filtering external perturbations, whereas the structural regularization of the VAE decoder and the semantic variance introduced by text prompts jointly mask embedding artifacts to improve security. Experimental results confirm that pixel-space mplementations maximize embedding capacity for lossless channels, while latent-space approaches offer superior robustness and security suitable for adversarial environments



## **32. A Lightweight Pipeline for Noisy Speech Voice Cloning and Accurate Lip Sync Synthesis**

cs.SD

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12831v1) [paper-pdf](https://arxiv.org/pdf/2509.12831v1)

**Confidence**: 0.85

**Authors**: Javeria Amir, Farwa Attaria, Mah Jabeen, Umara Noor, Zahid Rashid

**Abstract**: Recent developments in voice cloning and talking head generation demonstrate impressive capabilities in synthesizing natural speech and realistic lip synchronization. Current methods typically require and are trained on large scale datasets and computationally intensive processes using clean studio recorded inputs that is infeasible in noisy or low resource environments. In this paper, we introduce a new modular pipeline comprising Tortoise text to speech. It is a transformer based latent diffusion model that can perform high fidelity zero shot voice cloning given only a few training samples. We use a lightweight generative adversarial network architecture for robust real time lip synchronization. The solution will contribute to many essential tasks concerning less reliance on massive pre training generation of emotionally expressive speech and lip synchronization in noisy and unconstrained scenarios. The modular structure of the pipeline allows an easy extension for future multi modal and text guided voice modulation and it could be used in real world systems.



## **33. Emotion Detection Using Conditional Generative Adversarial Networks (cGAN): A Deep Learning Approach**

cs.LG

3 pages, 2 tables, submitted for arXiv preprint

**SubmitDate**: 2025-08-06    [abs](http://arxiv.org/abs/2508.04481v1) [paper-pdf](https://arxiv.org/pdf/2508.04481v1)

**Confidence**: 0.85

**Authors**: Anushka Srivastava

**Abstract**: This paper presents a deep learning-based approach to emotion detection using Conditional Generative Adversarial Networks (cGANs). Unlike traditional unimodal techniques that rely on a single data type, we explore a multimodal framework integrating text, audio, and facial expressions. The proposed cGAN architecture is trained to generate synthetic emotion-rich data and improve classification accuracy across multiple modalities. Our experimental results demonstrate significant improvements in emotion recognition performance compared to baseline models. This work highlights the potential of cGANs in enhancing human-computer interaction systems by enabling more nuanced emotional understanding.



## **34. RingFormer: A Neural Vocoder with Ring Attention and Convolution-Augmented Transformer**

cs.SD

Accepted for publication in IEEE Transactions on Human-Machine Systems (THMS)

**SubmitDate**: 2025-07-19    [abs](http://arxiv.org/abs/2501.01182v2) [paper-pdf](https://arxiv.org/pdf/2501.01182v2)

**Confidence**: 0.85

**Authors**: Seongho Hong, Yong-Hoon Choi

**Abstract**: While transformers demonstrate outstanding performance across various audio tasks, their application to neural vocoders remains challenging. Neural vocoders require the generation of long audio signals at the sample level, which demands high temporal resolution. This results in significant computational costs for attention map generation and limits their ability to efficiently process both global and local information. Additionally, the sequential nature of sample generation in neural vocoders poses difficulties for real-time processing, making the direct adoption of transformers impractical. To address these challenges, we propose RingFormer, a neural vocoder that incorporates the ring attention mechanism into a lightweight transformer variant, the convolution-augmented transformer (Conformer). Ring attention effectively captures local details while integrating global information, making it well-suited for processing long sequences and enabling real-time audio generation. RingFormer is trained using adversarial training with two discriminators. The proposed model is applied to the decoder of the text-to-speech model VITS and compared with state-of-the-art vocoders such as HiFi-GAN, iSTFT-Net, and BigVGAN under identical conditions using various objective and subjective metrics. Experimental results show that RingFormer achieves comparable or superior performance to existing models, particularly excelling in real-time audio generation. Our code and audio samples are available on GitHub.



## **35. Robust Watermarking Using Generative Priors Against Image Editing: From Benchmarking to Advances**

cs.CV

Accepted by ICLR 2025

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2410.18775v2) [paper-pdf](https://arxiv.org/pdf/2410.18775v2)

**Confidence**: 0.85

**Authors**: Shilin Lu, Zihan Zhou, Jiayou Lu, Yuanzhi Zhu, Adams Wai-Kin Kong

**Abstract**: Current image watermarking methods are vulnerable to advanced image editing techniques enabled by large-scale text-to-image models. These models can distort embedded watermarks during editing, posing significant challenges to copyright protection. In this work, we introduce W-Bench, the first comprehensive benchmark designed to evaluate the robustness of watermarking methods against a wide range of image editing techniques, including image regeneration, global editing, local editing, and image-to-video generation. Through extensive evaluations of eleven representative watermarking methods against prevalent editing techniques, we demonstrate that most methods fail to detect watermarks after such edits. To address this limitation, we propose VINE, a watermarking method that significantly enhances robustness against various image editing techniques while maintaining high image quality. Our approach involves two key innovations: (1) we analyze the frequency characteristics of image editing and identify that blurring distortions exhibit similar frequency properties, which allows us to use them as surrogate attacks during training to bolster watermark robustness; (2) we leverage a large-scale pretrained diffusion model SDXL-Turbo, adapting it for the watermarking task to achieve more imperceptible and robust watermark embedding. Experimental results show that our method achieves outstanding watermarking performance under various image editing techniques, outperforming existing methods in both image quality and robustness. Code is available at https://github.com/Shilin-LU/VINE.



## **36. Efficient Generative Adversarial Networks for Color Document Image Enhancement and Binarization Using Multi-scale Feature Extraction**

cs.CV

Accepted to APSIPA ASC 2025

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2407.04231v2) [paper-pdf](https://arxiv.org/pdf/2407.04231v2)

**Confidence**: 0.85

**Authors**: Rui-Yang Ju, KokSheik Wong, Jen-Shiun Chiang

**Abstract**: The outcome of text recognition for degraded color documents is often unsatisfactory due to interference from various contaminants. To extract information more efficiently for text recognition, document image enhancement and binarization are often employed as preliminary steps in document analysis. Training independent generative adversarial networks (GANs) for each color channel can generate images where shadows and noise are effectively removed, which subsequently allows for efficient text information extraction. However, employing multiple GANs for different color channels requires long training and inference times. To reduce both the training and inference times of these preliminary steps, we propose an efficient method based on multi-scale feature extraction, which incorporates Haar wavelet transformation and normalization to process document images before submitting them to GANs for training. Experiment results show that our proposed method significantly reduces both the training and inference times while maintaining comparable performances when benchmarked against the state-of-the-art methods. In the best case scenario, a reduction of 10% and 26% are observed for training and inference times, respectively, while maintaining the model performance at 73.79 of Average-Score metric. The implementation of this work is available at https://github.com/RuiyangJu/Efficient_Document_Image_Binarization.



## **37. VocalCrypt: Novel Active Defense Against Deepfake Voice Based on Masking Effect**

cs.SD

9 pages, four figures

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2502.10329v1) [paper-pdf](https://arxiv.org/pdf/2502.10329v1)

**Confidence**: 0.85

**Authors**: Qingyuan Fei, Wenjie Hou, Xuan Hai, Xin Liu

**Abstract**: The rapid advancements in AI voice cloning, fueled by machine learning, have significantly impacted text-to-speech (TTS) and voice conversion (VC) fields. While these developments have led to notable progress, they have also raised concerns about the misuse of AI VC technology, causing economic losses and negative public perceptions. To address this challenge, this study focuses on creating active defense mechanisms against AI VC systems.   We propose a novel active defense method, VocalCrypt, which embeds pseudo-timbre (jamming information) based on SFS into audio segments that are imperceptible to the human ear, thereby forming systematic fragments to prevent voice cloning. This approach protects the voice without compromising its quality. In comparison to existing methods, such as adversarial noise incorporation, VocalCrypt significantly enhances robustness and real-time performance, achieving a 500\% increase in generation speed while maintaining interference effectiveness.   Unlike audio watermarking techniques, which focus on post-detection, our method offers preemptive defense, reducing implementation costs and enhancing feasibility. Extensive experiments using the Zhvoice and VCTK Corpus datasets show that our AI-cloned speech defense system performs excellently in automatic speaker verification (ASV) tests while preserving the integrity of the protected audio.




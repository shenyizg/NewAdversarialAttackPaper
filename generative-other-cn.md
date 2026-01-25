# 泛生成模型 - 其他攻防技术
**update at 2026-01-25 10:36:50**

按分类器置信度从高到低排序。

## **1. Safeguarding Facial Identity against Diffusion-based Face Swapping via Cascading Pathway Disruption**

通过级联路径破坏保护面部身份免受基于扩散模型的人脸交换攻击 cs.CV

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.14738v1) [paper-pdf](https://arxiv.org/pdf/2601.14738v1)

**Confidence**: 0.95

**Authors**: Liqin Wang, Qianyue Hu, Wei Lu, Xiangyang Luo

**Abstract**: The rapid evolution of diffusion models has democratized face swapping but also raises concerns about privacy and identity security. Existing proactive defenses, often adapted from image editing attacks, prove ineffective in this context. We attribute this failure to an oversight of the structural resilience and the unique static conditional guidance mechanism inherent in face swapping systems. To address this, we propose VoidFace, a systemic defense method that views face swapping as a coupled identity pathway. By injecting perturbations at critical bottlenecks, VoidFace induces cascading disruption throughout the pipeline. Specifically, we first introduce localization disruption and identity erasure to degrade physical regression and semantic embeddings, thereby impairing the accurate modeling of the source face. We then intervene in the generative domain by decoupling attention mechanisms to sever identity injection, and corrupting intermediate diffusion features to prevent the reconstruction of source identity. To ensure visual imperceptibility, we perform adversarial search in the latent manifold, guided by a perceptual adaptive strategy to balance attack potency with image quality. Extensive experiments show that VoidFace outperforms existing defenses across various diffusion-based swapping models, while producing adversarial faces with superior visual quality.

摘要: 扩散模型的快速发展使得人脸交换技术普及化，但也引发了隐私和身份安全方面的担忧。现有的主动防御方法通常改编自图像编辑攻击，在此场景中被证明效果有限。我们将这种失效归因于对人脸交换系统固有的结构鲁棒性及独特静态条件引导机制的忽视。为此，我们提出VoidFace——一种将人脸交换视为耦合身份通路的系统性防御方法。通过在关键瓶颈处注入扰动，VoidFace能在整个流程中引发级联破坏。具体而言，我们首先引入定位破坏和身份擦除来降低物理回归和语义嵌入的质量，从而削弱对源人脸的精确建模。随后，我们通过解耦注意力机制来切断身份注入，并破坏中间扩散特征以防止源身份重建，以此干预生成域。为确保视觉不可感知性，我们在潜在流形中进行对抗性搜索，并采用感知自适应策略来平衡攻击效力与图像质量。大量实验表明，VoidFace在多种基于扩散的交换模型中均优于现有防御方法，同时能生成具有卓越视觉质量的对抗性人脸。



## **2. PhaseMark: A Post-hoc, Optimization-Free Watermarking of AI-generated Images in the Latent Frequency Domain**

PhaseMark：潜在频域中AI生成图像的后处理、免优化水印技术 cs.CV

Accepted to the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2026

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2601.13128v1) [paper-pdf](https://arxiv.org/pdf/2601.13128v1)

**Confidence**: 0.95

**Authors**: Sung Ju Lee, Nam Ik Cho

**Abstract**: The proliferation of hyper-realistic images from Latent Diffusion Models (LDMs) demands robust watermarking, yet existing post-hoc methods are prohibitively slow due to iterative optimization or inversion processes. We introduce PhaseMark, a single-shot, optimization-free framework that directly modulates the phase in the VAE latent frequency domain. This approach makes PhaseMark thousands of times faster than optimization-based techniques while achieving state-of-the-art resilience against severe attacks, including regeneration, without degrading image quality. We analyze four modulation variants, revealing a clear performance-quality trade-off. PhaseMark demonstrates a new paradigm where efficient, resilient watermarking is achieved by exploiting intrinsic latent properties.

摘要: 潜在扩散模型（LDMs）生成的高度逼真图像激增，亟需鲁棒的水印技术，但现有后处理方法因迭代优化或反演过程而速度极慢。我们提出PhaseMark，一种单次处理、免优化的框架，直接在VAE潜在频域中调制相位。该方法使PhaseMark比基于优化的技术快数千倍，同时在对包括再生在内的严重攻击中实现最先进的鲁棒性，且不降低图像质量。我们分析了四种调制变体，揭示了清晰的性能-质量权衡。PhaseMark展示了一种新范式：通过利用内在潜在特性实现高效、鲁棒的水印技术。



## **3. Semantic Mismatch and Perceptual Degradation: A New Perspective on Image Editing Immunity**

语义失配与感知退化：图像编辑免疫的新视角 cs.CV

11 pages, 4 figures

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14320v1) [paper-pdf](https://arxiv.org/pdf/2512.14320v1)

**Confidence**: 0.95

**Authors**: Shuai Dong, Jie Zhang, Guoying Zhao, Shiguang Shan, Xilin Chen

**Abstract**: Text-guided image editing via diffusion models, while powerful, raises significant concerns about misuse, motivating efforts to immunize images against unauthorized edits using imperceptible perturbations. Prevailing metrics for evaluating immunization success typically rely on measuring the visual dissimilarity between the output generated from a protected image and a reference output generated from the unprotected original. This approach fundamentally overlooks the core requirement of image immunization, which is to disrupt semantic alignment with attacker intent, regardless of deviation from any specific output. We argue that immunization success should instead be defined by the edited output either semantically mismatching the prompt or suffering substantial perceptual degradations, both of which thwart malicious intent. To operationalize this principle, we propose Synergistic Intermediate Feature Manipulation (SIFM), a method that strategically perturbs intermediate diffusion features through dual synergistic objectives: (1) maximizing feature divergence from the original edit trajectory to disrupt semantic alignment with the expected edit, and (2) minimizing feature norms to induce perceptual degradations. Furthermore, we introduce the Immunization Success Rate (ISR), a novel metric designed to rigorously quantify true immunization efficacy for the first time. ISR quantifies the proportion of edits where immunization induces either semantic failure relative to the prompt or significant perceptual degradations, assessed via Multimodal Large Language Models (MLLMs). Extensive experiments show our SIFM achieves the state-of-the-art performance for safeguarding visual content against malicious diffusion-based manipulation.

摘要: 基于扩散模型的文本引导图像编辑虽然功能强大，但也引发了严重的滥用担忧，这促使人们通过不可察觉的扰动来免疫图像以防止未经授权的编辑。评估免疫成功的主流指标通常依赖于测量受保护图像生成的输出与未受保护原始图像生成的参考输出之间的视觉差异。这种方法从根本上忽视了图像免疫的核心要求，即破坏与攻击者意图的语义对齐，而无论其与任何特定输出的偏离程度如何。我们认为，免疫成功应定义为编辑输出要么与提示语义失配，要么遭受严重的感知退化，这两种情况都能挫败恶意意图。为实践这一原则，我们提出了协同中间特征操纵（SIFM），该方法通过双重协同目标策略性地扰动扩散中间特征：（1）最大化特征与原始编辑轨迹的差异，以破坏与预期编辑的语义对齐；（2）最小化特征范数以诱发感知退化。此外，我们引入了免疫成功率（ISR），这是一种新颖的指标，首次设计用于严格量化真实的免疫效能。ISR通过多模态大语言模型（MLLMs）评估，量化了免疫导致编辑相对于提示语义失败或显著感知退化的比例。大量实验表明，我们的SIFM在保护视觉内容免受基于扩散的恶意操纵方面实现了最先进的性能。



## **4. GeMM-GAN: A Multimodal Generative Model Conditioned on Histopathology Images and Clinical Descriptions for Gene Expression Profile Generation**

GeMM-GAN：基于组织病理学图像和临床描述的多模态生成模型用于基因表达谱生成 cs.AI

12 pages, 2 figures. Published at Image Analysis and Processing - ICIAP 2025 Workshops

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.15392v1) [paper-pdf](https://arxiv.org/pdf/2601.15392v1)

**Confidence**: 0.95

**Authors**: Francesca Pia Panaccione, Carlo Sgaravatti, Pietro Pinoli

**Abstract**: Biomedical research increasingly relies on integrating diverse data modalities, including gene expression profiles, medical images, and clinical metadata. While medical images and clinical metadata are routinely collected in clinical practice, gene expression data presents unique challenges for widespread research use, mainly due to stringent privacy regulations and costly laboratory experiments. To address these limitations, we present GeMM-GAN, a novel Generative Adversarial Network conditioned on histopathology tissue slides and clinical metadata, designed to synthesize realistic gene expression profiles. GeMM-GAN combines a Transformer Encoder for image patches with a final Cross Attention mechanism between patches and text tokens, producing a conditioning vector to guide a generative model in generating biologically coherent gene expression profiles. We evaluate our approach on the TCGA dataset and demonstrate that our framework outperforms standard generative models and generates more realistic and functionally meaningful gene expression profiles, improving by more than 11\% the accuracy on downstream disease type prediction compared to current state-of-the-art generative models. Code will be available at: https://github.com/francescapia/GeMM-GAN

摘要: 生物医学研究日益依赖于整合多种数据模态，包括基因表达谱、医学影像和临床元数据。虽然医学影像和临床元数据在临床实践中常规收集，但基因表达数据在大规模研究应用中面临独特挑战，主要源于严格的隐私法规和昂贵的实验室实验。为应对这些限制，我们提出GeMM-GAN——一种基于组织病理学切片和临床元数据的新型生成对抗网络，旨在合成逼真的基因表达谱。GeMM-GAN结合了用于图像块的Transformer编码器，以及图像块与文本标记之间的最终交叉注意力机制，生成用于指导生成模型产生生物学一致性基因表达谱的条件向量。我们在TCGA数据集上评估该方法，证明该框架优于标准生成模型，能生成更逼真且功能有意义的基因表达谱，在下游疾病类型预测任务中，相比当前最先进的生成模型准确率提升超过11%。代码将在以下网址公开：https://github.com/francescapia/GeMM-GAN



## **5. Universal Adversarial Purification with DDIM Metric Loss for Stable Diffusion**

基于DDIM度量损失的稳定扩散通用对抗净化方法 cs.CV

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07253v1) [paper-pdf](https://arxiv.org/pdf/2601.07253v1)

**Confidence**: 0.95

**Authors**: Li Zheng, Liangbin Xie, Jiantao Zhou, He YiMin

**Abstract**: Stable Diffusion (SD) often produces degraded outputs when the training dataset contains adversarial noise. Adversarial purification offers a promising solution by removing adversarial noise from contaminated data. However, existing purification methods are primarily designed for classification tasks and fail to address SD-specific adversarial strategies, such as attacks targeting the VAE encoder, UNet denoiser, or both. To address the gap in SD security, we propose Universal Diffusion Adversarial Purification (UDAP), a novel framework tailored for defending adversarial attacks targeting SD models. UDAP leverages the distinct reconstruction behaviors of clean and adversarial images during Denoising Diffusion Implicit Models (DDIM) inversion to optimize the purification process. By minimizing the DDIM metric loss, UDAP can effectively remove adversarial noise. Additionally, we introduce a dynamic epoch adjustment strategy that adapts optimization iterations based on reconstruction errors, significantly improving efficiency without sacrificing purification quality. Experiments demonstrate UDAP's robustness against diverse adversarial methods, including PID (VAE-targeted), Anti-DreamBooth (UNet-targeted), MIST (hybrid), and robustness-enhanced variants like Anti-Diffusion (Anti-DF) and MetaCloak. UDAP also generalizes well across SD versions and text prompts, showcasing its practical applicability in real-world scenarios.

摘要: 当训练数据集中包含对抗性噪声时，稳定扩散（SD）模型常产生质量退化的输出。对抗净化通过从受污染数据中移除对抗性噪声，为此问题提供了有前景的解决方案。然而，现有净化方法主要针对分类任务设计，未能应对SD特有的对抗策略，例如针对VAE编码器、UNet去噪器或两者的攻击。为填补SD安全领域的空白，我们提出了通用扩散对抗净化（UDAP），这是一个专为防御针对SD模型的对抗攻击而设计的新框架。UDAP利用干净图像与对抗图像在去噪扩散隐式模型（DDIM）反转过程中不同的重建行为来优化净化过程。通过最小化DDIM度量损失，UDAP能有效移除对抗性噪声。此外，我们引入了动态轮次调整策略，该策略根据重建误差自适应优化迭代次数，在保持净化质量的同时显著提升效率。实验表明，UDAP对多种对抗方法均表现出鲁棒性，包括PID（VAE靶向）、Anti-DreamBooth（UNet靶向）、MIST（混合攻击）以及鲁棒性增强变体如Anti-Diffusion（Anti-DF）和MetaCloak。UDAP在不同SD版本和文本提示下也展现出良好的泛化能力，证明了其在真实场景中的实际适用性。



## **6. Attack-Resistant Watermarking for AIGC Image Forensics via Diffusion-based Semantic Deflection**

基于扩散语义偏转的抗攻击AIGC图像取证水印技术 cs.CR

**SubmitDate**: 2026-01-10    [abs](http://arxiv.org/abs/2601.06639v1) [paper-pdf](https://arxiv.org/pdf/2601.06639v1)

**Confidence**: 0.95

**Authors**: Qingyu Liu, Yitao Zhang, Zhongjie Ba, Chao Shuai, Peng Cheng, Tianhang Zheng, Zhibo Wang

**Abstract**: Protecting the copyright of user-generated AI images is an emerging challenge as AIGC becomes pervasive in creative workflows. Existing watermarking methods (1) remain vulnerable to real-world adversarial threats, often forced to trade off between defenses against spoofing and removal attacks; and (2) cannot support semantic-level tamper localization. We introduce PAI, a training-free inherent watermarking framework for AIGC copyright protection, plug-and-play with diffusion-based AIGC services. PAI simultaneously provides three key functionalities: robust ownership verification, attack detection, and semantic-level tampering localization. Unlike existing inherent watermark methods that only embed watermarks at noise initialization of diffusion models, we design a novel key-conditioned deflection mechanism that subtly steers the denoising trajectory according to the user key. Such trajectory-level coupling further strengthens the semantic entanglement of identity and content, thereby further enhancing robustness against real-world threats. Moreover, we also provide a theoretical analysis proving that only the valid key can pass verification. Experiments across 12 attack methods show that PAI achieves 98.43\% verification accuracy, improving over SOTA methods by 37.25\% on average, and retains strong tampering localization performance even against advanced AIGC edits. Our code is available at https://github.com/QingyuLiu/PAI.

摘要: 随着AIGC在创意工作流中的普及，保护用户生成AI图像的版权成为新兴挑战。现有水印方法存在两大局限：(1) 仍易受现实对抗性攻击威胁，常被迫在防伪造与防去除攻击间权衡；(2) 无法支持语义级篡改定位。本文提出PAI——一种免训练的AIGC版权保护固有水印框架，可与基于扩散的AIGC服务即插即用。PAI同时提供三大核心功能：鲁棒所有权验证、攻击检测及语义级篡改定位。不同于现有仅通过扩散模型噪声初始化嵌入水印的固有水印方法，我们设计了新颖的密钥条件偏转机制，根据用户密钥精妙引导去噪轨迹。这种轨迹级耦合进一步强化了身份与内容的语义纠缠，从而显著提升对现实威胁的鲁棒性。此外，我们通过理论分析证明仅有效密钥可通过验证。在12种攻击方法上的实验表明，PAI达到98.43%的验证准确率，较现有最优方法平均提升37.25%，即使面对高级AIGC编辑仍保持强大的篡改定位性能。代码已开源：https://github.com/QingyuLiu/PAI。



## **7. Training-Free Color-Aware Adversarial Diffusion Sanitization for Diffusion Stegomalware Defense at Security Gateways**

面向安全网关的无训练色彩感知对抗扩散净化：用于扩散隐写恶意软件防御 cs.CR

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.24499v1) [paper-pdf](https://arxiv.org/pdf/2512.24499v1)

**Confidence**: 0.95

**Authors**: Vladimir Frants, Sos Agaian

**Abstract**: The rapid expansion of generative AI has normalized large-scale synthetic media creation, enabling new forms of covert communication. Recent generative steganography methods, particularly those based on diffusion models, can embed high-capacity payloads without fine-tuning or auxiliary decoders, creating significant challenges for detection and remediation. Coverless diffusion-based techniques are difficult to counter because they generate image carriers directly from secret data, enabling attackers to deliver stegomalware for command-and-control, payload staging, and data exfiltration while bypassing detectors that rely on cover-stego discrepancies. This work introduces Adversarial Diffusion Sanitization (ADS), a training-free defense for security gateways that neutralizes hidden payloads rather than detecting them. ADS employs an off-the-shelf pretrained denoiser as a differentiable proxy for diffusion-based decoders and incorporates a color-aware, quaternion-coupled update rule to reduce artifacts under strict distortion limits. Under a practical threat model and in evaluation against the state-of-the-art diffusion steganography method Pulsar, ADS drives decoder success rates to near zero with minimal perceptual impact. Results demonstrate that ADS provides a favorable security-utility trade-off compared to standard content transformations, offering an effective mitigation strategy against diffusion-driven steganography.

摘要: 生成式AI的快速扩张使大规模合成媒体创作常态化，催生了新的隐蔽通信形式。最近的生成式隐写方法，特别是基于扩散模型的技术，无需微调或辅助解码器即可嵌入高容量载荷，给检测与修复带来重大挑战。无载体的扩散隐写技术难以应对，因为它们直接从秘密数据生成图像载体，使攻击者能够绕过依赖载体-隐写差异的检测器，传递用于命令控制、载荷暂存和数据渗漏的隐写恶意软件。本研究提出对抗扩散净化（ADS），一种面向安全网关的无训练防御方案，通过消除隐藏载荷而非检测它们来实现防护。ADS采用现成的预训练去噪器作为基于扩散的解码器的可微分代理，并结合色彩感知的四元数耦合更新规则，在严格失真限制下减少伪影。在实际威胁模型下，针对最先进的扩散隐写方法Pulsar进行评估，ADS在最小感知影响下将解码成功率降至接近零。结果表明，与标准内容转换相比，ADS提供了更优的安全-效用权衡，为对抗扩散驱动的隐写提供了有效的缓解策略。



## **8. Smark: A Watermark for Text-to-Speech Diffusion Models via Discrete Wavelet Transform**

Smark：基于离散小波变换的文本到语音扩散模型水印技术 cs.SD

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18791v1) [paper-pdf](https://arxiv.org/pdf/2512.18791v1)

**Confidence**: 0.95

**Authors**: Yichuan Zhang, Chengxin Li, Yujie Gu

**Abstract**: Text-to-Speech (TTS) diffusion models generate high-quality speech, which raises challenges for the model intellectual property protection and speech tracing for legal use. Audio watermarking is a promising solution. However, due to the structural differences among various TTS diffusion models, existing watermarking methods are often designed for a specific model and degrade audio quality, which limits their practical applicability. To address this dilemma, this paper proposes a universal watermarking scheme for TTS diffusion models, termed Smark. This is achieved by designing a lightweight watermark embedding framework that operates in the common reverse diffusion paradigm shared by all TTS diffusion models. To mitigate the impact on audio quality, Smark utilizes the discrete wavelet transform (DWT) to embed watermarks into the relatively stable low-frequency regions of the audio, which ensures seamless watermark-audio integration and is resistant to removal during the reverse diffusion process. Extensive experiments are conducted to evaluate the audio quality and watermark performance in various simulated real-world attack scenarios. The experimental results show that Smark achieves superior performance in both audio quality and watermark extraction accuracy.

摘要: 文本到语音（TTS）扩散模型能够生成高质量语音，这对模型知识产权保护和合法使用场景下的语音溯源提出了挑战。音频水印是一种有前景的解决方案。然而，由于不同TTS扩散模型的结构差异，现有水印方法通常针对特定模型设计且会降低音频质量，限制了其实际应用。为解决这一难题，本文提出了一种适用于TTS扩散模型的通用水印方案，称为Smark。该方案通过设计一个轻量级水印嵌入框架实现，该框架在所有TTS扩散模型共有的反向扩散范式下运行。为减少对音频质量的影响，Smark利用离散小波变换（DWT）将水印嵌入音频相对稳定的低频区域，这确保了水印与音频的无缝融合，并能抵抗反向扩散过程中的移除。通过大量实验评估了模拟真实攻击场景下的音频质量与水印性能。实验结果表明，Smark在音频质量和水印提取准确性方面均表现出优越性能。



## **9. Towards Transferable Defense Against Malicious Image Edits**

面向恶意图像编辑的可迁移防御方法研究 cs.CV

14 pages, 5 figures

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14341v1) [paper-pdf](https://arxiv.org/pdf/2512.14341v1)

**Confidence**: 0.95

**Authors**: Jie Zhang, Shuai Dong, Shiguang Shan, Xilin Chen

**Abstract**: Recent approaches employing imperceptible perturbations in input images have demonstrated promising potential to counter malicious manipulations in diffusion-based image editing systems. However, existing methods suffer from limited transferability in cross-model evaluations. To address this, we propose Transferable Defense Against Malicious Image Edits (TDAE), a novel bimodal framework that enhances image immunity against malicious edits through coordinated image-text optimization. Specifically, at the visual defense level, we introduce FlatGrad Defense Mechanism (FDM), which incorporates gradient regularization into the adversarial objective. By explicitly steering the perturbations toward flat minima, FDM amplifies immune robustness against unseen editing models. For textual enhancement protection, we propose an adversarial optimization paradigm named Dynamic Prompt Defense (DPD), which periodically refines text embeddings to align the editing outcomes of immunized images with those of the original images, then updates the images under optimized embeddings. Through iterative adversarial updates to diverse embeddings, DPD enforces the generation of immunized images that seek a broader set of immunity-enhancing features, thereby achieving cross-model transferability. Extensive experimental results demonstrate that our TDAE achieves state-of-the-art performance in mitigating malicious edits under both intra- and cross-model evaluations.

摘要: 近期研究表明，在输入图像中添加人眼难以察觉的扰动，在对抗基于扩散模型的恶意图像编辑系统方面展现出良好潜力。然而，现有方法在跨模型评估中表现出有限的可迁移性。为此，我们提出可迁移防御恶意图像编辑框架（TDAE），这是一种通过协调图像-文本优化来增强图像对恶意编辑免疫力的新型双模态框架。具体而言，在视觉防御层面，我们引入平坦梯度防御机制（FDM），将梯度正则化融入对抗目标。通过显式引导扰动趋向平坦最小值，FDM增强了针对未见编辑模型的免疫鲁棒性。在文本增强保护方面，我们提出名为动态提示防御（DPD）的对抗优化范式，该范式周期性地优化文本嵌入，使免疫化图像的编辑结果与原始图像对齐，随后在优化后的嵌入下更新图像。通过对多样化嵌入进行迭代对抗更新，DPD强制生成寻求更广泛免疫增强特征的免疫化图像，从而实现跨模型可迁移性。大量实验结果表明，我们的TDAE在模型内和跨模型评估中均实现了最先进的恶意编辑缓解性能。



## **10. MFE-GAN: Efficient GAN-based Framework for Document Image Enhancement and Binarization with Multi-scale Feature Extraction**

MFE-GAN：基于多尺度特征提取的高效GAN框架用于文档图像增强与二值化 cs.CV

Extended Journal Version of APSIPA ASC 2025

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14114v1) [paper-pdf](https://arxiv.org/pdf/2512.14114v1)

**Confidence**: 0.95

**Authors**: Rui-Yang Ju, KokSheik Wong, Yanlin Jin, Jen-Shiun Chiang

**Abstract**: Document image enhancement and binarization are commonly performed prior to document analysis and recognition tasks for improving the efficiency and accuracy of optical character recognition (OCR) systems. This is because directly recognizing text in degraded documents, particularly in color images, often results in unsatisfactory recognition performance. To address these issues, existing methods train independent generative adversarial networks (GANs) for different color channels to remove shadows and noise, which, in turn, facilitates efficient text information extraction. However, deploying multiple GANs results in long training and inference times. To reduce both training and inference times of document image enhancement and binarization models, we propose MFE-GAN, an efficient GAN-based framework with multi-scale feature extraction (MFE), which incorporates Haar wavelet transformation (HWT) and normalization to process document images before feeding them into GANs for training. In addition, we present novel generators, discriminators, and loss functions to improve the model's performance, and we conduct ablation studies to demonstrate their effectiveness. Experimental results on the Benchmark, Nabuco, and CMATERdb datasets demonstrate that the proposed MFE-GAN significantly reduces the total training and inference times while maintaining comparable performance with respect to state-of-the-art (SOTA) methods. The implementation of this work is available at https://ruiyangju.github.io/MFE-GAN.

摘要: 文档图像增强与二值化通常在文档分析与识别任务前执行，以提高光学字符识别（OCR）系统的效率与准确率。这是因为直接识别退化文档（特别是彩色图像）中的文本往往导致不理想的识别性能。现有方法通过为不同颜色通道训练独立的生成对抗网络（GAN）来消除阴影和噪声，从而促进高效的文本信息提取。然而，部署多个GAN会导致训练和推理时间过长。为减少文档图像增强与二值化模型的训练与推理时间，我们提出MFE-GAN——一种融合多尺度特征提取（MFE）的高效GAN框架，该框架结合哈尔小波变换（HWT）和归一化技术，在将文档图像输入GAN训练前进行预处理。此外，我们设计了新颖的生成器、判别器和损失函数以提升模型性能，并通过消融实验验证其有效性。在Benchmark、Nabuco和CMATERdb数据集上的实验结果表明，所提出的MFE-GAN在保持与前沿（SOTA）方法相当性能的同时，显著减少了总训练与推理时间。本工作的实现代码已公开于https://ruiyangju.github.io/MFE-GAN。



## **11. Beyond Memorization: Gradient Projection Enables Selective Learning in Diffusion Models**

超越记忆：梯度投影实现扩散模型的选择性学习 cs.LG

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.11194v1) [paper-pdf](https://arxiv.org/pdf/2512.11194v1)

**Confidence**: 0.95

**Authors**: Divya Kothandaraman, Jaclyn Pytlarz

**Abstract**: Memorization in large-scale text-to-image diffusion models poses significant security and intellectual property risks, enabling adversarial attribute extraction and the unauthorized reproduction of sensitive or proprietary features. While conventional dememorization techniques, such as regularization and data filtering, limit overfitting to specific training examples, they fail to systematically prevent the internalization of prohibited concept-level features. Simply discarding all images containing a sensitive feature wastes invaluable training data, necessitating a method for selective unlearning at the concept level.   To address this, we introduce a Gradient Projection Framework designed to enforce a stringent requirement of concept-level feature exclusion. Our defense operates during backpropagation by systematically identifying and excising training signals aligned with embeddings of prohibited attributes. Specifically, we project each gradient update onto the orthogonal complement of the sensitive feature's embedding space, thereby zeroing out its influence on the model's weights. Our method integrates seamlessly into standard diffusion model training pipelines and complements existing defenses. We analyze our method against an adversary aiming for feature extraction. In extensive experiments, we demonstrate that our framework drastically reduces memorization while rigorously preserving generation quality and semantic fidelity. By reframing memorization control as selective learning, our approach establishes a new paradigm for IP-safe and privacy-preserving generative AI.

摘要: 大规模文本到图像扩散模型中的记忆现象带来了严重的安全和知识产权风险，使得对抗性属性提取和敏感或专有特征的未经授权复制成为可能。虽然传统的去记忆化技术（如正则化和数据过滤）能限制对特定训练样本的过拟合，但无法系统性地防止模型内化禁止的概念级特征。简单丢弃所有包含敏感特征的图像会浪费宝贵的训练数据，因此需要一种在概念层面实现选择性遗忘的方法。为此，我们提出了梯度投影框架，旨在强制执行概念级特征排除的严格约束。我们的防御机制在反向传播过程中运行，通过系统性地识别并剔除与禁止属性嵌入对齐的训练信号。具体而言，我们将每个梯度更新投影到敏感特征嵌入空间的正交补空间上，从而消除其对模型权重的影响。该方法可无缝集成到标准扩散模型训练流程中，并与现有防御措施形成互补。我们针对旨在进行特征提取的对抗者分析了本方法的有效性。大量实验表明，我们的框架在严格保持生成质量和语义保真度的同时，显著降低了记忆效应。通过将记忆控制重新定义为选择性学习，我们的方法为知识产权安全和隐私保护的生成式AI建立了新范式。



## **12. Interpreting Structured Perturbations in Image Protection Methods for Diffusion Models**

解读扩散模型图像保护方法中的结构化扰动 cs.CV

32 pages, 17 figures, 1 table, 5 algorithms, preprint

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08329v1) [paper-pdf](https://arxiv.org/pdf/2512.08329v1)

**Confidence**: 0.95

**Authors**: Michael R. Martin, Garrick Chan, Kwan-Liu Ma

**Abstract**: Recent image protection mechanisms such as Glaze and Nightshade introduce imperceptible, adversarially designed perturbations intended to disrupt downstream text-to-image generative models. While their empirical effectiveness is known, the internal structure, detectability, and representational behavior of these perturbations remain poorly understood. This study provides a systematic, explainable AI analysis using a unified framework that integrates white-box feature-space inspection and black-box signal-level probing. Through latent-space clustering, feature-channel activation analysis, occlusion-based spatial sensitivity mapping, and frequency-domain characterization, we show that protection mechanisms operate as structured, low-entropy perturbations tightly coupled to underlying image content across representational, spatial, and spectral domains. Protected images preserve content-driven feature organization with protection-specific substructure rather than inducing global representational drift. Detectability is governed by interacting effects of perturbation entropy, spatial deployment, and frequency alignment, with sequential protection amplifying detectable structure rather than suppressing it. Frequency-domain analysis shows that Glaze and Nightshade redistribute energy along dominant image-aligned frequency axes rather than introducing diffuse noise. These findings indicate that contemporary image protection operates through structured feature-level deformation rather than semantic dislocation, explaining why protection signals remain visually subtle yet consistently detectable. This work advances the interpretability of adversarial image protection and informs the design of future defenses and detection strategies for generative AI systems.

摘要: 近期出现的图像保护机制（如Glaze和Nightshade）引入了难以察觉的对抗性扰动，旨在干扰下游文本到图像生成模型。尽管其经验有效性已得到验证，但这些扰动的内部结构、可检测性和表征行为仍鲜为人知。本研究通过一个统一框架，结合白盒特征空间检测与黑盒信号级探测，提供了系统化的可解释AI分析。通过潜在空间聚类、特征通道激活分析、基于遮挡的空间敏感性映射和频域表征，我们发现保护机制表现为结构化、低熵的扰动，在表征、空间和频谱域中与底层图像内容紧密耦合。受保护图像保留了内容驱动的特征组织，并具有保护特定的子结构，而非引发全局表征漂移。可检测性受扰动熵、空间部署和频率对齐的交互效应控制，顺序保护会放大而非抑制可检测结构。频域分析表明，Glaze和Nightshade沿图像主导频率轴重新分配能量，而非引入扩散噪声。这些发现表明，当代图像保护通过结构化特征级形变而非语义错位实现，解释了保护信号为何在视觉上保持微妙却始终可检测。本研究推进了对抗性图像保护的可解释性，并为生成式AI系统未来防御与检测策略的设计提供了参考。



## **13. TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows**

TwinFlow：基于自对抗流实现大模型的一步生成 cs.CV

arxiv v0

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.05150v1) [paper-pdf](https://arxiv.org/pdf/2512.05150v1)

**Confidence**: 0.95

**Authors**: Zhenglin Cheng, Peng Sun, Jianguo Li, Tao Lin

**Abstract**: Recent advances in large multi-modal generative models have demonstrated impressive capabilities in multi-modal generation, including image and video generation. These models are typically built upon multi-step frameworks like diffusion and flow matching, which inherently limits their inference efficiency (requiring 40-100 Number of Function Evaluations (NFEs)). While various few-step methods aim to accelerate the inference, existing solutions have clear limitations. Prominent distillation-based methods, such as progressive and consistency distillation, either require an iterative distillation procedure or show significant degradation at very few steps (< 4-NFE). Meanwhile, integrating adversarial training into distillation (e.g., DMD/DMD2 and SANA-Sprint) to enhance performance introduces training instability, added complexity, and high GPU memory overhead due to the auxiliary trained models. To this end, we propose TwinFlow, a simple yet effective framework for training 1-step generative models that bypasses the need of fixed pretrained teacher models and avoids standard adversarial networks during training, making it ideal for building large-scale, efficient models. On text-to-image tasks, our method achieves a GenEval score of 0.83 in 1-NFE, outperforming strong baselines like SANA-Sprint (a GAN loss-based framework) and RCGM (a consistency-based framework). Notably, we demonstrate the scalability of TwinFlow by full-parameter training on Qwen-Image-20B and transform it into an efficient few-step generator. With just 1-NFE, our approach matches the performance of the original 100-NFE model on both the GenEval and DPG-Bench benchmarks, reducing computational cost by $100\times$ with minor quality degradation. Project page is available at https://zhenglin-cheng.com/twinflow.

摘要: 近期大规模多模态生成模型在多模态生成（包括图像和视频生成）方面展现出令人印象深刻的能力。这些模型通常基于扩散模型和流匹配等多步框架构建，这本质上限制了其推理效率（需要40-100次函数评估）。虽然各种少步方法旨在加速推理，但现有解决方案存在明显局限。主流的基于蒸馏的方法（如渐进蒸馏和一致性蒸馏）要么需要迭代蒸馏过程，要么在极少数步骤（<4-NFE）下表现出显著性能下降。同时，将对抗训练融入蒸馏过程（如DMD/DMD2和SANA-Sprint）以提升性能会引入训练不稳定性、增加复杂性，并因辅助训练模型导致高GPU内存开销。为此，我们提出TwinFlow——一个简单而有效的训练一步生成模型的框架，它绕过了对固定预训练教师模型的需求，并避免了训练过程中的标准对抗网络，使其成为构建大规模高效模型的理想选择。在文本到图像任务中，我们的方法在1-NFE下获得0.83的GenEval分数，优于SANA-Sprint（基于GAN损失的框架）和RCGM（基于一致性的框架）等强基线。值得注意的是，我们通过在Qwen-Image-20B上进行全参数训练，展示了TwinFlow的可扩展性，并将其转化为高效的少步生成器。仅需1-NFE，我们的方法在GenEval和DPG-Bench基准测试中即可达到原始100-NFE模型的性能，以微小的质量损失将计算成本降低100倍。项目页面见https://zhenglin-cheng.com/twinflow。



## **14. Seedream 4.0: Toward Next-generation Multimodal Image Generation**

Seedream 4.0：迈向下一代多模态图像生成 cs.CV

Seedream 4.0/4.5 Technical Report

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2509.20427v3) [paper-pdf](https://arxiv.org/pdf/2509.20427v3)

**Confidence**: 0.95

**Authors**: Team Seedream, :, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, Xiaowen Jian, Huafeng Kuang, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, Wei Liu, Yanzuo Lu, Zhengxiong Luo, Tongtong Ou, Guang Shi, Yichun Shi, Shiqi Sun, Yu Tian, Zhi Tian, Peng Wang, Rui Wang, Xun Wang, Ye Wang, Guofeng Wu, Jie Wu, Wenxu Wu, Yonghui Wu, Xin Xia, Xuefeng Xiao, Shuang Xu, Xin Yan, Ceyuan Yang, Jianchao Yang, Zhonghua Zhai, Chenlin Zhang, Heng Zhang, Qi Zhang, Xinyu Zhang, Yuwei Zhang, Shijia Zhao, Wenliang Zhao, Wenjia Zhu

**Abstract**: We introduce Seedream 4.0, an efficient and high-performance multimodal image generation system that unifies text-to-image (T2I) synthesis, image editing, and multi-image composition within a single framework. We develop a highly efficient diffusion transformer with a powerful VAE which also can reduce the number of image tokens considerably. This allows for efficient training of our model, and enables it to fast generate native high-resolution images (e.g., 1K-4K). Seedream 4.0 is pretrained on billions of text-image pairs spanning diverse taxonomies and knowledge-centric concepts. Comprehensive data collection across hundreds of vertical scenarios, coupled with optimized strategies, ensures stable and large-scale training, with strong generalization. By incorporating a carefully fine-tuned VLM model, we perform multi-modal post-training for training both T2I and image editing tasks jointly. For inference acceleration, we integrate adversarial distillation, distribution matching, and quantization, as well as speculative decoding. It achieves an inference time of up to 1.8 seconds for generating a 2K image (without a LLM/VLM as PE model). Comprehensive evaluations reveal that Seedream 4.0 can achieve state-of-the-art results on both T2I and multimodal image editing. In particular, it demonstrates exceptional multimodal capabilities in complex tasks, including precise image editing and in-context reasoning, and also allows for multi-image reference, and can generate multiple output images. This extends traditional T2I systems into an more interactive and multidimensional creative tool, pushing the boundary of generative AI for both creativity and professional applications. We further scale our model and data as Seedream 4.5. Seedream 4.0 and Seedream 4.5 are accessible on Volcano Engine https://www.volcengine.com/experience/ark?launch=seedream.

摘要: 我们推出Seedream 4.0，这是一个高效、高性能的多模态图像生成系统，将文本到图像（T2I）合成、图像编辑和多图像组合统一在单一框架中。我们开发了一种高效的扩散Transformer，配备强大的VAE，能够显著减少图像token数量。这使得模型训练更加高效，并能快速生成原生高分辨率图像（例如1K-4K）。Seedream 4.0在涵盖多样化分类和知识中心概念的数十亿文本-图像对上进行预训练。通过数百个垂直场景的全面数据收集与优化策略，确保了稳定的大规模训练和强大的泛化能力。通过整合精心微调的VLM模型，我们执行多模态后训练，联合训练T2I和图像编辑任务。为加速推理，我们集成了对抗蒸馏、分布匹配、量化以及推测解码技术。在生成2K图像时（不使用LLM/VLM作为PE模型），推理时间最快可达1.8秒。综合评估表明，Seedream 4.0在T2I和多模态图像编辑任务上均能达到最先进水平。特别是在复杂任务中展现出卓越的多模态能力，包括精确图像编辑和上下文推理，同时支持多图像参考并生成多个输出图像。这将传统T2I系统扩展为更具交互性和多维度的创作工具，推动了生成式AI在创意和专业应用领域的边界。我们进一步扩展了模型和数据，推出Seedream 4.5。Seedream 4.0和Seedream 4.5已在火山引擎平台开放访问：https://www.volcengine.com/experience/ark?launch=seedream。



## **15. If generative AI is the answer, what is the question?**

如果生成式AI是答案，那么问题是什么？ cs.LG

To appear as a book chapter in a Springer book titled "Statistical Foundations and Applications of Artificial Intelligence, Machine Learning and Deep Learning" and edited by S. Ejaz Ahmed, Pierre Alquier, Yi Li, Shuangge Ma

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2509.06120v2) [paper-pdf](https://arxiv.org/pdf/2509.06120v2)

**Confidence**: 0.95

**Authors**: Ambuj Tewari

**Abstract**: Beginning with text and images, generative AI has expanded to audio, video, computer code, and molecules. Yet, if generative AI is the answer, what is the question? We explore the foundations of generation as a distinct machine learning task with connections to prediction, compression, and decision-making. We survey five major generative model families: autoregressive models, variational autoencoders, normalizing flows, generative adversarial networks, and diffusion models. We then introduce a probabilistic framework that emphasizes the distinction between density estimation and generation. We review a game-theoretic framework with a two-player adversary-learner setup to study generation. We discuss post-training modifications that prepare generative models for deployment. We end by highlighting some important topics in socially responsible generation such as privacy, detection of AI-generated content, and copyright and IP. We adopt a task-first framing of generation, focusing on what generation is as a machine learning problem, rather than only on how models implement it.

摘要: 从文本和图像开始，生成式AI已扩展到音频、视频、计算机代码和分子领域。然而，如果生成式AI是答案，那么问题是什么？我们探讨了生成作为一项独特机器学习任务的基础，及其与预测、压缩和决策的联系。我们调研了五大主要生成模型家族：自回归模型、变分自编码器、归一化流、生成对抗网络和扩散模型。随后，我们引入了一个强调密度估计与生成之间区别的概率框架。我们回顾了采用双玩家对抗-学习者设置来研究生成的博弈论框架。我们讨论了为部署生成模型所做的训练后修改。最后，我们重点介绍了社会负责任生成中的一些重要议题，如隐私、AI生成内容检测以及版权与知识产权。我们采用任务优先的生成框架，聚焦于生成作为机器学习问题的本质，而非仅关注模型如何实现它。



## **16. Vocoder-Projected Feature Discriminator**

声码器投影特征判别器 cs.SD

Accepted to Interspeech 2025. Project page: https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/vpfd/

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.17874v2) [paper-pdf](https://arxiv.org/pdf/2508.17874v2)

**Confidence**: 0.95

**Authors**: Takuhiro Kaneko, Hirokazu Kameoka, Kou Tanaka, Yuto Kondo

**Abstract**: In text-to-speech (TTS) and voice conversion (VC), acoustic features, such as mel spectrograms, are typically used as synthesis or conversion targets owing to their compactness and ease of learning. However, because the ultimate goal is to generate high-quality waveforms, employing a vocoder to convert these features into waveforms and applying adversarial training in the time domain is reasonable. Nevertheless, upsampling the waveform introduces significant time and memory overheads. To address this issue, we propose a vocoder-projected feature discriminator (VPFD), which uses vocoder features for adversarial training. Experiments on diffusion-based VC distillation demonstrated that a pretrained and frozen vocoder feature extractor with a single upsampling step is necessary and sufficient to achieve a VC performance comparable to that of waveform discriminators while reducing the training time and memory consumption by 9.6 and 11.4 times, respectively.

摘要: 在文本转语音（TTS）和语音转换（VC）任务中，梅尔频谱等声学特征因其紧凑性和易学习性，通常被用作合成或转换目标。然而，由于最终目标是生成高质量波形，使用声码器将这些特征转换为波形并在时域进行对抗训练是合理的。但波形上采样会带来显著的时间和内存开销。为解决此问题，我们提出了一种声码器投影特征判别器（VPFD），利用声码器特征进行对抗训练。基于扩散的语音转换蒸馏实验表明，使用预训练且固定的声码器特征提取器配合单次上采样步骤，即可实现与波形判别器相当的语音转换性能，同时将训练时间和内存消耗分别降低9.6倍和11.4倍。



## **17. On the detection of synthetic images generated by diffusion models**

基于扩散模型的合成图像检测研究 cs.CV

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2211.00680v1) [paper-pdf](https://arxiv.org/pdf/2211.00680v1)

**Confidence**: 0.95

**Authors**: Riccardo Corvi, Davide Cozzolino, Giada Zingarini, Giovanni Poggi, Koki Nagano, Luisa Verdoliva

**Abstract**: Over the past decade, there has been tremendous progress in creating synthetic media, mainly thanks to the development of powerful methods based on generative adversarial networks (GAN). Very recently, methods based on diffusion models (DM) have been gaining the spotlight. In addition to providing an impressive level of photorealism, they enable the creation of text-based visual content, opening up new and exciting opportunities in many different application fields, from arts to video games. On the other hand, this property is an additional asset in the hands of malicious users, who can generate and distribute fake media perfectly adapted to their attacks, posing new challenges to the media forensic community. With this work, we seek to understand how difficult it is to distinguish synthetic images generated by diffusion models from pristine ones and whether current state-of-the-art detectors are suitable for the task. To this end, first we expose the forensics traces left by diffusion models, then study how current detectors, developed for GAN-generated images, perform on these new synthetic images, especially in challenging social-networks scenarios involving image compression and resizing. Datasets and code are available at github.com/grip-unina/DMimageDetection.

摘要: 过去十年间，合成媒体技术取得了巨大进展，这主要得益于基于生成对抗网络（GAN）的强大方法的发展。最近，基于扩散模型（DM）的方法开始受到广泛关注。除了提供令人印象深刻的照片级真实感外，这些方法还能创建基于文本的视觉内容，为从艺术到视频游戏等众多应用领域开辟了新的机遇。然而，这一特性也为恶意用户提供了额外工具，他们可以生成并传播完全适配其攻击的虚假媒体，给媒体取证领域带来了新的挑战。本研究旨在探讨区分扩散模型生成的合成图像与原始图像的难度，并评估当前最先进的检测器是否适用于此任务。为此，我们首先揭示了扩散模型留下的取证痕迹，然后研究了当前针对GAN生成图像开发的检测器在这些新型合成图像上的表现，特别是在涉及图像压缩和尺寸调整的具有挑战性的社交媒体场景中。数据集和代码可在github.com/grip-unina/DMimageDetection获取。



## **18. CloneShield: A Framework for Universal Perturbation Against Zero-Shot Voice Cloning**

CloneShield：一种针对零样本语音克隆的通用扰动框架 cs.SD

10pages, 4figures

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2505.19119v1) [paper-pdf](https://arxiv.org/pdf/2505.19119v1)

**Confidence**: 0.95

**Authors**: Renyuan Li, Zhibo Liang, Haichuan Zhang, Tianyu Shi, Zhiyuan Cheng, Jia Shi, Carl Yang, Mingjie Tang

**Abstract**: Recent breakthroughs in text-to-speech (TTS) voice cloning have raised serious privacy concerns, allowing highly accurate vocal identity replication from just a few seconds of reference audio, while retaining the speaker's vocal authenticity. In this paper, we introduce CloneShield, a universal time-domain adversarial perturbation framework specifically designed to defend against zero-shot voice cloning. Our method provides protection that is robust across speakers and utterances, without requiring any prior knowledge of the synthesized text. We formulate perturbation generation as a multi-objective optimization problem, and propose Multi-Gradient Descent Algorithm (MGDA) to ensure the robust protection across diverse utterances. To preserve natural auditory perception for users, we decompose the adversarial perturbation via Mel-spectrogram representations and fine-tune it for each sample. This design ensures imperceptibility while maintaining strong degradation effects on zero-shot cloned outputs. Experiments on three state-of-the-art zero-shot TTS systems, five benchmark datasets and evaluations from 60 human listeners demonstrate that our method preserves near-original audio quality in protected inputs (PESQ = 3.90, SRS = 0.93) while substantially degrading both speaker similarity and speech quality in cloned samples (PESQ = 1.07, SRS = 0.08).

摘要: 近期文本转语音（TTS）语音克隆技术的突破引发了严重的隐私担忧，仅需几秒参考音频即可实现高度准确的声音身份复制，同时保持说话者的声音真实性。本文提出CloneShield，一种专门设计用于防御零样本语音克隆的通用时域对抗扰动框架。该方法提供跨说话者和话语的鲁棒保护，无需任何合成文本的先验知识。我们将扰动生成构建为多目标优化问题，并提出多梯度下降算法（MGDA）以确保对不同话语的鲁棒保护。为保持用户自然听觉感知，我们通过梅尔频谱图表示分解对抗扰动，并为每个样本进行微调。该设计在保持强效降解零样本克隆输出的同时确保不可感知性。在三个最先进的零样本TTS系统、五个基准数据集和60名人类听者的评估实验中，我们的方法在受保护输入中保持了接近原始的音频质量（PESQ = 3.90, SRS = 0.93），同时显著降低了克隆样本的说话人相似度和语音质量（PESQ = 1.07, SRS = 0.08）。



## **19. AvatarSync: Rethinking Talking-Head Animation through Phoneme-Guided Autoregressive Perspective**

AvatarSync：基于音素引导自回归视角的说话头部动画方法再思考 cs.CV

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2509.12052v2) [paper-pdf](https://arxiv.org/pdf/2509.12052v2)

**Confidence**: 0.90

**Authors**: Yuchen Deng, Xiuyang Wu, Hai-Tao Zheng, Suiyang Zhang, Yi He, Yuxing Han

**Abstract**: Talking-head animation focuses on generating realistic facial videos from audio input. Following Generative Adversarial Networks (GANs), diffusion models have become the mainstream, owing to their robust generative capacities. However, inherent limitations of the diffusion process often lead to inter-frame flicker and slow inference, restricting their practical deployment. To address this, we introduce AvatarSync, an autoregressive framework on phoneme representations that generates realistic and controllable talking-head animations from a single reference image, driven directly by text or audio input. To mitigate flicker and ensure continuity, AvatarSync leverages an autoregressive pipeline that enhances temporal modeling. In addition, to ensure controllability, we introduce phonemes, which are the basic units of speech sounds, and construct a many-to-one mapping from text/audio to phonemes, enabling precise phoneme-to-visual alignment. Additionally, to further accelerate inference, we adopt a two-stage generation strategy that decouples semantic modeling from visual dynamics, and incorporate a customized Phoneme-Frame Causal Attention Mask to support multi-step parallel acceleration. Extensive experiments conducted on both Chinese (CMLR) and English (HDTF) datasets demonstrate that AvatarSync outperforms existing talking-head animation methods in visual fidelity, temporal consistency, and computational efficiency, providing a scalable and controllable solution.

摘要: 说话头部动画旨在根据音频输入生成逼真的面部视频。继生成对抗网络（GANs）之后，扩散模型凭借其强大的生成能力已成为主流方法。然而，扩散过程固有的局限性常导致帧间闪烁和推理速度缓慢，限制了其实际部署。为此，我们提出AvatarSync——一种基于音素表征的自回归框架，能够通过单张参考图像，在文本或音频的直接驱动下生成逼真且可控的说话头部动画。为减少闪烁并确保连续性，AvatarSync采用增强时序建模的自回归流程。同时，为实现精确控制，我们引入语音基本单元“音素”，构建从文本/音频到音素的多对一映射，实现音素与视觉特征的精准对齐。此外，为加速推理，我们采用两阶段生成策略，将语义建模与视觉动态解耦，并设计定制化的音素-帧因果注意力掩码以支持多步并行加速。在中文（CMLR）和英文（HDTF）数据集上的大量实验表明，AvatarSync在视觉保真度、时序一致性和计算效率方面均优于现有说话头部动画方法，提供了可扩展且可控的解决方案。



## **20. HyperNet-Adaptation for Diffusion-Based Test Case Generation**

基于扩散模型的测试用例生成的超网络适配方法 cs.LG

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.15041v1) [paper-pdf](https://arxiv.org/pdf/2601.15041v1)

**Confidence**: 0.85

**Authors**: Oliver Weißl, Vincenzo Riccio, Severin Kacianka, Andrea Stocco

**Abstract**: The increasing deployment of deep learning systems requires systematic evaluation of their reliability in real-world scenarios. Traditional gradient-based adversarial attacks introduce small perturbations that rarely correspond to realistic failures and mainly assess robustness rather than functional behavior. Generative test generation methods offer an alternative but are often limited to simple datasets or constrained input domains. Although diffusion models enable high-fidelity image synthesis, their computational cost and limited controllability restrict their applicability to large-scale testing. We present HyNeA, a generative testing method that enables direct and efficient control over diffusion-based generation. HyNeA provides dataset-free controllability through hypernetworks, allowing targeted manipulation of the generative process without relying on architecture-specific conditioning mechanisms or dataset-driven adaptations such as fine-tuning. HyNeA employs a distinct training strategy that supports instance-level tuning to identify failure-inducing test cases without requiring datasets that explicitly contain examples of similar failures. This approach enables the targeted generation of realistic failure cases at substantially lower computational cost than search-based methods. Experimental results show that HyNeA improves controllability and test diversity compared to existing generative test generators and generalizes to domains where failure-labeled training data is unavailable.

摘要: 深度学习系统的日益广泛应用需要对其在真实场景中的可靠性进行系统评估。传统的基于梯度的对抗攻击引入微小扰动，这些扰动很少对应现实故障，主要评估鲁棒性而非功能行为。生成式测试生成方法提供了替代方案，但通常局限于简单数据集或受限输入域。尽管扩散模型能够实现高保真图像合成，但其计算成本和有限的可控性限制了其在大规模测试中的应用。我们提出HyNeA，一种通过超网络实现直接高效控制扩散模型生成的测试方法。HyNeA通过超网络提供无需数据集的精确控制能力，支持对生成过程进行针对性操控，无需依赖特定架构的条件机制或基于数据集的适配（如微调）。HyNeA采用独特的训练策略，支持实例级调优以识别引发故障的测试用例，无需包含类似故障示例的数据集。该方法能够以远低于基于搜索方法的计算成本，针对性生成现实故障案例。实验结果表明，与现有生成式测试生成器相比，HyNeA在可控性和测试多样性方面均有提升，并能泛化到缺乏故障标签训练数据的领域。



## **21. Diffusion-Driven Synthetic Tabular Data Generation for Enhanced DoS/DDoS Attack Classification**

基于扩散模型的合成表格数据生成以增强DoS/DDoS攻击分类 cs.CR

7 pages, 8 figures, 2025 International Conference on Signal Processing, Computation, Electronics, Power and Telecommunication (IConSCEPT), National Institute of Technology, Puducherry, India

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2601.13197v1) [paper-pdf](https://arxiv.org/pdf/2601.13197v1)

**Confidence**: 0.85

**Authors**: Aravind B, Anirud R. S., Sai Surya Teja N, Bala Subrahmanya Sriranga Navaneeth A, Karthika R, Mohankumar N

**Abstract**: Class imbalance refers to a situation where certain classes in a dataset have significantly fewer samples than oth- ers, leading to biased model performance. Class imbalance in network intrusion detection using Tabular Denoising Diffusion Probability Models (TabDDPM) for data augmentation is ad- dressed in this paper. Our approach synthesizes high-fidelity minority-class samples from the CIC-IDS2017 dataset through iterative denoising processes. For the minority classes that have smaller samples, synthetic samples were generated and merged with the original dataset. The augmented training data enables an ANN classifier to achieve near-perfect recall on previously underrepresented attack classes. These results establish diffusion models as an effective solution for tabular data imbalance in security domains, with potential applications in fraud detection and medical diagnostics.

摘要: 类别不平衡指数据集中某些类别的样本数量显著少于其他类别，导致模型性能出现偏差。本文针对网络入侵检测中的类别不平衡问题，采用表格去噪扩散概率模型（TabDDPM）进行数据增强。我们的方法通过迭代去噪过程，从CIC-IDS2017数据集中合成高保真度的少数类样本。针对样本量较少的少数类别，生成合成样本并与原始数据集合并。增强后的训练数据使ANN分类器在先前代表性不足的攻击类别上实现了接近完美的召回率。这些结果表明扩散模型是解决安全领域表格数据不平衡的有效方案，在欺诈检测和医疗诊断等领域具有潜在应用价值。



## **22. EmoLat: Text-driven Image Sentiment Transfer via Emotion Latent Space**

EmoLat：基于情感潜在空间的文本驱动图像情感迁移 cs.CV

10 pages, 5 figures

**SubmitDate**: 2026-01-17    [abs](http://arxiv.org/abs/2601.12079v1) [paper-pdf](https://arxiv.org/pdf/2601.12079v1)

**Confidence**: 0.85

**Authors**: Jing Zhang, Bingjie Fan, Jixiang Zhu, Zhe Wang

**Abstract**: We propose EmoLat, a novel emotion latent space that enables fine-grained, text-driven image sentiment transfer by modeling cross-modal correlations between textual semantics and visual emotion features. Within EmoLat, an emotion semantic graph is constructed to capture the relational structure among emotions, objects, and visual attributes. To enhance the discriminability and transferability of emotion representations, we employ adversarial regularization, aligning the latent emotion distributions across modalities. Building upon EmoLat, a cross-modal sentiment transfer framework is proposed to manipulate image sentiment via joint embedding of text and EmoLat features. The network is optimized using a multi-objective loss incorporating semantic consistency, emotion alignment, and adversarial regularization. To support effective modeling, we construct EmoSpace Set, a large-scale benchmark dataset comprising images with dense annotations on emotions, object semantics, and visual attributes. Extensive experiments on EmoSpace Set demonstrate that our approach significantly outperforms existing state-of-the-art methods in both quantitative metrics and qualitative transfer fidelity, establishing a new paradigm for controllable image sentiment editing guided by textual input. The EmoSpace Set and all the code are available at http://github.com/JingVIPLab/EmoLat.

摘要: 我们提出EmoLat，一种新颖的情感潜在空间，通过建模文本语义与视觉情感特征之间的跨模态关联，实现细粒度的文本驱动图像情感迁移。在EmoLat中，我们构建了情感语义图以捕捉情感、物体和视觉属性之间的关联结构。为增强情感表征的区分度与可迁移性，我们采用对抗正则化方法，对齐跨模态的潜在情感分布。基于EmoLat，我们提出跨模态情感迁移框架，通过文本与EmoLat特征的联合嵌入来操控图像情感。该网络采用多目标损失函数进行优化，包含语义一致性、情感对齐和对抗正则化。为支持有效建模，我们构建了EmoSpace Set大规模基准数据集，包含具有情感、物体语义和视觉属性密集标注的图像。在EmoSpace Set上的大量实验表明，我们的方法在定量指标和定性迁移保真度上均显著优于现有最先进方法，为文本引导的可控图像情感编辑建立了新范式。EmoSpace Set数据集及全部代码已公开于http://github.com/JingVIPLab/EmoLat。



## **23. Speak the Art: A Direct Speech to Image Generation Framework**

Speak the Art：一种直接语音到图像生成的框架 eess.AS

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.00827v2) [paper-pdf](https://arxiv.org/pdf/2601.00827v2)

**Confidence**: 0.85

**Authors**: Mariam Saeed, Manar Amr, Farida Adel, Nada Hassan, Nour Walid, Eman Mohamed, Mohamed Hussein, Marwan Torki

**Abstract**: Direct speech-to-image generation has recently shown promising results. However, compared to text-to-image generation, there is still a large gap to enclose. Current approaches use two stages to tackle this task: speech encoding network and image generative adversarial network (GAN). The speech encoding networks in these approaches produce embeddings that do not capture sufficient linguistic information to semantically represent the input speech. GANs suffer from issues such as non-convergence, mode collapse, and diminished gradient, which result in unstable model parameters, limited sample diversity, and ineffective generator learning, respectively. To address these weaknesses, we introduce a framework called Speak the Art (STA) which consists of a speech encoding network and a VQ-Diffusion network conditioned on speech embeddings. To improve speech embeddings, the speech encoding network is supervised by a large pre-trained image-text model during training. Replacing GANs with diffusion leads to more stable training and the generation of diverse images. Additionally, we investigate the feasibility of extending our framework to be multilingual. As a proof of concept, we trained our framework with two languages: English and Arabic. Finally, we show that our results surpass state-of-the-art models by a large margin.

摘要: 直接语音到图像生成最近显示出有前景的结果。然而，与文本到图像生成相比，仍存在较大差距。当前方法采用两阶段处理此任务：语音编码网络和图像生成对抗网络（GAN）。这些方法中的语音编码网络产生的嵌入未能捕获足够的语言信息以语义上表示输入语音。GAN存在非收敛、模式崩溃和梯度消失等问题，分别导致模型参数不稳定、样本多样性有限和生成器学习无效。为应对这些弱点，我们引入了一个名为Speak the Art（STA）的框架，它包含一个语音编码网络和一个基于语音嵌入的VQ-Diffusion网络。为改进语音嵌入，语音编码网络在训练期间受到大型预训练图像-文本模型的监督。用扩散模型替代GAN可实现更稳定的训练和多样化的图像生成。此外，我们探讨了将框架扩展至多语言的可行性。作为概念验证，我们使用英语和阿拉伯语两种语言训练了该框架。最后，我们展示了我们的结果大幅超越了最先进的模型。



## **24. CLOAK: Contrastive Guidance for Latent Diffusion-Based Data Obfuscation**

CLOAK：基于潜在扩散模型的数据混淆对比引导框架 cs.LG

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.12086v1) [paper-pdf](https://arxiv.org/pdf/2512.12086v1)

**Confidence**: 0.85

**Authors**: Xin Yang, Omid Ardakanian

**Abstract**: Data obfuscation is a promising technique for mitigating attribute inference attacks by semi-trusted parties with access to time-series data emitted by sensors. Recent advances leverage conditional generative models together with adversarial training or mutual information-based regularization to balance data privacy and utility. However, these methods often require modifying the downstream task, struggle to achieve a satisfactory privacy-utility trade-off, or are computationally intensive, making them impractical for deployment on resource-constrained mobile IoT devices. We propose Cloak, a novel data obfuscation framework based on latent diffusion models. In contrast to prior work, we employ contrastive learning to extract disentangled representations, which guide the latent diffusion process to retain useful information while concealing private information. This approach enables users with diverse privacy needs to navigate the privacy-utility trade-off with minimal retraining. Extensive experiments on four public time-series datasets, spanning multiple sensing modalities, and a dataset of facial images demonstrate that Cloak consistently outperforms state-of-the-art obfuscation techniques and is well-suited for deployment in resource-constrained settings.

摘要: 数据混淆是一种有前景的技术，可用于缓解半可信方通过访问传感器发出的时间序列数据进行的属性推断攻击。最新研究利用条件生成模型结合对抗训练或基于互信息的正则化来平衡数据隐私与效用。然而，这些方法通常需要修改下游任务，难以实现理想的隐私-效用权衡，或计算成本高昂，使其难以部署在资源受限的移动物联网设备上。我们提出了Cloak，一种基于潜在扩散模型的新型数据混淆框架。与先前工作不同，我们采用对比学习提取解耦表示，引导潜在扩散过程在保留有用信息的同时隐藏隐私信息。该方法使具有不同隐私需求的用户能够以最小化重新训练的方式实现隐私-效用权衡。在涵盖多种传感模态的四个公共时间序列数据集及一个人脸图像数据集上的大量实验表明，Cloak始终优于最先进的混淆技术，并非常适合在资源受限环境中部署。



## **25. MAGE-ID: A Multimodal Generative Framework for Intrusion Detection Systems**

MAGE-ID：一种用于入侵检测系统的多模态生成框架 cs.LG

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03375v1) [paper-pdf](https://arxiv.org/pdf/2512.03375v1)

**Confidence**: 0.85

**Authors**: Mahdi Arab Loodaricheh, Mohammad Hossein Manshaei, Anita Raja

**Abstract**: Modern Intrusion Detection Systems (IDS) face severe challenges due to heterogeneous network traffic, evolving cyber threats, and pronounced data imbalance between benign and attack flows. While generative models have shown promise in data augmentation, existing approaches are limited to single modalities and fail to capture cross-domain dependencies. This paper introduces MAGE-ID (Multimodal Attack Generator for Intrusion Detection), a diffusion-based generative framework that couples tabular flow features with their transformed images through a unified latent prior. By jointly training Transformer and CNN-based variational encoders with an EDM style denoiser, MAGE-ID achieves balanced and coherent multimodal synthesis. Evaluations on CIC-IDS-2017 and NSL-KDD demonstrate significant improvements in fidelity, diversity, and downstream detection performance over TabSyn and TabDDPM, highlighting the effectiveness of MAGE-ID for multimodal IDS augmentation.

摘要: 现代入侵检测系统（IDS）面临严峻挑战，包括异构网络流量、不断演变的网络威胁以及良性流量与攻击流量之间显著的数据不平衡。尽管生成模型在数据增强方面展现出潜力，但现有方法仅限于单一模态，无法捕捉跨域依赖关系。本文提出MAGE-ID（用于入侵检测的多模态攻击生成器），这是一种基于扩散的生成框架，通过统一的潜在先验将表格化流量特征与其转换后的图像耦合。通过联合训练基于Transformer和CNN的变分编码器与EDM风格去噪器，MAGE-ID实现了平衡且连贯的多模态合成。在CIC-IDS-2017和NSL-KDD数据集上的评估表明，相较于TabSyn和TabDDPM，MAGE-ID在保真度、多样性和下游检测性能方面均有显著提升，凸显了其在多模态IDS增强中的有效性。



## **26. DepFlow: Disentangled Speech Generation to Mitigate Semantic Bias in Depression Detection**

DepFlow：基于解耦语音生成缓解抑郁症检测中的语义偏差 cs.CL

**SubmitDate**: 2026-01-01    [abs](http://arxiv.org/abs/2601.00303v1) [paper-pdf](https://arxiv.org/pdf/2601.00303v1)

**Confidence**: 0.85

**Authors**: Yuxin Li, Xiangyu Zhang, Yifei Li, Zhiwei Guo, Haoyang Zhang, Eng Siong Chng, Cuntai Guan

**Abstract**: Speech is a scalable and non-invasive biomarker for early mental health screening. However, widely used depression datasets like DAIC-WOZ exhibit strong coupling between linguistic sentiment and diagnostic labels, encouraging models to learn semantic shortcuts. As a result, model robustness may be compromised in real-world scenarios, such as Camouflaged Depression, where individuals maintain socially positive or neutral language despite underlying depressive states. To mitigate this semantic bias, we propose DepFlow, a three-stage depression-conditioned text-to-speech framework. First, a Depression Acoustic Encoder learns speaker- and content-invariant depression embeddings through adversarial training, achieving effective disentanglement while preserving depression discriminability (ROC-AUC: 0.693). Second, a flow-matching TTS model with FiLM modulation injects these embeddings into synthesis, enabling control over depressive severity while preserving content and speaker identity. Third, a prototype-based severity mapping mechanism provides smooth and interpretable manipulation across the depression continuum. Using DepFlow, we construct a Camouflage Depression-oriented Augmentation (CDoA) dataset that pairs depressed acoustic patterns with positive/neutral content from a sentiment-stratified text bank, creating acoustic-semantic mismatches underrepresented in natural data. Evaluated across three depression detection architectures, CDoA improves macro-F1 by 9%, 12%, and 5%, respectively, consistently outperforming conventional augmentation strategies in depression Detection. Beyond enhancing robustness, DepFlow provides a controllable synthesis platform for conversational systems and simulation-based evaluation, where real clinical data remains limited by ethical and coverage constraints.

摘要: 语音作为早期心理健康筛查的可扩展且非侵入性生物标志物，在抑郁症检测中具有重要价值。然而，DAIC-WOZ等广泛使用的抑郁症数据集存在语言情感与诊断标签的强耦合性，导致模型易学习语义捷径。这种语义偏差会削弱模型在真实场景（如伪装性抑郁症）中的鲁棒性——患者虽处于抑郁状态，却使用社会正面或中性语言。为缓解此偏差，我们提出DepFlow：一个三阶段的抑郁症条件文本转语音框架。首先，抑郁症声学编码器通过对抗训练学习说话人和内容无关的抑郁症嵌入，在保持抑郁症判别能力（ROC-AUC：0.693）的同时实现有效解耦。其次，采用FiLM调制的流匹配TTS模型将嵌入注入合成过程，实现在控制抑郁严重程度的同时保持内容与说话人身份。第三，基于原型的严重程度映射机制提供跨抑郁连续体的平滑可解释操控。利用DepFlow，我们构建了面向伪装性抑郁症的增强数据集CDoA，该数据集将抑郁声学模式与情感分层文本库中的正面/中性内容配对，创建了自然数据中 underrepresented 的声学-语义错配。在三种抑郁症检测架构上的评估表明，CDoA分别将宏平均F1分数提升了9%、12%和5%， consistently 优于传统增强策略。除提升鲁棒性外，DepFlow还为对话系统和基于模拟的评估提供了可控合成平台，在真实临床数据受伦理和覆盖范围限制的场景中具有重要价值。



## **27. Arabic TTS with FastPitch: Reproducible Baselines, Adversarial Training, and Oversmoothing Analysis**

基于FastPitch的阿拉伯语TTS：可复现基线、对抗训练与过平滑分析 eess.AS

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2512.00937v1) [paper-pdf](https://arxiv.org/pdf/2512.00937v1)

**Confidence**: 0.85

**Authors**: Lars Nippert

**Abstract**: Arabic text-to-speech (TTS) remains challenging due to limited resources and complex phonological patterns. We present reproducible baselines for Arabic TTS built on the FastPitch architecture and introduce cepstral-domain metrics for analyzing oversmoothing in mel-spectrogram prediction. While traditional Lp reconstruction losses yield smooth but over-averaged outputs, the proposed metrics reveal their temporal and spectral effects throughout training. To address this, we incorporate a lightweight adversarial spectrogram loss, which trains stably and substantially reduces oversmoothing. We further explore multi-speaker Arabic TTS by augmenting FastPitch with synthetic voices generated using XTTSv2, resulting in improved prosodic diversity without loss of stability. The code, pretrained models, and training recipes are publicly available at: https://github.com/nipponjo/tts-arabic-pytorch.

摘要: 阿拉伯语文本转语音（TTS）因资源有限和音系模式复杂而面临挑战。我们基于FastPitch架构构建了可复现的阿拉伯语TTS基线，并引入倒谱域指标用于分析梅尔频谱预测中的过平滑现象。传统Lp重建损失会产生平滑但过度平均化的输出，而所提指标揭示了其在训练过程中的时域和频域效应。为此，我们引入轻量级对抗频谱损失，该损失训练稳定且显著减少过平滑。我们进一步通过XTTSv2生成的合成语音增强FastPitch，探索多说话人阿拉伯语TTS，在保持稳定性的同时提升了韵律多样性。代码、预训练模型和训练方案已公开：https://github.com/nipponjo/tts-arabic-pytorch。



## **28. RealGen: Photorealistic Text-to-Image Generation via Detector-Guided Rewards**

RealGen：通过检测器引导奖励实现照片级真实感的文本到图像生成 cs.CV

**SubmitDate**: 2025-11-29    [abs](http://arxiv.org/abs/2512.00473v1) [paper-pdf](https://arxiv.org/pdf/2512.00473v1)

**Confidence**: 0.85

**Authors**: Junyan Ye, Leiqi Zhu, Yuncheng Guo, Dongzhi Jiang, Zilong Huang, Yifan Zhang, Zhiyuan Yan, Haohuan Fu, Conghui He, Weijia Li

**Abstract**: With the continuous advancement of image generation technology, advanced models such as GPT-Image-1 and Qwen-Image have achieved remarkable text-to-image consistency and world knowledge However, these models still fall short in photorealistic image generation. Even on simple T2I tasks, they tend to produce " fake" images with distinct AI artifacts, often characterized by "overly smooth skin" and "oily facial sheens". To recapture the original goal of "indistinguishable-from-reality" generation, we propose RealGen, a photorealistic text-to-image framework. RealGen integrates an LLM component for prompt optimization and a diffusion model for realistic image generation. Inspired by adversarial generation, RealGen introduces a "Detector Reward" mechanism, which quantifies artifacts and assesses realism using both semantic-level and feature-level synthetic image detectors. We leverage this reward signal with the GRPO algorithm to optimize the entire generation pipeline, significantly enhancing image realism and detail. Furthermore, we propose RealBench, an automated evaluation benchmark employing Detector-Scoring and Arena-Scoring. It enables human-free photorealism assessment, yielding results that are more accurate and aligned with real user experience. Experiments demonstrate that RealGen significantly outperforms general models like GPT-Image-1 and Qwen-Image, as well as specialized photorealistic models like FLUX-Krea, in terms of realism, detail, and aesthetics. The code is available at https://github.com/yejy53/RealGen.

摘要: 随着图像生成技术的持续进步，GPT-Image-1和Qwen-Image等先进模型在文本到图像一致性和世界知识方面取得了显著成就。然而，这些模型在照片级真实感图像生成方面仍存在不足。即使在简单的T2I任务上，它们也倾向于生成带有明显AI伪影的“虚假”图像，通常表现为“过度光滑的皮肤”和“油光满面的面部光泽”。为了重拾“与现实难以区分”的生成初衷，我们提出了RealGen，一个照片级真实感的文本到图像生成框架。RealGen集成了用于提示优化的LLM组件和用于真实感图像生成的扩散模型。受对抗生成的启发，RealGen引入了“检测器奖励”机制，该机制通过语义级和特征级的合成图像检测器来量化伪影并评估真实感。我们利用GRPO算法结合该奖励信号来优化整个生成流程，显著提升了图像的真实感和细节。此外，我们提出了RealBench，一个采用检测器评分和竞技场评分的自动化评估基准。它实现了无需人工干预的照片级真实感评估，产生的结果更准确且更符合真实用户体验。实验表明，在真实感、细节和美学方面，RealGen显著优于GPT-Image-1和Qwen-Image等通用模型，以及FLUX-Krea等专业照片级真实感模型。代码可在https://github.com/yejy53/RealGen获取。



## **29. One-Step Diffusion Transformer for Controllable Real-World Image Super-Resolution**

用于可控真实世界图像超分辨率的一步扩散Transformer cs.CV

**SubmitDate**: 2025-11-27    [abs](http://arxiv.org/abs/2511.17138v3) [paper-pdf](https://arxiv.org/pdf/2511.17138v3)

**Confidence**: 0.85

**Authors**: Yushun Fang, Yuxiang Chen, Shibo Yin, Qiang Hu, Jiangchao Yao, Ya Zhang, Xiaoyun Zhang, Yanfeng Wang

**Abstract**: Recent advances in diffusion-based real-world image super-resolution (Real-ISR) have demonstrated remarkable perceptual quality, yet the balance between fidelity and controllability remains a problem: multi-step diffusion-based methods suffer from generative diversity and randomness, resulting in low fidelity, while one-step methods lose control flexibility due to fidelity-specific finetuning. In this paper, we present ODTSR, a one-step diffusion transformer based on Qwen-Image that performs Real-ISR considering fidelity and controllability simultaneously: a newly introduced visual stream receives low-quality images (LQ) with adjustable noise (Control Noise), and the original visual stream receives LQs with consistent noise (Prior Noise), forming the Noise-hybrid Visual Stream (NVS) design. ODTSR further employs Fidelity-aware Adversarial Training (FAA) to enhance controllability and achieve one-step inference. Extensive experiments demonstrate that ODTSR not only achieves state-of-the-art (SOTA) performance on generic Real-ISR, but also enables prompt controllability on challenging scenarios such as real-world scene text image super-resolution (STISR) of Chinese characters without training on specific datasets. Codes are available at https://github.com/RedMediaTech/ODTSR.

摘要: 基于扩散的真实世界图像超分辨率（Real-ISR）近期取得了显著进展，展现出卓越的感知质量，但保真度与可控性之间的平衡仍是难题：多步扩散方法受生成多样性和随机性影响导致保真度较低，而一步方法因针对保真度的微调而丧失控制灵活性。本文提出ODTSR——一种基于Qwen-Image的一步扩散Transformer，能同时兼顾Real-ISR的保真度与可控性：新引入的视觉流接收带可调噪声（控制噪声）的低质量图像（LQ），原始视觉流接收带一致噪声（先验噪声）的LQ，构成噪声混合视觉流（NVS）设计。ODTSR进一步采用保真度感知对抗训练（FAA）以增强可控性并实现一步推理。大量实验表明，ODTSR不仅在通用Real-ISR上达到最先进（SOTA）性能，还能在挑战性场景（如无需特定数据集训练的中文字符真实世界场景文本图像超分辨率STISR）中实现提示可控性。代码发布于https://github.com/RedMediaTech/ODTSR。



## **30. SceneGuard: Training-Time Voice Protection with Scene-Consistent Audible Background Noise**

SceneGuard：基于场景一致性可听背景噪声的训练时语音保护方法 cs.SD

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16114v1) [paper-pdf](https://arxiv.org/pdf/2511.16114v1)

**Confidence**: 0.85

**Authors**: Rui Sang, Yuxuan Liu

**Abstract**: Voice cloning technology poses significant privacy threats by enabling unauthorized speech synthesis from limited audio samples. Existing defenses based on imperceptible adversarial perturbations are vulnerable to common audio preprocessing such as denoising and compression. We propose SceneGuard, a training-time voice protection method that applies scene-consistent audible background noise to speech recordings. Unlike imperceptible perturbations, SceneGuard leverages naturally occurring acoustic scenes (e.g., airport, street, park) to create protective noise that is contextually appropriate and robust to countermeasures. We evaluate SceneGuard on text-to-speech training attacks, demonstrating 5.5% speaker similarity degradation with extremely high statistical significance (p < 10^{-15}, Cohen's d = 2.18) while preserving 98.6% speech intelligibility (STOI = 0.986). Robustness evaluation shows that SceneGuard maintains or enhances protection under five common countermeasures including MP3 compression, spectral subtraction, lowpass filtering, and downsampling. Our results suggest that audible, scene-consistent noise provides a more robust alternative to imperceptible perturbations for training-time voice protection. The source code are available at: https://github.com/richael-sang/SceneGuard.

摘要: 语音克隆技术能够通过有限音频样本实现未经授权的语音合成，对隐私构成重大威胁。现有基于不可感知对抗扰动的防御方法容易受到去噪和压缩等常见音频预处理的影响。我们提出SceneGuard，一种在训练时应用的语音保护方法，通过为语音录音添加场景一致的可听背景噪声。与不可感知扰动不同，SceneGuard利用自然声学场景（如机场、街道、公园）生成上下文适配且对抗措施鲁棒的保护性噪声。我们在文本转语音训练攻击场景中评估SceneGuard，结果显示在保持98.6%语音可懂度（STOI=0.986）的同时，说话人相似度降低5.5%且具有极高统计显著性（p<10^{-15}，Cohen's d=2.18）。鲁棒性评估表明，SceneGuard在MP3压缩、谱减法、低通滤波和下采样等五种常见对抗措施下均保持或增强保护效果。研究结果表明，可听且场景一致的噪声为训练时语音保护提供了比不可感知扰动更鲁棒的替代方案。源代码已开源：https://github.com/richael-sang/SceneGuard。



## **31. Approximate Gaussian Mapping for Generative Image Steganography**

生成式图像隐写术的近似高斯映射方法 cs.CR

13 pages

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2510.07219v2) [paper-pdf](https://arxiv.org/pdf/2510.07219v2)

**Confidence**: 0.85

**Authors**: Yuhua Xu, Wei Sun, Chengpei Tang, Jiaxing Lu, Jingying Zhou, Chen Gu

**Abstract**: Ordinary differential equation (ODE)-based diffusion models enable deterministic image synthesis, establishing a reversible mapping suitable for generative steganography. While prevailing methods strictly adhere to a standard normal prior, empirical evidence indicates that controlled deviations from this distribution reduce numerical inversion errors without compromising perceptual quality. Leveraging this observation, the Approximate Gaussian Mapping (AGM) is proposed as a linear transformation strategy that embeds secrets by modulating noise scale and variance. To balance retrieval numerical consistence and security, a two-stage decoupled optimization strategy is introduced to minimize the Kullback-Leibler divergence subject to target bit accuracy constraints. Beyond the proposed method, we conduct a mechanistic analysis of the divergent behaviors between pixel-space and latent-space architectures. The experimental results reveal that the VAE encoder enhances robustness by filtering external perturbations, whereas the structural regularization of the VAE decoder and the semantic variance introduced by text prompts jointly mask embedding artifacts to improve security. Experimental results confirm that pixel-space mplementations maximize embedding capacity for lossless channels, while latent-space approaches offer superior robustness and security suitable for adversarial environments

摘要: 基于常微分方程（ODE）的扩散模型实现了确定性图像合成，建立了适用于生成式隐写的可逆映射。尽管主流方法严格遵循标准正态先验，但实证表明，在保持感知质量的前提下，对该分布进行可控偏离可降低数值反演误差。基于此观察，本文提出近似高斯映射（AGM）作为一种线性变换策略，通过调制噪声尺度和方差来嵌入秘密信息。为平衡检索数值一致性与安全性，引入两阶段解耦优化策略，在目标比特精度约束下最小化Kullback-Leibler散度。除所提方法外，本文对像素空间与潜在空间架构的差异行为进行了机理分析。实验结果表明：VAE编码器通过过滤外部扰动增强鲁棒性，而VAE解码器的结构正则化与文本提示引入的语义方差共同掩盖嵌入伪影以提升安全性。实验证实，像素空间方案在无损信道中可实现最大嵌入容量，而潜在空间方法在对抗环境下具备更优的鲁棒性与安全性。



## **32. A Lightweight Pipeline for Noisy Speech Voice Cloning and Accurate Lip Sync Synthesis**

一种用于噪声语音克隆与精确唇形同步合成的轻量级流程 cs.SD

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12831v1) [paper-pdf](https://arxiv.org/pdf/2509.12831v1)

**Confidence**: 0.85

**Authors**: Javeria Amir, Farwa Attaria, Mah Jabeen, Umara Noor, Zahid Rashid

**Abstract**: Recent developments in voice cloning and talking head generation demonstrate impressive capabilities in synthesizing natural speech and realistic lip synchronization. Current methods typically require and are trained on large scale datasets and computationally intensive processes using clean studio recorded inputs that is infeasible in noisy or low resource environments. In this paper, we introduce a new modular pipeline comprising Tortoise text to speech. It is a transformer based latent diffusion model that can perform high fidelity zero shot voice cloning given only a few training samples. We use a lightweight generative adversarial network architecture for robust real time lip synchronization. The solution will contribute to many essential tasks concerning less reliance on massive pre training generation of emotionally expressive speech and lip synchronization in noisy and unconstrained scenarios. The modular structure of the pipeline allows an easy extension for future multi modal and text guided voice modulation and it could be used in real world systems.

摘要: 近期语音克隆与说话头部生成技术的发展，在合成自然语音和逼真唇形同步方面展现出卓越能力。现有方法通常需要大规模数据集进行训练，并依赖计算密集型处理流程，使用清洁的录音棚录制输入，这在噪声环境或资源受限场景中难以实现。本文提出一种新型模块化流程，包含Tortoise文本转语音系统——这是一种基于Transformer的潜在扩散模型，仅需少量训练样本即可实现高保真度的零样本语音克隆。我们采用轻量级生成对抗网络架构实现鲁棒的实时唇形同步。该解决方案将有助于减少对大规模预训练的依赖，在噪声和无约束场景中生成情感丰富的语音与唇形同步。流程的模块化结构便于未来扩展至多模态及文本引导的语音调制，并适用于实际系统部署。



## **33. Emotion Detection Using Conditional Generative Adversarial Networks (cGAN): A Deep Learning Approach**

基于条件生成对抗网络（cGAN）的情感检测：一种深度学习方法 cs.LG

3 pages, 2 tables, submitted for arXiv preprint

**SubmitDate**: 2025-08-06    [abs](http://arxiv.org/abs/2508.04481v1) [paper-pdf](https://arxiv.org/pdf/2508.04481v1)

**Confidence**: 0.85

**Authors**: Anushka Srivastava

**Abstract**: This paper presents a deep learning-based approach to emotion detection using Conditional Generative Adversarial Networks (cGANs). Unlike traditional unimodal techniques that rely on a single data type, we explore a multimodal framework integrating text, audio, and facial expressions. The proposed cGAN architecture is trained to generate synthetic emotion-rich data and improve classification accuracy across multiple modalities. Our experimental results demonstrate significant improvements in emotion recognition performance compared to baseline models. This work highlights the potential of cGANs in enhancing human-computer interaction systems by enabling more nuanced emotional understanding.

摘要: 本文提出了一种基于条件生成对抗网络（cGANs）的深度学习情感检测方法。与依赖单一数据类型的传统单模态技术不同，我们探索了一种融合文本、音频和面部表情的多模态框架。所提出的cGAN架构经过训练，能够生成合成的情感丰富数据，并提高跨多模态的分类准确性。实验结果表明，与基线模型相比，该方法在情感识别性能上有显著提升。这项工作突显了cGANs在增强人机交互系统方面的潜力，通过实现更细致的情感理解。



## **34. RingFormer: A Neural Vocoder with Ring Attention and Convolution-Augmented Transformer**

RingFormer：一种融合环形注意力与卷积增强Transformer的神经声码器 cs.SD

Accepted for publication in IEEE Transactions on Human-Machine Systems (THMS)

**SubmitDate**: 2025-07-19    [abs](http://arxiv.org/abs/2501.01182v2) [paper-pdf](https://arxiv.org/pdf/2501.01182v2)

**Confidence**: 0.85

**Authors**: Seongho Hong, Yong-Hoon Choi

**Abstract**: While transformers demonstrate outstanding performance across various audio tasks, their application to neural vocoders remains challenging. Neural vocoders require the generation of long audio signals at the sample level, which demands high temporal resolution. This results in significant computational costs for attention map generation and limits their ability to efficiently process both global and local information. Additionally, the sequential nature of sample generation in neural vocoders poses difficulties for real-time processing, making the direct adoption of transformers impractical. To address these challenges, we propose RingFormer, a neural vocoder that incorporates the ring attention mechanism into a lightweight transformer variant, the convolution-augmented transformer (Conformer). Ring attention effectively captures local details while integrating global information, making it well-suited for processing long sequences and enabling real-time audio generation. RingFormer is trained using adversarial training with two discriminators. The proposed model is applied to the decoder of the text-to-speech model VITS and compared with state-of-the-art vocoders such as HiFi-GAN, iSTFT-Net, and BigVGAN under identical conditions using various objective and subjective metrics. Experimental results show that RingFormer achieves comparable or superior performance to existing models, particularly excelling in real-time audio generation. Our code and audio samples are available on GitHub.

摘要: 尽管Transformer在各种音频任务中表现出色，但其在神经声码器中的应用仍面临挑战。神经声码器需要在样本级别生成长音频信号，这要求高时间分辨率，导致注意力图生成的计算成本显著，并限制了其高效处理全局与局部信息的能力。此外，神经声码器中样本生成的顺序性给实时处理带来困难，使得直接采用Transformer不切实际。为解决这些问题，我们提出了RingFormer，这是一种将环形注意力机制融入轻量级Transformer变体——卷积增强Transformer（Conformer）的神经声码器。环形注意力在整合全局信息的同时有效捕捉局部细节，使其非常适合处理长序列并实现实时音频生成。RingFormer通过使用两个判别器的对抗训练进行训练。该模型被应用于文本转语音模型VITS的解码器，并在相同条件下与HiFi-GAN、iSTFT-Net和BigVGAN等先进声码器进行了多种客观和主观指标的比较。实验结果表明，RingFormer实现了与现有模型相当或更优的性能，尤其在实时音频生成方面表现突出。我们的代码和音频样本已在GitHub上提供。



## **35. Robust Watermarking Using Generative Priors Against Image Editing: From Benchmarking to Advances**

基于生成先验的鲁棒水印技术对抗图像编辑：从基准测试到进展 cs.CV

Accepted by ICLR 2025

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2410.18775v2) [paper-pdf](https://arxiv.org/pdf/2410.18775v2)

**Confidence**: 0.85

**Authors**: Shilin Lu, Zihan Zhou, Jiayou Lu, Yuanzhi Zhu, Adams Wai-Kin Kong

**Abstract**: Current image watermarking methods are vulnerable to advanced image editing techniques enabled by large-scale text-to-image models. These models can distort embedded watermarks during editing, posing significant challenges to copyright protection. In this work, we introduce W-Bench, the first comprehensive benchmark designed to evaluate the robustness of watermarking methods against a wide range of image editing techniques, including image regeneration, global editing, local editing, and image-to-video generation. Through extensive evaluations of eleven representative watermarking methods against prevalent editing techniques, we demonstrate that most methods fail to detect watermarks after such edits. To address this limitation, we propose VINE, a watermarking method that significantly enhances robustness against various image editing techniques while maintaining high image quality. Our approach involves two key innovations: (1) we analyze the frequency characteristics of image editing and identify that blurring distortions exhibit similar frequency properties, which allows us to use them as surrogate attacks during training to bolster watermark robustness; (2) we leverage a large-scale pretrained diffusion model SDXL-Turbo, adapting it for the watermarking task to achieve more imperceptible and robust watermark embedding. Experimental results show that our method achieves outstanding watermarking performance under various image editing techniques, outperforming existing methods in both image quality and robustness. Code is available at https://github.com/Shilin-LU/VINE.

摘要: 当前图像水印方法易受大规模文本到图像模型支持的先进图像编辑技术攻击。这些模型在编辑过程中会破坏嵌入的水印，对版权保护构成重大挑战。本研究提出首个综合性基准测试W-Bench，用于评估水印方法对抗多种图像编辑技术的鲁棒性，包括图像再生、全局编辑、局部编辑及图像到视频生成。通过对11种代表性水印方法在主流编辑技术下的广泛评估，我们发现多数方法在编辑后无法检测水印。为解决此问题，我们提出VINE水印方法，在保持高图像质量的同时显著提升对抗各类图像编辑的鲁棒性。该方法包含两项关键创新：(1) 分析图像编辑的频率特性，发现模糊失真具有相似频率特征，可将其作为训练时的替代攻击以增强水印鲁棒性；(2) 利用大规模预训练扩散模型SDXL-Turbo，将其适配于水印任务以实现更隐蔽且鲁棒的水印嵌入。实验结果表明，我们的方法在各种图像编辑技术下均表现出卓越的水印性能，在图像质量和鲁棒性方面均超越现有方法。代码发布于https://github.com/Shilin-LU/VINE。



## **36. Efficient Generative Adversarial Networks for Color Document Image Enhancement and Binarization Using Multi-scale Feature Extraction**

基于多尺度特征提取的高效生成对抗网络用于彩色文档图像增强与二值化 cs.CV

Accepted to APSIPA ASC 2025

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2407.04231v2) [paper-pdf](https://arxiv.org/pdf/2407.04231v2)

**Confidence**: 0.85

**Authors**: Rui-Yang Ju, KokSheik Wong, Jen-Shiun Chiang

**Abstract**: The outcome of text recognition for degraded color documents is often unsatisfactory due to interference from various contaminants. To extract information more efficiently for text recognition, document image enhancement and binarization are often employed as preliminary steps in document analysis. Training independent generative adversarial networks (GANs) for each color channel can generate images where shadows and noise are effectively removed, which subsequently allows for efficient text information extraction. However, employing multiple GANs for different color channels requires long training and inference times. To reduce both the training and inference times of these preliminary steps, we propose an efficient method based on multi-scale feature extraction, which incorporates Haar wavelet transformation and normalization to process document images before submitting them to GANs for training. Experiment results show that our proposed method significantly reduces both the training and inference times while maintaining comparable performances when benchmarked against the state-of-the-art methods. In the best case scenario, a reduction of 10% and 26% are observed for training and inference times, respectively, while maintaining the model performance at 73.79 of Average-Score metric. The implementation of this work is available at https://github.com/RuiyangJu/Efficient_Document_Image_Binarization.

摘要: 由于各种污染物的干扰，退化彩色文档的文本识别结果往往不尽如人意。为更有效地提取文本识别所需信息，文档图像增强与二值化常作为文档分析的预处理步骤。为每个颜色通道训练独立的生成对抗网络（GANs）可生成有效去除阴影和噪声的图像，从而提升文本信息提取效率。然而，为不同颜色通道使用多个GANs会导致训练和推理时间过长。为缩短这些预处理步骤的时间，我们提出一种基于多尺度特征提取的高效方法，该方法结合哈尔小波变换和归一化技术对文档图像进行预处理后，再提交给GANs进行训练。实验结果表明，与现有先进方法相比，我们提出的方法在保持相当性能的同时，显著减少了训练和推理时间。在最佳情况下，训练时间和推理时间分别减少10%和26%，同时模型性能保持在Average-Score指标73.79的水平。本工作的实现代码已发布于https://github.com/RuiyangJu/Efficient_Document_Image_Binarization。



## **37. VocalCrypt: Novel Active Defense Against Deepfake Voice Based on Masking Effect**

VocalCrypt：基于掩蔽效应的新型深度伪造语音主动防御方法 cs.SD

9 pages, four figures

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2502.10329v1) [paper-pdf](https://arxiv.org/pdf/2502.10329v1)

**Confidence**: 0.85

**Authors**: Qingyuan Fei, Wenjie Hou, Xuan Hai, Xin Liu

**Abstract**: The rapid advancements in AI voice cloning, fueled by machine learning, have significantly impacted text-to-speech (TTS) and voice conversion (VC) fields. While these developments have led to notable progress, they have also raised concerns about the misuse of AI VC technology, causing economic losses and negative public perceptions. To address this challenge, this study focuses on creating active defense mechanisms against AI VC systems.   We propose a novel active defense method, VocalCrypt, which embeds pseudo-timbre (jamming information) based on SFS into audio segments that are imperceptible to the human ear, thereby forming systematic fragments to prevent voice cloning. This approach protects the voice without compromising its quality. In comparison to existing methods, such as adversarial noise incorporation, VocalCrypt significantly enhances robustness and real-time performance, achieving a 500\% increase in generation speed while maintaining interference effectiveness.   Unlike audio watermarking techniques, which focus on post-detection, our method offers preemptive defense, reducing implementation costs and enhancing feasibility. Extensive experiments using the Zhvoice and VCTK Corpus datasets show that our AI-cloned speech defense system performs excellently in automatic speaker verification (ASV) tests while preserving the integrity of the protected audio.

摘要: 机器学习驱动的AI语音克隆技术快速发展，显著推动了文本转语音（TTS）和语音转换（VC）领域的进步。然而，这些进展也引发了对AI VC技术滥用的担忧，导致经济损失和负面公众认知。为应对这一挑战，本研究专注于构建针对AI VC系统的主动防御机制。我们提出了一种新型主动防御方法VocalCrypt，该方法基于SFS将伪音色（干扰信息）嵌入人耳难以察觉的音频片段中，从而形成系统性片段以阻止语音克隆。此方法能在保护语音的同时不损害其质量。与现有方法（如对抗性噪声注入）相比，VocalCrypt显著增强了鲁棒性和实时性能，在保持干扰效果的同时将生成速度提升500%。不同于侧重事后检测的音频水印技术，我们的方法提供先发制人的防御，降低了实施成本并增强了可行性。基于Zhvoice和VCTK Corpus数据集的广泛实验表明，我们的AI克隆语音防御系统在自动说话人验证（ASV）测试中表现优异，同时保持了受保护音频的完整性。




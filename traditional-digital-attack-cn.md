# 传统深度学习模型 - 数字攻击
**update at 2026-01-25 10:36:50**

按分类器置信度从高到低排序。

## **1. VTarbel: Targeted Label Attack with Minimal Knowledge on Detector-enhanced Vertical Federated Learning**

VTarbel：基于检测器增强的纵向联邦学习最小知识目标标签攻击 cs.CR

Accepted by ACM Transactions on Sensor Networks (TOSN)

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2507.14625v2) [paper-pdf](https://arxiv.org/pdf/2507.14625v2)

**Confidence**: 0.95

**Authors**: Juntao Tan, Anran Li, Quanchao Liu, Peng Ran, Lan Zhang

**Abstract**: Vertical federated learning (VFL) enables multiple parties with disjoint features to collaboratively train models without sharing raw data. While privacy vulnerabilities of VFL are extensively-studied, its security threats-particularly targeted label attacks-remain underexplored. In such attacks, a passive party perturbs inputs at inference to force misclassification into adversary-chosen labels. Existing methods rely on unrealistic assumptions (e.g., accessing VFL-model's outputs) and ignore anomaly detectors deployed in real-world systems. To bridge this gap, we introduce VTarbel, a two-stage, minimal-knowledge attack framework explicitly designed to evade detector-enhanced VFL inference. During the preparation stage, the attacker selects a minimal set of high-expressiveness samples (via maximum mean discrepancy), submits them through VFL protocol to collect predicted labels, and uses these pseudo-labels to train estimated detector and surrogate model on local features. In attack stage, these models guide gradient-based perturbations of remaining samples, crafting adversarial instances that induce targeted misclassifications and evade detection. We implement VTarbel and evaluate it against four model architectures, seven multimodal datasets, and two anomaly detectors. Across all settings, VTarbel outperforms four state-of-the-art baselines, evades detection, and retains effective against three representative privacy-preserving defenses. These results reveal critical security blind spots in current VFL deployments and underscore urgent need for robust, attack-aware defenses.

摘要: 纵向联邦学习（VFL）允许多个特征空间互不相交的参与方在不共享原始数据的情况下协同训练模型。尽管VFL的隐私漏洞已被广泛研究，但其安全威胁——特别是目标标签攻击——仍未得到充分探索。在此类攻击中，被动方在推理阶段扰动输入，迫使模型将样本误分类为攻击者指定的标签。现有方法依赖不现实的假设（例如访问VFL模型的完整输出），且忽略了实际系统中部署的异常检测器。为填补这一空白，我们提出了VTarbel——一个两阶段、最小知识攻击框架，专门设计用于规避检测器增强的VFL推理系统。在准备阶段，攻击者通过最大均值差异选择最小规模的高表达能力样本集，通过VFL协议提交以收集预测标签，并利用这些伪标签在本地特征上训练估计的检测器和替代模型。在攻击阶段，这些模型指导基于梯度的样本扰动，生成既能诱导目标误分类又能规避检测的对抗样本。我们实现了VTarbel，并在四种模型架构、七个多模态数据集和两种异常检测器上进行了评估。在所有实验设置中，VTarbel均优于四种最先进的基线方法，成功规避检测，并对三种代表性隐私保护防御机制保持攻击有效性。这些结果揭示了当前VFL部署中关键的安全盲点，并凸显了开发鲁棒性、攻击感知型防御机制的紧迫性。



## **2. Power to the Clients: Federated Learning in a Dictatorship Setting**

赋权客户端：专制设定下的联邦学习 cs.LG

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2510.22149v2) [paper-pdf](https://arxiv.org/pdf/2510.22149v2)

**Confidence**: 0.95

**Authors**: Mohammadsajad Alipour, Mohammad Mohammadi Amiri

**Abstract**: Federated learning (FL) has emerged as a promising paradigm for decentralized model training, enabling multiple clients to collaboratively learn a shared model without exchanging their local data. However, the decentralized nature of FL also introduces vulnerabilities, as malicious clients can compromise or manipulate the training process. In this work, we introduce dictator clients, a novel, well-defined, and analytically tractable class of malicious participants capable of entirely erasing the contributions of all other clients from the server model, while preserving their own. We propose concrete attack strategies that empower such clients and systematically analyze their effects on the learning process. Furthermore, we explore complex scenarios involving multiple dictator clients, including cases where they collaborate, act independently, or form an alliance in order to ultimately betray one another. For each of these settings, we provide a theoretical analysis of their impact on the global model's convergence. Our theoretical algorithms and findings about the complex scenarios including multiple dictator clients are further supported by empirical evaluations on both computer vision and natural language processing benchmarks.

摘要: 联邦学习（FL）作为一种去中心化的模型训练范式崭露头角，允许多个客户端在不交换本地数据的情况下协作学习共享模型。然而，FL的去中心化特性也带来了脆弱性，恶意客户端可能破坏或操纵训练过程。本文提出“专制客户端”这一新颖、定义明确且可分析处理的恶意参与者类别，这类客户端能够完全消除其他所有客户端对服务器模型的贡献，同时保留自身贡献。我们提出了赋能此类客户端的具体攻击策略，并系统分析了其对学习过程的影响。此外，我们探讨了涉及多个专制客户端的复杂场景，包括它们协作、独立行动或结盟后最终相互背叛的情况。针对每种设定，我们提供了其对全局模型收敛影响的理论分析。关于多专制客户端复杂场景的理论算法与发现，进一步通过计算机视觉和自然语言处理基准测试的实证评估得到验证。



## **3. BREPS: Bounding-Box Robustness Evaluation of Promptable Segmentation**

BREPS：可提示分割模型的边界框鲁棒性评估 cs.CV

Accepted by AAAI2026

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.15123v1) [paper-pdf](https://arxiv.org/pdf/2601.15123v1)

**Confidence**: 0.95

**Authors**: Andrey Moskalenko, Danil Kuznetsov, Irina Dudko, Anastasiia Iasakova, Nikita Boldyrev, Denis Shepelev, Andrei Spiridonov, Andrey Kuznetsov, Vlad Shakhuro

**Abstract**: Promptable segmentation models such as SAM have established a powerful paradigm, enabling strong generalization to unseen objects and domains with minimal user input, including points, bounding boxes, and text prompts. Among these, bounding boxes stand out as particularly effective, often outperforming points while significantly reducing annotation costs. However, current training and evaluation protocols typically rely on synthetic prompts generated through simple heuristics, offering limited insight into real-world robustness. In this paper, we investigate the robustness of promptable segmentation models to natural variations in bounding box prompts. First, we conduct a controlled user study and collect thousands of real bounding box annotations. Our analysis reveals substantial variability in segmentation quality across users for the same model and instance, indicating that SAM-like models are highly sensitive to natural prompt noise. Then, since exhaustive testing of all possible user inputs is computationally prohibitive, we reformulate robustness evaluation as a white-box optimization problem over the bounding box prompt space. We introduce BREPS, a method for generating adversarial bounding boxes that minimize or maximize segmentation error while adhering to naturalness constraints. Finally, we benchmark state-of-the-art models across 10 datasets, spanning everyday scenes to medical imaging. Code - https://github.com/emb-ai/BREPS.

摘要: 以SAM为代表的可提示分割模型建立了一种强大的范式，能够通过点、边界框和文本提示等最小用户输入，实现对未见对象和领域的强泛化能力。其中，边界框提示尤为有效，通常优于点提示，同时显著降低标注成本。然而，当前的训练和评估协议通常依赖简单启发式生成的合成提示，对现实世界鲁棒性的洞察有限。本文研究了可提示分割模型对边界框提示自然变化的鲁棒性。首先，我们进行了受控用户研究，收集了数千个真实边界框标注。分析表明，同一模型和实例在不同用户间的分割质量存在显著差异，说明SAM类模型对自然提示噪声高度敏感。其次，由于穷举测试所有可能的用户输入在计算上不可行，我们将鲁棒性评估重新表述为边界框提示空间上的白盒优化问题。我们提出了BREPS方法，用于生成在遵循自然性约束的同时最小化或最大化分割误差的对抗性边界框。最后，我们在涵盖日常场景到医学影像的10个数据集上对最先进模型进行了基准测试。代码地址：https://github.com/emb-ai/BREPS。



## **4. Diffusion-Driven Deceptive Patches: Adversarial Manipulation and Forensic Detection in Facial Identity Verification**

扩散驱动的欺骗性补丁：面部身份验证中的对抗性操纵与取证检测 cs.CV

This manuscript is a preprint. A revised version of this work has been accepted for publication in the Springer Nature book Artificial Intelligence-Driven Forensics. This version includes one additional figure for completeness

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.09806v1) [paper-pdf](https://arxiv.org/pdf/2601.09806v1)

**Confidence**: 0.95

**Authors**: Shahrzad Sayyafzadeh, Hongmei Chi, Shonda Bernadin

**Abstract**: This work presents an end-to-end pipeline for generating, refining, and evaluating adversarial patches to compromise facial biometric systems, with applications in forensic analysis and security testing. We utilize FGSM to generate adversarial noise targeting an identity classifier and employ a diffusion model with reverse diffusion to enhance imperceptibility through Gaussian smoothing and adaptive brightness correction, thereby facilitating synthetic adversarial patch evasion. The refined patch is applied to facial images to test its ability to evade recognition systems while maintaining natural visual characteristics. A Vision Transformer (ViT)-GPT2 model generates captions to provide a semantic description of a person's identity for adversarial images, supporting forensic interpretation and documentation for identity evasion and recognition attacks. The pipeline evaluates changes in identity classification, captioning results, and vulnerabilities in facial identity verification and expression recognition under adversarial conditions. We further demonstrate effective detection and analysis of adversarial patches and adversarial samples using perceptual hashing and segmentation, achieving an SSIM of 0.95.

摘要: 本研究提出了一种端到端流程，用于生成、优化和评估对抗性补丁以攻击面部生物识别系统，应用于取证分析和安全测试。我们利用FGSM生成针对身份分类器的对抗性噪声，并采用扩散模型结合反向扩散过程，通过高斯平滑和自适应亮度校正增强不可感知性，从而实现合成对抗性补丁的规避。优化后的补丁应用于面部图像，测试其在保持自然视觉特征的同时规避识别系统的能力。采用Vision Transformer (ViT)-GPT2模型生成对抗性图像的身份语义描述，为身份规避和识别攻击提供取证解释与文档支持。该流程评估了对抗条件下身份分类、描述结果的变化以及面部身份验证与表情识别的脆弱性。我们进一步展示了利用感知哈希和分割技术有效检测分析对抗性补丁与对抗样本的方法，实现了0.95的SSIM指标。



## **5. DiMEx: Breaking the Cold Start Barrier in Data-Free Model Extraction via Latent Diffusion Priors**

DiMEx：通过潜在扩散先验突破数据无关模型窃取中的冷启动障碍 cs.LG

8 pages, 3 figures, 4 tables

**SubmitDate**: 2026-01-10    [abs](http://arxiv.org/abs/2601.01688v2) [paper-pdf](https://arxiv.org/pdf/2601.01688v2)

**Confidence**: 0.95

**Authors**: Yash Thesia, Meera Suthar

**Abstract**: Model stealing attacks pose an existential threat to Machine Learning as a Service (MLaaS), allowing adversaries to replicate proprietary models for a fraction of their training cost. While Data-Free Model Extraction (DFME) has emerged as a stealthy vector, it remains fundamentally constrained by the "Cold Start" problem: GAN-based adversaries waste thousands of queries converging from random noise to meaningful data. We propose DiMEx, a framework that weaponizes the rich semantic priors of pre-trained Latent Diffusion Models to bypass this initialization barrier entirely. By employing Random Embedding Bayesian Optimization (REMBO) within the generator's latent space, DiMEx synthesizes high-fidelity queries immediately, achieving 52.1 percent agreement on SVHN with just 2,000 queries - outperforming state-of-the-art GAN baselines by over 16 percent. To counter this highly semantic threat, we introduce the Hybrid Stateful Ensemble (HSE) defense, which identifies the unique "optimization trajectory" of latent-space attacks. Our results demonstrate that while DiMEx evades static distribution detectors, HSE exploits this temporal signature to suppress attack success rates to 21.6 percent with negligible latency.

摘要: 模型窃取攻击对机器学习即服务（MLaaS）构成生存性威胁，使攻击者能以极低的训练成本复制专有模型。虽然数据无关模型窃取（DFME）已成为一种隐蔽的攻击途径，但其本质上仍受限于“冷启动”问题：基于GAN的攻击者需要耗费数千次查询才能从随机噪声收敛至有意义的数据。我们提出DiMEx框架，该框架利用预训练潜在扩散模型的丰富语义先验，完全绕过此初始化障碍。通过在生成器的潜在空间中采用随机嵌入贝叶斯优化（REMBO），DiMEx能立即合成高保真查询，仅用2,000次查询即在SVHN数据集上达到52.1%的模型一致性——超越最先进的GAN基线方法超过16%。为应对这种高语义威胁，我们提出混合状态集成（HSE）防御机制，通过识别潜在空间攻击特有的“优化轨迹”进行检测。实验表明，虽然DiMEx能规避静态分布检测器，但HSE利用其时序特征将攻击成功率压制至21.6%，且延迟可忽略不计。



## **6. ZQBA: Zero Query Black-box Adversarial Attack**

ZQBA：零查询黑盒对抗攻击 cs.CV

Accepted in ICAART 2026 Conference

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2510.00769v2) [paper-pdf](https://arxiv.org/pdf/2510.00769v2)

**Confidence**: 0.95

**Authors**: Joana C. Costa, Tiago Roxo, Hugo Proença, Pedro R. M. Inácio

**Abstract**: Current black-box adversarial attacks either require multiple queries or diffusion models to produce adversarial samples that can impair the target model performance. However, these methods require training a surrogate loss or diffusion models to produce adversarial samples, which limits their applicability in real-world settings. Thus, we propose a Zero Query Black-box Adversarial (ZQBA) attack that exploits the representations of Deep Neural Networks (DNNs) to fool other networks. Instead of requiring thousands of queries to produce deceiving adversarial samples, we use the feature maps obtained from a DNN and add them to clean images to impair the classification of a target model. The results suggest that ZQBA can transfer the adversarial samples to different models and across various datasets, namely CIFAR and Tiny ImageNet. The experiments also show that ZQBA is more effective than state-of-the-art black-box attacks with a single query, while maintaining the imperceptibility of perturbations, evaluated both quantitatively (SSIM) and qualitatively, emphasizing the vulnerabilities of employing DNNs in real-world contexts. All the source code is available at https://github.com/Joana-Cabral/ZQBA.

摘要: 当前的黑盒对抗攻击方法通常需要多次查询或依赖扩散模型来生成能够损害目标模型性能的对抗样本。然而，这些方法需要训练替代损失函数或扩散模型来生成对抗样本，这限制了它们在现实场景中的适用性。为此，我们提出了一种零查询黑盒对抗攻击方法，该方法利用深度神经网络的表征来欺骗其他网络。我们无需进行数千次查询来生成欺骗性对抗样本，而是使用从DNN获取的特征图，将其添加到干净图像中以损害目标模型的分类性能。结果表明，ZQBA能够将对抗样本迁移到不同模型及跨数据集，包括CIFAR和Tiny ImageNet。实验还表明，ZQBA在单次查询条件下比现有最先进的黑盒攻击方法更有效，同时保持扰动的不可感知性，并通过定量和定性评估验证，强调了在现实场景中部署DNN的脆弱性。所有源代码可在https://github.com/Joana-Cabral/ZQBA获取。



## **7. Vulnerabilities in AI-generated Image Detection: The Challenge of Adversarial Attacks**

AI生成图像检测的脆弱性：对抗性攻击的挑战 cs.CV

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2407.20836v5) [paper-pdf](https://arxiv.org/pdf/2407.20836v5)

**Confidence**: 0.95

**Authors**: Yunfeng Diao, Naixin Zhai, Changtao Miao, Zitong Yu, Xingxing Wei, Xun Yang, Meng Wang

**Abstract**: Recent advancements in image synthesis, particularly with the advent of GAN and Diffusion models, have amplified public concerns regarding the dissemination of disinformation. To address such concerns, numerous AI-generated Image (AIGI) Detectors have been proposed and achieved promising performance in identifying fake images. However, there still lacks a systematic understanding of the adversarial robustness of AIGI detectors. In this paper, we examine the vulnerability of state-of-the-art AIGI detectors against adversarial attack under white-box and black-box settings, which has been rarely investigated so far. To this end, we propose a new method to attack AIGI detectors. First, inspired by the obvious difference between real images and fake images in the frequency domain, we add perturbations under the frequency domain to push the image away from its original frequency distribution. Second, we explore the full posterior distribution of the surrogate model to further narrow this gap between heterogeneous AIGI detectors, e.g., transferring adversarial examples across CNNs and ViTs. This is achieved by introducing a novel post-train Bayesian strategy that turns a single surrogate into a Bayesian one, capable of simulating diverse victim models using one pre-trained surrogate, without the need for re-training. We name our method as Frequency-based Post-train Bayesian Attack, or FPBA. Through FPBA, we demonstrate that adversarial attacks pose a real threat to AIGI detectors. FPBA can deliver successful black-box attacks across various detectors, generators, defense methods, and even evade cross-generator and compressed image detection, which are crucial real-world detection scenarios. Our code is available at https://github.com/onotoa/fpba.

摘要: 近年来，随着GAN和Diffusion模型等图像合成技术的进步，公众对虚假信息传播的担忧日益加剧。为应对这一问题，众多AI生成图像（AIGI）检测器被提出，并在识别伪造图像方面展现出良好性能。然而，目前对AIGI检测器对抗鲁棒性的系统性理解仍显不足。本文首次系统研究了白盒与黑盒设置下先进AIGI检测器面对对抗攻击的脆弱性。为此，我们提出一种新的AIGI检测器攻击方法：首先，基于真实图像与伪造图像在频域的显著差异，通过在频域添加扰动使图像偏离原始频率分布；其次，通过探索代理模型的全后验分布，进一步缩小异构AIGI检测器（如CNN与ViT模型）间的差异。我们创新性地引入后训练贝叶斯策略，将单一代理模型转化为贝叶斯模型，仅需一个预训练代理即可模拟多样化受害模型，无需重新训练。该方法被命名为基于频率的后训练贝叶斯攻击（FPBA）。通过FPBA实验，我们证明对抗攻击对AIGI检测器构成实质性威胁：FPBA能成功实现跨检测器、跨生成器、跨防御方法的黑盒攻击，甚至可规避跨生成器检测与压缩图像检测等关键现实场景。代码已开源：https://github.com/onotoa/fpba。



## **8. DiffProtect: Generate Adversarial Examples with Diffusion Models for Facial Privacy Protection**

DiffProtect：利用扩散模型生成对抗样本以保护面部隐私 cs.CV

Code is at https://github.com/joellliu/DiffProtect/

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2305.13625v4) [paper-pdf](https://arxiv.org/pdf/2305.13625v4)

**Confidence**: 0.95

**Authors**: Jiang Liu, Chun Pong Lau, Zhongliang Guo, Yuxiang Guo, Zhaoyang Wang, Rama Chellappa

**Abstract**: The increasingly pervasive facial recognition (FR) systems raise serious concerns about personal privacy, especially for billions of users who have publicly shared their photos on social media. Several attempts have been made to protect individuals from being identified by unauthorized FR systems utilizing adversarial attacks to generate encrypted face images. However, existing methods suffer from poor visual quality or low attack success rates, which limit their utility. Recently, diffusion models have achieved tremendous success in image generation. In this work, we ask: can diffusion models be used to generate adversarial examples to improve both visual quality and attack performance? We propose DiffProtect, which utilizes a diffusion autoencoder to generate semantically meaningful perturbations on FR systems. Extensive experiments demonstrate that DiffProtect produces more natural-looking encrypted images than state-of-the-art methods while achieving significantly higher attack success rates, e.g., 24.5% and 25.1% absolute improvements on the CelebA-HQ and FFHQ datasets.

摘要: 日益普及的面部识别（FR）系统引发了严重的个人隐私担忧，尤其是对于在社交媒体上公开分享照片的数十亿用户。已有多种尝试通过利用对抗攻击生成加密人脸图像来保护个人免受未经授权的FR系统识别。然而，现有方法存在视觉质量差或攻击成功率低的问题，限制了其实用性。最近，扩散模型在图像生成领域取得了巨大成功。在本研究中，我们探讨：扩散模型能否用于生成对抗样本，以同时提升视觉质量和攻击性能？我们提出了DiffProtect，该方法利用扩散自编码器在FR系统上生成具有语义意义的扰动。大量实验表明，DiffProtect生成的加密图像比现有最先进方法看起来更自然，同时实现了显著更高的攻击成功率，例如在CelebA-HQ和FFHQ数据集上分别实现了24.5%和25.1%的绝对提升。



## **9. MORE: Multi-Objective Adversarial Attacks on Speech Recognition**

MORE：语音识别的多目标对抗攻击 eess.AS

19 pages

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.01852v2) [paper-pdf](https://arxiv.org/pdf/2601.01852v2)

**Confidence**: 0.95

**Authors**: Xiaoxue Gao, Zexin Li, Yiming Chen, Nancy F. Chen

**Abstract**: The emergence of large-scale automatic speech recognition (ASR) models such as Whisper has greatly expanded their adoption across diverse real-world applications. Ensuring robustness against even minor input perturbations is therefore critical for maintaining reliable performance in real-time environments. While prior work has mainly examined accuracy degradation under adversarial attacks, robustness with respect to efficiency remains largely unexplored. This narrow focus provides only a partial understanding of ASR model vulnerabilities. To address this gap, we conduct a comprehensive study of ASR robustness under multiple attack scenarios. We introduce MORE, a multi-objective repetitive doubling encouragement attack, which jointly degrades recognition accuracy and inference efficiency through a hierarchical staged repulsion-anchoring mechanism. Specifically, we reformulate multi-objective adversarial optimization into a hierarchical framework that sequentially achieves the dual objectives. To further amplify effectiveness, we propose a novel repetitive encouragement doubling objective (REDO) that induces duplicative text generation by maintaining accuracy degradation and periodically doubling the predicted sequence length. Overall, MORE compels ASR models to produce incorrect transcriptions at a substantially higher computational cost, triggered by a single adversarial input. Experiments show that MORE consistently yields significantly longer transcriptions while maintaining high word error rates compared to existing baselines, underscoring its effectiveness in multi-objective adversarial attack.

摘要: Whisper等大规模自动语音识别（ASR）模型的出现极大地拓展了其在各类实际应用中的采用。因此，确保对微小输入扰动的鲁棒性对于在实时环境中维持可靠性能至关重要。现有研究主要关注对抗攻击下的准确率下降，而关于效率方面的鲁棒性仍鲜有探索。这种局限视角仅提供了对ASR模型漏洞的部分理解。为填补这一空白，我们对多种攻击场景下的ASR鲁棒性进行了全面研究。我们提出了MORE——一种多目标重复倍增激励攻击，通过分层分阶段的排斥-锚定机制，联合降低识别准确率和推理效率。具体而言，我们将多目标对抗优化重构为分层框架，依次实现双重目标。为进一步增强攻击效果，我们提出了一种新颖的重复激励倍增目标（REDO），通过维持准确率下降并周期性地倍增预测序列长度，诱导重复文本生成。总体而言，MORE迫使ASR模型在显著更高的计算成本下产生错误转录，且仅需单个对抗输入即可触发。实验表明，与现有基线相比，MORE在保持高词错误率的同时，持续生成显著更长的转录文本，这突显了其在多目标对抗攻击中的有效性。



## **10. Instruct2Attack: Language-Guided Semantic Adversarial Attacks**

Instruct2Attack：语言引导的语义对抗攻击 cs.CV

under submission, code coming soon

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.15551v1) [paper-pdf](https://arxiv.org/pdf/2311.15551v1)

**Confidence**: 0.95

**Authors**: Jiang Liu, Chen Wei, Yuxiang Guo, Heng Yu, Alan Yuille, Soheil Feizi, Chun Pong Lau, Rama Chellappa

**Abstract**: We propose Instruct2Attack (I2A), a language-guided semantic attack that generates semantically meaningful perturbations according to free-form language instructions. We make use of state-of-the-art latent diffusion models, where we adversarially guide the reverse diffusion process to search for an adversarial latent code conditioned on the input image and text instruction. Compared to existing noise-based and semantic attacks, I2A generates more natural and diverse adversarial examples while providing better controllability and interpretability. We further automate the attack process with GPT-4 to generate diverse image-specific text instructions. We show that I2A can successfully break state-of-the-art deep neural networks even under strong adversarial defenses, and demonstrate great transferability among a variety of network architectures.

摘要: 我们提出了Instruct2Attack（I2A），一种语言引导的语义攻击方法，能够根据自由形式的语言指令生成具有语义意义的扰动。我们利用最先进的潜在扩散模型，通过对抗性地引导反向扩散过程，在输入图像和文本指令的条件下搜索对抗性潜在编码。与现有的基于噪声和语义的攻击相比，I2A生成的对抗样本更自然、更多样化，同时提供更好的可控性和可解释性。我们进一步使用GPT-4自动化攻击过程，生成多样化的图像特定文本指令。实验表明，即使在强大的对抗防御下，I2A也能成功突破最先进的深度神经网络，并在多种网络架构间展现出卓越的迁移性。



## **11. Cyberattack Detection in Virtualized Microgrids Using LightGBM and Knowledge-Distilled Classifiers**

基于LightGBM与知识蒸馏分类器的虚拟化微电网网络攻击检测 eess.SY

12 pages

**SubmitDate**: 2026-01-07    [abs](http://arxiv.org/abs/2601.03495v1) [paper-pdf](https://arxiv.org/pdf/2601.03495v1)

**Confidence**: 0.95

**Authors**: Osasumwen Cedric Ogiesoba-Eguakun, Suman Rath

**Abstract**: Modern microgrids depend on distributed sensing and communication interfaces, making them increasingly vulnerable to cyber physical disturbances that threaten operational continuity and equipment safety. In this work, a complete virtual microgrid was designed and implemented in MATLAB/Simulink, integrating heterogeneous renewable sources and secondary controller layers. A structured cyberattack framework was developed using MGLib to inject adversarial signals directly into the secondary control pathways. Multiple attack classes were emulated, including ramp, sinusoidal, additive, coordinated stealth, and denial of service behaviors. The virtual environment was used to generate labeled datasets under both normal and attack conditions. The datasets trained Light Gradient Boosting Machine (LightGBM) models to perform two functions: detecting the presence of an intrusion (binary) and distinguishing among attack types (multiclass). The multiclass model attained 99.72% accuracy and a 99.62% F1 score, while the binary model attained 94.8% accuracy and a 94.3% F1 score. A knowledge-distillation step reduced the size of the multiclass model, allowing faster predictions with only a small drop in performance. Real-time tests showed a processing delay of about 54 to 67 ms per 1000 samples, demonstrating suitability for CPU-based edge deployment in microgrid controllers. The results confirm that lightweight machine learning based intrusion detection methods can provide fast, accurate, and efficient cyberattack detection without relying on complex deep learning models. Key contributions include: (1) development of a complete MATLAB-based virtual microgrid, (2) structured attack injection at the control layer, (3) creation of multiclass labeled datasets, and (4) design of low-cost AI models suitable for practical microgrid cybersecurity.

摘要: 现代微电网依赖分布式传感与通信接口，使其日益易受网络物理扰动威胁，危及运行连续性与设备安全。本研究在MATLAB/Simulink中设计并实现了完整的虚拟微电网，集成异构可再生能源与二次控制层。利用MGLib开发结构化网络攻击框架，将对抗信号直接注入二次控制通道。模拟了多种攻击类型，包括斜坡、正弦、叠加、协同隐蔽及拒绝服务行为。该虚拟环境用于生成正常与攻击状态下的标注数据集。数据集训练轻量梯度提升机（LightGBM）模型执行两项功能：检测入侵存在（二分类）及区分攻击类型（多分类）。多分类模型获得99.72%准确率与99.62% F1分数，二分类模型获得94.8%准确率与94.3% F1分数。通过知识蒸馏步骤压缩多分类模型规模，在性能小幅下降前提下实现更快预测。实时测试显示每处理1000个样本约54-67毫秒延迟，证明其适用于微电网控制器中基于CPU的边缘部署。结果表明，基于轻量级机器学习的入侵检测方法无需依赖复杂深度学习模型即可提供快速、准确、高效的网络攻击检测。主要贡献包括：（1）开发完整的MATLAB虚拟微电网；（2）在控制层实现结构化攻击注入；（3）创建多分类标注数据集；（4）设计适用于实际微电网网络安全的低成本AI模型。



## **12. Engineering Attack Vectors and Detecting Anomalies in Additive Manufacturing**

增材制造中的工程攻击向量与异常检测 cs.CR

This paper has been accepted to EAI SmartSP 2025. This is the preprint version

**SubmitDate**: 2026-01-01    [abs](http://arxiv.org/abs/2601.00384v1) [paper-pdf](https://arxiv.org/pdf/2601.00384v1)

**Confidence**: 0.95

**Authors**: Md Mahbub Hasan, Marcus Sternhagen, Krishna Chandra Roy

**Abstract**: Additive manufacturing (AM) is rapidly integrating into critical sectors such as aerospace, automotive, and healthcare. However, this cyber-physical convergence introduces new attack surfaces, especially at the interface between computer-aided design (CAD) and machine execution layers. In this work, we investigate targeted cyberattacks on two widely used fused deposition modeling (FDM) systems, Creality's flagship model K1 Max, and Ender 3. Our threat model is a multi-layered Man-in-the-Middle (MitM) intrusion, where the adversary intercepts and manipulates G-code files during upload from the user interface to the printer firmware. The MitM intrusion chain enables several stealthy sabotage scenarios. These attacks remain undetectable by conventional slicer software or runtime interfaces, resulting in structurally defective yet externally plausible printed parts. To counter these stealthy threats, we propose an unsupervised Intrusion Detection System (IDS) that analyzes structured machine logs generated during live printing. Our defense mechanism uses a frozen Transformer-based encoder (a BERT variant) to extract semantic representations of system behavior, followed by a contrastively trained projection head that learns anomaly-sensitive embeddings. Later, a clustering-based approach and a self-attention autoencoder are used for classification. Experimental results demonstrate that our approach effectively distinguishes between benign and compromised executions.

摘要: 增材制造（AM）正快速融入航空航天、汽车和医疗等关键领域。然而，这种信息物理融合引入了新的攻击面，特别是在计算机辅助设计（CAD）与机器执行层之间的接口处。本研究针对两种广泛使用的熔融沉积成型（FDM）系统——Creality旗舰型号K1 Max和Ender 3——探究了定向网络攻击。我们的威胁模型采用多层中间人（MitM）入侵方式，攻击者在用户界面上传G代码文件至打印机固件的过程中拦截并篡改文件。这种MitM入侵链可实现多种隐蔽的破坏场景，这些攻击无法被传统切片软件或运行时接口检测，导致打印部件存在结构性缺陷但外观正常。为应对此类隐蔽威胁，我们提出了一种无监督入侵检测系统（IDS），通过分析实时打印过程中生成的结构化机器日志进行检测。该防御机制采用冻结的基于Transformer的编码器（BERT变体）提取系统行为的语义表征，随后通过对比训练投影头学习异常敏感嵌入，最后结合基于聚类的方法和自注意力自编码器进行分类。实验结果表明，我们的方法能有效区分正常与受攻击的执行过程。



## **13. PHANTOM: Physics-Aware Adversarial Attacks against Federated Learning-Coordinated EV Charging Management System**

PHANTOM：针对联邦学习协同电动汽车充电管理系统的物理感知对抗攻击 cs.ET

**SubmitDate**: 2025-12-26    [abs](http://arxiv.org/abs/2512.22381v1) [paper-pdf](https://arxiv.org/pdf/2512.22381v1)

**Confidence**: 0.95

**Authors**: Mohammad Zakaria Haider, Amit Kumar Podder, Prabin Mali, Aranya Chakrabortty, Sumit Paudyal, Mohammad Ashiqur Rahman

**Abstract**: The rapid deployment of electric vehicle charging stations (EVCS) within distribution networks necessitates intelligent and adaptive control to maintain the grid's resilience and reliability. In this work, we propose PHANTOM, a physics-aware adversarial network that is trained and optimized through a multi-agent reinforcement learning model. PHANTOM integrates a physics-informed neural network (PINN) enabled by federated learning (FL) that functions as a digital twin of EVCS-integrated systems, ensuring physically consistent modeling of operational dynamics and constraints. Building on this digital twin, we construct a multi-agent RL environment that utilizes deep Q-networks (DQN) and soft actor-critic (SAC) methods to derive adversarial false data injection (FDI) strategies capable of bypassing conventional detection mechanisms. To examine the broader grid-level consequences, a transmission and distribution (T and D) dual simulation platform is developed, allowing us to capture cascading interactions between EVCS disturbances at the distribution level and the operations of the bulk transmission system. Results demonstrate how learned attack policies disrupt load balancing and induce voltage instabilities that propagate across T and D boundaries. These findings highlight the critical need for physics-aware cybersecurity to ensure the resilience of large-scale vehicle-grid integration.

摘要: 电动汽车充电站在配电网中的快速部署需要智能自适应控制来维持电网的韧性和可靠性。本研究提出PHANTOM——一种通过多智能体强化学习模型训练优化的物理感知对抗网络。PHANTOM集成了基于联邦学习的物理信息神经网络，作为EVCS集成系统的数字孪生体，确保对运行动态和约束进行物理一致的建模。基于该数字孪生体，我们构建了采用深度Q网络和柔性演员-评论家方法的多智能体强化学习环境，以生成能够规避传统检测机制的对抗性虚假数据注入策略。为探究更广泛的电网级影响，开发了输配电网双仿真平台，用以捕捉配电层EVCS扰动与主干输电系统运行之间的级联交互。结果表明，学习得到的攻击策略会破坏负载平衡并引发跨输配电网边界的电压失稳传播。这些发现凸显了物理感知网络安全对保障大规模车网融合韧性的迫切需求。



## **14. Satellite Cybersecurity Across Orbital Altitudes: Analyzing Ground-Based Threats to LEO, MEO, and GEO**

不同轨道高度的卫星网络安全：分析对LEO、MEO和GEO的地基威胁 cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.21367v1) [paper-pdf](https://arxiv.org/pdf/2512.21367v1)

**Confidence**: 0.95

**Authors**: Mark Ballard, Guanqun Song, Ting Zhu

**Abstract**: The rapid proliferation of satellite constellations, particularly in Low Earth Orbit (LEO), has fundamentally altered the global space infrastructure, shifting the risk landscape from purely kinetic collisions to complex cyber-physical threats. While traditional safety frameworks focus on debris mitigation, ground-based adversaries increasingly exploit radio-frequency links, supply chain vulnerabilities, and software update pathways to degrade space assets. This paper presents a comparative analysis of satellite cybersecurity across LEO, Medium Earth Orbit (MEO), and Geostationary Earth Orbit (GEO) regimes. By synthesizing data from 60 publicly documented security incidents with key vulnerability proxies--including Telemetry, Tracking, and Command (TT&C) anomalies, encryption weaknesses, and environmental stressors--we characterize how orbital altitude dictates attack feasibility and impact. Our evaluation reveals distinct threat profiles: GEO systems are predominantly targeted via high-frequency uplink exposure, whereas LEO constellations face unique risks stemming from limited power budgets, hardware constraints, and susceptibility to thermal and radiation-induced faults. We further bridge the gap between security and sustainability, arguing that unmitigated cyber vulnerabilities accelerate hardware obsolescence and debris accumulation, undermining efforts toward carbon-neutral space operations. The results demonstrate that weak encryption and command path irregularities are the most consistent predictors of adversarial success across all orbits.

摘要: 卫星星座（特别是低地球轨道LEO）的快速扩张从根本上改变了全球空间基础设施，将风险格局从单纯的动能碰撞转向复杂的网络物理威胁。传统安全框架主要关注碎片减缓，而地基攻击者正日益利用射频链路、供应链漏洞和软件更新路径来破坏空间资产。本文对低地球轨道（LEO）、中地球轨道（MEO）和地球静止轨道（GEO）的卫星网络安全进行了比较分析。通过综合60起公开记录的安全事件数据及关键漏洞指标——包括遥测、跟踪与指令（TT&C）异常、加密弱点和环境压力因素——我们揭示了轨道高度如何决定攻击可行性和影响程度。评估显示明显的威胁特征差异：GEO系统主要通过高频上行链路暴露遭受攻击，而LEO星座则因有限的功率预算、硬件限制以及对热辐射诱发故障的敏感性面临独特风险。我们进一步弥合了安全与可持续性之间的鸿沟，指出未缓解的网络漏洞会加速硬件老化和碎片积累，破坏碳中和空间运营的努力。结果表明，弱加密和指令路径异常是所有轨道中最具一致性的攻击成功预测指标。



## **15. What You Read Isn't What You Hear: Linguistic Sensitivity in Deepfake Speech Detection**

所见非所闻：深度伪造语音检测中的语言敏感性 cs.LG

15 pages, 2 fogures

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17513v1) [paper-pdf](https://arxiv.org/pdf/2505.17513v1)

**Confidence**: 0.95

**Authors**: Binh Nguyen, Shuji Shi, Ryan Ofman, Thai Le

**Abstract**: Recent advances in text-to-speech technologies have enabled realistic voice generation, fueling audio-based deepfake attacks such as fraud and impersonation. While audio anti-spoofing systems are critical for detecting such threats, prior work has predominantly focused on acoustic-level perturbations, leaving the impact of linguistic variation largely unexplored. In this paper, we investigate the linguistic sensitivity of both open-source and commercial anti-spoofing detectors by introducing transcript-level adversarial attacks. Our extensive evaluation reveals that even minor linguistic perturbations can significantly degrade detection accuracy: attack success rates surpass 60% on several open-source detector-voice pairs, and notably one commercial detection accuracy drops from 100% on synthetic audio to just 32%. Through a comprehensive feature attribution analysis, we identify that both linguistic complexity and model-level audio embedding similarity contribute strongly to detector vulnerability. We further demonstrate the real-world risk via a case study replicating the Brad Pitt audio deepfake scam, using transcript adversarial attacks to completely bypass commercial detectors. These results highlight the need to move beyond purely acoustic defenses and account for linguistic variation in the design of robust anti-spoofing systems. All source code will be publicly available.

摘要: 近期文本转语音技术的进步实现了逼真的语音生成，助长了欺诈和冒充等基于音频的深度伪造攻击。虽然音频反欺骗系统对于检测此类威胁至关重要，但先前的研究主要集中于声学层面的扰动，很大程度上忽略了语言变异的影响。本文通过引入文本层面的对抗性攻击，研究了开源和商业反欺骗检测器的语言敏感性。我们的广泛评估表明，即使微小的语言扰动也能显著降低检测准确率：在多个开源检测器-语音配对中，攻击成功率超过60%，尤其值得注意的是，一款商业检测器的准确率从合成音频的100%骤降至仅32%。通过全面的特征归因分析，我们发现语言复杂度和模型级音频嵌入相似性均对检测器脆弱性有重要贡献。我们进一步通过复制布拉德·皮特音频深度伪造骗局的案例研究，展示了现实世界风险，利用文本对抗攻击完全绕过了商业检测器。这些结果凸显了超越纯声学防御、在鲁棒反欺骗系统设计中考虑语言变异的必要性。所有源代码将公开提供。



## **16. Trading Devil: Robust backdoor attack via Stochastic investment models and Bayesian approach**

交易恶魔：基于随机投资模型与贝叶斯方法的鲁棒后门攻击 cs.CR

(Last update!, a constructive comment from arxiv led to this latest update ) Stochastic investment models and a Bayesian approach to better modeling of uncertainty : adversarial machine learning or Stochastic market. arXiv admin note: substantial text overlap with arXiv:2402.05967 (see this link to the paper by : Orson Mengara)

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2406.10719v5) [paper-pdf](https://arxiv.org/pdf/2406.10719v5)

**Confidence**: 0.95

**Authors**: Orson Mengara

**Abstract**: With the growing use of voice-activated systems and speech recognition technologies, the danger of backdoor attacks on audio data has grown significantly. This research looks at a specific type of attack, known as a Stochastic investment-based backdoor attack (MarketBack), in which adversaries strategically manipulate the stylistic properties of audio to fool speech recognition systems. The security and integrity of machine learning models are seriously threatened by backdoor attacks, in order to maintain the reliability of audio applications and systems, the identification of such attacks becomes crucial in the context of audio data. Experimental results demonstrated that MarketBack is feasible to achieve an average attack success rate close to 100% in seven victim models when poisoning less than 1% of the training data.

摘要: 随着语音激活系统和语音识别技术的广泛应用，针对音频数据的后门攻击风险显著增加。本研究探讨了一种基于随机投资的后门攻击方法（MarketBack），攻击者通过策略性地操纵音频的风格特征来欺骗语音识别系统。后门攻击严重威胁机器学习模型的安全性与完整性，为维护音频应用系统的可靠性，在音频数据场景中识别此类攻击至关重要。实验结果表明，当训练数据污染率低于1%时，MarketBack在七个受害模型上平均攻击成功率接近100%。



## **17. SoK: Challenges in Tabular Membership Inference Attacks**

SoK：表格数据成员推断攻击面临的挑战 cs.LG

This paper is currently under review for the EuroS&P conference

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2601.15874v1) [paper-pdf](https://arxiv.org/pdf/2601.15874v1)

**Confidence**: 0.95

**Authors**: Cristina Pêra, Tânia Carvalho, Maxime Cordy, Luís Antunes

**Abstract**: Membership Inference Attacks (MIAs) are currently a dominant approach for evaluating privacy in machine learning applications. Despite their significance in identifying records belonging to the training dataset, several concerns remain unexplored, particularly with regard to tabular data. In this paper, first, we provide an extensive review and analysis of MIAs considering two main learning paradigms: centralized and federated learning. We extend and refine the taxonomy for both. Second, we demonstrate the efficacy of MIAs in tabular data using several attack strategies, also including defenses. Furthermore, in a federated learning scenario, we consider the threat posed by an outsider adversary, which is often neglected. Third, we demonstrate the high vulnerability of single-outs (records with a unique signature) to MIAs. Lastly, we explore how MIAs transfer across model architectures. Our results point towards a general poor performance of these attacks in tabular data which contrasts with previous state-of-the-art. Notably, even attacks with limited attack performance can still successfully expose a large portion of single-outs. Moreover, our findings suggest that using different surrogate models makes MIAs more effective.

摘要: 成员推断攻击（MIAs）目前是评估机器学习应用隐私性的主流方法。尽管其在识别训练数据集记录方面具有重要意义，但仍有若干问题尚未得到充分探讨，特别是在表格数据领域。本文首先对MIAs进行了全面回顾与分析，涵盖两种主要学习范式：集中式学习和联邦学习，并对两者的分类体系进行了扩展与完善。其次，我们通过多种攻击策略（包括防御措施）展示了MIAs在表格数据中的有效性。此外，在联邦学习场景中，我们考虑了常被忽视的外部对手威胁。第三，我们证明了具有唯一特征的单一记录对MIAs具有高度脆弱性。最后，我们探讨了MIAs在不同模型架构间的迁移性。我们的研究结果表明，这些攻击在表格数据中普遍表现不佳，这与先前的最先进成果形成对比。值得注意的是，即使攻击性能有限，仍能成功暴露大量单一记录。此外，我们的发现表明，使用不同的代理模型可提升MIAs的有效性。



## **18. Deep Leakage with Generative Flow Matching Denoiser**

基于生成流匹配去噪器的深度泄露攻击 cs.CV

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.15049v1) [paper-pdf](https://arxiv.org/pdf/2601.15049v1)

**Confidence**: 0.95

**Authors**: Isaac Baglin, Xiatian Zhu, Simon Hadfield

**Abstract**: Federated Learning (FL) has emerged as a powerful paradigm for decentralized model training, yet it remains vulnerable to deep leakage (DL) attacks that reconstruct private client data from shared model updates. While prior DL methods have demonstrated varying levels of success, they often suffer from instability, limited fidelity, or poor robustness under realistic FL settings. We introduce a new DL attack that integrates a generative Flow Matching (FM) prior into the reconstruction process. By guiding optimization toward the distribution of realistic images (represented by a flow matching foundation model), our method enhances reconstruction fidelity without requiring knowledge of the private data. Extensive experiments on multiple datasets and target models demonstrate that our approach consistently outperforms state-of-the-art attacks across pixel-level, perceptual, and feature-based similarity metrics. Crucially, the method remains effective across different training epochs, larger client batch sizes, and under common defenses such as noise injection, clipping, and sparsification. Our findings call for the development of new defense strategies that explicitly account for adversaries equipped with powerful generative priors.

摘要: 联邦学习（FL）已成为去中心化模型训练的强大范式，但其仍易受深度泄露（DL）攻击，即从共享的模型更新中重构私有客户端数据。虽然先前的DL方法已展现出不同程度的成功，但在实际FL设置下，它们常面临不稳定性、保真度有限或鲁棒性差的问题。我们提出了一种新的DL攻击方法，将生成流匹配（FM）先验集成到重构过程中。通过引导优化朝向真实图像的分布（由流匹配基础模型表示），我们的方法在无需了解私有数据的情况下提升了重构保真度。在多个数据集和目标模型上的广泛实验表明，我们的方法在像素级、感知和基于特征的相似度指标上均持续优于现有最先进的攻击方法。关键的是，该方法在不同训练轮次、较大客户端批次大小以及常见防御措施（如噪声注入、裁剪和稀疏化）下仍保持有效。我们的研究结果呼吁开发新的防御策略，以明确应对配备强大生成先验的对手。



## **19. Beyond Denial-of-Service: The Puppeteer's Attack for Fine-Grained Control in Ranking-Based Federated Learning**

超越拒绝服务攻击：基于排名的联邦学习中实现细粒度控制的傀儡师攻击 cs.LG

12 pages. To appear in The Web Conference 2026

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.14687v1) [paper-pdf](https://arxiv.org/pdf/2601.14687v1)

**Confidence**: 0.95

**Authors**: Zhihao Chen, Zirui Gong, Jianting Ning, Yanjun Zhang, Leo Yu Zhang

**Abstract**: Federated Rank Learning (FRL) is a promising Federated Learning (FL) paradigm designed to be resilient against model poisoning attacks due to its discrete, ranking-based update mechanism. Unlike traditional FL methods that rely on model updates, FRL leverages discrete rankings as a communication parameter between clients and the server. This approach significantly reduces communication costs and limits an adversary's ability to scale or optimize malicious updates in the continuous space, thereby enhancing its robustness. This makes FRL particularly appealing for applications where system security and data privacy are crucial, such as web-based auction and bidding platforms. While FRL substantially reduces the attack surface, we demonstrate that it remains vulnerable to a new class of local model poisoning attack, i.e., fine-grained control attacks. We introduce the Edge Control Attack (ECA), the first fine-grained control attack tailored to ranking-based FL frameworks. Unlike conventional denial-of-service (DoS) attacks that cause conspicuous disruptions, ECA enables an adversary to precisely degrade a competitor's accuracy to any target level while maintaining a normal-looking convergence trajectory, thereby avoiding detection. ECA operates in two stages: (i) identifying and manipulating Ascending and Descending Edges to align the global model with the target model, and (ii) widening the selection boundary gap to stabilize the global model at the target accuracy. Extensive experiments across seven benchmark datasets and nine Byzantine-robust aggregation rules (AGRs) show that ECA achieves fine-grained accuracy control with an average error of only 0.224%, outperforming the baseline by up to 17x. Our findings highlight the need for stronger defenses against advanced poisoning attacks. Our code is available at: https://github.com/Chenzh0205/ECA

摘要: 联邦排名学习（FRL）是一种有前景的联邦学习（FL）范式，因其基于离散排名的更新机制而被设计为能够抵御模型投毒攻击。与传统FL方法依赖模型更新不同，FRL利用离散排名作为客户端与服务器之间的通信参数。这种方法显著降低了通信成本，并限制了攻击者在连续空间中扩展或优化恶意更新的能力，从而增强了其鲁棒性。这使得FRL在系统安全性和数据隐私至关重要的应用中特别具有吸引力，例如基于网络的拍卖和竞价平台。尽管FRL大幅减少了攻击面，但我们证明它仍然容易受到一类新的本地模型投毒攻击，即细粒度控制攻击。我们引入了边缘控制攻击（ECA），这是首个针对基于排名的FL框架量身定制的细粒度控制攻击。与导致明显破坏的传统拒绝服务（DoS）攻击不同，ECA使攻击者能够将竞争对手的准确率精确降低到任何目标水平，同时保持看似正常的收敛轨迹，从而避免检测。ECA分两个阶段操作：（i）识别并操纵上升和下降边缘，使全局模型与目标模型对齐；（ii）扩大选择边界间隙，使全局模型稳定在目标准确率。在七个基准数据集和九种拜占庭鲁棒聚合规则（AGR）上的大量实验表明，ECA实现了细粒度准确率控制，平均误差仅为0.224%，比基线方法高出多达17倍。我们的研究结果强调了对高级投毒攻击需要更强的防御措施。我们的代码可在以下网址获取：https://github.com/Chenzh0205/ECA



## **20. Gradient Structure Estimation under Label-Only Oracles via Spectral Sensitivity**

基于谱敏感度的仅标签预言机下的梯度结构估计 cs.LG

**SubmitDate**: 2026-01-17    [abs](http://arxiv.org/abs/2601.14300v1) [paper-pdf](https://arxiv.org/pdf/2601.14300v1)

**Confidence**: 0.95

**Authors**: Jun Liu, Leo Yu Zhang, Fengpeng Li, Isao Echizen, Jiantao Zhou

**Abstract**: Hard-label black-box settings, where only top-1 predicted labels are observable, pose a fundamentally constrained yet practically important feedback model for understanding model behavior. A central challenge in this regime is whether meaningful gradient information can be recovered from such discrete responses. In this work, we develop a unified theoretical perspective showing that a wide range of existing sign-flipping hard-label attacks can be interpreted as implicitly approximating the sign of the true loss gradient. This observation reframes hard-label attacks from heuristic search procedures into instances of gradient sign recovery under extremely limited feedback. Motivated by this first-principles understanding, we propose a new attack framework that combines a zero-query frequency-domain initialization with a Pattern-Driven Optimization (PDO) strategy. We establish theoretical guarantees demonstrating that, under mild assumptions, our initialization achieves higher expected cosine similarity to the true gradient sign compared to random baselines, while the proposed PDO procedure attains substantially lower query complexity than existing structured search approaches. We empirically validate our framework through extensive experiments on CIFAR-10, ImageNet, and ObjectNet, covering standard and adversarially trained models, commercial APIs, and CLIP-based models. The results show that our method consistently surpasses SOTA hard-label attacks in both attack success rate and query efficiency, particularly in low-query regimes. Beyond image classification, our approach generalizes effectively to corrupted data, biomedical datasets, and dense prediction tasks. Notably, it also successfully circumvents Blacklight, a SOTA stateful defense, resulting in a $0\%$ detection rate. Our code will be released publicly soon at https://github.com/csjunjun/DPAttack.git.

摘要: 硬标签黑盒设置（仅可观测top-1预测标签）为理解模型行为提供了一个本质上受限但实际重要的反馈模型。该场景的核心挑战在于能否从这种离散响应中恢复有意义的梯度信息。本文建立了一个统一的理论视角，表明现有多种符号翻转硬标签攻击可被解释为对真实损失梯度符号的隐式近似。这一观察将硬标签攻击从启发式搜索过程重新定义为极端有限反馈下的梯度符号恢复实例。基于这一基本原理的理解，我们提出了一种新的攻击框架，结合了零查询频域初始化和模式驱动优化（PDO）策略。我们在理论证明中表明，在温和假设下，我们的初始化相比随机基线能获得与真实梯度符号更高的期望余弦相似度，而所提出的PDO过程则能以远低于现有结构化搜索方法的查询复杂度实现目标。我们在CIFAR-10、ImageNet和ObjectNet数据集上进行了广泛实验，覆盖标准模型、对抗训练模型、商业API和CLIP模型，实证验证了该框架。结果表明，我们的方法在攻击成功率和查询效率上均持续超越SOTA硬标签攻击，尤其在低查询场景下表现突出。除图像分类外，该方法还能有效泛化至损坏数据、生物医学数据集和密集预测任务。值得注意的是，该方法成功规避了SOTA有状态防御系统Blacklight，实现了0%的检测率。代码即将发布于https://github.com/csjunjun/DPAttack.git。



## **21. Adversarial Attacks on Medical Hyperspectral Imaging Exploiting Spectral-Spatial Dependencies and Multiscale Features**

针对医学高光谱成像的对抗攻击：利用光谱-空间依赖性与多尺度特征 cs.CV

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2601.07056v1) [paper-pdf](https://arxiv.org/pdf/2601.07056v1)

**Confidence**: 0.95

**Authors**: Yunrui Gu, Zhenzhe Gao, Cong Kong, Zhaoxia Yin

**Abstract**: Medical hyperspectral imaging (HSI) enables accurate disease diagnosis by capturing rich spectral-spatial tissue information, but recent advances in deep learning have exposed its vulnerability to adversarial attacks. In this work, we identify two fundamental causes of this fragility: the reliance on local pixel dependencies for preserving tissue structure and the dependence on multiscale spectral-spatial representations for hierarchical feature encoding. Building on these insights, we propose a targeted adversarial attack framework for medical HSI, consisting of a Local Pixel Dependency Attack that exploits spatial correlations among neighboring pixels, and a Multiscale Information Attack that perturbs features across hierarchical spectral-spatial scales. Experiments on the Brain and MDC datasets demonstrate that our attacks significantly degrade classification performance, especially in tumor regions, while remaining visually imperceptible. Compared with existing methods, our approach reveals the unique vulnerabilities of medical HSI models and underscores the need for robust, structure-aware defenses in clinical applications.

摘要: 医学高光谱成像（HSI）通过捕获丰富的光谱-空间组织信息实现精准疾病诊断，但深度学习的最新进展暴露了其易受对抗攻击的脆弱性。本研究识别了这种脆弱性的两个根本原因：依赖局部像素相关性以保持组织结构，以及依赖多尺度光谱-空间表示进行层次特征编码。基于这些发现，我们提出了一种针对医学HSI的定向对抗攻击框架，包含利用相邻像素空间相关性的局部像素依赖攻击，以及扰动跨层次光谱-空间尺度特征的多尺度信息攻击。在Brain和MDC数据集上的实验表明，我们的攻击显著降低了分类性能（尤其在肿瘤区域），同时保持视觉不可感知性。与现有方法相比，我们的方法揭示了医学HSI模型的独特脆弱性，并强调了临床应用中需要构建具有结构感知能力的鲁棒防御机制。



## **22. Sparse Neural Approximations for Bilevel Adversarial Problems in Power Grids**

稀疏神经逼近方法在电网双层对抗问题中的应用 eess.SY

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.06187v1) [paper-pdf](https://arxiv.org/pdf/2512.06187v1)

**Confidence**: 0.95

**Authors**: Young-ho Cho, Harsha Nagarajan, Deepjyoti Deka, Hao Zhu

**Abstract**: The adversarial worst-case load shedding (AWLS) problem is pivotal for identifying critical contingencies under line outages. It is naturally cast as a bilevel program: the upper level simulates an attacker determining worst-case line failures, and the lower level corresponds to the defender's generator redispatch operations. Conventional techniques using optimality conditions render the bilevel, mixed-integer formulation computationally prohibitive due to the combinatorial number of topologies and the nonconvexity of AC power flow constraints. To address these challenges, we develop a novel single-level optimal value-function (OVF) reformulation and further leverage a data-driven neural network (NN) surrogate of the follower's optimal value. To ensure physical realizability, we embed the trained surrogate in a physics-constrained NN (PCNN) formulation that couples the OVF inequality with (relaxed) AC feasibility, yielding a mixed-integer convex model amenable to off-the-shelf solvers. To achieve scalability, we learn a sparse, area-partitioned NN via spectral clustering; the resulting block-sparse architecture scales essentially linearly with system size while preserving accuracy. Notably, our approach produces near-optimal worst-case failures and generalizes across loading conditions and unseen topologies, enabling rapid online recomputation. Numerical experiments on the IEEE 14- and 118-bus systems demonstrate the method's scalability and solution quality for large-scale contingency analysis, with an average optimality gap of 5.8% compared to conventional methods, while maintaining computation times under one minute.

摘要: 对抗性最坏情况减负荷（AWLS）问题对于识别线路停运下的关键事故至关重要。该问题自然表述为双层规划：上层模拟攻击者确定最坏情况线路故障，下层对应防御者的发电机再调度操作。传统方法利用最优性条件处理该双层混合整数规划时，由于拓扑组合数量巨大及交流潮流约束的非凸性，计算量难以承受。为解决这些挑战，我们开发了一种新颖的单层最优值函数（OVF）重构方法，并进一步利用跟随者最优值的数据驱动神经网络（NN）代理。为确保物理可实现性，我们将训练后的代理嵌入物理约束神经网络（PCNN）框架，将OVF不等式与（松弛的）交流可行性耦合，得到可直接用现成求解器处理的混合整数凸模型。为实现可扩展性，我们通过谱聚类学习稀疏的区域划分神经网络；所得块稀疏架构在保持精度的同时，其规模随系统大小基本呈线性增长。值得注意的是，我们的方法能产生近似最优的最坏情况故障，并在不同负荷条件和未见拓扑中具有良好泛化能力，支持快速在线重计算。在IEEE 14节点和118节点系统上的数值实验表明，该方法在大规模事故分析中具有优越的可扩展性和求解质量——与传统方法相比平均最优性差距为5.8%，同时计算时间控制在一分钟以内。



## **23. Towards Trustworthy Wi-Fi Sensing: Systematic Evaluation of Deep Learning Model Robustness to Adversarial Attacks**

迈向可信的Wi-Fi感知：深度学习模型对抗攻击鲁棒性的系统评估 cs.LG

19 pages, 8 figures, 7 tables

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.20456v1) [paper-pdf](https://arxiv.org/pdf/2511.20456v1)

**Confidence**: 0.95

**Authors**: Shreevanth Krishnaa Gopalakrishnan, Stephen Hailes

**Abstract**: Machine learning has become integral to Channel State Information (CSI)-based human sensing systems and is expected to power applications such as device-free activity recognition and identity detection in future cellular and Wi-Fi generations. However, these systems rely on models whose decisions can be subtly perturbed, raising concerns for security and reliability in ubiquitous sensing. Quantifying and understanding the robustness of such models, defined as their ability to maintain accurate predictions under adversarial perturbations, is therefore critical before wireless sensing can be safely deployed in real-world environments.   This work presents a systematic evaluation of the robustness of CSI deep learning models under diverse threat models (white-box, black-box/transfer, and universal perturbations) and varying degrees of attack realism. We establish a framework to compare compact temporal autoencoder models with larger deep architectures across three public datasets, quantifying how model scale, training regime, and physical constraints influence robustness. Our experiments show that smaller models, while efficient and equally performant on clean data, are markedly less robust. We further confirm that physically realizable signal-space perturbations, designed to be feasible in real wireless channels, significantly reduce attack success compared to unconstrained feature-space attacks. Adversarial training mitigates these vulnerabilities, improving mean robust accuracy with only moderate degradation in clean performance across both model classes. As wireless sensing advances towards reliable, cross-domain operation, these findings provide quantitative baselines for robustness estimation and inform design principles for secure and trustworthy human-centered sensing systems.

摘要: 机器学习已成为基于信道状态信息（CSI）的人类感知系统的核心组成部分，并有望在未来蜂窝和Wi-Fi世代中支持设备无关的活动识别和身份检测等应用。然而，这些系统依赖的模型决策可能受到细微干扰，引发了泛在感知中安全性和可靠性的担忧。因此，在无线感知安全部署于现实环境之前，量化并理解此类模型的鲁棒性——即其在对抗扰动下保持准确预测的能力——至关重要。本研究系统评估了CSI深度学习模型在不同威胁模型（白盒、黑盒/迁移和通用扰动）及不同攻击现实性程度下的鲁棒性。我们建立了一个框架，在三个公共数据集上比较紧凑型时序自编码器模型与大型深度架构，量化模型规模、训练机制和物理约束如何影响鲁棒性。实验表明，较小模型虽然在干净数据上效率高且性能相当，但鲁棒性显著较低。我们进一步证实，与无约束特征空间攻击相比，物理可实现的信号空间扰动（设计为在实际无线信道中可行）能显著降低攻击成功率。对抗训练可缓解这些漏洞，在两类模型中仅以清洁性能适度下降为代价，提高了平均鲁棒精度。随着无线感知向可靠、跨域操作发展，这些发现为鲁棒性评估提供了量化基准，并为设计安全可信的人本感知系统提供了原则指导。



## **24. Algorithmic detection of false data injection attacks in cyber-physical systems**

网络物理系统中虚假数据注入攻击的算法检测 math.OC

13 pages, 6 figures

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2511.18588v1) [paper-pdf](https://arxiv.org/pdf/2511.18588v1)

**Confidence**: 0.95

**Authors**: Souvik Das, Avishek Ghosh, Debasish Chatterjee

**Abstract**: This article introduces an anomaly detection based algorithm (AD-CPS) to detect false data injection attacks that fall under the category of data deception/integrity attacks, but with arbitrary information structure, in cyber-physical systems (CPSs) modeled as stochastic linear time-invariant systems. The core idea of this data-driven algorithm is based on the fact that an honest state (one not compromised by adversaries) generated by the CPS should concentrate near its weighted empirical mean of the immediate past samples. As the first theoretical result, we provide non-asymptotic guarantees on the false positive error incurred by the algorithm for attacks that are 2-step honest, referring to adversaries that act intermittently rather than successively. Moreover, we establish that for adversaries possessing a certain minimum energy, the false negative error incurred by AD-CPS is low. Extensive experiments were conducted on partially observed stochastic LTI systems to demonstrate these properties and to quantitatively compare AD-CPS with an optimal CUSUM-based test.

摘要: 本文介绍了一种基于异常检测的算法（AD-CPS），用于检测网络物理系统（CPS）中属于数据欺骗/完整性攻击类别但具有任意信息结构的虚假数据注入攻击，该系统建模为随机线性时不变系统。该数据驱动算法的核心思想基于一个事实：由CPS生成的诚实状态（未受对手破坏）应集中在其近期样本的加权经验均值附近。作为首个理论结果，我们为算法在2步诚实攻击（指对手间歇性而非连续行动）下产生的误报错误提供了非渐近保证。此外，我们证明对于具有特定最小能量的对手，AD-CPS产生的漏报错误较低。我们在部分观测的随机LTI系统上进行了大量实验，以验证这些特性，并将AD-CPS与基于最优CUSUM的测试进行定量比较。



## **25. A Novel and Practical Universal Adversarial Perturbations against Deep Reinforcement Learning based Intrusion Detection Systems**

一种针对基于深度强化学习的入侵检测系统的新型实用通用对抗扰动 cs.CR

13 pages, 7 Figures,

**SubmitDate**: 2025-11-22    [abs](http://arxiv.org/abs/2511.18223v1) [paper-pdf](https://arxiv.org/pdf/2511.18223v1)

**Confidence**: 0.95

**Authors**: H. Zhang, L. Zhang, G. Epiphaniou, C. Maple

**Abstract**: Intrusion Detection Systems (IDS) play a vital role in defending modern cyber physical systems against increasingly sophisticated cyber threats. Deep Reinforcement Learning-based IDS, have shown promise due to their adaptive and generalization capabilities. However, recent studies reveal their vulnerability to adversarial attacks, including Universal Adversarial Perturbations (UAPs), which can deceive models with a single, input-agnostic perturbation. In this work, we propose a novel UAP attack against Deep Reinforcement Learning (DRL)-based IDS under the domain-specific constraints derived from network data rules and feature relationships. To the best of our knowledge, there is no existing study that has explored UAP generation for the DRL-based IDS. In addition, this is the first work that focuses on developing a UAP against a DRL-based IDS under realistic domain constraints based on not only the basic domain rules but also mathematical relations between the features. Furthermore, we enhance the evasion performance of the proposed UAP, by introducing a customized loss function based on the Pearson Correlation Coefficient, and we denote it as Customized UAP. To the best of our knowledge, this is also the first work using the PCC value in the UAP generation, even in the broader context. Four additional established UAP baselines are implemented for a comprehensive comparison. Experimental results demonstrate that our proposed Customized UAP outperforms two input-dependent attacks including Fast Gradient Sign Method (FGSM), Basic Iterative Method (BIM), and four UAP baselines, highlighting its effectiveness for real-world adversarial scenarios.

摘要: 入侵检测系统（IDS）在防御现代网络物理系统抵御日益复杂的网络威胁方面发挥着至关重要的作用。基于深度强化学习的IDS因其自适应和泛化能力而展现出潜力。然而，最近的研究揭示了它们对对抗攻击的脆弱性，包括通用对抗扰动（UAPs），这种扰动可以通过单一、与输入无关的干扰欺骗模型。在本研究中，我们提出了一种针对基于深度强化学习（DRL）的IDS的新型UAP攻击，该攻击遵循从网络数据规则和特征关系中推导出的领域特定约束。据我们所知，目前尚无研究探索针对基于DRL的IDS的UAP生成。此外，这是首个基于基本领域规则及特征间数学关系，在现实领域约束下开发针对基于DRL的IDS的UAP的研究。进一步地，我们通过引入基于皮尔逊相关系数的定制化损失函数来增强所提出UAP的规避性能，并将其称为定制化UAP。据我们所知，这也是首个在UAP生成中使用PCC值的研究，即使在更广泛的背景下也是如此。为进行全面比较，我们实现了四种已建立的UAP基线方法。实验结果表明，我们提出的定制化UAP在性能上优于两种输入相关攻击（包括快速梯度符号方法（FGSM）和基本迭代方法（BIM））以及四种UAP基线方法，突显了其在现实对抗场景中的有效性。



## **26. BadPatches: Routing-aware Backdoor Attacks on Vision Mixture of Experts**

BadPatches：针对视觉专家混合模型的路由感知型后门攻击 cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2505.01811v3) [paper-pdf](https://arxiv.org/pdf/2505.01811v3)

**Confidence**: 0.95

**Authors**: Cedric Chan, Jona te Lintelo, Stjepan Picek

**Abstract**: Mixture of Experts (MoE) architectures have gained popularity for reducing computational costs in deep neural networks by activating only a subset of parameters during inference. While this efficiency makes MoE attractive for vision tasks, the patch-based processing in vision models introduces new methods for adversaries to perform backdoor attacks. In this work, we investigate the vulnerability of vision MoE models for image classification, specifically the patch-based MoE (pMoE) models and MoE-based vision transformers, against backdoor attacks. We propose a novel routing-aware trigger application method BadPatches, which is designed for patch-based processing in vision MoE models. BadPatches applies triggers on image patches rather than on the entire image. We show that BadPatches achieves high attack success rates (ASRs) with lower poisoning rates than routing-agnostic triggers and is successful at poisoning rates as low as 0.01% with an ASR above 80% on pMoE. Moreover, BadPatches is still effective when an adversary does not have complete knowledge of the patch routing configuration of the considered models. Next, we explore how trigger design affects pMoE patch routing. Finally, we investigate fine-pruning as a defense. Results show that only the fine-tuning stage of fine-pruning removes the backdoor from the model.

摘要: 专家混合（MoE）架构通过推理时仅激活部分参数来降低深度神经网络的计算成本，因而广受欢迎。虽然这种高效性使MoE在视觉任务中颇具吸引力，但视觉模型中的基于图像块的处理方式为攻击者实施后门攻击提供了新途径。本研究探讨了视觉MoE模型（特别是基于图像块的MoE模型和基于MoE的视觉Transformer）在图像分类任务中面对后门攻击的脆弱性。我们提出了一种新颖的路由感知型触发器应用方法BadPatches，专为视觉MoE模型中的基于图像块的处理而设计。BadPatches将触发器应用于图像块而非整张图像。实验表明，与路由无关的触发器相比，BadPatches能以更低的投毒率实现高攻击成功率（ASR），在pMoE模型上仅需0.01%的投毒率即可达到80%以上的ASR。此外，即使攻击者不完全了解目标模型的图像块路由配置，BadPatches依然有效。接下来，我们探究了触发器设计如何影响pMoE的图像块路由。最后，我们研究了精细剪枝作为防御手段的效果。结果显示，仅精细剪枝中的微调阶段能够消除模型中的后门。



## **27. Boosting Adversarial Transferability with Low-Cost Optimization via Maximin Expected Flatness**

通过最大化最小期望平坦度的低成本优化提升对抗样本可迁移性 cs.CV

Accepted by IEEE T-IFS

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2405.16181v3) [paper-pdf](https://arxiv.org/pdf/2405.16181v3)

**Confidence**: 0.95

**Authors**: Chunlin Qiu, Ang Li, Yiheng Duan, Shenyi Zhang, Yuanjie Zhang, Lingchen Zhao, Qian Wang

**Abstract**: Transfer-based attacks craft adversarial examples on white-box surrogate models and directly deploy them against black-box target models, offering model-agnostic and query-free threat scenarios. While flatness-enhanced methods have recently emerged to improve transferability by enhancing the loss surface flatness of adversarial examples, their divergent flatness definitions and heuristic attack designs suffer from unexamined optimization limitations and missing theoretical foundation, thus constraining their effectiveness and efficiency. This work exposes the severely imbalanced exploitation-exploration dynamics in flatness optimization, establishing the first theoretical foundation for flatness-based transferability and proposing a principled framework to overcome these optimization pitfalls. Specifically, we systematically unify fragmented flatness definitions across existing methods, revealing their imbalanced optimization limitations in over-exploration of sensitivity peaks or over-exploitation of local plateaus. To resolve these issues, we rigorously formalize average-case flatness and transferability gaps, proving that enhancing zeroth-order average-case flatness minimizes cross-model discrepancies. Building on this theory, we design a Maximin Expected Flatness (MEF) attack that enhances zeroth-order average-case flatness while balancing flatness exploration and exploitation. Extensive evaluations across 22 models and 24 current transfer-based attacks demonstrate MEF's superiority: it surpasses the state-of-the-art PGN attack by 4% in attack success rate at half the computational cost and achieves 8% higher success rate under the same budget. When combined with input augmentation, MEF attains 15% additional gains against defense-equipped models, establishing new robustness benchmarks. Our code is available at https://github.com/SignedQiu/MEFAttack.

摘要: 基于迁移的攻击方法在白盒代理模型上生成对抗样本，并直接部署于黑盒目标模型，提供了模型无关且无需查询的威胁场景。虽然近期出现的平坦度增强方法通过提升对抗样本损失表面的平坦度来改善可迁移性，但其不同的平坦度定义和启发式攻击设计存在未经验证的优化局限性和理论基础的缺失，从而限制了其有效性和效率。本研究揭示了平坦度优化中严重失衡的利用-探索动态，首次建立了基于平坦度的可迁移性理论基础，并提出了一个原则性框架来克服这些优化缺陷。具体而言，我们系统性地统一了现有方法中零散的平坦度定义，揭示了它们在过度探索敏感峰值或过度利用局部平台方面的优化局限性。为解决这些问题，我们严格形式化了平均情况平坦度和可迁移性差距，证明了增强零阶平均情况平坦度可以最小化跨模型差异。基于该理论，我们设计了最大化最小期望平坦度攻击方法，在平衡平坦度探索与利用的同时增强零阶平均情况平坦度。在22个模型和24种当前迁移攻击方法上的广泛评估证明了MEF的优越性：其攻击成功率比最先进的PGN攻击高出4%，而计算成本仅为其一半；在相同预算下获得高出8%的成功率。当结合输入增强时，MEF在具备防御机制的模型上额外获得15%的性能提升，建立了新的鲁棒性基准。代码发布于https://github.com/SignedQiu/MEFAttack。



## **28. A Measurement of Genuine Tor Traces for Realistic Website Fingerprinting**

真实Tor流量测量：面向现实网站指纹识别的数据集 cs.CR

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2404.07892v2) [paper-pdf](https://arxiv.org/pdf/2404.07892v2)

**Confidence**: 0.95

**Authors**: Rob Jansen, Ryan Wails, Aaron Johnson

**Abstract**: Website fingerprinting (WF) is a dangerous attack on web privacy because it enables an adversary to predict the website a user is visiting, despite the use of encryption, VPNs, or anonymizing networks such as Tor. Previous WF work almost exclusively uses synthetic datasets to evaluate the performance and estimate the feasibility of WF attacks despite evidence that synthetic data misrepresents the real world. In this paper we present GTT23, the first WF dataset of genuine Tor traces, which we obtain through a large-scale measurement of the Tor network and which is intended especially for WF. It represents real Tor user behavior better than any existing WF dataset, is larger than any existing WF dataset by at least an order of magnitude, and will help ground the future study of realistic WF attacks and defenses. In a detailed evaluation, we survey 28 WF datasets published since 2008 and compare their characteristics to those of GTT23. We discover common deficiencies of synthetic datasets that make them inferior to GTT23 for drawing meaningful conclusions about the effectiveness of WF attacks directed at real Tor users. We have made GTT23 available to promote reproducible research and to help inspire new directions for future work.

摘要: 网站指纹识别（WF）是对网络隐私的严重威胁，它使攻击者能够预测用户访问的网站，即使用户使用了加密、VPN或Tor等匿名网络。尽管有证据表明合成数据无法准确反映现实情况，但以往的WF研究几乎完全依赖合成数据集来评估攻击性能与可行性。本文提出GTT23——首个专门为WF设计的真实Tor流量数据集，通过大规模Tor网络测量获得。该数据集比现有任何WF数据集更能体现真实Tor用户行为，规模至少大一个数量级，将为未来研究现实WF攻击与防御提供坚实基础。通过详细评估，我们梳理了2008年以来发布的28个WF数据集，并将其特征与GTT23对比，发现合成数据集普遍存在缺陷，导致其难以针对真实Tor用户得出有效的WF攻击效能结论。我们已公开GTT23数据集以促进可重复研究，并为未来工作开拓新方向。



## **29. PINNsFailureRegion Localization and Refinement through White-box AdversarialAttack**

基于白盒对抗攻击的PINNs失效区域定位与优化方法 cs.LG

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2310.11789v2) [paper-pdf](https://arxiv.org/pdf/2310.11789v2)

**Confidence**: 0.95

**Authors**: Shengzhu Shi, Yao Li, Zhichang Guo, Boying Wu, Yang Zhao

**Abstract**: Physics-informed neural networks (PINNs) have shown great promise in solving partial differential equations (PDEs). However, vanilla PINNs often face challenges when solving complex PDEs, especially those involving multi-scale behaviors or solutions with sharp or oscillatory characteristics. To precisely and adaptively locate the critical regions that fail in the solving process we propose a sampling strategy grounded in white-box adversarial attacks, referred to as WbAR. WbAR search for failure regions in the direction of the loss gradient, thus directly locating the most critical positions. WbAR generates adversarial samples in a random walk manner and iteratively refines PINNs to guide the model's focus towards dynamically updated critical regions during training. We implement WbAR to the elliptic equation with multi-scale coefficients, Poisson equation with multi-peak solutions, high-dimensional Poisson equations, and Burgers equation with sharp solutions. The results demonstrate that WbAR can effectively locate and reduce failure regions. Moreover, WbAR is suitable for solving complex PDEs, since locating failure regions through adversarial attacks is independent of the size of failure regions or the complexity of the distribution.

摘要: 物理信息神经网络（PINNs）在求解偏微分方程（PDEs）方面展现出巨大潜力。然而，传统PINNs在求解复杂PDEs时常常面临挑战，特别是涉及多尺度行为或具有尖锐/振荡特性的解。为了精确且自适应地定位求解过程中的关键失效区域，我们提出了一种基于白盒对抗攻击的采样策略（WbAR）。WbAR沿损失梯度方向搜索失效区域，从而直接定位最关键的位置。该方法以随机游走方式生成对抗样本，并通过迭代优化PINNs来引导模型在训练过程中关注动态更新的关键区域。我们将WbAR应用于多尺度系数椭圆方程、多峰解泊松方程、高维泊松方程以及具有尖锐解的Burgers方程。结果表明，WbAR能有效定位并减少失效区域。此外，由于通过对抗攻击定位失效区域的方法不依赖于失效区域的大小或分布复杂度，WbAR特别适用于求解复杂PDEs。



## **30. Memory Backdoor Attacks on Neural Networks**

针对神经网络的内存后门攻击 cs.CR

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2411.14516v2) [paper-pdf](https://arxiv.org/pdf/2411.14516v2)

**Confidence**: 0.90

**Authors**: Eden Luzon, Guy Amit, Roy Weiss, Torsten Kraub, Alexandra Dmitrienko, Yisroel Mirsky

**Abstract**: Neural networks are often trained on proprietary datasets, making them attractive attack targets. We present a novel dataset extraction method leveraging an innovative training time backdoor attack, allowing a malicious federated learning server to systematically and deterministically extract complete client training samples through a simple indexing process. Unlike prior techniques, our approach guarantees exact data recovery rather than probabilistic reconstructions or hallucinations, provides precise control over which samples are memorized and how many, and shows high capacity and robustness. Infected models output data samples when they receive a patternbased index trigger, enabling systematic extraction of meaningful patches from each clients local data without disrupting global model utility. To address small model output sizes, we extract patches and then recombined them. The attack requires only a minor modification to the training code that can easily evade detection during client-side verification. Hence, this vulnerability represents a realistic FL supply-chain threat, where a malicious server can distribute modified training code to clients and later recover private data from their updates. Evaluations across classifiers, segmentation models, and large language models demonstrate that thousands of sensitive training samples can be recovered from client models with minimal impact on task performance, and a clients entire dataset can be stolen after multiple FL rounds. For instance, a medical segmentation dataset can be extracted with only a 3 percent utility drop. These findings expose a critical privacy vulnerability in FL systems, emphasizing the need for stronger integrity and transparency in distributed training pipelines.

摘要: 神经网络通常在专有数据集上进行训练，这使其成为有吸引力的攻击目标。我们提出了一种新颖的数据集提取方法，利用创新的训练时后门攻击，使恶意的联邦学习服务器能够通过简单的索引过程系统且确定性地提取完整的客户端训练样本。与先前技术不同，我们的方法保证精确的数据恢复而非概率性重建或幻觉生成，提供对记忆样本及其数量的精确控制，并展现出高容量和鲁棒性。受感染的模型在接收到基于模式的索引触发时输出数据样本，从而能够在不破坏全局模型效用的情况下，系统地从每个客户端的本地数据中提取有意义的补丁。针对模型输出尺寸较小的问题，我们提取补丁后进行重组。该攻击仅需对训练代码进行微小修改，极易在客户端验证期间逃避检测。因此，这种漏洞构成了现实的联邦学习供应链威胁，恶意服务器可向客户端分发修改后的训练代码，随后从其更新中恢复私有数据。通过对分类器、分割模型和大语言模型的评估表明，可从客户端模型中恢复数千个敏感训练样本，且对任务性能影响极小；经过多轮联邦学习后，可窃取客户端的完整数据集。例如，医疗分割数据集仅需付出3%的效用下降即可被提取。这些发现揭示了联邦学习系统中严重的隐私漏洞，强调了分布式训练管道需要更强的完整性和透明度。



## **31. Backdoor Attacks on Multi-modal Contrastive Learning**

多模态对比学习中的后门攻击 cs.LG

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.11006v1) [paper-pdf](https://arxiv.org/pdf/2601.11006v1)

**Confidence**: 0.85

**Authors**: Simi D Kuniyilh, Rita Machacy

**Abstract**: Contrastive learning has become a leading self- supervised approach to representation learning across domains, including vision, multimodal settings, graphs, and federated learning. However, recent studies have shown that contrastive learning is susceptible to backdoor and data poisoning attacks. In these attacks, adversaries can manipulate pretraining data or model updates to insert hidden malicious behavior. This paper offers a thorough and comparative review of backdoor attacks in contrastive learning. It analyzes threat models, attack methods, target domains, and available defenses. We summarize recent advancements in this area, underline the specific vulnerabilities inherent to contrastive learning, and discuss the challenges and future research directions. Our findings have significant implications for the secure deployment of systems in industrial and distributed environments.

摘要: 对比学习已成为跨领域（包括视觉、多模态设置、图学习和联邦学习）表示学习的主要自监督方法。然而，近期研究表明，对比学习易受后门攻击和数据投毒攻击。在这些攻击中，攻击者可通过操纵预训练数据或模型更新来植入隐藏的恶意行为。本文对对比学习中的后门攻击进行了全面且比较性的综述，分析了威胁模型、攻击方法、目标领域及现有防御措施。我们总结了该领域的最新进展，强调了对比学习固有的特定漏洞，并讨论了挑战与未来研究方向。我们的发现对工业和分布式环境中系统的安全部署具有重要意义。



## **32. Abusing the Internet of Medical Things: Evaluating Threat Models and Forensic Readiness for Multi-Vector Attacks on Connected Healthcare Devices**

滥用医疗物联网：评估联网医疗设备多向量攻击的威胁模型与取证准备 cs.CR

In review at IEEE Euro S&P 2026

**SubmitDate**: 2026-01-18    [abs](http://arxiv.org/abs/2601.12593v1) [paper-pdf](https://arxiv.org/pdf/2601.12593v1)

**Confidence**: 0.85

**Authors**: Isabel Straw, Akhil Polamarasetty, Mustafa Jaafar

**Abstract**: Individuals experiencing interpersonal violence (IPV), who depend on medical devices, represent a uniquely vulnerable population as healthcare technologies become increasingly connected. Despite rapid growth in MedTech innovation and "health-at-home" ecosystems, the intersection of MedTech cybersecurity and technology-facilitated abuse remains critically under-examined. IPV survivors who rely on therapeutic devices encounter a qualitatively different threat environment from the external, technically sophisticated adversaries typically modeled in MedTech cybersecurity research. We address this gap through two complementary methods: (1) the development of hazard-integrated threat models that fuse Cyber physical system security modeling with tech-abuse frameworks, and (2) an immersive simulation with practitioners, deploying a live version of our model, identifying gaps in digital forensic practice.   Our hazard-integrated CIA threat models map exploits to acute and chronic biological effects, uncovering (i) Integrity attack pathways that facilitate "Medical gaslighting" and "Munchausen-by-IoMT", (ii) Availability attacks that create life-critical and sub-acute harms (glycaemic emergencies, blindness, mood destabilization), and (iii) Confidentiality threats arising from MedTech advertisements (geolocation tracking from BLE broadcasts). Our simulation demonstrates that these attack surfaces are unlikely to be detected in practice: participants overlooked MedTech, misclassified reproductive and assistive technologies, and lacked awareness of BLE broadcast artifacts. Our findings show that MedTech cybersecurity in IPV contexts requires integrated threat modeling and improved forensic capabilities for detecting, preserving and interpreting harms arising from compromised patient-technology ecosystems.

摘要: 随着医疗技术日益互联，依赖医疗设备的人际暴力（IPV）受害者构成了一个特别脆弱的群体。尽管医疗技术创新和'居家健康'生态系统快速发展，但医疗技术网络安全与技术助长虐待之间的交叉领域仍严重缺乏研究。依赖治疗设备的IPV幸存者面临的威胁环境，与医疗技术网络安全研究中通常建模的外部技术复杂对手存在本质差异。我们通过两种互补方法填补这一空白：（1）开发融合信息物理系统安全建模与技术滥用框架的危害集成威胁模型；（2）与从业者进行沉浸式模拟，部署我们模型的实时版本，识别数字取证实践中的差距。我们的危害集成CIA威胁模型将漏洞利用映射到急性和慢性生物效应，揭示了：（i）促成'医疗煤气灯效应'和'IoMT代理孟乔森综合征'的完整性攻击路径；（ii）造成生命危急和亚急性伤害（血糖急症、失明、情绪失稳）的可用性攻击；（iii）医疗技术广告引发的保密性威胁（BLE广播的地理位置追踪）。我们的模拟表明这些攻击面在实践中难以被检测：参与者忽视了医疗技术设备，误分类了生殖和辅助技术，且对BLE广播痕迹缺乏认知。研究结果表明，IPV背景下的医疗技术网络安全需要集成威胁建模，并提升检测、保存和解释受损患者-技术生态系统所造成伤害的取证能力。



## **33. Social Engineering Attacks: A Systemisation of Knowledge on People Against Humans**

社会工程攻击：针对人类的知识系统化研究 cs.CR

10 pages, 6 Figures, 3 Tables

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2601.04215v1) [paper-pdf](https://arxiv.org/pdf/2601.04215v1)

**Confidence**: 0.85

**Authors**: Scott Thomson, Michael Bewong, Arash Mahboubi, Tanveer Zia

**Abstract**: Our systematisation of knowledge on Social Engineering Attacks (SEAs), identifies the human, organisational, and adversarial dimensions of cyber threats. It addresses the growing risks posed by SEAs, highly relevant in the context physical cyber places, such as travellers at airports and residents in smart cities, and synthesizes findings from peer reviewed studies, industry and government reports to inform effective countermeasures that can be embedded into future smart city strategies. SEAs increasingly sidestep technical controls by weaponising leaked personal data and behavioural cues, an urgency underscored by the Optus, Medibank and now Qantas (2025) mega breaches that placed millions of personal records in criminals' hands. Our review surfaces three critical dimensions: (i) human factors of knowledge, abilities and behaviours (KAB) (ii) organisational culture and informal norms that shape those behaviours and (iii) attacker motivations, techniques and return on investment calculations. Our contributions are threefold: (1) TriLayer Systematisation: to the best of our knowledge, we are the first to unify KAB metrics, cultural drivers and attacker economics into a single analytical lens, enabling practitioners to see how vulnerabilities, norms and threat incentives coevolve. (2) Risk Weighted HAISQ Meta analysis: By normalising and ranking HAISQ scores across recent field studies, we reveal persistent high risk clusters (Internet and Social Media use) and propose impact weightings that make the instrument predictive rather than descriptive. (3) Adaptive 'Segment and Simulate' Training Blueprint: Building on clustering evidence, we outline a differentiated programme that matches low, medium, high risk user cohorts to experiential learning packages including phishing simulations, gamified challenges and realtime feedback thereby aligning effort with measured exposure.

摘要: 我们对社会工程攻击（SEAs）的知识系统化研究，识别了网络威胁中的人为、组织和对抗维度。该研究针对SEAs带来的日益增长的风险——在物理网络空间（如机场旅客和智慧城市居民）中尤为突出，综合了同行评审研究、行业和政府报告的研究成果，为未来智慧城市战略中嵌入有效对策提供依据。SEAs通过利用泄露的个人数据和行为线索规避技术控制，Optus、Medibank以及近期Qantas（2025年）大规模数据泄露事件使数千万个人记录落入犯罪分子之手，凸显了问题的紧迫性。我们的研究揭示了三个关键维度：（i）知识、能力与行为（KAB）的人为因素；（ii）塑造这些行为的组织文化与隐性规范；（iii）攻击者动机、技术及投资回报计算。本研究的贡献有三方面：（1）三层系统化框架：据我们所知，我们首次将KAB指标、文化驱动因素和攻击者经济学纳入统一分析视角，使从业者能够洞察漏洞、规范与威胁诱因如何协同演化；（2）风险加权HAISQ元分析：通过对近期实地研究的HAISQ分数进行标准化排序，揭示了持续存在的高风险集群（互联网与社交媒体使用），并提出使该工具具备预测性而非描述性的影响权重；（3）自适应‘分群模拟’培训蓝图：基于聚类证据，我们设计了差异化培训方案，将低、中、高风险用户群体匹配至体验式学习模块（包括钓鱼模拟、游戏化挑战和实时反馈），使培训投入与实测风险暴露程度相匹配。



## **34. To See or Not to See -- Fingerprinting Devices in Adversarial Environments Amid Advanced Machine Learning**

可见与否——高级机器学习环境下对抗性设备指纹识别研究 cs.CR

10 pages, 4 figures

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2504.08264v2) [paper-pdf](https://arxiv.org/pdf/2504.08264v2)

**Confidence**: 0.85

**Authors**: Justin Feng, Amirmohammad Haddad, Nader Sehatbakhsh

**Abstract**: The increasing use of the Internet of Things raises security concerns. To address this, device fingerprinting is often employed to authenticate devices, detect adversaries, and identify eavesdroppers in an environment. This requires the ability to discern between legitimate and malicious devices which is achieved by analyzing the unique physical and/or operational characteristics of IoT devices. In the era of the latest progress in machine learning, particularly generative models, it is crucial to methodically examine the current studies in device fingerprinting. This involves explaining their approaches and underscoring their limitations when faced with adversaries armed with these ML tools. To systematically analyze existing methods, we propose a generic, yet simplified, model for device fingerprinting. Additionally, we thoroughly investigate existing methods to authenticate devices and detect eavesdropping, using our proposed model. We further study trends and similarities between works in authentication and eavesdropping detection and present the existing threats and attacks in these domains. Finally, we discuss future directions in fingerprinting based on these trends to develop more secure IoT fingerprinting schemes.

摘要: 物联网的日益普及引发了安全担忧。为此，设备指纹识别常被用于环境中的设备认证、攻击者检测和窃听者识别。这需要能够区分合法与恶意设备，通过分析物联网设备独特的物理和/或操作特征来实现。在机器学习特别是生成模型取得最新进展的时代，系统性地审视当前设备指纹识别研究至关重要。这包括解释其方法，并强调在面对掌握这些机器学习工具的对手时的局限性。为系统分析现有方法，我们提出了一个通用且简化的设备指纹识别模型。此外，我们利用该模型深入研究了现有的设备认证和窃听检测方法。我们进一步探讨了认证与窃听检测研究之间的趋势和相似性，并呈现了这些领域现有的威胁和攻击。最后，基于这些趋势，我们讨论了指纹识别技术的未来发展方向，以构建更安全的物联网指纹识别方案。



## **35. NoisyHate: Mining Online Human-Written Perturbations for Realistic Robustness Benchmarking of Content Moderation Models**

NoisyHate：挖掘在线人工扰动文本用于内容审核模型的现实鲁棒性基准测试 cs.LG

Accepted to International AAAI Conference on Web and Social Media (ICWSM 2025)

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2303.10430v2) [paper-pdf](https://arxiv.org/pdf/2303.10430v2)

**Confidence**: 0.85

**Authors**: Yiran Ye, Thai Le, Dongwon Lee

**Abstract**: Online texts with toxic content are a clear threat to the users on social media in particular and society in general. Although many platforms have adopted various measures (e.g., machine learning-based hate-speech detection systems) to diminish their effect, toxic content writers have also attempted to evade such measures by using cleverly modified toxic words, so-called human-written text perturbations. Therefore, to help build automatic detection tools to recognize those perturbations, prior methods have developed sophisticated techniques to generate diverse adversarial samples. However, we note that these ``algorithms"-generated perturbations do not necessarily capture all the traits of ``human"-written perturbations. Therefore, in this paper, we introduce a novel, high-quality dataset of human-written perturbations, named as NoisyHate, that was created from real-life perturbations that are both written and verified by human-in-the-loop. We show that perturbations in NoisyHate have different characteristics than prior algorithm-generated toxic datasets show, and thus can be in particular useful to help develop better toxic speech detection solutions. We thoroughly validate NoisyHate against state-of-the-art language models, such as BERT and RoBERTa, and black box APIs, such as Perspective API, on two tasks, such as perturbation normalization and understanding.

摘要: 含有毒性内容的在线文本对社交媒体用户乃至整个社会构成明显威胁。尽管许多平台已采取多种措施（例如基于机器学习的仇恨言论检测系统）来减少其影响，但毒性内容发布者也试图通过巧妙修改毒性词汇（即所谓的人工文本扰动）来规避这些检测。因此，为帮助构建识别此类扰动的自动检测工具，先前方法已开发出复杂技术来生成多样化对抗样本。然而，我们注意到这些“算法”生成的扰动未必能完全捕捉“人工”书写扰动的所有特征。为此，本文引入一个新颖、高质量的人工书写扰动数据集NoisyHate，该数据集源自真实场景中通过人机协同编写与验证的扰动文本。研究表明，NoisyHate中的扰动特征与先前算法生成的毒性数据集存在显著差异，因而特别有助于开发更优的毒性言论检测方案。我们通过扰动文本归一化与理解两项任务，对NoisyHate在BERT、RoBERTa等前沿语言模型及Perspective API等黑盒API上的表现进行了全面验证。



## **36. Serverless AI Security: Attack Surface Analysis and Runtime Protection Mechanisms for FaaS-Based Machine Learning**

无服务器AI安全：基于FaaS的机器学习攻击面分析与运行时保护机制 cs.CR

17 Pages, 2 Figures, 4 Tables

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.11664v1) [paper-pdf](https://arxiv.org/pdf/2601.11664v1)

**Confidence**: 0.85

**Authors**: Chetan Pathade, Vinod Dhimam, Sheheryar Ahmad, Ilsa Lareb

**Abstract**: Serverless computing has achieved widespread adoption, with over 70% of AWS organizations using serverless solutions [1]. Meanwhile, machine learning inference workloads increasingly migrate to Function-as-a-Service (FaaS) platforms for their scalability and cost-efficiency [2], [3], [4]. However, this convergence introduces critical security challenges, with recent reports showing a 220% increase in AI/ML vulnerabilities [5] and serverless computing's fragmented architecture raises new security concerns distinct from traditional cloud deployments [6], [7]. This paper presents the first comprehensive security analysis of machine learning workloads in serverless environments. We systematically characterize the attack surface across five categories: function-level vulnerabilities (cold start exploitation, dependency poisoning), model-specific threats (API-based extraction, adversarial inputs), infrastructure attacks (cross-function contamination, privilege escalation), supply chain risks (malicious layers, backdoored libraries), and IAM complexity (ephemeral nature, serverless functions). Through empirical assessments across AWS Lambda, Azure Functions, and Google Cloud Functions, we demonstrate real-world attack scenarios and quantify their security impact. We propose Serverless AI Shield (SAS), a multi-layered defense framework providing pre-deployment validation, runtime monitoring, and post-execution forensics. Our evaluation shows SAS achieves 94% detection rates while maintaining performance overhead below 9% for inference latency. We release an open-source security toolkit to enable practitioners to assess and harden their serverless AI deployments, advancing the field toward more resilient cloud-native machine learning systems.

摘要: 无服务器计算已获得广泛应用，超过70%的AWS组织采用无服务器解决方案[1]。与此同时，机器学习推理工作负载因其可扩展性和成本效益，正日益迁移至函数即服务（FaaS）平台[2]、[3]、[4]。然而，这种融合带来了关键的安全挑战：近期报告显示AI/ML漏洞数量激增220%[5]，且无服务器计算的碎片化架构引发了不同于传统云部署的新型安全问题[6]、[7]。本文首次对无服务器环境中的机器学习工作负载进行全面安全分析。我们系统性地从五个维度刻画攻击面：函数级漏洞（冷启动利用、依赖项投毒）、模型特定威胁（基于API的提取、对抗性输入）、基础设施攻击（跨函数污染、权限提升）、供应链风险（恶意层、后门库）以及IAM复杂性（临时性、无服务器函数特性）。通过对AWS Lambda、Azure Functions和Google Cloud Functions的实证评估，我们展示了真实攻击场景并量化其安全影响。我们提出Serverless AI Shield（SAS）多层防御框架，提供部署前验证、运行时监控与执行后取证功能。评估显示SAS实现94%的检测率，同时将推理延迟的性能开销控制在9%以下。我们开源安全工具包，帮助从业者评估并加固无服务器AI部署，推动云原生机器学习系统向更高韧性发展。



## **37. Constraint-Guided Prediction Refinement via Deterministic Diffusion Trajectories**

基于确定性扩散轨迹的约束引导预测优化 cs.AI

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2506.12911v2) [paper-pdf](https://arxiv.org/pdf/2506.12911v2)

**Confidence**: 0.85

**Authors**: Pantelis Dogoulis, Fabien Bernier, Félix Fourreau, Karim Tit, Maxime Cordy

**Abstract**: Many real-world machine learning tasks require outputs that satisfy hard constraints, such as physical conservation laws, structured dependencies in graphs, or column-level relationships in tabular data. Existing approaches rely either on domain-specific architectures and losses or on strong assumptions on the constraint space, restricting their applicability to linear or convex constraints. We propose a general-purpose framework for constraint-aware refinement that leverages denoising diffusion implicit models (DDIMs). Starting from a coarse prediction, our method iteratively refines it through a deterministic diffusion trajectory guided by a learned prior and augmented by constraint gradient corrections. The approach accommodates a wide class of non-convex and nonlinear equality constraints and can be applied post hoc to any base model. We demonstrate the method in two representative domains: constrained adversarial attack generation on tabular data with column-level dependencies and in AC power flow prediction under Kirchhoff's laws. Across both settings, our diffusion-guided refinement improves both constraint satisfaction and performance while remaining lightweight and model-agnostic.

摘要: 许多现实世界的机器学习任务要求输出满足硬约束条件，例如物理守恒定律、图中的结构化依赖关系或表格数据中的列级关系。现有方法要么依赖特定领域的架构和损失函数，要么对约束空间做出强假设，仅适用于线性或凸约束。我们提出了一种通用的约束感知优化框架，利用去噪扩散隐式模型（DDIMs）。该方法从粗略预测出发，通过确定性扩散轨迹进行迭代优化，该轨迹由学习到的先验引导，并通过约束梯度校正进行增强。该框架适用于广泛的非凸和非线性等式约束，并且可以后验地应用于任何基础模型。我们在两个代表性领域展示了该方法：具有列级依赖关系的表格数据上的约束对抗攻击生成，以及基尔霍夫定律下的交流潮流预测。在这两种场景中，我们的扩散引导优化在保持轻量级和模型无关性的同时，提高了约束满足度和性能。




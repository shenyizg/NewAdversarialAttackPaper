# LLM / MLLM（语言模型） - 其他攻防技术
**update at 2026-01-25 10:36:50**

按分类器置信度从高到低排序。

## **1. KinGuard: Hierarchical Kinship-Aware Fingerprinting to Defend Against Large Language Model Stealing**

KinGuard：基于层次化亲属关系感知的指纹技术，防御大型语言模型窃取 cs.CR

Accepted by ICASSP2026

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.12986v2) [paper-pdf](https://arxiv.org/pdf/2601.12986v2)

**Confidence**: 0.95

**Authors**: Zhenhua Xu, Xiaoning Tian, Wenjun Zeng, Wenpeng Xing, Tianliang Lu, Gaolei Li, Chaochao Chen, Meng Han

**Abstract**: Protecting the intellectual property of large language models requires robust ownership verification. Conventional backdoor fingerprinting, however, is flawed by a stealth-robustness paradox: to be robust, these methods force models to memorize fixed responses to high-perplexity triggers, but this targeted overfitting creates detectable statistical artifacts. We resolve this paradox with KinGuard, a framework that embeds a private knowledge corpus built on structured kinship narratives. Instead of memorizing superficial triggers, the model internalizes this knowledge via incremental pre-training, and ownership is verified by probing its conceptual understanding. Extensive experiments demonstrate KinGuard's superior effectiveness, stealth, and resilience against a battery of attacks including fine-tuning, input perturbation, and model merging. Our work establishes knowledge-based embedding as a practical and secure paradigm for model fingerprinting.

摘要: 保护大型语言模型的知识产权需要可靠的所有权验证方法。然而，传统的后门指纹技术存在隐蔽性与鲁棒性悖论：为达到鲁棒性，这些方法迫使模型记忆对高困惑度触发器的固定响应，但这种针对性过拟合会产生可检测的统计伪影。我们通过KinGuard框架解决了这一悖论，该框架嵌入了一个基于结构化亲属关系叙事的私有知识语料库。模型通过增量预训练内化这些知识，而非记忆表面触发器，并通过探测其概念理解来验证所有权。大量实验证明，KinGuard在对抗微调、输入扰动和模型合并等多种攻击时，展现出卓越的有效性、隐蔽性和抗攻击能力。我们的工作确立了基于知识嵌入的模型指纹技术作为一种实用且安全的范式。



## **2. DNF: Dual-Layer Nested Fingerprinting for Large Language Model Intellectual Property Protection**

DNF：面向大语言模型知识产权保护的双层嵌套指纹技术 cs.CR

Accepted by ICASSP2026

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.08223v3) [paper-pdf](https://arxiv.org/pdf/2601.08223v3)

**Confidence**: 0.95

**Authors**: Zhenhua Xu, Yiran Zhao, Mengting Zhong, Dezhang Kong, Changting Lin, Tong Qiao, Meng Han

**Abstract**: The rapid growth of large language models raises pressing concerns about intellectual property protection under black-box deployment. Existing backdoor-based fingerprints either rely on rare tokens -- leading to high-perplexity inputs susceptible to filtering -- or use fixed trigger-response mappings that are brittle to leakage and post-hoc adaptation. We propose \textsc{Dual-Layer Nested Fingerprinting} (DNF), a black-box method that embeds a hierarchical backdoor by coupling domain-specific stylistic cues with implicit semantic triggers. Across Mistral-7B, LLaMA-3-8B-Instruct, and Falcon3-7B-Instruct, DNF achieves perfect fingerprint activation while preserving downstream utility. Compared with existing methods, it uses lower-perplexity triggers, remains undetectable under fingerprint detection attacks, and is relatively robust to incremental fine-tuning and model merging. These results position DNF as a practical, stealthy, and resilient solution for LLM ownership verification and intellectual property protection.

摘要: 大语言模型的快速发展引发了黑盒部署场景下知识产权保护的迫切需求。现有基于后门的指纹方法要么依赖罕见标记——导致高困惑度输入易被过滤，要么使用固定的触发-响应映射，容易因泄露和事后适应而失效。我们提出双层嵌套指纹技术（DNF），这是一种通过将领域特定风格线索与隐式语义触发器耦合来嵌入分层后门的黑盒方法。在Mistral-7B、LLaMA-3-8B-Instruct和Falcon3-7B-Instruct上的实验表明，DNF在保持下游任务性能的同时实现了完美的指纹激活。与现有方法相比，DNF使用更低困惑度的触发器，在指纹检测攻击下保持不可察觉性，并对增量微调和模型合并具有相对鲁棒性。这些结果表明DNF为大语言模型所有权验证和知识产权保护提供了一种实用、隐蔽且具有韧性的解决方案。



## **3. PrivTune: Efficient and Privacy-Preserving Fine-Tuning of Large Language Models via Device-Cloud Collaboration**

PrivTune：基于设备-云协作的高效隐私保护大语言模型微调框架 cs.CR

Accepted at IEEE INFOCOM 2026 (full version). Update the cited references

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2512.08809v3) [paper-pdf](https://arxiv.org/pdf/2512.08809v3)

**Confidence**: 0.95

**Authors**: Yi Liu, Weixiang Han, Chengjun Cai, Xingliang Yuan, Cong Wang

**Abstract**: With the rise of large language models, service providers offer language models as a service, enabling users to fine-tune customized models via uploaded private datasets. However, this raises concerns about sensitive data leakage. Prior methods, relying on differential privacy within device-cloud collaboration frameworks, struggle to balance privacy and utility, exposing users to inference attacks or degrading fine-tuning performance. To address this, we propose PrivTune, an efficient and privacy-preserving fine-tuning framework via Split Learning (SL). The key idea of PrivTune is to inject crafted noise into token representations from the SL bottom model, making each token resemble the $n$-hop indirect neighbors. PrivTune formulates this as an optimization problem to compute the optimal noise vector, aligning with defense-utility goals. On this basis, it then adjusts the parameters (i.e., mean) of the $d_χ$-Privacy noise distribution to align with the optimization direction and scales the noise according to token importance to minimize distortion. Experiments on five datasets (covering both classification and generation tasks) against three embedding inversion and three attribute inference attacks show that, using RoBERTa on the Stanford Sentiment Treebank dataset, PrivTune reduces the attack success rate to 10% with only a 3.33% drop in utility performance, outperforming state-of-the-art baselines.

摘要: 随着大语言模型的兴起，服务提供商提供语言模型即服务，允许用户通过上传私有数据集微调定制模型。然而，这引发了敏感数据泄露的担忧。现有方法依赖设备-云协作框架中的差分隐私技术，难以平衡隐私与效用，使用户面临推理攻击或导致微调性能下降。为解决这一问题，我们提出PrivTune——一种基于分割学习（SL）的高效隐私保护微调框架。PrivTune的核心思想是在SL底层模型的令牌表示中注入精心设计的噪声，使每个令牌近似于$n$跳间接邻居。PrivTune将此表述为优化问题以计算最优噪声向量，使其符合防御-效用目标。在此基础上，通过调整$d_χ$-隐私噪声分布的参数（即均值）以对齐优化方向，并根据令牌重要性缩放噪声以最小化失真。在五个数据集（涵盖分类与生成任务）上针对三种嵌入反演攻击和三种属性推理攻击的实验表明，在斯坦福情感树库数据集上使用RoBERTa时，PrivTune将攻击成功率降至10%，仅造成3.33%的效用性能下降，优于现有最优基线方法。



## **4. A Visual Semantic Adaptive Watermark grounded by Prefix-Tuning for Large Vision-Language Model**

基于前缀调优的视觉语义自适应水印技术用于大型视觉语言模型 cs.CV

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07291v1) [paper-pdf](https://arxiv.org/pdf/2601.07291v1)

**Confidence**: 0.95

**Authors**: Qi Zheng, Shuliang Liu, Yu Huang, Sihang Jia, Jungang Li, Lyuhao Chen, Junhao Chen, Hanqian Li, Aiwei Liu, Yibo Yan, Xuming Hu

**Abstract**: Watermarking has emerged as a pivotal solution for content traceability and intellectual property protection in Large Vision-Language Models (LVLMs). However, vision-agnostic watermarks introduce visually irrelevant tokens and disrupt visual grounding by enforcing indiscriminate pseudo-random biases, while some semantic-aware methods incur prohibitive inference latency due to rejection sampling. In this paper, we propose the VIsual Semantic Adaptive Watermark (VISA-Mark), a novel framework that embeds detectable signals while strictly preserving visual fidelity. Our approach employs a lightweight, efficiently trained prefix-tuner to extract dynamic Visual-Evidence Weights, which quantify the evidentiary support for candidate tokens based on the visual input. These weights guide an adaptive vocabulary partitioning and logits perturbation mechanism, concentrating watermark strength specifically on visually-supported tokens. By actively aligning the watermark with visual evidence, VISA-Mark effectively maintains visual fidelity. Empirical results confirm that VISA-Mark outperforms conventional methods with a 7.8% improvement in visual consistency (Chair-I) and superior semantic fidelity. The framework maintains highly competitive detection accuracy (96.88% AUC) and robust attack resilience (99.3%) without sacrificing inference efficiency, effectively establishing a new standard for reliability-preserving multimodal watermarking.

摘要: 水印技术已成为大型视觉语言模型（LVLMs）内容溯源和知识产权保护的关键解决方案。然而，视觉无关水印会引入视觉无关的标记，并通过强制施加无差别的伪随机偏差破坏视觉基础；而某些语义感知方法因拒绝采样导致推理延迟过高。本文提出视觉语义自适应水印（VISA-Mark），这是一种在严格保持视觉保真度的同时嵌入可检测信号的新型框架。该方法采用轻量级、高效训练的前缀调优器提取动态视觉证据权重，该权重基于视觉输入量化候选标记的证据支持度。这些权重引导自适应词汇分区和逻辑扰动机制，将水印强度集中作用于视觉支持的标记。通过主动将水印与视觉证据对齐，VISA-Mark有效保持了视觉保真度。实证结果表明，VISA-Mark在视觉一致性（Chair-I指标）上较传统方法提升7.8%，并具有更优的语义保真度。该框架在保持推理效率的同时，实现了极具竞争力的检测准确率（96.88% AUC）和强大的抗攻击能力（99.3%），为可靠性保护的多模态水印技术确立了新标准。



## **5. Large Language Models for Detecting Cyberattacks on Smart Grid Protective Relays**

基于大语言模型的智能电网保护继电器网络攻击检测 cs.CR

**SubmitDate**: 2026-01-07    [abs](http://arxiv.org/abs/2601.04443v1) [paper-pdf](https://arxiv.org/pdf/2601.04443v1)

**Confidence**: 0.95

**Authors**: Ahmad Mohammad Saber, Saeed Jafari, Zhengmao Ouyang, Paul Budnarain, Amr Youssef, Deepa Kundur

**Abstract**: This paper presents a large language model (LLM)-based framework for detecting cyberattacks on transformer current differential relays (TCDRs), which, if undetected, may trigger false tripping of critical transformers. The proposed approach adapts and fine-tunes compact LLMs such as DistilBERT to distinguish cyberattacks from actual faults using textualized multidimensional TCDR current measurements recorded before and after tripping. Our results demonstrate that DistilBERT detects 97.6% of cyberattacks without compromising TCDR dependability and achieves inference latency below 6 ms on a commercial workstation. Additional evaluations confirm the framework's robustness under combined time-synchronization and false-data-injection attacks, resilience to measurement noise, and stability across prompt formulation variants. Furthermore, GPT-2 and DistilBERT+LoRA achieve comparable performance, highlighting the potential of LLMs for enhancing smart grid cybersecurity. We provide the full dataset used in this study for reproducibility.

摘要: 本文提出了一种基于大语言模型（LLM）的框架，用于检测变压器电流差动继电器（TCDR）的网络攻击，若未及时发现此类攻击，可能导致关键变压器误跳闸。该方法通过适配和微调紧凑型LLM（如DistilBERT），利用跳闸前后记录的多维TCDR电流测量文本化数据，区分网络攻击与实际故障。实验结果表明，DistilBERT能检测97.6%的网络攻击，且不影响TCDR可靠性，在商用工作站上推理延迟低于6毫秒。进一步评估证实，该框架在时间同步与虚假数据注入组合攻击下具有鲁棒性，对测量噪声具有抗扰性，且在不同提示词变体中表现稳定。此外，GPT-2和DistilBERT+LoRA模型也实现了可比性能，凸显了LLM在增强智能电网网络安全方面的潜力。本研究提供了完整数据集以确保可复现性。



## **6. Multimodal Adversarial Defense for Vision-Language Models by Leveraging One-To-Many Relationships**

利用一对多关系的视觉语言模型多模态对抗防御 cs.CV

WACV 2026 Accepted. Code available at https://github.com/CyberAgentAILab/multimodal-adversarial-training

**SubmitDate**: 2026-01-05    [abs](http://arxiv.org/abs/2405.18770v6) [paper-pdf](https://arxiv.org/pdf/2405.18770v6)

**Confidence**: 0.95

**Authors**: Futa Waseda, Antonio Tejero-de-Pablos, Isao Echizen

**Abstract**: Pre-trained vision-language (VL) models are highly vulnerable to adversarial attacks. However, existing defense methods primarily focus on image classification, overlooking two key aspects of VL tasks: multimodal attacks, where both image and text can be perturbed, and the one-to-many relationship of images and texts, where a single image can correspond to multiple textual descriptions and vice versa (1:N and N:1). This work is the first to explore defense strategies against multimodal attacks in VL tasks, whereas prior VL defense methods focus on vision robustness. We propose multimodal adversarial training (MAT), which incorporates adversarial perturbations in both image and text modalities during training, significantly outperforming existing unimodal defenses. Furthermore, we discover that MAT is limited by deterministic one-to-one (1:1) image-text pairs in VL training data. To address this, we conduct a comprehensive study on leveraging one-to-many relationships to enhance robustness, investigating diverse augmentation techniques. Our analysis shows that, for a more effective defense, augmented image-text pairs should be well-aligned, diverse, yet avoid distribution shift -- conditions overlooked by prior research. This work pioneers defense strategies against multimodal attacks, providing insights for building robust VLMs from both optimization and data perspectives. Our code is publicly available at https://github.com/CyberAgentAILab/multimodal-adversarial-training.

摘要: 预训练的视觉语言（VL）模型极易受到对抗攻击。然而，现有的防御方法主要集中于图像分类，忽略了VL任务的两个关键方面：多模态攻击（图像和文本均可被扰动）以及图像与文本的一对多关系（单张图像可对应多个文本描述，反之亦然，即1:N和N:1）。本研究首次探索针对VL任务中多模态攻击的防御策略，而先前的VL防御方法仅关注视觉鲁棒性。我们提出了多模态对抗训练（MAT），在训练过程中同时引入图像和文本模态的对抗扰动，其性能显著优于现有的单模态防御方法。此外，我们发现MAT受限于VL训练数据中确定性的一对一（1:1）图像-文本配对。为解决此问题，我们全面研究了如何利用一对多关系增强鲁棒性，并探索了多种数据增强技术。分析表明，为实现更有效的防御，增强后的图像-文本对应应保持良好对齐、具有多样性，同时避免分布偏移——这些条件在先前研究中被忽视。本研究开创了针对多模态攻击的防御策略，从优化和数据两个角度为构建鲁棒的VLM提供了见解。代码已公开于https://github.com/CyberAgentAILab/multimodal-adversarial-training。



## **7. Who Can See Through You? Adversarial Shielding Against VLM-Based Attribute Inference Attacks**

谁能看透你？针对基于VLM的属性推断攻击的对抗性防护 cs.CV

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2512.18264v1) [paper-pdf](https://arxiv.org/pdf/2512.18264v1)

**Confidence**: 0.95

**Authors**: Yucheng Fan, Jiawei Chen, Yu Tian, Zhaoxia Yin

**Abstract**: As vision-language models (VLMs) become widely adopted, VLM-based attribute inference attacks have emerged as a serious privacy concern, enabling adversaries to infer private attributes from images shared on social media. This escalating threat calls for dedicated protection methods to safeguard user privacy. However, existing methods often degrade the visual quality of images or interfere with vision-based functions on social media, thereby failing to achieve a desirable balance between privacy protection and user experience. To address this challenge, we propose a novel protection method that jointly optimizes privacy suppression and utility preservation under a visual consistency constraint. While our method is conceptually effective, fair comparisons between methods remain challenging due to the lack of publicly available evaluation datasets. To fill this gap, we introduce VPI-COCO, a publicly available benchmark comprising 522 images with hierarchically structured privacy questions and corresponding non-private counterparts, enabling fine-grained and joint evaluation of protection methods in terms of privacy preservation and user experience. Building upon this benchmark, experiments on multiple VLMs demonstrate that our method effectively reduces PAR below 25%, keeps NPAR above 88%, maintains high visual consistency, and generalizes well to unseen and paraphrased privacy questions, demonstrating its strong practical applicability for real-world VLM deployments.

摘要: 随着视觉语言模型（VLMs）的广泛应用，基于VLM的属性推断攻击已成为严重的隐私威胁，使攻击者能够从社交媒体分享的图像中推断私人属性。这一日益严峻的威胁亟需专门的防护方法来保护用户隐私。然而，现有方法往往会降低图像的视觉质量或干扰社交媒体上的视觉功能，难以在隐私保护与用户体验之间取得理想平衡。为应对这一挑战，我们提出了一种新颖的防护方法，在视觉一致性约束下联合优化隐私抑制与功能保持。尽管我们的方法在概念上有效，但由于缺乏公开可用的评估数据集，方法间的公平比较仍具挑战性。为此，我们推出了VPI-COCO公开基准数据集，包含522张带有分层结构隐私问题及对应非隐私对照的图像，支持对防护方法在隐私保护与用户体验方面进行细粒度联合评估。基于该基准的实验表明，我们的方法在多个VLM上能将隐私属性识别率（PAR）降至25%以下，保持非隐私属性识别率（NPAR）高于88%，维持高视觉一致性，并能良好泛化至未见及转述的隐私问题，展现了其在真实世界VLM部署中的强大实用价值。



## **8. MoAPT: Mixture of Adversarial Prompt Tuning for Vision-Language Models**

MoAPT：视觉语言模型的对抗性提示调优混合方法 cs.CV

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2505.17509v2) [paper-pdf](https://arxiv.org/pdf/2505.17509v2)

**Confidence**: 0.95

**Authors**: Shiji Zhao, Qihui Zhu, Shukun Xiong, Shouwei Ruan, Maoxun Yuan, Jialing Tao, Jiexi Liu, Ranjie Duan, Jie Zhang, Jie Zhang, Xingxing Wei

**Abstract**: Large pre-trained Vision Language Models (VLMs) demonstrate excellent generalization capabilities but remain highly susceptible to adversarial examples, posing potential security risks. To improve the robustness of VLMs against adversarial examples, adversarial prompt tuning methods are proposed to align the text feature with the adversarial image feature without changing model parameters. However, when facing various adversarial attacks, a single learnable text prompt has insufficient generalization to align well with all adversarial image features, which ultimately results in overfitting. To address the above challenge, in this paper, we empirically find that increasing the number of learned prompts yields greater robustness improvements than simply extending the length of a single prompt. Building on this observation, we propose an adversarial tuning method named \textbf{Mixture of Adversarial Prompt Tuning (MoAPT)} to enhance the generalization against various adversarial attacks for VLMs. MoAPT aims to learn mixture text prompts to obtain more robust text features. To further enhance the adaptability, we propose a conditional weight router based on the adversarial images to predict the mixture weights of multiple learned prompts, which helps obtain sample-specific mixture text features aligning with different adversarial image features. Extensive experiments across 11 datasets under different settings show that our method can achieve better adversarial robustness than state-of-the-art approaches.

摘要: 大规模预训练视觉语言模型（VLMs）展现出优异的泛化能力，但仍极易受到对抗样本的攻击，存在潜在安全风险。为提升VLMs对抗样本的鲁棒性，研究者提出了对抗性提示调优方法，通过在不改变模型参数的情况下对齐文本特征与对抗性图像特征。然而，面对多种对抗攻击时，单一可学习文本提示的泛化能力不足，难以与所有对抗性图像特征良好对齐，最终导致过拟合。针对这一挑战，本文通过实证发现：增加学习提示的数量比单纯延长单个提示长度能带来更大的鲁棒性提升。基于此观察，我们提出名为\textbf{对抗性提示调优混合方法（MoAPT）}的对抗调优方法，以增强VLMs针对多种对抗攻击的泛化能力。MoAPT旨在通过学习混合文本提示来获得更鲁棒的文本特征。为进一步增强适应性，我们提出基于对抗性图像的条件权重路由器，用于预测多个学习提示的混合权重，从而获得与不同对抗性图像特征对齐的样本特定混合文本特征。在11个数据集上的大量实验表明，我们的方法在不同设置下均能取得优于现有最先进方法的对抗鲁棒性。



## **9. Proxy Robustness in Vision Language Models is Effortlessly Transferable**

视觉语言模型中的代理鲁棒性可轻松迁移 cs.CV

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2601.12865v1) [paper-pdf](https://arxiv.org/pdf/2601.12865v1)

**Confidence**: 0.95

**Authors**: Xiaowei Fu, Fuxiang Huang, Lei Zhang

**Abstract**: As a pivotal technique for improving the defense of deep models, adversarial robustness transfer via distillation has demonstrated remarkable success in conventional image classification tasks. However, this paradigm encounters critical challenges when applied to vision-language models (VLM) (e.g., CLIP): constructing adversarially robust teacher for large-scale multi-modal models demands prohibitively high computational resources. We bridge this gap by revealing an interesting phenomenon: vanilla CLIP (without adversarial training) exhibits intrinsic defensive capabilities against adversarial examples generated by another CLIP with different architectures. We formally define this as proxy adversarial robustness, and naturally propose a Heterogeneous Proxy Transfer (HPT) framework that establishes cross-architectural robustness distillation channels between CLIP variants, effortlessly enabling the VLM robustness transfer from proxy to target models. Yet, such proxy transfer paradigm easily induces severe overfitting, leading to a sharp degradation in zero-shot natural generalization. To resolve that, we design Generalization-Pivot Decoupling (GPD) by leveraging the difference in learning rate scheduling. This decouples the proxy transfer process into a generalization-anchored warm-up that maintains generalization and a generalization-pulled HPT that promotes adversarial robustness, to achieve an equilibrium between natural generalization and adversarial robustness. Extensive experiments on 15 zero-shot datasets demonstrate the effectiveness of our HPT-GPD method. The code is available at the website of github.com/fxw13/HPT-GPD.

摘要: 作为提升深度模型防御能力的关键技术，通过蒸馏实现对抗鲁棒性迁移在传统图像分类任务中已取得显著成功。然而，当应用于视觉语言模型（VLM）（如CLIP）时，该范式面临关键挑战：为大规模多模态模型构建对抗鲁棒的教师模型需要极高的计算资源。我们通过揭示一个有趣现象来弥合这一差距：普通CLIP（未经对抗训练）对由不同架构的CLIP生成的对抗样本表现出内在防御能力。我们正式将其定义为代理对抗鲁棒性，并自然提出异构代理迁移（HPT）框架，在CLIP变体间建立跨架构鲁棒性蒸馏通道，轻松实现VLM鲁棒性从代理模型到目标模型的迁移。然而，此类代理迁移范式易引发严重过拟合，导致零样本自然泛化能力急剧下降。为解决此问题，我们利用学习率调度差异设计了泛化枢轴解耦（GPD），将代理迁移过程解耦为保持泛化能力的泛化锚定预热阶段和提升对抗鲁棒性的泛化牵引HPT阶段，以实现自然泛化与对抗鲁棒性间的平衡。在15个零样本数据集上的大量实验证明了HPT-GPD方法的有效性。代码发布于github.com/fxw13/HPT-GPD。



## **10. SoK: Privacy-aware LLM in Healthcare: Threat Model, Privacy Techniques, Challenges and Recommendations**

SoK：医疗保健领域隐私感知大语言模型：威胁模型、隐私技术、挑战与建议 cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10004v1) [paper-pdf](https://arxiv.org/pdf/2601.10004v1)

**Confidence**: 0.95

**Authors**: Mohoshin Ara Tahera, Karamveer Singh Sidhu, Shuvalaxmi Dass, Sajal Saha

**Abstract**: Large Language Models (LLMs) are increasingly adopted in healthcare to support clinical decision-making, summarize electronic health records (EHRs), and enhance patient care. However, this integration introduces significant privacy and security challenges, driven by the sensitivity of clinical data and the high-stakes nature of medical workflows. These risks become even more pronounced across heterogeneous deployment environments, ranging from small on-premise hospital systems to regional health networks, each with unique resource limitations and regulatory demands. This Systematization of Knowledge (SoK) examines the evolving threat landscape across the three core LLM phases: Data preprocessing, Fine-tuning, and Inference within realistic healthcare settings. We present a detailed threat model that characterizes adversaries, capabilities, and attack surfaces at each phase, and we systematize how existing privacy-preserving techniques (PPTs) attempt to mitigate these vulnerabilities. While existing defenses show promise, our analysis identifies persistent limitations in securing sensitive clinical data across diverse operational tiers. We conclude with phase-aware recommendations and future research directions aimed at strengthening privacy guarantees for LLMs in regulated environments. This work provides a foundation for understanding the intersection of LLMs, threats, and privacy in healthcare, offering a roadmap toward more robust and clinically trustworthy AI systems.

摘要: 大语言模型（LLMs）在医疗保健领域的应用日益广泛，用于支持临床决策、总结电子健康记录（EHRs）以及提升患者护理水平。然而，这种整合带来了显著的隐私和安全挑战，这源于临床数据的敏感性以及医疗工作流程的高风险性。在异构部署环境中，这些风险尤为突出，范围涵盖从小型本地医院系统到区域健康网络，每个环境都有独特的资源限制和监管要求。本文通过知识系统化（SoK）方法，探讨了在现实医疗场景中，LLMs三个核心阶段（数据预处理、微调和推理）不断演变的威胁态势。我们提出了一个详细的威胁模型，描述了每个阶段的对手、能力和攻击面，并系统化分析了现有隐私保护技术（PPTs）如何尝试缓解这些漏洞。尽管现有防御措施显示出潜力，但我们的分析揭示了在不同操作层级保护敏感临床数据方面仍存在持续局限性。最后，我们提出了针对各阶段的建议和未来研究方向，旨在加强受监管环境中LLMs的隐私保障。这项工作为理解医疗保健领域LLMs、威胁和隐私的交叉点奠定了基础，为构建更稳健且临床可信的人工智能系统提供了路线图。



## **11. Rethinking On-Device LLM Reasoning: Why Analogical Mapping Outperforms Abstract Thinking for IoT DDoS Detection**

重新思考设备端LLM推理：为何类比映射在物联网DDoS检测中优于抽象思维 cs.CR

**SubmitDate**: 2026-01-20    [abs](http://arxiv.org/abs/2601.14343v1) [paper-pdf](https://arxiv.org/pdf/2601.14343v1)

**Confidence**: 0.90

**Authors**: William Pan, Guiran Liu, Binrong Zhu, Qun Wang, Yingzhou Lu, Beiyu Lin, Rose Qingyang Hu

**Abstract**: The rapid expansion of IoT deployments has intensified cybersecurity threats, notably Distributed Denial of Service (DDoS) attacks, characterized by increasingly sophisticated patterns. Leveraging Generative AI through On-Device Large Language Models (ODLLMs) provides a viable solution for real-time threat detection at the network edge, though limited computational resources present challenges for smaller ODLLMs. This paper introduces a novel detection framework that integrates Chain-of-Thought (CoT) reasoning with Retrieval-Augmented Generation (RAG), tailored specifically for IoT edge environments. We systematically evaluate compact ODLLMs, including LLaMA 3.2 (1B, 3B) and Gemma 3 (1B, 4B), using structured prompting and exemplar-driven reasoning strategies. Experimental results demonstrate substantial performance improvements with few-shot prompting, achieving macro-average F1 scores as high as 0.85. Our findings highlight the significant advantages of incorporating exemplar-based reasoning, underscoring that CoT and RAG approaches markedly enhance small ODLLMs' capabilities in accurately classifying complex network attacks under stringent resource constraints.

摘要: 物联网部署的快速扩张加剧了网络安全威胁，尤其是分布式拒绝服务（DDoS）攻击，其攻击模式日益复杂。通过设备端大语言模型（ODLLM）利用生成式AI为网络边缘的实时威胁检测提供了可行方案，但有限的计算资源对小型ODLLM构成了挑战。本文提出了一种新颖的检测框架，将思维链（CoT）推理与检索增强生成（RAG）相结合，专为物联网边缘环境设计。我们系统评估了紧凑型ODLLM（包括LLaMA 3.2的1B/3B版本和Gemma 3的1B/4B版本），采用结构化提示和示例驱动推理策略。实验结果表明，少样本提示带来显著性能提升，宏平均F1分数最高达0.85。我们的研究凸显了基于示例推理的重要优势，证明CoT和RAG方法能显著增强小型ODLLM在严格资源限制下准确分类复杂网络攻击的能力。



## **12. Multifaceted Evaluation of Audio-Visual Capability for MLLMs: Effectiveness, Efficiency, Generalizability and Robustness**

MLLMs音频-视觉能力的多维度评估：有效性、效率、泛化性与鲁棒性 cs.MM

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.16936v1) [paper-pdf](https://arxiv.org/pdf/2504.16936v1)

**Confidence**: 0.90

**Authors**: Yusheng Zhao, Junyu Luo, Xiao Luo, Weizhi Zhang, Zhiping Xiao, Wei Ju, Philip S. Yu, Ming Zhang

**Abstract**: Multi-modal large language models (MLLMs) have recently achieved great success in processing and understanding information from diverse modalities (e.g., text, audio, and visual signals). Despite their growing popularity, there remains a lack of comprehensive evaluation measuring the audio-visual capabilities of these models, especially in diverse scenarios (e.g., distribution shifts and adversarial attacks). In this paper, we present a multifaceted evaluation of the audio-visual capability of MLLMs, focusing on four key dimensions: effectiveness, efficiency, generalizability, and robustness. Through extensive experiments, we find that MLLMs exhibit strong zero-shot and few-shot generalization abilities, enabling them to achieve great performance with limited data. However, their success relies heavily on the vision modality, which impairs performance when visual input is corrupted or missing. Additionally, while MLLMs are susceptible to adversarial samples, they demonstrate greater robustness compared to traditional models. The experimental results and our findings provide insights into the audio-visual capabilities of MLLMs, highlighting areas for improvement and offering guidance for future research.

摘要: 多模态大语言模型（MLLMs）近年来在处理和理解来自不同模态（如文本、音频和视觉信号）的信息方面取得了巨大成功。尽管其日益普及，但目前仍缺乏对这些模型音频-视觉能力的全面评估，尤其是在多样化场景（如分布偏移和对抗攻击）中。本文对MLLMs的音频-视觉能力进行了多维度评估，重点关注四个关键维度：有效性、效率、泛化性和鲁棒性。通过大量实验，我们发现MLLMs展现出强大的零样本和少样本泛化能力，使其能够在有限数据下实现优异性能。然而，其成功高度依赖视觉模态，当视觉输入受损或缺失时，性能会受到损害。此外，尽管MLLMs易受对抗样本影响，但与传统模型相比，它们表现出更强的鲁棒性。实验结果和我们的发现为理解MLLMs的音频-视觉能力提供了见解，指出了改进方向，并为未来研究提供了指导。



## **13. PhishLumos: An Adaptive Multi-Agent System for Proactive Phishing Campaign Mitigation**

PhishLumos：一种用于主动式钓鱼攻击活动缓解的自适应多智能体系统 cs.CR

Accepted for publication at IEEE ICC 2026

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2509.21772v2) [paper-pdf](https://arxiv.org/pdf/2509.21772v2)

**Confidence**: 0.85

**Authors**: Daiki Chiba, Hiroki Nakano, Takashi Koide

**Abstract**: Phishing attacks are a significant societal threat, disproportionately harming vulnerable populations and eroding trust in essential digital services. Current defenses are often reactive, failing against modern evasive tactics like cloaking that conceal malicious content. To address this, we introduce PhishLumos, an adaptive multi-agent system that proactively mitigates entire attack campaigns. It confronts a core cybersecurity imbalance: attackers can easily scale operations, while defense remains an intensive expert task. Instead of being blocked by evasion, PhishLumos treats it as a critical signal to investigate the underlying infrastructure. Its Large Language Model (LLM)-powered agents uncover shared hosting, certificates, and domain registration patterns. On real-world data, our system identified 100% of campaigns in the median case, over a week before their confirmation by cybersecurity experts. PhishLumos demonstrates a practical shift from reactive URL blocking to proactive campaign mitigation, protecting users before they are harmed and making the digital world safer for all.

摘要: 钓鱼攻击是重大的社会威胁，尤其对弱势群体造成不成比例的伤害，并侵蚀对关键数字服务的信任。现有防御手段多为被动响应，难以应对如隐藏恶意内容的伪装技术等现代规避策略。为此，我们提出PhishLumos——一种自适应多智能体系统，能够主动缓解整个攻击活动。它直面网络安全的核心失衡：攻击者可轻易扩展其操作规模，而防御仍是一项依赖专家的高强度任务。PhishLumos不仅规避技术视为阻碍，更将其作为调查底层基础设施的关键信号。该系统通过基于大语言模型（LLM）的智能体，揭示共享托管、证书及域名注册模式。在真实数据测试中，本系统在网络安全专家确认前一周以上即识别出全部攻击活动（中位数情况）。PhishLumos实现了从被动URL拦截到主动攻击活动缓解的实践转变，在用户受侵害前提供保护，为所有人构建更安全的数字世界。



## **14. Large AI Model-Enabled Secure Communications in Low-Altitude Wireless Networks: Concepts, Perspectives and Case Study**

大型AI模型赋能的低空无线网络安全通信：概念、视角与案例研究 cs.NI

This paper has been accepted to IEEE Communications Magazine

**SubmitDate**: 2026-01-20    [abs](http://arxiv.org/abs/2508.00256v2) [paper-pdf](https://arxiv.org/pdf/2508.00256v2)

**Confidence**: 0.85

**Authors**: Chuang Zhang, Geng Sun, Yijing Lin, Weijie Yuan, Sinem Coleri, Dusit Niyato

**Abstract**: Low-altitude wireless networks (LAWNs) have the potential to revolutionize communications by supporting a range of applications, including urban parcel delivery, aerial inspections and air taxis. However, compared with traditional wireless networks, LAWNs face unique security challenges due to low-altitude operations, frequent mobility and reliance on unlicensed spectrum, making it more vulnerable to some malicious attacks. In this paper, we investigate some large artificial intelligence model (LAM)-enabled solutions for secure communications in LAWNs. Specifically, we first explore the amplified security risks and important limitations of traditional AI methods in LAWNs. Then, we introduce the basic concepts of LAMs and delve into the role of LAMs in addressing these challenges. To demonstrate the practical benefits of LAMs for secure communications in LAWNs, we propose a novel LAM-based optimization framework that leverages large language models (LLMs) to generate enhanced state features on top of handcrafted representations, and to design intrinsic rewards accordingly, thereby improving reinforcement learning performance for secure communication tasks. Through a typical case study, simulation results validate the effectiveness of the proposed framework. Finally, we outline future directions for integrating LAMs into secure LAWN applications.

摘要: 低空无线网络（LAWNs）通过支持城市包裹配送、空中巡检和空中出租车等一系列应用，具有革新通信的潜力。然而，与传统无线网络相比，LAWNs因低空运行、频繁移动及依赖非授权频谱而面临独特的安全挑战，使其更易遭受恶意攻击。本文研究了基于大型人工智能模型（LAMs）的LAWNs安全通信解决方案。具体而言，我们首先探讨了LAWNs中传统AI方法所加剧的安全风险和重要局限性。接着，介绍了LAMs的基本概念，并深入分析了LAMs在应对这些挑战中的作用。为展示LAMs在LAWNs安全通信中的实际优势，我们提出了一种新颖的基于LAM的优化框架，该框架利用大型语言模型（LLMs）在人工设计表征的基础上生成增强状态特征，并据此设计内在奖励，从而提升安全通信任务中强化学习的性能。通过典型案例研究，仿真结果验证了所提框架的有效性。最后，我们展望了将LAMs集成到安全LAWN应用中的未来方向。



## **15. Adversarial Defense in Vision-Language Models: An Overview**

视觉语言模型中的对抗防御：综述 cs.CV

**SubmitDate**: 2026-01-18    [abs](http://arxiv.org/abs/2601.12443v1) [paper-pdf](https://arxiv.org/pdf/2601.12443v1)

**Confidence**: 0.85

**Authors**: Xiaowei Fu, Lei Zhang

**Abstract**: The widespread use of Vision Language Models (VLMs, e.g. CLIP) has raised concerns about their vulnerability to sophisticated and imperceptible adversarial attacks. These attacks could compromise model performance and system security in cross-modal tasks. To address this challenge, three main defense paradigms have been proposed: Training-time Defense, Test-time Adaptation Defense, and Training-free Defense. Training-time Defense involves modifying the training process, typically through adversarial fine-tuning to improve the robustness to adversarial examples. While effective, this approach requires substantial computational resources and may not generalize across all adversarial attacks. Test-time Adaptation Defense focuses on adapting the model at inference time by updating its parameters to handle unlabeled adversarial examples, offering flexibility but often at the cost of increased complexity and computational overhead. Training-free Defense avoids modifying the model itself, instead focusing on altering the adversarial inputs or their feature embeddings, which enforces input perturbations to mitigate the impact of attacks without additional training. This survey reviews the latest advancements in adversarial defense strategies for VLMs, highlighting the strengths and limitations of such approaches and discussing ongoing challenges in enhancing the robustness of VLMs.

摘要: 视觉语言模型（VLMs，如CLIP）的广泛应用引发了对其易受复杂且难以察觉的对抗攻击的担忧。这些攻击可能损害跨模态任务中的模型性能和系统安全性。为应对这一挑战，已提出三种主要防御范式：训练时防御、测试时自适应防御和无训练防御。训练时防御通过修改训练过程（通常采用对抗性微调）来提升对对抗样本的鲁棒性。该方法虽有效，但需要大量计算资源，且可能无法泛化至所有对抗攻击。测试时自适应防御专注于在推理时通过更新模型参数来处理未标记的对抗样本，具有灵活性，但通常以增加复杂性和计算开销为代价。无训练防御避免修改模型本身，转而专注于改变对抗输入或其特征嵌入，通过强制输入扰动来减轻攻击影响，无需额外训练。本综述回顾了VLM对抗防御策略的最新进展，重点分析了各类方法的优势与局限，并探讨了增强VLM鲁棒性所面临的持续挑战。



## **16. Cisco Integrated AI Security and Safety Framework Report**

思科集成式AI安全与防护框架报告 cs.CR

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.12921v1) [paper-pdf](https://arxiv.org/pdf/2512.12921v1)

**Confidence**: 0.85

**Authors**: Amy Chang, Tiffany Saade, Sanket Mendapara, Adam Swanda, Ankit Garg

**Abstract**: Artificial intelligence (AI) systems are being readily and rapidly adopted, increasingly permeating critical domains: from consumer platforms and enterprise software to networked systems with embedded agents. While this has unlocked potential for human productivity gains, the attack surface has expanded accordingly: threats now span content safety failures (e.g., harmful or deceptive outputs), model and data integrity compromise (e.g., poisoning, supply-chain tampering), runtime manipulations (e.g., prompt injection, tool and agent misuse), and ecosystem risks (e.g., orchestration abuse, multi-agent collusion). Existing frameworks such as MITRE ATLAS, National Institute of Standards and Technology (NIST) AI 100-2 Adversarial Machine Learning (AML) taxonomy, and OWASP Top 10s for Large Language Models (LLMs) and Agentic AI Applications provide valuable viewpoints, but each covers only slices of this multi-dimensional space.   This paper presents Cisco's Integrated AI Security and Safety Framework ("AI Security Framework"), a unified, lifecycle-aware taxonomy and operationalization framework that can be used to classify, integrate, and operationalize the full range of AI risks. It integrates AI security and AI safety across modalities, agents, pipelines, and the broader ecosystem. The AI Security Framework is designed to be practical for threat identification, red-teaming, risk prioritization, and it is comprehensive in scope and can be extensible to emerging deployments in multimodal contexts, humanoids, wearables, and sensory infrastructures. We analyze gaps in prevailing frameworks, discuss design principles for our framework, and demonstrate how the taxonomy provides structure for understanding how modern AI systems fail, how adversaries exploit these failures, and how organizations can build defenses across the AI lifecycle that evolve alongside capability advancements.

摘要: 人工智能（AI）系统正被迅速广泛采用，日益渗透到关键领域：从消费平台和企业软件到嵌入智能体的网络系统。虽然这释放了提升人类生产力的潜力，但攻击面也相应扩大：威胁现已涵盖内容安全失效（如有害或欺骗性输出）、模型与数据完整性破坏（如投毒、供应链篡改）、运行时操控（如提示注入、工具与智能体滥用）以及生态系统风险（如编排滥用、多智能体串谋）。现有框架如MITRE ATLAS、美国国家标准与技术研究院（NIST）AI 100-2对抗性机器学习（AML）分类法，以及OWASP针对大语言模型（LLMs）和智能体AI应用的前十大风险清单提供了宝贵视角，但各自仅覆盖这一多维空间的局部。本文提出思科集成式AI安全与防护框架（“AI安全框架”），这是一个统一、感知生命周期的分类法与操作化框架，可用于对各类AI风险进行分类、整合和操作化实施。该框架将AI安全与AI防护整合到多模态、智能体、流水线及更广泛的生态系统中。AI安全框架设计注重实用性，适用于威胁识别、红队测试、风险优先级排序，其范围全面且可扩展至多模态场景、人形机器人、可穿戴设备和传感基础设施等新兴部署领域。我们分析了主流框架的不足，讨论了本框架的设计原则，并展示该分类法如何为理解现代AI系统失效机制、攻击者如何利用这些漏洞，以及组织如何在AI全生命周期构建与能力演进同步的防御体系提供结构化支撑。



## **17. BRIDG-ICS: AI-Grounded Knowledge Graphs for Intelligent Threat Analytics in Industry~5.0 Cyber-Physical Systems**

BRIDG-ICS：面向工业5.0信息物理系统的智能威胁分析——基于人工智能的知识图谱框架 cs.CR

44 Pages, To be published in Springer Cybersecurity Journal

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12112v1) [paper-pdf](https://arxiv.org/pdf/2512.12112v1)

**Confidence**: 0.85

**Authors**: Padmeswari Nandiya, Ahmad Mohsin, Ahmed Ibrahim, Iqbal H. Sarker, Helge Janicke

**Abstract**: Industry 5.0's increasing integration of IT and OT systems is transforming industrial operations but also expanding the cyber-physical attack surface. Industrial Control Systems (ICS) face escalating security challenges as traditional siloed defences fail to provide coherent, cross-domain threat insights. We present BRIDG-ICS (BRIDge for Industrial Control Systems), an AI-driven Knowledge Graph (KG) framework for context-aware threat analysis and quantitative assessment of cyber resilience in smart manufacturing environments. BRIDG-ICS fuses heterogeneous industrial and cybersecurity data into an integrated Industrial Security Knowledge Graph linking assets, vulnerabilities, and adversarial behaviours with probabilistic risk metrics (e.g. exploit likelihood, attack cost). This unified graph representation enables multi-stage attack path simulation using graph-analytic techniques. To enrich the graph's semantic depth, the framework leverages Large Language Models (LLMs): domain-specific LLMs extract cybersecurity entities, predict relationships, and translate natural-language threat descriptions into structured graph triples, thereby populating the knowledge graph with missing associations and latent risk indicators. This unified AI-enriched KG supports multi-hop, causality-aware threat reasoning, improving visibility into complex attack chains and guiding data-driven mitigation. In simulated industrial scenarios, BRIDG-ICS scales well, reduces potential attack exposure, and can enhance cyber-physical system resilience in Industry 5.0 settings.

摘要: 工业5.0中IT与OT系统的深度融合正在变革工业运营模式，同时也扩大了信息物理系统的攻击面。传统孤岛式防御难以提供跨域连贯的威胁洞察，工业控制系统面临日益严峻的安全挑战。本文提出BRIDG-ICS（工业控制系统桥梁框架），这是一种基于人工智能的知识图谱框架，用于智能制造环境下的情境感知威胁分析和网络弹性定量评估。该框架将异构工业数据与网络安全数据融合为统一的工业安全知识图谱，通过概率风险指标（如攻击利用可能性、攻击成本）关联资产、漏洞和攻击行为。这种统一的图表示支持使用图分析技术进行多阶段攻击路径模拟。为增强图谱语义深度，框架采用大语言模型：领域专用LLMs提取网络安全实体、预测关联关系，并将自然语言威胁描述转化为结构化图谱三元组，从而补充知识图谱中缺失的关联关系和潜在风险指标。这种AI增强的统一知识图谱支持多跳因果感知的威胁推理，提升对复杂攻击链的可见性，并指导数据驱动的缓解措施。在模拟工业场景中，BRIDG-ICS展现出良好的可扩展性，能有效降低潜在攻击暴露面，提升工业5.0环境下信息物理系统的网络弹性。



## **18. Enhancing Cloud Network Resilience via a Robust LLM-Empowered Multi-Agent Reinforcement Learning Framework**

通过稳健的LLM赋能多智能体强化学习框架增强云网络韧性 cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07122v1) [paper-pdf](https://arxiv.org/pdf/2601.07122v1)

**Confidence**: 0.85

**Authors**: Yixiao Peng, Hao Hu, Feiyang Li, Xinye Cao, Yingchang Jiang, Jipeng Tang, Guoshun Nan, Yuling Liu

**Abstract**: While virtualization and resource pooling empower cloud networks with structural flexibility and elastic scalability, they inevitably expand the attack surface and challenge cyber resilience. Reinforcement Learning (RL)-based defense strategies have been developed to optimize resource deployment and isolation policies under adversarial conditions, aiming to enhance system resilience by maintaining and restoring network availability. However, existing approaches lack robustness as they require retraining to adapt to dynamic changes in network structure, node scale, attack strategies, and attack intensity. Furthermore, the lack of Human-in-the-Loop (HITL) support limits interpretability and flexibility. To address these limitations, we propose CyberOps-Bots, a hierarchical multi-agent reinforcement learning framework empowered by Large Language Models (LLMs). Inspired by MITRE ATT&CK's Tactics-Techniques model, CyberOps-Bots features a two-layer architecture: (1) An upper-level LLM agent with four modules--ReAct planning, IPDRR-based perception, long-short term memory, and action/tool integration--performs global awareness, human intent recognition, and tactical planning; (2) Lower-level RL agents, developed via heterogeneous separated pre-training, execute atomic defense actions within localized network regions. This synergy preserves LLM adaptability and interpretability while ensuring reliable RL execution. Experiments on real cloud datasets show that, compared to state-of-the-art algorithms, CyberOps-Bots maintains network availability 68.5% higher and achieves a 34.7% jumpstart performance gain when shifting the scenarios without retraining. To our knowledge, this is the first study to establish a robust LLM-RL framework with HITL support for cloud defense. We will release our framework to the community, facilitating the advancement of robust and autonomous defense in cloud networks.

摘要: 虚拟化和资源池化虽然赋予云网络结构灵活性和弹性扩展能力，但也不可避免地扩大了攻击面并挑战网络韧性。基于强化学习（RL）的防御策略已被开发用于在对抗条件下优化资源部署和隔离策略，旨在通过维持和恢复网络可用性来增强系统韧性。然而，现有方法缺乏稳健性，因为它们需要重新训练以适应网络结构、节点规模、攻击策略和攻击强度的动态变化。此外，缺乏人在回路（HITL）支持限制了可解释性和灵活性。为应对这些局限，我们提出CyberOps-Bots——一个由大语言模型（LLMs）赋能的分层多智能体强化学习框架。受MITRE ATT&CK的战术-技术模型启发，CyberOps-Bots采用双层架构：（1）上层LLM智能体包含四个模块——ReAct规划、基于IPDRR的感知、长短时记忆和行动/工具集成——执行全局态势感知、人类意图识别和战术规划；（2）通过异构分离预训练开发的下层RL智能体，在局部网络区域内执行原子防御动作。这种协同机制在保持LLM适应性和可解释性的同时，确保了RL执行的可靠性。在真实云数据集上的实验表明，与最先进算法相比，CyberOps-Bots在网络可用性上保持高出68.5%的水平，并在场景切换无需重新训练时实现了34.7%的跳跃启动性能增益。据我们所知，这是首个为云防御建立具有HITL支持的稳健LLM-RL框架的研究。我们将向社区开源该框架，以推动云网络中稳健自主防御的进步。



